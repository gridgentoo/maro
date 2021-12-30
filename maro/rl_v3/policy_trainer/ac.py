from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

from maro.rl.utils import discount_cumsum
from maro.rl_v3.model import VNet
from maro.rl_v3.policy import DiscretePolicyGradient
from maro.rl_v3.replay_memory import FIFOReplayMemory
from maro.rl_v3.utils import AbsTransitionBatch, ndarray_to_tensor, TransitionBatch
from .abs_train_ops import AbsTrainOps
from .abs_trainer import SingleTrainer


class DiscreteActorCriticOps(AbsTrainOps):
    def __init__(
        self,
        name: str,
        device: str,
        get_policy_func: Callable[[], DiscretePolicyGradient],
        get_v_critic_net_func: Callable[[], VNet],
        enable_data_parallelism: bool = False,
        *,
        reward_discount: float = 0.9,
        critic_loss_coef: float = 0.1,
        critic_loss_cls: Callable = None,
        clip_ratio: float = None,
        lam: float = 0.9,
        min_logp: float = None,
    ) -> None:
        super(DiscreteActorCriticOps, self).__init__(
            name=name, device=device, is_single_scenario=True,
            get_policy_func=get_policy_func, enable_data_parallelism=enable_data_parallelism
        )

        assert isinstance(self._policy, DiscretePolicyGradient)

        self._reward_discount = reward_discount
        self._critic_loss_coef = critic_loss_coef
        self._critic_loss_func = critic_loss_cls() if critic_loss_cls is not None else torch.nn.MSELoss()
        self._clip_ratio = clip_ratio
        self._lam = lam
        self._min_logp = min_logp
        self._v_critic_net = get_v_critic_net_func()
        self._v_critic_net.to(self._device)

    def get_batch_grad(
        self,
        batch: TransitionBatch,
        tensor_dict: Dict[str, object] = None,
        scope: str = "all"
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Reference: https://tinyurl.com/2ezte4cr
        """
        assert scope in ("all", "actor", "critic"), \
            f"Unrecognized scope {scope}. Excepting 'all', 'actor', or 'critic'."

        grad_dict = {}
        if scope in ("all", "actor"):
            grad_dict["actor_grad"] = self._get_actor_grad(batch)

        if scope in ("all", "critic"):
            grad_dict["critic_grad"] = self._get_critic_grad(batch)

        return grad_dict

    def _dispatch_batch(self, batch: AbsTransitionBatch, num_sub_batches: int) -> List[AbsTransitionBatch]:
        raise NotImplementedError

    def _dispatch_tensor_dict(self, tensor_dict: Dict[str, object], num_sub_batches: int) -> List[Dict[str, object]]:
        raise NotImplementedError

    def _get_critic_grad(self, batch: TransitionBatch) -> Dict[str, torch.Tensor]:
        states = ndarray_to_tensor(batch.states, self._device)  # s

        self._policy.train()
        self._v_critic_net.train()

        state_values = self._v_critic_net.v_values(states)
        values = state_values.detach().numpy()
        values = np.concatenate([values, values[-1:]])
        rewards = np.concatenate([batch.rewards, values[-1:]])

        returns = ndarray_to_tensor(discount_cumsum(rewards, self._reward_discount)[:-1], self._device)
        critic_loss = self._critic_loss_func(state_values, returns)

        return self._v_critic_net.get_gradients(critic_loss * self._critic_loss_coef)

    def _get_actor_grad(self, batch: TransitionBatch) -> Dict[str, torch.Tensor]:
        assert isinstance(self._policy, DiscretePolicyGradient)

        states = ndarray_to_tensor(batch.states, self._device)  # s
        actions = ndarray_to_tensor(batch.actions, self._device).long()  # a

        if self._clip_ratio is not None:
            self._policy.eval()
            logps_old = self._policy.get_state_action_logps(states, actions)
        else:
            logps_old = None

        state_values = self._v_critic_net.v_values(states)
        values = state_values.detach().numpy()
        values = np.concatenate([values, values[-1:]])
        rewards = np.concatenate([batch.rewards, values[-1:]])

        deltas = rewards[:-1] + self._reward_discount * values[1:] - values[:-1]  # r + gamma * v(s') - v(s)
        advantages = ndarray_to_tensor(discount_cumsum(deltas, self._reward_discount * self._lam), self._device)

        action_probs = self._policy.get_action_probs(states)
        logps = torch.log(action_probs.gather(1, actions).squeeze())
        logps = torch.clamp(logps, min=self._min_logp, max=.0)
        if self._clip_ratio is not None:
            ratio = torch.exp(logps - logps_old)
            clipped_ratio = torch.clamp(ratio, 1 - self._clip_ratio, 1 + self._clip_ratio)
            actor_loss = -(torch.min(ratio * advantages, clipped_ratio * advantages)).mean()
        else:
            actor_loss = -(logps * advantages).mean()  # I * delta * log pi(a|s)

        return self._policy.get_gradients(actor_loss)

    def update(self, grad_iters: int) -> None:
        """Reference: https://tinyurl.com/2ezte4cr
        """
        for _ in range(grad_iters):
            grad_dict = self._get_batch_grad(self._batch, scope="all")
            self._policy.train()
            self._policy.apply_gradients(grad_dict["actor_grad"])
            self._v_critic_net.train()
            self._v_critic_net.apply_gradients(grad_dict["critic_grad"])

    def get_ops_state_dict(self, scope: str = "all") -> dict:
        ret_dict = {}
        if scope in ("all", "actor"):
            ret_dict["policy_state"] = self._policy.get_policy_state()
        if scope in ("all", "critic"):
            ret_dict["critic_state"] = self._v_critic_net.get_net_state()
        return ret_dict

    def set_ops_state_dict(self, ops_state_dict: dict, scope: str = "all") -> None:
        if scope in ("all", "actor"):
            self._policy.set_policy_state(ops_state_dict["policy_state"])
        if scope in ("all", "critic"):
            self._v_critic_net.set_net_state(ops_state_dict["critic_state"])


class DiscreteActorCritic(SingleTrainer):
    """Actor Critic algorithm with separate policy and value models.

    References:
        https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch.
        https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f

    Args:
        name (str): Unique identifier for the policy.
        get_policy_func_dict (Dict[str, Callable[[str], DiscretePolicyGradient]]): Dict of functions that used to
            create policies.
        get_v_critic_net_func (Callable[[], VNet]): Function to get V critic net.
        device (str): Identifier for the torch device. The policy will be moved to the specified device. If it is
            None, the device will be set to "cpu" if cuda is unavailable and "cuda" otherwise. Defaults to None.
        enable_data_parallelism (bool): Whether to enable data parallelism in this trainer. Defaults to False.
        dispatcher_address (Tuple[str, int]): The address of the dispatcher. This is used under only distributed
            model. Defaults to None.
        train_batch_size (int): Train batch size. Defaults to 128.

        replay_memory_capacity (int): Capacity of the replay memory. Defaults to 10000.
        grad_iters (int): Number of iterations to calculate gradients. Defaults to 1.

        reward_discount (float): Reward decay as defined in standard RL terminology. Defaults to 0.9.
        critic_loss_coef (float): Coefficient for critic loss in total loss. Defaults to 0.1.
        critic_loss_cls: A string indicating a loss class provided by torch.nn or a custom loss class for computing
            the critic loss. If it is a string, it must be a key in ``TORCH_LOSS``. Defaults to "mse".
        clip_ratio (float): Clip ratio in the PPO algorithm (https://arxiv.org/pdf/1707.06347.pdf). Defaults to None,
            in which case the actor loss is calculated using the usual policy gradient theorem.
        lam (float): Lambda value for generalized advantage estimation (TD-Lambda). Defaults to 0.9.
        min_logp (float): Lower bound for clamping logP values during learning. This is to prevent logP from becoming
            very large in magnitude and causing stability issues. Defaults to None, which means no lower bound.
    """
    def __init__(
        self,
        name: str,
        get_policy_func_dict: Dict[str, Callable[[str], DiscretePolicyGradient]],
        get_v_critic_net_func: Callable[[], VNet],
        device: str = None,
        enable_data_parallelism: bool = False,
        dispatcher_address: Tuple[str, int] = None,
        train_batch_size: int = 128,
        *,
        # Training params
        replay_memory_capacity: int = 10000,
        grad_iters: int = 1,
        # Ops params
        reward_discount: float = 0.9,
        critic_loss_coef: float = 0.1,
        critic_loss_cls: Callable = None,
        clip_ratio: float = None,
        lam: float = 0.9,
        min_logp: float = None,
    ) -> None:
        super(DiscreteActorCritic, self).__init__(
            name, get_policy_func_dict, dispatcher_address, train_batch_size
        )

        self._grad_iters = grad_iters
        self._replay_memory_size = replay_memory_capacity

        self._ops_params = {
            "device": device,
            "get_policy_func": self._get_policy_func,
            "get_v_critic_net_func": get_v_critic_net_func,
            "enable_data_parallelism": enable_data_parallelism,
            "reward_discount": reward_discount,
            "critic_loss_coef": critic_loss_coef,
            "critic_loss_cls": critic_loss_cls,
            "clip_ratio": clip_ratio,
            "lam": lam,
            "min_logp": min_logp,
        }

    def train_step(self):
        self._ops.set_batch(self._get_batch())
        self._ops.update(self._grad_iters)

    def _get_ops_creator_impl(self) -> Dict[str, Callable[[str], AbsTrainOps]]:
        ops_creator: Dict[str, Callable[[str], AbsTrainOps]] = {
            "ops": lambda ops_name: DiscreteActorCriticOps(name=ops_name, **self._ops_params)
        }
        return ops_creator

    def build(self) -> None:
        self._ops = self.get_ops("ops")
        self._replay_memory = FIFOReplayMemory(
            capacity=self._replay_memory_size,
            state_dim=self._ops.policy_state_dim,
            action_dim=self._ops.policy_action_dim
        )
