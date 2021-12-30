from abc import ABCMeta, abstractmethod
from typing import Callable, Dict, List

import torch

from maro.rl.utils import average_grads
from maro.rl_v3.policy import RLPolicy
from maro.rl_v3.utils import AbsTransitionBatch, MultiTransitionBatch, TransitionBatch


class AbsTrainOps(object, metaclass=ABCMeta):
    """The basic component for training a policy, which mainly takes charge of gradient computation and policy update.
    In trainer, train worker hosts a policy, and trainer hosts several train workers. In gradient workers,
    the train worker is an atomic representation of a policy, to perform parallel gradient computing.
    """
    def __init__(
        self,
        name: str,
        device: str,
        is_single_scenario: bool,
        get_policy_func: Callable[[], RLPolicy],
        enable_data_parallelism: bool = False,
    ) -> None:
        super(AbsTrainOps, self).__init__()
        self._name = name
        self._device = torch.device(device) if device is not None \
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._is_single_scenario = is_single_scenario
        self._enable_data_parallelism = enable_data_parallelism

        self._policy = get_policy_func()
        self._policy.to_device(self._device)

    @property
    def name(self) -> str:
        return self._name

    @property
    def policy_name(self) -> str:
        return self._policy.name

    @property
    def policy_state_dim(self) -> int:
        return self._policy.state_dim

    @property
    def policy_action_dim(self) -> int:
        return self._policy.action_dim

    def _is_valid_transition_batch(self, batch: AbsTransitionBatch) -> bool:
        return isinstance(batch, TransitionBatch) if self._is_single_scenario \
            else isinstance(batch, MultiTransitionBatch)

    def _get_batch_grad(
        self,
        batch: AbsTransitionBatch,
        tensor_dict: Dict[str, object] = None,
        scope: str = "all"
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        if self._enable_data_parallelism:
            gradients = self._remote_learn(batch, tensor_dict, scope)
            return average_grads(gradients)
        else:
            return self.get_batch_grad(batch, tensor_dict, scope)

    def _remote_learn(
        self,
        batch: AbsTransitionBatch,
        tensor_dict: Dict[str, object] = None,
        scope: str = "all"
    ) -> List[Dict[str, Dict[int, Dict[str, torch.Tensor]]]]:
        """Learn a batch of experience data from remote gradient workers.
        The task queue client will first request available gradient workers from task queue. If all workers are busy,
        it will keep waiting until at least 1 worker is available. Then the task queue client submits batch and state
        to the assigned workers to compute gradients.
        """
        pass  # TODO

    @abstractmethod
    def get_batch_grad(
        self,
        batch: AbsTransitionBatch,
        tensor_dict: Dict[str, object] = None,
        scope: str = "all"
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        raise NotImplementedError

    @abstractmethod
    def _dispatch_batch(self, batch: AbsTransitionBatch, num_sub_batches: int) -> List[AbsTransitionBatch]:
        """Divide experience data batch to several parts.
        For on-policy algorithms, like PG, the batch is divided into several complete trajectories.
        For off-policy algorithms, like DQN, the batch is treated as independent data points and divided evenly."""
        raise NotImplementedError

    @abstractmethod
    def _dispatch_tensor_dict(self, tensor_dict: Dict[str, object], num_sub_batches: int) -> List[Dict[str, object]]:
        raise NotImplementedError

    @abstractmethod
    def get_ops_state_dict(self, scope: str = "all") -> dict:
        """
        Returns:
            A dict that contains ops's state.
        """
        raise NotImplementedError

    @abstractmethod
    def set_ops_state_dict(self, ops_state_dict: dict, scope: str = "all") -> None:
        """Set ops's state."""
        raise NotImplementedError

    def set_batch(self, batch: AbsTransitionBatch) -> None:
        assert self._is_valid_transition_batch(batch)
        self._batch = batch

    def get_policy_state(self) -> object:
        return self._policy.get_policy_state()

    def set_policy_state(self, policy_state: object) -> None:
        self._policy.set_policy_state(policy_state)
