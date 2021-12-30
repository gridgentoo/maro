import asyncio
from abc import ABCMeta, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Union

from maro.rl_v3.distributed.remote import RemoteOps
from maro.rl_v3.policy import RLPolicy
from maro.rl_v3.policy_trainer.abs_train_ops import AbsTrainOps
from maro.rl_v3.replay_memory import MultiReplayMemory, ReplayMemory
from maro.rl_v3.utils import AbsTransitionBatch, MultiTransitionBatch, TransitionBatch


class AbsTrainer(object, metaclass=ABCMeta):
    """Policy trainer used to train policies. Trainer maintains several train ops and
    controls training logics of them, while train ops take charge of specific policy updating.
    """
    def __init__(
        self,
        name: str,
        get_policy_func_dict: Dict[str, Callable[[str], RLPolicy]],
        dispatcher_address: Tuple[str, int] = None,
        train_batch_size: int = 128
    ) -> None:
        """
        Args:
            name (str): Name of the trainer.
            get_policy_func_dict (Dict[str, Callable[[str], RLPolicy]]): Dict of functions that used to create policies.
            dispatcher_address (Tuple[str, int]): The address of the dispatcher. This is used under only distributed
                model. Defaults to None.
            train_batch_size (int): Train batch size. Defaults to 128.
        """
        self._name = name
        self._get_policy_func_dict = get_policy_func_dict
        self._dispatcher_address = dispatcher_address
        self._train_batch_size = train_batch_size

        self._ops_creator: Optional[Dict[str, Callable[[str], AbsTrainOps]]] = None

        print(f"Creating trainer {self.__class__.__name__} {name}.")

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def train_step(self) -> None:
        """Run a training step to update all the policies that this trainer is responsible for.
        """
        raise NotImplementedError

    @abstractmethod
    def get_policy_state_dict(self) -> Dict[str, object]:
        """Get policies' states.

        Returns:
            A double-deck dict with format: {policy_name: policy_state}.
        """
        raise NotImplementedError

    @abstractmethod
    def set_policy_state_dict(self, policy_state_dict: Dict[str, object]) -> None:
        """Set policies' states.

        Args:
            policy_state_dict (Dict[str, object]): A double-deck dict with format: {policy_name: policy_state}.
        """
        raise NotImplementedError

    @abstractmethod
    def build(self) -> None:
        raise NotImplementedError

    def get_ops_creator(self) -> Dict[str, Callable[[str], AbsTrainOps]]:
        if self._ops_creator is None:
            self._ops_creator = self._get_ops_creator_impl()
        return self._ops_creator

    @abstractmethod
    def _get_ops_creator_impl(self) -> Dict[str, Callable[[str], AbsTrainOps]]:
        raise NotImplementedError

    def get_ops(self, ops_name: str) -> Union[RemoteOps, AbsTrainOps]:
        if self._dispatcher_address:
            return RemoteOps(ops_name, self._dispatcher_address)
        else:
            return self.get_ops_creator()[ops_name](ops_name)


class SingleTrainer(AbsTrainer, metaclass=ABCMeta):
    """Policy trainer that trains only one policy.
    """
    def __init__(
        self,
        name: str,
        get_policy_func_dict: Dict[str, Callable[[str], RLPolicy]],
        dispatcher_address: Tuple[str, int] = None,
        train_batch_size: int = 128
    ) -> None:
        if len(get_policy_func_dict) > 1:
            raise ValueError(f"trainer {self._name} cannot have more than one policy assigned to it")

        super(SingleTrainer, self).__init__(
            name, get_policy_func_dict, dispatcher_address, train_batch_size
        )

        self._replay_memory: Optional[ReplayMemory] = None
        self._ops: Union[RemoteOps, AbsTrainOps, None] = None

        self._policy_name = list(get_policy_func_dict.keys())[0]
        self._get_policy_func = lambda: get_policy_func_dict[self._policy_name](self._policy_name)

    def record(self, transition_batch: TransitionBatch) -> None:
        """Record the experiences collected by external modules.

        Args:
            transition_batch (TransitionBatch): A TransitionBatch item that contains a batch of experiences.
        """
        assert isinstance(transition_batch, TransitionBatch)
        self._replay_memory.put(transition_batch)

    def _get_batch(self, batch_size: int = None) -> TransitionBatch:
        return self._replay_memory.sample(batch_size if batch_size is not None else self._train_batch_size)

    def get_policy_state_dict(self) -> Dict[str, object]:
        if not self._ops:
            raise ValueError("'init_ops' needs to be called to create an ops instance first.")
        return {self._ops.policy_name: self._ops.get_policy_state()}

    def set_policy_state_dict(self, policy_state_dict: Dict[str, object]) -> None:
        if not self._ops:
            raise ValueError("'init_ops' needs to be called to create an ops instance first.")
        assert len(policy_state_dict) == 1 and self._ops.policy_name in policy_state_dict
        self._ops.set_policy_state(policy_state_dict[self._ops.policy_name])


class MultiTrainer(AbsTrainer, metaclass=ABCMeta):
    """Policy trainer that trains multiple policies.
    """
    def __init__(
        self,
        name: str,
        get_policy_func_dict: Dict[str, Callable[[str], RLPolicy]],
        dispatcher_address: Tuple[str, int] = None,
        train_batch_size: int = 128
    ) -> None:
        super(MultiTrainer, self).__init__(
            name, get_policy_func_dict, dispatcher_address, train_batch_size
        )

        self._replay_memory: Optional[MultiReplayMemory] = None
        self._ops_list: List[Union[RemoteOps, AbsTrainOps]] = []
        self._policy_names = sorted(list(get_policy_func_dict.keys()))

    @property
    def num_policies(self) -> int:
        return len(self._ops_list)

    def record(self, transition_batch: MultiTransitionBatch) -> None:
        """Record the experiences collected by external modules.

        Args:
            transition_batch (MultiTransitionBatch): A TransitionBatch item that contains a batch of experiences.
        """
        self._replay_memory.put(transition_batch)

    def _get_batch(self, batch_size: int = None) -> MultiTransitionBatch:
        return self._replay_memory.sample(batch_size if batch_size is not None else self._train_batch_size)

    def get_policy_state_dict(self) -> Dict[str, object]:
        if len(self._ops_list) == 0:
            raise ValueError("'init_ops' needs to be called to create an ops instance first.")

        return {ops.policy_name: ops.get_policy_state() for ops in self._ops_list}

    def set_policy_state_dict(self, policy_state_dict: Dict[str, object]) -> None:
        if len(self._ops_list) == 0:
            raise ValueError("'init_ops' needs to be called to create an ops instance first.")

        assert len(policy_state_dict) == len(self._ops_list)
        for ops in self._ops_list:
            ops.set_policy_state(policy_state_dict[ops.policy_name])


class BatchTrainer:
    def __init__(self, trainers: List[AbsTrainer]) -> None:
        self._trainers = trainers
        self._trainer_dict = {trainer.name: trainer for trainer in self._trainers}

    def record(self, batch_by_trainer: Dict[str, AbsTransitionBatch]) -> None:
        for trainer_name, batch in batch_by_trainer.items():
            self._trainer_dict[trainer_name].record(batch)

    def train(self) -> None:
        try:
            asyncio.run(self._train_impl())
        except TypeError:
            for trainer in self._trainers:
                trainer.train_step()

    async def _train_impl(self) -> None:
        await asyncio.gather(*[trainer.train_step() for trainer in self._trainers])
