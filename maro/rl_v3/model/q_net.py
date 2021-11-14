from abc import ABCMeta, abstractmethod
from typing import Optional

import torch

from maro.rl_v3.utils import SHAPE_CHECK_FLAG, match_shape
from .abs_net import AbsNet


class QNet(AbsNet, metaclass=ABCMeta):
    """
    Net for Q functions.
    """
    def __init__(self, state_dim: int, action_dim: int) -> None:
        """
        Args:
            state_dim (int): Dimension of states.
            action_dim (int): Dimension of actions.
        """
        super(QNet, self).__init__()
        self._state_dim = state_dim
        self._action_dim = action_dim

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def _shape_check(self, states: torch.Tensor, actions: Optional[torch.Tensor] = None) -> bool:
        if not SHAPE_CHECK_FLAG:
            return True
        else:
            if states.shape[0] == 0 or not match_shape(states, (None, self.state_dim)):
                return False
            if actions is not None:
                if not match_shape(actions, (states.shape[0], self.action_dim)):
                    return False
            return True

    def q_values(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Get Q-values according to states and actions.

        Args:
            states (torch.Tensor): States.
            actions (torch.Tensor): Actions.

        Returns:
            Q-values with shape [batch_size]
        """
        assert self._shape_check(states=states, actions=actions), \
            f"States or action shape check failed. Expecting: " \
            f"states = {('BATCH_SIZE', self.state_dim)}, action = {('BATCH_SIZE', self.action_dim)}. " \
            f"Actual: states = {states.shape}, action = {actions.shape}."
        q = self._get_q_values(states, actions)
        assert match_shape(q, (states.shape[0],)), \
            f"Q-value shape check failed. Expecting: {(states.shape[0],)}, actual: {q.shape}."  # [B]
        return q

    @abstractmethod
    def _get_q_values(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Implementation of `q_values`.
        """
        raise NotImplementedError


class DiscreteQNet(QNet, metaclass=ABCMeta):
    """
    Net for Q functions with discrete actions.
    """
    def __init__(self, state_dim: int, action_num: int) -> None:
        """
        Args:
            state_dim (int): Dimension of states.
            action_num (int): Number of actions.
        """
        super(DiscreteQNet, self).__init__(state_dim=state_dim, action_dim=1)
        self._action_num = action_num

    @property
    def action_num(self) -> int:
        return self._action_num

    def q_values_for_all_actions(self, states: torch.Tensor) -> torch.Tensor:
        """
        Get Q-values for all actions according to states.

        Args:
            states (torch.Tensor): States.

        Returns:
            Q-values for all actions. The returned value has the shape [batch_size, action_num]
        """
        assert self._shape_check(states=states), \
            f"States shape check failed. Expecting: {('BATCH_SIZE', self.state_dim)}, actual: {states.shape}."
        q = self._get_q_values_for_all_actions(states)
        assert match_shape(q, (states.shape[0], self.action_num)), \
            f"Q-value matrix shape check failed. Expecting: {(states.shape[0], self.action_num)}, " \
            f"actual: {q.shape}."  # [B, action_num]
        return q

    def _get_q_values(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        q = self.q_values_for_all_actions(states)  # [B, action_num]
        return q.gather(1, actions.long()).reshape(-1)  # [B, action_num] + [B, 1] => [B]

    @abstractmethod
    def _get_q_values_for_all_actions(self, states: torch.Tensor) -> torch.Tensor:
        """
        Implementation of `q_values_for_all_actions`.
        """
        raise NotImplementedError


class ContinuousQNet(QNet, metaclass=ABCMeta):
    """
    Net for Q functions with continuous actions.
    """
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super(ContinuousQNet, self).__init__(state_dim=state_dim, action_dim=action_dim)
