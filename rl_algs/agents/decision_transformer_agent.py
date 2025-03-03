from typing import Callable, Optional, Sequence, Tuple

import torch
from torch import nn
import numpy as np

import rl_algs.utility.pytorch_util as ptu


class DecisionTransformerAgent:
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,
        infer_state,
        make_actor: Callable[[Tuple[int, ...], int], nn.Module],
        make_actor_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_actor_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        make_actor_loss,
        clip_grad_norm=0.25,
    ):
        self.actor = make_actor(observation_shape, action_dim)
        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.actor_lr_scheduler = make_actor_schedule(self.actor_optimizer)
        self.actor_loss = make_actor_loss()

        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.clip_grad_norm = clip_grad_norm
        self.infer_state = infer_state


    def get_action(
        self
    ) -> np.ndarray:
        # shall use eval mode here
        self.actor.eval()

        x = self.infer_state.get_transformer_input()
        _, y, _ = self.actor.forward(**x)

        self.actor.train()
        assert y.shape[0] == 1 and y.shape[-1] == self.action_dim

        return ptu.to_numpy(y[0, -1])
    
    # def reset_infer_state(self, observation_init, target_reward):
    #     self.infer_state.reset(observation_init, target_reward)

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        reward_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        """
        code adapted from trainner in https://github.com/kzl/decision-transformer
        the code shows that, dt is only perform prediction over actions, it do not
        model state and reward dynamics
        """
        target = actions.clone().detach()

        # forward
        # enable train mode
        self.actor.train()
        _, action_preds, _ = self.actor.forward(
            observations, actions, reward_to_go, timesteps, attention_mask
        )

        # compute loss on valid actions
        attention_mask = attention_mask.reshape((-1,))
        action_preds = action_preds.reshape(-1, self.action_dim)[attention_mask]
        target = target.reshape(-1, self.action_dim)[attention_mask]
        assert len(target) > 0

        loss: torch.Tensor = self.actor_loss(action_preds, target)

        # one step of optimization
        self.actor_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.clip_grad_norm
        )
        self.actor_optimizer.step()
        self.actor_lr_scheduler.step()

        # report update information
        info = {
            "actor_loss": loss.item(),
            "actor_lr": self.actor_lr_scheduler.get_last_lr()[0],
            "actor_grad_norm": grad_norm.item()
        }

        return info


class DtInferenceState:
    """
    Stores context required for sampling from a decision transformer
    """

    def __init__(
        self,
        observation_shape,
        action_dim,
        reward_decay: float,
        observation_mean: np.ndarray,
        observation_std: np.ndarray,
    ):
        self.observation_shape = observation_shape
        self.action_dim = action_dim

        self.observation_mean = observation_mean.reshape((-1,))
        self.observation_std = observation_std.reshape((-1,))

        # self.target_reward = 0.0
        self.reward_decay = reward_decay

        # historical states
        self.observations = []
        self.actions = []
        self.reward_to_go = []
        self.timesteps = []

    def reset(self, observation, target_reward):
        '''start a new sampling episode after env reset
        usage:
            obs, _ = env.reset()
            actstate.reset(obs, target_reward)
        '''
        # clear buffer
        self.observations = []
        self.actions = []
        self.reward_to_go = []
        self.timesteps = []

        # insert init state
        self._init(observation, target_reward)

    def update(self, action: np.ndarray, reward: float, new_state: np.ndarray):
        """update inner state after observe the agent's action and the env's response

        Args:
            action (np.ndarray): action taken by the agent, shape (action_dim,)
            reward (float): immediate reward
            new_state (np.ndarray): next state, shape == observation_shape
        """
        assert len(self.actions) > 0, 'must call reset before update'
        self.actions[-1] = action

        # compute target_rtg after observe reward
        rtg = self.reward_to_go[-1][0]
        rtg = rtg - reward / self.reward_decay

        # update buffers
        self._append_observation(new_state)
        self._append_action(None)
        self._append_reward_to_go(rtg)
        self._append_timesteps()

    def get_transformer_input(self):
        assert len(self.observations) > 0, 'reset is not called'

        out = {
            'states': np.stack(self.observations, axis=0)[None],
            'actions': np.stack(self.actions, axis=0)[None],
            'returns_to_go': np.stack(self.reward_to_go, axis=0)[None],
            'timesteps': np.stack(self.timesteps, axis=0)[None]
        }

        # check shape
        assert out['states'].shape == (1, len(self.timesteps), np.prod(self.observation_shape))
        assert out['actions'].shape == (1, len(self.timesteps), self.action_dim)
        assert out['returns_to_go'].shape == (1, len(self.timesteps), 1)
        assert out['timesteps'].shape == (1, len(self.timesteps))

        out = ptu.from_numpy(out)
        out['attention_mask'] = None

        return out

    def _append_observation(self, new_observation: np.ndarray):
        '''
        new_observation of shape (x, y) is normalized then reshaped
            into (x*y,) before append to observation list 
        '''
        assert new_observation.shape == self.observation_shape
        new_observation = new_observation.astype(np.float32).reshape((-1,))
        new_observation = (
            new_observation - self.observation_mean
        ) / self.observation_std
        self.observations.append(new_observation)

    def _append_action(self, new_action: Optional[np.ndarray]):
        '''
        new_action of shape (action_dim,)
        '''
        if new_action is None:
            new_action = np.zeros(self.action_dim, dtype=np.float32)

        assert new_action.shape == (self.action_dim,)
        new_action = new_action.astype(np.float32)
        self.actions.append(new_action)

    def _append_reward_to_go(self, rtg: float):
        '''
        keep rtg in self.reward_to_go has shape (1,)
        '''
        rtg_arr = np.zeros((1,), dtype=np.float32)
        rtg_arr[0] = rtg
        self.reward_to_go.append(rtg_arr)

    def _append_timesteps(self):
        sz = len(self.timesteps)
        self.timesteps.append(np.array(sz, dtype=np.int64))

    def _init(self, observation, target_reward):
        # insert state_0
        self._append_observation(observation)
        self._append_action(None)
        self._append_reward_to_go(target_reward / self.reward_decay)
        self._append_timesteps()
