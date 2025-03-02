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
        make_actor: Callable[[Tuple[int, ...], int], nn.Module],
        make_actor_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_actor_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        make_actor_loss,
        clip_grad_norm=0.25
    ):
        self.actor = make_actor(observation_shape, action_dim)
        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.actor_lr_scheduler = make_actor_schedule(self.actor_optimizer)
        self.actor_loss = make_actor_loss()

        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.clip_grad_norm = clip_grad_norm
    
    # TODO: implement a actor for get action
    def get_action(
            self, 
            observation: np.ndarray, 
            action: np.ndarray, 
            returns_to_go: float, 
            timestep: int
        ) -> np.ndarray:
        # shall use eval mode here
        raise NotImplementedError
    
    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        reward_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor
    ):
        '''
        code adapted from trainner in https://github.com/kzl/decision-transformer
        the code shows that, dt is only perform prediction over actions, it do not
        model state and reward dynamics
        '''
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
        grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad_norm)
        self.actor_optimizer.step()
        self.actor_lr_scheduler.step()

        # report update information
        info = {
            "actor_loss": loss.item(),
            "actor_lr": self.actor_lr_scheduler.get_last_lr()[0],
            "actor_grad_norm": grad_norm.item()
        }

        return info