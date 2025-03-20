from typing import Callable, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.functional import F
import numpy as np
import copy

import rl_algs.utility.pytorch_util as ptu
from rl_algs.networks.diffusion import Diffusion


class QLDiffuseAgent:
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,
        n_timesteps: int,
        # config for actor
        make_noise_model: Callable[[Tuple[int, ...], int, int], nn.Module],
        make_actor: Callable[[Tuple[int, ...], int, nn.Module, int], Diffusion],
        make_actor_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_actor_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        # config for critic
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_critic_optimizer: Callable[
            [torch.nn.ParameterList], torch.optim.Optimizer
        ],
        make_critic_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        # optional
        actor_target_update_rate: float = 0.005,
        critic_target_update_rate: float = 0.005,
        eta: float = 1.0,
        max_grad_norm: float = 1.0,
        t_dims: int =16,
        action_min_value: float = -1.0,
        action_max_value: float = 1.0
    ):
        # 1) actors
        noise_mode = make_noise_model(observation_shape, action_dim, t_dims)
        self.actor = make_actor(observation_shape, action_dim, noise_mode, n_timesteps)
        self.actor_target = copy.deepcopy(self.actor)

        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.actor_lr_schedule = make_actor_schedule(self.actor_optimizer)

        # 2) build critic
        # critic contains 2 nets with the same structure
        self.critics = nn.ModuleList(
            [
                make_critic(observation_shape, action_dim),
                make_critic(observation_shape, action_dim),
            ]
        )
        self.critics_target = copy.deepcopy(self.critics)

        self.critics_optimizer = make_critic_optimizer(self.critics.parameters())
        self.critics_lr_schedule = make_critic_schedule(self.critics_optimizer)

        # move actor and critic to device
        self.actor.to(ptu.device)
        self.actor_target.to(ptu.device)
        self.critics.to(ptu.device)
        self.critics_target.to(ptu.device)

        # bookmark
        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.n_timesteps = n_timesteps
        self.discount = discount
        self.actor_target_update_rate = actor_target_update_rate
        self.critic_target_update_rate = critic_target_update_rate
        self.qloss_weight = eta
        self.max_grad_norm = max_grad_norm
        self.action_min_value = action_min_value
        self.action_max_value = action_max_value

    def clip_sample(self, obs, use_latest_net=True):
        actor = self.actor if use_latest_net else self.actor_target
        action = actor.sample_action(obs)
        action_clipped = torch.clip(action, self.action_min_value, self.action_max_value)

        barred_ratio = ((action >= self.action_max_value) | (action <= self.action_min_value)).sum().item()
        barred_ratio = barred_ratio / np.prod(action.shape)

        return action_clipped, barred_ratio

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Compute the action for a given observation.
        """
        assert observation.shape == self.observation_shape
        observation = ptu.from_numpy(observation)[None]

        # TODO: diffusion model only support 1d state
        observation = observation.reshape(1, -1)

        with torch.no_grad():
            # action = self.actor.sample_action(observation)
            action, _ = self.clip_sample(observation, use_latest_net=True)
            action = action.squeeze(dim=0)

        assert action.shape == (self.action_dim,), \
            f"Expected action shape {(self.action_dim,)}, but got {action.shape}"
        
        # debug
        # print(f"action_mean = {action.mean()}, action_std = {action.std()}")
        return ptu.to_numpy(action)

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ):
        """update two critic using fitted QL
        loss = \| Q(s, a) - r(s, a) - gamma * not_done * Q(s', a') \|^2
        here a' is from target actor

        Args:
            obs (torch.Tensor): _description_
            action (torch.Tensor): _description_
            reward (torch.Tensor): _description_
            next_obs (torch.Tensor): _description_
            done (torch.Tensor): _description_
        """
        assert obs.ndim == 2 and obs.shape[1] == np.prod(self.observation_shape)
        assert next_obs.shape == obs.shape

        # sampling next action
        # compute training target
        not_done = 1.0 - done.float()
        with torch.no_grad():
            # next_action = self.actor_target.sample_action(next_obs)
            next_action, _ = self.clip_sample(next_obs, use_latest_net=False)
            qa0_next = self.critics_target[0].forward(next_obs, next_action)
            qa1_next = self.critics_target[1].forward(next_obs, next_action)
            values = torch.minimum(qa0_next, qa1_next)
            target = reward + self.discount * not_done * values

        # compute predictions
        pred0 = self.critics[0].forward(obs, action)
        pred1 = self.critics[1].forward(obs, action)

        assert target.shape == pred0.shape, \
            f"Expected target shape {pred0.shape}, but got {target.shape}"

        # compute td loss
        loss0 = F.mse_loss(pred0, target)
        loss1 = F.mse_loss(pred1, target)
        loss = (loss0 + loss1) / 2.0

        # perform one step of optimization
        critic_grad_norm = self._optimize_step(
            self.critics.parameters(),
            loss,
            self.critics_optimizer,
            self.critics_lr_schedule,
        )

        # bookmark infos
        infos = {
            "critic_loss": loss.item(),
            "q_values": (pred0.mean() + pred1.mean()).item(),
            "target_values": target.mean().item(),
            "critic_grad_norm": critic_grad_norm.item(),
            "critic_lr": self.critics_lr_schedule.get_last_lr()[0],
        }

        return infos

    def _optimize_step(self, params, loss, optimizer, lr_schedule):
        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        optimizer.step()
        lr_schedule.step()
        return grad_norm

    def update_actor(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        Update the actor by optimize both diffusion loss and Q loss.
        """
        assert obs.ndim == 2 and obs.shape[1] == np.prod(self.observation_shape)
        # compute diffusion loss
        diffuse_loss = self.actor.compute_loss(obs, actions)

        # compute Q loss over critic
        # the Q loss is computed in a reparameterized way
        # actions_sample = self.actor.sample_action(obs)
        actions_sample, cliped_ratio = self.clip_sample(obs, use_latest_net=True)
        qa0 = self.critics[0].forward(obs, actions_sample)
        qa1 = self.critics[1].forward(obs, actions_sample)
        # qreward0 = qa0.mean()
        # qreward1 = qa1.mean()

        coin = torch.rand(1)[0]
        if coin < 0.5:
            qloss = -qa0.mean() / qa1.detach().abs().mean()

        else:
            # qloss = qreward1 / qreward0.detach()
            qloss = -qa1.mean() / qa0.detach().abs().mean()

        # combine the two loss (add a coef parameter)
        loss = diffuse_loss + self.qloss_weight * qloss

        # call optimizer
        actor_grad_norm = self._optimize_step(
            self.actor.parameters(), loss, self.actor_optimizer, self.actor_lr_schedule
        )

        # return train infos
        action_l1_loss = (actions_sample - actions).abs().mean()
        infos = {
            "actor_loss": loss.item(),
            "actor_diffuse_loss": diffuse_loss.item(),
            "actor_qloss": qloss.item(),
            "actor_grad_norm": actor_grad_norm.item(),
            "actor_lr": self.actor_lr_schedule.get_last_lr()[0],
            "actor_action_l1_loss": action_l1_loss.item(),
            "actor_trained_action_clip_ratio": cliped_ratio
        }

        return infos

    def update_target_actor(self):
        ewm_update_weights(self.actor_target, self.actor, self.actor_target_update_rate)

    def update_target_critic(self):
        ewm_update_weights(
            self.critics_target, self.critics, self.critic_target_update_rate
        )

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        """
        Update the actor and critic networks.
        """
        # check shapes
        bsize = observations.shape[0]
        assert observations.shape[1:] == self.observation_shape
        assert actions.shape == (bsize, self.action_dim)
        assert rewards.shape == (bsize,)
        assert next_observations.shape == observations.shape
        assert dones.shape == (bsize,)

        # update critic
        critic_info = self.update_critic(
            observations, actions, rewards, next_observations, dones
        )

        # update actor
        actor_info = self.update_actor(observations, actions)

        # update actor_target and critic target
        self.update_target_critic()
        self.update_target_actor()

        # return infos
        infos = {**actor_info, **critic_info}

        return infos


# when update_rate = 1.0, source paramter copied to the target
def ewm_update_weights(target: nn.Module, source: nn.Module, update_rate: float):
    for param_tgt, param_src in zip(target.parameters(), source.parameters()):
        param_tgt.data.copy_(
            (1.0 - update_rate) * param_tgt.data + update_rate * param_src.data
        )
