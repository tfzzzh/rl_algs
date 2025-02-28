from typing import Callable, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.functional import F
import numpy as np
import rl_algs.utility.pytorch_util as ptu


class CrossQ:
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,
        make_actor: Callable[[Tuple[int, ...], int], nn.Module],
        make_actor_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_actor_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_critic_optimizer: Callable[
            [torch.nn.ParameterList], torch.optim.Optimizer
        ],
        make_critic_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        # Actor-critic configuration (use reparameterize trick)
        num_actor_samples: int = 1,
        num_critic_updates: int = 1,
        # Settings for multiple critics
        num_critic_networks: int = 1,
        use_entropy_bonus: bool = False,
        temperature: float = 0.0,
        backup_entropy: bool = True,
        use_reparameter_entropy = True
    ):
        # create actor
        self.actor = make_actor(observation_shape, action_dim)
        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.actor_lr_scheduler = make_actor_schedule(self.actor_optimizer)

        # create critics
        self.critics = nn.ModuleList([
            make_critic(observation_shape, action_dim)
            for _ in range(num_critic_networks)
        ])
        # self.critics = nn.ModuleList(
        #     [BNStateActionCritic(critic) for critic in critics]
        # )

        self.critic_optimizer = make_critic_optimizer(self.critics.parameters())
        self.critic_lr_scheduler = make_critic_schedule(self.critic_optimizer)

        # bookmark infos
        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.discount = discount
        self.num_actor_samples = num_actor_samples
        self.num_critic_updates = num_critic_updates
        self.use_entropy_bonus = use_entropy_bonus
        self.temperature = temperature
        self.backup_entropy = backup_entropy
        self.num_critic_networks = num_critic_networks
        self.use_reparameter_entropy = use_reparameter_entropy
        self._num_update = 0

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Compute the action for a given observation.
        """
        assert observation.shape == self.observation_shape
        observation = ptu.from_numpy(observation)[None]
        # set eval mode for actor
        self.actor.eval()

        dist: torch.distributions.Distribution = self.actor(observation)
        action = dist.sample()
        assert action.shape == (1, self.action_dim)

        # reset eval mode
        self.actor.train()

        return ptu.to_numpy(action.squeeze(0))

    def critic(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute the (ensembled) Q-values for the given state-action pair.
        returned tensor size [num_critic, bsize]
        keep: Q-nets is in train mode only in q-iteration

        output shape: [num_critic, batch_size]
        """
        return torch.stack([qnet(obs, action) for qnet in self.critics], dim=0)

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ):
        """
        Update the critic networks by computing target values and minimizing Bellman error.
        """
        batch_size = obs.shape[0]
        # when update critic shall in train mode
        self.critics.train()

        # sampling next_action
        self.actor.eval()
        next_pi = self.actor(next_obs)
        next_action = next_pi.sample()
        self.actor.train()

        # compute q-value with critic in train mode
        q_full = self.critic(
            torch.concat([obs, next_obs], dim=0),
            torch.concat([action, next_action], dim=0),
        )
        q_curr, q_next = q_full.split(batch_size, dim=1)

        assert q_curr.shape == q_next.shape
        assert q_curr.shape == (self.num_critic_networks, batch_size)

        # compute target; target = r + gamma * Q(s', a')
        with torch.no_grad():
            q_next = torch.min(q_next, dim=0).values  # [batchsize,]
            assert q_next.shape == (batch_size, )

            # when use entropy to guide actor in q-net
            if self.use_entropy_bonus and self.backup_entropy:
                next_action_entropy = self.entropy(next_pi)
                q_next += self.temperature * next_action_entropy

            target = reward + self.discount * (1.0 - done.float()) * q_next
            target = target[None].broadcast_to(q_curr.shape)

        # perform one-step of q-iteration
        loss = F.mse_loss(q_curr, target.detach()) # add detach change the result?

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        self.critic_lr_scheduler.step()

        return {
            "critic_loss": loss.item(),
            "q_values": q_curr.mean().item(),
            "target_values": target.mean().item(),
        }

    def entropy(self, action_distribution: torch.distributions.Distribution):
        """
        Compute the (approximate) entropy of the action distribution for each batch element.
        """
        # Note: Think about whether to use .rsample() or .sample() here...
        if self.use_reparameter_entropy:
            action = action_distribution.rsample()
            logpi = action_distribution.log_prob(action)
            # assert torch.all(logpi < 1e-4)
            return -logpi  # each action only sample one point

        return action_distribution.entropy()

    def actor_loss_reparametrize(self, obs: torch.Tensor):
        """
        TODO: shall I use eval mode for critic (yes)
        """
        # Sample from the actor
        action_distribution: torch.distributions.Distribution = self.actor(obs)

        # TODO(student): Sample actions
        # Note: Think about whether to use .rsample() or .sample() here...
        action = action_distribution.rsample()  # [bsize, act_dim]

        # TODO(student): Compute Q-values for the sampled state-action pair
        self.critics.eval()
        q_values = self.critic(obs, action)
        q_values = torch.min(q_values, dim=0).values
        self.critics.train()
        assert q_values.shape[0] == obs.shape[0] and q_values.ndim == 1

        # TODO(student): Compute the actor loss
        loss = -torch.mean(q_values)  # we need to maximize the q_value

        return loss, torch.mean(self.entropy(action_distribution))

    def update_actor(self, obs: torch.Tensor):
        """
        Update the actor by one gradient step
        """
        self.actor.train()
        loss, entropy = self.actor_loss_reparametrize(obs)

        # the actor shall generate distribution max expected qvalue and entropy
        if self.use_entropy_bonus:
            loss -= self.temperature * entropy

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_scheduler.step()

        return {"actor_loss": loss.item(), "entropy": entropy.item()}

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
        # update critic
        critic_infos = []
        for _ in range(self.num_critic_updates):
            critic_infos.append(
                self.update_critic(
                    observations, actions, rewards, next_observations, dones
                )
            )

        critic_info = {
            k: np.mean([info[k] for info in critic_infos]) for k in critic_infos[0]
        }

        # update actor (change back)
        actor_info = {}
        if (self._num_update + 1) % 3 == 0:
            actor_info = self.update_actor(observations)

        self._num_update += 1

        # return train info
        update_info = {
            **actor_info,
            **critic_info,
            "actor_lr": self.actor_lr_scheduler.get_last_lr()[0],
            "critic_lr": self.critic_lr_scheduler.get_last_lr()[0],
        }

        return update_info
