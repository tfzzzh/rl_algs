from typing import Callable, Sequence, Tuple, Union

import torch
from torch import nn
from torch.functional import F
import numpy as np

from .common import compute_gae_advantage
import rl_algs.utility.pytorch_util as ptu


class PPO:
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
        normalize_advantage: bool = False,
        clip_eps=0.2,
        clip_eps_vf: Union[None, float] = None,
        eps=1e-8,
        train_epoach = 1,
        train_batch_size = 128,
        gae_lambda = 1.0
    ):
        # check if data is valid
        assert discount < 1.0 and discount > 0.0, f"discount = {discount}"

        # create actor and its optimizer
        self.actor: nn.Module = make_actor(observation_shape, action_dim)
        self.actor_optimizer: torch.optim.Optimizer = make_actor_optimizer(
            self.actor.parameters()
        )
        self.actor_lr_scheduler = make_actor_schedule(self.actor_optimizer)

        # create reference actor
        # self.reference_actor: nn.Module = make_actor(observation_shape, action_dim)

        # create critic and crtic optimizer
        # assumption self.critic(x) is a tensor of shape [bsize, 1]
        self.critic: nn.Module = make_critic(observation_shape)
        self.critic_optimizer: torch.optim.Optimizer = make_critic_optimizer(
            self.critic.parameters()
        )
        self.critic_lr_scheduler = make_critic_schedule(self.critic_optimizer)

        # bookmark other terms
        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.discount = discount
        # self.reference_update_period = reference_update_period
        self.normalize_advantage = normalize_advantage
        self.clip_eps = clip_eps
        self.clip_eps_vf = clip_eps_vf
        self.eps = eps
        self.train_epoach = train_epoach
        self.train_batch_size = train_batch_size
        self.gae_lambda = gae_lambda

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Compute the action for a given observation.
        """
        assert (
            observation.shape == self.observation_shape
        ), f"observation.shape={observation.shape}"
        observation = ptu.from_numpy(observation)[None]

        action_distribution: torch.distributions.Distribution = self.actor(observation)
        action: torch.Tensor = action_distribution.sample()

        assert action.shape == (1, self.action_dim), action.shape
        return ptu.to_numpy(action).squeeze(0)

    def update(
        self,
        obs: Sequence[np.ndarray],
        actions: Sequence[np.ndarray],
        rewards: Sequence[np.ndarray],
        terminals: Sequence[np.ndarray],
        step: int
    ):
        """
        given sampled paths update agent parameters
        """
        # check inputs are not empty
        assert len(obs) > 0 and obs[0].shape[1:] == self.observation_shape, \
            f"obs has shape {obs[0].shape[1:]} while expect shape {self.observation_shape}"

        # concatenate obs, actions, rewards, terminals and q_values
        obs = np.concatenate(obs, axis=0)
        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        terminals = np.concatenate(terminals, axis=0)

        # compute reward to go (namely q_values)
        assert len(rewards.shape) == 1
        # q_values = self._compute_reward_to_go(rewards, terminals)

        # move above ndarrays to tensor
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        rewards = ptu.from_numpy(rewards)
        terminals = ptu.from_numpy(terminals)
        # q_values = ptu.from_numpy(q_values)

        # bookmark old policy and old value
        pi_old: torch.distributions.Distribution = self.actor(obs)
        logits_old = pi_old.log_prob(actions).detach()
        values_old = self.critic(obs).detach() # [bsize, 1]


        # compute advantage using these informations
        # adv = self._compute_advantage(obs, rewards, q_values, terminals)
        adv, q_values = compute_gae_advantage(ptu.to_numpy(rewards), ptu.to_numpy(values_old).reshape((-1,)), ptu.to_numpy(terminals), self.discount, self.gae_lambda)
        adv = ptu.from_numpy(adv)
        q_values = ptu.from_numpy(q_values)

        # update
        update_info = self.update_actor_critic(obs, actions, q_values, adv, logits_old, values_old)

        # use advantage to update actors
        #actor_info = self.update_actor(obs, actions, adv)

        # update reference actor when (step + 1) % reference_update_period == 0
        #if (step + 1) % self.reference_update_period == 0:
        # self.update_reference_actor()

        # use q_values (reward to go) to update critic
        #critic_info = self.update_critic(obs, q_values)

        # return udpate infos
        return update_info

    def update_actor_batch(
        self, obs: torch.Tensor, actions: torch.Tensor, advantage: torch.Tensor, logits_old: torch.Tensor
    ):
        # get policies at current state
        pi: torch.distributions.Distribution = self.actor(obs)
        # pi_ref: torch.distributions.Distribution = self.reference_actor(obs)

        # compute log-probs
        # compute log pi(a | s)
        # compute log pi_old(a | s)
        # actions = torch.tensor([ 0.9691, -0.3325, -0.6839,  1.0,  0.98, -0.9231], device='cuda:0') -> nan
        logits = pi.log_prob(actions)  # (bsize,)
        # logits_ref = pi_ref.log_prob(actions).detach() # shall not use gradient
        assert logits.ndim == 1 and logits.shape == logits_old.shape

        # compute prob ratio of current policy over the reference
        ratio = torch.exp(logits - logits_old)
        assert ratio.shape == advantage.shape
        if self.normalize_advantage:
            advantage = PPO.normalize(advantage, self.eps)

        # compute loss = min(ratio, clip(ratio, 1-eps, 1+eps)) * Advantage
        loss = torch.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
        loss = torch.minimum(ratio * advantage, loss * advantage)
        loss = -torch.mean(loss)

        # perform onestep of optimization
        loss.backward()
        
        # record gradient norm
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.actor.parameters(), 10.0
        )

        self.actor_optimizer.step()
        self.actor_lr_scheduler.step()
        self.actor_optimizer.zero_grad()

        # logging
        clip_fraction = torch.mean((torch.abs(ratio - 1.0) > self.clip_eps).float()).item()

        return {"actor_loss": loss.item(), "actor_grad_norm": grad_norm.item(), "actor_clip_fraction": clip_fraction}
    
    # def update_actor(
    #     self, obs: torch.Tensor, actions: torch.Tensor, advantage: torch.Tensor
    # ):
    #     infos = []
    #     n = len(obs)
    #     for _ in range(self.train_epoach):
    #         # generate shuffled indices
    #         indices = np.random.permutation(n)

    #         # handle each batch
    #         for i in range(0, n, self.train_batch_size):
    #             batch_idx = indices[i: i+self.train_batch_size]
    #             info = self.update_actor_batch(obs[batch_idx], actions[batch_idx], advantage[batch_idx])
    #             infos.append(info)

    #     info_agg = {key: np.mean(list(info[key] for info in infos)) for key in infos[0]}

    #     return info_agg

    def update_reference_actor(self):
        self.reference_actor.load_state_dict(self.actor.state_dict())

    def update_critic_batch(self, obs: torch.Tensor, q_values: torch.Tensor, values_old: torch.Tensor):
        """
        fit v(s) make v(s) predict cost to go at state s
        """
        # compute prediction at states using critic
        values = self.critic(obs)  # [bsize, 1]

        if self.clip_eps_vf is not None:
            diff = values - values_old
            values = values_old + torch.clip(diff, -self.clip_eps_vf, self.clip_eps_vf)

        assert values.shape[-1] == 1
        values = values.squeeze(1)
        assert values.shape == q_values.shape
        # compute mse loss between pred and target
        loss = F.smooth_l1_loss(values, q_values)

        # carry one step of optimization
        loss.backward()

        # record critic gradient norm
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), 10.0
        )

        self.critic_optimizer.step()
        self.critic_lr_scheduler.step()
        self.critic_optimizer.zero_grad()

        info = {"critic_loss": loss.item(),  "critic_grad_norm": grad_norm.item()}
        if self.clip_eps_vf is not None:
            clip_fraction = torch.mean((torch.abs(diff) > self.clip_eps_vf).float()).item()
            info['critic_clip_fraction'] = clip_fraction

        return info
    
    # def update_critic(
    #     self, obs: torch.Tensor, q_values: torch.Tensor
    # ):
    #     infos = []
    #     n = len(obs)
    #     for _ in range(self.train_epoach):
    #         # generate shuffled indices
    #         indices = np.random.permutation(n)

    #         # handle each batch
    #         for i in range(0, n, self.train_batch_size):
    #             batch_idx = indices[i: i+self.train_batch_size]
    #             info = self.update_critic_batch(obs[batch_idx], q_values[batch_idx])
    #             infos.append(info)

    #     info_agg = {key: np.mean(list(info[key] for info in infos)) for key in infos[0]}

    #     return info_agg
    
    def update_actor_critic(
        self, obs: torch.Tensor, actions: torch.Tensor, q_values: torch.Tensor, advantage: torch.Tensor,
        logits_old: torch.Tensor, values_old: torch.Tensor
    ):
        infos = []
        n = len(obs)

        # book mark old values
        # values_old = self.critic(obs).detach() # [bsize, 1]

        for _ in range(self.train_epoach):
            # generate shuffled indices
            indices = np.random.permutation(n)

            # handle each batch
            for i in range(0, n, self.train_batch_size):
                batch_idx = indices[i: i+self.train_batch_size]
                info_actor = self.update_actor_batch(obs[batch_idx], actions[batch_idx], advantage[batch_idx], logits_old[batch_idx])
                info_critic = self.update_critic_batch(obs[batch_idx], q_values[batch_idx], values_old[batch_idx])
                infos.append({**info_actor, **info_critic})

        info_agg = {key: np.mean(list(info[key] for info in infos)) for key in infos[0]}

        return info_agg

    @torch.no_grad()
    def _compute_advantage(
        self,
        obs: torch.Tensor,
        rewards: torch.Tensor,
        q_values: torch.Tensor,
        terminals: torch.Tensor,
    ):
        """
        compute advantage shall not result in gradient computation
        """
        # compute v(s) using critic
        values = self.critic(obs)

        # reshape v(s) to of shape (bsize, )
        assert values.ndim == 2 and values.shape[1] == 1
        values = values.squeeze(1)
        assert values.shape == q_values.shape

        # advantage = Q(s, a) - v(s)
        advantage = q_values - values

        # normalize advantage if required
        # if self.normalize_advantage:
        #     advantage = PPO.normalize(advantage, self.eps)

        return advantage    

    def _compute_reward_to_go(
        self, rewards: np.ndarray, terminals: np.ndarray
    ) -> np.ndarray:
        # check reward and terminal has the same shape
        # check both reward and terminal not empty
        # check terminal contains bool element
        assert len(rewards.shape) == 1
        assert rewards.shape == terminals.shape
        assert len(rewards) > 0
        assert isinstance(terminals[0], np.float32), f"terminal is of type {type(terminals[0])}"
        n = len(rewards)

        rtgs = []
        rtg = 0.0

        for i in range(n - 1, -1, -1):
            assert terminals[i] == 0.0 or terminals[i] == 1.0
            rtg = (1.0 - terminals[i]) * rtg

            rtg = rewards[i] + self.discount * rtg
            rtgs.append(rtg)

        rtgs.reverse()
        return np.array(rtgs)

    @staticmethod
    def normalize(adv: torch.Tensor, eps: float) -> torch.Tensor:
        """normalize advantage"""
        assert adv.ndim == 1
        return (adv - adv.mean()) / (adv.std() + eps) 
