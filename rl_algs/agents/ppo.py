from typing import Callable, Sequence, Tuple, Union, Dict, List

import torch
from torch import nn
from torch.functional import F
import numpy as np
import gymnasium as gym

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
        train_epoach=1,
        train_batch_size=128,
        gae_lambda=1.0,
        clip_grad_norm=10.0
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
        self.clip_grad_norm = clip_grad_norm

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

    @torch.no_grad()
    def get_action_and_inner_state(
        self, observation: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """When enconters an observation, returns action, logpi, values

        Args:
            observation (np.ndarray): of shape observation_shape

        Returns:
            Dict[str, np.ndarray]: values of {'action', 'logpi', 'value'}
        """
        assert (
            observation.shape == self.observation_shape
        ), f"observation.shape={observation.shape}"
        observation = ptu.from_numpy(observation)[None]

        action_distribution: torch.distributions.Distribution = self.actor(
            observation
        )  # shape [1, act_dim]
        value = self.critic(observation)  # shape [1, 1]
        action: torch.Tensor = action_distribution.sample()  # [1, act_dim]
        logpi: torch.Tensor = action_distribution.log_prob(
            action
        )  # [1] or [1, act_dim]

        assert value.shape == (1, 1)
        assert action.shape == (1, self.action_dim)
        assert logpi.shape == (1,) or logpi.shape == (1, self.action_dim)

        # remove batch dim
        results = {
            "action": action.squeeze(0),
            "logpi": logpi.squeeze(0),
            "value": value.squeeze(0),
        }

        return ptu.to_numpy(results)

    def update_actor_batch(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        advantage: torch.Tensor,
        logits_old: torch.Tensor,
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
        ratio_clip = torch.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
        loss = torch.minimum(ratio * advantage, ratio_clip * advantage)
        loss = -torch.mean(loss)

        # perform onestep of optimization
        loss.backward()

        # record gradient norm
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.actor.parameters(), self.clip_grad_norm
        )

        self.actor_optimizer.step()
        self.actor_lr_scheduler.step()
        self.actor_optimizer.zero_grad()

        # logging
        clip_fraction = torch.mean(
            (torch.abs(ratio - 1.0) > self.clip_eps).float()
        ).item()

        return {
            "actor_loss": loss.item(),
            "actor_grad_norm": grad_norm.item(),
            "actor_clip_fraction": clip_fraction,
        }

    def update_critic_batch(
        self, obs: torch.Tensor, q_values: torch.Tensor, values_old: torch.Tensor
    ):
        """
        fit v(s) make v(s) predict cost to go at state s
        """
        # compute prediction at states using critic
        values = self.critic(obs)  # [bsize, 1]
        assert values.shape == values_old.shape

        if self.clip_eps_vf is not None:
            diff = values - values_old
            values = values_old + torch.clip(diff, -self.clip_eps_vf, self.clip_eps_vf)

        assert values.shape[-1] == 1
        values = values.squeeze(1)
        assert values.shape == q_values.shape
        # compute mse loss between pred and target
        loss = F.smooth_l1_loss(values, q_values)
        # loss = F.mse_loss(values, q_values)

        # carry one step of optimization
        loss.backward()

        # record critic gradient norm
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm
        )

        self.critic_optimizer.step()
        self.critic_lr_scheduler.step()
        self.critic_optimizer.zero_grad()

        info = {"critic_loss": loss.item(), "critic_grad_norm": grad_norm.item()}
        if self.clip_eps_vf is not None:
            clip_fraction = torch.mean(
                (torch.abs(diff) > self.clip_eps_vf).float()
            ).item()
            info["critic_clip_fraction"] = clip_fraction

        return info

    def apply_optimizer(self, loss, parameters, optimizer, lr_scheduler):
        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(parameters, self.clip_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        return grad_norm

    @staticmethod
    def normalize(adv: torch.Tensor, eps: float) -> torch.Tensor:
        """normalize advantage"""
        assert adv.ndim == 1
        return (adv - adv.mean()) / (adv.std() + eps)


    def update_from_rb(self, rollout_buffer):
        # # debug code----------------
        # values_old = self.critic(ptu.from_numpy(rollout_buffer.observations)).detach()
        # mse = F.mse_loss(values_old, ptu.from_numpy(rollout_buffer.values))
        # assert (mse.item() < 1e-8)

        # pi_old: torch.distributions.Distribution = self.actor(ptu.from_numpy(rollout_buffer.observations))
        # logits_old = pi_old.log_prob(ptu.from_numpy(rollout_buffer.actions)).detach()
        # mse = F.mse_loss(logits_old, ptu.from_numpy(rollout_buffer.logits))
        # assert (mse.item() < 1e-8)
        # # debug code----------------

        infos = []
        for batch in rollout_buffer.sample_batch(self.train_batch_size, self.train_epoach):
            batch = ptu.from_numpy(batch)
            info = self.update_on_batch(**batch)
            infos.append(info)
        
        info_agg = {key: np.mean(list(info[key] for info in infos)) for key in infos[0]}

        return info_agg

    def update_on_batch(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        returns: torch.Tensor,
        dones: torch.Tensor,
        logits_old: torch.Tensor,
        advantages: torch.Tensor,
        values_old: torch.Tensor    
    ):
        info_actor = self.update_actor_batch(
            observations,
            actions,
            advantages,
            logits_old,
        )

        info_critic = self.update_critic_batch(
            observations, returns, values_old
        )
        
        return {**info_actor, **info_critic}


class RolloutBuffer:
    def __init__(self, max_length: int, gamma: float, gae_gamma: float, obs_type = np.float32):
        self.observations = None
        self.next_observations = None
        self.actions = None
        self.rewards = None
        self.returns = None
        self.dones = None
        self.logits = None
        self.advantages = None
        self.values = None

        self.max_length = max_length
        self.gamma = gamma
        self.gae_gamma = gae_gamma

        self.obs_type = obs_type
        self.done_type = np.float32
        self.rew_type = np.float32


    def _rollout_one_episode(self, agent: PPO, env: gym.Env) -> Dict[str, List]:

        # episode infos
        results = {
            "observations": [],
            "next_observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "logits": [],
            "values": [],
        }

        # episode start
        ob, init_info = env.reset()
        steps = 0
        loop_finish = False
        while not loop_finish:
            # step using agent
            act_state = agent.get_action_and_inner_state(ob)
            action, logpi, value = (
                act_state["action"],
                act_state["logpi"],
                act_state["value"],
            )

            next_ob, reward, terminated, truncated, info = env.step(action)

            # record episode
            results["observations"].append(ob)
            results["actions"].append(action)
            results["rewards"].append(reward)
            results["next_observations"].append(next_ob)
            results["dones"].append(terminated)
            results["logits"].append(logpi)
            results["values"].append(value)

            # update loop iterators
            ob = next_ob
            steps += 1
            loop_finish = terminated or truncated or steps >= self.max_length

        assert len(results["observations"]) == steps
        return results

    def _rollout_util_batchsize(self, agent: PPO, env: gym.Env, min_steps_per_batch: int) -> Dict[str, List]:
        """ rollout episodes util we get at least min_steps data points
        """
        trajs = {
            "observations": [],
            "next_observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "logits": [],
            "values": [],
        }

        while len(trajs['observations']) < min_steps_per_batch:
            traj = self._rollout_one_episode(agent, env)

            for key in trajs:
                trajs[key].extend(traj[key])
        
        return trajs
    
    def rollout(self, agent: PPO, env: gym.Env, min_steps_per_batch: int):
        """ rollout and construct buffers for training
        """
        trajs = self._rollout_util_batchsize(agent, env, min_steps_per_batch)
        self.observations = np.array(trajs['observations'])
        self.next_observations = np.array(trajs['next_observations'])
        self.actions = np.array(trajs['actions'])
        self.rewards = np.array(trajs['rewards'])
        self.dones = np.array(trajs['dones'])
        self.logits = np.array(trajs['logits'])
        self.values = np.array(trajs['values'])

        # handle types
        self.observations = self.observations.astype(self.obs_type)
        self.next_observations = self.next_observations.astype(self.obs_type)
        self.dones = self.dones.astype(self.done_type)
        self.rewards = self.rewards.astype(self.rew_type)

        assert self.actions.dtype == np.float32 or self.actions.dtype == np.int64
        assert self.logits.dtype == np.float32
        assert self.values.dtype == np.float32

        # compute advantages
        self.advantages, self.returns = compute_gae_advantage(self.rewards, self.values.reshape((-1,)), self.dones, self.gamma, self.gae_gamma)
        assert self.advantages.dtype == np.float32
        assert self.returns.dtype == np.float32

        # return rollout steps
        return len(self.observations)
    
    def sample_batch(self, batch_size, epoch):
        for _ in range(epoch):
            # generate shuffled indices
            n = len(self.observations)
            indices = np.random.permutation(n)

            # handle each batch
            for i in range(0, n, batch_size):
                batch_idx = indices[i : i + batch_size]
                batch = {
                    'observations': self.observations[batch_idx],
                    'actions': self.actions[batch_idx],
                    'rewards': self.rewards[batch_idx],
                    'returns': self.returns[batch_idx],
                    'dones':   self.dones[batch_idx],
                    'logits_old': self.logits[batch_idx],
                    'advantages': self.advantages[batch_idx],
                    'values_old': self.values[batch_idx]
                }

                yield batch