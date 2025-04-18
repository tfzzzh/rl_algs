from typing import Callable, Sequence, Tuple, Union, Dict, List, Optional

import torch
from torch import nn
from torch.functional import F
import numpy as np
import gymnasium as gym

from .common import compute_gae_advantage, explained_variance
import rl_algs.utility.pytorch_util as ptu
from rl_algs.networks.atari import CNNActorCritic


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
        clip_grad_norm=10.0,
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
        for batch in rollout_buffer.sample_batch(
            self.train_batch_size, self.train_epoach
        ):
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
        values_old: torch.Tensor,
    ):
        info_actor = self.update_actor_batch(
            observations,
            actions,
            advantages,
            logits_old,
        )

        info_critic = self.update_critic_batch(observations, returns, values_old)

        return {**info_actor, **info_critic}


class PPOAtari:
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_action: int,
        make_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_lr_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        share_encoder: bool = True,
        discount: float = 0.99,
        normalize_advantage: bool = True,
        clip_eps=0.2,
        clip_eps_vf: Optional[float] = None,
        eps=1e-8,
        train_epoach=1,
        train_batch_size=128,
        gae_lambda=1.0,
        clip_grad_norm=10.0,
        critic_loss_weight=1.0,
        entropy_weight=0.01,
    ):
        # check if data is valid
        assert discount < 1.0 and discount > 0.0, f"discount = {discount}"
        assert observation_shape == (4, 84, 84)

        self.model = CNNActorCritic(num_action, share_encoder)
        self.model.to(ptu.device)

        self.optimizer = make_optimizer(self.model.parameters())
        self.lr_scheduler = make_lr_schedule(self.optimizer)

        # bookmark other terms
        self.observation_shape = observation_shape
        self.num_action = num_action
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
        self.critic_loss_weight = critic_loss_weight
        self.entropy_weight = entropy_weight

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Compute the action for a given observation.
        """
        assert (
            observation.shape == self.observation_shape
        ), f"observation.shape={observation.shape}"
        observation = ptu.from_numpy(observation)[None]

        action_distribution, _ = self.model(observation)
        action: torch.Tensor = action_distribution.sample()

        assert action.shape == (1,), action.shape
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

        action_distribution, value = self.model(observation)
        action: torch.Tensor = action_distribution.sample()

        logpi: torch.Tensor = action_distribution.log_prob(
            action
        )  # [1] or [1, act_dim]

        assert value.shape == (1, 1) or value.shape == (1,)
        assert action.shape == (1,)
        assert logpi.shape == (1,)
        # remove batch dim
        results = {
            "action": action.squeeze(0),
            "logpi": logpi.squeeze(0),
            "value": value.squeeze(0),
        }

        return ptu.to_numpy(results)

    def compute_actor_loss(
        self,
        pi: torch.distributions.Categorical,
        actions: torch.Tensor,
        advantage: torch.Tensor,
        logpi_old: torch.Tensor,
    ):
        """compute ppo' actor loss and entropy

        Args:
            pi (torch.distributions.Categorical): policy generated by actor
            actions (torch.Tensor): action from rollout buffer
            advantage (torch.Tensor): advantage at current state
            logpi_old (torch.Tensor): previous policy

        Returns:
            actor_loss, entropy, likelihood_ratio
        """
        # check shape
        bsize = actions.shape[0]
        assert actions.shape == (bsize,)
        assert advantage.shape == (bsize,)
        assert logpi_old.shape == (bsize,)

        # compute log policy
        log_pi = pi.log_prob(actions)

        # compute likelihood ratio
        ratio = torch.exp(log_pi - logpi_old)
        ratio_clip = torch.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
        assert ratio.shape == (actions.shape[0],)  # [batchsize,]

        # normalize advantage
        if self.normalize_advantage:
            advantage = (advantage - advantage.mean()) / (advantage.std() + self.eps)
            # print(advantage)

        # ppo_gain = min(ratio * advantage, ratio_clip * advantage)
        ppo_loss = -torch.minimum(ratio * advantage, ratio_clip * advantage)
        ppo_loss = torch.mean(ppo_loss)

        # entropy
        entropy = pi.entropy()
        entropy = torch.mean(entropy)

        # return ppo loss + entropy
        return ppo_loss, entropy, ratio

    def compute_critic_loss(
        self, values: torch.Tensor, returns: torch.Tensor, values_old: torch.Tensor
    ):
        assert values.shape == values_old.shape
        assert values.shape == returns.shape

        # compute clipped prediction
        diff = values - values_old
        if self.clip_eps_vf is not None:
            diff_clip = torch.clip(diff, -self.clip_eps_vf, self.clip_eps_vf)
            pred = values_old + diff_clip
        else:
            pred = values

        # use l2 loss
        loss = F.mse_loss(pred, returns)

        return loss, diff

    def apply_optimizer(self, loss, parameters, optimizer, lr_scheduler):
        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(parameters, self.clip_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        return grad_norm

    def update_on_batch(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        returns: torch.Tensor,
        dones: torch.Tensor,
        logits_old: torch.Tensor,
        advantages: torch.Tensor,
        values_old: torch.Tensor,
    ):
        # get policy and value estimates
        policy, values = self.model(observations)

        # get losses from actor and critic
        actor_loss, actor_entropy, ratio = self.compute_actor_loss(
            policy, actions, advantages, logits_old
        )
        critic_loss, diff = self.compute_critic_loss(values, returns, values_old)

        # compose combined loss
        loss = (
            actor_loss
            + self.critic_loss_weight * critic_loss
            - self.entropy_weight * actor_entropy
        )

        # apply optimizer
        grad_norm = self.apply_optimizer(
            loss, self.model.parameters(), self.optimizer, self.lr_scheduler
        )

        # report infos
        infos = {
            "loss/loss": loss.item(),
            "loss/actor": actor_loss.item(),
            "loss/critic_loss": critic_loss.item(),
            "loss/entropy": actor_entropy.item(),
            "grad_norm": grad_norm.item(),
        }

        ## compute percent of clipped likelihood ratio
        infos["clip_fraction/actor"] = torch.mean(
            (torch.abs(ratio - 1.0) > self.clip_eps).float()
        ).item()

        ## compute percent of clipped values
        if self.clip_eps_vf is not None:
            infos["clip_fraction/critic"] = torch.mean(
                (torch.abs(diff) > self.clip_eps_vf).float().float()
            ).item()

        return infos

    def update_from_rb(self, rollout_buffer):
        infos = []

        for batch in rollout_buffer.sample_batch(
            self.train_batch_size, self.train_epoach
        ):
            batch = ptu.from_numpy(batch)
            info = self.update_on_batch(**batch)
            infos.append(info)

        info_agg = {key: np.mean(list(info[key] for info in infos)) for key in infos[0]}
        # compute stats from rb
        info_agg['explained_var'] = explained_variance(rollout_buffer.values.flatten(), rollout_buffer.returns.flatten())

        return info_agg

    def save_checkpoint(self, path: str):
        """
        save agent's state to a checkpoint file
        """
        checkpoint = {
            # model ckp
            "model_state_dict": self.model.state_dict(),
            # optimizer ckp
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_schedulerstate_dict": self.lr_scheduler.state_dict(),
        }

        torch.save(checkpoint, path)
        print(f"save model to checkpoint: {path}")

    def load_from_checkpoint(self, path, load_optimizer=False):
        """
        load model and optimizer state from disk,
        """
        checkpoint = torch.load(path)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if load_optimizer:
            print("load optimizer's states")
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            self.lr_scheduler.load_state_dict(checkpoint["lr_schedulerstate_dict"])

        # check model device
        assert next(self.model.parameters()).device == ptu.device


###################
# RolloutBuffer
###################
class RolloutBuffer:
    def __init__(
        self, max_length: int, gamma: float, gae_gamma: float, obs_type=np.float32,
        reward_trans = None,
    ):
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

        self.reward_trans = reward_trans

        self.obs_type = obs_type
        self.done_type = np.float32
        self.rew_type = np.float32

        self.episode_info = {'returns':[], 'lens': []}

        self._obs = None

    def _rollout_one_episode(self, agent: PPO, env: gym.Env, rollout_length) -> Dict[str, List]:
        assert rollout_length > 0

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
        if self._obs is None:
            ob, init_info = env.reset()
            self._obs = ob
        else:
            ob = self._obs

        steps = 0
        loop_finish = steps >= rollout_length
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
            loop_finish = terminated or truncated or steps >= self.max_length or steps >= rollout_length

        assert len(results["observations"]) == steps

        # add V(ST) to last reward
        act_state_last = agent.get_action_and_inner_state(ob)
        if act_state_last["value"].ndim == 0:
            results['rewards'][-1] += self.gamma * act_state_last["value"]
        else:
            results['rewards'][-1] += self.gamma * act_state_last["value"][0]

        if terminated or truncated:
            self._obs = None
        else:
            self._obs = ob

        if 'episode' in info:
            self.episode_info['returns'].append(info['episode']['r'])
            self.episode_info['lens'].append(info['episode']['l'])
        return results

    def _rollout_util_batchsize(
        self, agent: PPO, env: gym.Env, min_steps_per_batch: int
    ) -> Dict[str, List]:
        """rollout episodes util we get at least min_steps data points"""
        trajs = {
            "observations": [],
            "next_observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "logits": [],
            "values": [],
        }

        length_to_roll = min_steps_per_batch
        while len(trajs["observations"]) < min_steps_per_batch:
            traj = self._rollout_one_episode(agent, env, length_to_roll)

            for key in trajs:
                trajs[key].extend(traj[key])

            length_to_roll -= len(traj['observations'])

        return trajs

    def rollout(self, agent: PPO, env: gym.Env, min_steps_per_batch: int):
        """rollout and construct buffers for training"""
        self.episode_info = {'returns':[], 'lens': []}
        trajs = self._rollout_util_batchsize(agent, env, min_steps_per_batch)
        self.observations = np.array(trajs["observations"])
        self.next_observations = np.array(trajs["next_observations"])
        self.actions = np.array(trajs["actions"])
        self.rewards = np.array(trajs["rewards"])
        self.dones = np.array(trajs["dones"])
        self.logits = np.array(trajs["logits"])
        self.values = np.array(trajs["values"])

        # handle types
        self.observations = self.observations.astype(self.obs_type)
        self.next_observations = self.next_observations.astype(self.obs_type)
        self.dones = self.dones.astype(self.done_type)
        self.rewards = self.rewards.astype(self.rew_type)

        # transforms
        if self.reward_trans is not None:
            self.rewards = self.reward_trans(self.rewards).astype(self.rew_type)

        assert self.actions.dtype == np.float32 or self.actions.dtype == np.int64
        assert self.logits.dtype == np.float32
        assert self.values.dtype == np.float32

        # compute advantages
        self.advantages, self.returns = compute_gae_advantage(
            self.rewards,
            self.values.reshape((-1,)),
            self.dones,
            self.gamma,
            self.gae_gamma,
            recomputed_returns=True
        )
        assert self.advantages.dtype == np.float32
        assert self.returns.dtype == np.float32

        # return rollout steps
        ep_info = {}
        if len(self.episode_info['returns']) > 0:
            ep_info['rollout/returns'] = np.mean(self.episode_info['returns'])
            ep_info['rollout/lens'] = np.mean(self.episode_info['lens'])

        return len(self.observations), ep_info
    
    def sample_batch(self, batch_size, epoch):
        for _ in range(epoch):
            # generate shuffled indices
            n = len(self.observations)
            indices = np.random.permutation(n)

            # handle each batch
            for i in range(0, n, batch_size):
                if i + batch_size <= n:
                    batch_idx = indices[i : i + batch_size]

                else:
                    num_tail = n - i
                    num_head = batch_size - num_tail
                    batch_idx = np.concat([indices[i : n], indices[0: num_head]], axis=0)

                assert len(batch_idx) == batch_size
                batch = {
                    "observations": self.observations[batch_idx],
                    "actions": self.actions[batch_idx],
                    "rewards": self.rewards[batch_idx],
                    "returns": self.returns[batch_idx],
                    "dones": self.dones[batch_idx],
                    "logits_old": self.logits[batch_idx],
                    "advantages": self.advantages[batch_idx],
                    "values_old": self.values[batch_idx],
                }

                yield batch


class RolloutBufferMultEnvs:
    def __init__(
        self, 
        rollout_batch: int, gamma: float, gae_gamma: float, obs_type=np.float32,
        num_envs: int = 1
    ):
        self.observations = None
        self.next_observations = None
        self.actions = None
        self.rewards = None
        self.returns = None
        self.dones = None
        self.logits = None
        self.advantages = None
        self.values = None

        self.rollout_batch = rollout_batch
        self.gamma = gamma
        self.gae_gamma = gae_gamma

        self.obs_type = obs_type
        self.done_type = np.float32
        self.rew_type = np.float32
        self.num_envs = num_envs

        self.episode_info = {'returns':[], 'lens': []}

        self._obs = None

    @classmethod
    def create_buffer(cls):
        buffer = {
            "observations": [],
            "next_observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "logits": [],
            "values": [],
        }

        return buffer

    def _rollout_one_episode(self, agent: PPO, env: gym.Env, idx: int) -> Dict[str, List]:
        """ rollout one episode for rollout_batch steps

        Args:
            agent (PPO): _description_
            env (gym.Env): _description_
            idx (int): _description_

        Returns:
            Dict[str, List]: _description_
        """

        # episode infos
        results = self.create_buffer()

        ob = self._obs[idx]

        steps = 0
        while steps < self.rollout_batch:
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
            done = terminated or truncated

            if done:
                # end of current episode
                act_state_last = agent.get_action_and_inner_state(ob)
                if act_state_last["value"].ndim == 0:
                    results['rewards'][-1] += self.gamma * (1.0 - float(terminated)) * act_state_last["value"]
                else:
                    results['rewards'][-1] += self.gamma * (1.0 - float(terminated)) * act_state_last["value"][0]

                if 'episode' in info:
                    self.episode_info['returns'].append(info['episode']['r'])
                    self.episode_info['lens'].append(info['episode']['l'])

                # reset
                ob, _ = env.reset()


        assert len(results["observations"]) == self.rollout_batch

        # add V(ST) to last reward
        # when done is true ob is the first observation of next episode, its value
        # shall not add to previous path, however, when done is not true, ob is
        # the last obervation of current episode
        if not done:
            act_state_last = agent.get_action_and_inner_state(ob)
            if act_state_last["value"].ndim == 0:
                results['rewards'][-1] += self.gamma * (1.0 - float(terminated)) * act_state_last["value"]
            else:
                results['rewards'][-1] += self.gamma * (1.0 - float(terminated)) * act_state_last["value"][0]

        # for split different episodes
        results['dones'][-1] = True

        self._obs[idx] = ob
        return results
    
    def _get_trajs(self, agent: PPO, envs: List[gym.Env]):

        # init start observations
        if self._obs is None:
            self._obs = []
            for env in envs:
                obs_init, _ = env.reset()
                self._obs.append(obs_init)

        # init trajs buffer
        trajs = self.create_buffer()

        # roll over each envs
        for i in range(self.num_envs):
            traj = self._rollout_one_episode(agent, envs[i], i)

            for key in traj:
                trajs[key].extend(traj[key])  

        return trajs

    def rollout(self, agent: PPO, envs: List[gym.Env]):
        """rollout and construct buffers for training"""
        assert len(envs) == self.num_envs
        # refine it
        self.episode_info = {'returns':[], 'lens': []}
        trajs = self._get_trajs(agent, envs) 

        self.observations = np.array(trajs["observations"])
        self.next_observations = np.array(trajs["next_observations"])
        self.actions = np.array(trajs["actions"])
        self.rewards = np.array(trajs["rewards"])
        self.dones = np.array(trajs["dones"])
        self.logits = np.array(trajs["logits"])
        self.values = np.array(trajs["values"])

        # handle types
        self.observations = self.observations.astype(self.obs_type)
        self.next_observations = self.next_observations.astype(self.obs_type)
        self.dones = self.dones.astype(self.done_type)
        self.rewards = self.rewards.astype(self.rew_type)

        assert self.actions.dtype == np.float32 or self.actions.dtype == np.int64
        assert self.logits.dtype == np.float32
        assert self.values.dtype == np.float32

        # compute advantages
        self.advantages, self.returns = compute_gae_advantage(
            self.rewards,
            self.values.reshape((-1,)),
            self.dones,
            self.gamma,
            self.gae_gamma,
            recomputed_returns=True
        )
        assert self.advantages.dtype == np.float32
        assert self.returns.dtype == np.float32

        # return rollout steps
        ep_info = {}
        if len(self.episode_info['returns']) > 0:
            ep_info['rollout/returns'] = np.mean(self.episode_info['returns'])
            ep_info['rollout/lens'] = np.mean(self.episode_info['lens'])

        return len(self.observations), ep_info
    
    def sample_batch(self, batch_size, epoch):
        for _ in range(epoch):
            # generate shuffled indices
            n = len(self.observations)
            indices = np.random.permutation(n)

            # handle each batch
            for i in range(0, n, batch_size):
                if i + batch_size <= n:
                    batch_idx = indices[i : i + batch_size]

                else:
                    num_tail = n - i
                    num_head = batch_size - num_tail
                    batch_idx = np.concat([indices[i : n], indices[0: num_head]], axis=0)

                assert len(batch_idx) == batch_size
                batch = {
                    "observations": self.observations[batch_idx],
                    "actions": self.actions[batch_idx],
                    "rewards": self.rewards[batch_idx],
                    "returns": self.returns[batch_idx],
                    "dones": self.dones[batch_idx],
                    "logits_old": self.logits[batch_idx],
                    "advantages": self.advantages[batch_idx],
                    "values_old": self.values[batch_idx],
                }

                yield batch
