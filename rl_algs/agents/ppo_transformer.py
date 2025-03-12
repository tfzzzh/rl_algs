from typing import Callable, Sequence, Tuple, Union
import torch
import numpy as np
import gymnasium as gym
from torch import nn
from torch.functional import F
from typing import Optional, Dict, Any, Sequence

from .common import compute_gae_advantage

# from rl_algs.agents.ppo_transformer import RolloutBuffer
# from .ppo_transformer import RolloutBuffer
import rl_algs.utility.pytorch_util as ptu
from rl_algs.networks.transformer_actor_critic import TransformerActorCritic


class PPOTransformer:
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,
        max_ep_len: int,
        action_space: gym.spaces.Space,
        make_actor_critic: Callable[
            [Tuple[int, ...], int, int], TransformerActorCritic
        ],
        make_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_lr_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        disable_critic: bool = False,
        normalize_advantage: bool = False,
        clip_eps=0.2,
        clip_eps_vf: float = 5.0,
        eps: float = 1e-8,
        train_epoach: int = 1,
        train_batch_size: int = 128,
        gae_lambda: float = 1.0,
        max_context_len: Optional[int] = None,
        critic_loss_coef: float = 0.5,
        max_grad_norm=0.5,
    ):
        self.actor_critic: TransformerActorCritic = make_actor_critic(
            observation_shape, action_dim, max_ep_len, action_space
        )
        self.optimizer: torch.optim.Optimizer = make_optimizer(
            self.actor_critic.parameters()
        )
        self.lr_scheduler = make_lr_schedule(self.optimizer)

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
        self.max_ep_len = max_ep_len
        self.max_context_len = max_context_len
        self.critic_loss_coef = critic_loss_coef
        self.max_grad_norm = max_grad_norm
        self.disable_critic = disable_critic

        if self.disable_critic:
            self.critic_loss_coef = 0.0

        self.state_processor = StandardObservationScaler()

    def create_rollout_buffer(self, env, is_train_env) -> "RolloutBuffer":
        """
        Compute the action for a given observation.
        call it once for each env is enough
        """
        buffer = RolloutBuffer(
            obs_shape=self.observation_shape,
            action_dim=self.action_dim,
            max_ep_len=self.max_ep_len,
            env=env,
            actor_critic=self.actor_critic,
            gamma=self.discount,
            gae_lambda=self.gae_lambda,
            state_processor=self.state_processor,
            max_context_len=self.max_context_len,
            is_training=is_train_env
        )
        return buffer

    def update(self, rollout_buffer: "RolloutBuffer"):
        if not self.disable_critic:
            infos = {
                "loss": [],
                "actor_loss": [],
                "actor_clip_fraction": [],
                "grad_norm": [],
                "grad_norm_critic": [],
                "critic_loss": [],
                "critic_clip_fraction": [],
            }
        else:
            infos = {
                "actor_loss": [],
                "actor_clip_fraction": [],
                "grad_norm": []
            }

        #TODO remove it just fortest
        # rollout_buffer.returns[:len(rollout_buffer)] = self.normalize(
        #     rollout_buffer.returns[:len(rollout_buffer)],
        #     self.eps
        # )

        # foreach iterator in train epoch
        for itr in range(self.train_epoach):
            # sampling data from buffer
            # TODO: assumption: batch contains one episode, with full attention
            # mask handle the batch case in the future
            batch = rollout_buffer.get_sample(self.train_batch_size)

            # compute loss
            # call encoder get features for the batch
            features, bsize = self._extract_features_from_batch(batch)

            # compute prob ratio and actor loss
            ratio, actor_loss = self._compute_actor_loss(batch, features, bsize)

            # 4) compute critic value
            if not self.disable_critic:
                diff, critic_loss = self._compute_critic_loss(batch, features, bsize)

            # c) combine actor and critic loss
            if not self.disable_critic:
                loss = actor_loss + self.critic_loss_coef * critic_loss
            else:
                loss = actor_loss

            # call optimizer
            self.optimizer.zero_grad()
            loss.backward()

            # clip record gradnorm
            if not self.disable_critic:
                normed_parameters = list(self.actor_critic.encoder.parameters()) + \
                    list(self.actor_critic.actor_head.parameters())
                grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
                    normed_parameters, self.max_grad_norm
                )

                grad_norm_critic = torch.nn.utils.clip_grad.clip_grad_norm_(
                    self.actor_critic.critic_head.parameters(), 10.0
                )
            else:
                normed_parameters = list(self.actor_critic.encoder.parameters()) + \
                    list(self.actor_critic.actor_head.parameters())
                grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
                    normed_parameters, self.max_grad_norm
                )

            self.optimizer.step()
            self.lr_scheduler.step()

            # bookmark infos
            actor_clip_fraction = torch.mean(
                (torch.abs(ratio - 1.0) > self.clip_eps).float()
            ).item()
            infos["actor_loss"].append(actor_loss.item())
            infos["actor_clip_fraction"].append(actor_clip_fraction)
            infos["grad_norm"].append(grad_norm.item())

            if not self.disable_critic:
                infos["loss"].append(loss.item())
                infos["critic_loss"].append(critic_loss.item())
                critic_clip_fraction = torch.mean(
                    (torch.abs(diff) > self.clip_eps_vf).float()
                ).item()
                infos["critic_clip_fraction"].append(critic_clip_fraction)
                infos['grad_norm_critic'].append(grad_norm_critic.item())

        # when train finish, update preprocessor
        self.state_processor.fit(rollout_buffer.observations[:len(rollout_buffer)])
        # aggregate info and output
        info_agg = {key: np.mean(value) for key, value in infos.items()}

        return info_agg

    def _compute_critic_loss(self, batch, features, bsize):
        # critic loss: cliped loss: \|value_old + clip(delta) - return\|
        # detach encoder from critic's loss
        features = features.detach()

        # compute clipped value
        values = self.actor_critic.critic_head(features)
        values = values.reshape((bsize,))
        values_old = batch["values"].reshape((bsize,))
        diff = values - values_old
        values = values_old + torch.clip(diff, -self.clip_eps_vf, self.clip_eps_vf)
        assert values.shape == (bsize,)

        # compute loss
        returns = batch["returns"].reshape((bsize,))
        critic_loss = F.smooth_l1_loss(values, returns)
        return diff, critic_loss

    def _compute_actor_loss(self, batch, features, bsize):
        assert features.shape == (bsize, self.actor_critic.encoder_out_dim)

        # get action distribution from features
        actor_head = self.actor_critic.actor_head
        action_dist: torch.distributions.Distribution = actor_head(features)
        log_prob = action_dist.log_prob(batch["actions"].reshape(bsize, -1))
        assert log_prob.shape == (bsize, self.action_dim)
        log_prob = log_prob.sum(dim=1)

        # compute prob ratio
        log_prob_old = batch["log_probs"].reshape((bsize,))
        ratio = torch.exp(log_prob - log_prob_old)

        # normalize advantage
        if not self.disable_critic:
            advantage = batch["advantages"].reshape((bsize,))
        else:
            advantage = batch["returns"].reshape((bsize,))

        if self.normalize_advantage:
            advantage = self.normalize(advantage, eps=self.eps)

        # actor loss: -min(ratio * advantage, clip(ratio) * advantage)
        actor_gain = torch.min(
            ratio * advantage,
            torch.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantage,
        )
        actor_loss = -actor_gain.mean()
        return ratio, actor_loss

    def _extract_features_from_batch(self, batch):
        features = self.actor_critic.encode(
            observations=batch["observations"],
            actions=batch["actions"],
            timesteps=batch["timesteps"],
        )
        fdim = self.actor_critic.encoder_out_dim
        assert (features.shape[0], features.shape[2]) == (1, fdim)
        bsize = features.shape[0] * features.shape[1]
        assert bsize >= self.train_batch_size, "not enough states are sampled"

        features = features.reshape(bsize, -1)
        return features, bsize

    @staticmethod
    def normalize(adv: torch.Tensor, eps: float) -> torch.Tensor:
        """normalize advantage"""
        assert adv.ndim == 1
        return (adv - adv.mean()) / (adv.std() + eps)


# Currently it only support for rolling one env
class RolloutBuffer:
    def __init__(
        self,
        obs_shape: Sequence[int],
        action_dim: int,
        max_ep_len: int,
        env: gym.Env,
        actor_critic: TransformerActorCritic,
        gamma: float,
        gae_lambda: float,
        state_processor: 'StandardObservationScaler',
        max_context_len: Optional[int] = None,
        is_training = True,
    ):
        """The buffer stores one trajactory of a transformer actor.
        And provide (truncated) inference context for it

        Args:
            obs_shape (Sequence[int]): shape of observation
            action_dim (int): dimension of actions, here we assume the action is continuous
            max_ep_len (int): max episode length
            env (gym.Env):
            actor_critic (TransformerActorCritic):
            gamma (float): discount factor (in (0,1))
            gae_lambda (float): discount factor of GAE
            max_context_len (Optional[int], optional): inference window size. when set to None,
                all the trajactory will be used to predict next action
        """
        self.obs_shape = obs_shape
        self.max_ep_len = max_ep_len
        self.env = env
        self.actor_critic = actor_critic
        self.max_context_len = max_context_len
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.length = 0
        self.observations = np.zeros(
            (self.max_ep_len, *self.obs_shape), dtype=np.float32
        )
        self.actions = np.zeros((self.max_ep_len, self.action_dim), dtype=np.float32)
        self.timesteps = np.zeros(self.max_ep_len, dtype=np.long)
        self.rewards = np.zeros(self.max_ep_len, dtype=np.float32)
        self.returns = np.zeros(self.max_ep_len, dtype=np.float32)
        self.values = np.zeros(self.max_ep_len, dtype=np.float32)
        self.log_probs = np.zeros(self.max_ep_len, dtype=np.float32)
        self.advantages = np.zeros(self.max_ep_len, dtype=np.float32)
        self.dones = np.ones(self.max_ep_len, dtype=np.float32)

        # record running mean and std of obs
        # self.run_mean = np.zeros(self.obs_shape, dtype=np.float32)
        # self.run_std = np.ones(self.obs_shape, dtype=np.float32)
        # self.mom = mom
        self.state_processor = state_processor
        self.is_training = is_training

    def expand_episode(self):
        """fill the buffer with new episode
        TODO: change it to fill the buffer until #samples have been collected
        """
        env = self.env
        actor_critic = self.actor_critic

        # use eval mode
        if self.is_training:
            actor_critic.train()
        else:
            actor_critic.eval()

        # reset environment and buffer for a brandly new episode
        ob, init_info = env.reset()
        self.reset()
        steps = 0
        ob = self.state_processor.transform(ob)
        self.observations[0][:] = ob

        features_prev = None # Debug
        while True:
            # once at here, observation is known, but other component
            # like action is set to 0.0
            self.length += 1  # length set to 0 at reset
            self.timesteps[steps] = self.length - 1

            # update running mean and std
            # self.run_mean[:] = self.run_mean * (1.0 - self.mom)

            # get action from actor
            with torch.no_grad():
                # construct context for actor
                context = self.get_context()
                features = actor_critic.encode(**context)
                action_dist = actor_critic.actor_forward(features)
                value = actor_critic.critic_forward(features)

                action = action_dist.sample()
                assert action.shape == (1, self.action_dim)
                log_prob = action_dist.log_prob(action)
                assert log_prob.shape == (1, self.action_dim)
                log_prob = ptu.to_numpy(log_prob.sum(dim=-1).squeeze(0))
                action = ptu.to_numpy(action.squeeze(0))
                value = ptu.to_numpy(value.squeeze(0))

                assert action.shape == (self.action_dim,)
                assert len(value.shape) == 0
                assert len(log_prob.shape) == 0, f"log_prob.shape = {log_prob.shape}"

                features_prev = features

            # take that action and get reward and next ob
            next_ob, reward, terminated, truncated, info = env.step(action)

            # rollout can end due to done, or due to max_length
            rollout_done = terminated or truncated or (steps + 1) >= self.max_ep_len

            # record result of taking that action
            self.actions[steps][:] = action
            self.rewards[steps] = reward
            self.dones[steps] = terminated
            self.log_probs[steps] = log_prob
            self.values[steps] = value

            steps += 1
            # end the rollout if the rollout ended
            if rollout_done:
                break

            ob = next_ob  # jump to next timestep
            # next state when rollout_done is not recorded
            ob = self.state_processor.transform(ob)
            self.observations[steps][:] = ob

        env.close()
        assert self.length == steps

        # close eval mode (set to default)
        actor_critic.train()

        # episode ready compute returns and advantages
        self.compute_returns_and_advantages()

        # check if the dataset contains nan or inf
        assert np.all(~(np.isnan(self.observations) | np.isinf(self.observations)))
        assert np.all(~(np.isnan(self.actions) | np.isinf(self.actions)))
        assert np.all(~(np.isnan(self.rewards) | np.isinf(self.rewards)))
        assert np.all(~(np.isnan(self.returns) | np.isinf(self.returns)))
        assert np.all(~(np.isnan(self.values) | np.isinf(self.values)))
        assert np.all(~(np.isnan(self.log_probs) | np.isinf(self.log_probs)))
        assert np.all(~(np.isnan(self.advantages) | np.isinf(self.advantages)))
        assert np.all(~(np.isnan(self.dones) | np.isinf(self.dones)))

        # fit processor (when actor complete training)
        # self.state_processor.fit(self.observations[:self.length])

    def reset(self):
        self.length = 0
        self.observations[:] = 0.0
        self.actions[:] = 0.0
        self.timesteps[:] = 0
        self.rewards[:] = 0.0
        self.returns[:] = 0.0
        self.values[:] = 0.0
        self.log_probs[:] = 0.0
        self.advantages[:] = 0.0
        self.dones[:] = 1.0

    def get_context(self) -> Dict[str, torch.Tensor]:
        assert self.length != 0

        observations = self.observations
        actions = self.actions
        timesteps = self.timesteps
        max_len = self.max_context_len
        context_length = self.length

        # context_lengh <= max_context_len and <= length
        if max_len is not None:
            assert max_len > 0
            context_length = min(context_length, max_len)

        # latest context start is from [? to self.length)
        start_idx = self.length - context_length
        observations = observations[start_idx : self.length]

        # normalized observations (move out later)
        # observations = (observations - self.run_mean) / (self.run_std + 1e-8)

        observations = observations.reshape((context_length, -1))
        actions = actions[start_idx : self.length]
        timesteps = timesteps[start_idx : self.length]

        # fill first axis with batchsize = 1
        context = {
            "observations": observations[None],
            "actions": actions[None],
            "timesteps": timesteps[None],
        }

        # check shape
        self._check_context_shape(context, max_len)

        context = ptu.from_numpy(context)
        context["attention_mask"] = None

        return context

    def _check_context_shape(self, context, max_len):
        context_length = self.length if max_len is None else min(self.length, max_len)
        assert context["observations"].shape == (
            1,
            context_length,
            np.prod(self.obs_shape),
        ), f"observation context shape: {context["observations"].shape}"
        assert context["actions"].shape == (1, context_length, self.action_dim)
        assert context["timesteps"].shape == (1, context_length)

    def __len__(self):
        return self.length

    def get_sample(self, seqlen) -> Dict[str, torch.Tensor]:
        """sampling a subsequence with length seqlen from the buffer"""
        # make seqlen <= length
        seqlen = min(seqlen, self.length)
        assert seqlen > 0

        # sampling a valid start index
        start_idx = np.random.randint(0, self.length - seqlen + 1)

        # returns {obs, acs, values, log_probs, advantanges, returns}
        assert start_idx + seqlen <= self.length
        batch = {
            "observations": self.observations[start_idx : start_idx + seqlen][None],
            "actions": self.actions[start_idx : start_idx + seqlen][None],
            "timesteps": self.timesteps[start_idx : start_idx + seqlen][None],
            "rewards": self.rewards[start_idx : start_idx + seqlen][None],
            "returns": self.returns[start_idx : start_idx + seqlen][None],
            "values": self.values[start_idx : start_idx + seqlen][None],
            "log_probs": self.log_probs[start_idx : start_idx + seqlen][None],
            "advantages": self.advantages[start_idx : start_idx + seqlen][None],
            "dones": self.dones[start_idx : start_idx + seqlen][None],
        }

        batch = ptu.from_numpy(batch)

        return batch

    def compute_returns_and_advantages(self):
        adv, ret = compute_gae_advantage(
            self.rewards[: self.length],
            self.values[: self.length],
            self.dones[: self.length],
            self.gamma,
            self.gae_lambda,
        )

        self.advantages[: self.length] = adv
        self.returns[: self.length] = ret

    # def compute_metrics(self) -> Dict[str, float]:
    #     pass


# process input state
class StandardObservationScaler:
    r"""Standardization preprocessing.

    .. math::

        x' = (x - \mu) / \sigma


    Args:
        eps (float): Small constant value to avoid zero-division.
    """
    def __init__(self, eps=1e-3, mom=0.1):
        self.eps = eps
        self.mom = mom
        self.mean = None
        self.std = None


    # assumption: no mask in the observations
    def fit(self, observations):
        obs_shape = observations.shape[1:]
        observations = observations.reshape((-1, *obs_shape))

        mean_batch = observations.mean(axis=0)
        std_batch = observations.std(axis=0)

        if self.mean is None:
            self.mean = mean_batch
            self.std = std_batch
        
        else:
            self.mean[:] = (1.0 - self.mom) * self.mean + self.mom * mean_batch
            self.std[:] = (1.0 - self.mom) * self.std + self.mom * std_batch

    def transform(self, observations):
        if self.mean is None:
            return observations

        return (observations - self.mean) / (self.std + self.eps)