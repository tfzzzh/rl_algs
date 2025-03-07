import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from torch import nn
from typing import Optional, Dict, Any

from .gpt import GPT, GPTConfig
import rl_algs.utility.pytorch_util as ptu
from rl_algs.agents.common import compute_gae_advantage


class TransformerActorCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_ep_len: int,
        encoder_config: Dict[str, Any],
        actor_head_config: Dict[str, Any],
        critic_head_config: Dict[str, Any],
    ):
        """Construct An Actor, Critic pair which shares an transformer encoder

        Args:
            encoder_config (Dict[str, Any]): example:
            {
                "n_embd": 768,
                "n_layer": 12,
                "n_head": 12,
                "dropout": 0.0
            }

            actor_head_config (Dict[str, Any]): example:
            {
                # parameters for mlp head
                # "input_size": (equal to encoder's output dim) do not set it
                # "output_size": (equal to act_dim) do not set it
                "n_layers": 3,
                "size": 120, # layer size
                "activation": "tanh",
                "output_activation": "identity",
                # parameter for distribution head
                "action_space": env.action_space,
                "log_std_init_value": 1.0 # (ignored in discrete space)
            }

            critic_head_config (Dict[str, Any]): example:
            {
                # parameters for mlp
                # "input_size": (equal to encoder's output dim) do not set it
                # "output_size": (equal to act_dim) do not set it
                "n_layers": 3,
                "size": 120, # layer size
                "activation": "tanh",
                "output_activation": "identity"
            }

        TODO: now only continuous action is supported, to support descrete action
        """
        super().__init__()

        # init encoder
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.encoder = TransformerEncoder(
            state_dim=state_dim,
            act_dim=act_dim,
            max_ep_len=max_ep_len,
            **encoder_config,
        )

        # init actor_head by mlp
        self.encoder_out_dim = encoder_config["n_embd"]
        self.actor_head = self._build_actor_head(actor_head_config)

        # init critic
        self.critic_head = self._build_critic_head(critic_head_config)
        self.to(ptu.device)

    def _build_actor_head(self, actor_head_config):
        net = ptu.build_mlp(
            input_size=self.encoder_out_dim,
            output_size=self.act_dim,
            n_layers=actor_head_config["n_layers"],
            size=actor_head_config["size"],
            activation=actor_head_config["activation"],
            output_activation=actor_head_config["output_activation"],
        )

        net = DistributionHead(
            action_space=actor_head_config["action_space"],
            network=net,
            log_std_init_value=actor_head_config["log_std_init_value"],
        )

        return net

    def _build_critic_head(self, critic_head_config):
        net = ptu.build_mlp(
            input_size=self.encoder_out_dim,
            output_size=1,
            n_layers=critic_head_config["n_layers"],
            size=critic_head_config["size"],
            activation=critic_head_config["activation"],
            output_activation=critic_head_config["output_activation"],
        )

        return net

    def encode(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """given historical trace of (s0, a0), (s1, a1) ... (sn, an) return encode features

        Args:
            observations (torch.Tensor): shape [bsize, seqlen, dim_obs]
            actions (torch.Tensor): shape [bsize, seqlen, dim_act]
            timesteps (torch.Tensor): shape[bsize, seqlen] with type torch.long
            attention_mask (Optional[torch.Tensor], optional): shape[bsize, seqlen] type torch.bool if not none. Defaults to None.

        Returns:
            features: torch.Tensor: features of size  [bsize, seqlen, n_embed], use the feature with attention_mask
        """
        # features of shape [bsize, nseq, n_embed]
        # batch_size, seq_length = observations.shape[0], observations.shape[1]
        # if attention_mask is None:
        #     # attention mask for GPT: 1 if can be attended to, 0 if not
        #     attention_mask = torch.ones((batch_size, seq_length), dtype=torch.bool)

        features = self.encoder(observations, actions, timesteps, attention_mask)

        # return features, attention_mask
        return features

    def actor_forward(
        self, features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.distributions.Distribution:
        """given extracted features output action distribution

        Args:
            features (torch.Tensor): features of size  [bsize, seqlen, n_embed]
            attention_mask (Optional[torch.Tensor], optional): _description_. Defaults to None.

        Returns:
            torch.distributions.Distribution: _description_
        """
        batch_size, seq_length, dim = features.shape
        assert seq_length > 0 and dim == self.encoder_out_dim

        # use latest feature
        feature = features[:, -1, :]
        action_dist = self.actor_head.forward(feature)

        # action_dist is only valid under attention_mask[-1]
        return action_dist

    def critic_forward(
        self, features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """given extracted features compute critic

        Args:
            features (torch.Tensor): features of size  [bsize, seqlen, n_embed]
            attention_mask (Optional[torch.Tensor], optional): _description_. Defaults to None.

        Returns:
            torch.Tensor: critic value of shape [bsize, ]
        """

        batch_size, seq_length, dim = features.shape
        assert seq_length > 0 and dim == self.encoder_out_dim

        # use latest feature
        feature = features[:, -1, :]
        value = self.critic_head.forward(feature)

        assert value.shape == (batch_size, 1)
        value = value.squeeze(dim=-1)

        # action_dist is only valid under attention_mask[-1]
        return value


class DistributionHead(nn.Module):
    def __init__(
        self,
        action_space: spaces.Space,
        network: nn.Module,
        log_std_init_value: float = 1.0,
    ):
        super().__init__()

        self.action_space = action_space
        self.net = network

        if isinstance(action_space, spaces.Box):
            self.log_std = nn.Parameter(
                torch.tensor(log_std_init_value, dtype=torch.float32),
                requires_grad=True,
            )

        self.to(ptu.device)

    def forward(self, x: torch.Tensor) -> torch.distributions.Distribution:
        """_summary_

        Args:
            x (torch.Tensor): generally, x shall have shape [bsize, hidden_size]

        Returns:
            torch.distributions.Distribution: when sampled give a action of size [bsize,]
        """
        logits = self.net(x)

        # genenrate output according to type of the space
        if isinstance(self.action_space, spaces.Box):
            return torch.distributions.Normal(loc=logits, scale=torch.exp(self.log_std))

        elif isinstance(self.action_space, spaces.Discrete):
            prob = torch.softmax(logits, dim=-1)
            return torch.distributions.Categorical(prob)

        else:
            raise NotImplementedError(
                f"Action space {self.action_space} not supported."
            )


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        state_dim,
        act_dim,
        max_ep_len,
        # config for gpt
        n_embd,
        n_layer,
        n_head,
        dropout,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.n_embd = n_embd

        self.gpt_config = GPTConfig(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=n_embd,
            block_size=max_ep_len,
            n_layer=n_layer,
            n_head=n_head,
            dropout=dropout,
        )

        self.transformer = GPT(self.gpt_config)

        # construct embedding matrix for time, state and action
        self.embed_timestep = nn.Embedding(max_ep_len, n_embd)
        self.embed_state = torch.nn.Linear(self.state_dim, n_embd)
        self.embed_action = torch.nn.Linear(self.act_dim, n_embd)

        # use layer norm
        # TODO: implement batchnorm in replaybuffer or explore buffer
        self.embed_ln = nn.LayerNorm(self.n_embd)

        # push model to device
        self.to(ptu.device)

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            observations (torch.Tensor): shape [bsize, seqlen, dim_obs]
            actions (torch.Tensor): shape [bsize, seqlen, dim_act]
            timesteps (torch.Tensor): shape[bsize, seqlen] with type torch.long
            attention_mask (Optional[torch.Tensor], optional): shape[bsize, seqlen] type torch.bool if not none. Defaults to None.
        """
        batch_size, seq_length = observations.shape[0], observations.shape[1]
        assert observations.shape == (batch_size, seq_length, self.state_dim)
        assert actions.shape == (batch_size, seq_length, self.act_dim)
        assert timesteps.shape == (batch_size, seq_length)
        assert timesteps.dtype == torch.long

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.bool)
        else:
            assert (
                attention_mask.shape == (batch_size, seq_length)
                and attention_mask.dtype == torch.bool
            )

        # embed observations, actions, time
        obs_embed = self.embed_state(observations)  # -> [bsize, seqlen, n_embd]
        act_embed = self.embed_action(actions)
        pos_embed = self.embed_timestep(timesteps)

        obs_embed = obs_embed + pos_embed
        act_embed = act_embed + pos_embed

        # make sequence (s1, a1, s2, a2, ...)
        # [bsize, 2, seqlen, dim]
        stacked_inputs = torch.stack([obs_embed, act_embed], dim=1)
        stacked_inputs = stacked_inputs.permute(0, 2, 1, 3)
        assert stacked_inputs.shape == (batch_size, seq_length, 2, self.n_embd)
        stacked_inputs = stacked_inputs.reshape(batch_size, 2 * seq_length, self.n_embd)
        stacked_inputs = self.embed_ln(stacked_inputs)

        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 2 * seq_length)
        )

        # TODO: use mask to reduce computational cost
        transformer_outputs = self.transformer(stacked_inputs)
        assert transformer_outputs.shape == (batch_size, 2 * seq_length, self.n_embd)

        # extract encoding at observation
        # transformer_outputs arranged in (s1, a1, s2, a2) format
        # we only need feature at (s1, s2, ...)
        transformer_outputs = transformer_outputs.reshape(
            batch_size, seq_length, 2, self.n_embd
        )
        features = transformer_outputs[:, :, 0, :]

        return features


# context for the encoder when used for inference
# TODO: test it
class TransformerEncoderContext:
    def __init__(self, observation_shape, action_dim):
        self.observation_shape = observation_shape
        self.action_dim = action_dim

        self.observations = []
        self.actions = []
        self.timesteps = []

    def reset(self):
        self.observations = []
        self.actions = []
        self.timesteps = []

    def update(self, new_state: np.ndarray, prev_action: Optional[np.ndarray] = None):
        """given the latest state and previous action update context
        usage case:
        1) when env reset
        state,_ = env.reset()
        tfcontext.reset()
        tfcontext.update(state, None)

        2) in explore step
        action = agent.get_action(context)
        next_ob, rew, terminated, truncated, info = env.step(action)
        tfcontext(next_ob, action)

        Args:
            new_state (np.ndarray): _description_
            prev_action (Optional[np.ndarray], optional): _description_. Defaults to None.
        """
        # case: insert first state
        if prev_action is None:
            assert (
                len(self.observations) == 0
                and len(self.actions) == 0
                and len(self.timesteps) == 0
            )
            # do noting

        else:
            # make action's type would not change
            self.actions[-1][:] = prev_action

        self.actions.append(np.zeros(self.action_dim, dtype=np.float32))
        sz = len(self.timesteps)
        self.timesteps.append(np.array(sz, dtype=np.int64))
        self._append_observation(new_state)

    def _append_observation(self, new_observation: np.ndarray):
        """
        new_observation of shape (x, y) is normalized then reshaped
            into (x*y,) before append to observation list
        """
        assert new_observation.shape == self.observation_shape
        new_observation = new_observation.astype(np.float32).reshape((-1,))
        self.observations.append(new_observation)

    def get_context(self, max_len: Optional[int] = None) -> Dict[str, np.ndarray]:
        assert len(self.observations) != 0

        observations = self.observations
        actions = self.actions
        timesteps = self.timesteps

        if max_len is not None:
            assert max_len > 0
            observations = observations[-max_len:]
            actions = actions[-max_len:]
            timesteps = timesteps[-max_len:]

        context = {
            "observations": np.stack(observations, axis=0)[None],
            "actions": np.stack(actions, axis=0)[None],
            "timesteps": np.stack(timesteps, axis=0)[None],
        }

        # check shape
        self._check_context_shape(context, max_len)

        context = ptu.from_numpy(context)
        context["attention_mask"] = None

        return context

    def _check_context_shape(self, context, max_len):
        context_length = (
            len(self.timesteps)
            if max_len is None
            else min(len(self.timesteps), max_len)
        )
        assert context["observations"].shape == (
            1,
            context_length,
            np.prod(self.observation_shape),
        )
        assert context["actions"].shape == (1, context_length, self.action_dim)
        assert context["timesteps"].shape == (1, context_length)


# Currently it only support for rolling one env
class RolloutBuffer:
    def __init__(
        self,
        obs_shape,
        max_ep_len,
        action_dim,
        env: gym.Env,
        actor_critic: TransformerActorCritic,
        gamma: float,
        gae_lambda: float,
        max_context_len=None,
    ):
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

    def expand_episode(self):
        env = self.env
        actor_critic = self.actor_critic

        ob, init_info = env.reset()
        self.reset()
        steps = 0
        self.observations[0][:] = ob

        while True:
            # get action from actor
            # construct context for actor
            context = self.get_context()
            with torch.no_grad():
                features = actor_critic.encode(**context)
                action_dist = actor_critic.actor_forward(features)

                action = action_dist.sample()
                log_prob = action_dist.log_prob(action).squeeze(0).numpy()
                action = action.squeeze(0).numpy()
                value = actor_critic.critic_forward(features).squeeze(0).numpy()

                assert action.shape == (self.action_dim,)
                assert len(value.shape) == 0
                assert len(log_prob.shape) == 0

            # take that action and get reward and next ob
            next_ob, reward, terminated, truncated, info = env.step(action)

            # rollout can end due to done, or due to max_length
            rollout_done = terminated or truncated or (steps+1) >= self.max_ep_len

            # record result of taking that action
            self.timesteps[steps] = steps
            self.actions[steps][:] = action
            self.rewards[steps] = reward
            self.dones[steps] = terminated
            self.log_probs[steps] = log_prob
            self.length += 1

            steps += 1
            # end the rollout if the rollout ended
            if rollout_done:
                break

            ob = next_ob  # jump to next timestep
            # next state when rollout_done is not recorded
            self.observations[steps][:] = ob

        env.close()
        assert (self.length == steps )

        # episode ready compute returns and advantages
        self.compute_returns_and_advantages()

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

        if max_len is not None:
            assert max_len > 0
            context_length = min(context_length, max_len)

        # each of shape [T,?]
        start_idx = self.length - context_length
        observations = observations[start_idx : self.length]
        observations = observations.reshape((context_length, -1))
        actions = actions[start_idx : self.length]
        timesteps = timesteps[start_idx : self.length]

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
        context_length = (
            len(self.timesteps)
            if max_len is None
            else min(len(self.timesteps), max_len)
        )
        assert context["observations"].shape == (
            1,
            context_length,
            np.prod(self.observation_shape),
        )
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


# regression test
# if __name__ == '__main__':
#     state_dim = 10
#     act_dim = 5
#     max_ep_len = 17
#     # config for gpt
#     n_embd = 8
#     n_layer = 1
#     n_head = 1
#     dropout = 0.1
#     encoder = TransformerEncoder(state_dim, act_dim, max_ep_len, n_embd, n_layer, n_head, dropout)
