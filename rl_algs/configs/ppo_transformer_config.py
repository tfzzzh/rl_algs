from typing import Tuple, Optional

import gymnasium as gym

import numpy as np
import torch
import torch.nn as nn

from rl_algs.networks.transformer_actor_critic import TransformerActorCritic
from rl_algs.agents.ppo_transformer import RolloutBuffer
import rl_algs.utility.pytorch_util as ptu

from gymnasium.wrappers import RescaleAction
from gymnasium.wrappers import ClipAction
from gymnasium.wrappers import RecordEpisodeStatistics


def ppo_transformer_config(
    env_name: str,
    exp_name: Optional[str] = None,
    learning_rate: float = 1e-4,
    weight_decay:float =1e-4,
    discount: float = 0.99,
    normalize_advantage: bool = True,
    clip_eps: float = 0.2,
    clip_eps_vf: float = 5.0,
    eps: float = 1e-8,
    train_epoach: int = 1,
    train_batch_size: int = 128,
    gae_lambda: float = 1.0,
    max_context_len: Optional[int] = None,
    critic_loss_coef: float = 0.5,
    max_grad_norm=10.0,
    # gpt_config
    gpt_n_embd: int = 128,
    gpt_n_layer: int = 1,
    gpt_n_head: int = 3,
    dropout: float = 0.0,
    # actor_head_config
    actor_n_layers: int = 2,
    actor_size: int = 15,
    actor_activation: str = "tanh",
    actor_output_activation: str = "identity",
    log_std_init_value: float = 1.0,
    # critic_head_config
    critic_n_layers: int = 2,
    critic_size: int = 120,  # layer size
    critic_activation: str = "tanh",
    critic_output_activation: str = "identity",
    # script config
    total_steps: int = 300000,
    ep_len: int = 1000,
    num_eval_episode: int = 2,
):
    def make_actor_critic(
        observation_shape: Tuple[int, ...],
        action_dim: int,
        max_ep_len: int,
        action_space: gym.spaces.Space,
    ) -> TransformerActorCritic:
        encoder_config = {
            "n_embd": gpt_n_embd,
            "n_layer": gpt_n_layer,
            "n_head": gpt_n_head,
            "dropout": dropout,
        }
        actor_head_config = {
            # parameters for mlp head
            # "input_size": (equal to encoder's output dim) do not set it
            # "output_size": (equal to act_dim) do not set it
            "n_layers": actor_n_layers,
            "size": actor_size,  # layer size
            "activation": actor_activation,
            "output_activation": actor_output_activation,
            # parameter for distribution head
            "action_space": action_space,
            "log_std_init_value": log_std_init_value,  # (ignored in discrete space)
        }

        critic_head_config = {
            "n_layers": critic_n_layers,
            "size": critic_size,  # layer size
            "activation": critic_activation,
            "output_activation": critic_output_activation,
        }

        actor_critic = TransformerActorCritic(
            np.prod(observation_shape),
            action_dim,
            max_ep_len,
            encoder_config,
            actor_head_config,
            critic_head_config,
        )

        return actor_critic

    def make_optimizer(params: torch.nn.ParameterList) -> torch.optim.Optimizer:
        return torch.optim.Adam(params, lr=learning_rate)
        # return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)

    def make_lr_schedule(
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler._LRScheduler:
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    def make_env(render: bool = False):
        return RecordEpisodeStatistics(
            ClipAction(
                RescaleAction(
                    gym.make(env_name, render_mode="rgb_array" if render else None),
                    -1,
                    1,
                )
            )
        )

    log_string = f"{exp_name}_{env_name}_ed{gpt_n_embd}_an{actor_n_layers}_cn{critic_n_layers}_lr{learning_rate}_gae{gae_lambda}_d{discount}"

    config = {
        "agent_kwargs": {
            "make_actor_critic": make_actor_critic,
            "make_optimizer": make_optimizer,
            "make_lr_schedule": make_lr_schedule,
            "discount": discount,
            "normalize_advantage": normalize_advantage,
            "clip_eps": clip_eps,
            "clip_eps_vf": clip_eps_vf,
            "eps": eps,
            "train_epoach": train_epoach,
            "train_batch_size": train_batch_size,
            "gae_lambda": gae_lambda,
            "max_context_len": max_context_len,
            "critic_loss_coef": critic_loss_coef,
            "max_grad_norm": max_grad_norm,
        },
        "log_name": log_string,
        "total_steps": total_steps,
        "ep_len": ep_len,
        "num_eval_episode": num_eval_episode,
        "make_env": make_env,
    }

    print(config)

    return config
