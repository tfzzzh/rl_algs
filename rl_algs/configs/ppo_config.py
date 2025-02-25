from typing import Tuple, Optional

import gymnasium as gym

import numpy as np
import torch
import torch.nn as nn

from rl_algs.networks.mlp_policy import MLPPolicy
from rl_algs.networks.value_critic import ValueCritic

from gymnasium.wrappers import RescaleAction
from gymnasium.wrappers import ClipAction
from gymnasium.wrappers import RecordEpisodeStatistics


def ppo_config(
    env_name: str,
    exp_name: Optional[str] = None,
    hidden_size: int = 128,
    num_layers: int = 3,
    is_discrete: bool = False,
    actor_learning_rate: float = 3e-4,
    critic_learning_rate: float = 3e-4,
    total_steps: int = 300000,
    batch_size: int = 128,
    ep_len: Optional[int] = None,
    discount: float = 0.99,
    reference_update_period: int = 1,
    # Actor-critic configuration
    normalize_advantage: bool = False,
    clip_eps=0.2,
    clip_eps_vf=None,
    eps=1e-8,
    actor_fixed_std: Optional[float] = None,
    use_tanh: bool = True,
    train_epoach: int = 5,
    train_batch_size: int = 128
):
    def make_critic(observation_shape: Tuple[int, ...]) -> nn.Module:
        return ValueCritic(
            ob_dim=np.prod(observation_shape),
            n_layers=num_layers,
            layer_size=hidden_size,
        )

    def make_actor(observation_shape: Tuple[int, ...], action_dim: int) -> nn.Module:
        assert len(observation_shape) == 1
        if actor_fixed_std is not None:
            return MLPPolicy(
                ac_dim=action_dim,
                ob_dim=np.prod(observation_shape),
                discrete=is_discrete,
                n_layers=num_layers,
                layer_size=hidden_size,
                use_tanh=use_tanh,
                state_dependent_std=False,
                fixed_std=actor_fixed_std,
            )
        else:
            return MLPPolicy(
                ac_dim=action_dim,
                ob_dim=np.prod(observation_shape),
                discrete=is_discrete,
                n_layers=num_layers,
                layer_size=hidden_size,
                use_tanh=use_tanh,
                state_dependent_std=True,
            )

    def make_actor_optimizer(params: torch.nn.ParameterList) -> torch.optim.Optimizer:
        return torch.optim.Adam(params, lr=actor_learning_rate)

    def make_critic_optimizer(params: torch.nn.ParameterList) -> torch.optim.Optimizer:
        return torch.optim.Adam(params, lr=critic_learning_rate)

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

    log_string = "{}_{}_s{}_l{}_alr{}_clr{}_b{}_d{}".format(
        exp_name or "offpolicy_ac",
        env_name,
        hidden_size,
        num_layers,
        actor_learning_rate,
        critic_learning_rate,
        batch_size,
        discount,
    )

    config = {
        "agent_kwargs": {
            "make_critic": make_critic,
            "make_critic_optimizer": make_critic_optimizer,
            "make_critic_schedule": make_lr_schedule,
            "make_actor": make_actor,
            "make_actor_optimizer": make_actor_optimizer,
            "make_actor_schedule": make_lr_schedule,
            "discount": discount,
            "reference_update_period": reference_update_period,
            "normalize_advantage": normalize_advantage,
            "clip_eps": clip_eps,
            "clip_eps_vf": clip_eps_vf,
            "eps": eps,
            "train_epoach": train_epoach,
            "train_batch_size": train_batch_size
        },
        "log_name": log_string,
        "total_steps": total_steps,
        "ep_len": ep_len,
        "batch_size": batch_size,
        "make_env": make_env,
        "is_discrete": is_discrete
    }

    print(config)

    return config