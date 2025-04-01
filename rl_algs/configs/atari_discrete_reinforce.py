from typing import Tuple, Optional

import gymnasium as gym

import numpy as np
import torch
import torch.nn as nn

from rl_algs.networks.atari import DiscreteActor, DiscreteCritic, Temperature
from rl_algs.agents.reinforce import ActionGetter
from rl_algs.utility.env_atari import make_atari


def atari_discrete_reinforce_config(
    env_name: str,
    exp_name: Optional[str] = None,
    hidden_size: int = 128,
    num_layers: int = 3,
    # action_embed_dim: int = 512,
    actor_learning_rate: float = 0.5e-4,
    critic_learning_rate: float = 1.5e-4,
    temp_learming_rate: float = 0.5e-4,
    total_steps: int = 300000,
    random_steps: int = 10000,
    # action getter
    explore_eps_annealing_frames: int = 10000,
    explore_eps_initial: float = 1.0,
    explore_eps_final: float=0.01,
    explore_eps_final_frame: float=0.01,
    batch_size: int = 128,
    replay_buffer_capacity: int = 1000000,
    ep_len: Optional[int] = None,
    discount: float = 0.99,
    use_soft_target_update: bool = False,
    target_update_period: Optional[int] = None,
    soft_target_update_rate: Optional[float] = None,
    # Actor-critic configuration
    num_actor_samples: int = 1,
    num_critic_updates: int = 1,
    # Settings for multiple critics
    num_critic_networks: int = 1,
    target_critic_backup_type: str = "min_target",  # One of "doubleq", "min", or "mean"
    # Soft actor-critic
    backup_entropy: bool = True,
    use_entropy_bonus: bool = True,
    temperature: float = 0.1,
    log_temperature:float = 0.0,
    pretrained_model_path: Optional[str] = None,
):
    def make_actor(num_action: int) -> nn.Module:
        return DiscreteActor(num_action)

    def make_actor_optimizer(params: torch.nn.ParameterList) -> torch.optim.Optimizer:
        return torch.optim.Adam(params, lr=actor_learning_rate, eps=1e-4)

    def make_actor_schedule(
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler._LRScheduler:
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    def make_critic(num_action: int) -> nn.Module:
        # return DiscreteCritic(
        #     num_action, num_layers, hidden_size
        # )
        return DiscreteCritic(num_action, num_layers, hidden_size)

    def make_critic_optimizer(params: torch.nn.ParameterList) -> torch.optim.Optimizer:
        return torch.optim.Adam(params, lr=temp_learming_rate, eps=1e-4)

    def make_critic_schedule(
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler._LRScheduler:
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    
    def make_temperature_net() -> Temperature:
        return Temperature(log_temperature)
    
    def make_temperature_optimizer(params: torch.nn.ParameterList) -> torch.optim.Optimizer:
        return torch.optim.Adam(params, lr=critic_learning_rate, eps=1e-4)

    def make_env(render: bool = False) -> gym.Env:
        env = make_atari(env_name, render_mode="human" if render else None)
        return env
    
    def make_action_getter(num_action: int) -> ActionGetter:
        return ActionGetter(
            n_actions=num_action,
            eps_initial=explore_eps_initial,
            eps_final=explore_eps_final,
            eps_final_frame=explore_eps_final_frame,
            eps_evaluation=0.0,
            eps_annealing_frames=explore_eps_annealing_frames,
            replay_memory_start_size=random_steps,
            max_steps=total_steps
        )

    log_string = (
        f"{exp_name}_{env_name}_ch{hidden_size}_cl{num_layers}_alr{actor_learning_rate}_clr{critic_learning_rate}"
        + f"_b{batch_size}_d{discount}_bt{target_critic_backup_type}"
    )

    if use_entropy_bonus:
        log_string += f"_t{temperature}"

    if use_soft_target_update:
        log_string += f"_stu{soft_target_update_rate}"
    else:
        log_string += f"_htu{target_update_period}"

    return {
        "agent_kwargs": {
            "make_critic": make_critic,
            "make_critic_optimizer": make_critic_optimizer,
            "make_critic_schedule": make_critic_schedule,
            "make_actor": make_actor,
            "make_actor_optimizer": make_actor_optimizer,
            "make_actor_schedule": make_actor_schedule,
            "num_critic_updates": num_critic_updates,
            "discount": discount,
            "num_actor_samples": num_actor_samples,
            "num_critic_updates": num_critic_updates,
            "num_critic_networks": num_critic_networks,
            "target_critic_backup_type": target_critic_backup_type,
            "use_entropy_bonus": use_entropy_bonus,
            "backup_entropy": backup_entropy,
            "temperature": temperature,
            "target_update_period": (
                target_update_period if not use_soft_target_update else None
            ),
            "soft_target_update_rate": (
                soft_target_update_rate if use_soft_target_update else None
            ),
            "make_temperature_net": make_temperature_net,
            "make_temperature_optimizer": make_temperature_optimizer,
            "make_temperature_lr_schedule": make_actor_schedule
        },
        "make_action_getter": make_action_getter,
        "replay_buffer_capacity": replay_buffer_capacity,
        "log_name": log_string,
        "total_steps": total_steps,
        "random_steps": random_steps,
        "ep_len": ep_len,
        "batch_size": batch_size,
        "make_env": make_env,
        "pretrained_model_path": pretrained_model_path
    }
