from typing import Tuple, Optional

import gymnasium as gym

import numpy as np
import torch
import torch.nn as nn

from rl_algs.networks.diffusion import NoiseMLP, Diffusion
from rl_algs.networks.state_action_value_critic import StateActionCritic
import rl_algs.utility.pytorch_util as ptu

from rl_algs.dataset import DataHandler


def ql_diffuse_config(
    dataset_name: str,
    exp_name: str,
    max_ep_len: int,
    batch_size: int,
    critic_nlayers: int,
    critic_hidden_size: int,
    n_timesteps: int,
    discount: float,
    rb_capacity: int = int(1e6) + 10,
    actor_target_update_rate: float = 0.005,
    critic_target_update_rate: float = 0.005,
    eta: float = 1.0,
    clip_grad_norm: float = 1.0,
    actor_learning_rate: float = 3e-4,
    critic_learning_rate: float = 3e-4,
    total_steps: int = 1000000
):
    def make_datahandler() -> DataHandler:
        '''
        use datahandler to recover training env
        '''
        return DataHandler(dataset_name)

    def make_noise_model(observation_shape: Tuple[int, ...], action_dim: int, t_dim: int) -> nn.Module:
        return NoiseMLP(np.prod(observation_shape), action_dim, t_dim)
    
    def make_actor(observation_shape: Tuple[int, ...], action_dim: int, noise_mode: nn.Module, n_timesteps: int) -> Diffusion:
        state_dim = np.prod(observation_shape)
        return Diffusion(state_dim, action_dim, noise_mode, n_timesteps=n_timesteps)
    
    def make_critic(observation_shape: Tuple[int, ...], action_dim: int) -> nn.Module:
        return StateActionCritic(
            ob_dim=np.prod(observation_shape),
            ac_dim=action_dim,
            n_layers=critic_nlayers,
            size=critic_hidden_size,
        )

    def make_actor_optimizer(params: torch.nn.ParameterList) -> torch.optim.Optimizer:
        return torch.optim.Adam(params, lr=actor_learning_rate)

    def make_critic_optimizer(params: torch.nn.ParameterList) -> torch.optim.Optimizer:
        return torch.optim.Adam(params, lr=critic_learning_rate)
    
    def make_lr_schedule(
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler._LRScheduler:
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    
    log_string = f"{exp_name}_cl{critic_nlayers}_ch{critic_hidden_size}_nt{n_timesteps}_alr{actor_learning_rate}_clr{critic_learning_rate}"

    config_dict = {
        "agent_kwargs": {
            "n_timesteps": n_timesteps,
            "make_noise_model": make_noise_model,
            "make_actor": make_actor,
            "make_actor_optimizer": make_actor_optimizer,
            "make_actor_schedule": make_lr_schedule,
            "make_critic": make_critic,
            "make_critic_optimizer": make_critic_optimizer,
            "make_critic_schedule": make_lr_schedule,
            "discount": discount,
            "actor_target_update_rate": actor_target_update_rate,
            "critic_target_update_rate": critic_target_update_rate,
            "eta": eta,
            "max_grad_norm": clip_grad_norm
        },
        "log_name": log_string,
        "make_datahandler": make_datahandler,
        "ep_len": max_ep_len,
        "batch_size": batch_size,
        "rb_capacity": rb_capacity,
        "total_steps": total_steps
    }

    return config_dict