from typing import Tuple, Optional

import gymnasium as gym

import numpy as np
import torch
import torch.nn as nn

from rl_algs.utility.atari_wrappers import make_atari

def ppo_atari_config(
    env_name: str,
    exp_name: Optional[str] = None,
    share_encoder: bool = True,
    learning_rate: float = 3e-4,
    total_steps: int = 300000,
    ep_len: Optional[int] = None,
    discount: float = 0.99,
    # Actor-critic configuration
    normalize_advantage: bool = False,
    clip_eps=0.2,
    clip_eps_vf=None,
    eps=1e-8,
    train_epoach: int = 5,
    train_batch_size: int = 128,
    gae_lambda: float = 1.0,
    clip_grad_norm: float = 10.0,
    critic_loss_weight: float = 1.0,
    entropy_weight: float = 0.01,
    min_batch_per_step: int = 2048,
    num_train_env: int = 8,
    pretrained_model_path: Optional[str] = None,
):
    def make_optimizer(params: torch.nn.ParameterList) -> torch.optim.Optimizer:
        return torch.optim.Adam(params, lr=learning_rate)

    def make_lr_schedule(
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler._LRScheduler:
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    def make_env(render: bool = False) -> gym.Env:
        env = make_atari(env_name, render_mode="human" if render else None)
        return env


    log_string = f"{env_name}_{exp_name}_share:{share_encoder}_lr{learning_rate}_gamma{discount}" + \
        f"gae{gae_lambda}_gn{clip_grad_norm}_wcri{critic_loss_weight}_went{critic_loss_weight}"

    config = {
        "agent_kwargs": {
            "make_optimizer": make_optimizer,
            "make_lr_schedule": make_lr_schedule,
            "discount": discount,
            "share_encoder": share_encoder,
            "normalize_advantage": normalize_advantage,
            "clip_eps": clip_eps,
            "clip_eps_vf": clip_eps_vf,
            "eps": eps,
            "train_epoach": train_epoach,
            "train_batch_size": train_batch_size,
            "gae_lambda": gae_lambda,
            "clip_grad_norm": clip_grad_norm,
            "critic_loss_weight": critic_loss_weight,
            "entropy_weight": entropy_weight
        },
        "log_name": log_string,
        "total_steps": total_steps,
        "ep_len": ep_len,
        "min_batch_per_step": min_batch_per_step,
        "make_env": make_env,
        "num_train_env": num_train_env,
        "pretrained_model_path": pretrained_model_path
    }

    print(config)

    return config