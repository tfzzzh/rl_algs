from typing import Tuple, Optional

import gymnasium as gym

import numpy as np
import torch
import torch.nn as nn

from rl_algs.dataset import DataHandler
from rl_algs.networks.decision_transformer import DecisionTransformer
import rl_algs.utility.pytorch_util as ptu


def decision_transformer_config(
    # env and dataset
    dataset_name,
    exp_name,
    ep_len,
    eval_target_return: float,
    # dt config
    dt_max_infer_length: Optional[int]=None, # max inference length of dt
    action_tanh: bool =True,
    # gpt config
    n_layer: int = 3,
    n_head: int = 1,
    hidden_size: int = 128,
    dropout: float = 0.1,
    gpt_bias: bool = True,
    # trainner config
    actor_learning_rate: float = 1e-4,
    actor_weight_decay: float = 1e-4,
    warmup_steps: int = 10000,
    total_steps: int = 100000, # each 
    clip_grad_norm: float = 0.25,
    batch_size: int = 64,
    shuffle: bool = True,
    batch_max_length:int = 20,
    # evaluate config
    discount: float = 0.99,
    reward_scale:float = 1000.0,
):
    def make_datahandler() -> DataHandler:
        '''
        use datahandler to recover training env
        '''
        return DataHandler(dataset_name)

    def make_actor(observation_shape, action_dim) -> nn.Module:
        model = DecisionTransformer(
            state_dim=np.prod(observation_shape),
            act_dim=action_dim,
            hidden_size=hidden_size,
            max_ep_len=ep_len,
            action_tanh=action_tanh,
            n_layer=n_layer,
            n_head=n_head,
            dropout=dropout,
            bias=gpt_bias
        )
        return model

    def make_actor_optimizer(params: torch.nn.ParameterList) -> torch.optim.Optimizer:
        return torch.optim.AdamW(params, lr=actor_learning_rate, weight_decay=actor_weight_decay)
    
    def make_lr_schedule(
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler._LRScheduler:
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda steps: min((steps+1)/warmup_steps, 1)
        )
    
    def make_actor_loss():
        loss_fn = torch.nn.MSELoss()
        return loss_fn
    
    log_string = "{}_{}_layer{}_head{}_n_emb{}_alr{}_awd{}_bsize{}_d{}_K{}".format(
        "dt",
        exp_name,
        n_layer,
        n_head,
        hidden_size,
        actor_learning_rate,
        actor_weight_decay,
        batch_size,
        discount,
        batch_max_length
    )

    config_dict = {
        "agent_kwargs": {
            "make_actor": make_actor,
            "make_actor_optimizer": make_actor_optimizer,
            "make_actor_schedule": make_lr_schedule,
            "make_actor_loss": make_actor_loss,
            "clip_grad_norm": clip_grad_norm
        },
        "log_name": log_string,
        "total_steps": total_steps, # in dt, one batch is a step
        "ep_len": ep_len,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "batch_max_length": batch_max_length,
        "discount": discount,
        "make_datahandler": make_datahandler,
        "dt_max_length": dt_max_infer_length,
        "dataset_name": dataset_name,
        "reward_scale": reward_scale,
        "eval_target_return": eval_target_return
    }

    return config_dict