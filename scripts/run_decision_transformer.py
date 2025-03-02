import numpy as np
import torch
import tqdm
import gymnasium as gym

from rl_algs.dataset import DataHandler
from rl_algs.agents.decision_transformer_agent import DecisionTransformerAgent
from rl_algs.utility.script_util import make_logger, make_config
from rl_algs.utility.logger import Logger
from rl_algs.utility import pytorch_util as ptu
import argparse


def run_training_loop(config: dict, logger: Logger, args: argparse.Namespace):
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # load datahandle
    make_datahandler = config['make_datahandler']
    data_handler:DataHandler = make_datahandler()

    # create env
    env = data_handler.dataset_env
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    assert (
        not discrete
    ), "discrete action is not supported yet"

    # read meta data from configs
    ep_len = config["ep_len"] 

    # create dataloader
    device = ptu.device
    dataloader = data_handler.create_dataloader(
        batch_size=config['batch_size'],
        shuffle=config['shuffle'],
        device=device,
        max_length=config['batch_max_length']
    )

    # create agent
    ob_shape = env.observation_space.shape
    ac_dim = env.action_space.shape[0]

    agent = DecisionTransformerAgent(
        ob_shape,
        ac_dim,
        **config["agent_kwargs"],
    )

    dataloader_iter = iter(dataloader)
    for step in tqdm.trange(config["total_steps"], dynamic_ncols=True):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
        
        # train agent
        update_info = agent.update(
            observations=batch['observations'],
            actions=batch['actions'],
            reward_to_go=batch['rtgs'],
            timesteps=batch['timesteps'],
            attention_mask=batch['masks']
        )

        # logging
        if step % args.log_interval == 0:
            logger.log_metrics(update_info, step)
            logger.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)

    parser.add_argument("--eval_interval", "-ei", type=int, default=5000)
    parser.add_argument("--num_eval_trajectories", "-neval", type=int, default=10)
    parser.add_argument("--num_render_trajectories", "-nvid", type=int, default=0)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-g", default=0)
    parser.add_argument("--log_interval", type=int, default=10)

    args = parser.parse_args()

    # create directory for logging
    logdir_prefix = "dt_"  # keep for autograder

    config = make_config(args.config_file)
    logger = make_logger(logdir_prefix, config)

    run_training_loop(config, logger, args)


if __name__ == "__main__":
    main()
