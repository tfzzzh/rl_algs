from rl_algs.agents.ppo_transformer import PPOTransformer
import gymnasium as gym
import numpy as np
import torch
from rl_algs.utility import pytorch_util as ptu
from rl_algs.utility.logger import Logger

from rl_algs.utility.script_util import make_logger, make_config

import argparse

def run_training_loop(config: dict, logger: Logger, args: argparse.Namespace):
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # make the gym environment
    env = config["make_env"]()
    eval_env = config["make_env"]()
    ep_len = config["ep_len"] or env.spec.max_episode_steps
    total_steps = config['total_steps']
    action_space = env.action_space

    ob_shape = env.observation_space.shape
    ac_dim = env.action_space.shape[0]

    # make agent
    agent = PPOTransformer(ob_shape, ac_dim, ep_len, action_space, **config["agent_kwargs"])
    train_rollout_buffer = agent.create_rollout_buffer(env, is_train_env=True)
    eval_rollout_buffer = agent.create_rollout_buffer(eval_env, is_train_env=False)

    num_step = 0
    itr = 0
    while num_step < total_steps:
        # train
        train_rollout_buffer.expand_episode()
        num_step += len(train_rollout_buffer)
        train_info = agent.update(train_rollout_buffer)

        # log train info
        if itr % args.log_interval == 0:
            print(f"iter = {itr}, write log to tensorboard, steps={num_step}/{total_steps}")
            logger.log_metrics(train_info, num_step)

        # log eval info
        if itr % args.eval_interval == 0:
            returns = []
            ep_lens = []

            for _ in range(config['num_eval_episode']):
                eval_rollout_buffer.expand_episode()
                episode_len = len(eval_rollout_buffer)
                returns.append(eval_rollout_buffer.rewards[:episode_len].sum())
                ep_lens.append(episode_len)
            
            logger.log_scalar(np.mean(returns), "eval_return", num_step)
            logger.log_scalar(np.mean(ep_lens), "eval_ep_len", num_step)

            if len(returns) > 1:
                logger.log_scalar(np.std(returns), "eval/return_std", num_step)
                logger.log_scalar(np.max(returns), "eval/return_max", num_step)
                logger.log_scalar(np.min(returns), "eval/return_min", num_step)
                logger.log_scalar(np.std(ep_lens), "eval/ep_len_std", num_step)
                logger.log_scalar(np.max(ep_lens), "eval/ep_len_max", num_step)
                logger.log_scalar(np.min(ep_lens), "eval/ep_len_min", num_step)


        itr += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)

    parser.add_argument("--eval_interval", "-ei", type=int, default=5)
    parser.add_argument("--num_eval_trajectories", "-neval", type=int, default=10)
    parser.add_argument("--num_render_trajectories", "-nvid", type=int, default=0)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-g", default=0)
    parser.add_argument("--log_interval", type=int, default=5)

    args = parser.parse_args()

    # create directory for logging
    logdir_prefix = "ppotf_"  # keep for autograder

    config = make_config(args.config_file)
    logger = make_logger(logdir_prefix, config)

    run_training_loop(config, logger, args)


if __name__ == "__main__":
    main()