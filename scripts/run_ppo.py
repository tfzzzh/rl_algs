from rl_algs.agents.ppo import PPO, RolloutBuffer
import gymnasium as gym
import numpy as np
import torch
from rl_algs.utility import pytorch_util as ptu

from rl_algs.utility import utils
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
    batch_size = config["batch_size"]
    total_steps = config['total_steps']

    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    assert discrete == config["is_discrete"]

    ob_shape = env.observation_space.shape
    ac_dim = env.action_space.shape[0]

    # init agent
    agent = PPO(ob_shape, ac_dim, **config["agent_kwargs"])

    # init rollout buffer
    rollout_buffer = RolloutBuffer(max_length=ep_len, gamma=config['agent_kwargs']['discount'],
                                    gae_gamma=config['agent_kwargs']['gae_lambda'], obs_type=np.float32)
    
    num_step = 0
    itr = 0
    while num_step < total_steps:
        roll_step, _ = rollout_buffer.rollout(agent, env, batch_size)
        num_step += roll_step
        train_info = agent.update_from_rb(rollout_buffer=rollout_buffer)

        # write logs
        if itr % args.log_interval == 0:
            print(f"iter = {itr}, write log to tensorboard, steps={num_step}/{total_steps}")
            for k, v in train_info.items():
                logger.log_scalar(v, k, num_step)
            logger.flush()

        # perform evaluation
        if itr % args.eval_interval == 0:
            perf = utils.evaluate_performance(eval_env, agent, args.num_eval_trajectories, ep_len)
            logger.log_metrics(perf, num_step)

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
    logdir_prefix = "ppo_"  # keep for autograder

    config = make_config(args.config_file)
    logger = make_logger(logdir_prefix, config)

    run_training_loop(config, logger, args)


if __name__ == "__main__":
    main()
