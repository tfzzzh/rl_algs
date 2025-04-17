import argparse
import os
import gymnasium as gym
import numpy as np
import torch
from rl_algs.utility import pytorch_util as ptu
from rl_algs.agents.ppo import PPOAtari, RolloutBufferMultEnvs
from rl_algs.utility import utils
from rl_algs.utility.logger import Logger

from rl_algs.utility.script_util import make_logger, make_config


def run_training_loop(config: dict, logger: Logger, args: argparse.Namespace):
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # make the gym environment
    envs = [config["make_env"]() for _ in range(config['num_train_env'])]
    eval_env = config["make_env"]()
    env = envs[0]
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise NotImplementedError("only discrete action space is supported")

    ep_len = config["ep_len"] or env.spec.max_episode_steps
    total_steps = config["total_steps"]

    # init agent
    ob_shape = env.observation_space.shape
    num_action = env.action_space.n
    agent = PPOAtari(ob_shape, num_action, **config["agent_kwargs"])

    if config['pretrained_model_path'] is not None:
        print(f"use pretrained model in {config['pretrained_model_path']}")
        agent.load_from_checkpoint(config['pretrained_model_path'], load_optimizer=False)

    # init rollout buffer
    # rollout_buffer = RolloutBuffer(
    #     max_length=ep_len,
    #     gamma=config["agent_kwargs"]["discount"],
    #     gae_gamma=config["agent_kwargs"]["gae_lambda"],
    #     obs_type=np.uint8,
    #     reward_trans=lambda x: np.sign(x)
    # )
    rollout_buffer = RolloutBufferMultEnvs(
        rollout_batch=config['min_batch_per_step'],
        gamma=config["agent_kwargs"]["discount"],
        gae_gamma=config["agent_kwargs"]["gae_lambda"],
        obs_type=np.uint8,
        num_envs=config['num_train_env']
    )

    # init cpt
    checkpoint_dir = os.path.join(logger._log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    agent.save_checkpoint(os.path.join(checkpoint_dir, f"step=0.cpt"))

    num_step, itr = 0, 0
    while num_step < total_steps:
        rollout_step, episode_info = rollout_buffer.rollout(agent, envs)
        num_step += rollout_step
        train_info = agent.update_from_rb(rollout_buffer=rollout_buffer)

        # write logs
        if itr % args.log_interval == 0:
            print(
                f"iter = {itr}, write log to tensorboard, steps={num_step}/{total_steps}"
            )
            logger.log_metrics({**episode_info, **train_info}, num_step)
            logger.flush()

        # perform evaluation
        if itr % args.eval_interval == 0:
            perf = utils.evaluate_performance(
                eval_env, agent, args.num_eval_trajectories, ep_len
            )
            logger.log_metrics(perf, num_step)

        if itr % args.ckp_interval == 0 or num_step >= config["total_steps"]:
            agent.save_checkpoint(os.path.join(checkpoint_dir, f"step={num_step}.cpt"))

        itr += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)

    parser.add_argument("--eval_interval", "-ei", type=int, default=1)
    parser.add_argument("--num_eval_trajectories", "-neval", type=int, default=2)
    parser.add_argument("--num_render_trajectories", "-nvid", type=int, default=0)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-g", default=0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--ckp_interval", type=int, default=800)

    args = parser.parse_args()

    # create directory for logging
    logdir_prefix = "ppo_atari_"  # keep for autograder

    config = make_config(args.config_file)
    logger = make_logger(logdir_prefix, config)

    run_training_loop(config, logger, args)


if __name__ == "__main__":
    main()
