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
    render_env = config["make_env"](render=True)
    ep_len = config["ep_len"] or env.spec.max_episode_steps
    batch_size = config["batch_size"]
    total_steps = config['total_steps']

    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    assert discrete == config["is_discrete"]

    ob_shape = env.observation_space.shape
    ac_dim = env.action_space.shape[0]

    # simulation timestep, will be used for video saving
    if "model" in dir(env):
        fps = 1 / env.model.opt.timestep
    else:
        fps = env.env.metadata["render_fps"]

    # init agent
    agent = PPO(ob_shape, ac_dim, **config["agent_kwargs"])

    # init rollout buffer
    rollout_buffer = RolloutBuffer(max_length=ep_len, gamma=config['agent_kwargs']['discount'],
                                    gae_gamma=config['agent_kwargs']['gae_lambda'], obs_type=np.float32)
    
    num_step = 0
    itr = 0
    while num_step < total_steps:
        # rollout using current actor
        # trajs, step_batch = utils.sample_trajectories(
        #     env, agent, batch_size, ep_len
        # )
        # num_step += step_batch
        num_step += rollout_buffer.rollout(agent, env, batch_size)

        # update agent and bookmark infos
        # trajs_dict = {k: [traj[k] for traj in trajs] for k in trajs[0]}
        # train_info = agent.update(
        #     obs=trajs_dict["observation"],
        #     actions=trajs_dict["action"],
        #     rewards=trajs_dict["reward"],
        #     terminals=trajs_dict["terminal"],
        #     step=itr,
        # )
        train_info = agent.update(
            obs=[rollout_buffer.observations],
            actions=[rollout_buffer.actions],
            rewards=[rollout_buffer.rewards],
            terminals=[rollout_buffer.dones],
            step=itr,
        )

        # write logs
        if itr % args.log_interval == 0:
            print(f"iter = {itr}, write log to tensorboard, steps={num_step}/{total_steps}")
            for k, v in train_info.items():
                logger.log_scalar(v, k, num_step)
            logger.flush()

        # perform evaluation
        if itr % args.eval_interval == 0:
            trajectories = utils.sample_n_trajectories(
                eval_env,
                policy=agent,
                ntraj=args.num_eval_trajectories,
                max_length=ep_len,
            )
            returns = [t["episode_statistics"]["r"] for t in trajectories]
            ep_lens = [t["episode_statistics"]["l"] for t in trajectories]

            logger.log_scalar(np.mean(returns), "eval_return", num_step)
            logger.log_scalar(np.mean(ep_lens), "eval_ep_len", num_step)

            if len(returns) > 1:
                logger.log_scalar(np.std(returns), "eval/return_std", num_step)
                logger.log_scalar(np.max(returns), "eval/return_max", num_step)
                logger.log_scalar(np.min(returns), "eval/return_min", num_step)
                logger.log_scalar(np.std(ep_lens), "eval/ep_len_std", num_step)
                logger.log_scalar(np.max(ep_lens), "eval/ep_len_max", num_step)
                logger.log_scalar(np.min(ep_lens), "eval/ep_len_min", num_step)

            if args.num_render_trajectories > 0:
                video_trajectories = utils.sample_n_trajectories(
                    render_env,
                    agent,
                    args.num_render_trajectories,
                    ep_len,
                    render=True,
                )

                logger.log_paths_as_videos(
                    video_trajectories,
                    num_step,
                    fps=fps,
                    max_videos_to_save=args.num_render_trajectories,
                    video_title="eval_rollouts",
                )

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
