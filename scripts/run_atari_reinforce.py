from rl_algs.agents.reinforce import AtariDiscreteReinforce
from rl_algs.utility.replay_buffer import MemoryEfficientReplayBuffer, ReplayBuffer

import os
import gymnasium as gym
import numpy as np
import torch
from rl_algs.utility import pytorch_util as ptu
import tqdm

from rl_algs.utility import utils
from rl_algs.utility.logger import Logger

from rl_algs.utility.script_util import make_logger, make_config

import argparse


def run_training_loop(config: dict, logger: Logger, args: argparse.Namespace):
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # create env
    env = config["make_env"]()
    eval_env = config["make_env"]()
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise NotImplementedError("only discrete action space is supported")
    ep_len = config["ep_len"] or env.spec.max_episode_steps
    batch_size = config["batch_size"]

    # create agent
    num_action = env.action_space.n
    agent = AtariDiscreteReinforce(
        num_action=num_action, **config["agent_kwargs"]
    )

    # create action explorer
    action_getter = config['make_action_getter'](num_action)

    # init cpt
    checkpoint_dir = os.path.join(logger._log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    agent.save_checkpoint(os.path.join(checkpoint_dir, f"step=0.cpt"))

    # create replay buffer
    frame_history_len = env.observation_space.shape[0]
    assert frame_history_len == 4, "only support 4 stacked frames"
    # replay_buffer = MemoryEfficientReplayBuffer(frame_history_len=frame_history_len)
    replay_buffer = ReplayBuffer(capacity=config['replay_buffer_capacity'])

    # reset env
    observation, _ = env.reset()
    episode_stats = {'train_return': 0.0, "train_ep_len": 0}
    assert isinstance(observation, np.ndarray)
    # replay_buffer.on_reset(observation=observation[-1, ...])

    action_counter = np.zeros(num_action, dtype=np.int64)
    for step in tqdm.trange(config["total_steps"], dynamic_ncols=True):
        explore_eps, action = action_getter.get_action(step, agent.get_action(observation))
        action_counter[action] += 1

        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        episode_stats['train_return'] += reward
        episode_stats['train_ep_len'] += 1

        # TODO: test it
        # replay_buffer.insert(
        #     action=action,
        #     reward=clip_reward(reward),
        #     next_observation=next_observation[-1],
        #     done=terminated,
        # )
        replay_buffer.insert(observation, action, clip_reward(reward), next_observation, done)

        if done:
            # logger.log_scalar(info["episode"]["r"], "train_return", step)
            # logger.log_scalar(info["episode"]["l"], "train_ep_len", step)
            logger.log_metrics(episode_stats, step)
            episode_stats = {'train_return': 0.0, "train_ep_len": 0}
            observation, _ = env.reset()
            # replay_buffer.on_reset(observation=observation[-1, ...])
        else:
            observation = next_observation

        # update agent
        if step >= config["random_steps"] and step % args.update_freq == 0:
            batch = replay_buffer.sample(batch_size)
            batch = ptu.from_numpy(batch)
            update_info = agent.update(
                observations=batch["observations"],
                actions=batch["actions"],
                rewards=batch["rewards"],
                next_observations=batch["next_observations"],
                dones=batch["dones"],
                step=step,
            )

            if step % args.log_interval == 0:
                logger.log_metrics(update_info, step)
                action_sample_dist = action_counter / (action_counter.sum() + 1e-10)
                action_sample_dist_dict = {f"action_{a}": action_sample_dist[a] for a in range(num_action)}
                logger.log_scalars(action_sample_dist_dict, "action", step, "sampled")
                logger.log_scalar(explore_eps, "explore_eps", step)


            # run evaluation
            if step % args.eval_interval == 0:
                eval_info = utils.evaluate_performance(
                    env=eval_env, 
                    agent=agent, 
                    num_eval_trajectories=args.num_eval_trajectories, 
                    max_ep_len=ep_len
                )

                logger.log_metrics(eval_info, step)

            if step % args.ckp_interval == 0 or step == config["total_steps"] - 1:
                agent.save_checkpoint(os.path.join(checkpoint_dir, f"step={step}.cpt"))


def clip_reward(reward):
    if reward > 0:
        return 1.0
    elif reward == 0:
        return 0.0
    else:
        return -1.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)

    parser.add_argument("--eval_interval", "-ei", type=int, default=10000)
    parser.add_argument("--num_eval_trajectories", "-neval", type=int, default=3)
    parser.add_argument("--num_render_trajectories", "-nvid", type=int, default=0)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--log_interval", type=int, default=1000)
    parser.add_argument("--ckp_interval", type=int, default=21000)
    parser.add_argument("--update_freq", type=int, default=1)

    args = parser.parse_args()

    # create directory for logging
    logdir_prefix = "atari_0329"  # keep for autograder

    config = make_config(args.config_file)
    logger = make_logger(logdir_prefix, config)

    run_training_loop(config, logger, args)


if __name__ == "__main__":
    main()
