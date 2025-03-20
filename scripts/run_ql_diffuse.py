import numpy as np
import torch
import tqdm
import gymnasium as gym

from rl_algs.dataset import DataHandler
from rl_algs.agents.ql_diffuse import QLDiffuseAgent
from rl_algs.utility.script_util import make_logger, make_config
from rl_algs.utility.logger import Logger
from rl_algs.utility import pytorch_util as ptu
from rl_algs.utility.utils import sample_n_trajectories
import argparse


def run_training_loop(config: dict, logger: Logger, args: argparse.Namespace):
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # load datahandle
    make_datahandler = config["make_datahandler"]
    data_handler: DataHandler = make_datahandler()

    # create env
    env = data_handler.dataset_env
    env_name = env.unwrapped.spec.id
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    assert not discrete, "discrete action is not supported yet"

    # read meta data from configs
    ep_len = config["ep_len"]
    action_min_value = float(env.action_space.low[0])
    action_max_value = float(env.action_space.high[0])

    # Define shape variables first
    ob_shape = env.observation_space.shape
    ac_dim = env.action_space.shape[0]

    # create agent
    agent = QLDiffuseAgent(
        ob_shape,
        ac_dim,
        **config["agent_kwargs"],
        action_min_value=action_min_value,
        action_max_value=action_max_value
    )

    # get replay buffer
    replaybuffer = data_handler.create_replaybuffer(capacity=config["rb_capacity"])
    for step in tqdm.trange(config["total_steps"], dynamic_ncols=True):
        batch = replaybuffer.sample(batch_size=config["batch_size"])
        batch = ptu.from_numpy(batch)

        update_info = agent.update(**batch, step=step)

        if step % args.log_interval == 0:
            logger.log_metrics(update_info, step)
            logger.flush()

        if step % args.eval_interval == 0:
            trajectories = sample_n_trajectories(
                env,
                policy=agent,
                ntraj=args.num_eval_trajectories,
                max_length=ep_len,
            )

            returns = [t["episode_statistics"]["r"] for t in trajectories]
            ep_lens = [t["episode_statistics"]["l"] for t in trajectories]

            logger.log_scalar(np.mean(returns), "eval_return", step)
            logger.log_scalar(np.mean(ep_lens), "eval_ep_len", step)

            if len(returns) > 1:
                logger.log_scalar(np.std(returns), "eval/return_std", step)
                logger.log_scalar(np.max(returns), "eval/return_max", step)
                logger.log_scalar(np.min(returns), "eval/return_min", step)
                logger.log_scalar(np.std(ep_lens), "eval/ep_len_std", step)
                logger.log_scalar(np.max(ep_lens), "eval/ep_len_max", step)
                logger.log_scalar(np.min(ep_lens), "eval/ep_len_min", step)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)

    parser.add_argument("--eval_interval", "-ei", type=int, default=500)
    parser.add_argument("--num_eval_trajectories", "-neval", type=int, default=2)
    parser.add_argument("--num_render_trajectories", "-nvid", type=int, default=0)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-g", default=0)
    parser.add_argument("--log_interval", type=int, default=10)

    args = parser.parse_args()

    # create directory for logging
    logdir_prefix = "diffuse_"  # keep for autograder

    config = make_config(args.config_file)
    logger = make_logger(logdir_prefix, config)

    run_training_loop(config, logger, args)


if __name__ == "__main__":
    main()
