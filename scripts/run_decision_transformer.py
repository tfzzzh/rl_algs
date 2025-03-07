import numpy as np
import torch
import tqdm
import gymnasium as gym

from rl_algs.dataset import DataHandler
from rl_algs.agents.decision_transformer_agent import DecisionTransformerAgent, DtInferenceState
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
    env_name = env.unwrapped.spec.id
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    assert (
        not discrete
    ), "discrete action is not supported yet"

    # print banner
    dataset_info = data_handler.statistics_info
    print('=' * 50)
    print(f'Starting new experiment: {env_name}, {config['dataset_name']}')
    print(f'{dataset_info['num_trajectories']} trajectories, {dataset_info['num_timesteps']} timesteps found')
    print(f'Average return: {dataset_info['mean_returns']:.2f}, std: {dataset_info['std_returns']:.2f}')
    print(f'Max return: {dataset_info['max_returns']:.2f}, min: {dataset_info['min_returns']:.2f}')
    print(f'each epoch has {dataset_info['num_trajectories'] / config['batch_size']:.2f} steps')
    print('=' * 50)

    # read meta data from configs
    ep_len = config["ep_len"] 
    reward_scale = config['reward_scale']

    # Define shape variables first
    ob_shape = env.observation_space.shape
    ac_dim = env.action_space.shape[0]

    # create dataloader
    device = ptu.device
    dataloader = data_handler.create_dataloader(
        reward_scale=reward_scale,
        batch_size=config['batch_size'],
        shuffle=config['shuffle'],
        device=device,
        max_length=config['batch_max_length']
    )

    # create inference state
    infer_state = DtInferenceState(
        ob_shape, 
        ac_dim, 
        reward_scale, 
        observation_mean = dataset_info['state_mean'],
        observation_std = dataset_info['state_std']
    )

    # create agent
    agent = DecisionTransformerAgent(
        ob_shape,
        ac_dim,
        infer_state,
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

        # evaluation
        if step % args.eval_interval == 0:
            trajectories = rollout_trajectory(
                env,
                policy=agent,
                ntraj=args.num_eval_trajectories,
                max_length=ep_len,
                target_return=config['eval_target_return']
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


def rollout_trajectory(
    env: gym.Env, agent: DecisionTransformerAgent, max_length: int, target_return: float
):
    """Sample a rollout in the environment from a policy."""
    ob, init_info = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    
    # reset inference state
    agent.infer_state.reset(ob, target_return)
    while True:
        # TODO use the most recent ob to decide what to do
        ac = agent.get_action()

        # TODO: take that action and get reward and next ob
        next_ob, rew, terminated, truncated, info = env.step(ac)
        done = terminated or truncated

        # TODO rollout can end due to done, or due to max_length
        steps += 1
        rollout_done = done or steps > max_length  # HINT: this is either 0 or 1

        # update inference state
        agent.infer_state.update(ac, rew, next_ob)

        # record result of taking that action
        obs.append(ob)
        acs.append(ac)
        rewards.append(rew)
        next_obs.append(next_ob)
        terminals.append(terminated) # when trace truncated the done is not seted

        ob = next_ob  # jump to next timestep

        # end the rollout if the rollout ended
        if rollout_done:
            break

    episode_statistics = {"l": steps, "r": np.sum(rewards)}
    if "episode" in info:
        episode_statistics.update(info["episode"])

    env.close()

    return {
        "observation": np.array(obs, dtype=np.float32),
        "image_obs": np.array(image_obs, dtype=np.uint8),
        "reward": np.array(rewards, dtype=np.float32),
        "action": np.array(acs, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminal": np.array(terminals, dtype=np.float32),
        "episode_statistics": episode_statistics,
    }

def sample_n_trajectories(
    env: gym.Env, policy: DecisionTransformerAgent, ntraj: int, max_length: int, render: bool = False
):
    """Collect ntraj rollouts."""
    trajs = []
    for _ in range(ntraj):
        # collect rollout
        traj = rollout_trajectory(env, policy, max_length, render)
        trajs.append(traj)
    return trajs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)

    parser.add_argument("--eval_interval", "-ei", type=int, default=20)
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
