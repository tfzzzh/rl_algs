import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
from gymnasium.wrappers import RecordEpisodeStatistics
import ale_py
from typing import Optional
import numpy as np


# def wrap_deepmind(env: gym.Env):
#     """
#     Wraps a Gym environment with a series of preprocessing steps commonly used 
#     for training reinforcement learning agents on Atari games.
#     """
#     # Standard Atari preprocessing
#     env = AtariPreprocessing(
#         env,
#         noop_max=30,
#         frame_skip=4,
#         screen_size=84,
#         terminal_on_life_loss=True,
#         grayscale_obs=True,
#     )
#     env = FrameStackObservation(env, stack_size=4)
#     env = FireResetEnv(env)

#     # Record the statistics of the _underlying_ environment, before frame-skip/reward-clipping/etc.
#     env = RecordEpisodeStatistics(env)
#     return env

def wrap_deepmind(env: gym.Env):
    """
    Wraps a Gym environment with a series of preprocessing steps commonly used 
    for training reinforcement learning agents on Atari games.
    """
    # Record the statistics of the _underlying_ environment, before frame-skip/reward-clipping/etc.
    # env = RecordEpisodeStatistics(env)

    # Standard Atari preprocessing
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=True,
        grayscale_obs=True,
    )

    env = FireResetEnv(env)
    env = FrameStackObservation(env, stack_size=4)
    return env


def make_atari(name:str, render_mode: Optional[str] = None, firereset=True):
    """make a atari game environment in openai style

    example:
        make_atari("ALE/Breakout-v5")
        make_atari("ALE/Breakout-v5", render_mode="human")
    """
    args = {
        'id': name,
        'frameskip': 1
    }

    if render_mode is not None:
        args['render_mode'] = render_mode
    
    env = gym.make(**args)

    return wrap_deepmind(env)

# class FireResetEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
#     """
#     Take action on reset for environments that are fixed until firing.

#     :param env: Environment to wrap
#     """

#     def __init__(self, env: gym.Env) -> None:
#         super().__init__(env)
#         assert env.unwrapped.get_action_meanings()[1] == "FIRE"  # type: ignore[attr-defined]
#         assert len(env.unwrapped.get_action_meanings()) >= 3  # type: ignore[attr-defined]


#     def reset(self, **kwargs):
#         self.env.reset(**kwargs)
#         obs, _, terminated, truncated, _ = self.env.step(1)
#         if terminated or truncated:
#             self.env.reset(**kwargs)
#         obs, _, terminated, truncated, info = self.env.step(2)
#         if terminated or truncated:
#             self.env.reset(**kwargs)
#         return obs, info
    

#     def step(self, ac):
#         return self.env.step(ac)

class FireResetEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Take action on reset for environments that are fixed until firing.

    :param env: Environment to wrap
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"  # type: ignore[attr-defined]
        assert len(env.unwrapped.get_action_meanings()) >= 3  # type: ignore[attr-defined]

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(1)
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(2)
        if terminated or truncated:
            self.env.reset(**kwargs)
        return obs, {}