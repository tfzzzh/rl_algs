import gymnasium as gym

class RandomAgent:
    def __init__(self, env: gym.Env):
        self.env = env

    def get_action(self, *args, **kwargs):
        return self.env.action_space.sample()