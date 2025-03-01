import minari
import numpy as np
from typing import List
import torch
from torch.utils.data import DataLoader


class DataHandler:
    def __init__(self, dataset_name: str, download=True):
        self.dataset = minari.load_dataset(dataset_name, download=download)
        self.dataset_env = self.dataset.recover_environment()
        # self.env_name = self.env.unwrapped.spec.id

    def create_dataloader(self, batch_size: int, shuffle: bool, device: torch.device, max_length: int) -> torch.utils.data.DataLoader:
        collate_fn = CollateFunc(device, max_length)
        dataloader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
        )
        return dataloader

    def get_statistics(self):
        # TODO: refine computation of this function
        states, traj_lens, returns = [], [], []
        for episode_data in self.dataset.iterate_episodes():
            states.append(episode_data.observations)
            traj_lens.append(len(episode_data.observations))
            returns.append(episode_data.rewards.sum())
        traj_lens, returns = np.array(traj_lens), np.array(returns)

        # used for input normalization
        states = np.concatenate(states, axis=0)
        state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        num_timesteps = sum(traj_lens)

        info = {
            "num_trajectories": len(traj_lens),
            "num_timesteps": num_timesteps,
            "mean_returns": np.mean(returns),
            "std_returns": np.std(returns),
            "max_returns": np.max(returns),
            "min_returns": np.min(returns),
            "state_mean": state_mean,
            "state_std": state_std,
        }

        return info


class CollateFunc:
    def __init__(self, device: torch.device, max_length: int):
        self.device = device
        self.max_length = max_length

    def __call__(self, batch):
        lengths = [len(x.observations) for x in batch]

        # sampling start index
        starts = [np.random.randint(0, length) for length in lengths]

        # get ends of each sequence
        ends = [
            min(start + self.max_length, length)
            for (start, length) in zip(starts, lengths)
        ]

        # compose upadded mask
        # masks = [np.ones(end - start, dtype=np.float32) for (start, end) in zip(starts, ends)]

        # return sampled data
        observations = []
        actions = []
        rewards = []
        terminations = []
        truncations = []
        timesteps = []  # int64
        rtgs = []  # reward to go, with gamma set to 1.0
        masks = []

        # foreach sampled episode perform transformation
        for i, x in enumerate(batch):
            start, end = starts[i], ends[i]

            observations.append(x.observations[start:end])
            actions.append(x.actions[start:end])
            rewards.append(x.rewards[start:end])
            terminations.append(x.terminations[start:end])
            truncations.append(x.truncations[start:end])
            timesteps.append(np.arange(start, end))

            # reverse reward them compute cumsum from back to forward
            reward_to_go: np.ndarray = x.rewards[start:end]
            reward_to_go = reward_to_go[::-1].cumsum()
            reward_to_go = reward_to_go[::-1]
            reward_to_go = np.expand_dims(reward_to_go, axis=1)
            rtgs.append(np.ascontiguousarray(reward_to_go))

            # attention mask
            masks.append(np.ones(end - start, dtype=np.bool_))

        # padd and for new batch
        device = self.device
        batch_in_device = {
            "id": torch.Tensor([x.id for x in batch]),
            "observations": _pad_sequences(
                observations, device=device, dtype=torch.float32
            ),
            "actions": _pad_sequences(actions, device=device, dtype=torch.float32),
            "rewards": _pad_sequences(rewards, device=device, dtype=torch.float32),
            "terminations": _pad_sequences(
                terminations, device=device, dtype=torch.float32
            ),
            "truncations": _pad_sequences(
                truncations, device=device, dtype=torch.float32
            ),
            "timesteps": _pad_sequences(timesteps, device=device, dtype=torch.long),
            "rtgs": _pad_sequences(rtgs, device=device, dtype=torch.float32),
            "masks": _pad_sequences(masks, device=device, dtype=torch.bool),
        }

        return batch_in_device


def _pad_sequences(seqs: List[np.ndarray], device: torch.device, dtype) -> torch.Tensor:
    return torch.nn.utils.rnn.pad_sequence(
        [torch.as_tensor(x, device=device, dtype=dtype) for x in seqs], batch_first=True
    )
