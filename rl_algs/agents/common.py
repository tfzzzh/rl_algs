from typing import Tuple
import numpy as np


def compute_gae_advantage(
    rewards: np.ndarray,
    values: np.ndarray,
    terminals: np.ndarray,
    gamma: float,
    gae_lambda: float
) -> Tuple[np.ndarray, np.ndarray]:

    n = len(rewards)
    assert n > 0
    assert len(values) == n and len(terminals) == n and len(rewards) == n
    assert len(rewards.shape) == 1 and len(values.shape) == 1 and len(terminals.shape) == 1
    assert gamma >= 0.0 and gamma <= 1.0
    assert gae_lambda >= 0.0 and gae_lambda <= 1.0

    advantages = np.zeros(n, dtype=rewards.dtype)
    for t in reversed(range(n)):
        v = values[t]
        v_next = values[t+1] if t+1 < n else 0.0
        adv_next = advantages[t+1] if t+1 < n else 0.0

        # delta_t = r_t + gamma V_{t+1} - V_{t}
        delta = rewards[t] + gamma * (1.0 - terminals[t]) * v_next - v

        # A_t = delta_t + gamma * lambda * A_{t+1}
        adv = delta + gamma * gae_lambda * (1.0 - terminals[t]) * adv_next
        advantages[t] = adv
    
    # return advantages and cumulative reward under the advantage
    returns = advantages + values
    assert returns.shape == (n,)

    return advantages, returns


def compute_reward_to_go(
    rewards: np.ndarray, terminals: np.ndarray, gamma: float
) -> np.ndarray:
    # check reward and terminal has the same shape
    # check both reward and terminal not empty
    # check terminal contains bool element
    assert len(rewards.shape) == 1
    assert rewards.shape == terminals.shape
    assert len(rewards) > 0
    assert isinstance(terminals[0], np.float32), f"terminal is of type {type(terminals[0])}"
    n = len(rewards)

    rtgs = []
    rtg = 0.0

    for i in range(n - 1, -1, -1):
        assert terminals[i] == 0.0 or terminals[i] == 1.0
        rtg = (1.0 - terminals[i]) * rtg

        rtg = rewards[i] + gamma * rtg
        rtgs.append(rtg)

    rtgs.reverse()
    return np.array(rtgs)