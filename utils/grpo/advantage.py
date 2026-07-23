import numpy as np

def compute_advantages(rewards: np.ndarray, n_rollouts: int, B: int) -> np.ndarray:
    """
    Compute group-relative advantages for B prompt groups, each with n_rollouts.
    """
    advantages = np.zeros_like(rewards)
    for b in range(B):
        start = b * n_rollouts
        end   = start + n_rollouts
        grp   = rewards[start:end]
        mu = grp.mean()
        sigma = grp.std() + 1e-8
        advantages[start:end] = (grp - mu) / sigma
    return advantages
