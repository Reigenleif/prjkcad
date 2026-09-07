import numpy as np
import torch

def compute_advantages(rewards: np.ndarray | torch.Tensor, n_rollouts: int, B: int | None = None) -> np.ndarray | torch.Tensor:
    """
    Compute group-relative advantages for B prompt groups, each with n_rollouts.
    """
    if isinstance(rewards, torch.Tensor):
        if B is None:
            B = max(1, rewards.size(0) // n_rollouts)
        rewards_np = rewards.detach().cpu().numpy()
        adv_np = compute_advantages(rewards_np, n_rollouts, B)
        return torch.tensor(adv_np, dtype=torch.float32, device=rewards.device)

    if B is None:
        B = max(1, len(rewards) // n_rollouts)

    advantages = np.zeros_like(rewards)
    for b in range(B):
        start = b * n_rollouts
        end   = start + n_rollouts
        grp   = rewards[start:end]
        mu = grp.mean()
        sigma = grp.std() + 1e-8
        advantages[start:end] = (grp - mu) / sigma
    return advantages
