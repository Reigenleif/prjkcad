import torch
import torch.nn as nn


class PretrainCriterion(nn.Module):
    """
    Criterion for autoencoder-like pretraining.
    Combines reconstruction loss (Categorical Cross Entropy) and KL divergence regularizer.
    """
    def __init__(self, pad_id: int = 0, kl_weight: float = 1.0):
        super().__init__()
        self.pad_id = pad_id
        self.kl_weight = kl_weight
        self.cce_loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_id)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, V = logits.shape

        # 1. Reconstruction Loss (CCE)
        cce_loss = self.cce_loss_fn(logits.reshape(-1, V), targets.reshape(-1))

        # 2. KL Divergence Loss
        # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_div / B

        total_loss = cce_loss + self.kl_weight * kl_loss

        return total_loss, cce_loss, kl_loss
