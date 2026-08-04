from __future__ import annotations

from typing import Any, Dict, Union
import torch
import torch.nn as nn
from utils.criterion.base_criterion import BaseCriterion

class PretrainCriterion(BaseCriterion):
    def __init__(self, pad_id: int = 0, kl_weight: float = 1.0):
        super().__init__()
        # <-- Loss Init -->
        self.pad_id = pad_id
        self.kl_weight = kl_weight
        self.cce_loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_id)

    def forward(
        self,
        outputs: Union[Dict[str, Any], torch.Tensor],
        batch: Union[Dict[str, Any], torch.Tensor] = None,
        mu: torch.Tensor = None,
        logvar: torch.Tensor = None
    ) -> torch.Tensor:
        # <-- Input Unpacking Guard -->
        if isinstance(outputs, dict) and isinstance(batch, dict):
            logits = outputs["logits"]
            mu = outputs.get("mu")
            logvar = outputs.get("logvar")
            targets = batch.get("target_ids", batch.get("y"))
        else:
            logits = outputs
            targets = batch

        # <-- Reconstruction Loss Calculation -->
        B, L, V = logits.shape
        cce_loss = self.cce_loss_fn(logits.reshape(-1, V), targets.reshape(-1))

        # <-- KL Divergence Calculation -->
        if mu is not None and logvar is not None:
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = kl_div / B
        else:
            kl_loss = torch.tensor(0.0, device=logits.device)

        # <-- Total Loss Return -->
        return cce_loss + self.kl_weight * kl_loss
