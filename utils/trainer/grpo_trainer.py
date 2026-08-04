from __future__ import annotations

from typing import Any, Dict, Tuple, Union
import torch
from utils.trainer.base_trainer import BaseTrainer
from utils.grpo import compute_advantages, grpo_loss

class GRPOTrainer(BaseTrainer):
    """Data-type agnostic Group Relative Policy Optimization trainer."""

    def __init__(
        self,
        wrapper: Any,
        criterion: Any = None,
        optimizer: Any = None,
        scheduler: Any = None,
        n_rollouts: int = 8,
        clip_eps: float = 0.2,
        temperature: float = 1.0,
        *args,
        **kwargs
    ):
        super().__init__(wrapper, criterion, optimizer, scheduler, *args, **kwargs)
        # <-- GRPO Hyperparameters Setup -->
        self.n_rollouts = n_rollouts
        self.clip_eps = clip_eps
        self.temperature = temperature

    def training_step(self, batch: Union[Dict[str, Any], Tuple], batch_idx: int) -> torch.Tensor:
        # <-- Input Unpacking -->
        if isinstance(batch, dict):
            input_ids = batch.get("input_ids", batch.get("x"))
            attn_mask = batch.get("attention_mask", batch.get("attn_mask"))
        else:
            input_ids, attn_mask = batch[0], batch[1]

        # <-- Rollout Sampling -->
        rollout = self.wrapper.generate_rollout(input_ids, attn_mask, n_rollouts=self.n_rollouts, temperature=self.temperature)
        ref_log_probs = rollout["log_probs"].detach()

        # <-- Advantages & Loss Computation -->
        advantages = self._compute_dummy_advantages(rollout["sampled_cmds"].size(0))
        loss = grpo_loss(
            self.wrapper,
            input_ids,
            attn_mask,
            rollout["sampled_cmds"].cpu().numpy().tolist(),
            rollout["sampled_args"].cpu().numpy().tolist(),
            ref_log_probs,
            advantages,
            self.clip_eps,
            max_T=rollout["sampled_cmds"].size(1)
        )

        # <-- Metrics Logging -->
        self.log("grpo_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def _compute_dummy_advantages(self, n_samples: int) -> torch.Tensor:
        # <-- Advantage Estimation Helper -->
        rewards = torch.randn(n_samples, device=self.device)
        return compute_advantages(rewards, n_rollouts=self.n_rollouts)
