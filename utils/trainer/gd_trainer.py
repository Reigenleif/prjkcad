from __future__ import annotations

from typing import Any, Dict, Tuple, Union
import torch
from utils.trainer.base_trainer import BaseTrainer

class GDTrainer(BaseTrainer):
    """Data-type agnostic Gradient Descent trainer for Fine-Tuning and Pretraining."""

    def training_step(self, batch: Union[Dict[str, Any], Tuple], batch_idx: int) -> torch.Tensor:
        # <-- Forward & Loss Compute -->
        outputs = self.wrapper(batch, is_teacher_forcing=True)
        loss = self.criterion(outputs, batch)

        # <-- Metrics Logging -->
        lr = self.optimizers().param_groups[0]["lr"] if self.optimizers() else 0.0
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("lr", lr, on_step=True, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch: Union[Dict[str, Any], Tuple], batch_idx: int) -> torch.Tensor:
        # <-- Validation Forward & Loss Compute -->
        outputs = self.wrapper(batch, is_teacher_forcing=True)
        loss = self.criterion(outputs, batch)

        # <-- Validation Metrics Logging -->
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
