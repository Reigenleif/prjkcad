from __future__ import annotations

from typing import Any, Dict, Optional, Union
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from utils.trainer.base_trainer import BaseTrainer
from utils.trainer.gd_trainer import GDTrainer
from utils.trainer.grpo_trainer import GRPOTrainer

class CustomTrainer(BaseTrainer):
    """Factory controller for Trainer components configuring PyTorch Lightning execution."""

    def __init__(
        self,
        trainer_cfg: Any = None,
        wrapper: Optional[torch.nn.Module] = None,
        criterion: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        save_folder: Optional[str] = None,
        trainer_type: str = "gd",
        epochs: int = 10,
        **kwargs
    ):
        super().__init__(wrapper, criterion, optimizer, scheduler, save_folder=save_folder)
        # <-- Initialization Subroutines -->
        self.trainer_type = trainer_type
        self.epochs = getattr(trainer_cfg, "epochs", epochs) if trainer_cfg else epochs
        self.lightning_module = self._create_trainer_variant(trainer_cfg, wrapper, criterion, optimizer, scheduler, **kwargs)
        self.pl_trainer = self._build_pl_trainer(save_folder, self.epochs)

    def _create_trainer_variant(self, trainer_cfg, wrapper, criterion, optimizer, scheduler, **kwargs) -> BaseTrainer:
        # <-- Select Trainer Variant Guard -->
        if self.trainer_type == "grpo" or getattr(trainer_cfg, "type", None) == "grpo":
            return GRPOTrainer(wrapper, criterion, optimizer, scheduler, **kwargs)
        return GDTrainer(wrapper, criterion, optimizer, scheduler, **kwargs)

    def _build_pl_trainer(self, save_folder: Optional[str], epochs: int) -> pl.Trainer:
        # <-- PyTorch Lightning Trainer Builder -->
        callbacks = []
        if save_folder:
            callbacks.append(ModelCheckpoint(dirpath=save_folder, filename="checkpoint", save_top_k=1, monitor="val_loss", mode="min"))

        return pl.Trainer(
            max_epochs=epochs,
            callbacks=callbacks,
            enable_progress_bar=True,
            enable_checkpointing=bool(save_folder),
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1
        )

    def fit(self, train_loader: Any, val_loader: Optional[Any] = None) -> None:
        # <-- Execute Training Loop via PyTorch Lightning -->
        self.pl_trainer.fit(self.lightning_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
        if self.save_folder:
            self.lightning_module.save_progression(self.save_folder)
