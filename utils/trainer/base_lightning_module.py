from __future__ import annotations

import os
from typing import Any, Dict, Optional
import pandas as pd
import torch
import pytorch_lightning as pl

class BaseLightningModule(pl.LightningModule):
    """Base PyTorch Lightning module providing unified optimization and metric tracking."""

    def __init__(
        self,
        wrapper: Optional[torch.nn.Module] = None,
        criterion: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        max_grad_norm: float = 1.0,
        save_folder: Optional[str] = None,
        *args,
        **kwargs
    ):
        super().__init__()
        self.wrapper = wrapper
        self.criterion = criterion
        self.custom_optimizer = optimizer
        self.custom_scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        self.save_folder = save_folder
        self.progression: list[dict[str, float]] = []

    def forward(self, *args, **kwargs) -> Any:
        if self.wrapper is None:
            raise NotImplementedError("Module wrapper is not set.")
        return self.wrapper(*args, **kwargs)

    def configure_optimizers(self) -> Any:
        if self.custom_optimizer is None:
            opt = torch.optim.AdamW(self.parameters(), lr=1e-3)
            return opt

        if self.custom_scheduler is not None:
            return {
                "optimizer": self.custom_optimizer,
                "lr_scheduler": {
                    "scheduler": self.custom_scheduler,
                    "interval": "step"
                }
            }
        return self.custom_optimizer

    def on_validation_epoch_end(self) -> None:
        metrics = {}
        for k, v in self.trainer.callback_metrics.items():
            if isinstance(v, torch.Tensor):
                metrics[k] = float(v.detach().cpu().item())
            elif isinstance(v, (float, int)):
                metrics[k] = float(v)
        if metrics:
            self.progression.append(metrics)

    def save_progression(self, folder_path: str) -> None:
        if not self.progression:
            return
        os.makedirs(folder_path, exist_ok=True)
        pd.DataFrame(self.progression).to_csv(os.path.join(folder_path, "history.csv"), index=False)
