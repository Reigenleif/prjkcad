from __future__ import annotations

import os
from typing import Any, Dict, Optional, Union
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import TQDMProgressBar
import wandb

from utils.trainer.gd_trainer import GDTrainer
from utils.trainer.grpo_trainer import GRPOTrainer
from pytorch_lightning.callbacks import Callback
from tqdm import tqdm

class GlobalStepProgressBar(Callback):
    def __init__(self, total_steps):
        super().__init__()
        self.total_steps = total_steps
        self.pbar = None

    def on_fit_start(self, trainer, pl_module):
        self.pbar = tqdm(total=self.total_steps, desc="Training")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.pbar.update(1)
        self.pbar.set_postfix({
            "global_step": trainer.global_step,
            "epoch": trainer.current_epoch
        })

    def on_fit_end(self, trainer, pl_module):
        self.pbar.close()

class CustomTrainer:
    """Factory controller for Trainers using PyTorch Lightning Trainer."""

    def __init__(
        self,
        trainer_cfg: Any = None,
        wrapper: Optional[torch.nn.Module] = None,
        criterion: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        train_loader: Optional[Any] = None,
        val_loader: Optional[Any] = None,
        save_folder: Optional[str] = None,
        trainer_type: str = "gd",
        epochs: int = 10,
        eval_steps: int = 1000,
        run_name: Optional[str] = None,
        out_type: str = "FloatArgs",
        metadata: Any = None,
        **kwargs
    ):
        self.trainer_cfg = trainer_cfg
        self.use_wandb = kwargs.get("use_wandb", getattr(trainer_cfg, "use_wandb", True) if trainer_cfg else True)
        self.epochs = getattr(trainer_cfg, "epochs", epochs) if trainer_cfg else epochs
        self.eval_steps = getattr(trainer_cfg, "eval_steps", eval_steps) if trainer_cfg else eval_steps
        self.run_name = run_name or getattr(trainer_cfg, "run_name", None) or (os.path.basename(save_folder) if save_folder else "run")
        self.save_folder = save_folder
        self.out_type = out_type or getattr(trainer_cfg, "out_type", "FloatArgs")
        self.metadata = metadata
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.trainer_type = trainer_type
        self.kwargs = kwargs

        if trainer_type == "grpo":
            self.lightning_module = GRPOTrainer(
                wrapper=wrapper,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                save_folder=save_folder,
                **kwargs
            )
        else:
            self.lightning_module = GDTrainer(
                wrapper=wrapper,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                save_folder=save_folder,
                **kwargs
            )

    def fit(self, train_loader: Any = None, val_loader: Optional[Any] = None) -> list[dict[str, float]]:
        t_loader = train_loader if train_loader is not None else self.train_loader
        v_loader = val_loader if val_loader is not None else self.val_loader

        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        devices = 1 if torch.cuda.is_available() else "auto"

        wandb_project = os.environ.get("WANDB_PROJECT") or getattr(self.trainer_cfg, "wandb_project", None) or "PRJKCAD"
        if self.use_wandb:
            if wandb.run is not None:
                logger = WandbLogger(experiment=wandb.run)
            else:
                try:
                    logger = WandbLogger(project=wandb_project, name=self.run_name, save_dir=self.save_folder or "out")
                except Exception:
                    logger = CSVLogger(save_dir=self.save_folder or "out", name=self.run_name or "logs")
        else:
            logger = CSVLogger(save_dir=self.save_folder or "out", name=self.run_name or "logs")

        total_steps = None
        if t_loader is not None and hasattr(t_loader, "__len__"):
            total_steps = len(t_loader) * self.epochs

        val_check_interval = None
        check_val_every_n_epoch = 1
        if v_loader is not None and self.eval_steps is not None and self.eval_steps > 0:
            if total_steps is not None and total_steps > 0:
                eval_interval = min(self.eval_steps, total_steps)
            else:
                eval_interval = self.eval_steps
            if eval_interval > 0:
                val_check_interval = eval_interval
                check_val_every_n_epoch = None

        callbacks = []
        if total_steps is not None and total_steps > 0:
            callbacks.append(GlobalStepProgressBar(total_steps=total_steps))

        quant_type = self.kwargs.get("quant_type") or getattr(self.trainer_cfg, "quant_type", None)
        precision = "16-mixed" if (quant_type == "fp16" and accelerator == "gpu") else "32-true"

        pl_trainer = pl.Trainer(
            max_epochs=self.epochs,
            max_steps=total_steps if total_steps else -1,
            default_root_dir=self.save_folder or "out",
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            enable_checkpointing=False,
            logger=logger,
            val_check_interval=val_check_interval,
            check_val_every_n_epoch=check_val_every_n_epoch,
            log_every_n_steps=1,
            callbacks=callbacks, 
            enable_progress_bar=True,
        )

        pl_trainer.fit(self.lightning_module, train_dataloaders=t_loader, val_dataloaders=v_loader)

        if self.save_folder:
            os.makedirs(self.save_folder, exist_ok=True)
            pt_path = os.path.join(self.save_folder, "checkpoint.pt")
            torch.save(self.lightning_module.state_dict(), pt_path)
            ckpt_path = os.path.join(self.save_folder, "checkpoint.ckpt")
            pl_trainer.save_checkpoint(ckpt_path)
            print(f"Saved checkpoints to: {self.save_folder}")

        return self.lightning_module.progression

    def eval(self, val_loader: Optional[Any] = None) -> dict[str, float]:
        v_loader = val_loader if val_loader is not None else self.val_loader
        if v_loader is None:
            raise ValueError("val_loader must be provided for evaluation.")

        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        devices = 1 if torch.cuda.is_available() else "auto"

        wandb_project = os.environ.get("WANDB_PROJECT") or getattr(self.trainer_cfg, "wandb_project", None) or "PRJKCAD"
        if self.use_wandb:
            if wandb.run is not None:
                logger = WandbLogger(experiment=wandb.run)
            else:
                try:
                    logger = WandbLogger(project=wandb_project, name=self.run_name, save_dir=self.save_folder or "out")
                except Exception:
                    logger = CSVLogger(save_dir=self.save_folder or "out", name=self.run_name or "eval")
        else:
            logger = CSVLogger(save_dir=self.save_folder or "out", name=self.run_name or "eval")

        pl_trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            enable_checkpointing=False,
            logger=logger,
            enable_progress_bar=False,
        )
        results = pl_trainer.validate(self.lightning_module, dataloaders=v_loader)
        return results[0] if results else {}

    def validate(self, val_loader: Optional[Any] = None) -> dict[str, float]:
        return self.eval(val_loader)
