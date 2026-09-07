from __future__ import annotations

import os
from typing import Any, Dict, Optional, Union
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, CSVLogger
import wandb

from utils.trainer.gd_lightning_module import GDLightningModule
from utils.trainer.grpo_lightning_module import GRPOLightningModule
from utils.trainer.components.progress_bar import GlobalStepProgressBar

class SaveStateDictCallback(pl.Callback):
    """Callback to save PyTorch state_dict checkpoint.pt on every validation end."""

    def __init__(self, save_folder: str):
        super().__init__()
        self.save_folder = save_folder

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        os.makedirs(self.save_folder, exist_ok=True)
        pt_path = os.path.join(self.save_folder, "checkpoint.pt")
        torch.save(pl_module.state_dict(), pt_path)

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
        pipeline: Optional[Any] = None,
        subset_refresh_every: int = 0,
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
        self.pipeline = pipeline if pipeline is not None else kwargs.pop("pipeline", None)
        self.subset_refresh_every = subset_refresh_every if subset_refresh_every else kwargs.pop("subset_refresh_every", 0)
        self.kwargs = kwargs

        module_kwargs = {"out_type": self.out_type, "metadata": self.metadata, **kwargs}

        if trainer_type == "grpo":
            self.lightning_module = GRPOLightningModule(
                wrapper=wrapper,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                save_folder=save_folder,
                pipeline=self.pipeline,
                subset_refresh_every=self.subset_refresh_every,
                **module_kwargs
            )
        else:
            self.lightning_module = GDLightningModule(
                wrapper=wrapper,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                save_folder=save_folder,
                **module_kwargs
            )

    def fit(self, train_loader: Any = None, val_loader: Optional[Any] = None) -> list[dict[str, float]]:
        t_loader = train_loader if train_loader is not None else self.train_loader
        v_loader = val_loader if val_loader is not None else self.val_loader

        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        devices = 1 if torch.cuda.is_available() else "auto"

        wandb_project = os.environ.get("WANDB_PROJECT") or getattr(self.trainer_cfg, "wandb_project", None) or "PRJKCAD"
        if self.use_wandb:
            if wandb.run is not None:
                logger = WandbLogger(experiment=wandb.run, log_model=False)
            else:
                try:
                    logger = WandbLogger(project=wandb_project, name=self.run_name, save_dir=self.save_folder or "out", log_model=False)
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
            if t_loader is not None and hasattr(t_loader, "__len__") and len(t_loader) > 0:
                n_batches_per_epoch = len(t_loader)
                if self.eval_steps >= n_batches_per_epoch:
                    check_val_every_n_epoch = max(1, round(self.eval_steps / n_batches_per_epoch))
                    val_check_interval = 1.0
                else:
                    val_check_interval = self.eval_steps
                    check_val_every_n_epoch = 1
            else:
                val_check_interval = self.eval_steps
                check_val_every_n_epoch = 1

        callbacks = []
        if total_steps is not None and total_steps > 0:
            callbacks.append(GlobalStepProgressBar(total_steps=total_steps))

        if self.save_folder:
            monitor_metric = getattr(self.trainer_cfg, "checkpoint_monitor", "val/loss" if v_loader is not None else "train/loss")
            mode = getattr(self.trainer_cfg, "checkpoint_mode", "min")
            save_top_k = getattr(self.trainer_cfg, "save_top_k", 1)

            ckpt_callback = ModelCheckpoint(
                dirpath=self.save_folder,
                filename="best_checkpoint",
                monitor=monitor_metric,
                mode=mode,
                save_top_k=save_top_k,
                save_last=True,
            )
            callbacks.append(ckpt_callback)
            callbacks.append(SaveStateDictCallback(save_folder=self.save_folder))

        extra_callbacks = self.kwargs.get("callbacks", [])
        if extra_callbacks:
            if isinstance(extra_callbacks, list):
                callbacks.extend(extra_callbacks)
            else:
                callbacks.append(extra_callbacks)

        log_every_n_steps = getattr(self.trainer_cfg, "log_every_n_steps", 1) if self.trainer_cfg else 1
        if total_steps is not None and total_steps > 0:
            log_every_n_steps = min(log_every_n_steps, total_steps)
        log_every_n_steps = max(1, log_every_n_steps)

        quant_type = self.kwargs.get("quant_type") or getattr(self.trainer_cfg, "quant_type", None)
        precision = "16-mixed" if (quant_type == "fp16" and accelerator == "gpu") else "32-true"

        pl_trainer = pl.Trainer(
            max_epochs=self.epochs,
            max_steps=total_steps if total_steps else -1,
            default_root_dir=self.save_folder or "out",
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            enable_checkpointing=True,
            logger=logger,
            val_check_interval=val_check_interval,
            check_val_every_n_epoch=check_val_every_n_epoch,
            log_every_n_steps=log_every_n_steps,
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
            print(f"Saved final checkpoints to: {self.save_folder}")
            self.save_progression(self.save_folder)

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

    def save_progression(self, folder_path: str) -> None:
        if hasattr(self.lightning_module, "save_progression"):
            self.lightning_module.save_progression(folder_path)
