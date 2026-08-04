from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import wandb

from utils.wandb import log_wandb_train, log_wandb_eval

def _move_to_device(batch: Any, device: torch.device):
    if isinstance(batch, Mapping):
        return {key: _move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(_move_to_device(value, device) for value in batch)
    if isinstance(batch, list):
        return [_move_to_device(value, device) for value in batch]
    if hasattr(batch, "to"):
        return batch.to(device)
    return batch

def format_log_header(metrics: dict[str, Any]) -> str:
    special_keys = ["epoch", "step", "train_loss"]
    other_keys = sorted([k for k in metrics.keys() if k not in special_keys])
    ordered_keys = [k for k in special_keys if k in metrics] + other_keys
    header = " | ".join(f"{k:<18}" for k in ordered_keys)
    separator = "-" * len(header)
    return f"{header}\n{separator}"

def format_log_row(metrics: dict[str, Any], header_metrics: dict[str, Any]) -> str:
    special_keys = ["epoch", "step", "train_loss"]
    other_keys = sorted([k for k in header_metrics.keys() if k not in special_keys])
    ordered_keys = [k for k in special_keys if k in header_metrics] + other_keys
    row_parts = []
    for k in ordered_keys:
        val = metrics.get(k, "")
        if isinstance(val, float):
            row_parts.append(f"{val:<18.4f}")
        elif isinstance(val, (int, bool)):
            row_parts.append(f"{str(val):<18}")
        else:
            row_parts.append(f"{str(val):<18}")
    return " | ".join(row_parts)

class BaseTrainer:
    def __init__(
        self,
        model_wrapper: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader = None,
        val_loader = None,
        device: torch.device | None = None,
        max_grad_norm: float = 1.0,
        quant_type: str | None = None,
        save_folder: str | None = None,
        best_metric_key: str = "val_loss",
        best_metric_mode: str = "min",
        eval_steps: int = 1000,
        scheduler = None,
        log_artifacts: bool = False,
        *args,
        **kwargs
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wrapper = model_wrapper.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_grad_norm = max_grad_norm
        self.quant_type = quant_type.lower() if quant_type is not None else None
        self.save_folder = save_folder
        self.best_metric_key = best_metric_key
        self.best_metric_mode = best_metric_mode
        self.eval_steps = eval_steps
        self.scheduler = scheduler
        self.log_artifacts = log_artifacts

    def train_step(self, batch: Tuple, epoch: int, use_amp: bool = False, dtype = None, scaler = None) -> dict[str, float]:
        raise NotImplementedError

    def eval_step(self, batch: Tuple) -> dict[str, float]:
        raise NotImplementedError

    def save_progression(self, folder_path: str, progression: list[dict[str, float]]):
        os.makedirs(folder_path, exist_ok=True)
        history_df = pd.DataFrame(progression)
        history_file_path = os.path.join(folder_path, "history.csv")
        history_df.to_csv(history_file_path, index=False)

    def save_on_best_epoch(self, folder_path: str, best_epoch: int, best_epoch_metrics: dict[str, float]):
        os.makedirs(folder_path, exist_ok=True)
        df = pd.DataFrame([{**{"epoch": best_epoch}, **best_epoch_metrics}])
        file_path = os.path.join(folder_path, "best_epoch.csv")
        df.to_csv(file_path, index=False)
        self.wrapper.save(folder_path)
        if self.optimizer is not None:
            torch.save(self.optimizer.state_dict(), os.path.join(folder_path, "optimizer.pt"))
        if self.scheduler is not None:
            torch.save(self.scheduler.state_dict(), os.path.join(folder_path, "scheduler.pt"))

    def _do_eval(
        self,
        global_step: int,
        epoch: int,
        avg_loss: float,
        step_metrics: dict[str, float],
        use_amp: bool,
        dtype: torch.dtype,
        history: list[dict[str, float]],
        best_metric_value: float,
        best_epoch_or_step: int,
        pbar: tqdm
    ) -> tuple[float, int]:
        summary: dict[str, float] = {
            "epoch": round(global_step / len(self.train_loader), 2) if self.train_loader else epoch + 1,
            "step": global_step,
            "train_loss": float(avg_loss),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "grad_norm": float(step_metrics.get("grad_norm", 0.0))
        }
        if self.val_loader is not None:
            eval_metrics = []
            for val_batch in self.val_loader:
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=dtype):
                        eval_metrics.append(self.eval_step(val_batch))
                else:
                    eval_metrics.append(self.eval_step(val_batch))
            if eval_metrics:
                for key in set(k for m in eval_metrics for k in m.keys()):
                    vals = [m[key] for m in eval_metrics if m.get(key) is not None]
                    if vals:
                        summary[f"val_{key}"] = float(np.mean(vals))

        history.append(summary)
        if len(history) == 1:
            pbar.write(format_log_header(summary))
        pbar.write(format_log_row(summary, history[0]))

        log_wandb_eval(
            summary=summary,
            step=global_step,
            wrapper=self.wrapper,
            save_folder=self.save_folder,
            log_artifacts=self.log_artifacts
        )

        if self.val_loader is not None and self.best_metric_key in summary:
            current_metric_value = summary[self.best_metric_key]
            is_best = False
            if self.best_metric_mode == "min":
                if current_metric_value < best_metric_value:
                    best_metric_value = current_metric_value
                    is_best = True
            else:
                if current_metric_value > best_metric_value:
                    best_metric_value = current_metric_value
                    is_best = True
            if is_best and self.save_folder is not None:
                best_epoch_or_step = global_step
                self.save_on_best_epoch(self.save_folder, best_epoch_or_step, summary)
        return best_metric_value, best_epoch_or_step

    def fit(self, epochs: int, verbose: bool = True) -> list[dict[str, float]]:
        init_str = f"Starting training for {epochs} epochs\n"
        if self.quant_type is not None:
            init_str += f"Quantization: {self.quant_type}.\n"
        try:
            model_param_cnt = sum(p.numel() for p in self.wrapper.model.parameters())
            trainable_param_cnt = sum(p.numel() for p in self.wrapper.model.parameters() if p.requires_grad)
            init_str += f"Model parameters: {model_param_cnt:,} (trainable: {trainable_param_cnt:,})\n"
        except Exception:
            pass

        init_str += f"Number of train batches: {len(self.train_loader) if self.train_loader else 0}\n"
        if self.val_loader is not None:
            init_str += f"Number of val batches: {len(self.val_loader)}\n"
        else:
            init_str += "No validation loader provided\n"
        if verbose:
            print(init_str)

        history: list[dict[str, float]] = []
        best_metric_value = float('inf') if self.best_metric_mode == "min" else float('-inf')
        best_epoch_or_step = -1

        total_steps = epochs * len(self.train_loader) if self.train_loader else 0
        global_step = 0
        train_losses = []
        avg_loss = 0.0
        step_metrics = {}

        pbar = tqdm(total=total_steps, desc="Training")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        use_amp = self.quant_type == "fp16"
        scaler = torch.amp.GradScaler("cuda") if (use_amp and dtype == torch.float16) else None

        for epoch in range(epochs):
            for batch in (self.train_loader or []):
                step_metrics = self.train_step(batch, epoch, use_amp=use_amp, dtype=dtype, scaler=scaler)
                loss_val = step_metrics["loss"]
                train_losses.append(loss_val)
                avg_loss = float(np.mean(train_losses))
                global_step += 1

                log_dict = {f"train/{k}": v for k, v in step_metrics.items()}

                pbar.update(1)
                pbar.set_description(f"Step {global_step}/{total_steps} | Loss: {avg_loss:.4f}")

                if global_step % self.eval_steps == 0:
                    best_metric_value, best_epoch_or_step = self._do_eval(
                        global_step, epoch, avg_loss, step_metrics, use_amp, dtype, history, best_metric_value, best_epoch_or_step, pbar
                    )

                log_wandb_train(log_dict, step=global_step)

        if global_step % self.eval_steps != 0:
            best_metric_value, best_epoch_or_step = self._do_eval(
                global_step, epochs - 1, avg_loss, step_metrics, use_amp, dtype, history, best_metric_value, best_epoch_or_step, pbar
            )

        pbar.close()
        if self.save_folder is not None:
            self.save_progression(self.save_folder, history)
            if best_epoch_or_step == -1:
                self.wrapper.save(self.save_folder)
                if self.optimizer is not None:
                    torch.save(self.optimizer.state_dict(), os.path.join(self.save_folder, "optimizer.pt"))
                if self.scheduler is not None:
                    torch.save(self.scheduler.state_dict(), os.path.join(self.save_folder, "scheduler.pt"))
        return history

    def train(self, epochs: int, verbose: bool = True) -> list[dict[str, float]]:
        return self.fit(epochs=epochs, verbose=verbose)
