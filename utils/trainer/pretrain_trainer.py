from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm
import pandas as pd

from utils.wrapper.pretrain_wrapper import PretrainWrapper
from utils.evaluate import eval_reconstruction


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


class PretrainTrainer:
    """
    Trainer for autoencoder-like pretraining.
    Trains the reconstructor model using reconstruction and KL divergence losses.
    """
    def __init__(
        self,
        model_wrapper: PretrainWrapper,
        criterion,
        optimizer,
        train_loader=None,
        val_loader=None,
        device: torch.device | None = None,
        max_grad_norm: float = 1.0,
        save_folder: str | None = None,
        best_metric_key: str = "val_f1",
        best_metric_mode: str = "max",
        eval_steps: int = 1000,
        scheduler = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wrapper = model_wrapper.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_grad_norm = max_grad_norm
        self.save_folder = save_folder
        self.best_metric_key = best_metric_key
        self.best_metric_mode = best_metric_mode
        self.eval_steps = eval_steps
        self.scheduler = scheduler

    def train_step(self, batch: Tuple) -> dict[str, float]:
        self.wrapper.train()
        self.optimizer.zero_grad(set_to_none=True)

        batch = _move_to_device(batch, self.device)
        input_ids, attention_mask, target_ids = batch

        logits, mu, logvar = self.wrapper(batch)

        loss, cce_loss, kl_loss = self.criterion(logits, target_ids, mu, logvar)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.wrapper.parameters(), self.max_grad_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return {
            "loss": float(loss.detach().cpu().item()),
            "cce_loss": float(cce_loss.detach().cpu().item()),
            "kl_loss": float(kl_loss.detach().cpu().item())
        }

    @torch.no_grad()
    def eval_step(self, batch: Tuple) -> dict[str, float]:
        self.wrapper.eval()
        batch = _move_to_device(batch, self.device)
        input_ids, attention_mask, target_ids = batch

        logits, mu, logvar = self.wrapper(batch)

        loss, cce_loss, kl_loss = self.criterion(logits, target_ids, mu, logvar)

        pred_ids = logits.argmax(dim=-1)

        pred_list = pred_ids.cpu().numpy().tolist()
        target_list = target_ids.cpu().numpy().tolist()

        pad_token_id = self.wrapper.text_tokenizer.pad_token_id or 0
        eval_metrics = eval_reconstruction(pred_list, target_list, pad_token=pad_token_id)

        metrics = {
            "loss": float(loss.detach().cpu().item()),
            "cce_loss": float(cce_loss.detach().cpu().item()),
            "kl_loss": float(kl_loss.detach().cpu().item()),
            "precision": eval_metrics["precision"],
            "recall": eval_metrics["recall"],
            "f1": eval_metrics["f1"]
        }
        return metrics

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

    def fit(self, epochs: int, verbose: bool = True) -> list[dict[str, float]]:
        print(f"Starting autoencoder pretraining for {epochs} epochs")
        history: list[dict[str, float]] = []
        best_metric_value = float('inf') if self.best_metric_mode == "min" else float('-inf')
        best_epoch_or_step = -1
        
        total_steps = epochs * len(self.train_loader) if self.train_loader else 0
        global_step = 0
        train_losses = []
        
        pbar = tqdm(total=total_steps, desc="Pretrain Training")
        
        for epoch in range(epochs):
            for batch in (self.train_loader or []):
                step_metrics = self.train_step(batch)
                loss_val = step_metrics["loss"]
                train_losses.append(loss_val)
                avg_loss = np.mean(train_losses)
                global_step += 1
                
                pbar.update(1)
                pbar.set_description(f"Step {global_step}/{total_steps} | Loss: {avg_loss:.4f}")
                
                if global_step % self.eval_steps == 0:
                    summary: dict[str, float] = {
                        "epoch": round(global_step / len(self.train_loader), 2) if self.train_loader else epoch + 1,
                        "step": global_step,
                        "train_loss": float(avg_loss)
                    }
                    
                    if self.val_loader is not None:
                        eval_metrics: list[dict[str, float]] = []
                        for val_batch in self.val_loader:
                            eval_metrics.append(self.eval_step(val_batch))
                        if eval_metrics:
                            all_val_keys = set()
                            for m in eval_metrics:
                                all_val_keys.update(m.keys())
                            for key in all_val_keys:
                                vals = [metric[key] for metric in eval_metrics if metric.get(key) is not None]
                                if vals:
                                    summary[f"val_{key}"] = float(np.mean(vals))
                    
                    history.append(summary)
                    if len(history) == 1:
                        pbar.write(format_log_header(summary))
                    pbar.write(format_log_row(summary, history[0]))
                    
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
            
        pbar.close()
        
        if self.save_folder is not None:
            self.save_progression(self.save_folder, history)
            if best_epoch_or_step == -1:
                self.wrapper.save(self.save_folder)
                
        return history

    def train(self, epochs: int, verbose: bool = True) -> list[dict[str, float]]:
        return self.fit(epochs=epochs, verbose=verbose)
