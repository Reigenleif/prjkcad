from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Tuple

import numpy as np
import torch
from tqdm import tqdm
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
        best_metric_mode: str = "max"
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
        best_epoch = -1

        for epoch in range(epochs):
            train_metrics: list[dict[str, float]] = []
            for batch in tqdm(self.train_loader or [], desc=f"Pretrain Train {epoch + 1}/{epochs}"):
                train_metrics.append(self.train_step(batch))

            summary: dict[str, float] = {}
            if train_metrics:
                all_train_keys = set()
                for m in train_metrics:
                    all_train_keys.update(m.keys())
                for key in all_train_keys:
                    vals = [metric[key] for metric in train_metrics if metric.get(key) is not None]
                    if vals:
                        summary[f"train_{key}"] = float(np.mean(vals))

            if self.val_loader is not None:
                eval_metrics: list[dict[str, float]] = []
                for batch in tqdm(self.val_loader, desc=f"Pretrain Eval {epoch + 1}/{epochs}"):
                    eval_metrics.append(self.eval_step(batch))
                if eval_metrics:
                    all_val_keys = set()
                    for m in eval_metrics:
                        all_val_keys.update(m.keys())
                    for key in all_val_keys:
                        vals = [metric[key] for metric in eval_metrics if metric.get(key) is not None]
                        if vals:
                            summary[f"val_{key}"] = float(np.mean(vals))

            history.append(summary)
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}: {summary}")

            if self.best_metric_key in summary:
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

                if is_best:
                    best_epoch = epoch
                    if self.save_folder is not None:
                        self.save_on_best_epoch(self.save_folder, best_epoch, summary)

        if self.save_folder is not None:
            self.save_progression(self.save_folder, history)

        return history

    def train(self, epochs: int, verbose: bool = True) -> list[dict[str, float]]:
        return self.fit(epochs=epochs, verbose=verbose)
