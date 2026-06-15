from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, Tuple

import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

from utils.wrapper.dual_seq_wrapper import DualSeqWrapper
from utils.dual_seq import get_dualseq_schema
from utils.evaluate import eval_cmd_and_args


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


class DualSeqTrainer:
    def __init__(
        self,
        model_wrapper: DualSeqWrapper,
        criterion,
        optimizer,
        train_loader=None,
        val_loader=None,
        device: torch.device | None = None,
        max_grad_norm: float = 1.0,
        teacher_forcing_ratio: float = 1.0,
        teacher_forcing_decay: float = 1.0,
        min_teacher_forcing_ratio: float = 0.0,
        schedule_fn: Callable[[int], float] | None = None,
        max_new_cmds: int = None,
        quant_type: str | None = None,
        save_folder: str | None = None,
        best_metric_key: str = "val_avg_f1",
        best_metric_mode: str = "max"
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wrapper = model_wrapper.to(self.device)
        self.model = model_wrapper.model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_grad_norm = max_grad_norm
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.teacher_forcing_decay = teacher_forcing_decay
        self.min_teacher_forcing_ratio = min_teacher_forcing_ratio
        self.schedule_fn = schedule_fn
        self.max_new_cmds = max_new_cmds if max_new_cmds is not None else self.model.max_new_cmds

        quant_type = quant_type.lower() if quant_type is not None else None
        self.quant_type = quant_type
        
        self.save_folder = save_folder
        self.best_metric_key = best_metric_key
        self.best_metric_mode = best_metric_mode
        
    def _scheduled_ratio(self, epoch: int) -> float:
        if self.schedule_fn is not None:
            return float(self.schedule_fn(epoch))
        ratio = self.teacher_forcing_ratio * (self.teacher_forcing_decay ** epoch)
        return float(max(self.min_teacher_forcing_ratio, min(1.0, ratio)))

    def _forward(self, batch: Tuple, ratio: float):
        is_teacher_forcing = np.random.rand() < ratio
        return self.wrapper.forward(batch, is_teacher_forcing=is_teacher_forcing)


    def _loss(self, outputs, batch):
        # batch = (input_ids, cmd_targets, arg_targets, attention_mask)
        cmd_logits, cmd_preds, arg_preds = outputs
        _, cmd_targets, arg_targets, _ = batch
        return self.criterion(
            cmd_logits,
            cmd_targets,
            arg_preds,
            arg_targets,
        )

    def train_step(self, batch: Tuple, ratio: float):
        
        self.wrapper.train()
        self.optimizer.zero_grad(set_to_none=True)
        
        batch = _move_to_device(batch, self.device)
        outputs = self._forward(batch, ratio)
        cmd_logits, cmd_preds, arg_preds = outputs
        
        # pad targets to match output length for loss computation
        if cmd_logits.shape[1] > batch[1].shape[1]:
            pad_length = cmd_logits.shape[1] - batch[1].shape[1]
            pad_tensor = torch.full((batch[1].shape[0], pad_length), self.wrapper.model.pad_id, device=batch[1].device, dtype=batch[1].dtype)
            batch = (batch[0], torch.cat([batch[1], pad_tensor], dim=1), batch[2], batch[3])
        
        loss = self._loss(outputs, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.wrapper.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return {"loss": float(loss.detach().cpu().item())}

    @torch.no_grad()
    def eval_step(self, batch: Tuple):
        metrics = {}
        
        # loss and inference (always teacher-forced for consistent loss)
        self.wrapper.eval()
        batch = _move_to_device(batch, self.device)
        _, cmd_targets, arg_targets, _ = batch
        outputs = self._forward(batch, ratio=1.0)
        cmd_logits, cmd_preds, arg_preds = outputs
        preds = cmd_preds
        
        loss = self._loss(outputs, batch)
        metrics = {
            "loss": float(loss.detach().cpu().item()),
        }
        
        # perplexity (proxy via total loss)
        metrics["perplexity"] = float(torch.exp(loss.detach().cpu()).item())

        # CMD-ARG level metrics
        schema = get_dualseq_schema()['id_to_command']
        cmd_arg_metric_list = []
        for i in range(batch[1].shape[0]):
            pred_cmds = [schema.get(id, '<UNK>') for id in preds[i].cpu().numpy().tolist()]
            true_cmds = [schema.get(id, '<UNK>') for id in batch[1][i].cpu().numpy().tolist()]
            pred_args = arg_preds[i].cpu().numpy().tolist()
            true_args = arg_targets[i].cpu().numpy().tolist()
            cmd_arg_metric_list.append(eval_cmd_and_args(pred_cmds, true_cmds, pred_args, true_args))

        cmd_arg_metrics = {key: float(np.mean([metric[key] for metric in cmd_arg_metric_list])) for key in cmd_arg_metric_list[0].keys()}
        metrics.update({k : v for k, v in cmd_arg_metrics.items()})
        return metrics  

    
    def save_progression(self, 
                   folder_path: str, 
                   progression: list[dict[str, float]]):
        
        # Save all progression metrics in history.csv
        history_df = pd.DataFrame(progression)
        history_file_path = f"{folder_path}/history.csv"
        history_df.to_csv(history_file_path, index=False)
        
    def save_on_best_epoch(self, 
                   folder_path: str, 
                   best_epoch: int, 
                   best_epoch_metrics: dict[str, float]):
        
        # Save best epoch metrics as column epoch, metric1, metric2, ...
        df = pd.DataFrame([{**{"epoch": best_epoch}, **best_epoch_metrics}])
        file_path = f"{folder_path}/best_epoch.csv"
        df.to_csv(file_path, index=False)
        
        # Save model state dict
        self.wrapper.save(folder_path)
        
    def fit(self, epochs: int, verbose: bool = True):
        init_str = f"Starting training for {epochs} epochs"
        if self.quant_type is not None:
            init_str += f" with quantization: {self.quant_type}"
        
        try:
            model_param_cnt = sum(p.numel() for p in self.wrapper.model.parameters())
            init_str += f". Model parameters: {model_param_cnt:,}"
        except Exception:
            pass    
        
        print(init_str)
        
        history: list[dict[str, float]] = []
        best_metric_value = float('-inf')
        best_epoch = -1

        for epoch in range(epochs):
            train_ratio = self._scheduled_ratio(epoch)
            train_metrics: list[dict[str, float]] = []
            for batch in tqdm(self.train_loader or [], desc=f"Train {epoch + 1}/{epochs}"):
                train_metrics.append(self.train_step(batch, train_ratio))

            summary: dict[str, float] = {}
            if train_metrics:
                for key in train_metrics[0].keys():
                    summary[f"train_{key}"] = float(np.mean([metric[key] for metric in train_metrics]))

            if self.val_loader is not None:
                eval_metrics: list[dict[str, float]] = []
                for batch in tqdm(self.val_loader, desc=f"Eval {epoch + 1}/{epochs}"):
                    eval_metrics.append(self.eval_step(batch))
                if eval_metrics:
                    for key in eval_metrics[0].keys():
                        summary[f"val_{key}"] = float(np.mean([metric[key] for metric in eval_metrics]))

            history.append(summary)
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}: {summary}")
            
            if self.best_metric_key not in summary:
                raise ValueError(f"Best metric '{self.best_metric_key}' not found in summary metrics: {summary.keys()}")
            
            current_metric_value = summary[self.best_metric_key]
            if self.best_metric_mode == "min":
                if current_metric_value < best_metric_value and self.save_folder is not None:
                    best_metric_value = current_metric_value
                    best_epoch = epoch
                    self.save_on_best_epoch(self.save_folder, best_epoch, summary)
            else:
                if current_metric_value > best_metric_value and self.save_folder is not None:
                    best_metric_value = current_metric_value
                    best_epoch = epoch
                    self.save_on_best_epoch(self.save_folder, best_epoch, summary)

        if self.save_folder is not None:
            self.save_progression(self.save_folder, history)
                
        return history

    def train(self, epochs: int, verbose: bool = True):
        if self.quant_type is not None and self.device != torch.device("cuda"):
            raise ValueError(f"Quantization with type {self.quant_type} is only supported on CUDA devices.")
        
        if self.quant_type == "fp16" :
            with torch.amp.autocast("cuda", dtype=torch.float16):
                return self.fit(epochs=epochs, verbose=verbose)
        
        return self.fit(epochs=epochs, verbose=verbose)
