from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Callable, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm
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
        best_metric_mode: str = "max",
        eval_steps: int = 1000,
        scheduler = None
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
        self.eval_steps = eval_steps
        self.scheduler = scheduler

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
        
        # Trim targets if outputs are shorter (due to max_new_cmds truncation)
        T_out = cmd_logits.size(1)
        if cmd_targets.size(1) > T_out:
            cmd_targets = cmd_targets[:, :T_out]
            arg_targets = arg_targets[:, :T_out, :]
            
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
        if self.scheduler is not None:
            self.scheduler.step()
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

        # Trim targets to match outputs length (in case of truncation/padding mismatches)
        T_out = preds.size(1)
        if cmd_targets.size(1) > T_out:
            cmd_targets = cmd_targets[:, :T_out]
            arg_targets = arg_targets[:, :T_out, :]
        elif cmd_targets.size(1) < T_out:
            preds = preds[:, :cmd_targets.size(1)]
            arg_preds = arg_preds[:, :cmd_targets.size(1), :]
            cmd_logits = cmd_logits[:, :cmd_targets.size(1), :]
            outputs = (cmd_logits, preds, arg_preds)
        
        loss = self._loss(outputs, (batch[0], cmd_targets, arg_targets, batch[3]))
        metrics = {
            "loss": float(loss.detach().cpu().item()),
        }
        
        # perplexity (proxy via total loss)
        metrics["perplexity"] = float(torch.exp(loss.detach().cpu()).item())

        # CMD-ARG level metrics
        schema = get_dualseq_schema()['id_to_command']
        cmd_arg_metric_list = []
        for i in range(cmd_targets.shape[0]):
            pred_cmds = [schema.get(id, '<UNK>') for id in preds[i].cpu().numpy().tolist()]
            true_cmds = [schema.get(id, '<UNK>') for id in cmd_targets[i].cpu().numpy().tolist()]
            pred_args = arg_preds[i].float().cpu().numpy().tolist()
            true_args = arg_targets[i].cpu().numpy().tolist()
            cmd_arg_metric_list.append(eval_cmd_and_args(pred_cmds, true_cmds, pred_args, true_args))

        cmd_arg_metrics = {}
        if cmd_arg_metric_list:
            for key in cmd_arg_metric_list[0].keys():
                vals = [metric[key] for metric in cmd_arg_metric_list if metric.get(key) is not None]
                if vals:
                    cmd_arg_metrics[key] = float(np.mean(vals))
        metrics.update(cmd_arg_metrics)
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
        
        # Save optimizer and scheduler
        if self.optimizer is not None:
            torch.save(self.optimizer.state_dict(), os.path.join(folder_path, "optimizer.pt"))
        if self.scheduler is not None:
            torch.save(self.scheduler.state_dict(), os.path.join(folder_path, "scheduler.pt"))

    def fit(self, epochs: int, verbose: bool = True):
        init_str = f"Starting training for {epochs} epochs\n"
        if self.quant_type is not None:
            init_str += f"Quantization: {self.quant_type}.\n"
        
        try:
            model_param_cnt = sum(p.numel() for p in self.wrapper.model.parameters())
            trainable_param_cnt = sum(p.numel() for p in self.wrapper.model.parameters() if p.requires_grad)
            init_str += f"Model parameters: {model_param_cnt:,} (trainable: {trainable_param_cnt:,})\n"
        except Exception:
            pass    

        # Number of train batches
        init_str += f"Number of train batches: {len(self.train_loader)}\n"
        # Number of val batches
        if self.val_loader is not None:
            init_str += f"Number of val batches: {len(self.val_loader)}\n"
        else:
            init_str += "No validation loader provided\n"
        
        print(init_str)
        
        history: list[dict[str, float]] = []
        best_metric_value = float('inf') if self.best_metric_mode == "min" else float('-inf')
        best_epoch_or_step = -1
        
        total_steps = epochs * len(self.train_loader) if self.train_loader else 0
        global_step = 0
        train_losses = []
        
        pbar = tqdm(total=total_steps, desc="Training")
        
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        use_amp = self.quant_type == "fp16"
        scaler = torch.amp.GradScaler("cuda") if (use_amp and dtype == torch.float16) else None
        
        for epoch in range(epochs):
            train_ratio = self._scheduled_ratio(epoch)
            for batch in (self.train_loader or []):
                self.wrapper.train()
                self.optimizer.zero_grad(set_to_none=True)
                
                batch = _move_to_device(batch, self.device)
                
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=dtype):
                        outputs = self._forward(batch, train_ratio)
                        cmd_logits, cmd_preds, arg_preds = outputs
                        if cmd_logits.shape[1] > batch[1].shape[1]:
                            pad_length = cmd_logits.shape[1] - batch[1].shape[1]
                            pad_tensor = torch.full((batch[1].shape[0], pad_length), self.wrapper.model.pad_id, device=batch[1].device, dtype=batch[1].dtype)
                            batch = (batch[0], torch.cat([batch[1], pad_tensor], dim=1), batch[2], batch[3])
                        loss = self._loss(outputs, batch)
                else:
                    outputs = self._forward(batch, train_ratio)
                    cmd_logits, cmd_preds, arg_preds = outputs
                    if cmd_logits.shape[1] > batch[1].shape[1]:
                        pad_length = cmd_logits.shape[1] - batch[1].shape[1]
                        pad_tensor = torch.full((batch[1].shape[0], pad_length), self.wrapper.model.pad_id, device=batch[1].device, dtype=batch[1].dtype)
                        batch = (batch[0], torch.cat([batch[1], pad_tensor], dim=1), batch[2], batch[3])
                    loss = self._loss(outputs, batch)
                
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.wrapper.model.parameters(), self.max_grad_norm)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.wrapper.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    
                if self.scheduler is not None:
                    self.scheduler.step()
                    
                loss_val = float(loss.detach().cpu().item())
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
                            if use_amp:
                                with torch.amp.autocast("cuda", dtype=dtype):
                                    eval_metrics.append(self.eval_step(val_batch))
                            else:
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
                if self.optimizer is not None:
                    torch.save(self.optimizer.state_dict(), os.path.join(self.save_folder, "optimizer.pt"))
                if self.scheduler is not None:
                    torch.save(self.scheduler.state_dict(), os.path.join(self.save_folder, "scheduler.pt"))
                
        return history

    def train(self, epochs: int, verbose: bool = True):
        if self.quant_type is not None and self.device != torch.device("cuda"):
            raise ValueError(f"Quantization with type {self.quant_type} is only supported on CUDA devices.")
        
        return self.fit(epochs=epochs, verbose=verbose)
