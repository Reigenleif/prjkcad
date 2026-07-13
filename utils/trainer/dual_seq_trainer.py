from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Callable, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm
import pandas as pd
import wandb

from utils.wrapper.dual_seq_wrapper import DualSeqWrapper
from utils.evaluate import eval_cmd_only

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
        self.quant_type = quant_type.lower() if quant_type is not None else None
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
        cmd_logits, arg_logits, cmd_preds, arg_preds = outputs
        _, cmd_targets, arg_targets, _ = batch
        return self.criterion(cmd_logits, cmd_targets, arg_logits, arg_targets)

    def train_step(self, batch: Tuple, ratio: float):
        self.wrapper.train()
        self.optimizer.zero_grad(set_to_none=True)
        batch = _move_to_device(batch, self.device)
        outputs = self._forward(batch, ratio)
        loss = self._loss(outputs, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.wrapper.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return {"loss": float(loss.detach().cpu().item())}

    @torch.no_grad()
    def eval_step(self, batch: Tuple):
        self.wrapper.eval()
        batch = _move_to_device(batch, self.device)
        _, cmd_targets, arg_targets, _ = batch
        outputs = self._forward(batch, ratio=1.0)
        cmd_logits, arg_logits, cmd_preds, arg_preds = outputs
        loss = self._loss(outputs, batch)
        metrics = {"loss": float(loss.detach().cpu().item())}
        metrics["perplexity"] = float(torch.exp(loss.detach().cpu()).item())

        from utils.evaluate import eval_cmd_and_args, eval_cmd_only
        
        schema = self.wrapper.schema
        id_to_cmd = schema['id_to_command']
        cmd_arg_metric_list = []
        is_cmdonly = getattr(self.model, "is_cmd_only", False) or getattr(getattr(self.model, "cfg", None), "is_cmd_only", False)
        
        for i in range(cmd_targets.shape[0]):
            pred_cmds = [id_to_cmd.get(id, '<UNK>') for id in cmd_preds[i].cpu().numpy().tolist()]
            true_cmds = [id_to_cmd.get(id, '<UNK>') for id in cmd_targets[i].cpu().numpy().tolist()]
            
            arg_pred_list = arg_preds[i].cpu().numpy().tolist()
            arg_true_list = arg_targets[i].cpu().numpy().tolist()
            
            try:
                arg_true_list = arg_true_list[:arg_true_list.index(schema["arg_eos_id"])]
            except: pass
            
            try:
                arg_pred_list = arg_pred_list[:arg_pred_list.index(schema["arg_eos_id"])]
            except: pass

            if is_cmdonly:
                metric = eval_cmd_only(pred_cmds, true_cmds)
            else:
                metric = eval_cmd_and_args(pred_cmds, true_cmds, arg_pred_list, arg_true_list, schema)
                correct = sum(1 for j in range(min(len(arg_pred_list), len(arg_true_list))) if arg_pred_list[j] == arg_true_list[j])
                metric["arg_token_accuracy"] = correct / max(len(arg_true_list), 1)
                
            cmd_arg_metric_list.append(metric)

        if cmd_arg_metric_list:
            for key in cmd_arg_metric_list[0].keys():
                vals = [m[key] for m in cmd_arg_metric_list if m.get(key) is not None]
                if vals:
                    metrics[key] = float(np.mean(vals))
        return metrics


    def save_progression(self, folder_path: str, progression: list[dict[str, float]]):
        history_df = pd.DataFrame(progression)
        history_file_path = f"{folder_path}/history.csv"
        history_df.to_csv(history_file_path, index=False)
        
    def save_on_best_epoch(self, folder_path: str, best_epoch: int, best_epoch_metrics: dict[str, float]):
        df = pd.DataFrame([{**{"epoch": best_epoch}, **best_epoch_metrics}])
        file_path = f"{folder_path}/best_epoch.csv"
        df.to_csv(file_path, index=False)
        self.wrapper.save(folder_path)
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

        init_str += f"Number of train batches: {len(self.train_loader)}\n"
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
                        loss = self._loss(outputs, batch)
                else:
                    outputs = self._forward(batch, train_ratio)
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
                
                if wandb.run:
                    wandb.log({"train/loss": loss_val}, step=global_step)
                
                pbar.update(1)
                pbar.set_description(f"Step {global_step}/{total_steps} | Loss: {avg_loss:.4f}")
                
                if global_step % self.eval_steps == 0:
                    summary: dict[str, float] = {
                        "epoch": round(global_step / len(self.train_loader), 2) if self.train_loader else epoch + 1,
                        "step": global_step,
                        "train_loss": float(avg_loss)
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
                    
                    if wandb.run:
                        val_log = {k.replace("val_", "val/"): v for k, v in summary.items() if k.startswith("val_")}
                        grad_dict = {}
                        for name, param in self.wrapper.named_parameters():
                            if param.grad is not None:
                                grad_dict[f"gradients/{name}"] = param.grad.norm().item()
                        wandb.log({**val_log, **grad_dict}, step=global_step)
                    
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

                            if wandb.run:
                                artifact = wandb.Artifact(name=f"best_model_{wandb.run.id}", type="model")
                                for fname in ["encoder.pt", "adaptive_layer.pt", "checkpoint.pt"]:
                                    fpath = os.path.join(self.save_folder, fname)
                                    if os.path.exists(fpath):
                                        artifact.add_file(fpath)
                                wandb.log_artifact(artifact)
            
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
        return self.fit(epochs=epochs, verbose=verbose)
