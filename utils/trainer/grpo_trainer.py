from __future__ import annotations

import os
from collections import deque
from typing import Any, Callable, Tuple, Mapping

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from utils.wrapper.grpo_wrapper import GRPOWrapper
from utils.dual_seq import get_dualseq_schema
from utils.evaluate import eval_cmd_and_args, compute_cd, compute_reward


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


def decode_sequence(cmd_ids, arg_vals, schema):
    id_to_command = schema["id_to_command"]
    arg_names = schema["arg_names"]
    command_to_slice = schema["command_to_slice"]
    pad_id = schema["pad_id"]
    eos_id = schema["eos_id"]
    
    cmds = []
    args = []
    
    for cmd_id, args_row in zip(cmd_ids, arg_vals):
        if cmd_id in (pad_id, eos_id):
            break
        cmd_name = id_to_command.get(cmd_id, None)
        if cmd_name is None or cmd_name in ("SOS", "EOS", "PAD"):
            continue
            
        cmds.append(cmd_name)
        arg_dict = {}
        if cmd_name in command_to_slice:
            start, end = command_to_slice[cmd_name]
            cmd_arg_names = arg_names[start:end]
            cmd_arg_vals = args_row[start:end]
            for name, val in zip(cmd_arg_names, cmd_arg_vals):
                arg_dict[name] = float(val)
        args.append(arg_dict)
        
    return cmds, args


def format_log_header(metrics: dict[str, Any]) -> str:
    special_keys = ["epoch", "step", "train_loss", "train_reward"]
    other_keys = sorted([k for k in metrics.keys() if k not in special_keys])
    ordered_keys = [k for k in special_keys if k in metrics] + other_keys
    header = " | ".join(f"{k:<18}" for k in ordered_keys)
    separator = "-" * len(header)
    return f"{header}\n{separator}"


def format_log_row(metrics: dict[str, Any], header_metrics: dict[str, Any]) -> str:
    special_keys = ["epoch", "step", "train_loss", "train_reward"]
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


class GRPOTrainer:
    def __init__(
        self,
        model_wrapper: GRPOWrapper,
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
        best_metric_key: str = "val_mean_reward",
        best_metric_mode: str = "max",
        eval_steps: int = 1000,
        scheduler = None,
        n_rollouts: int = 8,
        clip_eps: float = 0.2,
        min_cd: float = 1e-5,
        max_cd: float = 0.5,
        eval_fraction: float = 0.1,
        temperature: float = 1.0
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
        
        # GRPO specific parameters
        self.n_rollouts = n_rollouts
        self.clip_eps = clip_eps
        self.min_cd = min_cd
        self.max_cd = max_cd
        self.eval_fraction = eval_fraction
        self.temperature = temperature

    def _forward(self, batch: Tuple, ratio: float):
        # Forward pass used for loss evaluation (with teacher forcing)
        return self.wrapper.forward(batch, is_teacher_forcing=True)

    def _loss(self, outputs, batch):
        cmd_logits, cmd_preds, arg_preds = outputs
        _, cmd_targets, arg_targets, _ = batch
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

    def train_step(self, batch: Tuple) -> dict[str, float]:
        self.wrapper.train()
        
        # batch = (input_ids, cmd_targets, arg_targets, attention_mask)
        batch = _move_to_device(batch, self.device)
        input_ids, cmd_targets, arg_targets, attention_mask = batch
        B = input_ids.size(0)
        
        # Generate rollouts
        sampled_cmds, sampled_args, log_probs = self.wrapper.generate_rollout(
            input_ids, attention_mask, n_rollouts=self.n_rollouts, temperature=self.temperature
        )
        
        # Compute rewards and advantages
        rewards = torch.zeros(B * self.n_rollouts, device=self.device)
        schema = self.wrapper.dual_seq_schema
        
        for i in range(B):
            gt_cmds, gt_args = decode_sequence(
                cmd_targets[i].cpu().numpy().tolist(),
                arg_targets[i].cpu().numpy().tolist(),
                schema
            )
            for j in range(self.n_rollouts):
                k = i * self.n_rollouts + j
                gen_cmds, gen_args = decode_sequence(
                    sampled_cmds[k].cpu().numpy().tolist(),
                    sampled_args[k].cpu().numpy().tolist(),
                    schema
                )
                cd = compute_cd(gen_cmds, gen_args, gt_cmds, gt_args)
                rewards[k] = compute_reward(cd, self.min_cd, self.max_cd)
                
        # Group relative normalization
        advantages = torch.zeros_like(rewards)
        for i in range(B):
            group_slice = slice(i * self.n_rollouts, (i + 1) * self.n_rollouts)
            group_rewards = rewards[group_slice]
            mean_r = group_rewards.mean()
            std_r = group_rewards.std()
            if std_r > 1e-6:
                advantages[group_slice] = (group_rewards - mean_r) / (std_r + 1e-8)
            else:
                advantages[group_slice] = group_rewards - mean_r
                
        # Policy gradient loss calculation
        self.optimizer.zero_grad(set_to_none=True)
        
        pad_id = self.wrapper.model.pad_id
        mask = (sampled_cmds != pad_id).float()
        
        ratio = torch.exp(log_probs - log_probs.detach())
        surr1 = ratio * advantages.unsqueeze(-1)
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages.unsqueeze(-1)
        
        loss = - (torch.min(surr1, surr2) * mask).sum() / (mask.sum() + 1e-8)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.wrapper.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
            
        return {
            "loss": float(loss.detach().cpu().item()),
            "reward": float(rewards.mean().cpu().item())
        }

    @torch.no_grad()
    def eval_step(self, batch: Tuple) -> dict[str, float]:
        metrics = {}
        
        self.wrapper.eval()
        batch = _move_to_device(batch, self.device)
        _, cmd_targets, arg_targets, _ = batch
        outputs = self._forward(batch, ratio=1.0)
        cmd_logits, cmd_preds, arg_preds = outputs
        preds = cmd_preds

        # Trim targets to match outputs length
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
        metrics["perplexity"] = float(torch.exp(loss.detach().cpu()).item())

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
        init_str = f"Starting GRPO training for {epochs} epochs"
        if self.quant_type is not None:
            init_str += f" with quantization: {self.quant_type}"
        
        try:
            model_param_cnt = sum(p.numel() for p in self.wrapper.model.parameters())
            trainable_param_cnt = sum(p.numel() for p in self.wrapper.model.parameters() if p.requires_grad)
            init_str += f". Model parameters: {model_param_cnt:,} (trainable: {trainable_param_cnt:,})"
        except Exception:
            pass    
        print(init_str)
        
        history: list[dict[str, float]] = []
        best_metric_value = float('inf') if self.best_metric_mode == "min" else float('-inf')
        best_epoch_or_step = -1
        
        total_steps = epochs * len(self.train_loader) if self.train_loader else 0
        global_step = 0
        train_losses = []
        train_rewards = []
        
        pbar = tqdm(total=total_steps, desc="Training")
        
        for epoch in range(epochs):
            for batch in (self.train_loader or []):
                step_metrics = self.train_step(batch)
                train_losses.append(step_metrics["loss"])
                train_rewards.append(step_metrics["reward"])
                global_step += 1
                
                pbar.update(1)
                pbar.set_description(
                    f"Step {global_step}/{total_steps} | Loss: {np.mean(train_losses):.4f} | Reward: {np.mean(train_rewards):.4f}"
                )
                
                if global_step % self.eval_steps == 0:
                    summary: dict[str, float] = {
                        "epoch": round(global_step / len(self.train_loader), 2) if self.train_loader else epoch + 1,
                        "step": global_step,
                        "train_loss": float(np.mean(train_losses)),
                        "train_reward": float(np.mean(train_rewards))
                    }
                    
                    if self.val_loader is not None:
                        # 1. Run normal eval_step
                        eval_metrics = []
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
                                    
                        # 2. Run portion of val set for Reward & CD evaluation
                        self.wrapper.eval()
                        val_dataset = self.val_loader.dataset
                        n_eval_samples = int(np.ceil(self.eval_fraction * len(val_dataset)))
                        val_indices = torch.randperm(len(val_dataset))[:n_eval_samples].tolist()
                        
                        val_cds = []
                        val_rewards = []
                        schema = self.wrapper.dual_seq_schema
                        
                        for idx in val_indices:
                            # val_dataset[idx] returns (X, y_cmds, y_args)
                            desc, gt_cmds, gt_args = val_dataset[idx]
                            # Generate greedy sequence
                            gen_seq = self.wrapper.generate(desc, max_new_tokens=self.max_new_cmds)
                            gen_cmds = [item[0] for item in gen_seq]
                            gen_args = [item[1] for item in gen_seq]
                            
                            cd = compute_cd(gen_cmds, gen_args, gt_cmds, gt_args)
                            val_cds.append(cd)
                            val_rewards.append(compute_reward(cd, self.min_cd, self.max_cd))
                            
                        summary["val_mean_cd"] = float(np.mean(val_cds)) if val_cds else 1.0
                        summary["val_mean_reward"] = float(np.mean(val_rewards)) if val_rewards else 0.0
                    
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
        
        if self.quant_type == "fp16" :
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.amp.autocast("cuda", dtype=dtype):
                return self.fit(epochs=epochs, verbose=verbose)
        
        return self.fit(epochs=epochs, verbose=verbose)
