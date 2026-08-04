from __future__ import annotations

from typing import Tuple
import numpy as np
import torch
from sklearn.metrics import r2_score
import wandb

from utils.trainer.components import BaseTrainer, _move_to_device
from utils.evaluate import eval_cmd_only, eval_cmd_and_args
from utils.dual_seq import get_dualseq_schema

def eval_float_args(pred_cmds, gt_cmds, pred_args, gt_args):
    metrics = eval_cmd_only(pred_cmds, gt_cmds)
    pred_flat = []
    gt_flat = []
    for p_seq, g_seq in zip(pred_args, gt_args):
        pred_flat.extend(p_seq)
        gt_flat.extend(g_seq)
    if gt_flat:
        p_arr = np.array(pred_flat)
        g_arr = np.array(gt_flat)
        metrics["arg_float_mse"] = float(np.mean((p_arr - g_arr) ** 2))
        try:
            metrics["arg_float_r2"] = float(r2_score(g_arr, p_arr))
        except Exception:
            metrics["arg_float_r2"] = 0.0
    else:
        metrics["arg_float_mse"] = 0.0
        metrics["arg_float_r2"] = 1.0
    return metrics

class FloatArgsTrainer(BaseTrainer):
    def _scheduled_ratio(self, epoch: int) -> float:
        tf_ratio = getattr(self.wrapper, "teacher_forcing_ratio", 1.0)
        tf_decay = getattr(self.wrapper, "teacher_forcing_decay", 1.0)
        min_tf = getattr(self.wrapper, "min_teacher_forcing_ratio", 0.0)
        ratio = tf_ratio * (tf_decay ** epoch)
        return float(max(min_tf, min(1.0, ratio)))

    def train_step(self, batch: Tuple, epoch: int, use_amp: bool = False, dtype = None, scaler = None) -> dict[str, float]:
        self.wrapper.train()
        self.optimizer.zero_grad(set_to_none=True)
        batch = _move_to_device(batch, self.device)
        ratio = self._scheduled_ratio(epoch)
        is_teacher_forcing = np.random.rand() < ratio
        
        if use_amp:
            with torch.amp.autocast("cuda", dtype=dtype):
                outputs = self.wrapper(batch, is_teacher_forcing=is_teacher_forcing)
                loss = self.criterion(outputs[0], outputs[1], batch[1], batch[2])
        else:
            outputs = self.wrapper(batch, is_teacher_forcing=is_teacher_forcing)
            loss = self.criterion(outputs[0], outputs[1], batch[1], batch[2])

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.wrapper.parameters(), self.max_grad_norm)
            scaler.step(self.optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.wrapper.parameters(), self.max_grad_norm)
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return {
            "loss": float(loss.detach().cpu().item()),
            "grad_norm": float(grad_norm),
            "lr": float(self.optimizer.param_groups[0]["lr"])
        }

    @torch.no_grad()
    def eval_step(self, batch: Tuple) -> dict[str, float]:
        self.wrapper.eval()
        batch = _move_to_device(batch, self.device)
        outputs = self.wrapper(batch, is_teacher_forcing=True)
        loss = self.criterion(outputs[0], outputs[1], batch[1], batch[2])
        
        gen_outputs = self.wrapper(batch, is_teacher_forcing=False)
        if len(gen_outputs) == 3:
            cmd_logits, arg_preds, cmd_preds = gen_outputs
        else:
            cmd_logits, _, cmd_preds, arg_preds = gen_outputs
        
        schema = get_dualseq_schema()
        id_to_cmd = schema["id_to_command"]
        
        metrics = {
            "loss": float(loss.detach().cpu().item()),
            "perplexity": float(torch.exp(loss.detach().cpu()).item())
        }
        
        cmd_targets = batch[1]
        arg_targets = batch[2]
        
        cmd_arg_metric_list = []
        for i in range(cmd_targets.shape[0]):
            pred_cmds = [id_to_cmd.get(id, '<UNK>') for id in cmd_preds[i].cpu().numpy().tolist()]
            true_cmds = [id_to_cmd.get(id, '<UNK>') for id in cmd_targets[i].cpu().numpy().tolist()]
            pred_args = arg_preds[i].float().cpu().numpy().tolist()
            true_args = arg_targets[i].cpu().numpy().tolist()
            
            try: pred_cmds = pred_cmds[:pred_cmds.index("EOS")]
            except ValueError: pass
            try: true_cmds = true_cmds[:true_cmds.index("EOS")]
            except ValueError: pass
            
            cmd_arg_metric_list.append(eval_float_args(pred_cmds, true_cmds, pred_args, true_args))
            
        if cmd_arg_metric_list:
            for key in cmd_arg_metric_list[0].keys():
                vals = [m[key] for m in cmd_arg_metric_list if m.get(key) is not None]
                if vals:
                    metrics[key] = float(np.mean(vals))
        return metrics

class TokenizedOneSequenceArgsTrainer(BaseTrainer):
    def _scheduled_ratio(self, epoch: int) -> float:
        tf_ratio = getattr(self.wrapper, "teacher_forcing_ratio", 1.0)
        tf_decay = getattr(self.wrapper, "teacher_forcing_decay", 1.0)
        min_tf = getattr(self.wrapper, "min_teacher_forcing_ratio", 0.0)
        ratio = tf_ratio * (tf_decay ** epoch)
        return float(max(min_tf, min(1.0, ratio)))

    def train_step(self, batch: Tuple, epoch: int, use_amp: bool = False, dtype = None, scaler = None) -> dict[str, float]:
        self.wrapper.train()
        self.optimizer.zero_grad(set_to_none=True)
        batch = _move_to_device(batch, self.device)
        ratio = self._scheduled_ratio(epoch)
        is_teacher_forcing = np.random.rand() < ratio
        
        if use_amp:
            with torch.amp.autocast("cuda", dtype=dtype):
                outputs = self.wrapper(batch, is_teacher_forcing=is_teacher_forcing)
                loss = self.criterion(*outputs, batch[1], batch[2])
        else:
            outputs = self.wrapper(batch, is_teacher_forcing=is_teacher_forcing)
            loss = self.criterion(*outputs, batch[1], batch[2])

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.wrapper.parameters(), self.max_grad_norm)
            scaler.step(self.optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.wrapper.parameters(), self.max_grad_norm)
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return {
            "loss": float(loss.detach().cpu().item()),
            "grad_norm": float(grad_norm),
            "lr": float(self.optimizer.param_groups[0]["lr"])
        }

    @torch.no_grad()
    def eval_step(self, batch: Tuple) -> dict[str, float]:
        self.wrapper.eval()
        batch = _move_to_device(batch, self.device)
        outputs = self.wrapper(batch, is_teacher_forcing=True)
        loss = self.criterion(outputs[0], outputs[1], batch[1], batch[2])
        
        gen_outputs = self.wrapper(batch, is_teacher_forcing=False)
        cmd_logits, arg_logits, cmd_preds, arg_preds = gen_outputs[:4]
        
        schema = get_dualseq_schema()
        id_to_cmd = schema["id_to_command"]
        
        metrics = {
            "loss": float(loss.detach().cpu().item()),
            "perplexity": float(torch.exp(loss.detach().cpu()).item())
        }
        
        cmd_targets = batch[1]
        arg_targets = batch[2]
        
        cmd_arg_metric_list = []
        for i in range(cmd_targets.shape[0]):
            pred_cmds = [id_to_cmd.get(id, '<UNK>') for id in cmd_preds[i].cpu().numpy().tolist()]
            true_cmds = [id_to_cmd.get(id, '<UNK>') for id in cmd_targets[i].cpu().numpy().tolist()]
            pred_args = arg_preds[i].cpu().numpy().tolist()
            true_args = arg_targets[i].cpu().numpy().tolist()
            
            try: pred_cmds = pred_cmds[:pred_cmds.index("EOS")]
            except ValueError: pass
            try: true_cmds = true_cmds[:true_cmds.index("EOS")]
            except ValueError: pass
            
            metric = eval_cmd_and_args(pred_cmds, true_cmds, pred_args, true_args, schema, skip_rendering=True)
            correct = sum(1 for j in range(min(len(pred_args), len(true_args))) if pred_args[j] == true_args[j])
            metric["arg_token_accuracy"] = correct / max(len(true_args), 1)
            cmd_arg_metric_list.append(metric)
            
        if cmd_arg_metric_list:
            for key in cmd_arg_metric_list[0].keys():
                vals = [m[key] for m in cmd_arg_metric_list if m.get(key) is not None]
                if vals:
                    metrics[key] = float(np.mean(vals))
        return metrics

class EightBitBinarizedArgsTrainer(BaseTrainer):
    def __init__(self, *args, metadata=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata = metadata

    def _scheduled_ratio(self, epoch: int) -> float:
        tf_ratio = getattr(self.wrapper, "teacher_forcing_ratio", 1.0)
        tf_decay = getattr(self.wrapper, "teacher_forcing_decay", 1.0)
        min_tf = getattr(self.wrapper, "min_teacher_forcing_ratio", 0.0)
        ratio = tf_ratio * (tf_decay ** epoch)
        return float(max(min_tf, min(1.0, ratio)))

    def train_step(self, batch: Tuple, epoch: int, use_amp: bool = False, dtype = None, scaler = None) -> dict[str, float]:
        self.wrapper.train()
        self.optimizer.zero_grad(set_to_none=True)
        batch = _move_to_device(batch, self.device)
        ratio = self._scheduled_ratio(epoch)
        is_teacher_forcing = np.random.rand() < ratio
        
        if use_amp:
            with torch.amp.autocast("cuda", dtype=dtype):
                outputs = self.wrapper(batch, is_teacher_forcing=is_teacher_forcing)
                loss = self.criterion(*outputs, batch[1], batch[2])
        else:
            outputs = self.wrapper(batch, is_teacher_forcing=is_teacher_forcing)
            loss = self.criterion(*outputs, batch[1], batch[2])

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.wrapper.parameters(), self.max_grad_norm)
            scaler.step(self.optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.wrapper.parameters(), self.max_grad_norm)
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return {
            "loss": float(loss.detach().cpu().item()),
            "grad_norm": float(grad_norm),
            "lr": float(self.optimizer.param_groups[0]["lr"])
        }

    @torch.no_grad()
    def eval_step(self, batch: Tuple) -> dict[str, float]:
        self.wrapper.eval()
        batch = _move_to_device(batch, self.device)
        outputs = self.wrapper(batch, is_teacher_forcing=True)
        loss = self.criterion(*outputs, batch[1], batch[2])
        
        gen_outputs = self.wrapper(batch, is_teacher_forcing=False)
        cmd_logits, arg_logits, cmd_preds, arg_preds = gen_outputs[:4]
        
        schema = get_dualseq_schema()
        id_to_cmd = schema["id_to_command"]
        
        metrics = {
            "loss": float(loss.detach().cpu().item()),
            "perplexity": float(torch.exp(loss.detach().cpu()).item())
        }
        
        cmd_targets = batch[1]
        arg_targets = batch[2]
        
        cmd_arg_metric_list = []
        for i in range(cmd_targets.shape[0]):
            pred_cmds = [id_to_cmd.get(id, '<UNK>') for id in cmd_preds[i].cpu().numpy().tolist()]
            true_cmds = [id_to_cmd.get(id, '<UNK>') for id in cmd_targets[i].cpu().numpy().tolist()]
            pred_args = arg_preds[i].cpu().numpy().tolist()
            true_args = arg_targets[i].cpu().numpy().tolist()
            
            try: pred_cmds = pred_cmds[:pred_cmds.index("EOS")]
            except ValueError: pass
            try: true_cmds = true_cmds[:true_cmds.index("EOS")]
            except ValueError: pass
            
            metric = eval_cmd_only(pred_cmds, true_cmds)
            
            correct = 0
            total = 0
            pred_floats = []
            true_floats = []
            arg_names = schema["arg_names"]
            for step_idx in range(min(len(pred_args), len(true_args))):
                for arg_idx, arg_name in enumerate(arg_names):
                    p_bin = pred_args[step_idx][arg_idx]
                    t_bin = true_args[step_idx][arg_idx]
                    if t_bin != 256:
                        total += 1
                        if p_bin == t_bin:
                            correct += 1
                        if self.metadata is not None:
                            pred_floats.append(self.metadata.bin_to_float(arg_name, p_bin))
                            true_floats.append(self.metadata.bin_to_float(arg_name, t_bin))
            
            metric["arg_token_accuracy"] = correct / max(total, 1)
            
            if true_floats:
                p_arr = np.array(pred_floats)
                t_arr = np.array(true_floats)
                metric["arg_float_mse"] = float(np.mean((p_arr - t_arr) ** 2))
                try:
                    metric["arg_float_r2"] = float(r2_score(t_arr, p_arr))
                except Exception:
                    metric["arg_float_r2"] = 0.0
            else:
                metric["arg_float_mse"] = 0.0
                metric["arg_float_r2"] = 1.0
            cmd_arg_metric_list.append(metric)
            
        if cmd_arg_metric_list:
            for key in cmd_arg_metric_list[0].keys():
                vals = [m[key] for m in cmd_arg_metric_list if m.get(key) is not None]
                if vals:
                    metrics[key] = float(np.mean(vals))
        return metrics
