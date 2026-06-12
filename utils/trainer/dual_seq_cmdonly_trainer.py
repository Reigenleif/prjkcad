from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, Tuple

import numpy as np
import torch
from tqdm import tqdm

from utils.wrapper.dual_seq_cmdonly import DualSeqCMDOnlyWrapper


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


class DualSeqCMDOnlyTrainer:
    def __init__(
        self,
        model_wrapper: DualSeqCMDOnlyWrapper,
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
        if quant_type == "fp16":
            self.wrapper.model.half()
        self.quant_type = quant_type

    def _scheduled_ratio(self, epoch: int) -> float:
        if self.schedule_fn is not None:
            return float(self.schedule_fn(epoch))
        ratio = self.teacher_forcing_ratio * (self.teacher_forcing_decay ** epoch)
        return float(max(self.min_teacher_forcing_ratio, min(1.0, ratio)))

    def _forward(self, batch: Mapping[str, Any], ratio: float):
        is_teacher_forcing = np.random.rand() < ratio
        return self.wrapper.forward(batch, is_teacher_forcing=is_teacher_forcing)

    def _loss(self, outputs, batch):
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        return self.criterion(logits, batch[1])

    def train_step(self, batch: Tuple, ratio: float):
        
        self.wrapper.train()
        self.optimizer.zero_grad(set_to_none=True)
        
        batch = _move_to_device(batch, self.device)
        outputs, _ = self._forward(batch, ratio)
        
        # pad targets to match output length for loss computation
        if outputs.shape[1] > batch[1].shape[1]:
            pad_length = outputs.shape[1] - batch[1].shape[1]
            pad_tensor = torch.full((batch[1].shape[0], pad_length), self.wrapper.model.pad_id, device=batch[1].device, dtype=batch[1].dtype)
            batch = (batch[0], torch.cat([batch[1], pad_tensor], dim=1), batch[2])
        
        loss = self._loss(outputs, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.wrapper.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return {"loss": float(loss.detach().cpu().item())}

    @torch.no_grad()
    def eval_step(self, batch: Mapping[str, Any]):
        metrics = {}
        
        # loss and inference
        self.wrapper.eval()
        batch = _move_to_device(batch, self.device)
        outputs, preds = self._forward(batch, ratio=1.0)
        loss = self._loss(outputs, batch)
        metrics = {"loss": float(loss.detach().cpu().item())}
        
        # perplexity
        loss_value = metrics.get("loss")
        if loss_value is not None:
            metrics["perplexity"] = float(np.exp(loss_value))

        # Token accuracy
        targets = batch[1]
        T = min(preds.size(1), targets.size(1))
        preds = preds[:, :T]
        targets = targets[:, :T]
        correct = preds.eq(targets)
        accuracy = correct.float().sum() / correct.numel()
        metrics["accuracy"] = float(accuracy.detach().cpu().item())
        return metrics

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

        return history

    def train(self, epochs: int, verbose: bool = True):
        return self.fit(epochs=epochs, verbose=verbose)
