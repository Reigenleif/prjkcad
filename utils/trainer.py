from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable

import numpy as np
import torch
from tqdm import tqdm


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
        model,
        criterion,
        optimizer,
        train_loader=None,
        val_loader=None,
        device: torch.device | None = None,
        max_grad_norm: float = 1.0,
        side_teacher_forcing_ratio: float = 1.0,
        side_teacher_forcing_decay: float = 1.0,
        min_side_teacher_forcing_ratio: float = 0.0,
        schedule_fn: Callable[[int], float] | None = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_grad_norm = max_grad_norm
        self.side_teacher_forcing_ratio = side_teacher_forcing_ratio
        self.side_teacher_forcing_decay = side_teacher_forcing_decay
        self.min_side_teacher_forcing_ratio = min_side_teacher_forcing_ratio
        self.schedule_fn = schedule_fn

    def _scheduled_ratio(self, epoch: int) -> float:
        if self.schedule_fn is not None:
            return float(self.schedule_fn(epoch))
        ratio = self.side_teacher_forcing_ratio * (self.side_teacher_forcing_decay ** epoch)
        return float(max(self.min_side_teacher_forcing_ratio, min(1.0, ratio)))

    def _build_side_inputs(self, batch: Mapping[str, Any], ratio: float):
        decoder_input_ids = batch["decoder_input_ids"]
        decoder_input_args = batch["decoder_input_args"]
        decoder_attention_mask = batch["decoder_attention_mask"]

        if ratio >= 1.0:
            return decoder_input_ids, decoder_input_args

        with torch.no_grad():
            warm_cmd_logits, warm_arg_preds = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                decoder_input_ids=decoder_input_ids,
                decoder_input_args=decoder_input_args,
                decoder_attention_mask=decoder_attention_mask,
            )

        predicted_cmds = warm_cmd_logits.argmax(dim=-1)
        keep_mask = torch.rand(decoder_input_ids.shape, device=decoder_input_ids.device) < ratio
        keep_mask = keep_mask | ~decoder_attention_mask.bool()

        side_input_ids = torch.where(keep_mask, decoder_input_ids, predicted_cmds)
        side_input_args = torch.where(keep_mask.unsqueeze(-1), decoder_input_args, warm_arg_preds.detach())
        return side_input_ids, side_input_args

    def _forward(self, batch: Mapping[str, Any], ratio: float):
        side_input_ids, side_input_args = self._build_side_inputs(batch, ratio)
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            decoder_input_args=batch["decoder_input_args"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            side_input_ids=side_input_ids,
            side_input_args=side_input_args,
        )

    def _loss(self, outputs, batch):
        cmd_logits, arg_preds = outputs
        return self.criterion(
            cmd_logits,
            arg_preds,
            batch["cmd_targets"],
            batch["arg_targets"],
            batch.get("arg_masks"),
            batch.get("decoder_attention_mask"),
        )

    def train_step(self, batch: Mapping[str, Any], ratio: float):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        batch = _move_to_device(batch, self.device)
        outputs = self._forward(batch, ratio)
        loss_dict = self._loss(outputs, batch)
        loss_dict["loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return {key: float(value.detach().cpu().item()) for key, value in loss_dict.items()}

    @torch.no_grad()
    def eval_step(self, batch: Mapping[str, Any]):
        self.model.eval()
        batch = _move_to_device(batch, self.device)
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            decoder_input_args=batch["decoder_input_args"],
            decoder_attention_mask=batch["decoder_attention_mask"],
        )
        loss_dict = self._loss(outputs, batch)
        return {key: float(value.detach().cpu().item()) for key, value in loss_dict.items()}

    def fit(self, epochs: int, verbose: bool = True):
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
