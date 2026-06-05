from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


class Backend:
    def __init__(self):
        self._device = torch.device("cuda")

    def device(self):
        return self._device

    def move_to_device(self, batch: Any):
        return self._move(batch)

    def optimizer_step(self, optimizer):
        optimizer.step()

    def _move(self, batch: Any):
        if isinstance(batch, Mapping):
            return {key: self._move(value) for key, value in batch.items()}
        if isinstance(batch, tuple):
            return tuple(self._move(value) for value in batch)
        if isinstance(batch, list):
            return [self._move(value) for value in batch]
        if hasattr(batch, "to"):
            return batch.to(self.device())
        return batch


class CustomTrainer:
    def __init__(
        self,
        model,
        optimizer=None,
        criterion=None,
        backend=None,
        train_loader=None,
        val_loader=None,
        max_grad_norm: float = 1.0,
        eval_fn=None,
    ):
        self.backend = backend or Backend()
        self.device = self.backend.device()
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_grad_norm = max_grad_norm
        self.eval_fn = eval_fn

    def _forward(self, batch: Any):
        if hasattr(self.model, "training_step") and self.criterion is not None:
            return self.model.training_step(batch, self.criterion)

        if isinstance(batch, Mapping):
            signature = inspect.signature(self.model.forward)
            accepted = {name for name in signature.parameters if name != "self"}
            forwarded = {key: value for key, value in batch.items() if key in accepted}
            if forwarded:
                return self.model(**forwarded)
            return self.model(**batch)

        if isinstance(batch, Sequence):
            return self.model(*batch)

        return self.model(batch)

    def _get_batch_value(self, batch: Any, keys: tuple[str, ...]):
        if not isinstance(batch, Mapping):
            return None
        for key in keys:
            if key in batch:
                return batch[key]
        return None

    def _compute_loss(self, outputs: Any, batch: Any):
        if isinstance(outputs, Mapping):
            loss = outputs.get("loss")
            if loss is not None:
                return loss

            if self.criterion is not None:
                token_logits = outputs.get("token_logits", outputs.get("cmd_logits", outputs.get("logits")))
                float_preds = outputs.get("float_preds", outputs.get("param_preds"))
                token_targets = self._get_batch_value(batch, ("labels", "token_targets", "cmd_targets"))
                float_targets = self._get_batch_value(batch, ("float_targets", "param_targets"))

                if token_logits is not None and token_targets is not None and float_preds is not None and float_targets is not None:
                    return self.criterion(token_logits, float_preds, token_targets, float_targets)
                if token_logits is not None and token_targets is not None:
                    return self.criterion(token_logits, token_targets)

        if isinstance(outputs, (tuple, list)) and self.criterion is not None:
            token_targets = self._get_batch_value(batch, ("labels", "token_targets", "cmd_targets"))
            float_targets = self._get_batch_value(batch, ("float_targets", "param_targets"))

            if len(outputs) >= 2 and token_targets is not None and float_targets is not None:
                return self.criterion(outputs[0], outputs[1], token_targets, float_targets)
            if len(outputs) >= 1 and token_targets is not None:
                return self.criterion(outputs[0], token_targets)

        if torch.is_tensor(outputs) and outputs.ndim == 0:
            return outputs

        raise ValueError("Unable to derive a loss from model outputs")

    def single_train_step(self, batch: Any):
        if self.optimizer is None:
            raise ValueError("optimizer is required for training")

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        batch = self.backend.move_to_device(batch)
        outputs = self._forward(batch)
        loss = self._compute_loss(outputs, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.backend.optimizer_step(self.optimizer)
        return loss.detach()

    def single_eval_step(self, batch: Any):
        batch = self.backend.move_to_device(batch)
        if self.eval_fn is not None:
            return self.eval_fn(self.model, batch)

        with torch.no_grad():
            outputs = self._forward(batch)
            loss = self._compute_loss(outputs, batch)

        return {"loss": float(loss.detach().cpu().item())}

    def train(self, epochs: int, verbose: bool = True):
        history: list[dict[str, float]] = []
        train_loader = self.train_loader or []

        for epoch in range(epochs):
            total_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                total_loss += float(self.single_train_step(batch).item())

            avg_loss = total_loss / max(len(train_loader), 1)
            summary: dict[str, float] = {"loss": avg_loss}

            if self.eval_fn is not None and self.val_loader is not None:
                self.model.eval()
                eval_results = []
                with torch.no_grad():
                    for batch in tqdm(self.val_loader, desc=f"Eval Epoch {epoch + 1}/{epochs}"):
                        eval_results.append(self.single_eval_step(batch))
                if eval_results:
                    for key in eval_results[0].keys():
                        summary[key] = float(np.mean([item[key] for item in eval_results]))

            history.append(summary)
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
                if self.eval_fn is not None and self.val_loader is not None:
                    print(f"Eval Metrics: {summary}")

        return history
