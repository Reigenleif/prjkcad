from __future__ import annotations

from typing import Any, Dict, Tuple, Union
import numpy as np
import torch
from utils.trainer.base_trainer import BaseTrainer
from utils.evaluate import eval_batch

class GDTrainer(BaseTrainer):
    """Data-type agnostic Gradient Descent trainer for Fine-Tuning and Pretraining."""

    def _scheduled_ratio(self) -> float:
        wrapper = getattr(self.wrapper, "wrapper", self.wrapper)
        tf_ratio = getattr(wrapper, "teacher_forcing_ratio", 1.0)
        tf_decay = getattr(wrapper, "teacher_forcing_decay", 1.0)
        min_tf = getattr(wrapper, "min_teacher_forcing_ratio", 0.0)
        ratio = tf_ratio * (tf_decay ** self.current_epoch)
        return float(max(min_tf, min(1.0, ratio)))

    def training_step(self, batch: Union[Dict[str, Any], Tuple], batch_idx: int) -> torch.Tensor:
        # <-- Forward & Loss Compute -->
        ratio = self._scheduled_ratio()
        is_tf = bool(np.random.rand() < ratio)
        outputs = self.wrapper(batch, is_teacher_forcing=is_tf)
        loss = self.criterion(outputs, batch)

        # <-- Metrics Logging -->
        lr = self.optimizers().param_groups[0]["lr"] if self.optimizers() else 0.0
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("lr", lr, on_step=True, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch: Union[Dict[str, Any], Tuple], batch_idx: int) -> torch.Tensor:
        # <-- Validation Forward & Loss Compute -->
        outputs = self.wrapper(batch, is_teacher_forcing=True)
        loss = self.criterion(outputs, batch)

        # <-- Validation Metrics Logging -->
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_perplexity", torch.exp(loss.detach()), on_step=False, on_epoch=True, prog_bar=False, logger=True)

        # <-- Non-Loss Metrics Evaluation -->
        try:
            with torch.no_grad():
                gen_outputs = self.wrapper(batch, is_teacher_forcing=False)
                cmd_targets = batch[1] if isinstance(batch, (tuple, list)) and len(batch) > 1 else (batch.get("cmd_targets") if isinstance(batch, dict) else None)
                arg_targets = batch[2] if isinstance(batch, (tuple, list)) and len(batch) > 2 else (batch.get("arg_targets") if isinstance(batch, dict) else None)

                if isinstance(gen_outputs, dict):
                    cmd_preds = gen_outputs.get("cmd_preds")
                    arg_preds = gen_outputs.get("arg_preds")
                elif isinstance(gen_outputs, (tuple, list)):
                    cmd_preds = gen_outputs[2] if len(gen_outputs) > 2 else gen_outputs[0]
                    arg_preds = gen_outputs[3] if len(gen_outputs) > 3 else gen_outputs[1]
                else:
                    cmd_preds, arg_preds = None, None

                if cmd_preds is not None and cmd_targets is not None:
                    out_type = getattr(self.wrapper, "out_type", None) or getattr(getattr(self.wrapper, "wrapper", None), "out_type", "FloatArgs")
                    metadata = getattr(self.wrapper, "metadata", None) or getattr(getattr(self.wrapper, "wrapper", None), "metadata", None)
                    eval_metrics = eval_batch(cmd_preds, cmd_targets, arg_preds, arg_targets, out_type=out_type, metadata=metadata)
                    for k, v in eval_metrics.items():
                        self.log(f"val_{k}", v, on_step=False, on_epoch=True, prog_bar=(k in ["avg_f1", "arg_float_mse", "arg_float_r2"]), logger=True)
        except Exception:
            pass

        return loss

