from __future__ import annotations

from typing import Any, Dict, Tuple, Union
import numpy as np
import torch

from utils.trainer.base_lightning_module import BaseLightningModule
from utils.evaluate import eval_batch

class GDLightningModule(BaseLightningModule):
    """Gradient descent module for fine-tuning and pretraining."""

    def _scheduled_ratio(self) -> float:
        wrapper = getattr(self.wrapper, "wrapper", self.wrapper)
        tf_ratio = getattr(wrapper, "teacher_forcing_ratio", 1.0)
        tf_decay = getattr(wrapper, "teacher_forcing_decay", 1.0)
        min_tf = getattr(wrapper, "min_teacher_forcing_ratio", 0.0)
        ratio = tf_ratio * (tf_decay ** self.current_epoch)
        return float(max(min_tf, min(1.0, ratio)))

    def forward(self, batch: Any, is_teacher_forcing: bool = False, **kwargs) -> Any:
        if self.wrapper is None:
            raise NotImplementedError("Module wrapper is not set.")
        return self.wrapper(batch, is_teacher_forcing=is_teacher_forcing, **kwargs)

    def training_step(self, batch: Union[Dict[str, Any], Tuple], batch_idx: int) -> torch.Tensor:
        ratio = self._scheduled_ratio()
        is_tf = bool(np.random.rand() < ratio)
        outputs = self(batch, is_teacher_forcing=is_tf)
        loss = self.criterion(outputs, batch)
        if isinstance(loss, tuple):
            loss = loss[0]

        try:
            opts = self.optimizers()
            lr = opts.param_groups[0]["lr"] if opts and hasattr(opts, "param_groups") else 0.0
        except Exception:
            lr = 0.0
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/lr", lr, on_step=True, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch: Union[Dict[str, Any], Tuple], batch_idx: int) -> torch.Tensor:
        outputs = self(batch, is_teacher_forcing=True)
        loss = self.criterion(outputs, batch)
        if isinstance(loss, tuple):
            loss = loss[0]

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/perplexity", torch.exp(loss.detach()), on_step=False, on_epoch=True, prog_bar=False, logger=True)

        try:
            with torch.no_grad():
                gen_outputs = self(batch, is_teacher_forcing=False)
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
                        self.log(k, v, on_step=False, on_epoch=True, prog_bar=(k in ["eval/avg_f1", "eval/arg_float_mse", "eval/arg_float_r2"]), logger=True)
        except Exception as e:
            print(f"[GDLightningModule] Eval error (suppressed): {type(e).__name__}: {e}")

        return loss
