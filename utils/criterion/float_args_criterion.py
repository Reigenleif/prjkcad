from __future__ import annotations

from typing import Any, Dict, Union
import torch
from torch import nn

from utils.dual_seq import get_dualseq_schema
from utils.criterion.base_criterion import BaseCriterion

class FloatArgsCriterion(BaseCriterion):
    def __init__(self, lambda_args: float = 1.0, *args, **kwargs):
        super().__init__()
        # <-- Initialization -->
        schema = get_dualseq_schema()
        self.cmd_pad_id = schema["cmd_pad_id"]
        self.cmd_eos_id = schema["cmd_eos_id"]
        self.lambda_args = lambda_args
        self.cmd_loss_fn = nn.CrossEntropyLoss(ignore_index=self.cmd_pad_id)
        self.arg_loss_fn = nn.MSELoss()

    def forward(
        self,
        outputs: Union[Dict[str, Any], torch.Tensor],
        batch: Union[Dict[str, Any], torch.Tensor] = None,
        cmd_targets: torch.Tensor = None,
        arg_targets: torch.Tensor = None
    ) -> torch.Tensor:
        # <-- Argument Extraction -->
        if isinstance(outputs, dict) and isinstance(batch, dict):
            cmd_logits = outputs["cmd_logits"]
            arg_preds = outputs["arg_preds"]
            cmd_targets = batch.get("cmd_targets", batch.get("y", {}).get("cmd_targets"))
            arg_targets = batch.get("arg_targets", batch.get("y", {}).get("arg_targets"))
        elif not isinstance(outputs, dict):
            cmd_logits = outputs
            arg_preds = batch

        # <-- Command Loss Calculation -->
        B, T_cmd_pred, V_cmd = cmd_logits.shape
        T_cmd_tgt = cmd_targets.size(1)
        T_cmd = min(T_cmd_pred, T_cmd_tgt)
        cmd_logits_trim = cmd_logits[:, :T_cmd, :]
        cmd_targets_trim = cmd_targets[:, :T_cmd]
        cmd_loss = self.cmd_loss_fn(cmd_logits_trim.reshape(-1, V_cmd), cmd_targets_trim.reshape(-1))

        # <-- Argument Loss Calculation -->
        if arg_targets.ndim == 2:
            arg_targets = arg_targets.unsqueeze(-1)
        if arg_preds.ndim == 2:
            arg_preds = arg_preds.unsqueeze(-1)
        n_args = arg_targets.size(-1)
        T_arg = min(arg_preds.size(1), arg_targets.size(1))
        arg_preds_trim = arg_preds[:, :T_arg, :n_args]
        arg_targets_trim = arg_targets[:, :T_arg, :]
        arg_loss = self.arg_loss_fn(arg_preds_trim, arg_targets_trim)

        # <-- Total Loss Return -->
        return cmd_loss + self.lambda_args * arg_loss
