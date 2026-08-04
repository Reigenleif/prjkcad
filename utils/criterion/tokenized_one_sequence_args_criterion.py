from __future__ import annotations

from typing import Any, Dict, Union
import torch
from torch import nn

from utils.dual_seq import get_dualseq_schema
from utils.criterion.base_criterion import BaseCriterion

class TokenizedOneSequenceArgsCriterion(BaseCriterion):
    def __init__(self, lambda_args: float = 1.0):
        super().__init__()
        # <-- Schema & Loss Init -->
        schema = get_dualseq_schema()
        self.cmd_pad_id = schema["cmd_pad_id"]
        self.cmd_eos_id = schema["cmd_eos_id"]
        self.arg_pad_id = schema["arg_pad_id"]
        self.arg_eos_id = schema["arg_eos_id"]
        self.cmd_loss_fn = nn.CrossEntropyLoss(ignore_index=self.cmd_pad_id)
        self.arg_loss_fn = nn.CrossEntropyLoss(ignore_index=self.arg_pad_id)
        self.lambda_args = lambda_args

    def forward(
        self,
        outputs: Union[Dict[str, Any], torch.Tensor],
        batch: Union[Dict[str, Any], torch.Tensor] = None,
        *args
    ) -> torch.Tensor:
        # <-- Input Unpacking Guard -->
        if isinstance(outputs, dict) and isinstance(batch, dict):
            cmd_logits = outputs["cmd_logits"]
            arg_logits = outputs["arg_logits"]
            cmd_targets = batch.get("cmd_targets", batch.get("y", {}).get("cmd_targets"))
            arg_targets = batch.get("arg_targets", batch.get("y", {}).get("arg_targets"))
        else:
            cmd_logits = outputs
            arg_logits = batch
            cmd_targets = args[0] if args else None
            arg_targets = args[1] if len(args) > 1 else None

        # <-- Command Loss Calculation -->
        B, T_cmd_pred, V_cmd = cmd_logits.shape
        T_cmd = min(T_cmd_pred, cmd_targets.size(1))
        cmd_logits_trim = cmd_logits[:, :T_cmd, :]
        cmd_targets_trim = cmd_targets[:, :T_cmd]
        cmd_loss = self.cmd_loss_fn(cmd_logits_trim.reshape(-1, V_cmd), cmd_targets_trim.reshape(-1))

        # <-- Argument Loss Calculation -->
        B, T_arg_pred, V_arg = arg_logits.shape
        T_arg = min(T_arg_pred, arg_targets.size(1))
        arg_logits_trim = arg_logits[:, :T_arg, :]
        arg_targets_trim = arg_targets[:, :T_arg]
        arg_loss = self.arg_loss_fn(arg_logits_trim.reshape(-1, V_arg), arg_targets_trim.reshape(-1))

        # <-- Combined Loss Return -->
        return cmd_loss + self.lambda_args * arg_loss
