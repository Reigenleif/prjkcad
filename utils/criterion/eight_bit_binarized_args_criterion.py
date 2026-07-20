from __future__ import annotations

import torch
from torch import nn
from utils.dual_seq import get_dualseq_schema
from utils.criterion.structural_validity_loss import StructuralValidityLoss


class EightBitBinarizedArgsCriterion(nn.Module):
    def __init__(self, lambda_args: float = 1.0, label_smoothing: float = 0.0,
                 eos_weight: float = 10.0, lambda_structural: float = 0.0):
        super().__init__()
        schema = get_dualseq_schema()
        self.cmd_pad_id = schema["cmd_pad_id"]
        self.cmd_eos_id = schema["cmd_eos_id"]
        self.arg_pad_id = 256
        self.lambda_args = lambda_args
        self.lambda_structural = lambda_structural

        cmd_class_weights = None
        if eos_weight > 0.0:
            cmd_weights = torch.ones(schema["cmd_n_tokens"])
            cmd_weights[self.cmd_eos_id] = eos_weight
            self.register_buffer("cmd_class_weights", cmd_weights)
            cmd_class_weights = self.cmd_class_weights

        self.cmd_loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.cmd_pad_id,
            weight=cmd_class_weights,
            label_smoothing=label_smoothing,
        )
        self.arg_loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.arg_pad_id,
            label_smoothing=label_smoothing,
        )

        if lambda_structural > 0.0:
            self.structural_loss = StructuralValidityLoss(schema)
        else:
            self.structural_loss = None

    def forward(self, cmd_logits, arg_logits, cmd_preds, arg_preds, cmd_targets, arg_targets):
        if hasattr(self, "cmd_class_weights"):
            device = cmd_logits.device
            if self.cmd_class_weights.device != device:
                self.cmd_class_weights = self.cmd_class_weights.to(device)
                self.cmd_loss_fn.weight = self.cmd_class_weights

        B, T_cmd_pred, V_cmd = cmd_logits.shape
        T_cmd_tgt = cmd_targets.size(1)
        T_cmd = min(T_cmd_pred, T_cmd_tgt)

        cmd_logits_trim = cmd_logits[:, :T_cmd, :]
        cmd_targets_trim = cmd_targets[:, :T_cmd]

        cmd_loss = self.cmd_loss_fn(
            cmd_logits_trim.reshape(-1, V_cmd),
            cmd_targets_trim.reshape(-1),
        )

        T_arg_pred = arg_logits.size(1)
        T_arg_tgt = arg_targets.size(1)
        T_arg = min(T_arg_pred, T_arg_tgt)

        arg_logits_trim = arg_logits[:, :T_arg, :, :]
        arg_targets_trim = arg_targets[:, :T_arg, :]

        arg_loss = self.arg_loss_fn(
            arg_logits_trim.reshape(-1, 257),
            arg_targets_trim.reshape(-1),
        )

        total_loss = cmd_loss + self.lambda_args * arg_loss

        if self.structural_loss is not None:
            total_loss = total_loss + self.lambda_structural * self.structural_loss(cmd_logits_trim)

        return total_loss
