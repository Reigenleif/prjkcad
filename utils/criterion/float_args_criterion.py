from __future__ import annotations

import torch
from torch import nn
from utils.dual_seq import get_dualseq_schema

class FloatArgsCriterion(nn.Module):
    def __init__(self, lambda_args: float = 1.0):
        super().__init__()
        schema = get_dualseq_schema()
        self.cmd_pad_id = schema["cmd_pad_id"]
        self.cmd_eos_id = schema["cmd_eos_id"]
        
        # UNCOMMENT THIS IF YOU WANT TO USE CLASS WEIGHTS
        # cmd_weights = torch.ones(schema["cmd_n_tokens"])
        # cmd_weights[self.cmd_eos_id] = 10.0 
        # self.register_buffer("cmd_class_weights", cmd_weights)
        self.lambda_args = lambda_args

        # UNCOMMENT THIS IF YOU WANT TO USE CLASS WEIGHTS
        # self.cmd_loss_fn = nn.CrossEntropyLoss(ignore_index=self.cmd_pad_id, weight=self.cmd_class_weights)
        self.cmd_loss_fn = nn.CrossEntropyLoss(ignore_index=self.cmd_pad_id)
        self.arg_loss_fn = nn.MSELoss()

    def forward(self, cmd_logits, arg_preds, cmd_targets, arg_targets):
        # UNCOMMENT THIS IF YOU WANT TO USE CLASS WEIGHTS
        # device = cmd_logits.device
        # if self.cmd_class_weights.device != device:
        #     self.cmd_class_weights = self.cmd_class_weights.to(device)
        #     self.cmd_loss_fn.weight = self.cmd_class_weights

        B, T_cmd_pred, V_cmd = cmd_logits.shape
        T_cmd_tgt = cmd_targets.size(1)
        T_cmd = min(T_cmd_pred, T_cmd_tgt)

        cmd_logits_trim = cmd_logits[:, :T_cmd, :]
        cmd_targets_trim = cmd_targets[:, :T_cmd]

        cmd_loss = self.cmd_loss_fn(
            cmd_logits_trim.reshape(-1, V_cmd),
            cmd_targets_trim.reshape(-1),
        )

        T_arg = min(arg_preds.size(1), arg_targets.size(1))
        arg_preds_trim = arg_preds[:, :T_arg, :]
        arg_targets_trim = arg_targets[:, :T_arg, :]

        arg_loss = self.arg_loss_fn(
            arg_preds_trim,
            arg_targets_trim
        )

        return cmd_loss + self.lambda_args * arg_loss
