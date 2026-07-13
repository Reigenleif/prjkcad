from __future__ import annotations

import torch
from torch import nn

from utils.dual_seq import get_dualseq_schema

class DualSeqCriterion(nn.Module):
    def __init__(
        self,
        lambda_args: float = 1.0,
    ):
        super().__init__()
        schema = get_dualseq_schema()
        
        self.cmd_pad_id = schema["cmd_pad_id"]
        self.cmd_eos_id = schema["cmd_eos_id"]
        
        self.arg_pad_id = schema["arg_pad_id"]
        self.arg_eos_id = schema["arg_eos_id"]
        
        cmd_weights = torch.ones(schema["cmd_n_tokens"])
        cmd_weights[self.cmd_eos_id] = 10.0 
        self.register_buffer("cmd_class_weights", cmd_weights)
        
        arg_weights = torch.ones(schema["args_n_tokens"])
        arg_weights[self.arg_eos_id] = 10.0
        self.register_buffer("arg_class_weights", arg_weights)

        self.cmd_loss_fn = nn.CrossEntropyLoss(ignore_index=self.cmd_pad_id, weight=self.cmd_class_weights)
        self.arg_loss_fn = nn.CrossEntropyLoss(ignore_index=self.arg_pad_id, weight=self.arg_class_weights)

        self.lambda_args = lambda_args

    def forward(self, cmd_logits, cmd_targets, arg_logits, arg_targets):
        device = cmd_logits.device
        if self.cmd_class_weights.device != device:
            self.cmd_class_weights = self.cmd_class_weights.to(device)
            self.cmd_loss_fn.weight = self.cmd_class_weights
            
        if self.arg_class_weights.device != device:
            self.arg_class_weights = self.arg_class_weights.to(device)
            self.arg_loss_fn.weight = self.arg_class_weights

        B, T_cmd_pred, V_cmd = cmd_logits.shape
        T_cmd_tgt = cmd_targets.size(1)

        T_cmd = min(T_cmd_pred, T_cmd_tgt)

        cmd_logits_trim = cmd_logits[:, :T_cmd, :]
        cmd_targets_trim = cmd_targets[:, :T_cmd]

        cmd_loss = self.cmd_loss_fn(
            cmd_logits_trim.reshape(-1, V_cmd),
            cmd_targets_trim.reshape(-1),
        )

        B, T_arg_pred, V_arg = arg_logits.shape
        T_arg_tgt = arg_targets.size(1)
        
        T_arg = min(T_arg_pred, T_arg_tgt)
        
        arg_logits_trim = arg_logits[:, :T_arg, :]
        arg_targets_trim = arg_targets[:, :T_arg]
        
        arg_loss = self.arg_loss_fn(
            arg_logits_trim.reshape(-1, V_arg),
            arg_targets_trim.reshape(-1),
        )

        total_loss = cmd_loss + self.lambda_args * arg_loss

        return total_loss
