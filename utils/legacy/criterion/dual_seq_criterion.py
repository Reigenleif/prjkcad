from __future__ import annotations

import torch
from torch import nn

from utils.dual_seq import get_dualseq_schema


class DualSeqCriterion(nn.Module):
    def __init__(
        self,
        pad_id: int | None = None,
        lambda_after_pad: float = 3.0,
        lambda_overgen: float = 3.0,
        lambda_args: float = 0.1,
    ):
        super().__init__()
        schema = get_dualseq_schema()
        self.pad_id = schema["pad_id"] if pad_id is None else pad_id
        self.eos_id = schema["eos_id"]
        self.n_args = schema["n_args"]
        self.n_tokens = schema["n_tokens"]
        
        # Increase EOS penalty using class weights
        weights = torch.ones(self.n_tokens)
        weights[self.eos_id] = 10.0 # Heavier penalty for missing/mispredicting EOS
        self.register_buffer("class_weights", weights)

        self.cmd_loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_id, weight=self.class_weights)
        self.arg_loss_fn = nn.MSELoss()

        self.lambda_args = lambda_args

    def forward(self, cmd_logits, cmd_targets, arg_preds, arg_targets):
        device = cmd_logits.device
        if self.class_weights.device != device:
            self.class_weights = self.class_weights.to(device)
            self.cmd_loss_fn.weight = self.class_weights

        B, T_pred, V = cmd_logits.shape
        T_tgt = cmd_targets.size(1)

        T = min(T_pred, T_tgt)

        # CMD loss 
        logits_trim = cmd_logits[:, :T, :]
        targets_trim = cmd_targets[:, :T]

        cmd_loss = self.cmd_loss_fn(
            logits_trim.reshape(-1, V),
            targets_trim.reshape(-1),
        )

        # ARG loss (MSE, only over the overlapping time steps) 
        T_arg = min(arg_preds.size(1), arg_targets.size(1))
        arg_loss = self.arg_loss_fn(
            arg_preds[:, :T_arg, :],
            arg_targets[:, :T_arg, :].to(arg_preds.dtype),
        )

        # Total 
        total_loss = cmd_loss + self.lambda_args * arg_loss

        return total_loss