from __future__ import annotations

from typing import Any

import torch
from torch import nn

from utils.dual_seq import get_dualseq_schema


class DualSeqCriterion(nn.Module):
    def __init__(
        self,
        cmd_loss_weight: float = 1.0,
        arg_loss_weight: float = 1.0,
        pad_id: int | None = None,
    ):
        super().__init__()
        schema = get_dualseq_schema()
        self.cmd_loss_weight = cmd_loss_weight
        self.arg_loss_weight = arg_loss_weight
        self.pad_id = schema["pad_id"] if pad_id is None else pad_id
        self.n_args = schema["n_args"]
        self.cmd_loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_id)

        command_arg_masks = torch.zeros(schema["command_vocab_size"], self.n_args, dtype=torch.float32)
        for command_name, command_id in schema["command_to_id"].items():
            command_arg_masks[command_id] = torch.tensor(schema["command_to_mask"][command_name], dtype=torch.float32)

        self.register_buffer("command_arg_masks", command_arg_masks, persistent=False)

    def forward(
        self,
        cmd_logits: torch.Tensor,
        arg_preds: torch.Tensor,
        cmd_targets: torch.Tensor,
        arg_targets: torch.Tensor,
        arg_masks: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if arg_masks is None:
            arg_masks = self.command_arg_masks[cmd_targets]

        cmd_loss = self.cmd_loss_fn(
            cmd_logits.reshape(-1, cmd_logits.size(-1)),
            cmd_targets.reshape(-1),
        )

        valid_mask = cmd_targets.ne(self.pad_id)
        if decoder_attention_mask is not None:
            valid_mask = valid_mask & decoder_attention_mask.bool()

        predicted_cmds = cmd_logits.argmax(dim=-1)
        correct_mask = predicted_cmds.eq(cmd_targets) & valid_mask

        active_mask = arg_masks.to(arg_preds.dtype) * correct_mask.unsqueeze(-1).to(arg_preds.dtype)
        squared_error = (arg_preds - arg_targets).pow(2) * active_mask
        active_count = active_mask.sum().clamp(min=1.0)
        arg_loss = squared_error.sum() / active_count

        total_loss = self.cmd_loss_weight * cmd_loss + self.arg_loss_weight * arg_loss
        cmd_accuracy = correct_mask.float().sum() / valid_mask.float().sum().clamp(min=1.0)

        return {
            "loss": total_loss,
            "cmd_loss": cmd_loss,
            "arg_loss": arg_loss,
            "cmd_accuracy": cmd_accuracy,
        }
