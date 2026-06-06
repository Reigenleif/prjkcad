from __future__ import annotations

from typing import Any

import torch
from torch import nn

from utils.dual_seq import get_dualseq_schema


class DualSeqCMDOnlyCriterion(nn.Module):
    def __init__(
        self,
        pad_id: int | None = None,
    ):
        super().__init__()
        schema = get_dualseq_schema()
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
        cmd_targets: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        
        # Cross-entropy command loss
        cmd_loss = self.cmd_loss_fn(
            cmd_logits.reshape(-1, cmd_logits.size(-1)),
            cmd_targets.reshape(-1),
        )

        valid_mask = cmd_targets.ne(self.pad_id)
        if attention_mask is not None:
            valid_mask = valid_mask & attention_mask.bool()

        predicted_cmds = cmd_logits.argmax(dim=-1)
        correct_mask = predicted_cmds.eq(cmd_targets) & valid_mask

        cmd_accuracy = correct_mask.float().sum() / valid_mask.float().sum().clamp(min=1.0)

        return {
            "loss": cmd_loss,
            "cmd_loss": cmd_loss,
            "cmd_accuracy": cmd_accuracy,
        }
