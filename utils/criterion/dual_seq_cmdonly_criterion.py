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
        self.cmd_loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_id)

    def forward(self, cmd_logits, cmd_targets):
        T = min(cmd_logits.size(1), cmd_targets.size(1))
        
        cmd_logits = cmd_logits[:, :T, :]
        cmd_targets = cmd_targets[:, :T]

        cmd_loss = self.cmd_loss_fn(
            cmd_logits.reshape(-1, cmd_logits.size(-1)),
            cmd_targets.reshape(-1),
        )
        return cmd_loss
