from __future__ import annotations

from typing import Any

import torch
from torch import nn

from utils.dual_seq import get_dualseq_schema


class DualSeqCMDOnlyCriterion(nn.Module):
    def __init__(
        self,
        pad_id: int | None = None,
        lambda_after_pad: float = 3.0,
        lambda_overgen: float = 3.0,
    ):
        super().__init__()
        schema = get_dualseq_schema()
        self.pad_id = schema["pad_id"] if pad_id is None else pad_id

        self.cmd_loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_id)

        self.lambda_after_pad = lambda_after_pad
        self.lambda_overgen = lambda_overgen

    def forward(self, cmd_logits, cmd_targets):
        B, T_pred, V = cmd_logits.shape
        T_tgt = cmd_targets.size(1)

        T = min(T_pred, T_tgt)

        # Base CE loss
        logits_trim = cmd_logits[:, :T, :]
        targets_trim = cmd_targets[:, :T]

        cmd_loss = self.cmd_loss_fn(
            logits_trim.reshape(-1, V),
            targets_trim.reshape(-1),
        )

        preds = cmd_logits.argmax(dim=-1)  # (B, T_pred)

        # 1) penalty: non-pad AFTER first pad in prediction
        pred_is_pad = preds.eq(self.pad_id)

        # find first PAD index per sequence
        # mask trick: large index for non-pad, then min
        idxs = torch.arange(T_pred, device=preds.device).unsqueeze(0).expand(B, -1)
        first_pad_idx = torch.where(
            pred_is_pad,
            idxs,
            torch.full_like(idxs, T_pred)
        ).min(dim=1).values  # (B,)

        # build mask: positions AFTER first PAD
        after_pad_mask = idxs > first_pad_idx.unsqueeze(1)

        # violation: non-pad after pad
        non_pad_after_pad = after_pad_mask & (~pred_is_pad)

        penalty_after_pad = non_pad_after_pad.float().sum() / B

        # 2) penalty: generating beyond target length
        if T_pred > T_tgt:
            extra_preds = preds[:, T_tgt:]  # (B, extra_len)

            # count non-pad tokens in extra region
            overgen_mask = extra_preds.ne(self.pad_id)
            penalty_overgen = overgen_mask.float().sum() / B
        else:
            penalty_overgen = torch.tensor(0.0, device=cmd_logits.device)

        # FINAL
        total_loss = (
            cmd_loss
            + self.lambda_after_pad * penalty_after_pad
            + self.lambda_overgen * penalty_overgen
        )

        return total_loss