from __future__ import annotations

from typing import Any, Dict, Union
import torch
from torch import nn

from utils.criterion.base_criterion import BaseCriterion
from utils.criterion.float_args_criterion import FloatArgsCriterion
from utils.criterion.eight_bit_binarized_args_criterion import EightBitBinarizedArgsCriterion
from utils.criterion.tokenized_one_sequence_args_criterion import TokenizedOneSequenceArgsCriterion
from utils.criterion.pretrain_criterion import PretrainCriterion

class CustomCriterion(BaseCriterion):
    """Factory controller for criterion components based on configuration."""

    def __init__(self, criterion_cfg: Any = None, out_type: str = "FloatArgs", **kwargs):
        super().__init__()
        # <-- Private Init Routing -->
        self.criterion = self._create_criterion(criterion_cfg, out_type, **kwargs)

    def _create_criterion(self, criterion_cfg: Any, out_type: str, **kwargs) -> nn.Module:
        # <-- Configuration Unpacking Guard -->
        cls_name = getattr(criterion_cfg, "cls", None) or out_type
        c_kwargs = getattr(criterion_cfg, "kwargs", {}) if criterion_cfg else {}
        merged_kwargs = {**c_kwargs, **kwargs}

        # <-- Variant Controller Guard Chain -->
        if cls_name in ["FloatArgsCriterion", "FloatArgs"]:
            return FloatArgsCriterion(**merged_kwargs)
        if cls_name in ["EightBitBinarizedArgsCriterion", "EightBitBinarizedArgs"]:
            return EightBitBinarizedArgsCriterion(**merged_kwargs)
        if cls_name in ["TokenizedOneSequenceArgsCriterion", "TokenizedOneSequenceArgs"]:
            return TokenizedOneSequenceArgsCriterion(**merged_kwargs)
        if cls_name in ["PretrainCriterion", "pretrain"]:
            return PretrainCriterion(**merged_kwargs)

        # <-- Fallback Default -->
        return FloatArgsCriterion(**merged_kwargs)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        # <-- Forward Pass to Instantiated Criterion Variant -->
        return self.criterion(*args, **kwargs)
