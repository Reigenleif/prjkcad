from __future__ import annotations

from typing import Any, Dict
import torch
from torch import nn

# <-- Base Criterion Class -->
class BaseCriterion(nn.Module):
    """Base class for all loss criterion modules receiving dict outputs and batches."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, outputs: Dict[str, Any], batch: Dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward()")
