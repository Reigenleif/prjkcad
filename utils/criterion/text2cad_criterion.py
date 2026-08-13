from __future__ import annotations

from typing import Any, Dict, Tuple, Union
import torch
from torch import nn

from utils.criterion.base_criterion import BaseCriterion
from models.text2cad_ori.loss import CELoss


class Text2CADCriterion(BaseCriterion):
    """Criterion for Text2CAD model training wrapping CELoss."""

    def __init__(self, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", **kwargs):
        super().__init__()
        self.device = torch.device(device) if isinstance(device, str) else device
        self.loss_fn = CELoss(device=self.device)

    def forward(self, inputs: Any, targets: Any = None) -> torch.Tensor:
        if isinstance(inputs, dict):
            cad_dict = dict(inputs)
        elif isinstance(inputs, (tuple, list)) and len(inputs) >= 1 and isinstance(inputs[0], dict):
            cad_dict = dict(inputs[0])
        else:
            cad_dict = {"pred": inputs}

        if targets is not None and "target" not in cad_dict:
            cad_dict["target"] = targets

        loss, _ = self.loss_fn(cad_dict)
        return loss
