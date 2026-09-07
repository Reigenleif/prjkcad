from __future__ import annotations

from typing import Any, Dict, Tuple, Union
import torch
from torch import nn

from models.text2cad_ori.loss import CELoss
from utils.criterion.base_criterion import BaseCriterion
from utils.representations.CadSeqProc.utility.macro import END_TOKEN


class Text2CADCriterion(BaseCriterion):
    """Criterion for Text2CAD model training wrapping CELoss."""

    def __init__(self, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", **kwargs):
        super().__init__()
        self.device = torch.device(device) if isinstance(device, str) else device
        self.loss_fn = CELoss(device=self.device)

    def forward(self, inputs: Any, targets: Any = None) -> torch.Tensor:
        pad_id = END_TOKEN.index("PADDING")
        if isinstance(inputs, dict):
            cad_dict = dict(inputs)
        elif isinstance(inputs, (tuple, list)) and len(inputs) >= 1 and isinstance(inputs[0], dict):
            cad_dict = dict(inputs[0])
        elif isinstance(inputs, (tuple, list)):
            cad_dict = {"pred": inputs[0]}
        else:
            cad_dict = {"pred": inputs}

        if targets is not None and "target" not in cad_dict:
            if isinstance(targets, (tuple, list)) and len(targets) > 1 and isinstance(targets[1], torch.Tensor):
                tgt = targets[1]
                if tgt.dim() == 3 and tgt.shape[1] == 2 and tgt.shape[2] != 2:
                    tgt = tgt.permute(0, 2, 1)
                cad_dict["target"] = tgt[:, 1:, :]
                if "key_padding_mask" not in cad_dict:
                    cad_dict["key_padding_mask"] = (tgt[:, 1:, 0] != pad_id).float()
            elif isinstance(targets, torch.Tensor):
                tgt = targets
                if tgt.dim() == 3 and tgt.shape[1] == 2 and tgt.shape[2] != 2:
                    tgt = tgt.permute(0, 2, 1)
                cad_dict["target"] = tgt[:, 1:, :] if tgt.shape[1] > (cad_dict["pred"].shape[1] if hasattr(cad_dict.get("pred"), "shape") else 0) else tgt
                if "key_padding_mask" not in cad_dict:
                    cad_dict["key_padding_mask"] = (cad_dict["target"][:, :, 0] != pad_id).float()
            else:
                cad_dict["target"] = targets

        # Ensure length consistency between pred and target if needed
        if "pred" in cad_dict and "target" in cad_dict and isinstance(cad_dict["pred"], torch.Tensor) and isinstance(cad_dict["target"], torch.Tensor):
            p_len = cad_dict["pred"].shape[1]
            t_len = cad_dict["target"].shape[1]
            min_len = min(p_len, t_len)
            cad_dict["pred"] = cad_dict["pred"][:, :min_len]
            cad_dict["target"] = cad_dict["target"][:, :min_len]
            if "key_padding_mask" in cad_dict and isinstance(cad_dict["key_padding_mask"], torch.Tensor):
                cad_dict["key_padding_mask"] = cad_dict["key_padding_mask"][:, :min_len]

        loss, _ = self.loss_fn(cad_dict)
        return loss
