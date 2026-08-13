from typing import Any
import torch
import torch.nn as nn


class CELoss(nn.Module):
    """
    Cross Entropy Loss for Text2CAD
    """

    def __init__(self, device):
        super(CELoss, self).__init__()

        self.ce_cad = nn.CrossEntropyLoss(reduction="none", label_smoothing=0.1)
        self.ce_pc = nn.CrossEntropyLoss()
        self.mseloss = nn.MSELoss()

    def forward(self, cad_dict: dict | tuple | list, target_arg: Any = None):
        """
        cad_dict: dictionary or tuple containing predictions and targets.
                pred: shape (B, N, 2, C) or dict with 'cmd_logits'/'arg_logits'
                target: shape (B, N, 2) or tuple/dict of targets
                key_padding_mask: shape (B, N) or (B, N, 2)
        """
        pred = None
        target = None
        key_padding_mask = None

        if isinstance(cad_dict, (tuple, list)):
            pred = cad_dict[0]
            if len(cad_dict) > 1:
                target = cad_dict[1]
            if len(cad_dict) > 2:
                key_padding_mask = cad_dict[2]
        elif isinstance(cad_dict, dict):
            # Check for native Text2CAD 4D pred tensor
            for k in ["pred", "prediction", "predictions", "S_output"]:
                if k in cad_dict:
                    pred = cad_dict[k]
                    break

            # Check target
            for k in ["target", "targets", "gt", "labels", "cad_vec"]:
                if k in cad_dict:
                    target = cad_dict[k]
                    break

            # Check padding mask
            for k in ["key_padding_mask", "padding_mask", "mask", "attn_mask"]:
                if k in cad_dict:
                    key_padding_mask = cad_dict[k]
                    break

        if target is None and target_arg is not None:
            target = target_arg

        # Handle DualSeq wrapper dict format: ['cmd_logits', 'arg_logits', 'cmd_preds', 'arg_preds']
        if pred is None and isinstance(cad_dict, dict) and ("cmd_logits" in cad_dict or "logits" in cad_dict):
            pred_x = cad_dict.get("cmd_logits", cad_dict.get("logits"))
            pred_y = cad_dict.get("arg_logits", cad_dict.get("arg_preds"))

            if isinstance(target, (tuple, list)):
                target_x = target[0]
                target_y = target[1] if len(target) > 1 else None
            elif isinstance(target, dict):
                target_x = target.get("cmd_targets", target.get("target_x"))
                target_y = target.get("arg_targets", target.get("target_y"))
            elif isinstance(target, torch.Tensor) and target.dim() == 3 and target.size(-1) == 2:
                target_x = target[:, :, 0]
                target_y = target[:, :, 1]
            elif isinstance(target, torch.Tensor):
                target_x = target
                target_y = cad_dict.get("arg_targets")
            else:
                target_x = cad_dict.get("cmd_targets")
                target_y = cad_dict.get("arg_targets")

            # Calculate loss_seq_x
            if pred_x is not None and target_x is not None:
                if pred_x.dim() == 3:
                    loss_x_raw = self.ce_cad(pred_x.permute(0, 2, 1), target_x.long())
                else:
                    loss_x_raw = self.ce_cad(pred_x, target_x.long())

                if key_padding_mask is not None:
                    m_x = key_padding_mask[:, :, 0] if key_padding_mask.dim() == 3 else key_padding_mask
                    self.loss_seq_x = torch.sum(loss_x_raw * m_x) / torch.clamp(torch.sum(m_x * 1), min=1.0)
                else:
                    self.loss_seq_x = torch.mean(loss_x_raw)
            else:
                device = pred_x.device if pred_x is not None else "cpu"
                self.loss_seq_x = torch.tensor(0.0, device=device)

            # Calculate loss_seq_y
            if pred_y is not None and target_y is not None:
                if pred_y.dim() == 3 and target_y.dim() == 2:
                    loss_y_raw = self.ce_cad(pred_y.permute(0, 2, 1), target_y.long())
                elif pred_y.shape == target_y.shape:
                    loss_y_raw = self.mseloss(pred_y, target_y.float())
                else:
                    loss_y_raw = self.ce_cad(pred_y.permute(0, 2, 1), target_y.long())

                if key_padding_mask is not None and loss_y_raw.dim() > 0:
                    m_y = key_padding_mask[:, :, 1] if key_padding_mask.dim() == 3 else key_padding_mask
                    self.loss_seq_y = torch.sum(loss_y_raw * m_y) / torch.clamp(torch.sum(m_y * 1), min=1.0)
                else:
                    self.loss_seq_y = torch.mean(loss_y_raw)
            else:
                self.loss_seq_y = torch.tensor(0.0, device=self.loss_seq_x.device)

            self.loss_seq_x = torch.nan_to_num(self.loss_seq_x, nan=0.0)
            self.loss_seq_y = torch.nan_to_num(self.loss_seq_y, nan=0.0)
            self.loss_seq = (self.loss_seq_x + self.loss_seq_y) / 2
            self.loss_seq = torch.nan_to_num(self.loss_seq, nan=0.0)
            return self.loss_seq, {"loss_seq": self.loss_seq.detach().item()}

        if pred is None or target is None:
            keys_info = list(cad_dict.keys()) if isinstance(cad_dict, dict) else type(cad_dict)
            raise KeyError(f"CELoss expected 'pred' and 'target' in cad_dict, but got: {keys_info}")

        if key_padding_mask is None:
            key_padding_mask = torch.ones_like(target, dtype=torch.float32, device=pred.device)

        if key_padding_mask.dim() == 2:
            key_padding_mask = key_padding_mask.unsqueeze(-1).expand(-1, -1, 2)

        self.loss_seq_x = torch.sum(
            self.ce_cad(
                pred[:, :, 0].permute(0, 2, 1),
                target[:, :, 0].long(),
            )
            * key_padding_mask[:, :, 0]
        ) / torch.clamp(torch.sum(key_padding_mask[:, :, 0] * 1), min=1.0)

        self.loss_seq_y = torch.sum(
            self.ce_cad(
                pred[:, :, 1].permute(0, 2, 1),
                target[:, :, 1].long(),
            )
            * key_padding_mask[:, :, 1]
        ) / torch.clamp(torch.sum(key_padding_mask[:, :, 1] * 1), min=1.0)

        self.loss_seq_x = torch.nan_to_num(self.loss_seq_x, nan=0.0)
        self.loss_seq_y = torch.nan_to_num(self.loss_seq_y, nan=0.0)
        self.loss_seq = (self.loss_seq_x + self.loss_seq_y) / 2
        self.loss_seq = torch.nan_to_num(self.loss_seq, nan=0.0)
        loss_keys = ["loss_seq"]

        result_dict = {key: getattr(self, key).detach().item() for key in loss_keys}

        loss = self.loss_seq
        return loss, result_dict
