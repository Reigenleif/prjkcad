import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.dual_seq import get_dualseq_schema


class StructuralValidityLoss(nn.Module):
    """Differentiable loss penalizing invalid CAD command transitions."""

    def __init__(self, schema=None):
        super().__init__()
        if schema is None:
            schema = get_dualseq_schema()
        n_cmds = schema["cmd_n_tokens"]
        valid = self._build_valid_transitions(n_cmds, schema)
        self.register_buffer("invalid_mask", (~valid).float())

    def _build_valid_transitions(self, n_cmds, schema):
        cmd_to_id = schema["command_to_id"]
        valid = torch.zeros(n_cmds, n_cmds, dtype=torch.bool)

        PAD = cmd_to_id["PAD"]
        SOS = cmd_to_id["SOS"]
        EOS = cmd_to_id["EOS"]
        COOR = cmd_to_id["COOR"]
        FACE = cmd_to_id["FACE"]
        LOOP = cmd_to_id["LOOP"]
        LINE = cmd_to_id["LINE"]
        CIRCLE = cmd_to_id["CIRCLE"]
        ARC = cmd_to_id["ARC"]
        EXT_NEW = cmd_to_id["EXTRUDE_NEW"]
        EXT_JOIN = cmd_to_id["EXTRUDE_JOIN"]
        EXT_CUT = cmd_to_id["EXTRUDE_CUT"]
        EXT_INT = cmd_to_id["EXTRUDE_INTERSECT"]

        sketch_prims = [LINE, CIRCLE, ARC]
        extrude_cmds = [EXT_NEW, EXT_JOIN, EXT_CUT, EXT_INT]

        # Valid Grammars
        valid[PAD, PAD] = True # Pad -> Pad (the tail)
        valid[SOS, COOR] = True # Start -> COOR 
        valid[EOS, PAD] = True # EOS -> Pad (the tail)
        valid[COOR, FACE] = True # COOR -> FACE
        valid[FACE, LOOP] = True # FACE -> LOOP

        for p in sketch_prims:
            valid[LOOP, p] = True

        for p in sketch_prims:
            for q in sketch_prims:
                valid[p, q] = True
            valid[p, LOOP] = True
            valid[p, FACE] = True
            for e in extrude_cmds:
                valid[p, e] = True
            valid[p, COOR] = True
            valid[p, EOS] = True

        for e in extrude_cmds:
            valid[e, COOR] = True
            valid[e, EOS] = True

        return valid

    def forward(self, cmd_logits):
        if self.invalid_mask.device != cmd_logits.device:
            self.invalid_mask = self.invalid_mask.to(cmd_logits.device)
        probs = F.softmax(cmd_logits, dim=-1)
        prev_probs = probs[:, :-1, :]
        next_probs = probs[:, 1:, :]
        weighted_invalid = torch.matmul(prev_probs, self.invalid_mask)
        penalty = (weighted_invalid * next_probs).sum(dim=-1)
        return penalty.mean()
