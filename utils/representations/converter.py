from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch

from utils.representations.dual_seq.dual_seq import (
    DualSeq,
    DualSeqMetadata,
    float_to_tokens,
    tokens_to_float,
)
from utils.representations.dual_seq.schema import DEFAULT_COMMANDS
from utils.representations.CadSeqProc.cad_sequence import CADSequence
from utils.representations.CadSeqProc.utility.macro import END_TOKEN, MAX_CAD_SEQUENCE_LENGTH, N_BIT

def basic_to_one_head_tokenized(cmds: list[str], args: list[dict], schema: dict) -> list[int]:
    all_tokens = []
    for i, cmd in enumerate(cmds):
        cmd_arg_names = DEFAULT_COMMANDS.get(cmd, [])
        if not cmd_arg_names:
            continue
        arg_dict = args[i] if i < len(args) else {}
        for arg_name in cmd_arg_names:
            val = arg_dict.get(arg_name, 0.0)
            all_tokens.extend(float_to_tokens(val, schema))
            all_tokens.append(schema["arg_sep_id"])
    return all_tokens

def one_head_tokenized_to_basic(cmds: list[str], tokens: list[int], schema: dict) -> list[dict]:
    sep_id = schema["arg_sep_id"]
    arg_groups = []
    current_group = []
    for t in tokens:
        if t == sep_id:
            arg_groups.append(current_group)
            current_group = []
        else:
            current_group.append(t)
    if current_group:
        arg_groups.append(current_group)
        
    decoded_args_dict = []
    arg_group_idx = 0
    for cmd in cmds:
        arg_dict = {}
        if cmd in DEFAULT_COMMANDS:
            for arg_name in DEFAULT_COMMANDS[cmd]:
                if arg_group_idx < len(arg_groups):
                    val = tokens_to_float(arg_groups[arg_group_idx], schema)
                    arg_dict[arg_name] = val
                    arg_group_idx += 1
                else:
                    arg_dict[arg_name] = 0.0
        decoded_args_dict.append(arg_dict)
    return decoded_args_dict

def basic_to_eight_bit_binarized(cmds: list[str], args: list[dict], metadata: DualSeqMetadata) -> list[dict]:
    binned_args = []
    for i, cmd in enumerate(cmds):
        arg_dict = args[i] if i < len(args) else {}
        binned_dict = {}
        if cmd in DEFAULT_COMMANDS:
            for arg_name in DEFAULT_COMMANDS[cmd]:
                val = arg_dict.get(arg_name, 0.0)
                binned_dict[arg_name] = metadata.float_to_bin(arg_name, val)
        binned_args.append(binned_dict)
    return binned_args

def eight_bit_binarized_to_basic(cmds: list[str], binned_args: list[dict], metadata: DualSeqMetadata) -> list[dict]:
    float_args = []
    for i, cmd in enumerate(cmds):
        binned_dict = binned_args[i] if i < len(binned_args) else {}
        float_dict = {}
        if cmd in DEFAULT_COMMANDS:
            for arg_name in DEFAULT_COMMANDS[cmd]:
                bin_val = binned_dict.get(arg_name, 0)
                float_dict[arg_name] = metadata.bin_to_float(arg_name, bin_val)
        float_args.append(float_dict)
    return float_args

def dualseq_to_minimal_json(ds: Any, args_list: Optional[List[Dict[str, Any]]] = None) -> dict:
    if hasattr(ds, "json_object") and isinstance(ds.json_object, dict) and "parts" in ds.json_object:
        return ds.json_object

    if isinstance(ds, DualSeq):
        cmds = getattr(ds, "cmds", [])
        args_list = getattr(ds, "args_dict", getattr(ds, "args", []))
    else:
        cmds = ds if isinstance(ds, list) else []
        args_list = args_list if isinstance(args_list, list) else []

    parts = {}
    part_idx, face_idx, loop_idx, seg_idx = 0, 0, 0, 0
    curr_part, curr_face, curr_loop = None, None, None
    op_map = {
        "EXTRUDE_NEW": "NewBodyFeatureOperation",
        "EXTRUDE_JOIN": "JoinFeatureOperation",
        "EXTRUDE_CUT": "CutFeatureOperation",
        "EXTRUDE_INTERSECT": "IntersectFeatureOperation",
    }

    def ensure_part():
        nonlocal part_idx, curr_part, face_idx, loop_idx, seg_idx
        if curr_part is None:
            part_idx += 1
            curr_part = {
                "coordinate_system": {
                    "Euler Angles": [0.0, 0.0, 0.0],
                    "Translation Vector": [0.0, 0.0, 0.0],
                },
                "sketch": {},
                "extrusion": {
                    "operation": "NewBodyFeatureOperation",
                    "extrude_depth_towards_normal": 0.0,
                    "extrude_depth_opposite_normal": 0.0,
                    "sketch_scale": 1.0,
                },
            }
            parts[f"part_{part_idx}"] = curr_part
            face_idx, loop_idx, seg_idx = 0, 0, 0

    def ensure_face():
        nonlocal face_idx, curr_face, loop_idx, seg_idx
        ensure_part()
        if curr_face is None:
            face_idx += 1
            curr_face = {}
            curr_part["sketch"][f"face_{face_idx}"] = curr_face
            loop_idx, seg_idx = 0, 0

    def ensure_loop():
        nonlocal loop_idx, curr_loop, seg_idx
        ensure_face()
        if curr_loop is None:
            loop_idx += 1
            curr_loop = {}
            curr_face[f"loop_{loop_idx}"] = curr_loop
            seg_idx = 0

    for cmd, arg in zip(cmds, args_list):
        if not isinstance(arg, dict):
            arg = {}
        if cmd == "COOR":
            part_idx += 1
            curr_part = {
                "coordinate_system": {
                    "Euler Angles": [arg.get("coor_euax", 0.0), arg.get("coor_euay", 0.0), arg.get("coor_euaz", 0.0)],
                    "Translation Vector": [arg.get("coor_tx", 0.0), arg.get("coor_ty", 0.0), arg.get("coor_tz", 0.0)],
                },
                "sketch": {},
                "extrusion": {
                    "operation": "NewBodyFeatureOperation",
                    "extrude_depth_towards_normal": 0.0,
                    "extrude_depth_opposite_normal": 0.0,
                    "sketch_scale": 1.0,
                },
            }
            parts[f"part_{part_idx}"] = curr_part
            curr_face = None
            curr_loop = None
            face_idx, loop_idx, seg_idx = 0, 0, 0
        elif cmd == "FACE":
            ensure_part()
            face_idx += 1
            curr_face = {}
            curr_part["sketch"][f"face_{face_idx}"] = curr_face
            curr_loop = None
            loop_idx, seg_idx = 0, 0
        elif cmd == "LOOP":
            ensure_face()
            loop_idx += 1
            curr_loop = {}
            curr_face[f"loop_{loop_idx}"] = curr_loop
            seg_idx = 0
        elif cmd == "LINE":
            ensure_loop()
            seg_idx += 1
            curr_loop[f"line_{seg_idx}"] = {
                "Start Point": [arg.get("line_sx", 0.0), arg.get("line_sy", 0.0)],
                "End Point": [arg.get("line_ex", 0.0), arg.get("line_ey", 0.0)],
            }
        elif cmd == "CIRCLE":
            ensure_loop()
            seg_idx += 1
            curr_loop[f"circle_{seg_idx}"] = {
                "Center": [arg.get("circle_cx", 0.0), arg.get("circle_cy", 0.0)],
                "Radius": arg.get("circle_r", 0.0),
            }
        elif cmd == "ARC":
            ensure_loop()
            seg_idx += 1
            curr_loop[f"arc_{seg_idx}"] = {
                "Start Point": [arg.get("arc_sx", 0.0), arg.get("arc_sy", 0.0)],
                "Mid Point": [arg.get("arc_mx", 0.0), arg.get("arc_my", 0.0)],
                "End Point": [arg.get("arc_ex", 0.0), arg.get("arc_ey", 0.0)],
            }
        elif cmd in op_map:
            ensure_part()
            prefix = cmd.lower()
            curr_part["extrusion"] = {
                "operation": op_map[cmd],
                "extrude_depth_towards_normal": arg.get(f"{prefix}_dtn", 0.0),
                "extrude_depth_opposite_normal": arg.get(f"{prefix}_don", 0.0),
                "sketch_scale": arg.get(f"{prefix}_scale", 1.0),
            }

    return {"parts": parts}

def dualseq_to_cadseq(ds: Any, args_list: Optional[List[Dict[str, Any]]] = None) -> Optional[CADSequence]:
    try:
        minimal_json = dualseq_to_minimal_json(ds, args_list=args_list)
        return CADSequence.from_minimal_json(minimal_json)
    except Exception:
        return None

def cadseq_to_vec_2t(cad_seq: CADSequence) -> torch.Tensor:
    cad_seq = cad_seq.numericalize(bit=N_BIT)
    cad_seq = cad_seq.to_vec(padding=False)
    # Output as a vector with shape (2, T)
    return cad_seq.cad_vec.T.contiguous().long()

def vec_2t_to_cadseq(vec: Union[torch.Tensor, np.ndarray]) -> Optional[CADSequence]:
    if isinstance(vec, torch.Tensor):
        vec = vec.detach().cpu().numpy()
    if vec.ndim == 2 and vec.shape[0] == 2:
        vec = vec.T
    if vec.ndim == 2 and vec.shape[1] == 2 and vec.shape[0] > 0:
        # Ensure a closing START token [1, 0] exists after index 0 for valid CADSequence splitting
        if not (vec[1:, 0] == END_TOKEN.index("START")).any():
            start_tok = np.array([[END_TOKEN.index("START"), 0]], dtype=vec.dtype)
            vec = np.concatenate([vec, start_tok], axis=0)
    try:
        return CADSequence.from_vec(vec)
    except Exception:
        return None
