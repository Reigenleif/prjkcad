from __future__ import annotations

from functools import lru_cache

DEFAULT_COMMANDS = {
    "COOR": ["coor_euax", "coor_euay", "coor_euaz", "coor_tx", "coor_ty", "coor_tz"],
    "FACE": [],
    "LOOP": [],
    "LINE": ["line_sx", "line_sy", "line_ex", "line_ey"],
    "CIRCLE": ["circle_cx", "circle_cy", "circle_r"],
    "ARC": ["arc_sx", "arc_sy", "arc_mx", "arc_my", "arc_ex", "arc_ey"],
    "EXTRUDE_NEW": ["extrude_new_dtn", "extrude_new_don", "extrude_new_scale"],
    "EXTRUDE_JOIN": ["extrude_join_dtn", "extrude_join_don", "extrude_join_scale"],
    "EXTRUDE_CUT": ["extrude_cut_dtn", "extrude_cut_don", "extrude_cut_scale"],
    "EXTRUDE_INTERSECT": ["extrude_intersect_dtn", "extrude_intersect_don", "extrude_intersect_scale"],
}
SPECIAL_COMMANDS = ("PAD", "SOS", "EOS")


@lru_cache(maxsize=1)
def build_dualseq_schema() -> dict[str, object]:
    """
    Dual Sequence token schema Principle:
    - Starts with special commands (SOS, EOS, PAD)
    - For each face:
        - COOR command with 6 arguments (euax, euay, euaz, tx, ty, tz)
        - For each loop in the face:
            - LOOP command with no argument
            - For each segment in the loop:
                - LINE command with 4 arguments (line_sx, line_sy, line_ex, line_ey) for line segment
                - CIRCLE command with 3 arguments (circle_cx, circle_cy, circle_r) for circle segment
                - ARC command with 6 arguments (arc_sx, arc_sy, arc_mx, arc_my, arc_ex, arc_ey) for arc segment
    - For each extrusion:
        - EXTRUDE_NEW, EXTRUDE_JOIN, EXTRUDE_CUT, EXTRUDE_INTERSECT command with 3 arguments (extrude_dtn, extrude_don, extrude_scale)
    """
    
    
    sos_id = 1
    pad_id = 0
    eos_id = 2
    command_names = list(DEFAULT_COMMANDS)
    command_to_id = {name: index + len(SPECIAL_COMMANDS) for index, name in enumerate(command_names)}
    command_to_id.update({name: index for index, name in enumerate(SPECIAL_COMMANDS)})
    arg_names: list[str] = []
    command_to_slice: dict[str, tuple[int, int]] = {}
    command_to_mask: dict[str, list[int]] = {}
    cursor = 0
    for command_name in command_names:
        command_args = list(DEFAULT_COMMANDS[command_name])
        arg_names.extend(command_args)
        command_to_slice[command_name] = (cursor, cursor + len(command_args))
        cursor += len(command_args)
    arg_dim = len(arg_names)
    for command_name in command_names:
        start, end = command_to_slice[command_name]
        command_to_mask[command_name] = [1 if start <= index < end else 0 for index in range(arg_dim)]

    return {"command_names": command_names, 
            "command_to_id": command_to_id, 
            "id_to_command": {index: name for name, index in command_to_id.items()}, 
            "arg_names": arg_names, 
            "command_to_slice": command_to_slice, 
            "command_to_mask": command_to_mask, 
            "n_cmds": len(command_names), 
            "n_tokens": len(command_names) + len(SPECIAL_COMMANDS),  # including PAD, SOS, EOS
            "n_args": arg_dim, 
            "sos_id": sos_id, 
            "pad_id": pad_id, 
            "eos_id": eos_id}


def get_dualseq_schema() -> dict[str, object]:
    return build_dualseq_schema()
