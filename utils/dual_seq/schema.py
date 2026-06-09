from __future__ import annotations

from functools import lru_cache

DEFAULT_COMMANDS = {
    "COOR": ["coor_euax", "coor_euay", "coor_euaz", "coor_tx", "coor_ty", "coor_tz"],
    "FACE": [],
    "LOOP": [],
    "LINE": ["line_sx", "line_sy", "line_ex", "line_ey"],
    "CIRCLE": ["circle_cx", "circle_cy", "circle_r"],
    "EXTRUDE_NEW": ["extrude_new_dtn", "extrude_new_don", "extrude_new_scale"],
    "EXTRUDE_JOIN": ["extrude_join_dtn", "extrude_join_don", "extrude_join_scale"],
    "EXTRUDE_CUT": ["extrude_cut_dtn", "extrude_cut_don", "extrude_cut_scale"],
    "EXTRUDE_INTERSECT": ["extrude_intersect_dtn", "extrude_intersect_don", "extrude_intersect_scale"],
}
SPECIAL_COMMANDS = ("<SOS>", "<PAD>", "<EOS>")


@lru_cache(maxsize=1)
def build_dualseq_schema() -> dict[str, object]:
    command_names = list(DEFAULT_COMMANDS)
    command_to_id = {name: index + 2 for index, name in enumerate(command_names)}
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
    sos_id = 1
    pad_id = 0
    eos_id = 2
    return {"command_names": command_names, 
            "command_to_id": command_to_id, 
            "id_to_command": {index: name for name, index in command_to_id.items()}, 
            "arg_names": arg_names, 
            "command_to_slice": command_to_slice, 
            "command_to_mask": command_to_mask, 
            "n_cmds": len(command_names), 
            "n_tokens": len(command_names) + 2,
            "n_args": arg_dim, 
            "sos_id": sos_id, 
            "pad_id": pad_id, 
            "eos_id": eos_id}


def get_dualseq_schema() -> dict[str, object]:
    return build_dualseq_schema()
