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
SPECIAL_ARGS = ("PAD", "SOS", "EOS", "DEC", "SEP", "NEG")


@lru_cache(maxsize=1)
def build_dualseq_schema() -> dict[str, object]:
    """
    Dual Sequence token schema for both commands and arguments.
    """
    # Commands Schema
    cmd_pad_id = 0
    cmd_sos_id = 1
    cmd_eos_id = 2
    command_names = list(DEFAULT_COMMANDS.keys())
    command_to_id = {name: index + len(SPECIAL_COMMANDS) for index, name in enumerate(command_names)}
    for index, name in enumerate(SPECIAL_COMMANDS):
        command_to_id[name] = index
        
    cmd_n_tokens = len(command_names) + len(SPECIAL_COMMANDS)

    # Args Schema
    arg_pad_id = 0
    arg_sos_id = 1
    arg_eos_id = 2
    arg_dec_id = 3
    arg_sep_id = 4
    arg_neg_id = 5
    
    arg_to_id = {name: index for index, name in enumerate(SPECIAL_ARGS)}
    # Tokens for numbers 0-999
    number_offset = len(SPECIAL_ARGS)
    for i in range(1000):
        arg_to_id[str(i)] = number_offset + i
        
    args_n_tokens = len(SPECIAL_ARGS) + 1000

    id_to_arg = {index: name for name, index in arg_to_id.items()}

    # Original arg_names and slices for compatibility (if needed for evaluation/decoding mapping)
    arg_names: list[str] = []
    cursor = 0
    command_to_slice: dict[str, tuple[int, int]] = {}
    for command_name in command_names:
        command_args = list(DEFAULT_COMMANDS[command_name])
        arg_names.extend(command_args)
        command_to_slice[command_name] = (cursor, cursor + len(command_args))
        cursor += len(command_args)

    return {
        # Commands
        "command_names": command_names, 
        "command_to_id": command_to_id, 
        "id_to_command": {index: name for name, index in command_to_id.items()}, 
        "cmd_n_tokens": cmd_n_tokens,
        "cmd_pad_id": cmd_pad_id,
        "cmd_sos_id": cmd_sos_id,
        "cmd_eos_id": cmd_eos_id,
        
        # Args
        "arg_to_id": arg_to_id,
        "id_to_arg": id_to_arg,
        "args_n_tokens": args_n_tokens,
        "arg_pad_id": arg_pad_id,
        "arg_sos_id": arg_sos_id,
        "arg_eos_id": arg_eos_id,
        "arg_dec_id": arg_dec_id,
        "arg_sep_id": arg_sep_id,
        "arg_neg_id": arg_neg_id,
        
        # Legacy mappings (useful for reference or decoding back to dicts)
        "arg_names": arg_names, 
        "command_to_slice": command_to_slice,
        
        # Backwards compatibility names
        "pad_id": cmd_pad_id,
        "sos_id": cmd_sos_id,
        "eos_id": cmd_eos_id,
        "n_tokens": cmd_n_tokens,
    }
    

def get_dualseq_schema() -> dict[str, object]:
    return build_dualseq_schema()
