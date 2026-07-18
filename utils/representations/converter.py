from __future__ import annotations

from utils.representations.dual_seq.dual_seq import float_to_tokens, tokens_to_float, DualSeqMetadata
from utils.representations.dual_seq.schema import DEFAULT_COMMANDS

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
