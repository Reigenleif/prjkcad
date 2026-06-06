from __future__ import annotations

from .schema import DEFAULT_COMMANDS, build_dualseq_schema


def encode_command(command: str) -> int:
    return build_dualseq_schema()["command_to_id"][command]


def encode_args(command: str, arg_values: dict[str, float]) -> list[float]:
    schema = build_dualseq_schema()
    arg_vector = [0.0] * len(schema["arg_names"])
    start, _ = schema["command_to_slice"][command]
    for offset, arg_name in enumerate(DEFAULT_COMMANDS[command]):
        arg_vector[start + offset] = float(arg_values.get(arg_name, 0.0))
    return arg_vector
