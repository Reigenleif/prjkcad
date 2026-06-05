from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

from utils.converter.DualSeq import DualSeq, encode_args, get_dualseq_schema
from utils.dataset.cadstr import CADSTRDatasetLoader


@dataclass
class DualSeqSample:
    text: str
    dual_seq: DualSeq
    path: str = ""
    source: str = "dualseq"


class DualSeqDataset(torch.utils.data.Dataset):
    def __init__(self, samples: Sequence[Any]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Any:
        return self.samples[index]


def _resolve_sample(sample: Any) -> tuple[str, DualSeq, str, str]:
    if isinstance(sample, DualSeqSample):
        return sample.text, sample.dual_seq, sample.path, sample.source

    if isinstance(sample, tuple) and len(sample) == 2 and isinstance(sample[1], DualSeq):
        text, dual_seq = sample
        return str(text), dual_seq, "", "dualseq"

    if isinstance(sample, dict):
        dual_seq = sample.get("dual_seq") or sample.get("dualseq") or sample.get("sequence")
        if not isinstance(dual_seq, DualSeq):
            raise TypeError("Expected a DualSeq-compatible dict with a 'dual_seq' entry")
        return (
            str(sample.get("text", "")),
            dual_seq,
            str(sample.get("path", "")),
            str(sample.get("source", "dualseq")),
        )

    if isinstance(sample, DualSeq):
        return str(getattr(sample, "text", "")), sample, str(getattr(sample, "path", "")), str(getattr(sample, "source", "dualseq"))

    dual_seq = getattr(sample, "dual_seq", None)
    if isinstance(dual_seq, DualSeq):
        return str(getattr(sample, "text", "")), dual_seq, str(getattr(sample, "path", "")), str(getattr(sample, "source", "dualseq"))

    raise TypeError(f"Unsupported DualSeq sample type: {type(sample)!r}")


def _build_dualseq_example(sample: Any) -> dict[str, Any]:
    text, dual_seq, path, source = _resolve_sample(sample)
    schema = get_dualseq_schema()
    command_to_id = schema["command_to_id"]
    command_to_mask = schema["command_to_mask"]
    pad_id = schema["pad_id"]
    sos_id = schema["sos_id"]
    eos_id = schema["eos_id"]
    arg_dim = schema["n_args"]

    command_ids = [command_to_id[command] for command in dual_seq.cmds]
    command_targets = command_ids + [eos_id]
    decoder_input_ids = [sos_id] + command_ids

    arg_targets: list[torch.Tensor] = []
    arg_masks: list[torch.Tensor] = []
    zero_args = torch.zeros(arg_dim, dtype=torch.float32)

    for command, arg_values in zip(dual_seq.cmds, dual_seq.args):
        arg_targets.append(torch.tensor(encode_args(command, arg_values), dtype=torch.float32))
        arg_masks.append(torch.tensor(command_to_mask[command], dtype=torch.float32))

    arg_targets.append(zero_args.clone())
    arg_masks.append(torch.zeros(arg_dim, dtype=torch.float32))

    decoder_input_args = [zero_args.clone()] + [tensor.clone() for tensor in arg_targets[:-1]]

    return {
        "text": text,
        "path": path,
        "source": source,
        "dual_seq": dual_seq,
        "input_ids": None,
        "command_targets": torch.tensor(command_targets, dtype=torch.long),
        "decoder_input_ids": torch.tensor(decoder_input_ids, dtype=torch.long),
        "decoder_input_args": torch.stack(decoder_input_args, dim=0),
        "decoder_attention_mask": torch.ones(len(command_targets), dtype=torch.long),
        "cmd_targets": torch.tensor(command_targets, dtype=torch.long),
        "arg_targets": torch.stack(arg_targets, dim=0),
        "arg_masks": torch.stack(arg_masks, dim=0),
        "labels": torch.tensor(command_targets, dtype=torch.long),
        "param_targets": torch.stack(arg_targets, dim=0),
        "float_targets": torch.stack(arg_targets, dim=0),
        "pad_id": pad_id,
    }


def _pad_tensor_list(tensors: list[torch.Tensor], pad_value: float | int) -> torch.Tensor:
    if not tensors:
        raise ValueError("Cannot pad an empty tensor list")

    max_len = max(tensor.size(0) for tensor in tensors)
    trailing_shape = tensors[0].shape[1:]
    result_shape = (len(tensors), max_len, *trailing_shape)
    result = torch.full(result_shape, pad_value, dtype=tensors[0].dtype)
    for index, tensor in enumerate(tensors):
        length = tensor.size(0)
        result[index, :length] = tensor
    return result


def collate_dualseq_batch(batch: Sequence[Any], tokenizer: Any) -> dict[str, Any]:
    examples = [_build_dualseq_example(sample) for sample in batch]
    texts = [example["text"] for example in examples]

    tokenizer_output = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    schema = get_dualseq_schema()
    pad_id = schema["pad_id"]

    decoder_input_ids = [example["decoder_input_ids"] for example in examples]
    decoder_input_args = [example["decoder_input_args"] for example in examples]
    decoder_attention_mask = [example["decoder_attention_mask"] for example in examples]
    cmd_targets = [example["cmd_targets"] for example in examples]
    arg_targets = [example["arg_targets"] for example in examples]
    arg_masks = [example["arg_masks"] for example in examples]

    return {
        "texts": texts,
        "paths": [example["path"] for example in examples],
        "source": [example["source"] for example in examples],
        "dual_seq": [example["dual_seq"] for example in examples],
        "input_ids": tokenizer_output["input_ids"],
        "attention_mask": tokenizer_output["attention_mask"],
        "decoder_input_ids": _pad_tensor_list(decoder_input_ids, pad_id),
        "decoder_input_args": _pad_tensor_list(decoder_input_args, 0.0),
        "decoder_attention_mask": _pad_tensor_list(decoder_attention_mask, 0),
        "cmd_targets": _pad_tensor_list(cmd_targets, pad_id),
        "arg_targets": _pad_tensor_list(arg_targets, 0.0),
        "arg_masks": _pad_tensor_list(arg_masks, 0.0),
        "labels": _pad_tensor_list(cmd_targets, pad_id),
        "param_targets": _pad_tensor_list(arg_targets, 0.0),
        "float_targets": _pad_tensor_list(arg_targets, 0.0),
    }


def make_dualseq_dataloader(
    tokenizer: Any,
    samples: Sequence[Any],
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    dataset = DualSeqDataset(samples)

    def collate_fn(batch):
        return collate_dualseq_batch(batch, tokenizer)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)


make_dualseq_dataloader = make_dualseq_dataloader


def make_dataloader(
    tokenizer: Any,
    data_roots: list[str | Path] | None = None,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    dataset = CADSTRDatasetLoader(roots=data_roots).load()

    def collate_fn(batch):
        texts = [item.text for item in batch]
        cadstr = [item.cadstr for item in batch]
        encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        encoded_targets = tokenizer(cadstr, padding=True, truncation=True, return_tensors="pt")
        return {
            "texts": texts,
            "cadstr": cadstr,
            "cadstr_json": [item.cadstr_json for item in batch],
            "paths": [item.path for item in batch],
            "source": [item.source for item in batch],
            "input_ids": encoded_inputs["input_ids"],
            "attention_mask": encoded_inputs["attention_mask"],
            "labels": encoded_targets["input_ids"],
        }

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)

