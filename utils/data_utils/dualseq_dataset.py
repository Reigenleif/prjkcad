from __future__ import annotations

from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Optional

from utils.dual_seq import DualSeq, get_dualseq_schema
from utils.representations.dual_seq.dual_seq import DualSeqMetadata
from utils.representations.dual_seq.schema import DEFAULT_COMMANDS
from utils.representations.converter import basic_to_one_head_tokenized

class DualSeqDataset(Dataset):
    DESCRIPTION_LEVELS: tuple[str, str, str, str] = ("abstract", "beginner", "intermediate", "expert")
    
    def __init__(self, 
                 dual_seqs: list[DualSeq],
                 description_tokenizer: PreTrainedTokenizerBase,
                 description_level: str = "abstract",
                 out_type: str = "FloatArgs",
                 metadata: Optional[DualSeqMetadata] = None):
        if description_level not in self.DESCRIPTION_LEVELS:
            raise ValueError(f"Invalid description level. Choose from {self.DESCRIPTION_LEVELS}")

        for dual_seq in dual_seqs:
            if description_level not in dual_seq.descriptions:
                raise ValueError(f"DualSeq with uid {dual_seq.uid} does not have description level '{description_level}'")
        
        self.dual_seqs: list[DualSeq] = dual_seqs
        self.description_tokenizer: PreTrainedTokenizerBase = description_tokenizer
        self.description_level: str = description_level
        self.out_type = out_type
        self.metadata = metadata
        self.schema = get_dualseq_schema()

    def __len__(self) -> int:
        return len(self.dual_seqs)

    def __getitem__(self, index: int) -> tuple[str, list[str], torch.Tensor | list[int]]:
        ds = self.dual_seqs[index]
        X = ds.descriptions[self.description_level]
        y_cmds = ds.cmds
        
        if self.out_type == "FloatArgs":
            # Prepare continuous float target of shape (T_cmd, 31)
            arg_targets = []
            for i, cmd in enumerate(ds.cmds):
                row = [0.0] * 31
                if cmd in DEFAULT_COMMANDS:
                    arg_dict = ds.args[i]
                    for arg_name in DEFAULT_COMMANDS[cmd]:
                        global_idx = self.schema["arg_name_to_id"][arg_name]
                        row[global_idx] = float(arg_dict.get(arg_name, 0.0))
                arg_targets.append(row)
            y_args = torch.tensor(arg_targets, dtype=torch.float32)
            
        elif self.out_type == "EightBitBinarizedArgs":
            # Prepare discrete binned target of shape (T_cmd, 31)
            arg_targets = []
            for i, cmd in enumerate(ds.cmds):
                row = [256] * 31 # 256 is the padding value
                if cmd in DEFAULT_COMMANDS:
                    arg_dict = ds.args[i]
                    for arg_name in DEFAULT_COMMANDS[cmd]:
                        global_idx = self.schema["arg_name_to_id"][arg_name]
                        val = arg_dict.get(arg_name, 0.0)
                        if self.metadata is not None:
                            row[global_idx] = self.metadata.float_to_bin(arg_name, val)
                        else:
                            row[global_idx] = 0
                arg_targets.append(row)
            y_args = torch.tensor(arg_targets, dtype=torch.long)
            
        else: # TokenizedOneSequenceArgs
            y_args = basic_to_one_head_tokenized(ds.cmds, ds.args, self.schema)
            
        return X, y_cmds, y_args

def _make_attn_masks(batch: torch.Tensor) -> torch.Tensor:
    return (batch != 0).long()

def _command_tokenizer(cmds: list[str], dualseq_schema: dict) -> list[int]:
    command_to_id = dualseq_schema["command_to_id"]
    cmd_eos_id = dualseq_schema["cmd_eos_id"]
    tokenized_cmds: list[int] = []
    for cmd in cmds:
        if cmd not in command_to_id:
            raise ValueError(f"Unknown command: {cmd}")
        tokenized_cmds.append(int(command_to_id[cmd]))
    tokenized_cmds.append(cmd_eos_id)
    return tokenized_cmds

def _collate_fn(batch, description_tokenizer: PreTrainedTokenizerBase, dualseq_schema: dict, out_type: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    descriptions, cmds, args = zip(*batch)
    
    max_len = description_tokenizer.model_max_length
    if max_len is None:
        max_len = 512
    desc_tokens = [torch.as_tensor(description_tokenizer(desc, truncation=True, max_length=max_len)['input_ids'], dtype=torch.long) for desc in descriptions]
    
    cmd_tokens_list = [torch.as_tensor(_command_tokenizer(c, dualseq_schema), dtype=torch.long) for c in cmds]
    
    cmd_pad_id = dualseq_schema["cmd_pad_id"]
    input_ids = torch.nn.utils.rnn.pad_sequence(desc_tokens, batch_first=True, padding_value=0)
    attention_mask = _make_attn_masks(input_ids)
    
    cmd_targets = torch.nn.utils.rnn.pad_sequence(cmd_tokens_list, batch_first=True, padding_value=cmd_pad_id)
    
    if out_type == "FloatArgs":
        arg_targets = torch.nn.utils.rnn.pad_sequence(args, batch_first=True, padding_value=0.0)
    elif out_type == "EightBitBinarizedArgs":
        arg_targets = torch.nn.utils.rnn.pad_sequence(args, batch_first=True, padding_value=256)
    else: # TokenizedOneSequenceArgs
        arg_tokens_list = [torch.as_tensor(a + [dualseq_schema["arg_eos_id"]], dtype=torch.long) for a in args]
        arg_targets = torch.nn.utils.rnn.pad_sequence(arg_tokens_list, batch_first=True, padding_value=dualseq_schema["arg_pad_id"])
        
    return input_ids, cmd_targets, arg_targets, attention_mask

def create_dualseq_data_loader(dual_seqs: list[DualSeq], 
                               description_tokenizer: PreTrainedTokenizerBase, 
                               description_level: str = "abstract", 
                               batch_size: int = 32, 
                               num_workers: int = 4,
                               val_ratio: float = 0,
                               shuffle: bool = True,
                               out_type: str = "FloatArgs",
                               metadata: Optional[DualSeqMetadata] = None):
    dualseq_schema = get_dualseq_schema()
    
    def custom_collate(batch):
        return _collate_fn(batch, description_tokenizer, dualseq_schema, out_type)
        
    if val_ratio > 0:
        total_size = len(dual_seqs)
        val_size = int(total_size * val_ratio)
        train_size = total_size - val_size
        base_dataset = DualSeqDataset(dual_seqs, description_tokenizer, description_level, out_type, metadata)
        train_dataset, val_dataset = torch.utils.data.random_split(base_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=shuffle, 
                                  num_workers=num_workers,
                                  collate_fn=custom_collate)
        val_loader = DataLoader(val_dataset, 
                                batch_size=batch_size, 
                                shuffle=shuffle, 
                                num_workers=num_workers,
                                collate_fn=custom_collate)
        return train_loader, val_loader
                             
    dataset = DualSeqDataset(dual_seqs, description_tokenizer, description_level, out_type, metadata)
    loader = DataLoader(dataset, 
                        batch_size=batch_size, 
                        shuffle=shuffle, 
                        num_workers=num_workers,
                        collate_fn=custom_collate)
    return loader
