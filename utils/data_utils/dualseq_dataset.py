from ..dual_seq import DualSeq, get_dualseq_schema
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from transformers import PreTrainedTokenizerBase
from typing_extensions import override

class DualSeqDataset(Dataset[tuple[str, list[str], list[int]]]):
    DESCRIPTION_LEVELS: tuple[str, str, str, str] = ("abstract", "beginner", "intermediate", "expert")
    
    def __init__(self, 
                 dual_seqs: list[DualSeq],
                 description_tokenizer: PreTrainedTokenizerBase,
                 description_level: str = "abstract"):
        if description_level not in self.DESCRIPTION_LEVELS:
            raise ValueError(f"Invalid description level. Choose from {self.DESCRIPTION_LEVELS}")

        for dual_seq in dual_seqs:
            if description_level not in dual_seq.descriptions:
                raise ValueError(f"DualSeq with uid {dual_seq.uid} does not have description level '{description_level}'")
        
        self.dual_seqs: list[DualSeq] = dual_seqs
        self.description_tokenizer: PreTrainedTokenizerBase = description_tokenizer
        self.description_level: str = description_level

    def __len__(self) -> int:
        return len(self.dual_seqs)

    @override
    def __getitem__(self, index: int) -> tuple[str, list[str], list[int]]:
        X = self.dual_seqs[index].descriptions[self.description_level]
        y_cmds = self.dual_seqs[index].cmds
        y_args = self.dual_seqs[index].args
        return X, y_cmds, y_args

def _make_attn_masks(batch: torch.Tensor) -> torch.Tensor:
    return (batch != 0).long()

def _command_and_args_tokenizer(cmds: list[str], args: list[int], dualseq_schema: dict) -> tuple[list[int], list[int]]:
    command_to_id = dualseq_schema["command_to_id"]
    cmd_eos_id = dualseq_schema["cmd_eos_id"]
    arg_eos_id = dualseq_schema["arg_eos_id"]

    tokenized_cmds: list[int] = []
    
    for cmd in cmds:
        if cmd not in command_to_id:
            raise ValueError(f"Unknown command: {cmd}")
        tokenized_cmds.append(int(command_to_id[cmd]))

    tokenized_cmds.append(cmd_eos_id)
    tokenized_args = list(args) + [arg_eos_id]
    
    return tokenized_cmds, tokenized_args

def _collate_fn(batch: list[tuple[str, list[str], list[int]]], 
                description_tokenizer: PreTrainedTokenizerBase, 
                dualseq_schema: dict) -> dict[str, torch.Tensor]:
    descriptions, cmds, args = zip(*batch)
    
    max_len = description_tokenizer.model_max_length
    if max_len is None :
        max_len = 512
    desc_tokens = [torch.as_tensor(description_tokenizer(desc, truncation=True, max_length=max_len)['input_ids'], dtype=torch.long) for desc in descriptions]
    
    cmd_tokens_list = []
    arg_tokens_list = []
    for c, a in zip(cmds, args):
        c_tok, a_tok = _command_and_args_tokenizer(c, a, dualseq_schema)
        cmd_tokens_list.append(torch.as_tensor(c_tok, dtype=torch.long))
        arg_tokens_list.append(torch.as_tensor(a_tok, dtype=torch.long))

    cmd_pad_id = dualseq_schema["cmd_pad_id"]
    arg_pad_id = dualseq_schema["arg_pad_id"]
    
    input_ids = torch.nn.utils.rnn.pad_sequence(desc_tokens, batch_first=True, padding_value=0)
    attention_mask = _make_attn_masks(input_ids)
    
    cmd_targets = torch.nn.utils.rnn.pad_sequence(cmd_tokens_list, batch_first=True, padding_value=cmd_pad_id)
    arg_targets = torch.nn.utils.rnn.pad_sequence(arg_tokens_list, batch_first=True, padding_value=arg_pad_id)
        
    return input_ids, cmd_targets, arg_targets, attention_mask

def create_dualseq_data_loader(dual_seqs: list[DualSeq], 
                       description_tokenizer: PreTrainedTokenizerBase, 
                       description_level: str = "abstract", 
                       batch_size: int = 32, 
                       num_workers: int = 4,
                       val_ratio: float = 0,
                       shuffle: bool = True):
    
    dualseq_schema = get_dualseq_schema()
    
    def custom_collate(batch):
        return _collate_fn(batch, description_tokenizer, dualseq_schema)
        
    if val_ratio > 0:
        total_size = len(dual_seqs)
        val_size = int(total_size * val_ratio)
        train_size = total_size - val_size
        base_dataset = DualSeqDataset(dual_seqs, description_tokenizer, description_level)
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
                             
    dataset = DualSeqDataset(dual_seqs, description_tokenizer, description_level)
    loader = DataLoader(dataset, 
                      batch_size=batch_size, 
                      shuffle=shuffle, 
                      num_workers=num_workers,
                      collate_fn=custom_collate)
    return loader
