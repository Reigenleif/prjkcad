from ..dual_seq import DualSeq, get_dualseq_schema
from torch.utils.data import Dataset, DataLoader
import torch


class DualSeqCmdonlyDataset(Dataset) :
    DESCRIPTION_LEVELS = ["abstract", "beginner", "intermediate", "expert"]
    
    def __init__(self, 
                 dual_seqs: list[DualSeq],
                 description_tokenizer,
                 description_level: str = "abstract"):
        # description_level validity
        if description_level not in self.DESCRIPTION_LEVELS :
            raise ValueError(f"Invalid description level. Choose from {self.DESCRIPTION_LEVELS}")

        for dual_seq in dual_seqs :
            if description_level not in dual_seq.descriptions :
                raise ValueError(f"DualSeq with uid {dual_seq.uid} does not have description level '{description_level}'")
        
        self.dual_seqs = dual_seqs
        self.description_tokenizer = description_tokenizer
        self.description_level = description_level

    def __len__(self) -> int:
        return len(self.dual_seqs)

    def __getitem__(self, index: int) -> DualSeq:
        X = self.dual_seqs[index].descriptions[self.description_level]
        y = self.dual_seqs[index].cmds
        return X, y
    

def _make_attn_masks(batch) :
    return (batch != 0).long()

def _collate_fn(batch, description_tokenizer=None, cmd_tokenizer=None) :
    descriptions, cmds = zip(*batch)
    if description_tokenizer is None:
        raise ValueError("description_tokenizer is required for collate")

    if cmd_tokenizer is None:
        cmd_tokenizer = description_tokenizer
    
    desc_tokens = [torch.as_tensor(description_tokenizer(desc)['input_ids'], dtype=torch.long) for desc in descriptions]
    cmd_tokens = [torch.as_tensor(cmd_tokenizer(cmd), dtype=torch.long) for cmd in cmds]

    desc_tokens = torch.nn.utils.rnn.pad_sequence(desc_tokens, batch_first=True, padding_value=0)
    cmd_tokens = torch.nn.utils.rnn.pad_sequence(cmd_tokens, batch_first=True, padding_value=0)
    attn_mask = _make_attn_masks(desc_tokens)

    return desc_tokens, cmd_tokens, attn_mask

def _command_tokenizer(cmds, dualseq_schema) :
    tokenized_cmds = []
    for cmd in cmds :
        if cmd not in dualseq_schema["command_to_id"] :
            raise ValueError(f"Unknown command: {cmd}")
        tokenized_cmds.append(dualseq_schema["command_to_id"][cmd])
    return tokenized_cmds
    
    
    
def create_cmdonly_data_loader(dual_seqs: list[DualSeq], 
                       description_tokenizer, 
                       description_level: str = "abstract", 
                       batch_size: int = 32, 
                       num_workers: int = 4,
                       val_ratio: float = 0,
                       shuffle: bool = True) -> DataLoader:
    """Creates a DataLoader for the CMDOnly dataset.
    If val_ratio > 0, it will split the dataset into train and validation sets and return two DataLoaders.
    else, it will return a single DataLoader for the entire dataset.
    """
    
    
    if val_ratio > 0 :
        # Shuffle and split the dataset into train and validation sets
        total_size = len(dual_seqs)
        val_size = int(total_size * val_ratio)
        train_size = total_size - val_size
        train_dual_seqs, val_dual_seqs = torch.utils.data.random_split(dual_seqs, [train_size, val_size])
    
        train_dataset = DualSeqCmdonlyDataset(train_dual_seqs, description_tokenizer, description_level)
        val_dataset = DualSeqCmdonlyDataset(val_dual_seqs, description_tokenizer, description_level)
        
        dualseq_schema = get_dualseq_schema()
        train_loader = DataLoader(train_dataset, 
                      batch_size=batch_size, 
                      shuffle=shuffle, 
                      num_workers=num_workers,
                      collate_fn=lambda batch: _collate_fn(batch, description_tokenizer, cmd_tokenizer=lambda cmds: _command_tokenizer(cmds, dualseq_schema)))
        val_loader = DataLoader(val_dataset, 
                      batch_size=batch_size, 
                      shuffle=shuffle, 
                      num_workers=num_workers,
                      collate_fn=lambda batch: _collate_fn(batch, description_tokenizer, cmd_tokenizer=lambda cmds: _command_tokenizer(cmds, dualseq_schema)))
        return train_loader, val_loader
                           
                                
    
    dataset = DualSeqCmdonlyDataset(dual_seqs, description_tokenizer, description_level)
    
    dualseq_schema = get_dualseq_schema()
    return DataLoader(dataset, 
                      batch_size=batch_size, 
                      shuffle=shuffle, 
                      num_workers=num_workers,
                      collate_fn=lambda batch: _collate_fn(batch, description_tokenizer, cmd_tokenizer=lambda cmds: _command_tokenizer(cmds, dualseq_schema)))
