import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase
from typing import Tuple, List
from utils.dual_seq import DualSeq


class PretrainDataset(Dataset[Tuple[str, str]]):
    DESCRIPTION_LEVELS: Tuple[str, str, str, str] = ("abstract", "beginner", "intermediate", "expert")

    def __init__(self, 
                 dual_seqs: List[DualSeq],
                 description_level: str = "abstract"):
        if description_level not in self.DESCRIPTION_LEVELS:
            raise ValueError(f"Invalid description level. Choose from {self.DESCRIPTION_LEVELS}")

        for dual_seq in dual_seqs:
            if description_level not in dual_seq.descriptions:
                raise ValueError(f"DualSeq with uid {dual_seq.uid} does not have description level '{description_level}'")

        self.dual_seqs: List[DualSeq] = dual_seqs
        self.description_level: str = description_level

    def __len__(self) -> int:
        return len(self.dual_seqs)

    def __getitem__(self, index: int) -> Tuple[str, str]:
        desc = self.dual_seqs[index].descriptions[self.description_level]
        return desc, desc


def _pretrain_collate_fn(batch: List[Tuple[str, str]], 
                        tokenizer: PreTrainedTokenizerBase) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    X_list, y_list = zip(*batch)

    max_len = tokenizer.model_max_length
    if max_len is None:
        max_len = 512

    X_tokens = [torch.as_tensor(tokenizer(x, truncation=True, max_length=max_len)['input_ids'], dtype=torch.long) for x in X_list]
    y_tokens = [torch.as_tensor(tokenizer(y, truncation=True, max_length=max_len)['input_ids'], dtype=torch.long) for y in y_list]

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    input_ids = torch.nn.utils.rnn.pad_sequence(X_tokens, batch_first=True, padding_value=pad_id)
    target_ids = torch.nn.utils.rnn.pad_sequence(y_tokens, batch_first=True, padding_value=pad_id)
    attention_mask = (input_ids != pad_id).long()

    return input_ids, attention_mask, target_ids


def create_pretrain_data_loader(dual_seqs: List[DualSeq], 
                               tokenizer: PreTrainedTokenizerBase, 
                               description_level: str = "abstract", 
                               batch_size: int = 32, 
                               num_workers: int = 4,
                               val_ratio: float = 0.0,
                               shuffle: bool = True):
    def custom_collate(batch):
        return _pretrain_collate_fn(batch, tokenizer)

    if val_ratio > 0.0:
        total_size = len(dual_seqs)
        val_size = int(total_size * val_ratio)
        train_size = total_size - val_size
        base_dataset = PretrainDataset(dual_seqs, description_level)
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

    dataset = PretrainDataset(dual_seqs, description_level)
    loader = DataLoader(dataset, 
                        batch_size=batch_size, 
                        shuffle=shuffle, 
                        num_workers=num_workers,
                        collate_fn=custom_collate)
    return loader
