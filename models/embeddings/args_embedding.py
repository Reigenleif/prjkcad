import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import RotaryPositionalEncoding, SelfAttentionBlock


# Args embedding variants
class CADArgsSideEmbedding(nn.Module):
    def __init__(self, n_args: int, d_model: int = 512, max_len: int = 1024):
        super().__init__()
        self.arg_embedding = nn.Linear(n_args, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, arg_sequences: torch.Tensor) -> torch.Tensor:
        seq_len = arg_sequences.size(1)
        positions = torch.arange(seq_len, device=arg_sequences.device).unsqueeze(0)
        return self.arg_embedding(arg_sequences) + self.pos_embedding(positions)


class CADArgsRoPEEmbedding(nn.Module):
    def __init__(self, n_args: int, d_model: int = 512, max_len: int = 1024):
        super().__init__()
        self.arg_embedding = nn.Linear(n_args, d_model)
        self.rope = RotaryPositionalEncoding(d_model, max_len=max_len)

    def forward(self, arg_sequences: torch.Tensor) -> torch.Tensor:
        x = self.arg_embedding(arg_sequences)
        return self.rope(x)


class CADArgsSDPAEmbedding(nn.Module):
    def __init__(self, n_args: int, d_model: int = 512, max_len: int = 1024, n_heads: int = 8):
        super().__init__()
        self.arg_embedding = nn.Linear(n_args, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.attn_blocks = nn.Sequential(
            SelfAttentionBlock(d_model, n_heads),
            SelfAttentionBlock(d_model, n_heads),
        )

    def forward(self, arg_sequences: torch.Tensor) -> torch.Tensor:
        seq_len = arg_sequences.size(1)
        positions = torch.arange(seq_len, device=arg_sequences.device).unsqueeze(0)
        x = self.arg_embedding(arg_sequences) + self.pos_embedding(positions)
        return self.attn_blocks(x)


class CADArgsRoPESDPAEmbedding(nn.Module):
    """
    Composite between rope and sdpa
    """
    def __init__(self, n_args: int, d_model: int = 512, max_len: int = 1024, n_heads: int = 8):
        super().__init__()
        self.arg_embedding = nn.Linear(n_args, d_model)
        self.rope = RotaryPositionalEncoding(d_model, max_len=max_len)
        self.attn_blocks = nn.Sequential(
            SelfAttentionBlock(d_model, n_heads),
            SelfAttentionBlock(d_model, n_heads),
        )

    def forward(self, arg_sequences: torch.Tensor) -> torch.Tensor:
        x = self.arg_embedding(arg_sequences)
        x = self.rope(x)
        return self.attn_blocks(x)


# Factory

def build_args_embedding(
    embedding_type: str,
    n_args: int,
    d_model: int,
    max_len: int = 1024,
    n_heads: int = 8,
) -> nn.Module:
    """
    Factory for args-side embeddings.
    """
    if embedding_type == "standard":
        return CADArgsSideEmbedding(n_args, d_model, max_len)
    elif embedding_type == "rope":
        return CADArgsRoPEEmbedding(n_args, d_model, max_len)
    elif embedding_type == "sdpa":
        return CADArgsSDPAEmbedding(n_args, d_model, max_len, n_heads)
    elif embedding_type == "rope_sdpa":
        return CADArgsRoPESDPAEmbedding(n_args, d_model, max_len, n_heads)
    else:
        raise ValueError(
            f"Unknown args embedding_type {embedding_type!r}. "
            "Must be one of: 'standard', 'rope', 'sdpa', 'rope_sdpa'."
        )