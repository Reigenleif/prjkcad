import math
import torch
import torch.nn as nn
import torch.nn.functional as F


from ..common import RotaryPositionalEncoding, SelfAttentionBlock


# Cmd embedding variants
class CADCmdSideEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, max_len: int = 1024):
        super().__init__()
        self.cmd_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, cmd_input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = cmd_input_ids.size(1)
        positions = torch.arange(seq_len, device=cmd_input_ids.device).unsqueeze(0)
        return self.cmd_embedding(cmd_input_ids) + self.pos_embedding(positions)


class CADCmdRoPEEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, max_len: int = 1024):
        super().__init__()
        self.cmd_embedding = nn.Embedding(vocab_size, d_model)
        self.rope = RotaryPositionalEncoding(d_model, max_len=max_len)

    def forward(self, cmd_input_ids: torch.Tensor) -> torch.Tensor:
        x = self.cmd_embedding(cmd_input_ids)
        return self.rope(x)


class CADCmdSDPAEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, max_len: int = 1024, n_heads: int = 8):
        super().__init__()
        self.cmd_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.attn_blocks = nn.Sequential(
            SelfAttentionBlock(d_model, n_heads, is_causal=True),
            SelfAttentionBlock(d_model, n_heads, is_causal=True),
        )

    def forward(self, cmd_input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = cmd_input_ids.size(1)
        positions = torch.arange(seq_len, device=cmd_input_ids.device).unsqueeze(0)
        x = self.cmd_embedding(cmd_input_ids) + self.pos_embedding(positions)
        return self.attn_blocks(x)


class CADCmdRoPESDPAEmbedding(nn.Module):
    """
    Composite between rope and sdpa
    """
    def __init__(self, vocab_size: int, d_model: int = 512, max_len: int = 1024, n_heads: int = 8):
        super().__init__()
        self.cmd_embedding = nn.Embedding(vocab_size, d_model)
        self.rope = RotaryPositionalEncoding(d_model, max_len=max_len)
        self.attn_blocks = nn.Sequential(
            SelfAttentionBlock(d_model, n_heads, is_causal=True),
            SelfAttentionBlock(d_model, n_heads, is_causal=True),
        )

    def forward(self, cmd_input_ids: torch.Tensor) -> torch.Tensor:
        x = self.cmd_embedding(cmd_input_ids)
        x = self.rope(x)
        return self.attn_blocks(x)


# Factory Function

def build_cmd_embedding(
    embedding_type: str,
    vocab_size: int,
    d_model: int,
    max_len: int = 1024,
    n_heads: int = 8,
) -> nn.Module:
    """
    Factory for command-side embeddings.

    Args:
        embedding_type: one of "standard", "rope", "sdpa", "rope_sdpa"
        vocab_size: vocabulary size
        d_model: model hidden dimension
        max_len: maximum sequence length
        n_heads: number of attention heads (used by sdpa variants)

    Returns:
        nn.Module embedding
    """
    if embedding_type == "standard":
        return CADCmdSideEmbedding(vocab_size, d_model, max_len)
    elif embedding_type == "rope":
        return CADCmdRoPEEmbedding(vocab_size, d_model, max_len)
    elif embedding_type == "sdpa":
        return CADCmdSDPAEmbedding(vocab_size, d_model, max_len, n_heads)
    elif embedding_type == "rope_sdpa":
        return CADCmdRoPESDPAEmbedding(vocab_size, d_model, max_len, n_heads)
    else:
        raise ValueError(
            f"Unknown cmd embedding_type {embedding_type!r}. "
            "Must be one of: 'standard', 'rope', 'sdpa', 'rope_sdpa'."
        )
