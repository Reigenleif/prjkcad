import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable

from ..common import SDPAttention, SwitchFFN, MixtralFFN

import sys
import types

# Mamba3 uses Triton kernel, this raises an error and somehow below script fixes it
try:
    from mamba_ssm import Mamba3
except ImportError as e:
    if "selective_scan_cuda" in str(e):
        if "selective_scan_cuda" not in sys.modules:
            sys.modules["selective_scan_cuda"] = types.ModuleType("selective_scan_cuda")
        from mamba_ssm import Mamba3
    else:
        raise e

# Module-level components
class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_hidden: int = 16):
        super().__init__()
        self.mamba = Mamba3(d_model=d_model, d_state=d_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mamba(x)


# SDPAttention imported from common
def _make_mamba_ffn(d_model: int, dim_feedforward: int, dropout: float) -> nn.Sequential:
    """Returns a standard FFN block for Mamba decoder layers."""
    return nn.Sequential(
        nn.Linear(d_model, dim_feedforward),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_feedforward, d_model),
    )


def _build_mamba_ffn_module(
    d_model: int,
    dim_feedforward: int,
    dropout: float,
    moe_conf: str,
) -> nn.Module:
    """
    For MoE
    """
    make_fn: Callable[[], nn.Module] = lambda: _make_mamba_ffn(d_model, dim_feedforward, dropout)

    if moe_conf == "Switch":
        return SwitchFFN(make_fn, d_model)
    elif moe_conf == "Mixtral":
        return MixtralFFN(make_fn, d_model)
    else:
        return make_fn()


# Decoder Layer
class MambaDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        d_hidden: int = 16,
        moe_conf: str = None,
    ):
        super().__init__()

        # 1. Causal Mamba Block
        self.mamba = MambaBlock(d_model, d_hidden=d_hidden)

        # 2. Cross-Attention
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.cross_attn = SDPAttention(d_model, n_heads, self.head_dim, dropout_p=dropout)

        # 3. Feed Forward
        self.ffn = _build_mamba_ffn_module(d_model, dim_feedforward, dropout, moe_conf)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # 1. Causal Mamba Block
        mamba_out = self.mamba(tgt)
        tgt = self.norm1(tgt + mamba_out)

        # 2. Cross-attention
        if memory is not None:
            cross_out = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask)
            tgt = self.norm2(tgt + cross_out)

        # 3. FFN
        ffn_out = self.ffn(tgt)
        tgt = self.norm3(tgt + ffn_out)
        return tgt


# Transformer Decoder
class MambaTransformerDecoder(nn.Module):
    """
    Decoder that uses Mamba Blocks for causal feature processing and Cross-Attention
    to query encoder features.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_layers: int = 4,
        max_len: int = 1024,
        side_embedding: nn.Module = None,
        d_hidden: int = 16,
        moe_conf: str = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.side_embedding = side_embedding
        self.layers = nn.ModuleList([
            MambaDecoderLayer(
                d_model, n_heads,
                d_hidden=d_hidden, moe_conf=moe_conf, dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(
        self,
        input_ids: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        encoder_hidden_states: torch.Tensor = None,
        encoder_attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            tgt_state = inputs_embeds
        elif input_ids is not None:
            if self.side_embedding is not None:
                tgt_state = self.side_embedding(input_ids)
            else:
                raise ValueError("No input embedding available for input_ids in MambaTransformerDecoder")
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        seq_len = tgt_state.size(1)
        positions = torch.arange(seq_len, device=tgt_state.device)
        pos_embed = self.pos_embedding(positions)[None, :, :]
        tgt_state = tgt_state + pos_embed

        memory_key_padding_mask = None
        if encoder_attention_mask is not None:
            memory_key_padding_mask = (encoder_attention_mask == 0).unsqueeze(1).unsqueeze(2)

        x = tgt_state
        for layer in self.layers:
            x = layer(x, encoder_hidden_states, memory_mask=memory_key_padding_mask)
        return x
