import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable

from ..common import SwitchFFN, MixtralFFN


# Module-level helpers
def _make_ffn(d_model: int, dim_feedforward: int, dropout: float) -> nn.Sequential:
    """Returns a standard FFN block (Linear → ReLU → Dropout → Linear)."""
    return nn.Sequential(
        nn.Linear(d_model, dim_feedforward),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_feedforward, d_model),
    )


def _build_ffn_module(
    d_model: int,
    dim_feedforward: int,
    dropout: float,
    moe_conf: str,
) -> nn.Module:
    make_fn: Callable[[], nn.Module] = lambda: _make_ffn(d_model, dim_feedforward, dropout)

    if moe_conf == "Switch":
        return SwitchFFN(make_fn, d_model)
    elif moe_conf == "Mixtral":
        return MixtralFFN(make_fn, d_model)
    else:
        return make_fn()


class MHAWrapper(nn.Module):
    def __init__(self, mha: nn.MultiheadAttention):
        super().__init__()
        self.mha = mha

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        out, _ = self.mha(query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return out

# Decoder Layer
class TorchTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        moe_conf: str = None,
    ):
        super().__init__()

        self.self_attn = MHAWrapper(
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        )
        self.cross_attn = MHAWrapper(
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        )
        self.ffn = _build_ffn_module(d_model, dim_feedforward, dropout, moe_conf)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # 1. Self Attention
        attn_out = self.self_attn(
            tgt, tgt, tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        tgt = self.norm1(tgt + self.dropout1(attn_out))

        # 2. Cross Attention
        if memory is not None:
            attn_out = self.cross_attn(
                tgt, memory, memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
            )
            tgt = self.norm2(tgt + self.dropout2(attn_out))

        # 3. FFN
        ffn_out = self.ffn(tgt)
        tgt = self.norm3(tgt + self.dropout3(ffn_out))
        return tgt


# Transformer Decoder
class TorchTransformerDecoder(nn.Module):
    """
    PyTorch Built in
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_layers: int = 4,
        max_len: int = 1024,
        side_embedding: nn.Module = None,
        moe_conf: str = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.side_embedding = side_embedding

        if moe_conf is not None:
            self.layers = nn.ModuleList([
                TorchTransformerDecoderLayer(
                    d_model=d_model, n_heads=n_heads, dim_feedforward=2048,
                    dropout=dropout, moe_conf=moe_conf,
                )
                for _ in range(n_layers)
            ])
            self.decoder = None
        else:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
            self.layers = None

        self.pos_embedding = nn.Embedding(max_len, d_model)

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)

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
                raise ValueError("No input embedding available for input_ids in TorchTransformerDecoder")
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        seq_len = tgt_state.size(1)
        positions = torch.arange(seq_len, device=tgt_state.device)
        pos_embed = self.pos_embedding(positions)[None, :, :]
        tgt_state = tgt_state + pos_embed

        causal_mask = self._causal_mask(seq_len, tgt_state.device)

        tgt_key_padding_mask = None
        if input_ids is not None:
            pad_id = getattr(self.side_embedding, "pad_id", 0)
            tgt_key_padding_mask = (input_ids == pad_id)

        memory_key_padding_mask = None
        if encoder_attention_mask is not None:
            memory_key_padding_mask = (encoder_attention_mask == 0)

        if self.decoder is not None:
            dec_out = self.decoder(
                tgt=tgt_state,
                memory=encoder_hidden_states,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        else:
            x = tgt_state
            for layer in self.layers:
                x = layer(
                    tgt=x,
                    memory=encoder_hidden_states,
                    tgt_mask=causal_mask,
                    memory_mask=None,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                )
            dec_out = x
        return dec_out
