from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Any

from .embeddings import build_cmd_embedding
from .encoders import PretrainedBERTEncoder
from .common import SelfAttentionBlock
from utils.representations.dual_seq.schema import get_dualseq_schema

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from utils.pipeline.config import ModelConfig


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8, dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, x: torch.Tensor, memory: torch.Tensor, memory_mask: torch.Tensor = None) -> torch.Tensor:
        B, T, D = x.shape
        _, T_mem, _ = memory.shape
        
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory).view(B, T_mem, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory).view(B, T_mem, self.n_heads, self.head_dim).transpose(1, 2)
        
        if memory_mask is not None:
            if memory_mask.dtype == torch.bool:
                memory_mask = ~memory_mask
                
        attn = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=memory_mask, 
            dropout_p=self.dropout1.p if self.training else 0.0
        )
        attn = attn.transpose(1, 2).contiguous().view(B, T, D)
        
        x = self.norm1(x + self.dropout1(self.out_proj(attn)))
        x = self.norm2(x + self.dropout2(self.ffn(x)))
        return x


class Text2CADModel(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        vocab_size: int,
        vocab_size_args: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.vocab_size_args = vocab_size_args

        self.encoder = PretrainedBERTEncoder(
            pretrained_model_name="bert-base-uncased",
            d_model=768,
        )

        if cfg.freeze_encoder:
            if hasattr(self.encoder, "encoder"):
                for param in self.encoder.encoder.parameters():
                    param.requires_grad = False
            else:
                for param in self.encoder.parameters():
                    param.requires_grad = False

        self.adaptive_layer = nn.Sequential(
            SelfAttentionBlock(d_model=768, n_heads=8, is_causal=False),
            SelfAttentionBlock(d_model=768, n_heads=8, is_causal=False),
        )

        self.downsampler = nn.Linear(768, 256)
        self.decoder_d_model = 256

        self.schema = get_dualseq_schema()
        self.pad_id = self.schema["cmd_pad_id"]
        self.sos_id = self.schema["cmd_sos_id"]
        self.eos_id = self.schema["cmd_eos_id"]
        
        self.arg_pad_id = self.schema["arg_pad_id"]
        self.arg_sos_id = self.schema["arg_sos_id"]
        self.arg_eos_id = self.schema["arg_eos_id"]

        self.max_new_cmds = cfg.max_new_cmds or 1024
        self.max_new_args = self.max_new_cmds
        self.total_vocab_size = self.vocab_size + self.vocab_size_args

        self.cmd_embedding = build_cmd_embedding(
            embedding_type=cfg.cmd_embedding_type,
            vocab_size=self.total_vocab_size,
            d_model=self.decoder_d_model,
            max_len=self.max_new_cmds,
        )

        if not cfg.is_cmd_only:
            if vocab_size_args is None:
                raise ValueError("vocab_size_args must be provided if is_cmd_only is False")
            self.arg_embedding = build_cmd_embedding(
                embedding_type=cfg.args_embedding_type,
                vocab_size=self.total_vocab_size,
                d_model=self.decoder_d_model,
                max_len=self.max_new_args,
            )

        self.decoder_self_attn = nn.ModuleList([
            SelfAttentionBlock(d_model=self.decoder_d_model, n_heads=8, is_causal=True),
            SelfAttentionBlock(d_model=self.decoder_d_model, n_heads=8, is_causal=True),
        ])

        self.decoder_cross_attn = nn.ModuleList([
            CrossAttentionBlock(d_model=self.decoder_d_model, n_heads=8)
            for _ in range(6)
        ])

        self.seq_head = nn.Linear(self.decoder_d_model, 2 * self.total_vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor = None,
        decoder_input_args: torch.Tensor = None,
        encoder_out_embeddings: torch.Tensor = None,
    ):
        if encoder_out_embeddings is not None:
            encoder_hidden_states = encoder_out_embeddings
        else:
            encoder_hidden_states = self.encoder(input_ids, attention_mask)
            encoder_hidden_states = self.adaptive_layer(encoder_hidden_states)
            encoder_hidden_states = self.downsampler(encoder_hidden_states)

        B = encoder_hidden_states.size(0)

        if decoder_input_ids is None:
            decoder_input_ids = torch.full(
                (B, 1), self.sos_id,
                device=encoder_hidden_states.device,
                dtype=torch.long,
            )

        tgt = self.cmd_embedding(decoder_input_ids)
        if not self.cfg.is_cmd_only:
            if decoder_input_args is None:
                decoder_input_args = torch.full(
                    (B, 1), self.arg_sos_id,
                    device=encoder_hidden_states.device,
                    dtype=torch.long,
                )
            shifted_args = decoder_input_args + self.vocab_size
            tgt = tgt + self.arg_embedding(shifted_args)

        for block in self.decoder_self_attn:
            tgt = block(tgt)

        memory_mask = None
        if attention_mask is not None:
            memory_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)

        for block in self.decoder_cross_attn:
            tgt = block(tgt, encoder_hidden_states, memory_mask=memory_mask)

        seq_logits = self.seq_head(tgt)
        seq_logits = seq_logits.view(B, -1, 2, self.total_vocab_size)
        seq_logits = seq_logits.permute(0, 2, 1, 3)

        return seq_logits, encoder_hidden_states
