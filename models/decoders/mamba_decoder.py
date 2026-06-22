import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba3
except ImportError:
    class Mamba3(nn.Module):
        """
        Fallback implementation of Mamba3 for environments without mamba_ssm.
        """
        def __init__(self, d_model: int, d_state: int, *args, **kwargs):
            super().__init__()
            self.linear = nn.Linear(d_model, d_model)
        def forward(self, x):
            return self.linear(x)

class MambaBlock(nn.Module):
    """
    Selective State Space Model (Mamba) block using Mamba3 from mamba_ssm.
    """
    def __init__(self, d_model: int, d_hidden: int = 16):
        super().__init__()
        self.mamba = Mamba3(d_model=d_model, d_state=d_hidden)

    def forward(self, x):
        return self.mamba(x)


class MambaDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1, d_hidden: int = 16, moe_type: str = None, moe_conf: str = None):
        super().__init__()
        
        # 1. Causal Mamba Block (Self-Attention alternative)
        def make_mamba():
            return MambaBlock(d_model, d_hidden=d_hidden)
            
        from .moe import MoEBlock
        
        # 2. Cross-Attention
        class MambaCrossAttention(nn.Module):
            def __init__(self, d_model, n_heads, head_dim):
                super().__init__()
                self.n_heads = n_heads
                self.head_dim = head_dim
                self.cross_q_proj = nn.Linear(d_model, d_model)
                self.cross_k_proj = nn.Linear(d_model, d_model)
                self.cross_v_proj = nn.Linear(d_model, d_model)
                self.cross_out_proj = nn.Linear(d_model, d_model)
                
            def forward(self, query, key, value, attn_mask=None):
                B, T, D = query.shape
                _, T_key, _ = key.shape
                cq = self.cross_q_proj(query).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
                ck = self.cross_k_proj(key).view(B, T_key, self.n_heads, self.head_dim).transpose(1, 2)
                cv = self.cross_v_proj(value).view(B, T_key, self.n_heads, self.head_dim).transpose(1, 2)
                
                if attn_mask is not None:
                    if attn_mask.dtype == torch.bool:
                        float_mask = torch.zeros(attn_mask.shape, device=query.device, dtype=query.dtype)
                        float_mask.masked_fill_(attn_mask, float('-inf'))
                    else:
                        float_mask = attn_mask
                else:
                    float_mask = None
                    
                cross_out = F.scaled_dot_product_attention(cq, ck, cv, attn_mask=float_mask)
                cross_out = cross_out.transpose(1, 2).contiguous().view(B, T, D)
                return self.cross_out_proj(cross_out)
                
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        def make_cross_attn():
            return MambaCrossAttention(d_model, n_heads, self.head_dim)

        if moe_type == "MoA" and moe_conf is not None:
            self.mamba = MoEBlock(make_mamba, moe_conf, d_model, is_sequence_level=True)
            self.cross_attn = MoEBlock(make_cross_attn, moe_conf, d_model, is_sequence_level=True)
        else:
            self.mamba = make_mamba()
            self.cross_attn = make_cross_attn()
            
        # 3. Feed Forward
        def make_ffn():
            return nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model)
            )
            
        if moe_type == "OnFFN" and moe_conf is not None:
            self.ffn = MoEBlock(make_ffn, moe_conf, d_model, is_sequence_level=False)
        else:
            self.ffn = make_ffn()
            
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt, memory, memory_mask=None):
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


class MambaTransformerDecoder(nn.Module):
    """
    Decoder that uses Mamba Blocks for causal feature processing and Cross-Attention to query encoder features.
    """
    def __init__(self, d_model: int, n_heads: int = 8, n_layers: int = 4, max_len: int = 1024, side_embedding: nn.Module = None, d_hidden: int = 16, moe_type: str = None, moe_conf: str = None):
        super().__init__()
        self.d_model = d_model
        self.side_embedding = side_embedding
        self.layers = nn.ModuleList([
            MambaDecoderLayer(d_model, n_heads, d_hidden=d_hidden, moe_type=moe_type, moe_conf=moe_conf)
            for _ in range(n_layers)
        ])
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
    def forward(self, input_ids=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None):
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
