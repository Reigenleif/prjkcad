import torch
import torch.nn as nn
import torch.nn.functional as F

class SDPADecoderLayer(nn.Module):
    def __init__(self, 
            d_model: int, 
            n_heads: int = 8, 
            dim_feedforward: int = 2048, 
            dropout: float = 0.1,
            moe_type: str = None,
            moe_conf: str = None
            ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"

        class SDPAAttention(nn.Module):
            def __init__(self, d_model, n_heads, head_dim, dropout_p=0.1):
                super().__init__()
                self.n_heads = n_heads
                self.head_dim = head_dim
                self.dropout_p = dropout_p
                self.q_proj = nn.Linear(d_model, d_model)
                self.k_proj = nn.Linear(d_model, d_model)
                self.v_proj = nn.Linear(d_model, d_model)
                self.out_proj = nn.Linear(d_model, d_model)
                
            def forward(self, query, key, value, attn_mask=None):
                B, T, D = query.shape
                _, T_key, _ = key.shape
                q = self.q_proj(query).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
                k = self.k_proj(key).view(B, T_key, self.n_heads, self.head_dim).transpose(1, 2)
                v = self.v_proj(value).view(B, T_key, self.n_heads, self.head_dim).transpose(1, 2)
                
                if attn_mask is not None:
                    if attn_mask.dtype == torch.bool:
                        float_mask = torch.zeros(attn_mask.shape, device=query.device, dtype=query.dtype)
                        float_mask.masked_fill_(attn_mask, float('-inf'))
                    else:
                        float_mask = attn_mask
                else:
                    float_mask = None
                    
                attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=float_mask, dropout_p=self.dropout_p if self.training else 0.0)
                attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
                return self.out_proj(attn_out)

        # 1. Self-attention with SDPA
        def make_self_attn():
            return SDPAAttention(d_model, n_heads, self.head_dim, dropout_p=dropout)
            
        from .moe import MoEBlock
        
        if moe_type == "MoA" and moe_conf is not None:
            self.self_attn = MoEBlock(make_self_attn, moe_conf, d_model, is_sequence_level=True)
            self.cross_attn = MoEBlock(lambda: SDPAAttention(d_model, n_heads, self.head_dim, dropout_p=dropout), moe_conf, d_model, is_sequence_level=True)
        else:
            self.self_attn = make_self_attn()
            self.cross_attn = SDPAAttention(d_model, n_heads, self.head_dim, dropout_p=dropout)

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

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # 1. Self-attention
        attn_out = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.norm1(tgt + self.dropout1(attn_out))

        # 2. Cross-attention
        if memory is not None:
            cross_out = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask)
            tgt = self.norm2(tgt + self.dropout2(cross_out))

        # 3. Feed Forward Network
        ffn_out = self.ffn(tgt)
        tgt = self.norm3(tgt + self.dropout3(ffn_out))
        return tgt


class SDPATransformerDecoder(nn.Module):
    """
    Transformer Decoder using PyTorch's native Scaled Dot-Product Attention (SDPA).
    """
    def __init__(self, d_model: int, n_heads: int = 8, n_layers: int = 4, dim_feedforward: int = 2048, max_len: int = 1024, side_embedding: nn.Module = None, moe_type: str = None, moe_conf: str = None):
        super().__init__()
        self.d_model = d_model
        self.side_embedding = side_embedding
        self.layers = nn.ModuleList([
            SDPADecoderLayer(d_model, n_heads, dim_feedforward, moe_type=moe_type, moe_conf=moe_conf)
            for _ in range(n_layers)
        ])
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, input_ids=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None):
        if inputs_embeds is not None:
            tgt_state = inputs_embeds
        elif input_ids is not None:
            if self.side_embedding is not None:
                tgt_state = self.side_embedding(input_ids)
            else:
                raise ValueError("No input embedding available for input_ids in SDPATransformerDecoder")
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
            tgt_key_padding_mask = (input_ids == pad_id).unsqueeze(1).unsqueeze(2) # Shape: B, 1, 1, T

        memory_key_padding_mask = None
        if encoder_attention_mask is not None:
            memory_key_padding_mask = (encoder_attention_mask == 0).unsqueeze(1).unsqueeze(2) # Shape: B, 1, 1, T_mem

        # Broadcast causal mask to shape: 1, 1, T, T
        full_tgt_mask = causal_mask.unsqueeze(0).unsqueeze(1)
        if tgt_key_padding_mask is not None:
            full_tgt_mask = full_tgt_mask | tgt_key_padding_mask

        x = tgt_state
        for layer in self.layers:
            x = layer(x, encoder_hidden_states, tgt_mask=full_tgt_mask, memory_mask=memory_key_padding_mask)
        return x
