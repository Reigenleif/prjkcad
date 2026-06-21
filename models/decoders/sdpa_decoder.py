import torch
import torch.nn as nn
import torch.nn.functional as F

class SDPADecoderLayer(nn.Module):
    def __init__(self, 
            d_model: int, 
            n_heads: int = 8, 
            dim_feedforward: int = 2048, 
            dropout: float = 0.1
            ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"

        # QKV projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Cross-attention projections
        self.cross_q_proj = nn.Linear(d_model, d_model)
        self.cross_k_proj = nn.Linear(d_model, d_model)
        self.cross_v_proj = nn.Linear(d_model, d_model)
        self.cross_out_proj = nn.Linear(d_model, d_model)

        # Feed Forward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        B, T, D = tgt.shape
        
        # 1. Self-attention with SDPA
        q = self.q_proj(tgt).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(tgt).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(tgt).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if tgt_mask is not None:
            if tgt_mask.dtype == torch.bool:
                float_tgt_mask = torch.zeros(tgt_mask.shape, device=tgt.device, dtype=tgt.dtype)
                float_tgt_mask.masked_fill_(tgt_mask, float('-inf'))
            else:
                float_tgt_mask = tgt_mask
        else:
            float_tgt_mask = None

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=float_tgt_mask, dropout_p=self.dropout.p if self.training else 0.0)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        tgt = self.norm1(tgt + self.dropout1(self.out_proj(attn_out)))

        # 2. Cross-attention with SDPA
        if memory is not None:
            B_mem, T_mem, _ = memory.shape
            cq = self.cross_q_proj(tgt).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            ck = self.cross_k_proj(memory).view(B, T_mem, self.n_heads, self.head_dim).transpose(1, 2)
            cv = self.cross_v_proj(memory).view(B, T_mem, self.n_heads, self.head_dim).transpose(1, 2)

            if memory_mask is not None:
                if memory_mask.dtype == torch.bool:
                    float_memory_mask = torch.zeros(memory_mask.shape, device=tgt.device, dtype=tgt.dtype)
                    float_memory_mask.masked_fill_(memory_mask, float('-inf'))
                else:
                    float_memory_mask = memory_mask
            else:
                float_memory_mask = None

            cross_out = F.scaled_dot_product_attention(cq, ck, cv, attn_mask=float_memory_mask, dropout_p=self.dropout.p if self.training else 0.0)
            cross_out = cross_out.transpose(1, 2).contiguous().view(B, T, D)
            tgt = self.norm2(tgt + self.dropout2(self.cross_out_proj(cross_out)))

        # 3. Feed Forward Network
        ffn_out = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(ffn_out))
        return tgt


class SDPATransformerDecoder(nn.Module):
    """
    Transformer Decoder using PyTorch's native Scaled Dot-Product Attention (SDPA).
    """
    def __init__(self, d_model: int, n_heads: int = 8, n_layers: int = 4, dim_feedforward: int = 2048, max_len: int = 1024, side_embedding: nn.Module = None):
        super().__init__()
        self.d_model = d_model
        self.side_embedding = side_embedding
        self.layers = nn.ModuleList([SDPADecoderLayer(d_model, n_heads, dim_feedforward) for _ in range(n_layers)])
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
