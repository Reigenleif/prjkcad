import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaBlock(nn.Module):
    """
    A pure-PyTorch lightweight selective State Space Model (Mamba) block.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.d_conv = d_conv
        
        dt_rank = d_model // 16 if d_model >= 16 else 1

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.x_proj = nn.Linear(self.d_inner, dt_rank + d_state + d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)

        A_log = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1))
        self.A_log = nn.Parameter(A_log)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        
        projected = self.in_proj(x)
        x_inner, res = projected.chunk(2, dim=-1)

        x_inner = x_inner.transpose(1, 2)
        x_inner = self.conv1d(x_inner)[:, :, :L]
        x_inner = x_inner.transpose(1, 2)
        
        x_inner = F.silu(x_inner)

        ssm_params = self.x_proj(x_inner)
        dt_rank = self.d_model // 16 if self.d_model >= 16 else 1
        dt, B_ssm, C_ssm = torch.split(ssm_params, [dt_rank, self.d_state, self.d_state], dim=-1)

        dt = F.softplus(self.dt_proj(dt))
        
        A = -torch.exp(self.A_log)
        
        # Causal SSM Scan loop
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        y = torch.zeros(B, L, self.d_inner, device=x.device, dtype=x.dtype)
        
        for t in range(L):
            dt_t = dt[:, t, :].unsqueeze(-1)
            A_bar = torch.exp(dt_t * A.unsqueeze(0))
            B_t = B_ssm[:, t, :].unsqueeze(1)
            B_bar = dt_t * B_t
            
            x_t = x_inner[:, t, :].unsqueeze(-1)
            h = A_bar * h + B_bar * x_t
            
            C_t = C_ssm[:, t, :].unsqueeze(-1)
            y[:, t, :] = (h @ C_t).squeeze(-1)

        y = y + x_inner * self.D.unsqueeze(0).unsqueeze(0)
        out = y * F.silu(res)
        return self.out_proj(out)


class MambaDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.mamba = MambaBlock(d_model)
        
        # Cross-attention to attend to encoder hidden states
        self.cross_q_proj = nn.Linear(d_model, d_model)
        self.cross_k_proj = nn.Linear(d_model, d_model)
        self.cross_v_proj = nn.Linear(d_model, d_model)
        self.cross_out_proj = nn.Linear(d_model, d_model)
        
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt, memory, memory_mask=None):
        # 1. Self-attention replacement: Causal Mamba Block
        mamba_out = self.mamba(tgt)
        tgt = self.norm1(tgt + mamba_out)
        
        # 2. Cross-attention
        if memory is not None:
            B, T, D = tgt.shape
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
                
            cross_out = F.scaled_dot_product_attention(cq, ck, cv, attn_mask=float_memory_mask)
            cross_out = cross_out.transpose(1, 2).contiguous().view(B, T, D)
            tgt = self.norm2(tgt + self.cross_out_proj(cross_out))
            
        # 3. FFN
        ffn_out = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = self.norm3(tgt + ffn_out)
        return tgt


class MambaTransformerDecoder(nn.Module):
    """
    Decoder that uses Mamba Blocks for causal feature processing and Cross-Attention to query encoder features.
    """
    def __init__(self, d_model: int, n_heads: int = 8, n_layers: int = 4, max_len: int = 1024, side_embedding: nn.Module = None):
        super().__init__()
        self.d_model = d_model
        self.side_embedding = side_embedding
        self.layers = nn.ModuleList([MambaDecoderLayer(d_model, n_heads) for _ in range(n_layers)])
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
