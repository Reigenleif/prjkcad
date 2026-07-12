import torch
import torch.nn as nn
import torch.nn.functional as F

class SDPAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, head_dim: int, dropout_p: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.dropout_p = dropout_p
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        B, T, D = query.shape
        _, T_key, _ = key.shape
        q = self.q_proj(query).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, T_key, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, T_key, self.n_heads, self.head_dim).transpose(1, 2)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_mask = ~attn_mask
            is_causal = False

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(attn_out)


class SelfAttentionBlock(nn.Module):
    """
    Designed only for Self-Attention
    """
    def __init__(self, d_model: int, n_heads: int = 8, is_causal: bool = False):
        super().__init__()
        self.is_causal = is_causal
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)
        attn = attn.transpose(1, 2).contiguous().view(B, T, D)
        x = self.norm1(x + self.out_proj(attn))
        x = self.norm2(x + self.ffn(x))
        return x
