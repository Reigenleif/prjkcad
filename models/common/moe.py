import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable


def _moe_forward(x: torch.Tensor, experts: nn.ModuleList, router: nn.Linear, top_k: int) -> torch.Tensor:
    """
    Shared token-wise MoE forward for FFN experts.

    Args:
        x: (B, T, D) input hidden states
        experts: list of FFN expert modules
        router: linear router (D -> num_experts)
        top_k: number of experts to activate per token

    Returns:
        (B, T, D) output
    """
    B, T, D = x.shape
    flat_x = x.contiguous().view(B * T, D)

    gate_logits = router(flat_x)                         # (B*T, num_experts)
    gate_weights = F.softmax(gate_logits, dim=-1)        # (B*T, num_experts)

    topk_weights, topk_indices = torch.topk(gate_weights, top_k, dim=-1)   # (B*T, top_k)
    topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)

    routed_out = torch.zeros_like(flat_x)

    for i, expert in enumerate(experts):
        mask = (topk_indices == i)
        if not mask.any():
            continue
        token_indices, k_indices = torch.where(mask)
        weight = topk_weights[token_indices, k_indices].unsqueeze(-1)
        expert_in = flat_x[token_indices]
        expert_out = expert(expert_in)
        routed_out.index_add_(0, token_indices, expert_out * weight)

    return routed_out.contiguous().view(B, T, D)


class SwitchFFN(nn.Module):
    """
    Switch Transformer FFN: 8 experts, top-1 routing.

    Args:
        make_ffn_fn: callable that returns an nn.Module FFN expert (token-wise)
        d_model: hidden dimension for the router
    """

    NUM_EXPERTS = 8
    TOP_K = 1

    def __init__(self, make_ffn_fn: Callable[[], nn.Module], d_model: int):
        super().__init__()
        self.experts = nn.ModuleList([make_ffn_fn() for _ in range(self.NUM_EXPERTS)])
        self.router = nn.Linear(d_model, self.NUM_EXPERTS, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _moe_forward(x, self.experts, self.router, self.TOP_K)


class MixtralFFN(nn.Module):
    """
    Mixtral-style FFN: 8 experts, top-2 routing.

    Args:
        make_ffn_fn: callable that returns an nn.Module FFN expert (token-wise)
        d_model: hidden dimension for the router
    """

    NUM_EXPERTS = 8
    TOP_K = 2

    def __init__(self, make_ffn_fn: Callable[[], nn.Module], d_model: int):
        super().__init__()
        self.experts = nn.ModuleList([make_ffn_fn() for _ in range(self.NUM_EXPERTS)])
        self.router = nn.Linear(d_model, self.NUM_EXPERTS, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _moe_forward(x, self.experts, self.router, self.TOP_K)
