import torch
import torch.nn as nn
import torch.nn.functional as F

class MoEBlock(nn.Module):
    """
    A reusable Mixture of Experts (MoE) block that routes inputs (hidden states)
    to top-k selected experts using a direct softmax router.
    """
    def __init__(self, make_expert_fn, moe_conf: str, d_model: int, is_sequence_level: bool = False):
        super().__init__()
        self.is_sequence_level = is_sequence_level
        self.d_model = d_model
        
        # Determine num_experts and top_k based on moe_conf
        if moe_conf == "Switch":
            self.num_experts = 8
            self.top_k = 1
        elif moe_conf == "Mixtral":
            self.num_experts = 8
            self.top_k = 2
        # Disable DeepSeek configuration for now
        # elif moe_conf == "DeepSeek":
        #     self.num_experts = 256
        #     self.top_k = 8
        else:
            raise ValueError(f"Unknown MoE config: {moe_conf}")
            
        # Instantiate experts
        self.experts = nn.ModuleList([make_expert_fn() for _ in range(self.num_experts)])
        
        # Router: Linear projection from hidden state dimension to number of experts
        self.router = nn.Linear(d_model, self.num_experts, bias=False)

    def forward(self, x, *args, **kwargs):
        # x shape: (B, T, D)
        B, T, D = x.shape
        flat_x = x.contiguous().view(B * T, D)
        
        # Compute router logits and routing weights using softmax
        gate_logits = self.router(flat_x)  # (B * T, num_experts)
        gate_weights = F.softmax(gate_logits, dim=-1)  # (B * T, num_experts)
        
        # Select top-k experts
        topk_weights, topk_indices = torch.topk(gate_weights, self.top_k, dim=-1)  # (B * T, top_k)
        
        # Normalize top-k weights so they sum to 1.0 for each token
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)
        
        # Prepare output tensor
        routed_out = torch.zeros_like(flat_x)
        
        # Process active experts
        for i in range(self.num_experts):
            mask = (topk_indices == i)
            if not mask.any():
                continue
            
            # Find the token indices (row indices in flat_x) and their top-k column rank indices
            token_indices, k_indices = torch.where(mask)
            weight = topk_weights[token_indices, k_indices].unsqueeze(-1)
            
            if self.is_sequence_level:
                # Sequence-level experts (Attention/Mamba) run on the full sequence, then we slice
                expert_out = self.experts[i](x, *args, **kwargs)
                flat_expert_out = expert_out.contiguous().view(B * T, D)[token_indices]
                routed_out.index_add_(0, token_indices, flat_expert_out * weight)
            else:
                # Token-wise experts (FFN) run directly on the selected tokens' hidden states
                expert_in = flat_x[token_indices]
                expert_out = self.experts[i](expert_in)
                routed_out.index_add_(0, token_indices, expert_out * weight)
                
        return routed_out.contiguous().view(B, T, D)
