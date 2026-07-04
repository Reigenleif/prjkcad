import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import SelfAttentionBlock


# SelfAttentionBlock imported from common


# Adaptive Layer
class AdaptiveLayer(nn.Module):
    """
    Adaptive Layer for enncoders (Proposed byKhan, 2024)
    """

    VALID_TYPES = {"none", "linear", "ffn_head", "sdpa"}

    def __init__(self, adaptive_type: str, d_model: int, n_heads: int = 8):
        super().__init__()
        if adaptive_type not in self.VALID_TYPES:
            raise ValueError(
                f"Unknown adaptive_layer type {adaptive_type!r}. "
                f"Must be one of: {sorted(self.VALID_TYPES)}"
            )
        self.adaptive_type = adaptive_type

        if adaptive_type == "none":
            self.layer = nn.Identity()
        elif adaptive_type == "linear":
            self.layer = nn.Linear(d_model, d_model)
        elif adaptive_type == "ffn_head":
            # Attn's FFN like adaptive layer
            self.layer = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        elif adaptive_type == "sdpa":
            
            self.layer = nn.Sequential(
                SelfAttentionBlock(d_model, n_heads),
                SelfAttentionBlock(d_model, n_heads),
            )

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.layer(encoder_hidden_states)
