import math
import torch
import torch.nn as nn

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))

    def _get_sincos(self, seq_len: int, device: torch.device):
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return torch.sin(emb), torch.cos(emb)

    def _rotate_half(self, x: torch.Tensor):
        x1 = x[..., :self.d_model // 2]
        x2 = x[..., self.d_model // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        sin, cos = self._get_sincos(seq_len, x.device)
        sin = sin.unsqueeze(0)   # (1, T, D)
        cos = cos.unsqueeze(0)   # (1, T, D)
        return x * cos + self._rotate_half(x) * sin
