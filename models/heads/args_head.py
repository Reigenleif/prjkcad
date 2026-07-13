import torch
import torch.nn as nn

class ArgsHead(nn.Module):
    """
    Decoder prediction head for discrete argument tokens.
    """
    def __init__(self, d_model: int, vocab_size_args: int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size_args)

    def forward(self, x):
        return self.linear(x)
