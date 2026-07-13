import torch
import torch.nn as nn

class ArgsHead(nn.Module):
    """
    Decoder prediction head for continuous argument values.
    """
    def __init__(self, d_model: int, n_args: int):
        super().__init__()
        self.linear = nn.Linear(d_model, n_args)

    def forward(self, x):
        return self.linear(x)
