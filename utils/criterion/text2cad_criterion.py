import torch
from torch import nn

class Text2CADCriterion(nn.Module):
    """Criterion from the reference baseline"""

    def __init__(self, vocab_size_cmd: int, vocab_size_args: int, lambda_args: float = 1.0):
        super().__init__()
        self.vocab_size_cmd = vocab_size_cmd
        self.vocab_size_args = vocab_size_args
        self.lambda_args = lambda_args
        self.cmd_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        self.arg_loss_fn = nn.CrossEntropyLoss(ignore_index=vocab_size_cmd)

    def forward(self, cmd_logits, cmd_targets, arg_logits, arg_targets):
        B, T_max, V_cmd = cmd_logits.shape
        
        # Align cmd_targets to T_max
        if cmd_targets.size(1) < T_max:
            pad_len = T_max - cmd_targets.size(1)
            padding = torch.zeros((B, pad_len), device=cmd_targets.device, dtype=cmd_targets.dtype)
            cmd_targets = torch.cat([cmd_targets, padding], dim=1)
        elif cmd_targets.size(1) > T_max:
            cmd_targets = cmd_targets[:, :T_max]
            
        # Align arg_targets to T_max
        if arg_targets.size(1) < T_max:
            pad_len = T_max - arg_targets.size(1)
            padding = torch.zeros((B, pad_len), device=arg_targets.device, dtype=arg_targets.dtype)
            arg_targets = torch.cat([arg_targets, padding], dim=1)
        elif arg_targets.size(1) > T_max:
            arg_targets = arg_targets[:, :T_max]

        # Shift arg_targets to unified vocab space
        arg_targets_shifted = arg_targets + self.vocab_size_cmd
        
        cmd_loss = self.cmd_loss_fn(
            cmd_logits.reshape(-1, V_cmd),
            cmd_targets.reshape(-1)
        )
        arg_loss = self.arg_loss_fn(
            arg_logits.reshape(-1, arg_logits.size(-1)),
            arg_targets_shifted.reshape(-1)
        )
        return cmd_loss + self.lambda_args * arg_loss
