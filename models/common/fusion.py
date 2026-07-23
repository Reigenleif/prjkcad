import torch
import torch.nn as nn

class CmdArgsFusion(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

    def forward(self, cmd_embeds: torch.Tensor, arg_embeds: torch.Tensor) -> torch.Tensor:
        T_cmd = cmd_embeds.size(1)
        T_arg = arg_embeds.size(1)
        T = min(T_cmd, T_arg)
        cmd_trim = cmd_embeds[:, :T, :]
        arg_trim = arg_embeds[:, :T, :]
        cat_embeds = torch.cat([cmd_trim, arg_trim], dim=-1)
        return self.fusion(cat_embeds)
