from typing import Tuple, Optional
import torch
import torch.nn as nn

from .attention import SDPAttention


class CmdArgsFusion(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.cmd_cross_attn = nn.MultiheadAttention(d_model, num_heads=8)
        self.arg_cross_attn = nn.MultiheadAttention(d_model, num_heads=8)

    def forward(self, cmd_embeds: torch.Tensor, arg_embeds: torch.Tensor) -> torch.Tensor:
        T_cmd = cmd_embeds.size(1)
        T_arg = arg_embeds.size(1)
        T = min(T_cmd, T_arg)
        cmd_trim = cmd_embeds[:, :T, :]
        arg_trim = arg_embeds[:, :T, :]
        cat_embeds = torch.cat([cmd_trim, arg_trim], dim=-1)

        cmd_out = self.cmd_cross_attn(query=cmd_trim, key=arg_trim, value=arg_trim)[0]
        arg_out = self.arg_cross_attn(query=arg_trim, key=cmd_trim, value=cmd_trim)[0]

        return self.fusion(cat_embeds)


class FusionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8, dim_feedforward: int = 768, dropout: float = 0.1):
        super().__init__()
        head_dim = d_model // n_heads

        self.cmd_self_attn = SDPAttention(d_model, n_heads, head_dim, dropout_p=dropout)
        self.arg_self_attn = SDPAttention(d_model, n_heads, head_dim, dropout_p=dropout)
        self.norm_self_cmd = nn.LayerNorm(d_model)
        self.norm_self_arg = nn.LayerNorm(d_model)

        self.cmd_args_attn = SDPAttention(d_model, n_heads, head_dim, dropout_p=dropout)
        self.arg_cmd_attn = SDPAttention(d_model, n_heads, head_dim, dropout_p=dropout)
        self.norm_cross_cmd = nn.LayerNorm(d_model)
        self.norm_cross_arg = nn.LayerNorm(d_model)

        self.cmd_enc_attn = SDPAttention(d_model, n_heads, head_dim, dropout_p=dropout)
        self.arg_enc_attn = SDPAttention(d_model, n_heads, head_dim, dropout_p=dropout)
        self.norm_enc_cmd = nn.LayerNorm(d_model)
        self.norm_enc_arg = nn.LayerNorm(d_model)

        self.cmd_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.arg_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm_ffn_cmd = nn.LayerNorm(d_model)
        self.norm_ffn_arg = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        cmd_h: torch.Tensor,
        arg_h: Optional[torch.Tensor],
        encoder_hidden_states: torch.Tensor,
        enc_mask: Optional[torch.Tensor] = None,
        cmd_self_mask: Optional[torch.Tensor] = None,
        arg_self_mask: Optional[torch.Tensor] = None,
        cmd_args_mask: Optional[torch.Tensor] = None,
        arg_cmd_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        cmd_self = self.cmd_self_attn(cmd_h, cmd_h, cmd_h, attn_mask=cmd_self_mask, is_causal=(cmd_self_mask is None))
        cmd_h = self.norm_self_cmd(cmd_h + self.dropout(cmd_self))

        if arg_h is not None:
            arg_self = self.arg_self_attn(arg_h, arg_h, arg_h, attn_mask=arg_self_mask, is_causal=(arg_self_mask is None))
            arg_h = self.norm_self_arg(arg_h + self.dropout(arg_self))

        if arg_h is not None:
            cmd_cross = self.cmd_args_attn(cmd_h, arg_h, arg_h, attn_mask=cmd_args_mask)
            arg_cross = self.arg_cmd_attn(arg_h, cmd_h, cmd_h, attn_mask=arg_cmd_mask)
            cmd_h = self.norm_cross_cmd(cmd_h + self.dropout(cmd_cross))
            arg_h = self.norm_cross_arg(arg_h + self.dropout(arg_cross))

        if encoder_hidden_states is not None:
            cmd_enc = self.cmd_enc_attn(cmd_h, encoder_hidden_states, encoder_hidden_states, attn_mask=enc_mask)
            cmd_h = self.norm_enc_cmd(cmd_h + self.dropout(cmd_enc))

            if arg_h is not None:
                arg_enc = self.arg_enc_attn(arg_h, encoder_hidden_states, encoder_hidden_states, attn_mask=enc_mask)
                arg_h = self.norm_enc_arg(arg_h + self.dropout(arg_enc))

        cmd_ffn_out = self.cmd_ffn(cmd_h)
        cmd_h = self.norm_ffn_cmd(cmd_h + self.dropout(cmd_ffn_out))

        if arg_h is not None:
            arg_ffn_out = self.arg_ffn(arg_h)
            arg_h = self.norm_ffn_arg(arg_h + self.dropout(arg_ffn_out))

        return cmd_h, arg_h


class FusionStack(nn.Module):
    def __init__(self, d_model: int, n_dec_blocks: int = 6, n_heads: int = 8, dim_feedforward: int = 768, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            FusionBlock(d_model=d_model, n_heads=n_heads, dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(n_dec_blocks)
        ])

    def forward(
        self,
        cmd_h: torch.Tensor,
        arg_h: Optional[torch.Tensor],
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        cmd_input_ids: Optional[torch.Tensor] = None,
        arg_input_args: Optional[torch.Tensor] = None,
        cmd_pad_id: int = 0,
        arg_pad_id: int = 256,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = cmd_h.device
        T_cmd = cmd_h.size(1)
        T_arg = arg_h.size(1) if arg_h is not None else 0

        enc_mask = None
        if encoder_attention_mask is not None:
            enc_mask = (encoder_attention_mask == 0).unsqueeze(1).unsqueeze(2)

        cmd_pad_mask = None
        if cmd_input_ids is not None:
            cmd_trim = cmd_input_ids[:, :T_cmd]
            cmd_pad_mask = (cmd_trim == cmd_pad_id).unsqueeze(1).unsqueeze(2)

        arg_pad_mask = None
        if arg_input_args is not None and arg_h is not None:
            arg_trim = arg_input_args[:, :T_arg]
            if arg_trim.dim() == 3:
                arg_pad_mask = (arg_trim == arg_pad_id).all(dim=-1).unsqueeze(1).unsqueeze(2)
            else:
                arg_pad_mask = (arg_trim == arg_pad_id).unsqueeze(1).unsqueeze(2)

        cmd_causal = torch.triu(torch.ones(T_cmd, T_cmd, device=device, dtype=torch.bool), diagonal=1).unsqueeze(0).unsqueeze(1)
        cmd_self_mask = cmd_causal | cmd_pad_mask if cmd_pad_mask is not None else cmd_causal

        arg_self_mask = None
        cmd_args_mask = None
        arg_cmd_mask = None

        if arg_h is not None:
            arg_causal = torch.triu(torch.ones(T_arg, T_arg, device=device, dtype=torch.bool), diagonal=1).unsqueeze(0).unsqueeze(1)
            arg_self_mask = arg_causal | arg_pad_mask if arg_pad_mask is not None else arg_causal

            cross_causal_cmd_arg = torch.triu(torch.ones(T_cmd, T_arg, device=device, dtype=torch.bool), diagonal=1).unsqueeze(0).unsqueeze(1)
            cmd_args_mask = cross_causal_cmd_arg | arg_pad_mask if arg_pad_mask is not None else cross_causal_cmd_arg

            cross_causal_arg_cmd = torch.triu(torch.ones(T_arg, T_cmd, device=device, dtype=torch.bool), diagonal=1).unsqueeze(0).unsqueeze(1)
            arg_cmd_mask = cross_causal_arg_cmd | cmd_pad_mask if cmd_pad_mask is not None else cross_causal_arg_cmd

        for block in self.blocks:
            cmd_h, arg_h = block(
                cmd_h,
                arg_h,
                encoder_hidden_states=encoder_hidden_states,
                enc_mask=enc_mask,
                cmd_self_mask=cmd_self_mask,
                arg_self_mask=arg_self_mask,
                cmd_args_mask=cmd_args_mask,
                arg_cmd_mask=arg_cmd_mask,
            )
        return cmd_h, arg_h
