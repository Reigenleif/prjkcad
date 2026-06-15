from __future__ import annotations

import torch
from torch import nn
from transformers import T5EncoderModel




class T5TorchTorch(nn.Module):
    """
    T5 encoder + Single Torch Transformer decoder for text-to-DualSeq training.
    Predicts both CMD and Args simultaneously using dual heads.
    """

    def __init__(
        self,
        t5_encoder: T5EncoderModel,
        d_model: int,
        n_cmds: int,
        n_args: int,
        n_layers: int = 4,
        n_heads: int = 8,
        max_len: int = 1024,
    ):
        super().__init__()

        self.encoder = t5_encoder
        self.cmd_vocab_size = n_cmds + 3
        self.pad_id = n_cmds + 1
        self.sos_id = n_cmds
        self.eos_id = n_cmds + 2

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
        )

        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.dec_embedding = nn.Embedding(self.cmd_vocab_size, d_model)
        self.arg_embedding = nn.Linear(n_args, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)

        self.cmd_head = nn.Linear(d_model, self.cmd_vocab_size)
        self.arg_head = nn.Linear(d_model, n_args)

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)

    def _default_args(self, batch_size: int, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch_size, seq_len, self.arg_head.out_features, device=device, dtype=dtype)

    def forward(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids,
        decoder_input_args=None,
        decoder_attention_mask=None,
        side_input_ids=None,
        side_input_args=None,
    ):
        enc_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

        if decoder_attention_mask is None:
            decoder_attention_mask = torch.ones_like(decoder_input_ids)

        if decoder_input_args is None:
            decoder_input_args = self._default_args(
                decoder_input_ids.size(0),
                decoder_input_ids.size(1),
                decoder_input_ids.device,
                enc_out.dtype,
            )

        positions = torch.arange(decoder_input_ids.size(1), device=decoder_input_ids.device)
        pos_embed = self.pos_embedding(positions)[None, :, :]

        decoder_state = self.dec_embedding(decoder_input_ids) + self.arg_embedding(decoder_input_args) + pos_embed

        padding_mask = decoder_attention_mask == 0
        memory_padding_mask = attention_mask == 0

        dec_out = self.decoder(
            tgt=decoder_state,
            memory=enc_out,
            tgt_mask=self._causal_mask(decoder_input_ids.size(1), decoder_input_ids.device),
            tgt_key_padding_mask=padding_mask,
            memory_key_padding_mask=memory_padding_mask,
        )

        cmd_logits = self.cmd_head(dec_out)
        arg_preds = self.arg_head(dec_out)
        return cmd_logits, arg_preds
