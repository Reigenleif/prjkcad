from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModelForSeq2SeqLM
import copy
from .components import CADCmdSideEmbedding, CADArgsSideEmbedding

class T5T5T5(nn.Module):
    """
    Model that consists of:
    - T5 Encoder
    - T5 Decoder (headless, driven via inputs_embeds)
    - CMD CAD head
    - Arg CAD head
    for text-to-DualSeq training (both command and argument sequences).

    The decoder receives a combined embedding of (cmd_embed + arg_embed + pos_embed)
    as `inputs_embeds`, bypassing the T5 token vocabulary entirely.
    """

    def __init__(
        self,
        pretrained_model_name: str,
        vocab_size: int,
        n_args: int,
        max_new_cmds: int = 1024,
    ):
        super().__init__()

        pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name)
        self.pretrained_model_name = pretrained_model_name
        self.encoder = pretrained_model.get_encoder()
        self.cmd_decoder = pretrained_model.get_decoder()
        self.arg_decoder = copy.deepcopy(self.cmd_decoder)

        d_model = self.encoder.config.d_model

        self.vocab_size = vocab_size
        self.max_new_cmds = max_new_cmds

        # CAD-domain decoder input embeddings
        self.cmd_embedding = CADCmdSideEmbedding(
            vocab_size=self.vocab_size,
            d_model=d_model,
            max_len=max_new_cmds
        )
        self.arg_embedding = CADArgsSideEmbedding(
            n_args=n_args,
            d_model=d_model,
            max_len=max_new_cmds
        )

        self.cmd_decoder.set_input_embeddings(self.cmd_embedding)

        # Prediction heads
        self.cmd_head = nn.Linear(d_model, self.vocab_size)
        self.arg_head = nn.Linear(d_model, n_args)

        # Hard coded spec tokens
        self.pad_id = 0
        self.sos_id = 1
        self.eos_id = 2
        

    def _default_args(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros(batch_size, seq_len, self.arg_head.out_features, device=device, dtype=dtype)

    def forward(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids=None,
        decoder_input_args=None,
        encoder_out_embeddings=None
    ):
        # PATH 1 : encoder out already computed and passed in
        if encoder_out_embeddings is not None:
            encoder_hidden_states = encoder_out_embeddings
            B = encoder_hidden_states.size(0)

            # Default decoder inputs
            if decoder_input_ids is None:
                decoder_input_ids = torch.full(
                    (B, 1), self.sos_id, device=encoder_hidden_states.device, dtype=torch.long
                )
            T_dec = decoder_input_ids.size(1)
            if decoder_input_args is None:
                decoder_input_args = self._default_args(
                    B, T_dec, encoder_hidden_states.device, encoder_hidden_states.dtype
                )

            # CMD decoder
            cmd_decoder_out = self.cmd_decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=attention_mask,
            )
            cmd_hidden = cmd_decoder_out.last_hidden_state  # (B, T_dec, d)

            # Arg decoder
            arg_embeds = self.arg_embedding(decoder_input_args)  # (B, T_dec, d)
            arg_decoder_out = self.arg_decoder(
                inputs_embeds=arg_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=attention_mask,
            )
            arg_hidden = arg_decoder_out.last_hidden_state  # (B, T_dec, d)

            cmd_logits = self.cmd_head(cmd_hidden)   # (B, T_dec, vocab_size)
            arg_preds = self.arg_head(arg_hidden)    # (B, T_dec, n_args)

            return cmd_logits, arg_preds, encoder_hidden_states

        # PATH 2 : encoder out not yet computed
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state  # (B, T_enc, d)

        B = encoder_hidden_states.size(0)

        # Default decoder inputs
        if decoder_input_ids is None:
            decoder_input_ids = torch.full(
                (B, 1), self.sos_id, device=input_ids.device, dtype=torch.long
            )
        T_dec = decoder_input_ids.size(1)
        if decoder_input_args is None:
            decoder_input_args = self._default_args(
                B, T_dec, input_ids.device, encoder_hidden_states.dtype
            )

        # CMD decoder — input_embeddings already replaced with cmd_embedding
        cmd_decoder_out = self.cmd_decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
        )
        cmd_hidden = cmd_decoder_out.last_hidden_state  # (B, T_dec, d)

        # Arg decoder — feed arg_embedding output as inputs_embeds
        arg_embeds = self.arg_embedding(decoder_input_args)  # (B, T_dec, d)
        arg_decoder_out = self.arg_decoder(
            inputs_embeds=arg_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
        )
        arg_hidden = arg_decoder_out.last_hidden_state  # (B, T_dec, d)

        cmd_logits = self.cmd_head(cmd_hidden)   # (B, T_dec, vocab_size)
        arg_preds = self.arg_head(arg_hidden)    # (B, T_dec, n_args)

        return cmd_logits, arg_preds, encoder_hidden_states

