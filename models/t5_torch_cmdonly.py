from __future__ import annotations

import torch
from torch import nn
from transformers import T5EncoderModel

class T5TorchCmdonly(nn.Module):
    """
    Model that consists of :
    - T5 Encoder 
    - Torch built-in Transformer Decoder
    for text-to-DualSeq-cmdonly training.
    """

    def __init__(
        self,
        t5_encoder: T5EncoderModel,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
        max_new_cmds: int = 1024,
    ):
        super().__init__()
        
        self.encoder = t5_encoder
        self.cmd_vocab_size = vocab_size
        self.max_new_cmds = max_new_cmds
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.cmd_embedding = nn.Embedding(self.cmd_vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_new_cmds, d_model)

        self.cmd_head = nn.Linear(d_model, self.cmd_vocab_size)
        self.pad_id = 0
        self.sos_id = 1
        self.eos_id = 2

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids=None,
        encoder_out_embeddings=None,
    ):
        
        if encoder_out_embeddings is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            encoder_hidden_states = encoder_outputs.last_hidden_state
        else:
            encoder_hidden_states = encoder_out_embeddings
            
        if decoder_input_ids is None:
            # Dummy decode for initialization or feature extraction like T5EncT5DecCAD does
            decoder_input_ids = torch.ones(encoder_hidden_states.shape[0], 
                                  1, 
                                  dtype=torch.long, 
                                  device=encoder_hidden_states.device)

        positions = torch.arange(decoder_input_ids.size(1), device=decoder_input_ids.device)
        pos_embed = self.pos_embedding(positions)[None, :, :]
        
        tgt_state = self.cmd_embedding(decoder_input_ids) + pos_embed
        
        tgt_key_padding_mask = decoder_input_ids == self.pad_id
        memory_key_padding_mask = attention_mask == 0
        causal_mask = self._causal_mask(decoder_input_ids.size(1), decoder_input_ids.device)

        dec_out = self.decoder(
            tgt=tgt_state,
            memory=encoder_hidden_states,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        cmd_logits = self.cmd_head(dec_out)
        
        return cmd_logits, encoder_hidden_states
