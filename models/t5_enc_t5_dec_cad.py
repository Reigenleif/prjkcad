from __future__ import annotations

import torch
from torch import nn
from transformers import T5EncoderModel

class CADCmdSideEncoder(nn.Module) :
    """
    Encodes command sequence to a d_model dimensional representation using a Transformer Encoder.
    """
    def __init__(self,
                 n_cmds: int,
                d_model: int = 512,
                n_heads: int = 8,
                n_layers: int = 6,
                max_len: int = 1024):
        
        super().__init__()
        self.cmd_embedding = nn.Embedding(n_cmds + 3, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
    def forward(self, cmd_input_ids, src_key_padding_mask=None):
        positions = torch.arange(cmd_input_ids.size(1), device=cmd_input_ids.device)
        pos_embed = self.pos_embedding(positions)[None, :, :]
        cmd_embed = self.cmd_embedding(cmd_input_ids) + pos_embed
        cmd_embed = cmd_embed.transpose(0, 1)
        encoded_cmds = self.transformer_encoder(cmd_embed, src_key_padding_mask=src_key_padding_mask)
        return encoded_cmds.transpose(0, 1)
    
      

class T5EncT5DecCAD(nn.Module):
    """
    Model that consists of :
    - T5 Encoder 
    - T5 Decoder Headless
    - CMD CAD Decoder head 
    - CMD CAD side encoder
    for text-to-DualSeq-cmdonly training.
    """

    def __init__(
        self,
        t5_encoder: T5EncoderModel,
        t5_decoder,
        embedding,
        n_cmds: int,
        side_encoder_heads: int = 8,
        side_encoder_layers: int = 6,
        max_new_cmds: int = 1024,
    ):
        super().__init__()

        self.embedding  = embedding
        self.encoder = t5_encoder
        self.decoder = t5_decoder
        self.cmd_vocab_size = n_cmds + 2 # 1 for SOS, 1 for PAD (EOS is same as PAD)
        self.max_new_cmds = max_new_cmds
        
        # Extract d_model from the T5 encoder config
        d_model = t5_encoder.config.d_model
        
        # CAD side encoder
        self.side_encoder = CADCmdSideEncoder(
            n_cmds=n_cmds,
            d_model=d_model, 
            n_heads=side_encoder_heads, 
            n_layers=side_encoder_layers)
        # Prediction heads
        
        self.cmd_head = nn.Linear(d_model, self.cmd_vocab_size)
        self.pad_id = 0
        

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)

    def _default_args(self, batch_size: int, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch_size, seq_len, self.arg_head.out_features, device=device, dtype=dtype)

    def forward(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids=None,
        decoder_out_embeddings=None,
        encoder_out_embeddings=None,
    ):
        # PATH 1 : encoder out already computed and passed in
        if encoder_out_embeddings is not None and decoder_input_ids is not None:
            encoder_hidden_states = encoder_out_embeddings  # (B, T_enc, d)

            side_encoded_cmds = self.side_encoder(
                decoder_input_ids,
                src_key_padding_mask=(decoder_input_ids == self.pad_id)
            )  # (B, T_dec, d)
            
            combined_hidden_states = encoder_hidden_states + side_encoded_cmds
            
            # Decode
            decoder_outputs = self.decoder(
                inputs_embeds=decoder_out_embeddings,
                encoder_hidden_states=combined_hidden_states
            )

            decoder_hidden_states = decoder_outputs.last_hidden_state  # (B, T_dec, d)
            cmd_logits = self.cmd_head(decoder_hidden_states)
            
            return cmd_logits, decoder_hidden_states, encoder_hidden_states
        
        
        # PATH 2 : encoder out not yet computed
        # Encode text (T5 token domain)
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state  # (B, T_enc, d)

        combined_hidden_states = encoder_hidden_states
        
        # Decode
        dec_out = self.decoder(
            input_ids=torch.zeros(combined_hidden_states.shape[0], 
                                  1, 
                                  dtype=torch.long, 
                                  device=combined_hidden_states.device),
            encoder_hidden_states=combined_hidden_states
        )

        decoder_hidden_states = dec_out.last_hidden_state  # (B, T_dec, d)
        cmd_logits = self.cmd_head(decoder_hidden_states)
        
        return cmd_logits, decoder_hidden_states, encoder_hidden_states