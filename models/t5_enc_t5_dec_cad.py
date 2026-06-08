from __future__ import annotations

import torch
from torch import nn
from transformers import T5EncoderModel


class CADCmdSideEmbedding(nn.Module) :
    """
    Encodes command sequence to a d_model dimensional representation using a Transformer Encoder.
    """
    def __init__(self,
                 n_cmds: int,
                d_model: int = 512,
                max_len: int = 1024):
    
        super().__init__()
        self.cmd_embedding = nn.Embedding(n_cmds + 2, d_model) # +2 for SOS and PAD
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        
    def forward(self, cmd_input_ids) :
        seq_len = cmd_input_ids.size(1)
        positions = torch.arange(seq_len, device=cmd_input_ids.device).unsqueeze(0)
        cmd_embeds = self.cmd_embedding(cmd_input_ids)
        pos_embeds = self.pos_embedding(positions)
        return cmd_embeds + pos_embeds
        
      

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
        
        # CAD side embedding
        self.side_embedding = CADCmdSideEmbedding(
            n_cmds=n_cmds,
            d_model=d_model, 
            max_len=max_new_cmds
        )
        
        # Replace decoder's emebdding with side embedding
        self.decoder.set_input_embeddings(self.side_embedding)
        
        
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
        if encoder_out_embeddings is not None :
            
            # Decode
            decoder_outputs = self.decoder(
                inputs_embeds=decoder_out_embeddings,
                encoder_hidden_states=encoder_out_embeddings,
                encoder_attention_mask=attention_mask,
            )

            decoder_hidden_states = decoder_outputs.last_hidden_state  # (B, T_dec, d)
            cmd_logits = self.cmd_head(decoder_hidden_states)
            
            return cmd_logits, decoder_hidden_states, encoder_out_embeddings
        
        
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