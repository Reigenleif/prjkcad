from __future__ import annotations

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from .components import CADCmdSideEmbedding
        
      

class T5EncT5DecCADCMDOnly(nn.Module):
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
        pretrained_model_name: str,
        vocab_size: int,
        max_new_cmds: int = 1024,
    ):
        super().__init__()
        pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name)
        self.pretrained_model_name = pretrained_model_name
        self.encoder = pretrained_model.get_encoder()
        self.decoder = pretrained_model.get_decoder()
        
        self.cmd_vocab_size = vocab_size
        self.max_new_cmds = max_new_cmds
        
        # Extract d_model from the T5 encoder config
        d_model = self.encoder.config.d_model
        
        # CAD side embedding
        self.side_embedding = CADCmdSideEmbedding(
            vocab_size=vocab_size,
            d_model=d_model, 
            max_len=max_new_cmds
        )
        
        # Replace decoder's emebdding with side embedding
        self.decoder.set_input_embeddings(self.side_embedding)
        

        # Prediction heads        
        self.cmd_head = nn.Linear(d_model, self.cmd_vocab_size)
        self.pad_id = 0
        self.sos_id = 1
        self.eos_id = 2
        

    # def _causal_mask(self, decoder_input_ids: torch.Tensor) -> torch.Tensor :
    #     batch_size, seq_len = decoder_input_ids.size()
    #     mask = torch.tril(
    #         torch.ones(seq_len, seq_len, device=decoder_input_ids.device, dtype=torch.bool)
    #     )
    #     mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    #     return mask

    def _default_args(self, batch_size: int, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch_size, seq_len, self.arg_head.out_features, device=device, dtype=dtype)

    def forward(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids=None,
        encoder_out_embeddings=None,
    ):
        
        # PATH 1 : encoder out already computed and passed in
        if encoder_out_embeddings is not None :
            
            # Decode
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_out_embeddings,
                encoder_attention_mask=attention_mask,
            )

            decoder_hidden_states = decoder_outputs.last_hidden_state  # (B, T_dec, d)
            cmd_logits = self.cmd_head(decoder_hidden_states)
            
            return cmd_logits, encoder_out_embeddings
        
        
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
            input_ids=torch.ones(combined_hidden_states.shape[0], 
                                  1, 
                                  dtype=torch.long, 
                                  device=combined_hidden_states.device),
            encoder_hidden_states=combined_hidden_states
        )

        decoder_hidden_states = dec_out.last_hidden_state  # (B, T_dec, d)
        cmd_logits = self.cmd_head(decoder_hidden_states)
        
        return cmd_logits, encoder_hidden_states