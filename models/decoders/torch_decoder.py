import torch
import torch.nn as nn

class TorchTransformerDecoder(nn.Module):
    """
    Wraps PyTorch's built-in nn.TransformerDecoder.
    """
    def __init__(self, d_model: int, n_heads: int = 8, n_layers: int = 4, max_len: int = 1024, side_embedding: nn.Module = None):
        super().__init__()
        self.d_model = d_model
        self.side_embedding = side_embedding
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, input_ids=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None):
        if inputs_embeds is not None:
            tgt_state = inputs_embeds
        elif input_ids is not None:
            if self.side_embedding is not None:
                tgt_state = self.side_embedding(input_ids)
            else:
                raise ValueError("No input embedding available for input_ids in TorchTransformerDecoder")
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        seq_len = tgt_state.size(1)
        positions = torch.arange(seq_len, device=tgt_state.device)
        pos_embed = self.pos_embedding(positions)[None, :, :]
        tgt_state = tgt_state + pos_embed

        causal_mask = self._causal_mask(seq_len, tgt_state.device)
        
        tgt_key_padding_mask = None
        if input_ids is not None:
            pad_id = getattr(self.side_embedding, "pad_id", 0)
            tgt_key_padding_mask = (input_ids == pad_id)

        memory_key_padding_mask = None
        if encoder_attention_mask is not None:
            memory_key_padding_mask = (encoder_attention_mask == 0)

        dec_out = self.decoder(
            tgt=tgt_state,
            memory=encoder_hidden_states,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return dec_out
