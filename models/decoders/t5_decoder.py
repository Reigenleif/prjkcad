import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM

class PretrainedT5Decoder(nn.Module):
    """
    Wraps the decoder of a pretrained T5 model.
    """
    def __init__(self, pretrained_model_name: str = "t5-small", d_model: int = 512, side_embedding: nn.Module = None):
        super().__init__()
        self.pretrained_model_name = pretrained_model_name
        self.d_model = d_model
        
        # Check suitability
        if pretrained_model_name == "t5-small" and d_model != 512:
            raise ValueError(f"t5-small decoder requires d_model to be 512, but got d_model={d_model}")
            
        pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name)
        self.decoder = pretrained_model.get_decoder()
        
        if side_embedding is not None:
            self.decoder.set_input_embeddings(side_embedding)

    def forward(self, input_ids=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None):
        outputs = self.decoder(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        return outputs.last_hidden_state
