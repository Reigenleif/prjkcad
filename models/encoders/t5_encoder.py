import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM

class PretrainedT5Encoder(nn.Module):
    """
    Wraps the encoder of a pretrained T5 model.
    """
    def __init__(self, pretrained_model_name: str = "t5-small", d_model: int = 512):
        super().__init__()
        self.pretrained_model_name = pretrained_model_name
        self.d_model = d_model
        
        # Check suitability
        if pretrained_model_name == "t5-small" and d_model != 512:
            raise ValueError(f"t5-small encoder requires d_model to be 512, but got d_model={d_model}")
            
        pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name)
        self.encoder = pretrained_model.get_encoder()

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.last_hidden_state
