import logging
import torch
import torch.nn as nn
from transformers import AutoModel

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


class PretrainedBERTEncoder(nn.Module):
    """
    Wraps the encoder of a pretrained BERT model (bert-base-uncased, native d=768).

    If the requested d_model differs from BERT's native 768, an internal
    nn.Linear(768, d_model) projection is applied to the encoder output,
    so any d_model is supported without an external adapter.
    """

    BERT_HIDDEN_SIZE = 768

    def __init__(self, pretrained_model_name: str = "bert-base-uncased", d_model: int = 512):
        super().__init__()
        self.pretrained_model_name = pretrained_model_name
        self.d_model = d_model

        self.encoder = AutoModel.from_pretrained(pretrained_model_name)

        # Optional projection when d_model != BERT native size
        if d_model != self.BERT_HIDDEN_SIZE:
            self.proj = nn.Linear(self.BERT_HIDDEN_SIZE, d_model)
        else:
            self.proj = None

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if attention_mask is not None and attention_mask.shape[1] != input_ids.shape[1]:
            if attention_mask.shape[1] > input_ids.shape[1]:
                attention_mask = attention_mask[:, :input_ids.shape[1]]
            else:
                pad_len = input_ids.shape[1] - attention_mask.shape[1]
                pad_tensor = torch.zeros((attention_mask.size(0), pad_len), dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, pad_tensor], dim=1)

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden = outputs.last_hidden_state   # (B, T, 768)
        if self.proj is not None:
            hidden = self.proj(hidden)       # (B, T, d_model)
        return hidden
