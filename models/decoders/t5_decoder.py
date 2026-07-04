import copy

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM

from ..common import SwitchFFN, MixtralFFN

def _make_t5_expert(original_dense_relu_dense: nn.Module) -> nn.Module:
    return copy.deepcopy(original_dense_relu_dense)


class T5LayerFFMoE(nn.Module):
    def __init__(self, original_ff_layer: nn.Module, moe_conf: str, d_model: int):
        super().__init__()
        self.layer_norm = original_ff_layer.layer_norm
        self.dropout = original_ff_layer.dropout

        dense = original_ff_layer.DenseReluDense
        make_fn = lambda: _make_t5_expert(dense)

        if moe_conf == "Switch":
            self.moe = SwitchFFN(make_fn, d_model)
        elif moe_conf == "Mixtral":
            self.moe = MixtralFFN(make_fn, d_model)
        else:
            raise ValueError(f"Unknown moe_conf for T5 MoE: {moe_conf!r}")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        norm_states = self.layer_norm(hidden_states)
        moe_out = self.moe(norm_states)
        return hidden_states + self.dropout(moe_out)

# Pretrained T5 Decoder
class PretrainedT5Decoder(nn.Module):

    def __init__(
        self,
        pretrained_model_name: str = "t5-small",
        d_model: int = 512,
        side_embedding: nn.Module = None,
        moe_conf: str = None,
    ):
        super().__init__()
        self.pretrained_model_name = pretrained_model_name
        self.d_model = d_model

        # Validate size compatibility
        if pretrained_model_name == "t5-small" and d_model != 512:
            raise ValueError(f"t5-small decoder requires d_model=512, got {d_model}")
        elif pretrained_model_name == "t5-base" and d_model != 768:
            raise ValueError(f"t5-base decoder requires d_model=768, got {d_model}")
        elif pretrained_model_name == "t5-large" and d_model != 1024:
            raise ValueError(f"t5-large decoder requires d_model=1024, got {d_model}")

        pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name)
        self.decoder = pretrained_model.get_decoder()

        if side_embedding is not None:
            self.decoder.set_input_embeddings(side_embedding)

        # Apply MoE to Feed-Forward layers if requested
        if moe_conf is not None:
            for i in range(len(self.decoder.block)):
                original_ff = self.decoder.block[i].layer[2]
                self.decoder.block[i].layer[2] = T5LayerFFMoE(original_ff, moe_conf, d_model)

    def forward(
        self,
        input_ids: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        encoder_hidden_states: torch.Tensor = None,
        encoder_attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        outputs = self.decoder(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        return outputs.last_hidden_state
