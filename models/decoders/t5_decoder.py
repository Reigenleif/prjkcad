import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM

class T5LayerFFMoE(nn.Module):
    def __init__(self, original_ff_layer: nn.Module, moe_conf: str, d_model: int):
        super().__init__()
        self.layer_norm = original_ff_layer.layer_norm
        self.dropout = original_ff_layer.dropout
        
        def make_expert():
            import copy
            return copy.deepcopy(original_ff_layer.DenseReluDense)
            
        from .moe import MoEBlock
        self.moe = MoEBlock(make_expert, moe_conf, d_model, is_sequence_level=False)
        
    def forward(self, hidden_states):
        norm_states = self.layer_norm(hidden_states)
        moe_out = self.moe(norm_states)
        return hidden_states + self.dropout(moe_out)

class PretrainedT5Decoder(nn.Module):
    """
    Wraps the decoder of a pretrained T5 model.
    """
    def __init__(self, pretrained_model_name: str = "t5-small", d_model: int = 512, side_embedding: nn.Module = None, moe_type: str = None, moe_conf: str = None):
        super().__init__()
        self.pretrained_model_name = pretrained_model_name
        self.d_model = d_model
        
        # Check suitability
        if pretrained_model_name == "t5-small" and d_model != 512:
            raise ValueError(f"t5-small decoder requires d_model to be 512, but got d_model={d_model}")
        elif pretrained_model_name == "t5-base" and d_model != 768:
            raise ValueError(f"t5-base decoder requires d_model to be 768, but got d_model={d_model}")
        elif pretrained_model_name == "t5-large" and d_model != 1024:
            raise ValueError(f"t5-large decoder requires d_model to be 1024, but got d_model={d_model}")
            
        if moe_type == "MoA":
            raise ValueError("MoA (Mixture of Attention) is not available for T5 models")
            
        pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name)
        self.decoder = pretrained_model.get_decoder()
        
        if side_embedding is not None:
            self.decoder.set_input_embeddings(side_embedding)

        # Apply MoE to Feed-Forward layers if requested
        if moe_type == "OnFFN":
            if moe_conf is None:
                raise ValueError("moe_conf must be specified when moe_type is OnFFN")
            for i in range(len(self.decoder.block)):
                original_ff = self.decoder.block[i].layer[2]
                self.decoder.block[i].layer[2] = T5LayerFFMoE(original_ff, moe_conf, d_model)

    def forward(self, input_ids=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None):
        outputs = self.decoder(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        return outputs.last_hidden_state
