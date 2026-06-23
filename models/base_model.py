from __future__ import annotations

import torch
from torch import nn
from typing import Optional, Any

from .embeddings import CADCmdSideEmbedding, CADArgsSideEmbedding
from .encoders import PretrainedT5Encoder
from .decoders import PretrainedT5Decoder, TorchTransformerDecoder, SDPATransformerDecoder, MambaTransformerDecoder
from .heads import CMDHead, ArgsHead

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from utils import Config

class BaseModel(nn.Module):
    """
    Base model designed for all experiments on the prjkcad repository.
    Dynamically configures and connects components based on the ModelConfig.

    """
    AVAILABLE_ENCODERS = {"t5-small", "t5-base", "t5-large"}
    AVAILABLE_DECODERS = {"torch", "sdpa", "t5-small", "t5-base", "t5-large", "mamba"}

    def __init__(self, cfg: Config.Model, vocab_size: int, n_args: Optional[int] = None, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.n_args = n_args
        
        # d_model defaults to 512 if not specified
        self.d_model = getattr(cfg, "d_model", 512)
        if self.d_model is None:
            self.d_model = 512
            
        # Ensure d_model is valid
        if self.d_model not in [512, 768, 1024]:
            raise ValueError(f"d_model must be one of [512, 768, 1024], but got d_model={self.d_model}")
            
        # Hard coded spec tokens
        self.pad_id = 0
        self.sos_id = 1
        self.eos_id = 2
        
        # max_new_cmds fallback
        self.max_new_cmds = getattr(cfg, "max_new_cmds", 1024)

        # 1. Instantiate Encoder
        if cfg.encoder_type not in self.AVAILABLE_ENCODERS:
            raise ValueError(f"Unsupported encoder type: {cfg.encoder_type}. Must be one of {self.AVAILABLE_ENCODERS}")
        
        if cfg.encoder_type.startswith("t5-"):
            if cfg.encoder_type == "t5-small" and self.d_model != 512:
                raise ValueError(f"t5-small encoder requires d_model to be 512, but got d_model={self.d_model}")
            elif cfg.encoder_type == "t5-base" and self.d_model != 768:
                raise ValueError(f"t5-base encoder requires d_model to be 768, but got d_model={self.d_model}")
            elif cfg.encoder_type == "t5-large" and self.d_model != 1024:
                raise ValueError(f"t5-large encoder requires d_model to be 1024, but got d_model={self.d_model}")
            self.encoder = PretrainedT5Encoder(cfg.encoder_type, d_model=self.d_model)

        # 2. Instantiate Embedding Components for Decoders
        self.cmd_embedding = CADCmdSideEmbedding(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            max_len=self.max_new_cmds
        )
        
        if not cfg.is_cmd_only:
            if n_args is None:
                raise ValueError("n_args must be provided if is_cmd_only is False")
            self.arg_embedding = CADArgsSideEmbedding(
                n_args=n_args,
                d_model=self.d_model,
                max_len=self.max_new_cmds
            )

        # 3. Instantiate Command Decoder
        if cfg.cmd_decoder_type not in self.AVAILABLE_DECODERS:
            raise ValueError(f"Unsupported cmd decoder type: {cfg.cmd_decoder_type}. Must be one of {self.AVAILABLE_DECODERS}")
            
        moe_type = getattr(cfg, "moe_type", None)
        moe_conf = getattr(cfg, "moe_conf", None)
        
        if moe_type == "MoA" and cfg.cmd_decoder_type.startswith("t5-"):
            raise ValueError("MoA (Mixture of Attention) is not available for T5 models")

        if cfg.cmd_decoder_type.startswith("t5-"):
            if cfg.cmd_decoder_type == "t5-small" and self.d_model != 512:
                raise ValueError(f"t5-small cmd decoder requires d_model to be 512, but got d_model={self.d_model}")
            elif cfg.cmd_decoder_type == "t5-base" and self.d_model != 768:
                raise ValueError(f"t5-base cmd decoder requires d_model to be 768, but got d_model={self.d_model}")
            elif cfg.cmd_decoder_type == "t5-large" and self.d_model != 1024:
                raise ValueError(f"t5-large cmd decoder requires d_model to be 1024, but got d_model={self.d_model}")
            self.cmd_decoder = PretrainedT5Decoder(cfg.cmd_decoder_type, d_model=self.d_model, side_embedding=self.cmd_embedding, moe_type=moe_type, moe_conf=moe_conf)
        elif cfg.cmd_decoder_type == "torch":
            self.cmd_decoder = TorchTransformerDecoder(d_model=self.d_model, side_embedding=self.cmd_embedding, moe_type=moe_type, moe_conf=moe_conf)
        elif cfg.cmd_decoder_type == "sdpa":
            self.cmd_decoder = SDPATransformerDecoder(d_model=self.d_model, side_embedding=self.cmd_embedding, moe_type=moe_type, moe_conf=moe_conf)
        elif cfg.cmd_decoder_type == "mamba":
            self.cmd_decoder = MambaTransformerDecoder(d_model=self.d_model, side_embedding=self.cmd_embedding, moe_type=moe_type, moe_conf=moe_conf)

        # 4. Instantiate Args Decoder (if applicable) 
        if not cfg.is_cmd_only:
            args_dec_type = cfg.args_decoder_type
            if args_dec_type not in self.AVAILABLE_DECODERS:
                raise ValueError(f"Unsupported args decoder type: {args_dec_type}. Must be one of {self.AVAILABLE_DECODERS}")
                
            if moe_type == "MoA" and args_dec_type.startswith("t5-"):
                raise ValueError("MoA (Mixture of Attention) is not available for T5 models")

            if args_dec_type.startswith("t5-"):
                if args_dec_type == "t5-small" and self.d_model != 512:
                    raise ValueError(f"t5-small args decoder requires d_model to be 512, but got d_model={self.d_model}")
                elif args_dec_type == "t5-base" and self.d_model != 768:
                    raise ValueError(f"t5-base args decoder requires d_model to be 768, but got d_model={self.d_model}")
                elif args_dec_type == "t5-large" and self.d_model != 1024:
                    raise ValueError(f"t5-large args decoder requires d_model to be 1024, but got d_model={self.d_model}")
                self.arg_decoder = PretrainedT5Decoder(args_dec_type, d_model=self.d_model, moe_type=moe_type, moe_conf=moe_conf) # feed arg_embeds as inputs_embeds in forward
            elif args_dec_type == "torch":
                self.arg_decoder = TorchTransformerDecoder(d_model=self.d_model, moe_type=moe_type, moe_conf=moe_conf)
            elif args_dec_type == "sdpa":
                self.arg_decoder = SDPATransformerDecoder(d_model=self.d_model, moe_type=moe_type, moe_conf=moe_conf)
            elif args_dec_type == "mamba":
                self.arg_decoder = MambaTransformerDecoder(d_model=self.d_model, moe_type=moe_type, moe_conf=moe_conf)



        # 6. Instantiate Heads
        self.cmd_head = CMDHead(self.d_model, self.vocab_size)
        if not cfg.is_cmd_only:
            self.arg_head = ArgsHead(self.d_model, n_args)

    def _default_args(self, batch_size: int, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch_size, seq_len, self.n_args, device=device, dtype=dtype)

    def forward(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids=None,
        decoder_input_args=None,
        encoder_out_embeddings=None
    ):
        # 1. Process encoder hidden states
        if encoder_out_embeddings is not None:
            encoder_hidden_states = encoder_out_embeddings
        else:
            encoder_hidden_states = self.encoder(input_ids, attention_mask)
            
        B = encoder_hidden_states.size(0)

        # Default decoder inputs if not provided
        if decoder_input_ids is None:
            decoder_input_ids = torch.full(
                (B, 1), self.sos_id, device=encoder_hidden_states.device, dtype=torch.long
            )

        # 2. Command Decoder Path
        cmd_hidden = self.cmd_decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask
        )

        cmd_logits = self.cmd_head(cmd_hidden)

        # 3. Arguments Decoder Path (if not cmd only)
        if not self.cfg.is_cmd_only:
            T_dec = decoder_input_ids.size(1)
            if decoder_input_args is None:
                decoder_input_args = self._default_args(
                    B, T_dec, encoder_hidden_states.device, encoder_hidden_states.dtype
                )

            # Args decoder uses arg_embedding (CADArgsSideEmbedding)
            arg_embeds = self.arg_embedding(decoder_input_args)
            
            # Args decoder computes representation
            arg_hidden = self.arg_decoder(
                inputs_embeds=arg_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=attention_mask
            )

            arg_preds = self.arg_head(arg_hidden)

            return cmd_logits, arg_preds, encoder_hidden_states
        else:
            return cmd_logits, encoder_hidden_states
