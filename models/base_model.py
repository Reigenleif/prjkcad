from __future__ import annotations

from typing import Optional, Tuple
import torch
from torch import nn

from .embeddings import build_cmd_embedding, build_args_embedding, BinarizedArgsSideEmbedding
from .encoders import PretrainedT5Encoder, PretrainedBERTEncoder
from .decoders import PretrainedT5Decoder, SDPATransformerDecoder, MambaTransformerDecoder
from .heads import CMDHead, ArgsHead
from .adaptive_layer import AdaptiveLayer
from .common import CmdArgsFusion, FusionStack
from utils.representations.dual_seq.schema import get_dualseq_schema


class BaseModel(nn.Module):
    AVAILABLE_ENCODERS = {"t5-small", "t5-base", "t5-large", "bert"}
    AVAILABLE_DECODERS = {"sdpa", "t5-small", "t5-base", "t5-large", "mamba"}

    def __init__(
        self,
        cfg,
        vocab_size: int,
        vocab_size_args: Optional[int] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.vocab_size_args = vocab_size_args

        self.d_model = cfg.d_model or 512
        if self.d_model not in [256, 512, 768, 1024]:
            raise ValueError(f"d_model must be one of [256, 512, 768, 1024], got {self.d_model}")

        self.schema = get_dualseq_schema()
        self.out_type = getattr(cfg, "out_type", "FloatArgs")
        
        self.pad_id = self.schema["cmd_pad_id"]
        self.sos_id = self.schema["cmd_sos_id"]
        self.eos_id = self.schema["cmd_eos_id"]
        
        if self.out_type == "TokenizedOneSequenceArgs":
            self.arg_pad_id = self.schema["arg_pad_id"]
            self.arg_sos_id = self.schema["arg_sos_id"]
            self.arg_eos_id = self.schema["arg_eos_id"]
        elif self.out_type == "EightBitBinarizedArgs":
            self.arg_pad_id = 256
            self.arg_sos_id = 0
            self.arg_eos_id = 256
        else: # FloatArgs
            self.arg_pad_id = 0
            self.arg_sos_id = 0
            self.arg_eos_id = 0

        self.max_new_cmds = cfg.max_new_cmds
        self.max_new_args = cfg.max_new_args or cfg.max_new_cmds

        self.use_cmd_args_fusion = getattr(cfg, "use_cmd_args_fusion", False)
        self.n_dec_blocks = getattr(cfg, "n_dec_blocks", 6)
        if self.use_cmd_args_fusion:
            dropout = cfg.drop_out_p if cfg.use_drop_out else 0.1
            self.fusion_stack = FusionStack(d_model=self.d_model, n_dec_blocks=self.n_dec_blocks, dropout=dropout)

        self._init_encoder()
        self._init_cmd_decoder()
        self._init_args_decoder()
        self._init_heads()

    def _init_encoder(self):
        cfg = self.cfg
        if cfg.encoder_type not in self.AVAILABLE_ENCODERS:
            raise ValueError(f"Unsupported encoder type: {cfg.encoder_type}")

        if cfg.encoder_type.startswith("t5-"):
            _t5_d = {"t5-small": 512, "t5-base": 768, "t5-large": 1024}
            required = _t5_d[cfg.encoder_type]
            if self.d_model != required:
                raise ValueError(f"{cfg.encoder_type} encoder requires d_model={required}")
            self.encoder = PretrainedT5Encoder(cfg.encoder_type, d_model=self.d_model)
        elif cfg.encoder_type == "bert":
            self.encoder = PretrainedBERTEncoder("google-bert/bert-base-uncased", d_model=self.d_model)

        if cfg.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.adaptive_layer = AdaptiveLayer(cfg.adaptive_layer, self.d_model)

    def _init_cmd_decoder(self):
        cfg = self.cfg
        self.cmd_embedding = build_cmd_embedding(
            embedding_type=cfg.cmd_embedding_type,
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            max_len=self.max_new_cmds,
        )

        if cfg.cmd_decoder_type not in self.AVAILABLE_DECODERS:
            raise ValueError(f"Unsupported cmd decoder type: {cfg.cmd_decoder_type}")

        self.cmd_decoder = self._build_decoder(
            cfg.cmd_decoder_type,
            side_embedding=self.cmd_embedding,
            dropout=cfg.drop_out_p if cfg.use_drop_out else 0.0,
            max_len=self.max_new_cmds,
            n_layers=getattr(cfg, "cmd_n_layers", None),
        )

        if cfg.freeze_cmd_decoder:
            for param in self.cmd_decoder.parameters():
                param.requires_grad = False

    def _init_args_decoder(self):
        cfg = self.cfg
        if cfg.is_cmd_only:
            return

        if self.out_type == "FloatArgs":
            self.arg_embedding = build_args_embedding(
                embedding_type=cfg.args_embedding_type,
                n_args=31,
                d_model=self.d_model,
                max_len=self.max_new_cmds,
            )
        elif self.out_type == "EightBitBinarizedArgs":
            self.arg_embedding = BinarizedArgsSideEmbedding(
                n_args=31,
                d_model=self.d_model,
                max_len=self.max_new_cmds,
                embedding_type=cfg.args_embedding_type,
            )
        else: # TokenizedOneSequenceArgs
            if self.vocab_size_args is None:
                raise ValueError("vocab_size_args must be provided for TokenizedOneSequenceArgs")
            self.arg_embedding = build_cmd_embedding(
                embedding_type=cfg.args_embedding_type,
                vocab_size=self.vocab_size_args,
                d_model=self.d_model,
                max_len=self.max_new_args,
            )

        args_dec_type = cfg.args_decoder_type
        if args_dec_type not in self.AVAILABLE_DECODERS:
            raise ValueError(f"Unsupported args decoder type: {args_dec_type}")

        self.arg_decoder = self._build_decoder(
            args_dec_type,
            side_embedding=self.arg_embedding,
            dropout=cfg.drop_out_p if cfg.use_drop_out else 0.0,
            max_len=self.max_new_args,
            n_layers=getattr(cfg, "args_n_layers", None),
        )

        if cfg.freeze_args_decoder:
            for param in self.arg_decoder.parameters():
                param.requires_grad = False

    def _init_heads(self):
        self.cmd_head = CMDHead(self.d_model, self.vocab_size)
        if self.cfg.is_cmd_only:
            return

        if self.out_type == "FloatArgs":
            self.arg_head = ArgsHead(self.d_model, 31)
        elif self.out_type == "EightBitBinarizedArgs":
            self.arg_head = nn.Linear(self.d_model, 31 * 257)
        else: # TokenizedOneSequenceArgs
            self.arg_head = ArgsHead(self.d_model, self.vocab_size_args)

    def _build_decoder(
        self,
        decoder_type: str,
        side_embedding: nn.Module,
        dropout: float,
        max_len: int,
        n_layers: Optional[int] = None,
    ) -> nn.Module:
        _t5_d = {"t5-small": 512, "t5-base": 768, "t5-large": 1024}
        moe_conf = self.cfg.moe_conf

        kwargs = {
            "d_model": self.d_model,
            "side_embedding": side_embedding,
            "moe_conf": moe_conf,
            "dropout": dropout,
            "max_len": max_len,
        }
        if n_layers is not None:
            kwargs["n_layers"] = n_layers

        if decoder_type.startswith("t5-"):
            required = _t5_d[decoder_type]
            if self.d_model != required:
                raise ValueError(f"{decoder_type} decoder requires d_model={required}")
            return PretrainedT5Decoder(
                pretrained_model_name=decoder_type,
                d_model=self.d_model,
                side_embedding=side_embedding,
                moe_conf=moe_conf,
            )
        elif decoder_type == "sdpa":
            return SDPATransformerDecoder(**kwargs)
        elif decoder_type == "mamba":
            return MambaTransformerDecoder(**kwargs)
        raise ValueError(f"Unsupported decoder type: {decoder_type}")

    def _forward_encoder(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        encoder_hidden_states = self.encoder(input_ids, attention_mask)
        return self.adaptive_layer(encoder_hidden_states)

    def _forward_cmd_decoder(self, decoder_input_ids: torch.Tensor, encoder_hidden_states: torch.Tensor, attention_mask: torch.Tensor, inputs_embeds: torch.Tensor = None) -> torch.Tensor:
        return self.cmd_decoder(
            input_ids=decoder_input_ids,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
        )

    def _forward_args_decoder(self, decoder_input_args: torch.Tensor, encoder_hidden_states: torch.Tensor, attention_mask: torch.Tensor, inputs_embeds: torch.Tensor = None) -> torch.Tensor:
        return self.arg_decoder(
            input_ids=decoder_input_args,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor = None,
        decoder_input_args: torch.Tensor = None,
        encoder_out_embeddings: torch.Tensor = None,
    ):
        # <--- Encoder pass
        if encoder_out_embeddings is not None:
            encoder_hidden_states = encoder_out_embeddings
        else:
            encoder_hidden_states = self._forward_encoder(input_ids, attention_mask)

        B = encoder_hidden_states.size(0)

        if decoder_input_ids is None:
            decoder_input_ids = torch.full(
                (B, 1), self.sos_id,
                device=encoder_hidden_states.device,
                dtype=torch.long,
            )

        # <--- Decoder Pass
        if self.use_cmd_args_fusion and decoder_input_args is not None:
            cmd_embeds = self.cmd_embedding(decoder_input_ids)
            arg_embeds = self.arg_embedding(decoder_input_args)
            cmd_hidden, arg_hidden = self.fusion_stack(
                cmd_embeds,
                arg_embeds,
                encoder_hidden_states,
                attention_mask,
                cmd_input_ids=decoder_input_ids,
                arg_input_args=decoder_input_args,
                cmd_pad_id=self.pad_id,
                arg_pad_id=self.arg_pad_id,
            )
        else:
            cmd_hidden = self._forward_cmd_decoder(decoder_input_ids, encoder_hidden_states, attention_mask)
            if not self.cfg.is_cmd_only and decoder_input_args is None:
                decoder_input_args_shape = (B, 1) if self.out_type == "TokenizedOneSequenceArgs" else (B, 1, 31)
                decoder_input_args = torch.full(
                    decoder_input_args_shape, self.arg_sos_id,
                    device=encoder_hidden_states.device,
                    dtype=torch.long if self.out_type != "FloatArgs" else torch.float,
                )
            if not self.cfg.is_cmd_only:
                arg_hidden = self._forward_args_decoder(decoder_input_args, encoder_hidden_states, attention_mask)
            else:
                arg_hidden = None
        # <--- Heads pass
        cmd_logits = self.cmd_head(cmd_hidden)
        if self.cfg.is_cmd_only:
            return cmd_logits, encoder_hidden_states

        if self.out_type == "FloatArgs":
            arg_logits = self.arg_head(arg_hidden)
        elif self.out_type == "EightBitBinarizedArgs":
            logits_flat = self.arg_head(arg_hidden)
            arg_logits = logits_flat.view(B, -1, 31, 257)
        else:
            arg_logits = self.arg_head(arg_hidden)

        return cmd_logits, arg_logits, encoder_hidden_states
