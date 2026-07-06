from __future__ import annotations

import torch
from torch import nn
from typing import Optional, Any

from .embeddings import build_cmd_embedding, build_args_embedding
from .encoders import PretrainedT5Encoder, PretrainedBERTEncoder
from .decoders import PretrainedT5Decoder, TorchTransformerDecoder, SDPATransformerDecoder, MambaTransformerDecoder
from .heads import CMDHead, ArgsHead
from .adaptive_layer import AdaptiveLayer

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from utils import Config


class BaseModel(nn.Module):
    """
    Base model designed for all experiments on the prjkcad repository.
    Dynamically configures and connects components based on the ModelConfig.

    """
    AVAILABLE_ENCODERS = {"t5-small", "t5-base", "t5-large", "bert"}
    AVAILABLE_DECODERS = {"torch", "sdpa", "t5-small", "t5-base", "t5-large", "mamba"}

    def __init__(
        self,
        cfg: Config.Model,
        vocab_size: int,
        n_args: Optional[int] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.n_args = n_args

        self.d_model = cfg.d_model or 512
        if self.d_model not in [512, 768, 1024]:
            raise ValueError(f"d_model must be one of [512, 768, 1024], got {self.d_model}")

        self.pad_id = 0
        self.sos_id = 1
        self.eos_id = 2

        self.max_new_cmds = cfg.max_new_cmds

        # Encoder
        if cfg.encoder_type not in self.AVAILABLE_ENCODERS:
            raise ValueError(
                f"Unsupported encoder type: {cfg.encoder_type}. "
                f"Must be one of {self.AVAILABLE_ENCODERS}"
            )

        if cfg.encoder_type.startswith("t5-"):
            _t5_d = {"t5-small": 512, "t5-base": 768, "t5-large": 1024}
            required = _t5_d[cfg.encoder_type]
            if self.d_model != required:
                raise ValueError(
                    f"{cfg.encoder_type} encoder requires d_model={required}, got {self.d_model}"
                )
            self.encoder = PretrainedT5Encoder(cfg.encoder_type, d_model=self.d_model)
        elif cfg.encoder_type == "bert":
            self.encoder = PretrainedBERTEncoder(
                pretrained_model_name="bert-base-uncased",
                d_model=self.d_model,
            )

        if cfg.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.adaptive_layer = AdaptiveLayer(cfg.adaptive_layer, self.d_model)

        self.cmd_embedding = build_cmd_embedding(
            embedding_type=cfg.cmd_embedding_type,
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            max_len=self.max_new_cmds,
        )

        if not cfg.is_cmd_only:
            if n_args is None:
                raise ValueError("n_args must be provided if is_cmd_only is False")
            self.arg_embedding = build_args_embedding(
                embedding_type=cfg.args_embedding_type,
                n_args=n_args,
                d_model=self.d_model,
                max_len=self.max_new_cmds,
            )

        moe_conf = cfg.moe_conf

        # Command Decoder
        if cfg.cmd_decoder_type not in self.AVAILABLE_DECODERS:
            raise ValueError(
                f"Unsupported cmd decoder type: {cfg.cmd_decoder_type}. "
                f"Must be one of {self.AVAILABLE_DECODERS}"
            )

        self.cmd_decoder = self._build_decoder(
            cfg.cmd_decoder_type,
            side_embedding=self.cmd_embedding,
            moe_conf=moe_conf,
            dropout=cfg.drop_out_p if cfg.use_drop_out else 0.0,
        )

        if cfg.freeze_cmd_decoder:
            for param in self.cmd_decoder.parameters():
                param.requires_grad = False

        # Argument Decoder
        if not cfg.is_cmd_only:
            args_dec_type = cfg.args_decoder_type
            if args_dec_type not in self.AVAILABLE_DECODERS:
                raise ValueError(
                    f"Unsupported args decoder type: {args_dec_type}. "
                    f"Must be one of {self.AVAILABLE_DECODERS}"
                )

            self.arg_decoder = self._build_decoder(
                args_dec_type,
                side_embedding=None,
                moe_conf=moe_conf,
                dropout=cfg.drop_out_p if cfg.use_drop_out else 0.0,
            )

            if cfg.freeze_args_decoder:
                for param in self.arg_decoder.parameters():
                    param.requires_grad = False

        self.cmd_head = CMDHead(self.d_model, self.vocab_size)
        if not cfg.is_cmd_only:
            self.arg_head = ArgsHead(self.d_model, n_args)

    def _build_decoder(
        self,
        decoder_type: str,
        side_embedding: nn.Module,
        moe_conf: Optional[str],
        dropout: float,
    ) -> nn.Module:
        _t5_d = {"t5-small": 512, "t5-base": 768, "t5-large": 1024}

        if decoder_type.startswith("t5-"):
            required = _t5_d[decoder_type]
            if self.d_model != required:
                raise ValueError(
                    f"{decoder_type} decoder requires d_model={required}, got {self.d_model}"
                )
            return PretrainedT5Decoder(
                pretrained_model_name=decoder_type,
                d_model=self.d_model,
                side_embedding=side_embedding,
                moe_conf=moe_conf,
            )
        elif decoder_type == "torch":
            return TorchTransformerDecoder(
                d_model=self.d_model,
                side_embedding=side_embedding,
                moe_conf=moe_conf,
                dropout=dropout,
                max_len=self.max_new_cmds,
            )
        elif decoder_type == "sdpa":
            return SDPATransformerDecoder(
                d_model=self.d_model,
                side_embedding=side_embedding,
                moe_conf=moe_conf,
                dropout=dropout,
                max_len=self.max_new_cmds,
            )
        elif decoder_type == "mamba":
            return MambaTransformerDecoder(
                d_model=self.d_model,
                side_embedding=side_embedding,
                moe_conf=moe_conf,
                dropout=dropout,
                max_len=self.max_new_cmds,
            )
        raise ValueError(f"Unsupported decoder type: {decoder_type!r}")

    def _default_args(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros(batch_size, seq_len, self.n_args, device=device, dtype=dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor = None,
        decoder_input_args: torch.Tensor = None,
        encoder_out_embeddings: torch.Tensor = None,
    ):
        if encoder_out_embeddings is not None:
            encoder_hidden_states = encoder_out_embeddings
        else:
            encoder_hidden_states = self.encoder(input_ids, attention_mask)

        encoder_hidden_states = self.adaptive_layer(encoder_hidden_states)

        B = encoder_hidden_states.size(0)

        if decoder_input_ids is None:
            decoder_input_ids = torch.full(
                (B, 1), self.sos_id,
                device=encoder_hidden_states.device,
                dtype=torch.long,
            )

        cmd_hidden = self.cmd_decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
        )
        cmd_logits = self.cmd_head(cmd_hidden)

        if self.cfg.is_cmd_only:
            return cmd_logits, encoder_hidden_states

        T_dec = decoder_input_ids.size(1)
        if decoder_input_args is None:
            decoder_input_args = self._default_args(
                B, T_dec,
                encoder_hidden_states.device,
                encoder_hidden_states.dtype,
            )

        arg_embeds = self.arg_embedding(decoder_input_args) + cmd_hidden # CMD Directed
        arg_hidden = self.arg_decoder(
            inputs_embeds=arg_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
        )
        arg_preds = self.arg_head(arg_hidden)

        return cmd_logits, arg_preds, encoder_hidden_states

        
