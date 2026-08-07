from __future__ import annotations

from typing import Optional
import torch
from torch import nn

from .embeddings import build_cmd_embedding, build_args_embedding, BinarizedArgsSideEmbedding
from .encoders import PretrainedT5Encoder, PretrainedBERTEncoder
from .heads import CMDHead, ArgsHead
from .adaptive_layer import AdaptiveLayer
from .common import FusionStack
from utils.representations.dual_seq.schema import get_dualseq_schema  # noqa: direct submodule import avoids utils/__init__ circular chain


class BaseModel(nn.Module):
    AVAILABLE_ENCODERS = {"t5-small", "t5-base", "t5-large", "bert"}

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
        else:  # FloatArgs
            self.arg_pad_id = 0
            self.arg_sos_id = 0
            self.arg_eos_id = 0

        self.max_new_cmds = cfg.max_new_cmds
        self.max_new_args = cfg.max_new_args or cfg.max_new_cmds

        self.n_dec_blocks = getattr(cfg, "n_dec_blocks", 6)
        dropout = cfg.drop_out_p if cfg.use_drop_out else 0.1

        self._init_encoder()
        self._init_embeddings()
        self._init_heads()

        self.fusion_stack = FusionStack(
            d_model=self.d_model,
            n_dec_blocks=self.n_dec_blocks,
            dropout=dropout,
        )

    # ── Encoder ──────────────────────────────────────────────────────────

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

    # ── Embeddings ───────────────────────────────────────────────────────

    def _init_embeddings(self):
        cfg = self.cfg

        self.cmd_embedding = build_cmd_embedding(
            embedding_type=cfg.cmd_embedding_type,
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            max_len=self.max_new_cmds,
        )

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
        else:  # TokenizedOneSequenceArgs
            if self.vocab_size_args is None:
                raise ValueError("vocab_size_args must be provided for TokenizedOneSequenceArgs")
            self.arg_embedding = build_cmd_embedding(
                embedding_type=cfg.args_embedding_type,
                vocab_size=self.vocab_size_args,
                d_model=self.d_model,
                max_len=self.max_new_args,
            )

    # ── Heads ────────────────────────────────────────────────────────────

    def _init_heads(self):
        self.cmd_head = CMDHead(self.d_model, self.vocab_size)
        if self.cfg.is_cmd_only:
            return

        if self.out_type == "FloatArgs":
            self.arg_head = ArgsHead(self.d_model, 31)
        elif self.out_type == "EightBitBinarizedArgs":
            self.arg_head = nn.Linear(self.d_model, 31 * 257)
        else:  # TokenizedOneSequenceArgs
            self.arg_head = ArgsHead(self.d_model, self.vocab_size_args)

    # ── Forward ──────────────────────────────────────────────────────────

    def _forward_encoder(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        enc_out = self.encoder(input_ids, attention_mask)
        return self.adaptive_layer(enc_out)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor = None,
        decoder_input_args: torch.Tensor = None,
        encoder_out_embeddings: torch.Tensor = None,
    ):
        # 1. Encoder -> adaptive layer -> enc_out
        if encoder_out_embeddings is not None:
            enc_out = encoder_out_embeddings
        else:
            enc_out = self._forward_encoder(input_ids, attention_mask)

        B = enc_out.size(0)

        if decoder_input_ids is None:
            decoder_input_ids = torch.full(
                (B, 1), self.sos_id,
                device=enc_out.device,
                dtype=torch.long,
            )

        # 2. Embed cmd and args streams
        cmd_h = self.cmd_embedding(decoder_input_ids)

        arg_h = None
        if not self.cfg.is_cmd_only:
            if decoder_input_args is None:
                shape = (B, 1) if self.out_type == "TokenizedOneSequenceArgs" else (B, 1, 31)
                dtype = torch.long if self.out_type != "FloatArgs" else torch.float
                decoder_input_args = torch.full(shape, self.arg_sos_id, device=enc_out.device, dtype=dtype)
            arg_h = self.arg_embedding(decoder_input_args)

        # 3. Cmd-arg fusion stack
        cmd_h, arg_h = self.fusion_stack(
            cmd_h,
            arg_h,
            enc_out,
            encoder_attention_mask=attention_mask,
            cmd_input_ids=decoder_input_ids,
            arg_input_args=decoder_input_args,
            cmd_pad_id=self.pad_id,
            arg_pad_id=self.arg_pad_id,
        )

        # 4. Heads
        cmd_logits = self.cmd_head(cmd_h)

        if self.cfg.is_cmd_only:
            return cmd_logits, enc_out

        if self.out_type == "EightBitBinarizedArgs":
            arg_logits = self.arg_head(arg_h).view(B, -1, 31, 257)
        else:
            arg_logits = self.arg_head(arg_h)

        return cmd_logits, arg_logits, enc_out
