from __future__ import annotations

from typing import Optional, Union, Dict, Tuple, Any, List
import torch
from torch import nn

from models.text2cad_ori.text2cad import Text2CAD
from utils.representations.CadSeqProc.utility.macro import END_TOKEN, MAX_CAD_SEQUENCE_LENGTH, N_BIT, CAD_CLASS_INFO
from utils.representations.CadSeqProc.utility.utils import generate_attention_mask
from utils.representations.CadSeqProc.cad_sequence import CADSequence


TEXT_CONFIG: Dict[str, Any] = {
    "text_embedder": {
        "model_name": "bert_large_uncased",
        "cache_dir": None,
        "max_seq_len": 512,
    },
    "adaptive_layer": {
        "in_dim": 1024,
        "out_dim": 1024,
        "num_heads": 8,
        "dropout": 0.1,
    },
}

CAD_CONFIG: Dict[str, Any] = {
    "tdim": 1024,
    "cdim": 256,
    "num_layers": 8,
    "num_heads": 8,
    "dropout": 0.1,
    "ca_level_start": 2,
    "cad_seq_len": 272,
}


class Text2CADModel(nn.Module):
    """Text2CAD wrapper module delegating to models/text2cad/text2cad.py."""

    def __init__(
        self,
        cfg: Any = None,
        vocab_size: int = 267,
        vocab_size_args: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.out_type = "CADSequence"
        self.vocab_size = vocab_size
        self.vocab_size_args = vocab_size_args

        text_config = kwargs.get("text_config", TEXT_CONFIG)
        cad_config = kwargs.get("cad_config", CAD_CONFIG)

        self.text2cad = Text2CAD(text_config=text_config, cad_config=cad_config)

        self.base_text_embedder = self.text2cad.base_text_embedder
        self.adaptive_layer = self.text2cad.adaptive_layer
        self.cad_decoder = self.text2cad.cad_decoder

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        decoder_input_args: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        encoder_out_embeddings: Optional[torch.Tensor] = None,
        vec_dict: Optional[Dict[str, torch.Tensor]] = None,
        texts: Optional[Union[str, List[str]]] = None,
        mask_cad_dict: Optional[Dict[str, torch.Tensor]] = None,
        metadata: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if texts is None and "texts" in kwargs:
            texts = kwargs["texts"]

        if texts is None:
            if input_ids is not None:
                texts = self.base_text_embedder.tokenizer.batch_decode(
                    input_ids, skip_special_tokens=True
                )
            else:
                texts = [""]

        if isinstance(texts, str):
            texts = [texts]

        device = next(self.parameters()).device if list(self.parameters()) else "cpu"

        if vec_dict is None:
            if isinstance(decoder_input_ids, dict):
                vec_dict = decoder_input_ids
            elif isinstance(decoder_input_ids, torch.Tensor):
                cad_vec = decoder_input_ids.to(device)
                if cad_vec.dim() == 2:
                    if isinstance(decoder_input_args, torch.Tensor) and decoder_input_args.dim() == 2:
                        args_t = decoder_input_args.to(device)
                        if args_t.size(1) > cad_vec.size(1):
                            args_t = args_t[:, :cad_vec.size(1)]
                        elif args_t.size(1) < cad_vec.size(1):
                            pad_len = cad_vec.size(1) - args_t.size(1)
                            pad_tensor = torch.zeros((cad_vec.size(0), pad_len), dtype=args_t.dtype, device=device)
                            args_t = torch.cat([args_t, pad_tensor], dim=1)
                        cad_vec = torch.stack([cad_vec, args_t], dim=-1)
                    else:
                        cad_vec = torch.stack([cad_vec, torch.zeros_like(cad_vec)], dim=-1)
                elif cad_vec.dim() == 3 and cad_vec.size(1) == 2:
                    cad_vec = cad_vec.transpose(1, 2)

                B, T_seq = cad_vec.size(0), cad_vec.size(1)
                flag_vec = torch.zeros((B, T_seq), dtype=torch.long, device=device)
                index_vec = torch.zeros((B, T_seq), dtype=torch.long, device=device)
                if isinstance(decoder_input_args, dict):
                    flag_vec = decoder_input_args.get("flag_vec", flag_vec).to(device)
                    index_vec = decoder_input_args.get("index_vec", index_vec).to(device)

                vec_dict = {
                    "cad_vec": cad_vec,
                    "flag_vec": flag_vec,
                    "index_vec": index_vec,
                }
            else:
                B = len(texts)
                cad_vec = torch.tensor([[[1, 0]]], device=device, dtype=torch.long).repeat(B, 1, 1)
                flag_vec = torch.zeros((B, 1), dtype=torch.long, device=device)
                index_vec = torch.zeros((B, 1), dtype=torch.long, device=device)
                vec_dict = {
                    "cad_vec": cad_vec,
                    "flag_vec": flag_vec,
                    "index_vec": index_vec,
                }

        if mask_cad_dict is None:
            T_seq = vec_dict["cad_vec"].size(1)
            pad_idx = END_TOKEN.index("PADDING")
            cad_v = vec_dict["cad_vec"]
            if cad_v.dim() == 3:
                key_pad = (cad_v[:, :, 0] == pad_idx)
            else:
                key_pad = (cad_v == pad_idx)

            mask_cad_dict = {
                "attn_mask": generate_attention_mask(T_seq, T_seq, device=device),
                "key_padding_mask": key_pad,
            }

        S_output, _ = self.text2cad(
            vec_dict=vec_dict,
            texts=texts,
            mask_cad_dict=mask_cad_dict,
            metadata=metadata,
        )

        cmd_logits = S_output[:, :, 0, :]
        arg_logits = S_output[:, :, 1, :]
        return cmd_logits, arg_logits, None

    def test_decode(
        self,
        texts: List[str],
        maxlen: int = 272,
        nucleus_prob: float = 0.0,
        topk_index: int = 1,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if device is None:
            device = next(self.parameters()).device if list(self.parameters()) else "cpu"
        return self.text2cad.test_decode(
            texts=texts,
            maxlen=maxlen,
            nucleus_prob=nucleus_prob,
            topk_index=topk_index,
            device=device,
        )

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        texts: Optional[List[str]] = None,
        max_len: int = 272,
        **kwargs
    ):
        if texts is None and input_ids is not None:
            texts = self.base_text_embedder.tokenizer.batch_decode(
                input_ids, skip_special_tokens=True
            )
        if texts is None:
            raise ValueError("Either texts or input_ids must be provided for generate")

        device = input_ids.device if input_ids is not None else (next(self.parameters()).device if list(self.parameters()) else "cpu")
        return self.test_decode(
            texts=texts,
            maxlen=max_len,
            nucleus_prob=kwargs.get("nucleus_prob", 0.0),
            topk_index=kwargs.get("topk_index", 1),
            device=device,
        )