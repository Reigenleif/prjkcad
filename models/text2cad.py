from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from transformers import BertModel, BertTokenizer

from models.text2cad_ori.decoder import CADDecoder
from models.text2cad_ori.layers.adaptive_layer import TextAdaptiveLayer
from models.text2cad_ori.layers.text_embed import prepare_cross_attention_mask_batch
from models.text2cad_ori.text2cad import Text2CAD
from utils.representations.CadSeqProc.utility.macro import (
    CAD_CLASS_INFO,
    END_TOKEN,
    MAX_CAD_SEQUENCE_LENGTH,
    N_BIT,
)
from utils.representations.CadSeqProc.utility.utils import generate_attention_mask

TEXT_CONFIG: Dict[str, Any] = {
    "text_embedder": {
        "model_name": "bert-base-uncased",
        "cache_dir": None,
        "max_seq_len": 512,
    },
    "adaptive_layer": {
        "in_dim": 768,
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


def build_flag_and_index_vecs(cad_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, _ = cad_vec.shape
    flag_vec = torch.zeros((B, T), dtype=torch.long, device=cad_vec.device)
    index_vec = torch.zeros((B, T), dtype=torch.long, device=cad_vec.device)
    
    pad_id = END_TOKEN.index("PADDING")
    start_id = END_TOKEN.index("START")
    end_ext_id = END_TOKEN.index("END_EXTRUSION")
    
    for b in range(B):
        curr_idx = 0
        curr_flag = 0
        is_extrusion = False
        ext_offset = 0
        
        for t in range(T):
            tok_0 = int(cad_vec[b, t, 0].item())
            
            if tok_0 == pad_id:
                flag_vec[b, t] = 11
                index_vec[b, t] = curr_idx
                continue
                
            if tok_0 == start_id:
                curr_flag = 0
                is_extrusion = False
                ext_offset = 0
            elif tok_0 in [7, 8, 9, 10]:
                is_extrusion = True
                ext_offset = 1
                curr_flag = 1
            elif is_extrusion:
                ext_offset += 1
                curr_flag = min(ext_offset, 10)
                if tok_0 == end_ext_id or ext_offset >= 10:
                    is_extrusion = False
                    curr_idx = min(curr_idx + 1, CAD_CLASS_INFO["index_size"] - 1)
            else:
                curr_flag = 0
                
            flag_vec[b, t] = curr_flag
            index_vec[b, t] = curr_idx
            
    return flag_vec, index_vec


class Text2CADModel(nn.Module):
    """Wrapper model for Text2CAD architecture supporting tuple and dictionary input/output formats."""

    def __init__(
        self,
        cfg: Any = None,
        vocab_size: Optional[int] = None,
        vocab_size_args: Optional[int] = None,
        text_config: Optional[Dict[str, Any]] = None,
        cad_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__()
        self.cfg = cfg
        
        # Merge configuration
        self.text_config = dict(TEXT_CONFIG)
        if text_config is not None:
            self.text_config.update(text_config)
            
        self.cad_config = dict(CAD_CONFIG)
        if cad_config is not None:
            self.cad_config.update(cad_config)
            
        model_name = getattr(getattr(cfg, "tokenizer", None), "model_name", None) or self.text_config["text_embedder"]["model_name"]
        if "large" in model_name:
            enc_dim = 1024
        else:
            enc_dim = 768
            
        self.text_config["adaptive_layer"]["in_dim"] = enc_dim
        
        # Initialize text encoder
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.encoder = BertModel.from_pretrained(self.model_name)
        
        if getattr(cfg, "freeze_encoder", True):
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        # Adaptive layer to project text embedding
        self.adaptive_layer = TextAdaptiveLayer.from_config(self.text_config["adaptive_layer"])
        
        # Transformer decoder for CAD sequence
        self.cad_decoder = CADDecoder.from_config(self.cad_config)
        self.cad_seq_len = self.cad_config["cad_seq_len"] - 1
        self.max_new_cmds = getattr(cfg, "max_new_cmds", 272) or 272
        self.attention_scores: Dict[str, Any] = {}

    def get_trainable_state_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.state_dict().items() if "encoder" not in k.split(".")}

    def _encode_text(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        texts: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.encoder.parameters()).device
        is_frozen = getattr(self.cfg, "freeze_encoder", True) or not any(p.requires_grad for p in self.encoder.parameters())
        
        context = torch.no_grad() if is_frozen else torch.enable_grad()
        with context:
            if input_ids is not None:
                input_ids = input_ids.to(device)
                if attention_mask is None:
                    attention_mask = (input_ids != 0).long()
                else:
                    attention_mask = attention_mask.to(device)
                outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                embedding = outputs.last_hidden_state
                key_padding_mask = (attention_mask == 0)
                return embedding, key_padding_mask
                
            if texts is not None:
                if isinstance(texts, str):
                    texts = [texts]
                tok_out = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True,
                ).to(device)
                outputs = self.encoder(**tok_out)
                embedding = outputs.last_hidden_state
                key_padding_mask = (tok_out["attention_mask"] == 0)
                return embedding, key_padding_mask
                
        raise ValueError("Either input_ids or texts must be provided to encode text.")

    def forward(
        self,
        input_ids: Any = None,
        attention_mask: Any = None,
        decoder_input_ids: Any = None,
        vec_dict: Any = None,
        mask_cad_dict: Any = None,
        texts: Any = None,
        metadata: bool = False,
        *args,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor, Optional[Dict[str, Any]]], torch.Tensor]:
        # Tuple batch unpack handling
        if isinstance(input_ids, (tuple, list)):
            batch = input_ids
            input_ids = batch[0] if len(batch) > 0 else None
            if len(batch) >= 4 and isinstance(batch[1], torch.Tensor) and batch[1].dim() == 3:
                decoder_input_ids = batch[1]
                attention_mask = batch[2]
            elif len(batch) >= 3 and isinstance(batch[1], torch.Tensor) and batch[1].dim() == 3:
                decoder_input_ids = batch[1]
                attention_mask = batch[2] if batch[2].dim() == 2 else None
            elif len(batch) > 1:
                decoder_input_ids = batch[1]
                attention_mask = batch[3] if len(batch) > 3 else None

        device = next(self.cad_decoder.parameters()).device
        
        # 1. Text embedding
        T, key_padding_mask = self._encode_text(input_ids=input_ids, attention_mask=attention_mask, texts=texts)
        T = T.to(device)
        key_padding_mask = key_padding_mask.to(device)
        
        # 2. Extract cad_vec
        cad_vec = None
        flag_vec = None
        index_vec = None
        
        if isinstance(vec_dict, dict):
            cad_vec = vec_dict.get("cad_vec")
            flag_vec = vec_dict.get("flag_vec")
            index_vec = vec_dict.get("index_vec")
        elif decoder_input_ids is not None:
            if isinstance(decoder_input_ids, dict):
                cad_vec = decoder_input_ids.get("cad_vec")
                flag_vec = decoder_input_ids.get("flag_vec")
                index_vec = decoder_input_ids.get("index_vec")
            elif isinstance(decoder_input_ids, torch.Tensor):
                if decoder_input_ids.dim() == 3 and decoder_input_ids.shape[1] == 2 and decoder_input_ids.shape[2] != 2:
                    # Shape (B, 2, T) -> Transpose to (B, T, 2)
                    cad_vec = decoder_input_ids.permute(0, 2, 1).contiguous()
                else:
                    cad_vec = decoder_input_ids

        if cad_vec is None:
            raise ValueError("CAD sequence vector must be provided for Text2CAD forward pass.")
            
        cad_vec = cad_vec.to(device)
        cad_seq_len = cad_vec.shape[1]
        
        if flag_vec is None or index_vec is None:
            flag_vec, index_vec = build_flag_and_index_vecs(cad_vec)
        else:
            flag_vec = flag_vec.to(device)
            index_vec = index_vec.to(device)
            
        vec_dict_built = {
            "cad_vec": cad_vec,
            "flag_vec": flag_vec,
            "index_vec": index_vec,
        }
        
        if mask_cad_dict is None or not isinstance(mask_cad_dict, dict):
            mask_cad_dict = {
                "attn_mask": generate_attention_mask(cad_seq_len, cad_seq_len, device=device),
                "key_padding_mask": (cad_vec[:, :, 0] == END_TOKEN.index("PADDING")),
            }
        else:
            mask_cad_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in mask_cad_dict.items()}

        # 3. Cross attention mask
        ca_mask = {
            "attn_mask": prepare_cross_attention_mask_batch(key_padding_mask, cad_seq_len=cad_seq_len),
            "key_padding_mask": key_padding_mask,
        }
        
        # 4. Adaptive layer
        T, text_attn_scores = self.adaptive_layer(
            T,
            {
                "attn_mask": None,
                "key_padding_mask": ca_mask["key_padding_mask"],
            },
            metadata,
        )
        # 5. CAD decoder
        S_output, cad_attn_scores = self.cad_decoder(
            vec_dict_built, T, mask_cad_dict, ca_mask, metadata
        )
        if metadata:
            if text_attn_scores is not None:
                self.attention_scores.update(text_attn_scores)
            if cad_attn_scores is not None:
                self.attention_scores.update(cad_attn_scores)
            return S_output, self.attention_scores
        return S_output, None

    def test_decode(
        self,
        texts: Union[str, List[str]],
        maxlen: int = 50,
        nucleus_prob: float = 0.0,
        topk_index: int = 1,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        if device is None:
            device = next(self.cad_decoder.parameters()).device
        device = torch.device(device)
        
        if isinstance(texts, str):
            texts = [texts]
            
        ZE, key_padding_mask = self._encode_text(texts=texts)
        ZE = ZE.to(device)
        key_padding_mask = key_padding_mask.to(device)
        
        ca_mask = {
            "attn_mask": prepare_cross_attention_mask_batch(key_padding_mask, cad_seq_len=1),
            "key_padding_mask": key_padding_mask,
        }
        ZE, _ = self.adaptive_layer(
            ZE,
            {
                "attn_mask": None,
                "key_padding_mask": ca_mask["key_padding_mask"],
            },
            False,
        )
        S_output = self.cad_decoder.decode(
            ZE=ZE,
            cross_attn_mask_dict=ca_mask,
            maxlen=maxlen,
            nucleus_prob=nucleus_prob,
            topk_index=topk_index,
            device=device,
        )
        if isinstance(S_output, dict) and "cad_vec" in S_output:
            return S_output["cad_vec"]
        return S_output

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        texts: Optional[Union[str, List[str]]] = None,
        max_new_tokens: int = 50,
        **kwargs,
    ) -> torch.Tensor:
        device = next(self.cad_decoder.parameters()).device
        if texts is not None:
            return self.test_decode(texts=texts, maxlen=max_new_tokens, device=device)
            
        if input_ids is not None:
            decoded_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            return self.test_decode(texts=decoded_texts, maxlen=max_new_tokens, device=device)
            
        raise ValueError("Either texts or input_ids must be provided for generate.")
