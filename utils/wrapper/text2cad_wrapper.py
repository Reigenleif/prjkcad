from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from utils.dual_seq import DualSeq
from utils.representations.CadSeqProc.cad_sequence import CADSequence
from utils.representations.CadSeqProc.utility.macro import END_TOKEN, MAX_CAD_SEQUENCE_LENGTH, N_BIT
from utils.representations.CadSeqProc.utility.utils import generate_attention_mask
from utils.representations.converter import cadseq_to_vec_2t, dualseq_to_cadseq, vec_2t_to_cadseq
from utils.wrapper.base_wrapper import BaseWrapper


class Text2CADWrapper(BaseWrapper):
    """Wrapper for Text2CAD model supporting training pipeline and CADSequence vector outputs."""

    def __init__(
        self,
        model: torch.nn.Module,
        text_tokenizer: PreTrainedTokenizerBase,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ):
        super().__init__(model, text_tokenizer, device)
        self.model = model.to(self.device)
        self.out_type = "Text2CAD"

    def extract_inputs(self, batch: Union[Dict[str, Any], Tuple]) -> Tuple[torch.Tensor, torch.Tensor, Any, Any]:
        device = next(self.model.parameters()).device
        _to_dev = lambda t: t.to(device) if isinstance(t, torch.Tensor) else t

        if isinstance(batch, dict):
            input_ids = batch.get("input_ids", batch.get("x"))
            attn_mask = batch.get("attention_mask", batch.get("attn_mask"))
            cad_targets = batch.get("cad_targets", batch.get("target", batch.get("decoder_input_ids")))
            return _to_dev(input_ids), _to_dev(attn_mask), _to_dev(cad_targets), None

        # Tuple batch parsing
        input_ids = batch[0]
        if len(batch) >= 4 and isinstance(batch[1], torch.Tensor) and batch[1].dim() == 3:
            # (input_ids, cad_targets, attention_mask, extra_info)
            cad_targets = batch[1]
            attn_mask = batch[2] if isinstance(batch[2], torch.Tensor) else (input_ids != 0).long()
            return _to_dev(input_ids), _to_dev(attn_mask), _to_dev(cad_targets), None
        elif len(batch) >= 3 and isinstance(batch[1], torch.Tensor) and batch[1].dim() == 3:
            # (input_ids, cad_targets, attention_mask)
            cad_targets = batch[1]
            attn_mask = batch[2] if isinstance(batch[2], torch.Tensor) and batch[2].dim() == 2 else (input_ids != 0).long()
            return _to_dev(input_ids), _to_dev(attn_mask), _to_dev(cad_targets), None
        elif len(batch) > 3 and isinstance(batch[3], torch.Tensor):
            # (input_ids, cmd_targets, arg_targets, attention_mask, ...)
            cmd_targets = batch[1]
            arg_targets = batch[2]
            attn_mask = batch[3]
            return _to_dev(input_ids), _to_dev(attn_mask), _to_dev(cmd_targets), _to_dev(arg_targets)

        attn_mask = (input_ids != 0).long()
        cad_targets = batch[1] if len(batch) > 1 else None
        return _to_dev(input_ids), _to_dev(attn_mask), _to_dev(cad_targets), None

    def forward(self, batch: Any, is_teacher_forcing: bool = True) -> Dict[str, Any]:
        pad_id = END_TOKEN.index("PADDING")
        input_ids, attn_mask, cad_targets, arg_targets = self.extract_inputs(batch)
        
        # Handle dict format
        if isinstance(batch, dict) and "texts" in batch:
            texts = batch["texts"]
            if is_teacher_forcing:
                out = self.model(texts=texts, decoder_input_ids=cad_targets)
            else:
                out = self.model.generate(texts=texts, max_new_tokens=self.max_new_cmds)
        elif is_teacher_forcing:
            out = self.model(input_ids=input_ids, attention_mask=attn_mask, decoder_input_ids=cad_targets)
        else:
            out = self.model.generate(input_ids=input_ids, attention_mask=attn_mask, max_new_tokens=self.max_new_cmds)

        if not is_teacher_forcing:
            # Autoregressive rollout prediction
            gen_cad = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(gen_cad, torch.Tensor) and gen_cad.dim() == 4:
                cmd_preds = torch.argmax(gen_cad[:, :, 0, :], dim=-1)
                arg_preds = torch.argmax(gen_cad[:, :, 1, :], dim=-1)
            elif isinstance(gen_cad, torch.Tensor) and gen_cad.dim() == 3:
                cmd_preds = gen_cad[:, :, 0]
                arg_preds = gen_cad[:, :, 1]
            else:
                cmd_preds = gen_cad
                arg_preds = gen_cad
                
            return {
                "pred": gen_cad,
                "cmd_preds": cmd_preds,
                "arg_preds": arg_preds,
            }

        # Teacher forcing forward
        if isinstance(out, tuple):
            S_output = out[0]
        else:
            S_output = out

        if cad_targets is not None:
            if cad_targets.dim() == 3 and cad_targets.shape[1] == 2 and cad_targets.shape[2] != 2:
                target = cad_targets.permute(0, 2, 1)[:, 1:, :].contiguous()
            else:
                target = cad_targets[:, 1:, :].contiguous()
        else:
            target = torch.zeros((S_output.shape[0], S_output.shape[1] - 1, 2), dtype=torch.long, device=S_output.device)

        min_len = min(S_output.shape[1] - 1, target.shape[1])
        pred = S_output[:, :min_len, :, :]
        target = target[:, :min_len, :]
        key_mask = (target[:, :, 0] != pad_id).float()

        loss_dict = {
            "pred": pred,
            "target": target,
            "key_padding_mask": key_mask,
            "cmd_preds": torch.argmax(pred[:, :, 0, :], dim=-1),
            "arg_preds": torch.argmax(pred[:, :, 1, :], dim=-1),
        }
        return loss_dict

    @torch.no_grad()
    def generate(
        self,
        input_text: Union[str, List[str]],
        max_new_tokens: int = 50,
    ) -> Union[Optional[CADSequence], List[Optional[CADSequence]]]:
        self.model.eval()
        is_single = isinstance(input_text, str)
        texts = [input_text] if is_single else input_text
        
        if hasattr(self.model, "test_decode"):
            S_output = self.model.test_decode(
                texts=texts,
                maxlen=max_new_tokens,
                nucleus_prob=0.0,
                topk_index=1,
                device=self.device,
            )
        else:
            S_output = self.model.generate(texts=texts, max_new_tokens=max_new_tokens)

        if isinstance(S_output, dict) and "cad_vec" in S_output:
            S_output = S_output["cad_vec"]

        results = []
        for i in range(S_output.shape[0]):
            cad_vec = S_output[i].detach().cpu()
            cad_seq = vec_2t_to_cadseq(cad_vec)
            results.append(cad_seq)

        if is_single:
            return results[0]
        return results

    def infer(self, input_text: str, max_new_tokens: int = 50) -> Optional[CADSequence]:
        return self.generate(input_text, max_new_tokens=max_new_tokens)

    def infer_batch(self, input_texts: List[str], batch_size: int = 32, max_new_tokens: int = 50) -> List[Optional[CADSequence]]:
        results = []
        for i in range(0, len(input_texts), batch_size):
            chunk = input_texts[i : i + batch_size]
            res = self.generate(chunk, max_new_tokens=max_new_tokens)
            if isinstance(res, list):
                results.extend(res)
            else:
                results.append(res)
        return results
