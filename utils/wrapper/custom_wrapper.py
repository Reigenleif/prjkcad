from __future__ import annotations

from typing import Any, Dict, Optional, Union
import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from utils.wrapper.base_wrapper import BaseWrapper
from utils.wrapper.float_args_wrapper import FloatArgsWrapper
from utils.wrapper.eight_bit_binarized_args_wrapper import EightBitBinarizedArgsWrapper
from utils.wrapper.tokenized_one_sequence_args_wrapper import TokenizedOneSequenceArgsWrapper
from utils.wrapper.pretrain_wrapper import PretrainWrapper
from utils.wrapper.grpo_wrapper import GRPOWrapper
from utils.representations.dual_seq.dual_seq import DualSeqMetadata, DualSeq

class CustomWrapper(BaseWrapper):
    """Factory controller for model wrappers based on model/pipeline configurations."""

    def __init__(
        self,
        model: torch.nn.Module,
        text_tokenizer: PreTrainedTokenizerBase,
        out_type: str = "FloatArgs",
        metadata: Optional[DualSeqMetadata] = None,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__(model, text_tokenizer, device)
        # <-- Instantiate Wrapper Variant -->
        self.wrapper = self._create_wrapper(model, text_tokenizer, out_type, metadata, device)

    def _create_wrapper(self, model, text_tokenizer, out_type, metadata, device) -> BaseWrapper:
        # <-- Out-Type Guard Clauses -->
        if out_type in ["FloatArgs", "float_args"]:
            return FloatArgsWrapper(model, text_tokenizer, device=device)
        if out_type in ["EightBitBinarizedArgs", "eight_bit"]:
            return EightBitBinarizedArgsWrapper(model, text_tokenizer, device=device, metadata=metadata)
        if out_type in ["TokenizedOneSequenceArgs", "tokenized"]:
            return TokenizedOneSequenceArgsWrapper(model, text_tokenizer, device=device)
        if out_type in ["PretrainWrapper", "pretrain"]:
            return PretrainWrapper(model, text_tokenizer, device=device)
        if out_type in ["GRPOWrapper", "grpo"]:
            return GRPOWrapper(model, text_tokenizer, device=device, metadata=metadata)

        # <-- Fallback Default -->
        return FloatArgsWrapper(model, text_tokenizer, device=device)

    def forward(self, batch: Any, is_teacher_forcing: bool = True) -> Dict[str, torch.Tensor]:
        # <-- Forward Pass Delegate -->
        return self.wrapper(batch, is_teacher_forcing=is_teacher_forcing)

    def generate(self, *args, **kwargs) -> Any:
        # <-- Generation Delegate -->
        return self.wrapper.generate(*args, **kwargs)

    def infer(self, input_text: str, max_new_tokens: int = 50) -> DualSeq:
        # <-- Inference Delegate returning DualSeq -->
        if hasattr(self.wrapper, "infer"):
            return self.wrapper.infer(input_text, max_new_tokens=max_new_tokens)
        res = self.wrapper.generate(input_text, max_new_tokens=max_new_tokens)
        if isinstance(res, DualSeq):
            return res
        if isinstance(res, list):
            return DualSeq(cmd_args_tuples=res)
        return res

