from __future__ import annotations

from typing import Any, Dict, Tuple, Union
import os
import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from utils.dual_seq import get_dualseq_schema

class BaseWrapper(torch.nn.Module):
    """Base class for model wrappers providing unified save and tokenization helpers."""

    def __init__(
        self,
        model: torch.nn.Module,
        text_tokenizer: PreTrainedTokenizerBase,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        # <-- Initialization -->
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model = model.to(self.device)
        self.text_tokenizer = text_tokenizer
        self.schema = get_dualseq_schema()
        self.dual_seq_schema = self.schema
        self.max_new_cmds = getattr(model, "max_new_cmds", 1024)
        self.max_new_args = getattr(model, "max_new_args", 1024)

    def extract_inputs(self, batch: Union[Dict[str, Any], Tuple]) -> Tuple[torch.Tensor, torch.Tensor, Any, Any]:
        # <-- Dict Input Parsing Guard Clause -->
        if isinstance(batch, dict):
            input_ids = batch.get("input_ids", batch.get("x"))
            attn_mask = batch.get("attention_mask", batch.get("attn_mask"))
            cmd_targets = batch.get("cmd_targets", batch.get("y", {}).get("cmd_targets") if isinstance(batch.get("y"), dict) else None)
            arg_targets = batch.get("arg_targets", batch.get("y", {}).get("arg_targets") if isinstance(batch.get("y"), dict) else None)
            return input_ids, attn_mask, cmd_targets, arg_targets

        # <-- Fallback Tuple Parsing -->
        input_ids = batch[0]
        cmd_targets = batch[1] if len(batch) > 1 else None
        arg_targets = batch[2] if len(batch) > 2 else None
        attn_mask = batch[3] if len(batch) > 3 else (input_ids != 0).long()
        return input_ids, attn_mask, cmd_targets, arg_targets

    def save(self, folder_path: str) -> None:
        # <-- Save Checkpoints -->
        os.makedirs(folder_path, exist_ok=True)
        if hasattr(self.model, "encoder") and self.model.encoder is not None:
            torch.save(self.model.encoder.state_dict(), os.path.join(folder_path, "encoder.pt"))
        if hasattr(self.model, "adaptive_layer") and self.model.adaptive_layer is not None:
            torch.save(self.model.adaptive_layer.state_dict(), os.path.join(folder_path, "adaptive_layer.pt"))
        torch.save(self.model.state_dict(), os.path.join(folder_path, "checkpoint.pt"))

    def tokenize_input(self, input_text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        # <-- Tokenize Prompt Input -->
        device = next(self.model.parameters()).device
        max_len = self.text_tokenizer.model_max_length or 512
        tokenized = self.text_tokenizer(input_text, truncation=True, max_length=max_len)
        input_ids = torch.as_tensor(tokenized['input_ids'], dtype=torch.long).unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids)
        return input_ids, attention_mask
