from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple, Union, TYPE_CHECKING
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
if TYPE_CHECKING:
    from models.base_model import BaseModel
from utils.wrapper.base_wrapper import BaseWrapper

class ReconstructorModel(nn.Module):
    """Reconstructor mapping: encoder -> adaptive_layer -> (mu/logvar) -> reconstructor head."""

    def __init__(self, encoder: nn.Module, adaptive_layer: nn.Module, vocab_size: int, d_model: int):
        super().__init__()
        # <-- Layers Init -->
        self.encoder = encoder
        self.adaptive_layer = adaptive_layer
        self.mu_proj = nn.Linear(d_model, d_model)
        self.logvar_proj = nn.Linear(d_model, d_model)
        self.reconstructor_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # <-- Forward Pass -->
        hidden = self.adaptive_layer(self.encoder(input_ids, attention_mask))
        mu = self.mu_proj(hidden)
        logvar = self.logvar_proj(hidden)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        logits = self.reconstructor_head(z)
        return logits, mu, logvar

class PretrainWrapper(BaseWrapper):
    """Wrapper for autoencoder-like pretraining."""

    def __init__(
        self,
        model: nn.Module,
        text_tokenizer: PreTrainedTokenizerBase,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__(model, text_tokenizer, device)
        # <-- Reconstructor Setup -->
        d_model = getattr(model, "d_model", 512)
        vocab_size = getattr(text_tokenizer, "vocab_size", 32100)
        self.reconstructor = ReconstructorModel(
            encoder=model.encoder,
            adaptive_layer=model.adaptive_layer,
            vocab_size=vocab_size,
            d_model=d_model
        ).to(self.device)

    def forward(self, batch: Union[Dict[str, Any], Tuple], is_teacher_forcing: bool = True) -> Dict[str, torch.Tensor]:
        # <-- Input Unpacking -->
        if isinstance(batch, dict):
            input_ids = batch.get("input_ids", batch.get("x"))
            attention_mask = batch.get("attention_mask", batch.get("attn_mask"))
        else:
            input_ids, attention_mask = batch[0], batch[1]

        # <-- Reconstructor Forward -->
        logits, mu, logvar = self.reconstructor(input_ids, attention_mask)
        return {"logits": logits, "mu": mu, "logvar": logvar}

    @torch.no_grad()
    def generate(self, input_text: Union[str, List[str]], deterministic: bool = True, skip_special_tokens: bool = True) -> Union[str, List[str]]:
        # <-- Evaluation Mode & Tokenization -->
        self.reconstructor.eval()
        max_len = self.text_tokenizer.model_max_length or 512
        is_single = isinstance(input_text, str)
        texts = [input_text] if is_single else list(input_text)

        tokenized = self.text_tokenizer(texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        input_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)

        # <-- Latent Decoding -->
        hidden = self.reconstructor.adaptive_layer(self.reconstructor.encoder(input_ids, attention_mask))
        mu = self.reconstructor.mu_proj(hidden)
        if deterministic:
            z = mu
        else:
            logvar = self.reconstructor.logvar_proj(hidden)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

        pred_ids = self.reconstructor.reconstructor_head(z).argmax(dim=-1)
        decoded = self.text_tokenizer.batch_decode(pred_ids, skip_special_tokens=skip_special_tokens)
        return decoded[0] if is_single else decoded
