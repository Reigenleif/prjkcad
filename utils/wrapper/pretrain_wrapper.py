import os
import torch
import torch.nn as nn
from typing import Any
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class ReconstructorModel(nn.Module):
    """
    Reconstructor Model mapping:
    encoder -> adaptive_layer -> (mu/logvar projection + sampling) -> reconstructor head
    """
    def __init__(self, encoder: nn.Module, adaptive_layer: nn.Module, vocab_size: int, d_model: int):
        super().__init__()
        self.encoder = encoder
        self.adaptive_layer = adaptive_layer

        self.mu_proj = nn.Linear(d_model, d_model)
        self.logvar_proj = nn.Linear(d_model, d_model)
        self.reconstructor_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoder_hidden_states = self.encoder(input_ids, attention_mask)
        encoder_hidden_states = self.adaptive_layer(encoder_hidden_states)

        mu = self.mu_proj(encoder_hidden_states)
        logvar = self.logvar_proj(encoder_hidden_states)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        logits = self.reconstructor_head(z)
        return logits, mu, logvar


class PretrainWrapper(nn.Module):
    """
    Wrapper for autoencoder-like pretraining.
    Extracts the encoder and adaptive layer from the main model.
    """
    def __init__(self, 
                 model: nn.Module,
                 text_tokenizer: PreTrainedTokenizerBase,
                 device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.text_tokenizer = text_tokenizer
        self.device = device

        self.encoder = model.encoder
        self.adaptive_layer = model.adaptive_layer

        d_model = getattr(model, "d_model", 512)
        vocab_size = getattr(text_tokenizer, "vocab_size", 32100)

        self.reconstructor = ReconstructorModel(
            encoder=self.encoder,
            adaptive_layer=self.adaptive_layer,
            vocab_size=vocab_size,
            d_model=d_model
        ).to(device)

    def forward(self, batch, is_teacher_forcing: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids, attention_mask, target_ids = batch
        logits, mu, logvar = self.reconstructor(input_ids, attention_mask)
        return logits, mu, logvar

    @torch.no_grad()
    def generate(self, input_text: str | list[str], deterministic: bool = True, skip_special_tokens: bool = True) -> str | list[str]:
        self.reconstructor.eval()
        
        max_len = self.text_tokenizer.model_max_length or 512
        is_single = isinstance(input_text, str)
        texts = [input_text] if is_single else list(input_text)
        
        tokenized = self.text_tokenizer(texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        input_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)
        
        encoder_hidden_states = self.reconstructor.encoder(input_ids, attention_mask)
        encoder_hidden_states = self.reconstructor.adaptive_layer(encoder_hidden_states)
        
        mu = self.reconstructor.mu_proj(encoder_hidden_states)
        if deterministic:
            z = mu
        else:
            logvar = self.reconstructor.logvar_proj(encoder_hidden_states)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            
        logits = self.reconstructor.reconstructor_head(z)
        pred_ids = logits.argmax(dim=-1)
        
        reconstructions = self.text_tokenizer.batch_decode(pred_ids, skip_special_tokens=skip_special_tokens)
        
        return reconstructions[0] if is_single else reconstructions


    def save(self, folder_path: str):
        os.makedirs(folder_path, exist_ok=True)
        torch.save(self.encoder.state_dict(), os.path.join(folder_path, "encoder.pt"))
        torch.save(self.adaptive_layer.state_dict(), os.path.join(folder_path, "adaptive_layer.pt"))
        torch.save(self.reconstructor.state_dict(), os.path.join(folder_path, "checkpoint.pt"))

    def train(self, mode: bool = True):
        self.reconstructor.train(mode)

    def eval(self):
        self.reconstructor.eval()

    def to(self, device):
        self.device = device
        self.reconstructor.to(device)
        return self

    def half(self):
        self.reconstructor.half()
        return self

    def parameters(self):
        return self.reconstructor.parameters()

    @classmethod
    def from_pretrained(cls, 
                        model_cls,
                        pretrained_model_path=None, 
                        seq2seq_model_name="t5-small", 
                        device="cuda" if torch.cuda.is_available() else "cpu", 
                        model_kwargs: dict[str, Any] = None):
        """
        Loads model from the specified folder path and pretrained model name.
        """
        from transformers import AutoTokenizer
        from models import BaseModel

        if isinstance(model_cls, str):
            pretrained_model_path = model_cls
            model_cls = BaseModel

        if pretrained_model_path is None:
            raise ValueError("pretrained_model_path must be specified.")

        hf_tokenizer = AutoTokenizer.from_pretrained(seq2seq_model_name)
        model: torch.nn.Module = model_cls(**(model_kwargs or {}))

        # Check if pretrained_model_path is a directory
        if os.path.isdir(pretrained_model_path):
            encoder_path = os.path.join(pretrained_model_path, "encoder.pt")
            adaptive_layer_path = os.path.join(pretrained_model_path, "adaptive_layer.pt")
            
            # If encoder.pt and adaptive_layer.pt do not exist, fall back to checkpoint.pt / model.pt
            if not os.path.exists(encoder_path) and not os.path.exists(adaptive_layer_path):
                checkpoint_file = os.path.join(pretrained_model_path, "checkpoint.pt")
                if not os.path.exists(checkpoint_file):
                    checkpoint_file = os.path.join(pretrained_model_path, "model.pt")
                pretrained_model_path = checkpoint_file
            else:
                # Load individual encoder and adaptive layer
                if os.path.exists(encoder_path):
                    encoder_state = torch.load(encoder_path, map_location="cpu")
                    model.encoder.load_state_dict(encoder_state)
                    print(f"Loaded encoder weights from {encoder_path}")
                else:
                    print(f"Warning: encoder weights not found at {encoder_path}")

                if os.path.exists(adaptive_layer_path):
                    adaptive_layer_state = torch.load(adaptive_layer_path, map_location="cpu")
                    model.adaptive_layer.load_state_dict(adaptive_layer_state)
                    print(f"Loaded adaptive_layer weights from {adaptive_layer_path}")
                else:
                    print(f"Warning: adaptive_layer weights not found at {adaptive_layer_path}")

                wrapper = cls(model=model, text_tokenizer=hf_tokenizer, device=device)
                return wrapper

        # If it's a file, check if it's one of the files
        if pretrained_model_path.endswith("encoder.pt"):
            encoder_path = pretrained_model_path
            adaptive_layer_path = os.path.join(os.path.dirname(pretrained_model_path), "adaptive_layer.pt")
            
            encoder_state = torch.load(encoder_path, map_location="cpu")
            model.encoder.load_state_dict(encoder_state)
            print(f"Loaded encoder weights from {encoder_path}")
            
            if os.path.exists(adaptive_layer_path):
                adaptive_layer_state = torch.load(adaptive_layer_path, map_location="cpu")
                model.adaptive_layer.load_state_dict(adaptive_layer_state)
                print(f"Loaded adaptive_layer weights from {adaptive_layer_path}")
            else:
                print(f"Warning: adaptive_layer weights not found at {adaptive_layer_path}")
                
        elif pretrained_model_path.endswith("adaptive_layer.pt"):
            adaptive_layer_path = pretrained_model_path
            encoder_path = os.path.join(os.path.dirname(pretrained_model_path), "encoder.pt")
            
            adaptive_layer_state = torch.load(adaptive_layer_path, map_location="cpu")
            model.adaptive_layer.load_state_dict(adaptive_layer_state)
            print(f"Loaded adaptive_layer weights from {adaptive_layer_path}")
            
            if os.path.exists(encoder_path):
                encoder_state = torch.load(encoder_path, map_location="cpu")
                model.encoder.load_state_dict(encoder_state)
                print(f"Loaded encoder weights from {encoder_path}")
            else:
                print(f"Warning: encoder weights not found at {encoder_path}")
                
        else:
            # Full model state dict loading
            state_dict = torch.load(pretrained_model_path, map_location="cpu")
            first_key = next(iter(state_dict.keys()))
            if first_key.startswith("model."):
                state_dict = {k[6:]: v for k, v in state_dict.items()}

            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys or unexpected_keys:
                print("Warnings during load_state_dict:")
                if missing_keys:
                    print(f"  Missing keys: {missing_keys}")
                if unexpected_keys:
                    print(f"  Unexpected keys: {unexpected_keys}")
            print(f"Model loaded from {pretrained_model_path}")

        wrapper = cls(model=model, text_tokenizer=hf_tokenizer, device=device)
        return wrapper

