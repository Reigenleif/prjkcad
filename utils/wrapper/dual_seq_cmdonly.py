import os

import torch
from typing import Any, Mapping, Callable
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils.dual_seq import get_dualseq_schema

class DualSeqCMDOnlyWrapper(torch.nn.Module):
    """A wrapper for the DualSeqCMDOnly model for inference and training"""
    
    def __init__(self, 
                 model: torch.nn.Module,
                 text_tokenizer,
                 device = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.model = model.to(device)
        self.text_tokenizer = text_tokenizer
        self.dual_seq_schema = get_dualseq_schema()

        self.max_new_cmds = model.max_new_cmds

    def forward(self, batch, is_teacher_forcing: bool = False):
        """
        Forward pass for both training and inference with autoregressive style.
        """

        input_ids, cmds, attention_mask = batch
        device = input_ids.device
        B = input_ids.size(0)
        
        _, enc_out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        if is_teacher_forcing:
            # Build decoder inputs: [SOS, c0, c1, c2, ...]
            sos = torch.full((B, 1), self.model.sos_id, device=device, dtype=cmds.dtype)
            decoder_input_ids = torch.cat([sos, cmds[:, :-1]], dim=1)

            logits, _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                encoder_out_embeddings=enc_out,
            )

            preds = logits.argmax(dim=-1)
            return logits, preds
        
        # Autoregressive generation (non-teacher forcing, greedy decoding)
        preds = torch.full((B, 1), self.model.sos_id, device=device, dtype=torch.long)
        outs = []

        for _ in range(self.max_new_cmds):
            logits, _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=preds,
                encoder_out_embeddings=enc_out,
            )

            next_logits = logits[:, -1:, :]                 # (B, 1, V)
            next_token = next_logits.argmax(dim=-1)         # (B, 1)

            outs.append(next_logits)
            preds = torch.cat([preds, next_token], dim=1)

            # stop if all sequences hit EOS
            if (next_token.squeeze(-1) == self.model.eos_id).all():
                break

        outs = torch.cat(outs, dim=1) if outs else torch.empty(0, device=device)
        return outs, preds
    
    @torch.no_grad()
    def generate(
        self,
        input_text,
        max_new_tokens=50,
    ):
        """
        Generator for autoregressive inference.
        input: text 
        output: generated command sequence
        """
        self.model.eval()
        
        # Tokenize input text
        input_ids = torch.as_tensor(self.text_tokenizer(input_text)['input_ids'], dtype=torch.long).unsqueeze(0).to(next(self.model.parameters()).device)
        attention_mask = torch.ones_like(input_ids)
        batch = (input_ids, None, attention_mask)
        logits, preds = self.forward(batch)
        
        # Convert predicted token ids to command names
        id_to_command = {v: k for k, v in self.dual_seq_schema["command_to_id"].items()}
        generated_cmds = []
        for token_id in preds[0].tolist():
            if token_id == self.model.pad_id:
                break
            command_name = id_to_command.get(token_id, None)
            if command_name is not None:
                generated_cmds.append(command_name)
            if len(generated_cmds) >= max_new_tokens:
                break
            
        return generated_cmds
    
    def save(self, folder_path):
        """
        Saves model to the specified folder path.
        """
        
        os.makedirs(folder_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(folder_path, "checkpoint.pt"))

    @classmethod
    def from_pretrained(cls, 
                        model_cls,
                        pretrained_model_path, 
                        seq2seq_model_name="t5-small", 
                        device="cuda" if torch.cuda.is_available() else "cpu", 
                        model_kwargs: dict[str, Any] = None):
        """
        Loads model from the specified folder path and pretrained model name.
        """
        
        
        hf_model = AutoModelForSeq2SeqLM.from_pretrained(seq2seq_model_name)
        hf_tokenizer = AutoTokenizer.from_pretrained(seq2seq_model_name)
        
        # Extract the encoder and decoder from the pretrained model
        encoder = hf_model.get_encoder()
        decoder = hf_model.get_decoder()
        
        # Initialize the model with the pretrained encoder and decoder
        model: torch.nn.Module = model_cls(
            **(model_kwargs or {})
        )
        
        text_tokenizer = hf_tokenizer
        
        # Load the saved state dict
        state_dict = torch.load(pretrained_model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        
        wrapper = cls(model=model, text_tokenizer=text_tokenizer)
        wrapper.to(device)
    
        print(f"Model loaded from {pretrained_model_path}")
        return wrapper
        
    def train(self, mode=True):
        self.model.train(mode)
        
    def eval(self):
        self.model.eval()
        
    def to(self, device):
        self.model.to(device)
        return self
    
    def half(self):
        self.model.half()
        return self