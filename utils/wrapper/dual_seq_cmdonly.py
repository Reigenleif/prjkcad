import os

import torch
from typing import Any, Mapping, Callable
from models import T5EncT5DecCAD
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils.dual_seq import get_dualseq_schema

class DualSeqCMDOnlyWrapper(torch.nn.Module):
    """A wrapper for the DualSeqCMDOnly model for inference and training"""
    
    def __init__(self, 
                 model:T5EncT5DecCAD,
                 text_tokenizer,
                 device = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.model = model.to(device)
        self.text_tokenizer = text_tokenizer
        self.dual_seq_schema = get_dualseq_schema()

        self.max_new_cmds = model.max_new_cmds

    def forward(self, batch: Mapping[str, Any], ratio: float):
        """
        Forward pass for both training and inference with autoregressive style.
        """

        input_ids, cmds, attention_mask = batch
        logits0, dec_out, enc_out =  self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=cmds if ratio > 0 else None,
        )
        outs = logits0
        preds = logits0.argmax(dim=-1)

        while outs.shape[1] < self.max_new_cmds:
            # Stop at EOS
            if (preds[:, -1] == self.model.pad_id).all():
                break

            logits, dec_out, _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=preds,
                encoder_out_embeddings=enc_out,
                decoder_out_embeddings=dec_out,
            )
            # Update
            outs = torch.cat([outs, logits[:, -1:, :]], dim=1)
            preds = torch.cat([preds, logits[:, -1:, :].argmax(dim=-1)], dim=1)

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
        logits, preds = self.forward(batch, ratio=0.0)
        
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
        torch.save(self.model.state_dict(), os.path.join(folder_path, "model.pt"))

    @classmethod
    def from_pretrained(cls, 
                        folder_path, 
                        pretrained_model_name, 
                        device="cuda" if torch.cuda.is_available() else "cpu", 
                        model_config: Mapping[str, Any] = None):
        """
        Loads model from the specified folder path and pretrained model name.
        """
        
        
        pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name)
        pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        
        # Extract the encoder and decoder from the pretrained model
        encoder = pretrained_model.get_encoder()
        decoder = pretrained_model.get_decoder()
        
        # Initialize the T5EncT5DecCAD model with the pretrained encoder and decoder
        model = T5EncT5DecCAD(
            t5_encoder=encoder,
            t5_decoder=decoder,
            n_cmds=len(get_dualseq_schema()["command_to_id"]),
            **(model_config or {})
        )
        
        text_tokenizer = pretrained_tokenizer
        
        # Load the saved state dict
        state_dict = torch.load(os.path.join(folder_path, "model.pt"), map_location="cpu")
        model.load_state_dict(state_dict)
        
        wrapper = cls(model=model, text_tokenizer=text_tokenizer)
        wrapper.to(device)
    
        print(f"Model loaded from {folder_path}")
        return wrapper
        
    def train(self, mode=True):
        self.model.train(mode)
        
    def eval(self):
        self.model.eval()
        
    def to(self, device):
        self.model.to(device)
        return self