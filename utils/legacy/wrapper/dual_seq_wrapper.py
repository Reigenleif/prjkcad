import os
import torch
from typing import Any, Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from utils.dual_seq import get_dualseq_schema
from models import BaseModel

class DualSeqWrapper(torch.nn.Module):
    """A wrapper for the full DualSeq models for inference and training"""

    def __init__(self, 
                 model: torch.nn.Module,
                 text_tokenizer: PreTrainedTokenizerBase,
                 device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.model = model.to(device)
        self.text_tokenizer = text_tokenizer
        self.dual_seq_schema = get_dualseq_schema()
        # Max new commands (could be accessed from model if exists, otherwise fallback to schema max_len or 1024)
        self.max_new_cmds = getattr(model, "max_new_cmds", 1024)

    def forward(self, batch: Tuple, is_teacher_forcing: bool = True):
        """
        Forward pass for both training and inference.

        batch = (input_ids, cmd_targets, arg_targets, attention_mask)

        Teacher-forcing: decoder is fed the ground-truth shifted sequence.
        Autoregressive:  decoder generates step-by-step up to max_new_cmds.

        Returns (cmd_logits, cmd_preds, arg_preds).
            cmd_logits : (B, T, V)   raw logits over the command vocabulary
            cmd_preds  : (B, T)      greedy argmax command token predictions
            arg_preds  : (B, T, A)   predicted argument values
        """
        input_ids, cmd_targets, arg_targets, attention_mask = batch
        device = input_ids.device
        B = input_ids.size(0)

        _, _, enc_out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        if is_teacher_forcing:
            # Decoder input: [SOS, c0, c1, ...] (right-shift by 1, preserving length)
            # Truncate target length to max_new_cmds to obey the positional embedding limits of the decoder
            T = min(cmd_targets.size(1), self.max_new_cmds)
            cmd_targets_limited = cmd_targets[:, :T]
            sos = torch.full((B, 1), self.model.sos_id, device=device, dtype=cmd_targets.dtype)
            decoder_input_ids = torch.cat([sos, cmd_targets_limited[:, :-1]], dim=1)

            # Decoder arg input: [zeros, a0, a1, ...] (right-shift by 1, preserving length)
            n_args = arg_targets.size(-1)
            zero_args = torch.zeros((B, 1, n_args), device=device, dtype=arg_targets.dtype)
            arg_targets_limited = arg_targets[:, :T, :]
            decoder_input_args = torch.cat([zero_args, arg_targets_limited[:, :-1, :]], dim=1)

            cmd_logits, arg_preds, _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_input_args=decoder_input_args,
                encoder_out_embeddings=enc_out
            )   
            cmd_preds = cmd_logits.argmax(dim=-1)  # (B, T)
            return cmd_logits, cmd_preds, arg_preds

        # Autoregressive generation (greedy decoding)
        preds = torch.full((B, 1), self.model.sos_id, device=device, dtype=torch.long)
        n_args = arg_targets.size(-1)
        pred_args = torch.zeros((B, 1, n_args), device=device, dtype=torch.float32)

        cmd_outs = []
        cmd_pred_outs = []
        arg_outs = []

        for _ in range(self.max_new_cmds):
            cmd_logits, arg_preds_step, _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=preds,
                decoder_input_args=pred_args,
                encoder_out_embeddings=enc_out
            )

            next_cmd_logits = cmd_logits[:, -1:, :]           # (B, 1, V)
            next_cmd_token = next_cmd_logits.argmax(dim=-1)   # (B, 1)
            next_args = arg_preds_step[:, -1:, :]             # (B, 1, A)

            cmd_outs.append(next_cmd_logits)
            cmd_pred_outs.append(next_cmd_token)
            arg_outs.append(next_args)

            preds = torch.cat([preds, next_cmd_token], dim=1)
            pred_args = torch.cat([pred_args, next_args], dim=1)

            if (next_cmd_token.squeeze(-1) == self.model.eos_id).all():
                break

        cmd_logits_out = torch.cat(cmd_outs, dim=1) if cmd_outs else torch.empty(0, device=device)
        cmd_preds_out = torch.cat(cmd_pred_outs, dim=1) if cmd_pred_outs else torch.empty(0, device=device, dtype=torch.long)
        arg_preds_out = torch.cat(arg_outs, dim=1) if arg_outs else torch.empty(0, device=device)
        return cmd_logits_out, cmd_preds_out, arg_preds_out
    
    @torch.no_grad()
    def generate(self, input_text, max_new_tokens=50):
        """
        Generator for autoregressive inference for both commands and arguments.
        output: list of tuples (command_name, arg_dict)
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        
        # Tokenize input text
        max_len = self.text_tokenizer.model_max_length
        if max_len is None:
            max_len = 512
        tokenized = self.text_tokenizer(input_text, truncation=True, max_length=max_len)
        input_ids = torch.as_tensor(tokenized['input_ids'], dtype=torch.long).unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids)
        
        # Initial inputs: SOS token and zeros for args
        sos_id = self.dual_seq_schema["sos_id"]
        n_args = self.dual_seq_schema["n_args"]
        pad_id = self.dual_seq_schema["pad_id"]
        eos_id = self.dual_seq_schema["eos_id"]
        
        decoder_input_ids = torch.full((1, 1), sos_id, device=device, dtype=torch.long)
        decoder_input_args = torch.zeros((1, 1, n_args), device=device, dtype=torch.float32)
        
        cmd_preds = []
        arg_preds = []

        _, _, enc_out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        
        for _ in range(max_new_tokens):
            cmd_logits, arg_preds_out, _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_input_args=decoder_input_args,
                encoder_out_embeddings=enc_out
            )
            
            # Predict next token and args
            next_cmd_logits = cmd_logits[:, -1:, :]
            next_cmd_token = next_cmd_logits.argmax(dim=-1)
            next_args = arg_preds_out[:, -1:, :]
            
            # Append predictions
            decoder_input_ids = torch.cat([decoder_input_ids, next_cmd_token], dim=1)
            decoder_input_args = torch.cat([decoder_input_args, next_args], dim=1)

            cmd_val = next_cmd_token.item()
            cmd_preds.append(cmd_val)
            arg_preds.append(next_args.squeeze().tolist())
            
            if cmd_val == eos_id:
                break
                
        # Parse output into schema
        id_to_command = {v: k for k, v in self.dual_seq_schema["command_to_id"].items()}
        arg_names = self.dual_seq_schema["arg_names"]
        command_to_slice = self.dual_seq_schema["command_to_slice"]
        
        generated_sequence = []
        for cmd_id, arg_vals in zip(cmd_preds, arg_preds):
            if cmd_id in (pad_id, eos_id):
                break
            command_name = id_to_command.get(cmd_id, None)
            if command_name is None or command_name in ("SOS", "EOS", "PAD"):
                continue
                
            arg_dict = {}
            if command_name in command_to_slice:
                start, end = command_to_slice[command_name]
                cmd_arg_names = arg_names[start:end]
                cmd_arg_vals = arg_vals[start:end]
                for name, val in zip(cmd_arg_names, cmd_arg_vals):
                    arg_dict[name] = val
                    
            generated_sequence.append((command_name, arg_dict))
            
        return generated_sequence

    def save(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        if hasattr(self.model, "encoder") and self.model.encoder is not None:
            torch.save(self.model.encoder.state_dict(), os.path.join(folder_path, "encoder.pt"))
        if hasattr(self.model, "adaptive_layer") and self.model.adaptive_layer is not None:
            torch.save(self.model.adaptive_layer.state_dict(), os.path.join(folder_path, "adaptive_layer.pt"))
        torch.save(self.model.state_dict(), os.path.join(folder_path, "checkpoint.pt"))

    @classmethod
    def from_pretrained(cls, 
                        model_cls,
                        pretrained_model_path=None, 
                        seq2seq_model_name="t5-small", 
                        device="cuda" if torch.cuda.is_available() else "cpu", 
                        model_kwargs: dict[str, Any] = None):
        if isinstance(model_cls, str):
            pretrained_model_path = model_cls
            model_cls = BaseModel

        if pretrained_model_path is None:
            raise ValueError("pretrained_model_path must be specified.")

        # Resolve checkpoint path
        if os.path.isdir(pretrained_model_path):
            checkpoint_file = os.path.join(pretrained_model_path, "checkpoint.pt")
            if not os.path.exists(checkpoint_file):
                checkpoint_file = os.path.join(pretrained_model_path, "model.pt")
            pretrained_model_path = checkpoint_file

        # Instantiate Tokenizer
        hf_tokenizer = AutoTokenizer.from_pretrained(seq2seq_model_name)
        model: torch.nn.Module = model_cls(**(model_kwargs or {}))
        
        state_dict = torch.load(pretrained_model_path, map_location="cpu")
        first_key = next(iter(state_dict.keys()))
        if first_key.startswith("model."):
            state_dict = {k[6:]: v for k, v in state_dict.items()}

        # Load Weights
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys or unexpected_keys:
            print("Warnings during load_state_dict:")
            if missing_keys:
                print(f"  Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"  Unexpected keys: {unexpected_keys}")

        # Instantiate Wrapper
        wrapper = cls(model=model, text_tokenizer=hf_tokenizer, device=device)
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

    def parameters(self):
        return self.model.parameters()
