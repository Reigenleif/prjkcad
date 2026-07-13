import os
import torch
from typing import Any, Tuple
from transformers import AutoTokenizer
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
        self.schema = get_dualseq_schema()
        self.max_new_cmds = getattr(model, "max_new_cmds", 1024)
        self.max_new_args = getattr(model, "max_new_args", 1024)

    def forward(self, batch: Tuple, is_teacher_forcing: bool = True):
        input_ids, cmd_targets, arg_targets, attention_mask = batch
        device = input_ids.device
        B = input_ids.size(0)

        _, _, enc_out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        if is_teacher_forcing:
            T_cmd = min(cmd_targets.size(1), self.max_new_cmds)
            cmd_targets_limited = cmd_targets[:, :T_cmd]
            cmd_sos = torch.full((B, 1), self.model.sos_id, device=device, dtype=cmd_targets.dtype)
            decoder_input_ids = torch.cat([cmd_sos, cmd_targets_limited[:, :-1]], dim=1)

            T_arg = min(arg_targets.size(1), self.max_new_args)
            arg_targets_limited = arg_targets[:, :T_arg]
            arg_sos = torch.full((B, 1), self.model.arg_sos_id, device=device, dtype=arg_targets.dtype)
            decoder_input_args = torch.cat([arg_sos, arg_targets_limited[:, :-1]], dim=1)

            cmd_logits, arg_logits, _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_input_args=decoder_input_args,
                encoder_out_embeddings=enc_out
            )   
            cmd_preds = cmd_logits.argmax(dim=-1)
            arg_preds = arg_logits.argmax(dim=-1)
            return cmd_logits, arg_logits, cmd_preds, arg_preds

        # Autoregressive generation
        cmd_preds_seq = torch.full((B, 1), self.model.sos_id, device=device, dtype=torch.long)
        arg_preds_seq = torch.full((B, 1), self.model.arg_sos_id, device=device, dtype=torch.long)

        cmd_outs = []
        cmd_pred_outs = []
        arg_outs = []
        arg_pred_outs = []

        cmd_done = torch.zeros(B, dtype=torch.bool, device=device)
        arg_done = torch.zeros(B, dtype=torch.bool, device=device)

        max_steps = max(self.max_new_cmds, self.max_new_args)

        for step in range(max_steps):
            if cmd_done.all() and arg_done.all():
                break

            cmd_logits, arg_logits, _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=cmd_preds_seq,
                decoder_input_args=arg_preds_seq,
                encoder_out_embeddings=enc_out
            )

            next_cmd_logits = cmd_logits[:, -1:, :]
            next_cmd_token = next_cmd_logits.argmax(dim=-1)
            
            next_arg_logits = arg_logits[:, -1:, :]
            next_arg_token = next_arg_logits.argmax(dim=-1)

            next_cmd_token[cmd_done] = self.model.pad_id
            next_arg_token[arg_done] = self.model.arg_pad_id

            if step < self.max_new_cmds:
                cmd_outs.append(next_cmd_logits)
                cmd_pred_outs.append(next_cmd_token)
                cmd_preds_seq = torch.cat([cmd_preds_seq, next_cmd_token], dim=1)
                cmd_done |= (next_cmd_token.squeeze(-1) == self.model.eos_id)
            
            if step < self.max_new_args:
                arg_outs.append(next_arg_logits)
                arg_pred_outs.append(next_arg_token)
                arg_preds_seq = torch.cat([arg_preds_seq, next_arg_token], dim=1)
                arg_done |= (next_arg_token.squeeze(-1) == self.model.arg_eos_id)

        cmd_logits_out = torch.cat(cmd_outs, dim=1) if cmd_outs else torch.empty(0, device=device)
        cmd_preds_out = torch.cat(cmd_pred_outs, dim=1) if cmd_pred_outs else torch.empty(0, device=device, dtype=torch.long)
        arg_logits_out = torch.cat(arg_outs, dim=1) if arg_outs else torch.empty(0, device=device)
        arg_preds_out = torch.cat(arg_pred_outs, dim=1) if arg_pred_outs else torch.empty(0, device=device, dtype=torch.long)

        return cmd_logits_out, arg_logits_out, cmd_preds_out, arg_preds_out
    
    @torch.no_grad()
    def generate(self, input_text, max_new_tokens=50):
        self.model.eval()
        device = next(self.model.parameters()).device
        
        max_len = self.text_tokenizer.model_max_length
        if max_len is None:
            max_len = 512
        tokenized = self.text_tokenizer(input_text, truncation=True, max_length=max_len)
        input_ids = torch.as_tensor(tokenized['input_ids'], dtype=torch.long).unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids)
        
        _, _, cmd_preds, arg_preds = self.forward((input_ids, None, None, attention_mask), is_teacher_forcing=False)
        
        from utils.dual_seq import DualSeq
        
        cmd_list = DualSeq.id_to_cmds(cmd_preds[0].cpu().numpy().tolist())
        arg_tokens = arg_preds[0].cpu().numpy().tolist()
        
        # Trim eos and pad
        try:
            cmd_eos_idx = cmd_list.index("EOS")
            cmd_list = cmd_list[:cmd_eos_idx]
        except ValueError:
            pass
        
        try:
            arg_eos_idx = arg_tokens.index(self.model.arg_eos_id)
            arg_tokens = arg_tokens[:arg_eos_idx]
        except ValueError:
            pass
            
        instance = DualSeq.from_sequences(cmd_list, arg_tokens)
        
        return list(zip(instance.cmds, instance.args_dict))

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

        if os.path.isdir(pretrained_model_path):
            checkpoint_file = os.path.join(pretrained_model_path, "checkpoint.pt")
            if not os.path.exists(checkpoint_file):
                checkpoint_file = os.path.join(pretrained_model_path, "model.pt")
            pretrained_model_path = checkpoint_file

        hf_tokenizer = AutoTokenizer.from_pretrained(seq2seq_model_name)
        model: torch.nn.Module = model_cls(**(model_kwargs or {}))
        
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
