import os
import torch
from typing import Any, Tuple
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from utils.dual_seq import get_dualseq_schema, DualSeq

class Text2CADWrapper(torch.nn.Module):
    def __init__(self, 
                 model: torch.nn.Module,
                 text_tokenizer: PreTrainedTokenizerBase,
                 device=None
    ):
        super().__init__()
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.text_tokenizer = text_tokenizer
        self.vocab_size_cmd = model.vocab_size
        self.vocab_size_args = model.vocab_size_args
        self.max_new_cmds = getattr(model, "max_new_cmds", 1024)
        self.max_new_args = getattr(model, "max_new_args", 1024)
        self.schema = get_dualseq_schema()

    def forward(self, batch: Tuple, is_teacher_forcing: bool = True):
        input_ids, cmd_targets, arg_targets, attention_mask = batch
        device = input_ids.device
        B = input_ids.size(0)

        _, enc_out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        if is_teacher_forcing:
            T_cmd = cmd_targets.size(1)
            T_arg = arg_targets.size(1)
            T_max = max(T_cmd, T_arg)
            T_max = min(T_max, self.max_new_cmds)

            # Pad cmd_targets to T_max
            cmd_targets_padded = torch.full((B, T_max), self.model.pad_id, device=device, dtype=cmd_targets.dtype)
            T_cmd_limit = min(T_cmd, T_max)
            cmd_targets_padded[:, :T_cmd_limit] = cmd_targets[:, :T_cmd_limit]
            
            # Pad arg_targets to T_max
            arg_targets_padded = torch.full((B, T_max), self.model.arg_pad_id, device=device, dtype=arg_targets.dtype)
            T_arg_limit = min(T_arg, T_max)
            arg_targets_padded[:, :T_arg_limit] = arg_targets[:, :T_arg_limit]

            # Shift inputs for autoregressive decoder
            cmd_sos = torch.full((B, 1), self.model.sos_id, device=device, dtype=cmd_targets.dtype)
            decoder_input_ids = torch.cat([cmd_sos, cmd_targets_padded[:, :-1]], dim=1)

            arg_sos = torch.full((B, 1), self.model.arg_sos_id, device=device, dtype=arg_targets.dtype)
            decoder_input_args = torch.cat([arg_sos, arg_targets_padded[:, :-1]], dim=1)

            seq_logits, _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_input_args=decoder_input_args,
                encoder_out_embeddings=enc_out
            )
            
            cmd_logits = seq_logits[:, 0]
            arg_logits = seq_logits[:, 1]
            
            cmd_preds = cmd_logits.argmax(dim=-1)
            arg_preds_unified = arg_logits.argmax(dim=-1)
            arg_preds = torch.clamp(arg_preds_unified - self.vocab_size_cmd, min=0)
            
            return cmd_logits, arg_logits, cmd_preds, arg_preds

        # Autoregressive generation
        cmd_preds_seq = torch.full((B, 1), self.model.sos_id, device=device, dtype=torch.long)
        arg_preds_seq = torch.full((B, 1), self.model.arg_sos_id, device=device, dtype=torch.long)

        cmd_outs = []
        arg_outs = []
        cmd_pred_outs = []
        arg_pred_outs = []

        cmd_done = torch.zeros(B, dtype=torch.bool, device=device)
        arg_done = torch.zeros(B, dtype=torch.bool, device=device)

        for step in range(self.max_new_cmds):
            if cmd_done.all() and arg_done.all():
                break

            seq_logits, _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=cmd_preds_seq,
                decoder_input_args=arg_preds_seq,
                encoder_out_embeddings=enc_out
            )

            next_cmd_logits = seq_logits[:, 0, -1:, :]
            next_cmd_token = next_cmd_logits.argmax(dim=-1)
            
            next_arg_logits = seq_logits[:, 1, -1:, :]
            next_arg_token_unified = next_arg_logits.argmax(dim=-1)
            next_arg_token = torch.clamp(next_arg_token_unified - self.vocab_size_cmd, min=0)

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
        arg_logits_out = torch.cat(arg_outs, dim=1) if arg_outs else torch.empty(0, device=device)
        cmd_preds_out = torch.cat(cmd_pred_outs, dim=1) if cmd_pred_outs else torch.empty(0, device=device, dtype=torch.long)
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
        
        cmd_list = DualSeq.id_to_cmds(cmd_preds[0].cpu().numpy().tolist())
        arg_tokens = arg_preds[0].cpu().numpy().tolist()
        
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
        
    def train(self, mode=True):
        self.model.train(mode)
        
    def eval(self):
        self.model.eval()
        
    def to(self, device):
        self.model = self.model.to(device)
        self.device = device
        return self
    
    def half(self):
        self.model.half()
        return self

    def parameters(self):
        return self.model.parameters()
