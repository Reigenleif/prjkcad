from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple, Union, TYPE_CHECKING
import torch
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from utils.dual_seq import get_dualseq_schema, DualSeq
if TYPE_CHECKING:
    from models.base_model import BaseModel
from utils.wrapper.base_wrapper import BaseWrapper

class TokenizedOneSequenceArgsWrapper(BaseWrapper):
    """Wrapper for 1-sequence tokenized command and argument predictions."""

    def forward(
        self,
        batch: Union[Dict[str, Any], Tuple],
        is_teacher_forcing: bool = True
    ) -> Dict[str, torch.Tensor]:
        # <-- Input Extraction Guard Clause -->
        input_ids, attention_mask, cmd_targets, arg_targets = self.extract_inputs(batch)
        device = input_ids.device
        B = input_ids.size(0)

        # <-- Encoder Forward Pass -->
        _, _, enc_out = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # <-- Branching Teacher Forcing vs Autoregressive -->
        if is_teacher_forcing:
            return self._teacher_forcing_forward(input_ids, attention_mask, cmd_targets, arg_targets, enc_out, B, device)
        return self._autoregressive_forward(input_ids, attention_mask, enc_out, B, device)

    def _teacher_forcing_forward(self, input_ids, attention_mask, cmd_targets, arg_targets, enc_out, B, device) -> Dict[str, torch.Tensor]:
        # <-- Prepare Decoder Targets -->
        T_cmd = min(cmd_targets.size(1), self.max_new_cmds)
        cmd_sos = torch.full((B, 1), self.model.sos_id, device=device, dtype=cmd_targets.dtype)
        decoder_input_ids = torch.cat([cmd_sos, cmd_targets[:, :T_cmd][:, :-1]], dim=1)

        T_arg = min(arg_targets.size(1), self.max_new_args)
        arg_sos = torch.full((B, 1), self.model.arg_sos_id, device=device, dtype=arg_targets.dtype)
        decoder_input_args = torch.cat([arg_sos, arg_targets[:, :T_arg][:, :-1]], dim=1)

        # <-- Model Forward -->
        cmd_logits, arg_logits, _ = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_input_args=decoder_input_args,
            encoder_out_embeddings=enc_out
        )
        return {
            "cmd_logits": cmd_logits,
            "arg_logits": arg_logits,
            "cmd_preds": cmd_logits.argmax(dim=-1),
            "arg_preds": arg_logits.argmax(dim=-1),
        }

    def _autoregressive_forward(self, input_ids, attention_mask, enc_out, B, device) -> Dict[str, torch.Tensor]:
        # <-- Decoding Initialization -->
        cmd_preds_seq = torch.full((B, 1), self.model.sos_id, device=device, dtype=torch.long)
        arg_preds_seq = torch.full((B, 1), self.model.arg_sos_id, device=device, dtype=torch.long)

        cmd_outs, cmd_pred_outs, arg_outs, arg_pred_outs = [], [], [], []
        cmd_done, arg_done = torch.zeros(B, dtype=torch.bool, device=device), torch.zeros(B, dtype=torch.bool, device=device)

        # <-- Generation Steps -->
        for step in range(max(self.max_new_cmds, self.max_new_args)):
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

        # <-- Return Dict Output -->
        return {
            "cmd_logits": torch.cat(cmd_outs, dim=1) if cmd_outs else torch.empty(0, device=device),
            "cmd_preds": torch.cat(cmd_pred_outs, dim=1) if cmd_pred_outs else torch.empty(0, device=device, dtype=torch.long),
            "arg_logits": torch.cat(arg_outs, dim=1) if arg_outs else torch.empty(0, device=device),
            "arg_preds": torch.cat(arg_pred_outs, dim=1) if arg_pred_outs else torch.empty(0, device=device, dtype=torch.long),
        }

    @torch.no_grad()
    def generate(self, input_text: str, max_new_tokens: int = 50) -> DualSeq:
        # <-- Evaluation Mode & Tokenization -->
        self.model.eval()
        input_ids, attention_mask = self.tokenize_input(input_text)
        out_dict = self.forward({"input_ids": input_ids, "attention_mask": attention_mask}, is_teacher_forcing=False)
        cmd_tokens = out_dict["cmd_preds"][0].cpu().numpy().tolist() if "cmd_preds" in out_dict else []

        id_to_command = {v: k for k, v in self.schema["command_to_id"].items()}
        eos_id = self.schema["cmd_eos_id"]
        cmds, args = [], []
        for cmd_id in cmd_tokens:
            if cmd_id == eos_id:
                break
            cmd_str = id_to_command.get(cmd_id)
            if not cmd_str or cmd_str in ("SOS", "EOS", "PAD"):
                continue
            cmds.append(cmd_str)
            args.append({})
        return DualSeq(cmds=cmds, args=args)

    def infer(self, input_text: str, max_new_tokens: int = 50) -> DualSeq:
        # <-- DualSeq Output Generation -->
        return self.generate(input_text, max_new_tokens=max_new_tokens)

