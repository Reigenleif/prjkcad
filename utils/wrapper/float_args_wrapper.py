from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple, Union, TYPE_CHECKING
import torch
from transformers import AutoTokenizer
if TYPE_CHECKING:
    from models.base_model import BaseModel
from utils.wrapper.base_wrapper import BaseWrapper

class FloatArgsWrapper(BaseWrapper):
    """Wrapper for FloatArgs continuous argument predictions."""

    def forward(
        self,
        batch: Union[Dict[str, Any], Tuple],
        is_teacher_forcing: bool = True
    ) -> Dict[str, torch.Tensor]:
        # <-- Input Extraction Guard Clause -->
        input_ids, attention_mask, cmd_targets, arg_targets = self.extract_inputs(batch)
        device = input_ids.device
        B = input_ids.size(0)

        # <-- Teacher Forcing Execution -->
        if is_teacher_forcing:
            return self._teacher_forcing_step(input_ids, attention_mask, cmd_targets, arg_targets, B, device)

        # <-- Autoregressive Generation Execution -->
        return self._autoregressive_step(input_ids, attention_mask, arg_targets, B, device)

    def _teacher_forcing_step(self, input_ids, attention_mask, cmd_targets, arg_targets, B, device) -> Dict[str, torch.Tensor]:
        # <-- Decoder Inputs Construction -->
        T = min(cmd_targets.size(1), self.max_new_cmds)
        sos = torch.full((B, 1), self.model.sos_id, device=device, dtype=cmd_targets.dtype)
        decoder_input_ids = torch.cat([sos, cmd_targets[:, :T][:, :-1]], dim=1)

        # Dimension Guard for arg_targets
        if arg_targets.ndim == 2:
            zero_args = torch.zeros((B, 1), device=device, dtype=arg_targets.dtype)
            decoder_input_args = torch.cat([zero_args, arg_targets[:, :T][:, :-1]], dim=1)
        else:
            zero_args = torch.zeros((B, 1, arg_targets.size(-1)), device=device, dtype=arg_targets.dtype)
            decoder_input_args = torch.cat([zero_args, arg_targets[:, :T, :][:, :-1, :]], dim=1)

        # <-- Model Forward Pass -->
        cmd_logits, arg_preds, _ = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_input_args=decoder_input_args
        )
        cmd_preds = cmd_logits.argmax(dim=-1)
        return {"cmd_logits": cmd_logits, "arg_preds": arg_preds, "cmd_preds": cmd_preds}

    def _autoregressive_step(self, input_ids, attention_mask, arg_targets, B, device) -> Dict[str, torch.Tensor]:
        # <-- Autoregressive Loop Setup -->
        preds = torch.full((B, 1), self.model.sos_id, device=device, dtype=torch.long)
        n_args = arg_targets.size(-1) if (arg_targets is not None and arg_targets.ndim == 3) else 31
        pred_args = torch.zeros((B, 1, n_args), device=device, dtype=torch.float32)

        cmd_outs, cmd_pred_outs, arg_outs = [], [], []

        # <-- Generation Loop -->
        for _ in range(self.max_new_cmds):
            cmd_logits, arg_preds_step, _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=preds,
                decoder_input_args=pred_args
            )
            next_cmd_logits = cmd_logits[:, -1:, :]
            next_cmd_token = next_cmd_logits.argmax(dim=-1)
            next_args = arg_preds_step[:, -1:, :]

            cmd_outs.append(next_cmd_logits)
            cmd_pred_outs.append(next_cmd_token)
            arg_outs.append(next_args)

            preds = torch.cat([preds, next_cmd_token], dim=1)
            pred_args = torch.cat([pred_args, next_args], dim=1)

            if (next_cmd_token.squeeze(-1) == self.model.eos_id).all():
                break

        # <-- Aggregation & Output Return -->
        return {
            "cmd_logits": torch.cat(cmd_outs, dim=1) if cmd_outs else torch.empty(0, device=device),
            "cmd_preds": torch.cat(cmd_pred_outs, dim=1) if cmd_pred_outs else torch.empty(0, device=device, dtype=torch.long),
            "arg_preds": torch.cat(arg_outs, dim=1) if arg_outs else torch.empty(0, device=device),
        }

    @torch.no_grad()
    def generate(self, input_text: str, max_new_tokens: int = 50) -> List[Tuple[str, Dict[str, float]]]:
        # <-- Evaluation Mode Guard -->
        self.model.eval()
        input_ids, attention_mask = self.tokenize_input(input_text)
        device = input_ids.device

        cmd_preds, arg_preds = self._autoregressive_decode(input_ids, attention_mask, device, max_new_tokens)
        return self._build_sequence_output(cmd_preds, arg_preds)

    def _autoregressive_decode(self, input_ids, attention_mask, device, max_new_tokens) -> Tuple[List[int], List[List[float]]]:
        # <-- Loop Initialization -->
        sos_id = self.dual_seq_schema["sos_id"]
        eos_id = self.dual_seq_schema["eos_id"]
        decoder_input_ids = torch.full((1, 1), sos_id, device=device, dtype=torch.long)
        decoder_input_args = torch.zeros((1, 1, 31), device=device, dtype=torch.float32)

        cmd_preds, arg_preds = [], []

        # <-- Decoding Steps -->
        for _ in range(max_new_tokens):
            cmd_logits, arg_preds_out, _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_input_args=decoder_input_args
            )
            next_cmd_token = cmd_logits[:, -1:, :].argmax(dim=-1)
            next_args = arg_preds_out[:, -1:, :]

            decoder_input_ids = torch.cat([decoder_input_ids, next_cmd_token], dim=1)
            decoder_input_args = torch.cat([decoder_input_args, next_args], dim=1)

            cmd_val = next_cmd_token.item()
            cmd_preds.append(cmd_val)
            arg_preds.append(next_args.squeeze().tolist())

            if cmd_val == eos_id:
                break

        return cmd_preds, arg_preds

    def _build_sequence_output(self, cmd_preds: List[int], arg_preds: List[List[float]]) -> List[Tuple[str, Dict[str, float]]]:
        # <-- Sequence Reconstruction -->
        id_to_command = {v: k for k, v in self.dual_seq_schema["command_to_id"].items()}
        arg_names = self.dual_seq_schema["arg_names"]
        command_to_slice = self.dual_seq_schema["command_to_slice"]
        pad_id = self.dual_seq_schema["pad_id"]
        eos_id = self.dual_seq_schema["eos_id"]

        generated_sequence = []
        for cmd_id, arg_vals in zip(cmd_preds, arg_preds):
            if cmd_id in (pad_id, eos_id):
                break
            command_name = id_to_command.get(cmd_id)
            if not command_name or command_name in ("SOS", "EOS", "PAD"):
                continue

            arg_dict = {}
            if command_name in command_to_slice:
                start, end = command_to_slice[command_name]
                for name, val in zip(arg_names[start:end], arg_vals[start:end]):
                    arg_dict[name] = val

            generated_sequence.append((command_name, arg_dict))

        return generated_sequence
