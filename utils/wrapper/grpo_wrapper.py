from __future__ import annotations

import copy
from typing import Any, Dict, Tuple, Union
import torch
import torch.nn.functional as F
from utils.wrapper.eight_bit_binarized_args_wrapper import EightBitBinarizedArgsWrapper


class GRPOWrapper(EightBitBinarizedArgsWrapper):
    """Wrapper for GRPO policy sampling, rollout generation, and reference policy."""

    def __init__(self, model, text_tokenizer, device="cuda", metadata=None):
        super().__init__(model, text_tokenizer, device=device, metadata=metadata)

        # Reference model: deepcopy → move to CUDA first → freeze
        # torch.cuda.synchronize() after .to() guarantees ref is in VRAM
        # before the policy model occupies any more VRAM.
        ref_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ref_model = copy.deepcopy(model).to(ref_device)
        if ref_device.type == "cuda":
            torch.cuda.synchronize()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval()
        self.ref_device = ref_device

    def sync_ref_model(self) -> None:
        """Copy current policy weights into the frozen reference model."""
        self.ref_model.load_state_dict(self.model.state_dict())
        self.ref_model.eval()
        if self.ref_device.type == "cuda":
            torch.cuda.synchronize()

    @torch.no_grad()
    def generate_rollout(
        self,
        batch: Union[Dict[str, Any], Tuple, torch.Tensor],
        attention_mask: torch.Tensor = None,
        n_rollouts: int = 8,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        if isinstance(batch, torch.Tensor):
            input_ids = batch
            if attention_mask is None:
                attention_mask = (input_ids != 0).long()
        else:
            input_ids, attention_mask, _, _ = self.extract_inputs(batch)

        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        B = input_ids.size(0)
        input_ids_r = input_ids.repeat_interleave(n_rollouts, dim=0)
        attention_mask_r = attention_mask.repeat_interleave(n_rollouts, dim=0)
        N = B * n_rollouts

        _, _, enc_out = self.model(input_ids=input_ids_r, attention_mask=attention_mask_r)
        rollout = self._rollout_loop(input_ids_r, attention_mask_r, enc_out, N, device, temperature)
        rollout["enc_out"] = enc_out
        rollout["input_ids_r"] = input_ids_r
        rollout["attention_mask_r"] = attention_mask_r
        return rollout

    def _rollout_loop(self, input_ids, attention_mask, enc_out, N, device, temperature) -> Dict[str, Any]:
        preds = torch.full((N, 1), self.model.sos_id, device=device, dtype=torch.long)
        pred_args = torch.full((N, 1, 31), self.model.arg_sos_id, device=device, dtype=torch.long)

        cmd_tokens_list, arg_tokens_list, step_log_probs = [], [], []
        finished = torch.zeros(N, dtype=torch.bool, device=device)

        for _ in range(self.max_new_cmds):
            cmd_logits, arg_logits, _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=preds,
                decoder_input_args=pred_args,
                encoder_out_embeddings=enc_out,
            )
            next_cmd_logits = cmd_logits[:, -1, :]       # (N, cmd_vocab)
            next_arg_logits = arg_logits[:, -1, :, :]    # (N, 31, 257)

            # Sample CMD
            if temperature > 0.0:
                cmd_probs = F.softmax(next_cmd_logits / temperature, dim=-1)
                cmd_probs = torch.nan_to_num(cmd_probs, nan=1e-6)
                cmd_probs = cmd_probs / cmd_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                next_cmd_token = torch.multinomial(cmd_probs, 1)  # (N, 1)
            else:
                next_cmd_token = next_cmd_logits.argmax(dim=-1, keepdim=True)

            # Sample ARGs (31 dims) — previously was argmax, now properly sampled
            if temperature > 0.0:
                arg_probs = F.softmax(next_arg_logits.reshape(N * 31, 257) / temperature, dim=-1)
                arg_probs = torch.nan_to_num(arg_probs, nan=1e-6)
                arg_probs = arg_probs / arg_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                next_arg_token = torch.multinomial(arg_probs, 1).view(N, 1, 31)  # (N, 1, 31)
            else:
                next_arg_token = next_arg_logits.argmax(dim=-1).unsqueeze(1)     # (N, 1, 31)

            # Joint log-prob: log p(cmd) + Σ log p(arg_i)  — fixes the previous cmd-only bug
            lp_cmd = F.log_softmax(next_cmd_logits, dim=-1).gather(1, next_cmd_token)  # (N, 1)
            lp_arg = (
                F.log_softmax(next_arg_logits, dim=-1)
                .gather(2, next_arg_token.transpose(1, 2))
                .squeeze(2)
                .sum(dim=-1, keepdim=True)
            )  # (N, 1)
            step_lp = lp_cmd + lp_arg  # (N, 1)

            # Mask unused arg slots for the predicted command
            active_slots = self.slot_mask.to(device)[next_cmd_token.squeeze(-1)].unsqueeze(1)  # (N, 1, 31)
            next_arg_token = torch.where(active_slots, next_arg_token, torch.full_like(next_arg_token, self.model.arg_pad_id))

            # Mask finished sequences
            next_cmd_token = next_cmd_token.masked_fill(finished.unsqueeze(1), self.model.pad_id)
            next_arg_token = next_arg_token.masked_fill(finished.unsqueeze(1).unsqueeze(2), self.model.arg_pad_id)
            step_lp = step_lp.masked_fill(finished.unsqueeze(1), 0.0)

            cmd_tokens_list.append(next_cmd_token)
            arg_tokens_list.append(next_arg_token)
            step_log_probs.append(step_lp)

            preds = torch.cat([preds, next_cmd_token], dim=1)
            pred_args = torch.cat([pred_args, next_arg_token], dim=1)
            finished = finished | (next_cmd_token.squeeze(1) == self.model.eos_id)
            if finished.all():
                break

        sampled_cmds = torch.cat(cmd_tokens_list, dim=1)          # (N, T)
        sampled_args = torch.cat(arg_tokens_list, dim=1)          # (N, T, 31)
        log_probs = torch.cat(step_log_probs, dim=1).squeeze(-1)  # (N, T)

        return {
            "sampled_cmds": sampled_cmds,
            "sampled_args": sampled_args,
            "log_probs": log_probs,
        }

    @torch.no_grad()
    def compute_ref_log_probs(
        self,
        input_ids_r: torch.Tensor,
        attention_mask_r: torch.Tensor,
        cmd_tensor: torch.Tensor,
        arg_tensor: torch.Tensor,
        max_T: int,
        enc_out=None,
    ) -> torch.Tensor:
        """Teacher-forced per-token log-prob under the frozen reference model.

        enc_out: cached encoder output from the rollout (encoder is frozen,
        so ref and policy share the same encoder values — passing it avoids
        a redundant encoder forward pass and ensures identical encoder context).

        Returns a (N,) per-token-mean log-prob tensor on ref_device.
        """
        dev = self.ref_device
        input_ids_r = input_ids_r.to(dev)
        attention_mask_r = attention_mask_r.to(dev)
        cmd_tensor = cmd_tensor.to(dev)
        arg_tensor = arg_tensor.to(dev)

        N = cmd_tensor.size(0)

        sos_cmd = torch.full((N, 1), self.ref_model.sos_id, device=dev, dtype=torch.long)
        sos_arg = torch.full((N, 1, 31), self.ref_model.arg_sos_id, device=dev, dtype=torch.long)
        dec_cmd = torch.cat([sos_cmd, cmd_tensor[:, :-1]], dim=1)
        dec_arg = torch.cat([sos_arg, arg_tensor[:, :-1, :]], dim=1)

        enc_out_arg = enc_out.to(dev) if enc_out is not None else None
        cmd_logits, arg_logits, _ = self.ref_model(
            input_ids=input_ids_r,
            attention_mask=attention_mask_r,
            decoder_input_ids=dec_cmd,
            decoder_input_args=dec_arg,
            encoder_out_embeddings=enc_out_arg,
        )

        mask = (cmd_tensor != self.ref_model.pad_id).float()
        seq_len = mask.sum(dim=1).clamp(min=1)

        lp_cmd = F.log_softmax(cmd_logits, dim=-1).gather(2, cmd_tensor.unsqueeze(2)).squeeze(2)
        lp_arg = (
            F.log_softmax(arg_logits, dim=-1)
            .gather(3, arg_tensor.unsqueeze(3))
            .squeeze(3)
            .mean(dim=-1)   # mean over 31 arg dims — matches loss.py
        )  # (N, max_T)

        # Per-token mean — matches grpo_loss normalization
        ref_lp = ((lp_cmd + lp_arg) * mask).sum(dim=1) / seq_len  # (N,)
        return torch.nan_to_num(ref_lp, nan=-100.0, posinf=0.0, neginf=-1000.0)
