from __future__ import annotations

from typing import Any, Dict, Tuple, Union
import torch
import torch.nn.functional as F
from utils.wrapper.eight_bit_binarized_args_wrapper import EightBitBinarizedArgsWrapper

class GRPOWrapper(EightBitBinarizedArgsWrapper):
    """Wrapper for GRPO policy sampling and rollout generation."""

    def generate_rollout(
        self,
        batch: Union[Dict[str, Any], Tuple, torch.Tensor],
        attention_mask: torch.Tensor = None,
        n_rollouts: int = 8,
        temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        # <-- Parse Input Dict or Tensors -->
        if isinstance(batch, dict):
            input_ids = batch.get("input_ids", batch.get("x"))
            attention_mask = batch.get("attention_mask", batch.get("attn_mask"))
        elif isinstance(batch, torch.Tensor):
            input_ids = batch
        else:
            input_ids, attention_mask = batch[0], batch[1]

        # <-- Duplicate Prompts for Rollouts -->
        B = input_ids.size(0)
        input_ids = input_ids.repeat_interleave(n_rollouts, dim=0)
        attention_mask = attention_mask.repeat_interleave(n_rollouts, dim=0)
        device = input_ids.device
        N = B * n_rollouts

        # <-- Encoder Pass & Rollout Loop -->
        _, _, enc_out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return self._rollout_loop(input_ids, attention_mask, enc_out, N, device, temperature)

    def _rollout_loop(self, input_ids, attention_mask, enc_out, N, device, temperature) -> Dict[str, torch.Tensor]:
        # <-- Setup Iteration Variables -->
        preds = torch.full((N, 1), self.model.sos_id, device=device, dtype=torch.long)
        pred_args = torch.full((N, 1, 31), self.model.arg_sos_id, device=device, dtype=torch.long)

        cmd_tokens_list, arg_preds_list, step_log_probs = [], [], []
        finished = torch.zeros(N, dtype=torch.bool, device=device)

        # <-- Step Sampling Loop -->
        for _ in range(self.max_new_cmds):
            cmd_logits, arg_logits, _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=preds,
                decoder_input_args=pred_args,
                encoder_out_embeddings=enc_out
            )
            next_cmd_logits = cmd_logits[:, -1, :]
            next_arg_logits = arg_logits[:, -1:, :, :]
            next_arg_token = next_arg_logits.argmax(dim=-1)

            # Categorical Sampling or Argmax
            if temperature > 0.0:
                probs = F.softmax(next_cmd_logits / temperature, dim=-1)
                next_cmd_token = torch.distributions.Categorical(probs).sample().unsqueeze(-1)
            else:
                next_cmd_token = next_cmd_logits.argmax(dim=-1).unsqueeze(-1)

            log_probs_all = F.log_softmax(next_cmd_logits, dim=-1)
            next_log_prob = log_probs_all.gather(dim=-1, index=next_cmd_token)

            next_cmd_token_masked = torch.where(finished.unsqueeze(-1), torch.tensor(self.model.pad_id, device=device), next_cmd_token)
            next_log_prob_masked = torch.where(finished.unsqueeze(-1), torch.tensor(0.0, device=device), next_log_prob)

            cmd_tokens_list.append(next_cmd_token_masked)
            arg_preds_list.append(next_arg_token)
            step_log_probs.append(next_log_prob_masked)

            preds = torch.cat([preds, next_cmd_token_masked], dim=1)
            pred_args = torch.cat([pred_args, next_arg_token], dim=1)

            finished = finished | (next_cmd_token_masked.squeeze(-1) == self.model.eos_id)
            if finished.all():
                break

        # <-- Return Dict Output -->
        return {
            "sampled_cmds": torch.cat(cmd_tokens_list, dim=1),
            "sampled_args": torch.cat(arg_preds_list, dim=1),
            "log_probs": torch.cat(step_log_probs, dim=1).squeeze(-1),
        }
