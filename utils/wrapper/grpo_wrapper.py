import torch
import torch.nn.functional as F
from utils.wrapper.dual_seq_wrapper import DualSeqWrapper

class GRPOWrapper(DualSeqWrapper):
    def generate_rollout(self, input_ids, attention_mask, n_rollouts=8, temperature=1.0):
        """
        Generate n_rollouts samples per prompt.
        input_ids: (B, L)
        attention_mask: (B, L)
        Returns:
            sampled_cmds: (B * n_rollouts, T)
            sampled_args: (B * n_rollouts, T, A)
            log_probs: (B * n_rollouts, T)
        """
        B = input_ids.size(0)
        # Repeat input_ids and attention_mask to generate multiple rollouts per prompt
        input_ids = input_ids.repeat_interleave(n_rollouts, dim=0)
        attention_mask = attention_mask.repeat_interleave(n_rollouts, dim=0)
        
        device = input_ids.device
        N = B * n_rollouts
        
        _, _, enc_out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        preds = torch.full((N, 1), self.model.sos_id, device=device, dtype=torch.long)
        n_args = self.dual_seq_schema["n_args"]
        pred_args = torch.zeros((N, 1, n_args), device=device, dtype=torch.float32)
        
        cmd_tokens_list = []
        arg_preds_list = []
        step_log_probs = []
        
        finished = torch.zeros(N, dtype=torch.bool, device=device)
        
        for _ in range(self.max_new_cmds):
            cmd_logits, arg_preds_step, _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=preds,
                decoder_input_args=pred_args,
                encoder_out_embeddings=enc_out
            )
            
            next_cmd_logits = cmd_logits[:, -1, :] # (N, V)
            next_args = arg_preds_step[:, -1:, :] # (N, 1, A)
            
            # Sample next command
            if temperature > 0.0:
                probs = F.softmax(next_cmd_logits / temperature, dim=-1)
                dist = torch.distributions.Categorical(probs)
                next_cmd_token = dist.sample().unsqueeze(-1) # (N, 1)
            else:
                next_cmd_token = next_cmd_logits.argmax(dim=-1).unsqueeze(-1) # (N, 1)
                
            # Gather log probs of the sampled tokens
            log_probs_all = F.log_softmax(next_cmd_logits, dim=-1)
            next_log_prob = log_probs_all.gather(dim=-1, index=next_cmd_token) # (N, 1)
            
            # If a sequence has finished, fill with pad_id and log_prob = 0
            next_cmd_token_masked = torch.where(finished.unsqueeze(-1), torch.tensor(self.model.pad_id, device=device), next_cmd_token)
            next_log_prob_masked = torch.where(finished.unsqueeze(-1), torch.tensor(0.0, device=device), next_log_prob)
            
            cmd_tokens_list.append(next_cmd_token_masked)
            arg_preds_list.append(next_args)
            step_log_probs.append(next_log_prob_masked)
            
            preds = torch.cat([preds, next_cmd_token_masked], dim=1)
            pred_args = torch.cat([pred_args, next_args], dim=1)
            
            # Update finished status
            finished = finished | (next_cmd_token_masked.squeeze(-1) == self.model.eos_id)
            if finished.all():
                break
                
        sampled_cmds = torch.cat(cmd_tokens_list, dim=1) # (N, T)
        sampled_args = torch.cat(arg_preds_list, dim=1) # (N, T, A)
        log_probs = torch.cat(step_log_probs, dim=1).squeeze(-1) # (N, T)
        
        return sampled_cmds, sampled_args, log_probs
