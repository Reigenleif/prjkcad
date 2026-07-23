import torch
import torch.nn.functional as F
from utils.wrapper.eight_bit_binarized_args_wrapper import EightBitBinarizedArgsWrapper

def decode_rollout(cmd_ids: list[int], arg_ids: list[list[int]], schema: dict):
    """
    Decode raw token id lists → (cmd_strings, arg_token_ints) trimmed at EOS.
    Returns None, None if the lists are empty.
    """
    id_to_cmd = schema["id_to_command"]

    cmd_strings = [id_to_cmd.get(i, "PAD") for i in cmd_ids]
    try:
        eos_idx = cmd_strings.index("EOS")
        cmd_strings = cmd_strings[:eos_idx]
        arg_ids = arg_ids[:eos_idx]
    except ValueError:
        pass

    cmd_strings = [c for c in cmd_strings if c not in ("PAD", "SOS")]
    return cmd_strings, arg_ids


@torch.no_grad()
def generate_rollouts_tokenized(
    wrapper: EightBitBinarizedArgsWrapper,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    n_rollouts: int,
    max_steps: int,
    temperature: float,
    schema: dict,
):
    """
    Generate n_rollouts autoregressive completions per prompt, collecting log-probs.
    uses autoregressive stochastic sampling with temperature parameter
    

    Returns
    --
    cmd_seqs   : list[list[int]]  length B*n_rollouts, each inner list is token ids
    arg_seqs   : list[list[list[int]]]  same shape
    log_probs  : list[torch.Tensor]  each (T,)  — sum of step log-probs
    """
    device = input_ids.device
    B = input_ids.size(0)
    N = B * n_rollouts

    # Tile inputs for parallel rollouts
    input_ids_r      = input_ids.repeat_interleave(n_rollouts, dim=0)
    attention_mask_r = attention_mask.repeat_interleave(n_rollouts, dim=0)

    model = wrapper.model

    # Pre-compute encoder output once
    _, _, enc_out = model(input_ids=input_ids_r, attention_mask=attention_mask_r)

    cmd_seq = torch.full((N, 1), model.sos_id,     device=device, dtype=torch.long)
    arg_seq = torch.full((N, 1, 31), model.arg_sos_id, device=device, dtype=torch.long)

    step_log_probs_cmd = []
    step_log_probs_arg = []
    finished = torch.zeros(N, dtype=torch.bool, device=device)

    for _ in range(max_steps):
        if finished.all():
            break

        cmd_logits, arg_logits, _ = model(
            input_ids=input_ids_r,
            attention_mask=attention_mask_r,
            decoder_input_ids=cmd_seq,
            decoder_input_args=arg_seq,
            encoder_out_embeddings=enc_out,
        )

        next_cmd_logits = cmd_logits[:, -1, :] # (N, cmd_vocab)
        next_arg_logits = arg_logits[:, -1, :, :] # (N, 31, 257)

        # 1. Sample CMD
        if temperature > 0.0:
            cmd_probs = F.softmax(next_cmd_logits / temperature, dim=-1)
            cmd_probs = torch.nan_to_num(cmd_probs, nan=1e-6, posinf=1e-6, neginf=1e-6)
            cmd_probs = cmd_probs / cmd_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            next_cmd = torch.multinomial(cmd_probs, 1) # (N, 1)
        else:
            next_cmd = next_cmd_logits.argmax(dim=-1, keepdim=True) # (N, 1)

        # 2. Sample ARGs (31 dimensions)
        if temperature > 0.0:
            arg_probs = F.softmax(next_arg_logits.reshape(N * 31, 257) / temperature, dim=-1)
            arg_probs = torch.nan_to_num(arg_probs, nan=1e-6, posinf=1e-6, neginf=1e-6)
            arg_probs = arg_probs / arg_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            next_arg = torch.multinomial(arg_probs, 1) # (N * 31, 1)
            next_arg = next_arg.view(N, 1, 31) # (N, 1, 31)
        else:
            next_arg = next_arg_logits.argmax(dim=-1).unsqueeze(1) # (N, 1, 31)

        # Collect log-probs for sampled tokens
        lp_cmd = F.log_softmax(next_cmd_logits, dim=-1).gather(1, next_cmd) # (N, 1)

        lp_arg_dist = F.log_softmax(next_arg_logits, dim=-1) # (N, 31, 257)
        gathered_lp_arg = lp_arg_dist.gather(2, next_arg.transpose(1, 2)).squeeze(2) # (N, 31)
        lp_arg = gathered_lp_arg.sum(dim=-1, keepdim=True) # (N, 1)

        # Mask finished sequences
        next_cmd = next_cmd.masked_fill(finished.unsqueeze(1), model.pad_id)
        next_arg = next_arg.masked_fill(finished.unsqueeze(1).unsqueeze(2), model.arg_pad_id)
        lp_cmd   = lp_cmd.masked_fill(finished.unsqueeze(1), 0.0)
        lp_arg   = lp_arg.masked_fill(finished.unsqueeze(1), 0.0)

        cmd_seq = torch.cat([cmd_seq, next_cmd], dim=1)
        arg_seq = torch.cat([arg_seq, next_arg], dim=1)
        step_log_probs_cmd.append(lp_cmd)
        step_log_probs_arg.append(lp_arg)

        finished = finished | (next_cmd.squeeze(1) == model.eos_id)

    # cmd_seq / arg_seq include the leading SOS token — strip it
    cmd_seq = cmd_seq[:, 1:]
    arg_seq = arg_seq[:, 1:]

    summed_lp = (
        torch.cat(step_log_probs_cmd, dim=1) + torch.cat(step_log_probs_arg, dim=1)
    ).sum(dim=1)  # (N,)

    summed_lp = torch.nan_to_num(summed_lp, nan=-100.0, posinf=0.0, neginf=-1000.0)
    cmd_seqs  = cmd_seq.cpu().tolist()
    arg_seqs  = arg_seq.cpu().tolist()
    log_probs = summed_lp  # (N,) still on device

    return cmd_seqs, arg_seqs, log_probs
