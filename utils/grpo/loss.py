import torch
import torch.nn.functional as F
from utils.wrapper.eight_bit_binarized_args_wrapper import EightBitBinarizedArgsWrapper

def grpo_loss(
    wrapper: EightBitBinarizedArgsWrapper,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    cmd_seqs: list[list[int]],
    arg_seqs: list[list[list[int]]],
    ref_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float,
    max_T: int,
):
    """
    PPO-clip surrogate loss over the sampled sequences.
    Uses teacher-forcing with the rollout tokens as targets.
    """
    device = input_ids.device
    N = len(cmd_seqs)

    model = wrapper.model

    # Pad rollout sequences to max_T
    cmd_tensor = torch.full((N, max_T), model.pad_id, dtype=torch.long, device=device)
    arg_tensor = torch.full((N, max_T, 31), model.arg_pad_id, dtype=torch.long, device=device)
    for i, (c, a) in enumerate(zip(cmd_seqs, arg_seqs)):
        L = min(len(c), max_T)
        cmd_tensor[i, :L] = torch.tensor(c[:L], dtype=torch.long, device=device)
        L2 = min(len(a), max_T)
        arg_tensor[i, :L2, :] = torch.tensor(a[:L2], dtype=torch.long, device=device)

    # Tile input_ids to N (B * n_rollouts)
    B = input_ids.size(0)
    n_rollouts = N // B
    input_ids_r      = input_ids.repeat_interleave(n_rollouts, dim=0)
    attention_mask_r = attention_mask.repeat_interleave(n_rollouts, dim=0)

    # Teacher-forced forward pass with rollout tokens
    sos_cmd = torch.full((N, 1), model.sos_id,     device=device, dtype=torch.long)
    sos_arg = torch.full((N, 1, 31), model.arg_sos_id, device=device, dtype=torch.long)
    dec_cmd = torch.cat([sos_cmd, cmd_tensor[:, :-1]], dim=1)
    dec_arg = torch.cat([sos_arg, arg_tensor[:, :-1, :]], dim=1)

    cmd_logits, arg_logits, _ = model(
        input_ids=input_ids_r,
        attention_mask=attention_mask_r,
        decoder_input_ids=dec_cmd,
        decoder_input_args=dec_arg,
    )

    # Log-probs of the rollout tokens under the current policy
    lp_cmd = F.log_softmax(cmd_logits, dim=-1)
    lp_arg = F.log_softmax(arg_logits, dim=-1)

    # Gather log-prob for each actual token; ignore pad positions
    mask = (cmd_tensor != model.pad_id).float()
    gathered_cmd = lp_cmd.gather(2, cmd_tensor.unsqueeze(2)).squeeze(2)
    gathered_arg = lp_arg.gather(3, arg_tensor.unsqueeze(3)).squeeze(3) # (N, max_T, 31)
    gathered_arg_sum = gathered_arg.sum(dim=-1) # (N, max_T)

    cur_log_probs = ((gathered_cmd + gathered_arg_sum) * mask).sum(dim=1)  # (N,)

    # PPO-clip ratio with safe numerical clamping
    log_diff = torch.clamp(cur_log_probs - ref_log_probs.detach(), min=-10.0, max=10.0)
    ratio = torch.exp(log_diff)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

    adv = advantages.to(device)
    loss_surr = -torch.min(ratio * adv, clipped_ratio * adv).mean()

    if torch.isnan(loss_surr) or torch.isinf(loss_surr):
        return torch.tensor(0.0, device=device, requires_grad=True)

    return loss_surr
