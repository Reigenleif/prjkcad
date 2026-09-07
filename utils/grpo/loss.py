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
    enc_out=None,
    kl_coef: float = 0.01,
) -> tuple[torch.Tensor, dict]:
    """PPO-clip surrogate loss with KL penalty, using per-token normalized log-probs.

    Per-token normalization prevents sequence-length from inflating the ratio
    (a sequence of T=84 tokens would otherwise produce log_diff ~ 56, making
    ratio = exp(56) even with tiny per-step drift — clamped but still ≈ 22026).

    Returns (loss_scalar, stats_dict).
    """
    model = wrapper.model
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    N = len(cmd_seqs)

    # Pad rollout sequences to max_T
    cmd_tensor = torch.full((N, max_T), model.pad_id, dtype=torch.long, device=device)
    arg_tensor = torch.full((N, max_T, 31), model.arg_pad_id, dtype=torch.long, device=device)
    for i, (c, a) in enumerate(zip(cmd_seqs, arg_seqs)):
        L = min(len(c), max_T)
        cmd_tensor[i, :L] = torch.tensor(c[:L], dtype=torch.long, device=device)
        L2 = min(len(a), max_T)
        arg_tensor[i, :L2, :] = torch.tensor(a[:L2], dtype=torch.long, device=device)

    B = input_ids.size(0)
    n_rollouts = N // B
    input_ids_r = input_ids.repeat_interleave(n_rollouts, dim=0)
    attention_mask_r = attention_mask.repeat_interleave(n_rollouts, dim=0)

    # Teacher-forced forward — reuse enc_out to avoid second encoder pass
    sos_cmd = torch.full((N, 1), model.sos_id, device=device, dtype=torch.long)
    sos_arg = torch.full((N, 1, 31), model.arg_sos_id, device=device, dtype=torch.long)
    dec_cmd = torch.cat([sos_cmd, cmd_tensor[:, :-1]], dim=1)
    dec_arg = torch.cat([sos_arg, arg_tensor[:, :-1, :]], dim=1)

    enc_out_arg = enc_out.to(device) if enc_out is not None else None
    cmd_logits, arg_logits, _ = model(
        input_ids=input_ids_r,
        attention_mask=attention_mask_r,
        decoder_input_ids=dec_cmd,
        decoder_input_args=dec_arg,
        encoder_out_embeddings=enc_out_arg,
    )

    # Per-token log-probs, masked at PAD positions
    mask = (cmd_tensor != model.pad_id).float()
    seq_len = mask.sum(dim=1).clamp(min=1)  # actual token count per rollout

    lp_cmd = F.log_softmax(cmd_logits, dim=-1).gather(2, cmd_tensor.unsqueeze(2)).squeeze(2)
    lp_arg = (
        F.log_softmax(arg_logits, dim=-1)
        .gather(3, arg_tensor.unsqueeze(3))
        .squeeze(3)
        .mean(dim=-1)   # mean over 31 arg dims, not sum — prevents 31x amplification
    )  # (N, max_T)

    # Normalize by sequence length — prevents T-fold amplification of log_diff
    cur_log_probs = ((lp_cmd + lp_arg) * mask).sum(dim=1) / seq_len  # (N,) per-token mean

    ref_lp = ref_log_probs.to(device)
    if ref_lp.ndim > 1:
        ref_lp = ref_lp.sum(dim=-1)

    # Clamp log_diff to ±5 (tighter than before — per-token scale is ~100x smaller)
    log_diff = torch.clamp(cur_log_probs - ref_lp.detach(), min=-5.0, max=5.0)
    ratio = torch.exp(log_diff)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

    adv = advantages.to(device)
    loss_ppo = -torch.min(ratio * adv, clipped_ratio * adv).mean()

    # KL: per-token scale, non-negative
    kl = (cur_log_probs - ref_lp.detach()).mean().clamp(min=0.0)
    loss = loss_ppo + kl_coef * kl

    if torch.isnan(loss) or torch.isinf(loss):
        loss = torch.tensor(0.0, device=device, requires_grad=True)

    clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean()

    stats = {
        "ratio_mean": ratio.mean().item(),
        "ratio_clip_frac": clip_frac.item(),
        "kl": kl.item(),
        "loss_ppo": loss_ppo.item(),
        "seq_len_mean": seq_len.mean().item(),
    }
    return loss, stats
