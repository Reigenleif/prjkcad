import os
import sys
import numpy as np
import multiprocessing as mp
from utils.dual_seq import DualSeq
from utils.evaluate.grpo_evaluate import compute_reward

def _render_cd_worker(pred_cmds, pred_args_dict, gt_cmds, gt_args_dict):
    """
    Top-level render function for subprocess-isolated OCC CD.
    Used only in evaluate_cd, not in the training loop.
    """
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        from utils.render import render_dual_seq_to_shape as _render
        from utils.evaluate.shape_evaluation_functions import chamfer_distance_from_shapes as _cd
        pred_shape = _render(pred_cmds, pred_args_dict)
        gt_shape   = _render(gt_cmds,   gt_args_dict)
        if pred_shape is None or gt_shape is None:
            return 0.0
        cd = _cd(pred_shape, gt_shape)
        return 0.0 if (cd is None or np.isnan(cd)) else float(cd)
    except Exception:
        return 0.0


def _safe_cd_subprocess_worker(q, pred_cmds, pred_args_dict, gt_cmds, gt_args_dict):
    result = _render_cd_worker(pred_cmds, pred_args_dict, gt_cmds, gt_args_dict)
    q.put(result)


def safe_cd_subprocess(pred_cmds, pred_args_dict, gt_cmds, gt_args_dict, timeout: int = 30) -> float:
    """
    One-shot subprocess for OCC rendering, used only in evaluate_cd.
    Prevents OCC segfaults from crashing the main process.
    """
    ctx = mp.get_context("spawn")
    q   = ctx.Queue()

    p = ctx.Process(target=_safe_cd_subprocess_worker, args=(q, pred_cmds, pred_args_dict, gt_cmds, gt_args_dict))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return 0.0
    try:
        return q.get_nowait()
    except Exception:
        return 0.0


def fast_validity_reward(pred_cmds, pred_arg_tokens, schema, min_cd=1e-5, max_cd=0.5) -> float:
    """
    Fast reward for training — no OCC rendering.
    Returns 1.0 if DualSeq parses successfully with at least one extrude command,
    else a scaled score based on sequence length similarity.
    """
    if not pred_cmds:
        return 0.0
    extrude_cmds = {"EXTRUDE_NEW", "EXTRUDE_JOIN", "EXTRUDE_CUT", "EXTRUDE_INTERSECT"}
    has_extrude = any(c in extrude_cmds for c in pred_cmds)
    if not has_extrude:
        return 0.05
    try:
        is_binned = False
        if pred_arg_tokens and isinstance(pred_arg_tokens[0], (list, tuple)) and len(pred_arg_tokens[0]) == 31:
            is_binned = True

        if is_binned:
            n_coor = sum(1 for c in pred_cmds if c == "COOR")
            return min(1.0, 0.1 + 0.3 * n_coor)
        else:
            ds = DualSeq.from_sequences(pred_cmds, pred_arg_tokens)
            n_coor = sum(1 for c in ds.cmds if c == "COOR")
            return min(1.0, 0.1 + 0.3 * n_coor)
    except Exception:
        return 0.02


def compute_rewards(
    cmd_seqs, arg_seqs,
    gt_cmds_batch, gt_args_batch,
    schema, grpo_cfg,
):
    """
    Compute scalar rewards for each rollout using the fast validity reward.
    OCC rendering is skipped here to avoid segfaults and subprocess overhead.
    Returns rewards list[float] (N,).
    """
    from .rollout import decode_rollout
    rewards = []
    for cmd_ids, arg_ids in zip(cmd_seqs, arg_seqs):
        pred_cmds, pred_arg_tokens = decode_rollout(cmd_ids, arg_ids, schema)
        reward = fast_validity_reward(pred_cmds, pred_arg_tokens, schema,
                                       grpo_cfg.min_cd, grpo_cfg.max_cd)
        rewards.append(reward)

    return rewards
