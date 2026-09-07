import os
import sys
import numpy as np
import multiprocessing as mp
from utils.dual_seq import DualSeq
from utils.evaluate.grpo_evaluate import compute_reward
from utils.grpo.rollout import decode_rollout
from utils.evaluate.t2c.t2c_report import eval_t2c_report
from utils.representations.converter import dualseq_to_cadseq
try:
    from utils.render import render_dual_seq_to_shape as _render
    from utils.evaluate.shape_evaluation_functions import chamfer_distance_from_shapes as _cd
except ImportError:
    _render = None
    _cd = None


def arg_lin_deviation_reward(
    pred_arg_tokens, gt_arg_tokens, deviation_limit: int = 2
) -> float:
    """
    Per-token linear deviation reward for arg sequences.

    For each arg token position that is relevant (both pred and gt present):
      - exact match       -> 1.0
      - |pred - gt| <= deviation_limit -> 1 - deviation / deviation_limit
      - |pred - gt| >  deviation_limit -> 0.0

    Returns the mean across all compared token positions, or 0.0 if none.
    """
    if not pred_arg_tokens or not gt_arg_tokens:
        return 0.0

    scores = []
    for pred_row, gt_row in zip(pred_arg_tokens, gt_arg_tokens):
        if not isinstance(pred_row, (list, tuple)) or not isinstance(gt_row, (list, tuple)):
            continue
        for p, g in zip(pred_row, gt_row):
            try:
                p, g = int(p), int(g)
            except (TypeError, ValueError):
                continue
            dev = abs(p - g)
            if dev == 0:
                scores.append(1.0)
            elif dev <= deviation_limit:
                scores.append(1.0 - dev / deviation_limit)
            else:
                scores.append(0.0)

    return float(np.mean(scores)) if scores else 0.0

def _render_cd_worker(pred_cmds, pred_args_dict, gt_cmds, gt_args_dict):
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        if _render is None or _cd is None:
            return 0.0
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


def decode_dualseq_from_tokens(pred_cmds, pred_arg_tokens, schema, metadata=None, uid="") -> DualSeq:
    if not pred_cmds:
        return DualSeq(cmds=[], args=[], uid=uid)

    try:
        is_binned = False
        if pred_arg_tokens and isinstance(pred_arg_tokens[0], (list, tuple)) and len(pred_arg_tokens[0]) == 31:
            is_binned = True

        if is_binned:
            arg_names = schema.get("arg_names", [])
            command_to_slice = schema.get("command_to_slice", {})
            cmds, args = [], []
            for cmd_str, bin_row in zip(pred_cmds, pred_arg_tokens):
                if not cmd_str or cmd_str in ("SOS", "EOS", "PAD"):
                    continue
                cmds.append(cmd_str)
                arg_dict = {}
                if cmd_str in command_to_slice:
                    start, end = command_to_slice[cmd_str]
                    for name, bin_val in zip(arg_names[start:end], bin_row[start:end]):
                        if metadata is not None:
                            arg_dict[name] = float(metadata.bin_to_float(name, bin_val))
                        else:
                            arg_dict[name] = float(bin_val)
                args.append(arg_dict)
            return DualSeq(cmds=cmds, args=args, uid=uid)
        elif hasattr(DualSeq, "from_sequences"):
            return DualSeq.from_sequences(pred_cmds, pred_arg_tokens, uid=uid)
        else:
            return DualSeq(cmds=pred_cmds, args=pred_arg_tokens if isinstance(pred_arg_tokens, list) else None, uid=uid)
    except Exception:
        return DualSeq(cmds=[], args=[], uid=uid)


def fast_validity_reward(pred_cmds, pred_arg_tokens, schema, metadata=None, min_cd=1e-5, max_cd=0.5) -> float:
    if not pred_cmds:
        return 0.0

    extrude_cmds = {"EXTRUDE_NEW", "EXTRUDE_JOIN", "EXTRUDE_CUT", "EXTRUDE_INTERSECT"}
    has_extrude = any(c in extrude_cmds for c in pred_cmds)
    if not has_extrude:
        return 0.0

    ds = decode_dualseq_from_tokens(pred_cmds, pred_arg_tokens, schema, metadata=metadata)
    if not ds.cmds:
        return 0.0

    n_coor = sum(1 for c in ds.cmds if c == "COOR")
    return min(1.0, 0.1 + 0.3 * n_coor)


def compute_rewards(
    cmd_seqs, arg_seqs,
    schema=None, grpo_cfg=None, metadata=None,
    gt_cmds_batch=None, gt_args_batch=None,
    uids=None,
    deviation_limit: int = 2,
):
    rewards = []
    arg_rewards = []
    dual_seqs = []
    N = len(cmd_seqs)

    for i, (cmd_ids, arg_ids) in enumerate(zip(cmd_seqs, arg_seqs)):
        uid = uids[i] if (uids is not None and i < len(uids)) else ""
        pred_cmds, pred_arg_tokens = decode_rollout(cmd_ids, arg_ids, schema)
        pred_ds = decode_dualseq_from_tokens(pred_cmds, pred_arg_tokens, schema, metadata=metadata, uid=uid)
        dual_seqs.append(pred_ds)

        gt_cad = None
        gt_arg_tokens_raw = None
        if gt_cmds_batch is not None:
            B = len(gt_cmds_batch)
            n_rollouts = max(1, N // B)
            b = min(i // n_rollouts, B - 1)
            gt_item = gt_cmds_batch[b]

            if isinstance(gt_item, DualSeq):
                gt_cad = dualseq_to_cadseq(gt_item)
            else:
                gt_cmds = gt_item
                gt_args = gt_args_batch[b] if (gt_args_batch is not None and b < len(gt_args_batch)) else []
                gt_arg_tokens_raw = gt_args
                if isinstance(gt_cmds, (list, tuple)) and gt_cmds and isinstance(gt_cmds[0], (int, np.integer)):
                    gt_cmds, gt_args = decode_rollout(gt_cmds, gt_args, schema)
                    gt_arg_tokens_raw = gt_args
                gt_ds = decode_dualseq_from_tokens(gt_cmds, gt_args, schema, metadata=metadata, uid=uid)
                gt_cad = dualseq_to_cadseq(gt_ds)

        t2c_reward = 0.0
        if gt_cad is not None:
            pred_cad = dualseq_to_cadseq(pred_ds)
            if pred_cad is not None:
                t2c_m = eval_t2c_report(gt_cad, pred_cad, uid=uid)
                t2c_reward = float(t2c_m.get("t2c/macro_f1", 0.0))

        arg_reward = arg_lin_deviation_reward(
            pred_arg_tokens, gt_arg_tokens_raw, deviation_limit=deviation_limit
        )

        rewards.append(t2c_reward)
        arg_rewards.append(arg_reward)

    return np.array(rewards, dtype=np.float32), np.array(arg_rewards, dtype=np.float32), dual_seqs

