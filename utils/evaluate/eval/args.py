from __future__ import annotations

from typing import Any, Dict, Sequence
import numpy as np
from sklearn.metrics import r2_score


def arg_mape(pred_args: Sequence[float], gt_args: Sequence[float]) -> float:
    pred_arr = np.array(pred_args, dtype=np.float64)
    gt_arr = np.array(gt_args, dtype=np.float64)
    mask = gt_arr != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((gt_arr[mask] - pred_arr[mask]) / gt_arr[mask]))) * 100.0


def arg_r2_score(pred_args: Sequence[float], gt_args: Sequence[float]) -> float:
    if not gt_args:
        return 1.0
    p_arr = np.clip(np.array(pred_args, dtype=np.float64), -1e5, 1e5)
    g_arr = np.clip(np.array(gt_args, dtype=np.float64), -1e5, 1e5)
    try:
        return float(r2_score(g_arr, p_arr))
    except Exception:
        return 0.0


def extract_flat_floats(item: Any) -> list[float]:
    if isinstance(item, (int, float, np.number)):
        return [float(item)]
    if isinstance(item, dict):
        res = []
        for v in item.values():
            if isinstance(v, (int, float, np.number)):
                res.append(float(v))
        return res
    if isinstance(item, (list, tuple, np.ndarray)):
        res = []
        for elem in item:
            res.extend(extract_flat_floats(elem))
        return res
    return []


def eval_float_args_metrics(pred_args: Sequence[Any], gt_args: Sequence[Any]) -> Dict[str, float]:
    pred_flat = []
    gt_flat = []
    for p_seq, g_seq in zip(pred_args, gt_args):
        pred_flat.extend(extract_flat_floats(p_seq))
        gt_flat.extend(extract_flat_floats(g_seq))

    if gt_flat:
        min_len = min(len(pred_flat), len(gt_flat))
        if min_len == 0:
            return {"eval/arg_float_mse": 0.0, "eval/arg_float_r2": 0.0}
        p_arr = np.clip(np.array(pred_flat[:min_len], dtype=np.float64), -1e5, 1e5)
        g_arr = np.clip(np.array(gt_flat[:min_len], dtype=np.float64), -1e5, 1e5)
        mse = float(np.mean((p_arr - g_arr) ** 2))
        r2 = arg_r2_score(pred_flat[:min_len], gt_flat[:min_len])
    else:
        mse = 0.0
        r2 = 1.0

    return {
        "eval/arg_float_mse": mse,
        "eval/arg_float_r2": r2,
    }


def eval_tokenized_args_metrics(pred_arg_tokens: Sequence[int], gt_arg_tokens: Sequence[int], schema: Any) -> Dict[str, float]:
    correct = sum(1 for p, g in zip(pred_arg_tokens, gt_arg_tokens) if p == g)
    precision = correct / len(pred_arg_tokens) if pred_arg_tokens else 0.0
    recall = correct / len(gt_arg_tokens) if gt_arg_tokens else 0.0
    arg_token_f1 = 0.0 if (precision + recall == 0) else (2 * precision * recall / (precision + recall))

    arg_token_accuracy = correct / max(len(gt_arg_tokens), 1)

    sep_id = schema.get("arg_sep_id", -1) if schema else -1
    pred_seps = list(pred_arg_tokens).count(sep_id)
    gt_seps = list(gt_arg_tokens).count(sep_id)
    arg_sep_count_mse = float((pred_seps - gt_seps) ** 2)

    return {
        "eval/arg_token_accuracy": arg_token_accuracy,
        "eval/arg_token_f1": arg_token_f1,
        "eval/arg_sep_count_mse": arg_sep_count_mse,
    }


def eval_binarized_args_metrics(p_a: Any, g_a: Any, schema: Any, metadata: Any = None) -> Dict[str, float]:
    correct = 0
    total = 0
    pred_floats = []
    true_floats = []
    arg_names = schema.get("arg_names", []) if schema else []

    for step_idx in range(min(len(p_a), len(g_a))):
        p_step = p_a[step_idx]
        g_step = g_a[step_idx]
        for arg_idx, arg_name in enumerate(arg_names):
            if isinstance(p_step, dict):
                p_bin = p_step.get(arg_name, 256)
            elif isinstance(p_step, (list, tuple, np.ndarray)):
                p_bin = p_step[arg_idx] if arg_idx < len(p_step) else 256
            else:
                p_bin = 256

            if isinstance(g_step, dict):
                t_bin = g_step.get(arg_name, 256)
            elif isinstance(g_step, (list, tuple, np.ndarray)):
                t_bin = g_step[arg_idx] if arg_idx < len(g_step) else 256
            else:
                t_bin = 256

            if t_bin != 256:
                total += 1
                if p_bin == t_bin:
                    correct += 1
                if metadata is not None and hasattr(metadata, "bin_to_float"):
                    try:
                        pred_floats.append(metadata.bin_to_float(arg_name, p_bin))
                        true_floats.append(metadata.bin_to_float(arg_name, t_bin))
                    except Exception:
                        pass

    accuracy = correct / max(total, 1)
    if true_floats:
        p_arr = np.clip(np.array(pred_floats, dtype=np.float64), -1e5, 1e5)
        t_arr = np.clip(np.array(true_floats, dtype=np.float64), -1e5, 1e5)
        mse = float(np.mean((p_arr - t_arr) ** 2))
        try:
            r2 = float(r2_score(t_arr, p_arr))
        except Exception:
            r2 = 0.0
    else:
        mse = 0.0
        r2 = 1.0

    return {
        "eval/arg_token_accuracy": accuracy,
        "eval/arg_float_mse": mse,
        "eval/arg_float_r2": r2,
    }
