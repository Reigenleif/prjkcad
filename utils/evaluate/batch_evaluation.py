from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from utils.dual_seq import DualSeq, get_dualseq_schema
from utils.representations.dual_seq.dual_seq import DEFAULT_COMMANDS
from utils.representations.CadSeqProc.cad_sequence import CADSequence
from utils.representations.converter import dualseq_to_minimal_json, dualseq_to_cadseq, vec_2t_to_cadseq
from utils.render import render_dual_seq_to_shape

from utils.evaluate.eval.cmds import (
    token_accuracy_from_cmd_list,
    token_f1_from_cmd_list,
    token_precision_from_cmd_list,
    token_recall_from_cmd_list,
    tokens_accuracy_from_cmd_list,
    eval_cmd_only,
)
from utils.evaluate.eval.args import (
    arg_mape,
    arg_r2_score,
    eval_float_args_metrics,
    eval_binarized_args_metrics,
    eval_tokenized_args_metrics,
)
from utils.evaluate.render.chamfer_distance import (
    chamfer_distance,
    chamfer_distance_from_shapes,
    invalidity_rate_from_shapes,
    eval_reconstruction,
)
from utils.evaluate.t2c.t2c_report import eval_t2c_report
from utils.grpo.reward import decode_dualseq_from_tokens


def get_canonical_evaluation_dict() -> Dict[str, float]:
    """Returns the standardized, canonical evaluation dictionary initialized with default metric values."""
    return {
        # eval/ command metrics
        "eval/LINE_precision": 0.0,
        "eval/LINE_recall": 0.0,
        "eval/LINE_f1": 0.0,
        "eval/CIRCLE_precision": 0.0,
        "eval/CIRCLE_recall": 0.0,
        "eval/CIRCLE_f1": 0.0,
        "eval/ARC_precision": 0.0,
        "eval/ARC_recall": 0.0,
        "eval/ARC_f1": 0.0,
        "eval/avg_f1": 0.0,
        "eval/EXTRUDE_accuracy": 0.0,
        # eval/ argument metrics
        "eval/arg_float_mse": 0.0,
        "eval/arg_float_r2": 1.0,
        "eval/arg_token_accuracy": 0.0,
        "eval/arg_token_f1": 0.0,
        "eval/arg_sep_count_mse": 0.0,
        # t2c/ report metrics
        "t2c/line_recall": 0.0,
        "t2c/line_precision": 0.0,
        "t2c/line_f1": 0.0,
        "t2c/line_correct_type": 0.0,
        "t2c/line_total_pred": 0.0,
        "t2c/line_total_gt": 0.0,
        "t2c/line_param_s_x": 0.0,
        "t2c/line_param_s_y": 0.0,
        "t2c/line_param_e_x": 0.0,
        "t2c/line_param_e_y": 0.0,
        "t2c/arc_recall": 0.0,
        "t2c/arc_precision": 0.0,
        "t2c/arc_f1": 0.0,
        "t2c/arc_correct_type": 0.0,
        "t2c/arc_total_pred": 0.0,
        "t2c/arc_total_gt": 0.0,
        "t2c/arc_param_s_x": 0.0,
        "t2c/arc_param_s_y": 0.0,
        "t2c/arc_param_m_x": 0.0,
        "t2c/arc_param_m_y": 0.0,
        "t2c/arc_param_e_x": 0.0,
        "t2c/arc_param_e_y": 0.0,
        "t2c/arc_param_x": 0.0,
        "t2c/arc_param_y": 0.0,
        "t2c/circle_recall": 0.0,
        "t2c/circle_precision": 0.0,
        "t2c/circle_f1": 0.0,
        "t2c/circle_total_pred": 0.0,
        "t2c/circle_total_gt": 0.0,
        "t2c/circle_correct_type": 0.0,
        "t2c/circle_param_c_x": 0.0,
        "t2c/circle_param_c_y": 0.0,
        "t2c/circle_param_r": 0.0,
        "t2c/macro_precision": 0.0,
        "t2c/macro_recall": 0.0,
        "t2c/macro_f1": 0.0,
        "t2c/micro_precision": 0.0,
        "t2c/micro_recall": 0.0,
        "t2c/micro_f1": 0.0,
        "t2c/num_skt_gt": 0.0,
        "t2c/num_skt_pred": 0.0,
        "t2c/num_ext_pred": 0.0,
        "t2c/num_ext_gt": 0.0,
        "t2c/num_ext": 0.0,
        "t2c/dist": 0.0,
        "t2c/o_x": 0.0,
        "t2c/o_y": 0.0,
        "t2c/o_z": 0.0,
        "t2c/theta": 0.0,
        "t2c/phi": 0.0,
        "t2c/gamma": 0.0,
        "t2c/scale": 0.0,
        "t2c/b": 0.0,
        "t2c/is_invalid": 0.0,
        "t2c/total_type_acc": 0.0,
        # render/ reconstruction & CD metrics
        "render/dualseq_invalidity_ratio": 0.0,
        "render/render_invalid_ratio": 0.0,
        "render/total_invalidity_ratio": 0.0,
        "render/chamfer_distance": -1.0,
    }





# ── Object Representation Evaluators ───────────────────────────────────────

def eval_cad_sequence(gt_cad_seq: Optional[CADSequence], pred_cad_seq: Optional[CADSequence], uid: str = "0000") -> Dict[str, float]:
    metrics = get_canonical_evaluation_dict()
    t2c_m = eval_t2c_report(gt_cad_seq, pred_cad_seq, uid=uid)
    metrics.update(t2c_m)
    return metrics


def eval_dual_seq(gt_dual_seq: Any, pred_dual_seq: Any, uid: str = "0000", schema: Any = None, metadata: Any = None) -> Dict[str, float]:
    metrics = get_canonical_evaluation_dict()
    gt_cmds = getattr(gt_dual_seq, "cmds", [])
    pred_cmds = getattr(pred_dual_seq, "cmds", [])
    cmd_m = eval_cmd_only(pred_cmds, gt_cmds)
    metrics.update(cmd_m)

    gt_cad = dualseq_to_cadseq(gt_dual_seq)
    pred_cad = dualseq_to_cadseq(pred_dual_seq)
    t2c_m = eval_t2c_report(gt_cad, pred_cad, uid=uid)
    metrics.update(t2c_m)

    return metrics


# ── Raw Output Representation Evaluators ───────────────────────────────────

def eval_float_args(pred_cmds: List[str], gt_cmds: List[str], pred_args: Any, gt_args: Any, schema: Any = None, metadata: Any = None) -> Dict[str, float]:
    metrics = get_canonical_evaluation_dict()
    cmd_m = eval_cmd_only(pred_cmds, gt_cmds)
    metrics.update(cmd_m)

    arg_m = eval_float_args_metrics(pred_args, gt_args)
    metrics.update(arg_m)

    try:
        gt_cad = dualseq_to_cadseq(gt_cmds, gt_args)
        pred_cad = dualseq_to_cadseq(pred_cmds, pred_args)
        t2c_m = eval_t2c_report(gt_cad, pred_cad)
        metrics.update(t2c_m)
    except Exception:
        pass

    return metrics


def eval_eight_bit_binarized_args(pred_cmds: List[str], gt_cmds: List[str], pred_args: Any, gt_args: Any, schema: Any = None, metadata: Any = None) -> Dict[str, float]:
    metrics = get_canonical_evaluation_dict()
    cmd_m = eval_cmd_only(pred_cmds, gt_cmds)
    metrics.update(cmd_m)

    schema = schema or get_dualseq_schema()
    arg_m = eval_binarized_args_metrics(pred_args, gt_args, schema=schema, metadata=metadata)
    metrics.update(arg_m)

    try:
        gt_ds = decode_dualseq_from_tokens(gt_cmds, gt_args, schema, metadata=metadata)
        pred_ds = decode_dualseq_from_tokens(pred_cmds, pred_args, schema, metadata=metadata)
        gt_cad = dualseq_to_cadseq(gt_ds)
        pred_cad = dualseq_to_cadseq(pred_ds)
        if gt_cad is not None and pred_cad is not None:
            t2c_m = eval_t2c_report(gt_cad, pred_cad)
            metrics.update(t2c_m)
    except Exception:
        pass

    return metrics


def eval_tokenized_one_sequence_args(pred_cmds: List[str], gt_cmds: List[str], pred_arg_tokens: Any, gt_arg_tokens: Any, schema: Any = None, metadata: Any = None) -> Dict[str, float]:
    metrics = get_canonical_evaluation_dict()
    cmd_m = eval_cmd_only(pred_cmds, gt_cmds)
    metrics.update(cmd_m)

    schema = schema or get_dualseq_schema()
    arg_m = eval_tokenized_args_metrics(pred_arg_tokens, gt_arg_tokens, schema=schema)
    metrics.update(arg_m)

    return metrics


def eval_cad_seq(pred_cmds: List[str], gt_cmds: List[str], pred_args: Any, gt_args: Any, schema: Any = None, metadata: Any = None) -> Dict[str, float]:
    metrics = get_canonical_evaluation_dict()
    cmd_m = eval_cmd_only(pred_cmds, gt_cmds)
    metrics.update(cmd_m)

    gt_cad = dualseq_to_cadseq(gt_cmds, gt_args)
    pred_cad = dualseq_to_cadseq(pred_cmds, pred_args)
    t2c_m = eval_t2c_report(gt_cad, pred_cad)
    metrics.update(t2c_m)

    return metrics


# ── Unified Master Batch Evaluator ──────────────────────────────────────────

def eval_batch(
    pred_cmd_tokens: Any,
    gt_cmd_tokens: Any,
    pred_args: Any,
    gt_args: Any,
    out_type: str = "FloatArgs",
    schema: Any = None,
    metadata: Any = None
) -> Dict[str, float]:
    schema = schema or get_dualseq_schema()
    id_to_cmd = schema["id_to_command"]

    if hasattr(pred_cmd_tokens, "cpu"):
        pred_cmd_tokens = pred_cmd_tokens.cpu().numpy().tolist()
    if hasattr(gt_cmd_tokens, "cpu"):
        gt_cmd_tokens = gt_cmd_tokens.cpu().numpy().tolist()
    if hasattr(pred_args, "cpu"):
        pred_args = pred_args.cpu().numpy().tolist()
    if hasattr(gt_args, "cpu"):
        gt_args = gt_args.cpu().numpy().tolist()

    B = len(gt_cmd_tokens)
    sample_metrics_list = []

    for i in range(B):
        true_cmds = [id_to_cmd.get(tok, "PAD") if isinstance(tok, (int, np.integer)) else str(tok) for tok in gt_cmd_tokens[i]]
        pred_cmds = [id_to_cmd.get(tok, "PAD") if isinstance(tok, (int, np.integer)) else str(tok) for tok in pred_cmd_tokens[i]]

        try:
            true_cmds = true_cmds[: true_cmds.index("EOS")]
        except ValueError:
            pass
        try:
            pred_cmds = pred_cmds[: pred_cmds.index("EOS")]
        except ValueError:
            pass

        p_a = pred_args[i] if i < len(pred_args) else []
        g_a = gt_args[i]

        if out_type in ["FloatArgs", "float_args"]:
            m = eval_float_args(pred_cmds, true_cmds, p_a, g_a, schema=schema, metadata=metadata)
        elif out_type in ["EightBitBinarizedArgs", "eight_bit", "grpo", "GRPO", "GRPOWrapper"]:
            m = eval_eight_bit_binarized_args(pred_cmds, true_cmds, p_a, g_a, schema=schema, metadata=metadata)
        elif out_type in ["TokenizedOneSequenceArgs", "tokenized"]:
            m = eval_tokenized_one_sequence_args(pred_cmds, true_cmds, p_a, g_a, schema=schema, metadata=metadata)
        elif out_type in ["CADSequence", "cad_seq"]:
            m = eval_cad_seq(pred_cmds, true_cmds, p_a, g_a, schema=schema, metadata=metadata)
        else:
            m = eval_float_args(pred_cmds, true_cmds, p_a, g_a, schema=schema, metadata=metadata)

        sample_metrics_list.append(m)

    canonical_keys = list(get_canonical_evaluation_dict().keys())
    avg_metrics = {}

    for key in canonical_keys:
        vals = [sm[key] for sm in sample_metrics_list if key in sm and sm[key] is not None]
        if vals:
            avg_metrics[key] = float(np.mean(vals))
        else:
            avg_metrics[key] = 0.0

    return avg_metrics
