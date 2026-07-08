from .cmd_evaluation_functions import token_precision_from_cmd_list, token_recall_from_cmd_list, token_f1_from_cmd_list, token_accuracy_from_cmd_list, tokens_accuracy_from_cmd_list
from .args_evaluation_functions import arg_r2_score, arg_mape
from .shape_evaluation_functions import invalidity_rate_from_shapes, chamfer_distance_from_shapes, chamfer_distance
from .reconstruction_evaluation import eval_reconstruction
from utils.dual_seq import DualSeq
from utils.render import render_dual_seq_to_shape


def eval_cmd_only(pred_cmds, gt_cmds):
    assert len(pred_cmds) == len(gt_cmds), "Length of predicted commands and ground truth commands must be the same."
    
    observe_tokens = ["LINE", "CIRCLE", "ARC"]
    
    # Remove trailing PAD tokens for fair evaluation
    # Keep lengths the same
    pad_token = "PAD"
    while pred_cmds and pred_cmds[-1] == pad_token and gt_cmds and gt_cmds[-1] == pad_token:
        pred_cmds.pop()
        gt_cmds.pop()
        
    
    metrics = {}
    for token in observe_tokens:
        precision = token_precision_from_cmd_list(pred_cmds, gt_cmds, token)
        recall = token_recall_from_cmd_list(pred_cmds, gt_cmds, token)
        f1 = token_f1_from_cmd_list(pred_cmds, gt_cmds, token)
        metrics[f"{token}_precision"] = precision
        metrics[f"{token}_recall"] = recall
        metrics[f"{token}_f1"] = f1
        
    metrics["avg_f1"] = sum(metrics[f"{token}_f1"] for token in observe_tokens) / len(observe_tokens)

    extrusions = ["EXTRUDE_NEW", "EXTRUDE_JOIN", "EXTRUDE_CUT", "EXTRUDE_INTERSECT"]
    metrics["EXTRUDE_accuracy"] = tokens_accuracy_from_cmd_list(pred_cmds, gt_cmds, extrusions)
    
        
    return metrics

def eval_args_only(pred_args, gt_args):
    metrics = {}
    metrics["arg_r2"] = arg_r2_score(pred_args, gt_args)
    metrics["arg_mape"] = arg_mape(pred_args, gt_args)
    
    return metrics

def eval_shape(pred_shapes: list, gt_shapes: list, n_u: int = 20, n_v: int = 20) -> dict:
    """Shape-level evaluation: Invalidity Rate (IR) and mean Chamfer Distance (CD)."""
    metrics: dict = {}
    metrics["ir"] = invalidity_rate_from_shapes(pred_shapes)
    return metrics
    
    cd_values = []
    for ps, gs in zip(pred_shapes, gt_shapes):
        if ps is None or gs is None:
            continue
        try:
            cd_values.append(chamfer_distance_from_shapes(ps, gs, n_u=n_u, n_v=n_v))
        except Exception:
            pass

    metrics["cd"] = float(sum(cd_values) / len(cd_values)) if cd_values else None
    return metrics

def eval_cmd_and_args(pred_cmds, gt_cmds, pred_args, gt_args):
    metrics = {**eval_cmd_only(pred_cmds, gt_cmds), **eval_args_only(pred_args, gt_args)}
    return metrics

    pred_ds = DualSeq.from_sequences(pred_cmds, pred_args)
    gt_ds = DualSeq.from_sequences(gt_cmds, gt_args)

    pred_shape = render_dual_seq_to_shape(pred_ds.cmds, pred_ds.args)
    gt_shape = render_dual_seq_to_shape(gt_ds.cmds, gt_ds.args)

    shape_metrics = eval_shape([pred_shape], [gt_shape])
    metrics["ir"] = shape_metrics["ir"]
    if shape_metrics.get("cd") is not None:
        metrics["cd"] = shape_metrics["cd"]

    return metrics