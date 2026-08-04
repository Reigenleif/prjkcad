import numpy as np
from sklearn.metrics import r2_score
from .cmd_evaluation_functions import token_precision_from_cmd_list, token_recall_from_cmd_list, token_f1_from_cmd_list, token_accuracy_from_cmd_list, tokens_accuracy_from_cmd_list
from .args_evaluation_functions import arg_r2_score, arg_mape
from .shape_evaluation_functions import invalidity_rate_from_shapes, chamfer_distance_from_shapes, chamfer_distance
from .reconstruction_evaluation import eval_reconstruction
from .text2cad_evaluator import evaluate_text2cad_style
from utils.dual_seq import DualSeq, get_dualseq_schema
from utils.representations.dual_seq.dual_seq import DEFAULT_COMMANDS, tokens_to_float
from utils.render import render_dual_seq_to_shape


def eval_cmd_only(pred_cmds, gt_cmds):
    observe_tokens = ["LINE", "CIRCLE", "ARC"]
    
    # Remove trailing PAD tokens for fair evaluation
    pad_token = "PAD"
    pred_cmds = list(pred_cmds)
    gt_cmds = list(gt_cmds)
    while pred_cmds and pred_cmds[-1] == pad_token:
        pred_cmds.pop()
    while gt_cmds and gt_cmds[-1] == pad_token:
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


def eval_cmd_and_args(pred_cmds, gt_cmds, pred_arg_tokens, gt_arg_tokens, schema, skip_rendering=True):
    """
    Evaluates both command predictions (list of strings) and argument predictions (list of token IDs).
    """
    # 1. Evaluate commands
    metrics = eval_cmd_only(pred_cmds, gt_cmds)
    
    # 2. Argument token-level matching (F1)
    correct = sum(1 for p, g in zip(pred_arg_tokens, gt_arg_tokens) if p == g)
    precision = correct / len(pred_arg_tokens) if pred_arg_tokens else 0.0
    recall = correct / len(gt_arg_tokens) if gt_arg_tokens else 0.0
    if precision + recall == 0:
        metrics["arg_token_f1"] = 0.0
    else:
        metrics["arg_token_f1"] = 2 * precision * recall / (precision + recall)
        
    # 3. Separator count MSE
    sep_id = schema["arg_sep_id"]
    pred_seps = pred_arg_tokens.count(sep_id)
    gt_seps = gt_arg_tokens.count(sep_id)
    metrics["arg_sep_count_mse"] = float((pred_seps - gt_seps) ** 2)
    
    # 4. Conversion to DualSeq and validation/rendering
    # Clean commands (remove PAD, SOS, EOS)
    def clean_cmds(cmd_list):
        cleaned = []
        for c in cmd_list:
            if c in ("PAD", "SOS", "EOS"):
                break
            cleaned.append(c)
        return cleaned

    pred_cmds_clean = clean_cmds(pred_cmds)
    gt_cmds_clean = clean_cmds(gt_cmds)
    
    is_valid_dualseq = True
    pred_ds = None
    try:
        pred_ds = DualSeq.from_sequences(pred_cmds_clean, pred_arg_tokens)
    except Exception:
        is_valid_dualseq = False
        
    # Ground truth is always valid DualSeq
    try:
        gt_ds = DualSeq.from_sequences(gt_cmds_clean, gt_arg_tokens)
    except Exception:
        gt_ds = None

    is_valid_render = False
    if not skip_rendering and is_valid_dualseq and pred_ds is not None:
        try:
            shape = render_dual_seq_to_shape(pred_ds.cmds, pred_ds.args_dict)
            if shape is not None:
                is_valid_render = True
        except Exception:
            is_valid_render = False
            
    metrics["dualseq_invalidity_ratio"] = 0.0 if is_valid_dualseq else 1.0
    metrics["render_invalid_ratio"] = 0.0 if (not is_valid_dualseq or is_valid_render) else 1.0
    metrics["total_invalidity_ratio"] = 0.0 if is_valid_render else 1.0
    
    # 5. Extract float args and calculate MSE and R2
    if is_valid_dualseq and pred_ds is not None and gt_ds is not None:
        pred_floats = []
        gt_floats = []
        for i, gt_cmd in enumerate(gt_ds.cmds):
            if gt_cmd in DEFAULT_COMMANDS:
                arg_names = DEFAULT_COMMANDS[gt_cmd]
                for name in arg_names:
                    gt_val = gt_ds.args_dict[i].get(name, 0.0)
                    gt_floats.append(gt_val)
                    if i < len(pred_ds.cmds) and pred_ds.cmds[i] == gt_cmd:
                        pred_val = pred_ds.args_dict[i].get(name, 0.0)
                        pred_floats.append(pred_val)
                    else:
                        pred_floats.append(0.0)
                        
        if gt_floats:
            pred_arr = np.clip(np.array(pred_floats, dtype=np.float64), -1e5, 1e5)
            gt_arr = np.clip(np.array(gt_floats, dtype=np.float64), -1e5, 1e5)
            metrics["arg_float_mse"] = float(np.mean((pred_arr - gt_arr) ** 2))
            try:
                metrics["arg_float_r2"] = float(r2_score(gt_arr, pred_arr))
            except Exception:
                metrics["arg_float_r2"] = 0.0
        else:
            metrics["arg_float_mse"] = 0.0
            metrics["arg_float_r2"] = 1.0
    else:
        # Fallback values if DualSeq parsing fails
        metrics["arg_float_mse"] = 1.0
        metrics["arg_float_r2"] = 0.0
        
    return metrics