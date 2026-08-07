import numpy as np
from sklearn.metrics import r2_score
from .cmd_evaluation_functions import token_precision_from_cmd_list, token_recall_from_cmd_list, token_f1_from_cmd_list, token_accuracy_from_cmd_list, tokens_accuracy_from_cmd_list
from .args_evaluation_functions import arg_r2_score, arg_mape
from .shape_evaluation_functions import invalidity_rate_from_shapes, chamfer_distance_from_shapes, chamfer_distance
from .reconstruction_evaluation import eval_reconstruction
# from .text2cad_evaluator import evaluate_text2cad_style
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


def eval_float_args(pred_cmds, gt_cmds, pred_args, gt_args):
    metrics = eval_cmd_only(pred_cmds, gt_cmds)
    pred_flat = []
    gt_flat = []
    for p_seq, g_seq in zip(pred_args, gt_args):
        if isinstance(p_seq, (list, tuple, np.ndarray)):
            pred_flat.extend(p_seq)
        else:
            pred_flat.append(p_seq)
        if isinstance(g_seq, (list, tuple, np.ndarray)):
            gt_flat.extend(g_seq)
        else:
            gt_flat.append(g_seq)
    if gt_flat:
        p_arr = np.clip(np.array(pred_flat, dtype=np.float64), -1e5, 1e5)
        g_arr = np.clip(np.array(gt_flat, dtype=np.float64), -1e5, 1e5)
        metrics["arg_float_mse"] = float(np.mean((p_arr - g_arr) ** 2))
        try:
            metrics["arg_float_r2"] = float(r2_score(g_arr, p_arr))
        except Exception:
            metrics["arg_float_r2"] = 0.0
    else:
        metrics["arg_float_mse"] = 0.0
        metrics["arg_float_r2"] = 1.0
    return metrics


def eval_batch(pred_cmd_tokens, gt_cmd_tokens, pred_args, gt_args, out_type="FloatArgs", schema=None, metadata=None):
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
        true_cmds = [id_to_cmd.get(tok, 'PAD') if isinstance(tok, (int, np.integer)) else str(tok) for tok in gt_cmd_tokens[i]]
        pred_cmds = [id_to_cmd.get(tok, 'PAD') if isinstance(tok, (int, np.integer)) else str(tok) for tok in pred_cmd_tokens[i]]

        try: true_cmds = true_cmds[:true_cmds.index("EOS")]
        except ValueError: pass
        try: pred_cmds = pred_cmds[:pred_cmds.index("EOS")]
        except ValueError: pass

        if out_type in ["FloatArgs", "float_args"]:
            p_a = pred_args[i] if i < len(pred_args) else []
            g_a = gt_args[i]
            m = eval_float_args(pred_cmds, true_cmds, p_a, g_a)
        elif out_type in ["EightBitBinarizedArgs", "eight_bit"]:
            m = eval_cmd_only(pred_cmds, true_cmds)
            p_a = pred_args[i] if i < len(pred_args) else []
            g_a = gt_args[i]
            correct = 0
            total = 0
            pred_floats = []
            true_floats = []
            arg_names = schema["arg_names"]
            for step_idx in range(min(len(p_a), len(g_a))):
                for arg_idx, arg_name in enumerate(arg_names):
                    p_bin = p_a[step_idx][arg_idx] if arg_idx < len(p_a[step_idx]) else 256
                    t_bin = g_a[step_idx][arg_idx] if arg_idx < len(g_a[step_idx]) else 256
                    if t_bin != 256:
                        total += 1
                        if p_bin == t_bin:
                            correct += 1
                        if metadata is not None:
                            pred_floats.append(metadata.bin_to_float(arg_name, p_bin))
                            true_floats.append(metadata.bin_to_float(arg_name, t_bin))
            m["arg_token_accuracy"] = correct / max(total, 1)
            if true_floats:
                p_arr = np.clip(np.array(pred_floats, dtype=np.float64), -1e5, 1e5)
                t_arr = np.clip(np.array(true_floats, dtype=np.float64), -1e5, 1e5)
                m["arg_float_mse"] = float(np.mean((p_arr - t_arr) ** 2))
                try:
                    m["arg_float_r2"] = float(r2_score(t_arr, p_arr))
                except Exception:
                    m["arg_float_r2"] = 0.0
            else:
                m["arg_float_mse"] = 0.0
                m["arg_float_r2"] = 1.0
        elif out_type in ["TokenizedOneSequenceArgs", "tokenized"]:
            p_a = pred_args[i] if i < len(pred_args) else []
            g_a = gt_args[i]
            m = eval_cmd_and_args(pred_cmds, true_cmds, p_a, g_a, schema, skip_rendering=True)
            correct = sum(1 for j in range(min(len(p_a), len(g_a))) if p_a[j] == g_a[j])
            m["arg_token_accuracy"] = correct / max(len(g_a), 1)
        else:
            m = eval_cmd_only(pred_cmds, true_cmds)

        sample_metrics_list.append(m)

    avg_metrics = {}
    if sample_metrics_list:
        all_keys = set(k for sm in sample_metrics_list for k in sm.keys())
        for key in all_keys:
            vals = [sm[key] for sm in sample_metrics_list if sm.get(key) is not None]
            if vals:
                avg_metrics[key] = float(np.mean(vals))
    return avg_metrics