from .cmd_evaluation_functions import token_precision_from_cmd_list, token_recall_from_cmd_list, token_f1_from_cmd_list, token_accuracy_from_cmd_list, tokens_accuracy_from_cmd_list
from .args_evaluation_functions import arg_r2_score, arg_mape


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

def eval_cmd_and_args(pred_cmds, gt_cmds, pred_args, gt_args):
    return {**eval_cmd_only(pred_cmds, gt_cmds), **eval_args_only(pred_args, gt_args)}