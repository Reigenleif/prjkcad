from .evaluation_functions import token_precision_from_cmd_list, token_recall_from_cmd_list, token_f1_from_cmd_list, token_accuracy_from_cmd_list, tokens_accuracy_from_cmd_list


def eval_cmd_only(pred_cmds, gt_cmds):
    observe_tokens = ["LINE", "CIRCLE", "ARC"]
    
    metrics = {}
    for token in observe_tokens:
        precision = token_precision_from_cmd_list(pred_cmds, gt_cmds, token)
        recall = token_recall_from_cmd_list(pred_cmds, gt_cmds, token)
        f1 = token_f1_from_cmd_list(pred_cmds, gt_cmds, token)
        metrics[f"{token}_precision"] = precision
        metrics[f"{token}_recall"] = recall
        metrics[f"{token}_f1"] = f1

    extrusions = ["EXTRUDE_NEW", "EXTRUDE_JOIN", "EXTRUDE_CUT", "EXTRUDE_INTERSECT"]
    metrics["EXTRUDE_accuracy"] = tokens_accuracy_from_cmd_list(pred_cmds, gt_cmds, extrusions)
    
        
    return metrics