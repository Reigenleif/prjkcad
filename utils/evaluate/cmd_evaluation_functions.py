# LINE
def token_precision_from_cmd_list(pred_cmds, gt_cmds, token_str):
    """
    Predicts the precision of a list of predicted commands against a list of ground truth commands.
    """
    # Guard case if there are no ground truth commands
    T_gt = len(gt_cmds)
    if T_gt == 0:
        return 1.0 if len(pred_cmds) == 0 else 0.0
    
    # Count the number of correct predictions for every LINE in preds
    correct = 0
    cnt = 0
    for i in range(len(pred_cmds)):
        if pred_cmds[i] != token_str :
            continue
            
        cnt += 1
        if gt_cmds[i] == token_str:
            correct += 1
            
    precision = correct / cnt if cnt > 0 else 0.0
    return precision

def token_recall_from_cmd_list(pred_cmds, gt_cmds, token_str):
    """
    Predicts the recall of a list of predicted commands against a list of ground truth commands.
    """
    # Guard case if there are no ground truth commands
    T_gt = len(gt_cmds)
    if T_gt == 0:
        return 1.0 if len(pred_cmds) == 0 else 0.0
    
    # Count the number of correct predictions for every LINE in gt
    correct = 0
    cnt = 0
    for i in range(len(gt_cmds)):
        if gt_cmds[i] != token_str :
            continue
            
        cnt += 1
        if pred_cmds[i] == token_str:
            correct += 1
            
    recall = correct / cnt if cnt > 0 else 0.0
    return recall

def token_f1_from_cmd_list(pred_cmds, gt_cmds, token_str):
    """
    Predicts the F1 score of a list of predicted commands against a list of ground truth commands.
    """
    precision = token_precision_from_cmd_list(pred_cmds, gt_cmds, token_str)
    recall = token_recall_from_cmd_list(pred_cmds, gt_cmds, token_str)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def token_accuracy_from_cmd_list(pred_cmds, gt_cmds, token_str):
    """
    Predicts the accuracy of a list of predicted commands against a list of ground truth commands.
    """
    correct = 0
    total = 0
    for pred_cmd, gt_cmd in zip(pred_cmds, gt_cmds):
        if gt_cmd == token_str:
            total += 1
            if pred_cmd == token_str:
                correct += 1
    accuracy = correct / total if total > 0 else 0.0
    return accuracy

def tokens_precision_from_cmd_list(pred_cmds, gt_cmds, token_strs):
    """
    Predicts the precision of a list of predicted commands against a list of ground truth commands for multiple token strings.
    """
    correct = 0
    cnt = 0
    for pred_cmd, gt_cmd in zip(pred_cmds, gt_cmds):
        if pred_cmd in token_strs:
            cnt += 1
            if pred_cmd == gt_cmd:
                correct += 1
    precision = correct / cnt if cnt > 0 else 0.0
    return precision

def tokens_recall_from_cmd_list(pred_cmds, gt_cmds, token_strs):
    """
    Predicts the recall of a list of predicted commands against a list of ground truth commands for multiple token strings.
    """
    correct = 0
    cnt = 0
    for pred_cmd, gt_cmd in zip(pred_cmds, gt_cmds):
        if gt_cmd in token_strs:
            cnt += 1
            if pred_cmd == gt_cmd:
                correct += 1
    recall = correct / cnt if cnt > 0 else 0.0
    return recall

def tokens_f1_from_cmd_list(pred_cmds, gt_cmds, token_strs):
    """
    Predicts the F1 score of a list of predicted commands against a list of ground truth commands for multiple token strings.
    """
    precision = tokens_precision_from_cmd_list(pred_cmds, gt_cmds, token_strs)
    recall = tokens_recall_from_cmd_list(pred_cmds, gt_cmds, token_strs)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def tokens_accuracy_from_cmd_list(pred_cmds, gt_cmds, token_strs):
    """
    Predicts the accuracy of a list of predicted commands against a list of ground truth commands for multiple token strings.
    """
    correct = 0
    total = 0
    for pred_cmd, gt_cmd in zip(pred_cmds, gt_cmds):
        if gt_cmd in token_strs:
            total += 1
            if pred_cmd == gt_cmd:
                correct += 1
    accuracy = correct / total if total > 0 else 0.0
    return accuracy