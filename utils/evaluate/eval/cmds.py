from __future__ import annotations

from typing import List, Dict, Any, Sequence


def token_precision_from_cmd_list(pred_cmds: Sequence[str], gt_cmds: Sequence[str], token: str) -> float:
    pred_count = sum(1 for c in pred_cmds if c == token)
    gt_count = sum(1 for c in gt_cmds if c == token)
    if pred_count == 0:
        return 0.0
    common_count = min(pred_count, gt_count)
    return common_count / pred_count


def token_recall_from_cmd_list(pred_cmds: Sequence[str], gt_cmds: Sequence[str], token: str) -> float:
    pred_count = sum(1 for c in pred_cmds if c == token)
    gt_count = sum(1 for c in gt_cmds if c == token)
    if gt_count == 0:
        return 0.0
    common_count = min(pred_count, gt_count)
    return common_count / gt_count


def token_f1_from_cmd_list(pred_cmds: Sequence[str], gt_cmds: Sequence[str], token: str) -> float:
    precision = token_precision_from_cmd_list(pred_cmds, gt_cmds, token)
    recall = token_recall_from_cmd_list(pred_cmds, gt_cmds, token)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def token_accuracy_from_cmd_list(pred_cmds: Sequence[str], gt_cmds: Sequence[str], token: str) -> float:
    return token_recall_from_cmd_list(pred_cmds, gt_cmds, token)


def tokens_accuracy_from_cmd_list(pred_cmds: Sequence[str], gt_cmds: Sequence[str], observe_tokens: List[str]) -> float:
    accuracies = [tokens_accuracy_single(pred_cmds, gt_cmds, tok) for tok in observe_tokens]
    if not accuracies:
        return 0.0
    return sum(accuracies) / len(accuracies)


def tokens_accuracy_single(pred_cmds: Sequence[str], gt_cmds: Sequence[str], token: str) -> float:
    return token_recall_from_cmd_list(pred_cmds, gt_cmds, token)


def eval_cmd_only(pred_cmds: Sequence[str], gt_cmds: Sequence[str]) -> Dict[str, float]:
    observe_tokens = ["LINE", "CIRCLE", "ARC"]

    pad_token = "PAD"
    pred_cmds_list = list(pred_cmds)
    gt_cmds_list = list(gt_cmds)
    while pred_cmds_list and pred_cmds_list[-1] == pad_token:
        pred_cmds_list.pop()
    while gt_cmds_list and gt_cmds_list[-1] == pad_token:
        gt_cmds_list.pop()

    metrics = {}
    for token in observe_tokens:
        precision = token_precision_from_cmd_list(pred_cmds_list, gt_cmds_list, token)
        recall = token_recall_from_cmd_list(pred_cmds_list, gt_cmds_list, token)
        f1 = token_f1_from_cmd_list(pred_cmds_list, gt_cmds_list, token)
        metrics[f"eval/{token}_precision"] = precision
        metrics[f"eval/{token}_recall"] = recall
        metrics[f"eval/{token}_f1"] = f1

    metrics["eval/avg_f1"] = sum(metrics[f"eval/{token}_f1"] for token in observe_tokens) / len(observe_tokens)

    extrusions = ["EXTRUDE_NEW", "EXTRUDE_JOIN", "EXTRUDE_CUT", "EXTRUDE_INTERSECT"]
    metrics["eval/EXTRUDE_accuracy"] = tokens_accuracy_from_cmd_list(pred_cmds_list, gt_cmds_list, extrusions)

    return metrics
