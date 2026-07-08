def eval_reconstruction(pred_tokens, gt_tokens, pad_token=0) -> dict[str, float]:
    """
    Evaluates reconstruction metrics (precision, recall, F1) for token sequences.
    Filters out the specified pad_token from calculations to ensure fair evaluation.
    """
    correct = 0
    total_pred = 0
    total_gt = 0

    for preds, gts in zip(pred_tokens, gt_tokens):
        for p, g in zip(preds, gts):
            if p != pad_token:
                total_pred += 1
            if g != pad_token:
                total_gt += 1
            if g != pad_token and p == g:
                correct += 1

    precision = correct / total_pred if total_pred > 0 else 0.0
    recall = correct / total_gt if total_gt > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
