import numpy as np
from sklearn.metrics import r2_score, mean_absolute_percentage_error

def arg_r2_score(pred_args, gt_args):
    """
    Computes the R2 score of a list of predicted arguments against a list of ground truth arguments.
    Inputs are expected to be list of list of float.
    """
    pred_flat = np.array([v for seq in pred_args for v in seq])
    gt_flat = np.array([v for seq in gt_args for v in seq])

    if len(gt_flat) == 0:
        return 1.0

    return float(r2_score(gt_flat, pred_flat))


def arg_mape(pred_args, gt_args):
    """
    Computes the MAPE (Mean Absolute Percentage Error) of a list of predicted arguments against a list of ground truth arguments.
    Inputs are expected to be list of list of float.
    """
    pred_flat = np.array([v for seq in pred_args for v in seq])
    gt_flat = np.array([v for seq in gt_args for v in seq])

    if len(gt_flat) == 0:
        return 0.0

    return float(mean_absolute_percentage_error(gt_flat, pred_flat))