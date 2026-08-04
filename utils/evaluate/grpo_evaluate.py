import numpy as np
from utils.dual_seq import DualSeq
from utils.render import render_dual_seq_to_shape
from utils.evaluate.shape_evaluation_functions import chamfer_distance_from_shapes

def compute_cd(pred_cmds, pred_args, gt_cmds, gt_args, default_invalid_cd: float = 1.0) -> float:
    """
    Compute Chamfer Distance between predicted and ground-truth DualSeq shapes.
    Returns default_invalid_cd (1.0) when rendering fails or error occurs, NOT 0.0.
    """
    if not pred_cmds or not gt_cmds:
        return default_invalid_cd
    try:
        pred_cmds_list = pred_cmds.cmds if hasattr(pred_cmds, "cmds") else pred_cmds
        pred_args_list = pred_cmds.args_dict if hasattr(pred_cmds, "args_dict") else pred_args
        gt_cmds_list = gt_cmds.cmds if hasattr(gt_cmds, "cmds") else gt_cmds
        gt_args_list = gt_cmds.args_dict if hasattr(gt_cmds, "args_dict") else gt_args

        pred_shape = render_dual_seq_to_shape(pred_cmds_list, pred_args_list)
        gt_shape = render_dual_seq_to_shape(gt_cmds_list, gt_args_list)

        if pred_shape is None or gt_shape is None:
            return default_invalid_cd
        cd = chamfer_distance_from_shapes(pred_shape, gt_shape)
        if cd is None or np.isnan(cd):
            return default_invalid_cd
        return float(cd)
    except Exception:
        return default_invalid_cd

def compute_reward(cd: float, min_cd: float = 1e-5, max_cd: float = 0.5, valid: bool = True) -> float:
    """
    Compute GRPO reward based on Chamfer Distance.
    Returns 1.0 ONLY if valid and cd < min_cd.
    If cd >= max_cd or invalid, returns 0.0.
    """
    if not valid or cd >= max_cd:
        return 0.0
    if cd < min_cd:
        return 1.0
    return round(0.01 + (1.0 - max_cd) * (max_cd - cd) / (max_cd - min_cd), 4)
