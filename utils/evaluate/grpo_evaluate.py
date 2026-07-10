import numpy as np
from utils.dual_seq import DualSeq
from utils.render import render_dual_seq_to_shape
from utils.evaluate.shape_evaluation_functions import chamfer_distance_from_shapes

def compute_cd(pred_cmds, pred_args, gt_cmds, gt_args) -> float:
    try:
        pred_ds = DualSeq.from_sequences(pred_cmds, pred_args)
        gt_ds = DualSeq.from_sequences(gt_cmds, gt_args)
        pred_shape = render_dual_seq_to_shape(pred_ds.cmds, pred_ds.args)
        gt_shape = render_dual_seq_to_shape(gt_ds.cmds, gt_ds.args)
        if pred_shape is None or gt_shape is None:
            return 1.0
        cd = chamfer_distance_from_shapes(pred_shape, gt_shape)
        if cd is None or np.isnan(cd):
            return 1.0
        return float(cd)
    except Exception:
        return 1.0

def compute_reward(cd: float, min_cd: float = 1e-5, max_cd: float = 0.5) -> float:
    if cd < min_cd:
        return 1.0
    return 0.01 + (1.0 - max_cd) * (max_cd - cd) / (max_cd - min_cd)
