from __future__ import annotations

from typing import Any, Dict, Optional, Sequence
import numpy as np
from scipy.spatial import KDTree


def chamfer_distance(pc1: np.ndarray, pc2: np.ndarray) -> float:
    if pc1 is None or pc2 is None or len(pc1) == 0 or len(pc2) == 0:
        return float("inf")

    tree1 = KDTree(pc1)
    tree2 = KDTree(pc2)

    dist1, _ = tree1.query(pc2)
    dist2, _ = tree2.query(pc1)

    cd = np.mean(dist1**2) + np.mean(dist2**2)
    return float(cd)


def chamfer_distance_from_shapes(shape1: Any, shape2: Any, num_samples: int = 1000) -> Optional[float]:
    if shape1 is None or shape2 is None:
        return None
    try:
        if hasattr(shape1, "sample_surface"):
            pc1, _ = shape1.sample_surface(num_samples)
        elif hasattr(shape1, "vertices"):
            pc1 = np.array(shape1.vertices)
        else:
            pc1 = np.array(shape1)

        if hasattr(shape2, "sample_surface"):
            pc2, _ = shape2.sample_surface(num_samples)
        elif hasattr(shape2, "vertices"):
            pc2 = np.array(shape2.vertices)
        else:
            pc2 = np.array(shape2)

        return chamfer_distance(pc1, pc2)
    except Exception:
        return None


def invalidity_rate_from_shapes(shapes: Sequence[Any]) -> float:
    if not shapes:
        return 1.0
    invalid_count = sum(1 for s in shapes if s is None)
    return float(invalid_count / len(shapes))


def eval_reconstruction(gt_shapes: Sequence[Any], pred_shapes: Sequence[Any], num_samples: int = 1000) -> Dict[str, float]:
    cds = []
    for gt, pred in zip(gt_shapes, pred_shapes):
        cd = chamfer_distance_from_shapes(gt, pred, num_samples=num_samples)
        if cd is not None and not np.isnan(cd) and not np.isinf(cd):
            cds.append(cd)

    mean_cd = float(np.mean(cds)) if cds else -1.0
    invalid_ratio = invalidity_rate_from_shapes(pred_shapes)

    return {
        "render/chamfer_distance": mean_cd,
        "render/render_invalid_ratio": invalid_ratio,
        "render/total_invalidity_ratio": invalid_ratio,
    }
