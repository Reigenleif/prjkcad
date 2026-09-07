from __future__ import annotations

from typing import Any, Dict, Optional, Sequence
import numpy as np
from scipy.spatial import KDTree
from OCC.Core.TopoDS import TopoDS_Shape
from utils.render.point_sampling import sample_shape


def chamfer_distance(pc1: np.ndarray, pc2: np.ndarray) -> float:
    if pc1 is None or pc2 is None or len(pc1) == 0 or len(pc2) == 0:
        return float("inf")

    pc1_arr = np.asarray(pc1, dtype=np.float64)
    pc2_arr = np.asarray(pc2, dtype=np.float64)

    if pc1_arr.ndim != 2 or pc1_arr.shape[1] != 3 or pc2_arr.ndim != 2 or pc2_arr.shape[1] != 3:
        return float("inf")

    # Filter out NaNs, Infs, and extreme coordinates
    valid1 = np.isfinite(pc1_arr).all(axis=1) & (np.abs(pc1_arr) < 1e5).all(axis=1)
    valid2 = np.isfinite(pc2_arr).all(axis=1) & (np.abs(pc2_arr) < 1e5).all(axis=1)
    pc1_clean = pc1_arr[valid1]
    pc2_clean = pc2_arr[valid2]

    if len(pc1_clean) == 0 or len(pc2_clean) == 0:
        return float("inf")

    tree1 = KDTree(pc1_clean)
    tree2 = KDTree(pc2_clean)

    dist1, _ = tree1.query(pc2_clean)
    dist2, _ = tree2.query(pc1_clean)

    cd = np.mean(dist1**2) + np.mean(dist2**2)
    return float(cd)


def _extract_point_cloud(shape: Any, num_samples: int = 1000, n_u: int = 20, n_v: int = 20) -> Optional[np.ndarray]:
    if shape is None:
        return None

    # TopoDS_Shape
    if isinstance(shape, TopoDS_Shape) or "TopoDS" in str(type(shape)):
        pts = sample_shape(shape, n_u=n_u, n_v=n_v, num_samples=num_samples)
        return pts if len(pts) > 0 else None

    # CADSequence
    if hasattr(shape, "create_cad_model"):
        try:
            occ_shape = shape.create_cad_model()
            if occ_shape is not None and not occ_shape.IsNull():
                pts = sample_shape(occ_shape, n_u=n_u, n_v=n_v, num_samples=num_samples)
                if len(pts) > 0:
                    return pts
        except Exception:
            pass

    # DualSeq
    if hasattr(shape, "cmds") and hasattr(shape, "args_dict"):
        from utils.render import render_dual_seq_to_shape
        try:
            occ_shape = render_dual_seq_to_shape(shape.cmds, shape.args_dict)
            if occ_shape is not None and not occ_shape.IsNull():
                pts = sample_shape(occ_shape, n_u=n_u, n_v=n_v, num_samples=num_samples)
                if len(pts) > 0:
                    return pts
        except Exception:
            pass

    # Trimesh or object with sample_surface
    if hasattr(shape, "sample_surface"):
        try:
            pts, _ = shape.sample_surface(num_samples)
            return np.asarray(pts, dtype=np.float64)
        except Exception:
            pass

    if hasattr(shape, "vertices"):
        pts = np.asarray(shape.vertices, dtype=np.float64)
        return pts if len(pts) > 0 else None

    if isinstance(shape, (list, np.ndarray)):
        pts = np.asarray(shape, dtype=np.float64)
        if pts.ndim == 2 and pts.shape[1] == 3 and len(pts) > 0:
            return pts

    return None


def chamfer_distance_from_shapes(shape1: Any, shape2: Any, num_samples: int = 1000, n_u: int = 20, n_v: int = 20) -> Optional[float]:
    if shape1 is None or shape2 is None:
        return None
    try:
        pc1 = _extract_point_cloud(shape1, num_samples=num_samples, n_u=n_u, n_v=n_v)
        pc2 = _extract_point_cloud(shape2, num_samples=num_samples, n_u=n_u, n_v=n_v)

        if pc1 is None or pc2 is None:
            return None

        cd = chamfer_distance(pc1, pc2)
        if np.isnan(cd) or np.isinf(cd):
            return None
        return float(cd)
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
