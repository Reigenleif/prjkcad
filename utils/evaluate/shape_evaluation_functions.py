"""utils/evaluate/shape_evaluation_functions.py: Shape-level evaluation metrics.

Metrics:
    - invalidity_rate (IR): fraction of model outputs that fail to render.
    - chamfer_distance (CD): mean bidirectional nearest-neighbour distance
      between two point clouds sampled from two OCC shapes.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

def invalidity_rate_from_shapes(shapes: list) -> float:
    if not shapes:
        return 0.0
    n_invalid = sum(1 for s in shapes if s is None)
    return n_invalid / len(shapes)


def invalidity_rate_from_dual_seqs(dual_seqs: list, render_fn) -> float:
    if not dual_seqs:
        return 0.0

    shapes: list = []
    for ds in dual_seqs:
        try:
            shape = render_fn(ds)
            shapes.append(shape)
        except Exception:
            shapes.append(None)

    return invalidity_rate_from_shapes(shapes)

def chamfer_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """Compute the symmetric Chamfer Distance between two point sets.

    Given two point clouds A and B:
        CD(A, B) = mean_{a∈A}(min_{b∈B} ‖a-b‖) + mean_{b∈B}(min_{a∈A} ‖b-a‖)

    Parameters
    ----------
    points_a : np.ndarray of shape (N, 3)
    points_b : np.ndarray of shape (M, 3)

    Returns
    -------
    float
        Symmetric Chamfer Distance.

    Raises
    ------
    ValueError
        If either point set is empty.
    """
    if len(points_a) == 0 or len(points_b) == 0:
        raise ValueError("Both point sets must be non-empty to compute Chamfer distance.")

    tree_a = cKDTree(points_a)
    tree_b = cKDTree(points_b)

    dist_a_to_b, _ = tree_b.query(points_a)
    dist_b_to_a, _ = tree_a.query(points_b)

    return float(dist_a_to_b.mean() + dist_b_to_a.mean())


def chamfer_distance_from_shapes(
    shape_a,
    shape_b,
    n_u: int = 20,
    n_v: int = 20,
) -> float:
    """Compute the Chamfer Distance between two OCC shapes via point sampling.

    Points are sampled uniformly across each face's parametric domain.

    Parameters
    ----------
    shape_a, shape_b : TopoDS_Shape
        The two shapes to compare.
    n_u : int, optional
        Number of U-direction samples per face (default 20).
    n_v : int, optional
        Number of V-direction samples per face (default 20).

    Returns
    -------
    float
        Chamfer Distance between the sampled point clouds.
    """
    from utils.render.point_sampling import sample_shape

    pts_a = sample_shape(shape_a, n_u=n_u, n_v=n_v)
    pts_b = sample_shape(shape_b, n_u=n_u, n_v=n_v)
    return chamfer_distance(pts_a, pts_b)
