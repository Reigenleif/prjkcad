"""utils/render/point_sampling.py: Point sampling utilities for OCC shapes."""
import numpy as np
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface


def sample_shape(shape, n_u: int = 20, n_v: int = 20) -> np.ndarray:
    """Sample points from the parametric surfaces of an OCC shape.

    For each face of the shape, a grid of (n_u × n_v) parameter samples
    is evaluated and the resulting 3D points are collected.

    Parameters
    ----------
    shape : TopoDS_Shape
        The OpenCASCADE shape to sample.
    n_u : int, optional
        Number of samples along the U parameter direction (default 20).
    n_v : int, optional
        Number of samples along the V parameter direction (default 20).

    Returns
    -------
    np.ndarray of shape (N, 3)
        Array of sampled 3D points. Returns an empty (0, 3) array if no
        faces are found.
    """
    points: list[list[float]] = []

    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = exp.Current()
        surf = BRepAdaptor_Surface(face)

        u_min, u_max = surf.FirstUParameter(), surf.LastUParameter()
        v_min, v_max = surf.FirstVParameter(), surf.LastVParameter()

        for i in range(n_u):
            for j in range(n_v):
                u = u_min + (u_max - u_min) * i / max(n_u - 1, 1)
                v = v_min + (v_max - v_min) * j / max(n_v - 1, 1)
                pnt = surf.Value(u, v)
                points.append([pnt.X(), pnt.Y(), pnt.Z()])

        exp.Next()

    if not points:
        return np.zeros((0, 3), dtype=np.float64)
    return np.array(points, dtype=np.float64)
