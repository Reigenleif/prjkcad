"""utils/render/point_sampling.py: Point sampling utilities for OCC shapes."""
import numpy as np
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopLoc import TopLoc_Location


def sample_shape(shape, n_u: int = 20, n_v: int = 20, num_samples: int = None, **kwargs) -> np.ndarray:
    if shape is None:
        return np.zeros((0, 3), dtype=np.float64)

    target_n = num_samples if num_samples is not None else (n_u * n_v)
    points: list[list[float]] = []

    try:
        # Mesh shape using incremental mesher
        BRepMesh_IncrementalMesh(shape, 0.05).Perform()

        topexp = TopologyExplorer(shape)
        for face in topexp.faces():
            if face is None:
                continue
            loc = TopLoc_Location()
            tri = BRep_Tool.Triangulation(face, loc)
            if tri is not None:
                trsf = loc.Transformation()
                nb_nodes = tri.NbNodes()
                for i in range(1, nb_nodes + 1):
                    p = tri.Node(i).Transformed(trsf)
                    x, y, z = p.X(), p.Y(), p.Z()
                    if np.isfinite(x) and np.isfinite(y) and np.isfinite(z) and max(abs(x), abs(y), abs(z)) < 1e5:
                        points.append([x, y, z])
    except Exception:
        pass

    # Fallback to adaptor sampling if mesh yielded no points
    if len(points) == 0:
        try:
            topexp = TopologyExplorer(shape)
            for face in topexp.faces():
                if face is None:
                    continue
                surf = BRepAdaptor_Surface(face)
                u_min, u_max = surf.FirstUParameter(), surf.LastUParameter()
                v_min, v_max = surf.FirstVParameter(), surf.LastVParameter()

                # Clamp unreasonable boundaries
                if not (np.isfinite(u_min) and np.isfinite(u_max) and np.isfinite(v_min) and np.isfinite(v_max)):
                    continue
                if abs(u_min) > 1e4 or abs(u_max) > 1e4 or abs(v_min) > 1e4 or abs(v_max) > 1e4:
                    continue

                for i in range(n_u):
                    for j in range(n_v):
                        u = u_min + (u_max - u_min) * i / max(n_u - 1, 1)
                        v = v_min + (v_max - v_min) * j / max(n_v - 1, 1)
                        pnt = surf.Value(u, v)
                        x, y, z = pnt.X(), pnt.Y(), pnt.Z()
                        if np.isfinite(x) and np.isfinite(y) and np.isfinite(z) and max(abs(x), abs(y), abs(z)) < 1e5:
                            points.append([x, y, z])
        except Exception:
            pass

    if not points:
        return np.zeros((0, 3), dtype=np.float64)

    pts_arr = np.asarray(points, dtype=np.float64)

    # Subsample or oversample to target_n if requested
    if target_n is not None and len(pts_arr) > target_n:
        indices = np.linspace(0, len(pts_arr) - 1, target_n, dtype=int)
        pts_arr = pts_arr[indices]

    return pts_arr

