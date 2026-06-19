"""render_img.py: Render OCC shape to isometric PNG via tessellation + matplotlib."""
import os, tempfile
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.StlAPI import StlAPI_Writer


def _to_triangles(shape, deflection=0.005):
    """
    Tessellate OCC shape → (N, 3, 3) triangle vertex array.

    Steps:
    1. BRepMesh_IncrementalMesh: discretize shape surface (deflection = accuracy tolerance;
       smaller = smoother but more triangles)
    2. StlAPI_Writer: dump tessellation to a temp binary STL file (avoids TopLoc transforms)
    3. trimesh.load: parse STL → mesh object; return vertices[faces] → (N_tri, 3, xyz)
    """
    BRepMesh_IncrementalMesh(shape, deflection, False, 0.5)     # Step 1: tessellate
    tmp = tempfile.NamedTemporaryFile(suffix=".stl", delete=False); tmp.close()
    writer = StlAPI_Writer()
    writer.SetASCIIMode(False); writer.Write(shape, tmp.name)   # Step 2: export STL
    mesh = trimesh.load(tmp.name, force="mesh"); os.unlink(tmp.name)  # Step 3: load
    if mesh is None or len(mesh.faces) == 0:
        return np.empty((0, 3, 3))
    return mesh.vertices[mesh.faces]


def _face_normals(triangles):
    """Compute unit normals for each triangle via cross product of edges."""
    e1 = triangles[:, 1] - triangles[:, 0]
    e2 = triangles[:, 2] - triangles[:, 0]
    n  = np.cross(e1, e2)
    norms = np.linalg.norm(n, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return n / norms


def render_to_image(shape, filepath, size=(512, 512)):
    """
    Render OCC shape to a solid grey isometric PNG (CAD-preview style) at filepath.

    Steps:
    1. Tessellate shape to (N,3,3) triangle array via STL intermediate
    2. Set up matplotlib 3D axes with orthographic projection (no perspective distortion)
    3. Apply standard isometric angles: elevation=35.264°, azimuth=45°
       (equal foreshortening on all 3 world axes → true isometric appearance)
    4. Per-triangle Lambert shading: dot each face normal with a fixed light direction,
       remap to a grey range [base_grey, highlight_grey], fully opaque — no transparency
    5. Save PNG
    """
    triangles = _to_triangles(shape)
    dpi = 100
    fig = plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi)
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()
    fig.patch.set_facecolor("#ffffff")          # white background, clean CAD look

    if len(triangles) > 0:
        ax.view_init(elev=35.264, azim=45)      # Step 3: isometric camera angles
        ax.set_proj_type("ortho")               # Step 2: orthographic — no near/far shrinkage

        # Step 4: Lambert shading → solid grey faces
        light_dir = np.array([0.6, 0.4, 1.0])
        light_dir /= np.linalg.norm(light_dir)
        normals   = _face_normals(triangles)
        intensity = np.clip(normals @ light_dir, 0.0, 1.0)  # [0, 1] per triangle

        base      = 0.45    # darkest grey for back faces
        highlight = 0.95    # brightest grey for lit faces
        grey      = base + intensity * (highlight - base)   # per-triangle grey value

        face_colors = np.stack([grey, grey, grey, np.ones_like(grey)], axis=1)  # RGBA, alpha=1

        poly = Poly3DCollection(
            triangles,
            facecolors=face_colors,
            edgecolor=(0.3, 0.3, 0.3, 0.15),   # very subtle dark edge lines
            linewidth=0.1,
            shade=False)                       # Step 4
        ax.add_collection3d(poly)               # must be added first — get_proj() needs axes
        poly.set_facecolor(face_colors)         # Step 4: per-face grey set after axes attach
        pts = triangles.reshape(-1, 3)
        ax.auto_scale_xyz(pts[:, 0], pts[:, 1], pts[:, 2])  # equal axis scaling

    plt.tight_layout(pad=0)
    plt.savefig(filepath, dpi=dpi, bbox_inches="tight")      # Step 5: save PNG
    plt.close(fig)
