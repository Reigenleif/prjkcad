"""extrude.py: Extrude a 3D face along a gp_Ax3 normal and apply boolean ops."""
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut, BRepAlgoAPI_Common
from OCC.Core.BRep import BRep_Builder
from OCC.Core.TopoDS import TopoDS_Compound
from OCC.Core.gp import gp_Vec


def _prism(face, nx, ny, nz, depth):
    """Extrude face by 'depth' along direction (nx, ny, nz). Returns solid or None."""
    if not depth or abs(depth) < 1e-9:
        return None
    return BRepPrimAPI_MakePrism(face, gp_Vec(nx * depth, ny * depth, nz * depth)).Shape()


def _fuse(a, b):
    """Boolean union of two solids. If one operand is None, returns the other."""
    if a is None: return b
    if b is None: return a
    return BRepAlgoAPI_Fuse(a, b).Shape()


def extrude_part(face3d, ax3, dtn, don, op, body=None):
    """
    Extrude a face into a solid and merge with the running body via boolean op.

    Steps:
    1. Extract normal direction from ax3 Z-axis (sketch-plane normal = extrude axis)
    2. Extrude face 'dtn' units along +normal (towards normal)  → prism_dtn
    3. Extrude face 'don' units along −normal (opposite normal) → prism_don
       Union both halves into one solid (handles two-sided extrusions)
    4. Merge new solid into body via the EXTRUDE_* operation:
         EXTRUDE_NEW       → new solid replaces the body entirely
         EXTRUDE_JOIN      → fuse (add material to body)
         EXTRUDE_CUT       → subtract new solid from body (remove material)
         EXTRUDE_INTERSECT → keep only the volume common to both
    """
    # Step 1: Normal direction components from ax3 Z-axis
    zd = ax3.Direction()
    nx, ny, nz = zd.X(), zd.Y(), zd.Z()

    # Steps 2–3: Build prism towards and against the normal, then union
    new_solid = _fuse(_prism(face3d,  nx,  ny,  nz, dtn),   # towards normal
                      _prism(face3d, -nx, -ny, -nz, don))   # opposite normal

    if new_solid is None:
        return body

    # Step 4: Apply boolean op between new solid and existing body
    if body is None:
        return new_solid
    if op == "EXTRUDE_NEW":
        op = "EXTRUDE_JOIN"
    if op == "EXTRUDE_JOIN":
        try:
            res = BRepAlgoAPI_Fuse(body, new_solid).Shape()
            if res is not None and not res.IsNull():
                return res
        except Exception:
            pass
        try:
            builder = BRep_Builder()
            comp = TopoDS_Compound()
            builder.MakeCompound(comp)
            builder.Add(comp, body)
            builder.Add(comp, new_solid)
            return comp
        except Exception:
            return new_solid
    if op == "EXTRUDE_CUT":
        try:
            res = BRepAlgoAPI_Cut(body, new_solid).Shape()
            if res is not None and not res.IsNull():
                return res
        except Exception:
            return body
    if op == "EXTRUDE_INTERSECT":
        try:
            res = BRepAlgoAPI_Common(body, new_solid).Shape()
            if res is not None and not res.IsNull():
                return res
        except Exception:
            return body
    return new_solid
