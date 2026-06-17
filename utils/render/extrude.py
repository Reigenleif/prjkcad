"""extrude.py: Extrude a 3D face along a gp_Ax3 normal and apply boolean ops."""
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut, BRepAlgoAPI_Common
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
    if body is None or op == "EXTRUDE_NEW":    return new_solid
    if op == "EXTRUDE_JOIN":                   return BRepAlgoAPI_Fuse(body,    new_solid).Shape()
    if op == "EXTRUDE_CUT":                    return BRepAlgoAPI_Cut(body,     new_solid).Shape()
    if op == "EXTRUDE_INTERSECT":              return BRepAlgoAPI_Common(body,  new_solid).Shape()
    return new_solid
