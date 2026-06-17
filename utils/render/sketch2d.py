"""sketch2d.py: Build OCC TopoDS_Face from 2D LOOP/segment tokens via a gp_Ax3 plane."""
import math
from OCC.Core.gp import gp_Pnt, gp_Ax2, gp_Circ
from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire,
                                      BRepBuilderAPI_MakeFace)
from OCC.Core.GC import GC_MakeArcOfCircle


def _to_3d(x, y, ax3, s):
    """Map local 2D sketch coord (x,y) → 3D world point using ax3 plane axes and scale s.
    Formula: world = origin + s*x*x_axis + s*y*y_axis (linear combination of local axes)."""
    o, xd, yd = ax3.Location(), ax3.XDirection(), ax3.YDirection()
    return gp_Pnt(o.X() + s*x*xd.X() + s*y*yd.X(),
                  o.Y() + s*x*xd.Y() + s*y*yd.Y(),
                  o.Z() + s*x*xd.Z() + s*y*yd.Z())

def _signed_area(pts):
    """Shoelace formula: positive area = CCW (outer boundary), negative = CW (hole)."""
    n = len(pts)
    return sum((pts[i][0]*pts[(i+1)%n][1] - pts[(i+1)%n][0]*pts[i][1])
               for i in range(n)) / 2.0

def _build_wire(segments, ax3, s):
    """
    Build TopoDS_Wire from (cmd, args) segments; return (wire, estimated_2d_area).
    Area used to classify outer (largest) vs inner (holes) loops:
      LINE/ARC loops → shoelace area of sample 2D points
      CIRCLE loop    → exact π·r² (shoelace fails for a single arc/circle)
    """
    wb, pts, ca = BRepBuilderAPI_MakeWire(), [], None
    for cmd, a in segments:
        if cmd == "LINE":
            # Straight segment: map two 2D endpoints to 3D, create edge
            wb.Add(BRepBuilderAPI_MakeEdge(_to_3d(a["line_sx"], a["line_sy"], ax3, s),
                                           _to_3d(a["line_ex"], a["line_ey"], ax3, s)).Edge())
            pts.append((a["line_sx"], a["line_sy"]))
        elif cmd == "ARC":
            # Three-point arc: start → mid → end; GC_MakeArcOfCircle fits the unique circle
            arc = GC_MakeArcOfCircle(_to_3d(a["arc_sx"], a["arc_sy"], ax3, s),
                                     _to_3d(a["arc_mx"], a["arc_my"], ax3, s),
                                     _to_3d(a["arc_ex"], a["arc_ey"], ax3, s)).Value()
            wb.Add(BRepBuilderAPI_MakeEdge(arc).Edge())
            pts.extend([(a["arc_sx"], a["arc_sy"]), (a["arc_mx"], a["arc_my"])])
        elif cmd == "CIRCLE":
            # Full circle: place gp_Circ in the sketch plane (normal = ax3.Direction())
            circ = gp_Circ(gp_Ax2(_to_3d(a["circle_cx"], a["circle_cy"], ax3, s),
                                   ax3.Direction()), s * a["circle_r"])
            wb.Add(BRepBuilderAPI_MakeEdge(circ).Edge())
            ca = math.pi * (a["circle_r"] * s) ** 2    # exact enclosed area for a full circle
    return wb.Wire(), (ca if ca is not None else abs(_signed_area(pts)))

def build_face_from_loops(loop_groups, ax3, scale):
    """
    Build a TopoDS_Face from multiple LOOP groups with correct inner/outer handling.

    Steps:
    1. Build each loop into a TopoDS_Wire and compute its 2D area for classification
    2. Sort descending by area: largest = outer contour, rest = inner holes
    3. BRepBuilderAPI_MakeFace from outer wire; punch inner holes with .Add()
    """
    wires_area = [_build_wire(segs, ax3, scale) for segs in loop_groups]
    wires_area.sort(key=lambda wa: -wa[1])                  # Step 2: outer wire first
    fb = BRepBuilderAPI_MakeFace(wires_area[0][0])          # Step 3: face from outer wire
    for wire, _ in wires_area[1:]:
        fb.Add(wire)                                        # Step 3: punch inner holes
    return fb.Face()
