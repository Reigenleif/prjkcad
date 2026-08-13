"""sketch2d.py: Build OCC TopoDS_Face from 2D LOOP/segment tokens via a gp_Ax3 plane."""
import math
from OCC.Core.gp import gp_Pnt, gp_Ax2, gp_Circ, gp_Pln
from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire,
                                      BRepBuilderAPI_MakeFace)
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.GC import GC_MakeArcOfCircle
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop


def _to_3d(x, y, ax3, s):
    """Map local 2D sketch coord (x,y) -> 3D world point using ax3 plane axes and scale s."""
    o, xd, yd = ax3.Location(), ax3.XDirection(), ax3.YDirection()
    return gp_Pnt(o.X() + s*x*xd.X() + s*y*yd.X(),
                  o.Y() + s*x*xd.Y() + s*y*yd.Y(),
                  o.Z() + s*x*xd.Z() + s*y*yd.Z())

def _signed_area(pts):
    """Shoelace formula: positive area = CCW, negative = CW."""
    n = len(pts)
    return sum((pts[i][0]*pts[(i+1)%n][1] - pts[(i+1)%n][0]*pts[i][1])
               for i in range(n)) / 2.0

def _build_wire(segments, ax3, s):
    """Build TopoDS_Wire from (cmd, args) segments; return (wire, estimated_2d_area)."""
    wb, pts, ca = BRepBuilderAPI_MakeWire(), [], None
    for cmd, a in segments:
        if cmd == "LINE":
            wb.Add(BRepBuilderAPI_MakeEdge(_to_3d(a["line_sx"], a["line_sy"], ax3, s),
                                           _to_3d(a["line_ex"], a["line_ey"], ax3, s)).Edge())
            pts.append((a["line_sx"], a["line_sy"]))
        elif cmd == "ARC":
            arc = GC_MakeArcOfCircle(_to_3d(a["arc_sx"], a["arc_sy"], ax3, s),
                                     _to_3d(a["arc_mx"], a["arc_my"], ax3, s),
                                     _to_3d(a["arc_ex"], a["arc_ey"], ax3, s)).Value()
            wb.Add(BRepBuilderAPI_MakeEdge(arc).Edge())
            pts.extend([(a["arc_sx"], a["arc_sy"]), (a["arc_mx"], a["arc_my"])])
        elif cmd == "CIRCLE":
            circ = gp_Circ(gp_Ax2(_to_3d(a["circle_cx"], a["circle_cy"], ax3, s),
                                   ax3.Direction()), s * a["circle_r"])
            wb.Add(BRepBuilderAPI_MakeEdge(circ).Edge())
            ca = math.pi * (a["circle_r"] * s) ** 2
    return wb.Wire(), (ca if ca is not None else abs(_signed_area(pts)))

def build_face_from_loops(loop_groups, ax3, scale):
    """Build TopoDS_Face from multiple LOOP groups by cutting inner hole faces from outer boundary face."""
    plane = gp_Pln(ax3.Location(), ax3.Direction())
    wires_area = [_build_wire(segs, ax3, scale) for segs in loop_groups]

    faces_with_area = []
    for wire, approx_area in wires_area:
        f_builder = BRepBuilderAPI_MakeFace(plane, wire)
        if f_builder.IsDone():
            f = f_builder.Face()
            props = GProp_GProps()
            brepgprop.SurfaceProperties(f, props)
            area = props.Mass()
            faces_with_area.append((f, area if area > 0 else approx_area))

    if not faces_with_area:
        raise ValueError("Failed to build any valid face from loops.")

    faces_with_area.sort(key=lambda fa: -fa[1])
    outer_face = faces_with_area[0][0]
    inner_faces = [f for f, a in faces_with_area[1:]]

    final_face = outer_face
    for f_in in inner_faces:
        try:
            cut_res = BRepAlgoAPI_Cut(final_face, f_in).Shape()
            if cut_res is not None:
                final_face = cut_res
        except Exception:
            pass

    return final_face
