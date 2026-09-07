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

def _build_wire(segments, ax3, s, tol=1e-4):
    """Build TopoDS_Wire from (cmd, args) segments; return (wire, estimated_2d_area) or None."""
    wb = BRepBuilderAPI_MakeWire()
    pts = []
    ca = None
    edges_added = 0
    first_pt = None
    last_pt = None

    for cmd, a in segments:
        if cmd == "LINE":
            p1 = _to_3d(a.get("line_sx", 0.0), a.get("line_sy", 0.0), ax3, s)
            p2 = _to_3d(a.get("line_ex", 0.0), a.get("line_ey", 0.0), ax3, s)
            if p1.Distance(p2) > tol:
                try:
                    edge = BRepBuilderAPI_MakeEdge(p1, p2).Edge()
                    wb.Add(edge)
                    edges_added += 1
                    pts.append((a["line_sx"], a["line_sy"]))
                    if first_pt is None:
                        first_pt = p1
                    last_pt = p2
                except Exception:
                    pass
        elif cmd == "ARC":
            p1 = _to_3d(a.get("arc_sx", 0.0), a.get("arc_sy", 0.0), ax3, s)
            pm = _to_3d(a.get("arc_mx", 0.0), a.get("arc_my", 0.0), ax3, s)
            p2 = _to_3d(a.get("arc_ex", 0.0), a.get("arc_ey", 0.0), ax3, s)
            added = False
            try:
                arc = GC_MakeArcOfCircle(p1, pm, p2).Value()
                edge = BRepBuilderAPI_MakeEdge(arc).Edge()
                wb.Add(edge)
                edges_added += 1
                pts.extend([(a["arc_sx"], a["arc_sy"]), (a["arc_mx"], a["arc_my"])])
                if first_pt is None:
                    first_pt = p1
                last_pt = p2
                added = True
            except Exception:
                pass
            if not added and p1.Distance(p2) > tol:
                try:
                    edge = BRepBuilderAPI_MakeEdge(p1, p2).Edge()
                    wb.Add(edge)
                    edges_added += 1
                    pts.append((a["arc_sx"], a["arc_sy"]))
                    if first_pt is None:
                        first_pt = p1
                    last_pt = p2
                except Exception:
                    pass
        elif cmd == "CIRCLE":
            r = a.get("circle_r", 0.0) * s
            if r > tol:
                try:
                    p = _to_3d(a.get("circle_cx", 0.0), a.get("circle_cy", 0.0), ax3, s)
                    circ = gp_Circ(gp_Ax2(p, ax3.Direction()), r)
                    edge = BRepBuilderAPI_MakeEdge(circ).Edge()
                    wb.Add(edge)
                    edges_added += 1
                    ca = math.pi * (r ** 2)
                except Exception:
                    pass

    if edges_added == 0:
        return None

    if first_pt is not None and last_pt is not None and first_pt.Distance(last_pt) > tol:
        try:
            close_edge = BRepBuilderAPI_MakeEdge(last_pt, first_pt).Edge()
            wb.Add(close_edge)
        except Exception:
            pass

    if not wb.IsDone():
        return None

    try:
        wire = wb.Wire()
        area = ca if ca is not None else abs(_signed_area(pts))
        return wire, area
    except Exception:
        return None

def build_face_from_loops(loop_groups, ax3, scale):
    """Build TopoDS_Face from multiple LOOP groups by cutting inner hole faces from outer boundary face."""
    plane = gp_Pln(ax3.Location(), ax3.Direction())
    wires_area = []
    for segs in loop_groups:
        res = _build_wire(segs, ax3, scale)
        if res is not None:
            wires_area.append(res)

    if not wires_area:
        raise ValueError("Failed to build any valid face from loops.")

    faces_with_area = []
    for wire, approx_area in wires_area:
        try:
            f_builder = BRepBuilderAPI_MakeFace(plane, wire)
            if f_builder.IsDone():
                f = f_builder.Face()
                props = GProp_GProps()
                brepgprop.SurfaceProperties(f, props)
                area = props.Mass()
                faces_with_area.append((f, area if area > 0 else approx_area))
        except Exception:
            pass

    if not faces_with_area:
        raise ValueError("Failed to build any valid face from loops.")

    faces_with_area.sort(key=lambda fa: -fa[1])
    outer_face = faces_with_area[0][0]
    inner_faces = [f for f, a in faces_with_area[1:]]

    final_face = outer_face
    for f_in in inner_faces:
        try:
            cut_res = BRepAlgoAPI_Cut(final_face, f_in).Shape()
            if cut_res is not None and not cut_res.IsNull():
                final_face = cut_res
        except Exception:
            pass

    return final_face
