import re
from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_MakeFace
)
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.TopoDS import TopoDS_Shape


class CADSequenceToOCC:
    def __init__(self, seq: str):
        self.tokens = seq.split()
        self.idx = 0

    def parse(self):
        solids = []

        while self.idx < len(self.tokens):
            token = self.tokens[self.idx]

            if token.startswith(("line", "arc", "circle")):
                face = self._parse_sketch()
                if face:
                    solid = self._parse_extrude(face)
                    if solid:
                        solids.append(solid)
            else:
                self.idx += 1

        return solids

    def _parse_sketch(self):
        edges = []
        points = []
        prev_point = None

        while self.idx < len(self.tokens):
            token = self.tokens[self.idx]

            # ---------------- LINE ----------------
            if token.startswith("line"):
                try:
                    _, x, y = token.split(",")
                    p = gp_Pnt(float(x), float(y), 0)

                    if prev_point:
                        edge = BRepBuilderAPI_MakeEdge(prev_point, p).Edge()
                        edges.append(edge)

                    prev_point = p
                    points.append(p)
                except:
                    pass

            # ---------------- ARC ----------------
            elif token.startswith("arc"):
                try:
                    _, x1, y1, x2, y2 = token.split(",")

                    if prev_point is None:
                        self.idx += 1
                        continue

                    p1 = prev_point
                    p2 = gp_Pnt(float(x1), float(y1), 0)
                    p3 = gp_Pnt(float(x2), float(y2), 0)

                    arc = GC_MakeArcOfCircle(p1, p2, p3)

                    if arc.IsDone():
                        edge = BRepBuilderAPI_MakeEdge(arc.Value()).Edge()
                        edges.append(edge)

                        prev_point = p3
                        points.append(p3)
                except:
                    pass

            # ---------------- CIRCLE ----------------
            elif token.startswith("circle"):
                try:
                    vals = list(map(float, token.split(",")[1:]))

                    if len(vals) != 8:
                        self.idx += 1
                        continue

                    p1 = gp_Pnt(vals[0], vals[1], 0)
                    p2 = gp_Pnt(vals[2], vals[3], 0)
                    p3 = gp_Pnt(vals[4], vals[5], 0)

                    circ = GC_MakeCircle(p1, p2, p3)

                    if circ.IsDone():
                        edge = BRepBuilderAPI_MakeEdge(circ.Value()).Edge()
                        edges.append(edge)

                        # circle is closed → no prev_point update
                except:
                    pass

            # ---------------- CONTROL TOKENS ----------------
            elif token == "<curve_end>":
                pass  # no-op now (edges are created immediately)

            elif token == "<loop_end>":
                self.idx += 1
                break

            self.idx += 1

        # ---------------- VALIDATION ----------------
        if not edges:
            return None

        wire_maker = BRepBuilderAPI_MakeWire()
        for e in edges:
            wire_maker.Add(e)

        if not wire_maker.IsDone():
            return None

        wire = wire_maker.Wire()

        face_maker = BRepBuilderAPI_MakeFace(wire)
        if not face_maker.IsDone():
            return None

        return face_maker.Face()

    def _parse_extrude(self, face):
        while self.idx < len(self.tokens):
            token = self.tokens[self.idx]

            if token.startswith("add"):
                try:
                    parts = list(map(float, token.split(",")[1:]))

                    dx, dy, dz = parts[-3:]
                    vec = gp_Vec(dx, dy, dz)

                    if vec.Magnitude() == 0:
                        return None

                    prism = BRepPrimAPI_MakePrism(face, vec)
                    if not prism.IsDone():
                        return None

                    return prism.Shape()

                except:
                    return None

            self.idx += 1

        return None