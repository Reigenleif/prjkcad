"""utils/render/__init__.py: DualSeq → isometric PNG."""
from .coord_system import make_coord_system
from .sketch2d    import build_face_from_loops
from .extrude     import extrude_part
from .render_img  import render_to_image
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut, BRepAlgoAPI_Common

_OPS = {"EXTRUDE_JOIN": BRepAlgoAPI_Fuse, "EXTRUDE_CUT": BRepAlgoAPI_Cut, "EXTRUDE_INTERSECT": BRepAlgoAPI_Common}

def _bool_op(solid, body, op):
    """EXTRUDE_* boolean: NEW→replace body, JOIN→fuse, CUT→cut, INTERSECT→common."""
    if body is None or op == "EXTRUDE_NEW" or op not in _OPS: return solid
    return _OPS[op](body, solid).Shape()

def _parse_parts(cmds, args):
    """Group COOR…EXTRUDE_* tokens into PART dicts {coor, faces, extrude_cmd, extrude_args}.
    A PART spans from a COOR token up to and including the next EXTRUDE_* token."""
    parts, i = [], 0
    while i < len(cmds):
        if cmds[i] != "COOR": i += 1; continue
        part = {"coor": args[i], "faces": [], "extrude_cmd": None, "extrude_args": None}
        cur_face = cur_loop = None; i += 1
        while i < len(cmds) and not cmds[i].startswith("EXTRUDE_"):
            c = cmds[i]
            if c == "FACE":
                if cur_face is not None: part["faces"].append(cur_face)
                cur_face, cur_loop = [], None
            elif c == "LOOP":
                if cur_loop is not None: cur_face.append(cur_loop)
                cur_loop = []
            elif c in ("LINE", "CIRCLE", "ARC") and cur_loop is not None:
                cur_loop.append((c, args[i]))
            i += 1
        if cur_loop: cur_face.append(cur_loop)
        if cur_face: part["faces"].append(cur_face)
        if i < len(cmds): part["extrude_cmd"], part["extrude_args"] = cmds[i], args[i]; i += 1
        parts.append(part)
    return parts

def render_dual_seq_to_img(dual_seq, img_path: str) -> None:
    """Convert DualSeq → isometric PNG at img_path.
    Per COOR→EXTRUDE_* block: build gp_Ax3 coord system, build 2D sketch faces
    (inner/outer loop detection in sketch2d), extrude each face, union all face
    solids into the PART solid, then apply the EXTRUDE_* boolean onto the running body."""
    body = None
    for part in _parse_parts(dual_seq.cmds, dual_seq.args):
        ea = part["extrude_args"]
        if ea is None: continue
        ax3   = make_coord_system(part["coor"])
        # Extract extrude params — key suffixes differ by operation type (_new/_join/_cut/etc.)
        scale = next((ea[k] for k in ea if k.endswith("_scale")), 1.0)
        dtn   = next((ea[k] for k in ea if k.endswith("_dtn")), 0.0) or 0.0
        don   = next((ea[k] for k in ea if k.endswith("_don")), 0.0) or 0.0
        part_solid = None
        for face_loops in part["faces"]:
            try:
                face3d = build_face_from_loops(face_loops, ax3, scale)
                solid  = extrude_part(face3d, ax3, dtn, don, "EXTRUDE_NEW")
                part_solid = solid if part_solid is None else BRepAlgoAPI_Fuse(part_solid, solid).Shape()
            except Exception as e:
                print(f"[render] face skipped: {e}")
        if part_solid is not None:
            body = _bool_op(part_solid, body, part["extrude_cmd"])
    if body is not None:
        render_to_image(body, img_path)
    else :
        raise ValueError("No body was created from the DualSeq, check the validity of it")