import os
import struct
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from OCC.Core.StlAPI import StlAPI_Writer
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepGProp import brepgprop
    OCC_AVAILABLE = True
except (ImportError, Exception):
    OCC_AVAILABLE = False
    StlAPI_Writer = None
    BRepMesh_IncrementalMesh = None
    Bnd_Box = None
    brepbndlib = None
    GProp_GProps = None
    brepgprop = None

try:
    from utils.representations.dual_seq.dual_seq import DualSeq
    from utils.representations.converter import dualseq_to_minimal_json
except (ImportError, Exception):
    DualSeq = None
    dualseq_to_minimal_json = None

try:
    from utils.render import render_dual_seq_to_shape, render_to_image
except (ImportError, Exception):
    render_dual_seq_to_shape = None
    render_to_image = None

try:
    from gui.backend.llm_harness import normalize_command_args, repair_dualseq_sequence
except ImportError:
    from llm_harness import normalize_command_args, repair_dualseq_sequence


DEFAULT_STL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "renders")
os.makedirs(DEFAULT_STL_DIR, exist_ok=True)


def export_shape_to_stl(shape, stl_path: str, deflection: float = 0.05) -> bool:
    if not OCC_AVAILABLE or shape is None or (hasattr(shape, "IsNull") and shape.IsNull()):
        return False

    try:
        mesh = BRepMesh_IncrementalMesh(shape, deflection)
        mesh.Perform()
        writer = StlAPI_Writer()
        writer.SetASCIIMode(False)
        writer.Write(shape, stl_path)
        if os.path.isfile(stl_path) and os.path.getsize(stl_path) > 84:
            return True
        return False
    except Exception as e:
        print(f"STL export error: {e}")
        return False


def get_shape_properties(shape) -> Dict[str, Any]:
    props = {
        "volume": 0.0,
        "bbox": {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]},
    }
    if shape is None or shape.IsNull():
        return props
    try:
        gprops = GProp_GProps()
        brepgprop.VolumeProperties(shape, gprops)
        props["volume"] = float(gprops.Mass())

        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox)
        if not bbox.IsVoid():
            xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
            props["bbox"] = {
                "min": [round(xmin, 4), round(ymin, 4), round(zmin, 4)],
                "max": [round(xmax, 4), round(ymax, 4), round(zmax, 4)],
            }
    except Exception:
        pass
    return props


def parse_solidworks_part_tree(
    cmds: List[str],
    args: List[Dict[str, float]],
) -> Dict[str, Any]:
    tree_nodes: List[Dict[str, Any]] = []
    part_count = 0
    i = 0
    total_len = len(cmds)

    while i < total_len:
        if cmds[i] != "COOR":
            i += 1
            continue

        part_count += 1
        part_id = f"part_{part_count}"
        coor_arg = normalize_command_args("COOR", args[i] if i < len(args) else {})
        coor_node = {
            "id": f"{part_id}_coor",
            "name": f"Coordinate System {part_count}",
            "type": "coor",
            "icon": "compass",
            "cmd": "COOR",
            "cmd_index": i,
            "params": dict(coor_arg),
            "children": [],
        }

        part_children = [coor_node]
        i += 1
        face_idx = 0
        loop_idx = 0

        cur_face_node = None
        cur_loop_node = None
        extrude_node = None

        while i < total_len and not cmds[i].startswith("EXTRUDE_"):
            c = cmds[i]
            raw_a = args[i] if i < len(args) else {}
            a = normalize_command_args(c, raw_a)

            if c == "FACE":
                face_idx += 1
                cur_face_node = {
                    "id": f"{part_id}_face_{face_idx}",
                    "name": f"Sketch {face_idx} (Face)",
                    "type": "face",
                    "icon": "layers",
                    "cmd": "FACE",
                    "cmd_index": i,
                    "params": {},
                    "children": [],
                }
                part_children.append(cur_face_node)
                cur_loop_node = None

            elif c == "LOOP":
                loop_idx += 1
                loop_label = "Outer Profile" if (cur_face_node and len(cur_face_node["children"]) == 0) else f"Internal Hole {loop_idx}"
                cur_loop_node = {
                    "id": f"{part_id}_loop_{loop_idx}",
                    "name": f"Loop {loop_idx} ({loop_label})",
                    "type": "loop",
                    "icon": "circle-dot",
                    "cmd": "LOOP",
                    "cmd_index": i,
                    "params": {},
                    "children": [],
                }
                if cur_face_node is not None:
                    cur_face_node["children"].append(cur_loop_node)
                else:
                    part_children.append(cur_loop_node)

            elif c in ("LINE", "CIRCLE", "ARC"):
                curve_node = {
                    "id": f"{part_id}_curve_{i}",
                    "name": f"{c.capitalize()} Feature",
                    "type": c.lower(),
                    "icon": "spline" if c == "ARC" else ("circle" if c == "CIRCLE" else "minus"),
                    "cmd": c,
                    "cmd_index": i,
                    "params": dict(a),
                    "children": [],
                }
                if cur_loop_node is not None:
                    cur_loop_node["children"].append(curve_node)
                elif cur_face_node is not None:
                    cur_face_node["children"].append(curve_node)
                else:
                    part_children.append(curve_node)

            i += 1

        if i < total_len and cmds[i].startswith("EXTRUDE_"):
            extrude_cmd = cmds[i]
            extrude_arg = normalize_command_args(extrude_cmd, args[i] if i < len(args) else {})
            extrude_name = "Boss-Extrude"
            if extrude_cmd == "EXTRUDE_CUT":
                extrude_name = "Cut-Extrude"
            elif extrude_cmd == "EXTRUDE_INTERSECT":
                extrude_name = "Intersect-Extrude"
            elif extrude_cmd in ("EXTRUDE_THREAD", "EXTRUDE_THREADS"):
                extrude_name = "Threaded-Boss"
            elif extrude_cmd == "EXTRUDE_THREAD_CUT":
                extrude_name = "Threaded-Cut"

            extrude_node = {
                "id": f"{part_id}_extrude",
                "name": f"{extrude_name} {part_count}",
                "type": "extrude",
                "icon": "box",
                "cmd": extrude_cmd,
                "cmd_index": i,
                "params": dict(extrude_arg),
                "children": [],
            }
            part_children.append(extrude_node)
            i += 1

        part_node = {
            "id": part_id,
            "name": f"Solid Body {part_count} ({extrude_node['name'] if extrude_node else 'Feature'})",
            "type": "part",
            "icon": "box",
            "cmd": extrude_node["cmd"] if extrude_node else "PART",
            "cmd_index": extrude_node["cmd_index"] if extrude_node else None,
            "is_assembly_group": False,
            "params": dict(extrude_node["params"]) if extrude_node else {},
            "children": part_children,
        }
        tree_nodes.append(part_node)

    return {
        "root_name": "CAD Assembly",
        "part_count": part_count,
        "total_commands": len(cmds),
        "tree": tree_nodes,
    }


def render_and_export_dualseq(
    cmds: List[str],
    args: List[Dict[str, float]],
    uid_or_name: str = "cad_model",
) -> Dict[str, Any]:
    safe_name = uid_or_name.replace("/", "_").replace(" ", "_")
    stl_filename = f"{safe_name}.stl"
    png_filename = f"{safe_name}.png"
    stl_filepath = os.path.join(DEFAULT_STL_DIR, stl_filename)
    png_filepath = os.path.join(DEFAULT_STL_DIR, png_filename)

    tuples = list(zip(cmds, args))
    tuples = repair_dualseq_sequence(tuples)
    cmds = [c for c, _ in tuples]
    args = [a for _, a in tuples]

    ds = DualSeq(cmds=cmds, args=args, uid=uid_or_name)
    tree_data = parse_solidworks_part_tree(cmds, args)

    shape = None
    render_success = False
    error_message = None

    if OCC_AVAILABLE and render_dual_seq_to_shape is not None:
        try:
            shape = render_dual_seq_to_shape(ds)
            if shape is not None and not shape.IsNull():
                render_success = export_shape_to_stl(shape, stl_filepath)
                try:
                    if render_to_image is not None:
                        render_to_image(shape, png_filepath)
                except Exception:
                    pass
            else:
                error_message = "OpenCASCADE could not construct a valid solid from this sequence."
        except Exception as e:
            error_message = str(e)
    else:
        error_message = "OpenCASCADE (pythonocc-core) is not available in cloud environment. DualSeq and tree data generated."

    props = get_shape_properties(shape) if render_success else {"volume": 0.0, "bbox": {"min": [0, 0, 0], "max": [0, 0, 0]}}

    return {
        "success": render_success,
        "error": error_message,
        "stl_filename": stl_filename if render_success else None,
        "png_filename": png_filename if (render_success and os.path.isfile(png_filepath)) else None,
        "stl_url": f"/api/renders/{stl_filename}" if render_success else None,
        "png_url": f"/api/renders/{png_filename}" if (render_success and os.path.isfile(png_filepath)) else None,
        "tree_data": tree_data,
        "properties": props,
        "dualseq_tuples": tuples,
    }
