import os
import tempfile
import textwrap
from PIL import Image, ImageDraw, ImageFont

from .coord_system    import make_coord_system
from .sketch2d        import build_face_from_loops
from .extrude         import extrude_part
from .render_img      import render_to_image, render_with_text_side_by_side, format_dual_seq_representations, render_debug_instance_image
from .point_sampling  import sample_shape
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut, BRepAlgoAPI_Common
from OCC.Core.BRep import BRep_Builder
from OCC.Core.TopoDS import TopoDS_Compound

_OPS = {"EXTRUDE_JOIN": BRepAlgoAPI_Fuse, "EXTRUDE_CUT": BRepAlgoAPI_Cut, "EXTRUDE_INTERSECT": BRepAlgoAPI_Common}

def _bool_op(solid, body, op):
    """EXTRUDE_* boolean: NEW on existing body is treated as JOIN, JOIN->fuse, CUT->cut, INTERSECT->common."""
    if body is None:
        return solid
    if op == "EXTRUDE_NEW":
        op = "EXTRUDE_JOIN"
    if op not in _OPS:
        return solid
    try:
        res = _OPS[op](body, solid).Shape()
        if res is not None and not res.IsNull():
            return res
    except Exception:
        pass
    if op == "EXTRUDE_JOIN":
        try:
            builder = BRep_Builder()
            comp = TopoDS_Compound()
            builder.MakeCompound(comp)
            builder.Add(comp, body)
            builder.Add(comp, solid)
            return comp
        except Exception:
            pass
    return body

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
                if cur_loop: cur_face.append(cur_loop)
                if cur_face: part["faces"].append(cur_face)
                cur_face, cur_loop = [], None
            elif c == "LOOP":
                if cur_loop: cur_face.append(cur_loop)
                cur_loop = []
            elif c in ("LINE", "CIRCLE", "ARC") and cur_loop is not None:
                cur_loop.append((c, args[i]))
            i += 1
        if cur_loop: cur_face.append(cur_loop)
        if cur_face: part["faces"].append(cur_face)
        if i < len(cmds): part["extrude_cmd"], part["extrude_args"] = cmds[i], args[i]; i += 1
        parts.append(part)
    return parts


def render_dual_seq_to_shape(cmds, args=None):
    """Convert command and argument sequences or CADSequence into an OCC shape, or None on failure."""
    if hasattr(cmds, "create_cad_model"):
        try:
            cad_solid = cmds.create_cad_model()
            if cad_solid is not None and not cad_solid.IsNull():
                return cad_solid
        except Exception:
            pass
        try:
            from utils.dual_seq import DualSeq
            json_data = cmds._json() if hasattr(cmds, "_json") else {}
            if json_data and "parts" in json_data:
                ds = DualSeq(json_object=json_data)
                cmds, args = ds.cmds, ds.args_dict
            else:
                return None
        except Exception:
            return None

    if hasattr(cmds, "cmds"):
        args = getattr(cmds, "args_dict", getattr(cmds, "args", []))
        cmds = cmds.cmds
    if args is None:
        args = []
    if not cmds:
        return None

    body = None
    try:
        for part in _parse_parts(cmds, args):
            ea = part["extrude_args"]
            if ea is None: continue
            ax3   = make_coord_system(part["coor"])
            scale = 1.0
            dtn   = next((ea[k] for k in ea if k.endswith("_dtn")), 0.0) or 0.0
            don   = next((ea[k] for k in ea if k.endswith("_don")), 0.0) or 0.0
            part_solid = None
            for face_loops in part["faces"]:
                try:
                    face3d = build_face_from_loops(face_loops, ax3, scale)
                    solid  = extrude_part(face3d, ax3, dtn, don, "EXTRUDE_NEW")
                    part_solid = solid if part_solid is None else BRepAlgoAPI_Fuse(part_solid, solid).Shape()
                except Exception:
                    pass
            if part_solid is not None:
                body = _bool_op(part_solid, body, part["extrude_cmd"])
    except Exception:
        return None
    return body



def render_dual_seq_to_img(dual_seq, img_path: str, with_str: bool = False, with_desc: str = None) -> None:
    """Convert DualSeq -> isometric PNG at img_path.
    Per COOR->EXTRUDE_* block: build gp_Ax3 coord system, build 2D sketch faces
    (inner/outer loop detection in sketch2d), extrude each face, union all face
    solids into the PART solid, then apply the EXTRUDE_* boolean onto the running body."""
    import os
    import tempfile
    
    # Ensure final file path has a valid image extension
    _, ext = os.path.splitext(img_path)
    if ext.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".tiff"):
        img_path = img_path + ".png"

    # 1. Create the body
    body = None
    for part in _parse_parts(dual_seq.cmds, dual_seq.args_dict):
        ea = part["extrude_args"]
        if ea is None: continue
        ax3   = make_coord_system(part["coor"])
        scale = 1.0
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

    # 2. Save the image
    if body is not None:
        if with_str or with_desc:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                render_to_image(body, tmp_path)
                
                text_parts = []
                if with_desc:
                    descriptions = getattr(dual_seq, "descriptions", {})
                    desc_val = ""
                    desc_key = "expert"
                    if isinstance(descriptions, dict):
                        if isinstance(with_desc, str) and with_desc in descriptions:
                            desc_key = with_desc
                            desc_val = descriptions[with_desc]
                        elif isinstance(with_desc, str) and with_desc not in ("expert", "beginner", "intermediate", "abstract"):
                            desc_val = with_desc
                            desc_key = "text"
                        else:
                            for k in ["expert", "intermediate", "beginner", "abstract"]:
                                if k in descriptions and descriptions[k]:
                                    desc_val = descriptions[k]
                                    desc_key = k
                                    break
                            if not desc_val and descriptions:
                                desc_key, desc_val = next(iter(descriptions.items()))
                    elif isinstance(descriptions, str) and descriptions.strip():
                        desc_val = descriptions.strip()
                        desc_key = "text"

                    if desc_val:
                        import textwrap
                        text_parts.append(f"=== INPUT DESCRIPTION ({desc_key}) ===")
                        text_parts.append("-" * 60)
                        text_parts.extend(textwrap.wrap(str(desc_val), width=60))
                        text_parts.append("=" * 60)
                        text_parts.append("")
                
                if with_str:
                    text_parts.append(str(dual_seq))
                
                combined_text = "\n".join(text_parts).rstrip("\n")
                render_with_text_side_by_side(combined_text, tmp_path, img_path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        else:
            render_to_image(body, img_path)
    else :
        raise ValueError("No body was created from the DualSeq, check the validity of it")

def render_dual_seq_with_representations_to_img(dual_seq, img_path: str, metadata=None) -> None:
    body = render_dual_seq_to_shape(dual_seq.cmds, dual_seq.args)
    if body is None:
        raise ValueError("Failed to create OCC shape from DualSeq.")
        
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        
    try:
        render_to_image(body, tmp_path)
        text = format_dual_seq_representations(dual_seq, metadata)
        render_with_text_side_by_side(text, tmp_path, img_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def render_model_comparison_grid(
    models_predictions: list,
    gt_cmds: list,
    gt_args: list,
    output_path: str,
    uid: str = "",
    prompt_text: str = "",
    cell_size: tuple = (320, 320),
) -> str:
    """
    Renders multiple model CAD predictions alongside Ground Truth horizontally into a grid image.
    models_predictions: list of (model_name, cmds, args, cd_score)
    gt_cmds, gt_args: ground truth CAD commands and arguments
    """
    font_paths = [
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    font_bold = None
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                font_bold = ImageFont.truetype(fp, size=15)
                break
            except Exception:
                pass
    if font_bold is None:
        font_bold = ImageFont.load_default()

    font_regular_paths = [
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    font_regular = None
    for fp in font_regular_paths:
        if os.path.exists(fp):
            try:
                font_regular = ImageFont.truetype(fp, size=13)
                break
            except Exception:
                pass
    if font_regular is None:
        font_regular = ImageFont.load_default()

    all_entries = list(models_predictions)
    all_entries.append(("Ground Truth", gt_cmds, gt_args, None))

    cell_width, cell_height = cell_size
    rendered_cells = []
    tmp_files = []

    try:
        for model_name, cmds, args, cd_score in all_entries:
            cell_img = Image.new("RGB", (cell_width, cell_height + 40), color=(255, 255, 255))
            draw = ImageDraw.Draw(cell_img)

            is_gt = (model_name == "Ground Truth")
            hdr_color = (230, 240, 250) if is_gt else (242, 244, 247)
            draw.rectangle([0, 0, cell_width, 38], fill=hdr_color)
            draw.line([(0, 38), (cell_width, 38)], fill=(200, 205, 215), width=1)

            title_text = f"{model_name}"
            if is_gt:
                sub_text = "GT Reference"
            elif cd_score is not None and not (isinstance(cd_score, float) and cd_score != cd_score):
                sub_text = f"CD: {cd_score:.4f}"
            else:
                sub_text = "CD: N/A"

            draw.text((10, 5), title_text, fill=(20, 20, 20), font=font_bold)
            draw.text((10, 21), sub_text, fill=(90, 100, 110), font=font_regular)

            shape = render_dual_seq_to_shape(cmds, args)
            if shape is not None:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_f:
                    tmp_cell_path = tmp_f.name
                    tmp_files.append(tmp_cell_path)
                try:
                    render_to_image(shape, tmp_cell_path, size=(cell_width, cell_height))
                    shape_img = Image.open(tmp_cell_path)
                    cell_img.paste(shape_img, (0, 40))
                except Exception:
                    draw.rectangle([0, 40, cell_width, cell_height + 40], fill=(248, 249, 250))
                    draw.text((20, cell_height // 2), "Render Error", fill=(200, 40, 40), font=font_bold)
            else:
                draw.rectangle([0, 40, cell_width, cell_height + 40], fill=(248, 249, 250))
                draw.text((20, cell_height // 2), "Invalid / Failed Shape", fill=(180, 50, 50), font=font_bold)

            draw.rectangle([0, 0, cell_width - 1, cell_height + 39], outline=(210, 215, 220), width=1)
            rendered_cells.append(cell_img)

        num_cols = len(rendered_cells)
        gap = 10
        total_width = num_cols * cell_width + (num_cols + 1) * gap

        short_prompt = prompt_text[:100] + "..." if len(prompt_text) > 100 else prompt_text
        prompt_lines = textwrap.wrap(short_prompt, width=100) if short_prompt else []
        banner_height = 36 + max(len(prompt_lines), 1) * 18
        total_height = banner_height + cell_height + 40 + gap * 2

        composite = Image.new("RGB", (total_width, total_height), color=(250, 252, 255))
        draw_comp = ImageDraw.Draw(composite)

        draw_comp.rectangle([gap, gap, total_width - gap, banner_height], fill=(255, 255, 255), outline=(215, 220, 230), width=1)
        draw_comp.text((gap + 12, gap + 6), f"Instance UID: {uid}", fill=(15, 23, 42), font=font_bold)
        y_text = gap + 24
        for pline in prompt_lines:
            draw_comp.text((gap + 12, y_text), f"Prompt: {pline}", fill=(71, 85, 105), font=font_regular)
            y_text += 18

        cell_y = banner_height + gap
        for idx, cell_img in enumerate(rendered_cells):
            cell_x = gap + idx * (cell_width + gap)
            composite.paste(cell_img, (cell_x, cell_y))

        _, ext = os.path.splitext(output_path)
        if ext.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".tiff"):
            output_path = output_path + ".png"

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        composite.save(output_path)
    finally:
        for tf in tmp_files:
            if os.path.exists(tf):
                try:
                    os.remove(tf)
                except Exception:
                    pass

    return output_path