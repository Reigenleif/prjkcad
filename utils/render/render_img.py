import os
import tempfile
from PIL import Image, ImageDraw, ImageFont
from OCC.Display.OCCViewer import Viewer3d
from OCC.Core.gp import gp_Dir
from OCC.Core.Quantity import Quantity_Color, Quantity_NOC_WHITE, Quantity_TOC_RGB
from OCC.Core.V3d import V3d_DirectionalLight
from OCC.Extend.TopologyUtils import TopologyExplorer

def render_to_image(
    shape,
    filepath: str,
    size=(512, 512),
    face_color_rgb=(0.2, 0.2, 0.2),
    edge_color_rgb=(0, 0, 0),
    show_face_boundary=True,
    margin=0.1
) -> None:
    """
    Render a TopoDS_Shape to a high-fidelity isometric PNG using Python-OCC's
    internal OpenGL rendering pipeline via Viewer3d.

    Parameters:
    -----------
    shape : TopoDS_Shape
        The OpenCASCADE shape to render.
    filepath : str
        The output path for the saved PNG image.
    size : tuple of int, optional
        The width and height of the rendered image (default is (512, 512)).
    face_color_rgb : tuple of float, optional
        The RGB values for faces, normalized between 0 and 1.
    edge_color_rgb : tuple of float, optional
        The RGB values for edges, normalized between 0 and 1.
    show_face_boundary : bool, optional
        Whether to draw face boundaries (default is True).
    margin : float, optional
        The margin coefficient for view borders (default is 0.1 for 10% padding).
    """
    width, height = size
    viewer = Viewer3d()
    viewer.Create(phong_shading=True, create_default_lights=True)
    viewer.set_bg_gradient_color([255, 255, 255], [255, 255, 255])
    viewer.SetModeShaded()
    viewer.hide_triedron()
    viewer.EnableAntiAliasing()
    
    dir_light = V3d_DirectionalLight(gp_Dir(0, 0.5, -1), Quantity_Color(Quantity_NOC_WHITE))
    dir_light.SetEnabled(True)
    dir_light.SetIntensity(500.0)
    viewer.Viewer.AddLight(dir_light)
    viewer.Viewer.SetLightOn()

    viewer.default_drawer.EnableDrawHiddenLine()
    viewer.default_drawer.SetFaceBoundaryDraw(show_face_boundary)
    ais_context = viewer.GetContext()
    
    # Increase tessellation/deviation resolution for smoother rendering
    dc = ais_context.DeviationCoefficient()
    da = ais_context.DeviationAngle()
    factor = 10
    ais_context.SetDeviationCoefficient(dc / factor)
    ais_context.SetDeviationAngle(da / factor)
    
    topexp = TopologyExplorer(shape)
    for face in topexp.faces():
        if face is not None:
            viewer.DisplayShape(face, color=Quantity_Color(*face_color_rgb, Quantity_TOC_RGB))
    for edge in topexp.edges():
        if edge is not None:
            viewer.DisplayShape(edge, color=Quantity_Color(*edge_color_rgb, Quantity_TOC_RGB))
            
    viewer.SetSize(width, height)
    viewer.View.FitAll(margin)
    viewer.View.ZFitAll()
    
    # Ensure file has a valid image extension
    _, ext = os.path.splitext(filepath)
    if ext.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".tiff"):
        filepath = filepath + ".png"

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    success = viewer.View.Dump(str(filepath))
    if not success:
        raise IOError(
            f"OpenCASCADE failed to dump view to {filepath}. "
            "Please check if the file format/extension is supported."
        )

def render_with_text_side_by_side(text: str, render_img_path: str, output_path: str) -> str:
    """
    Renders monospace text on the left panel and places the 3D rendered image on the right panel.
    Returns the final file path where the image is saved.
    """
    from PIL import Image, ImageDraw, ImageFont
    
    # Candidate monospace font paths for code font rendering on Linux/Ubuntu
    font_paths = [
        "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]
    font = None
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, size=14)
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()

    lines = text.split("\n")
    
    # Open the 3D rendered image
    if not os.path.exists(render_img_path):
        raise FileNotFoundError(f"3D render image not found at {render_img_path}")
    render_img = Image.open(render_img_path)
    render_w, render_h = render_img.size

    # Measure text height and width to size the left panel
    dummy_img = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy_img)

    max_line_w = 0
    line_heights = []
    for line in lines:
        try:
            bbox = draw.textbbox((0, 0), line, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
        except AttributeError:
            w, h = draw.textsize(line, font=font)
        max_line_w = max(max_line_w, w)
        line_heights.append(h)

    padding = 20
    line_spacing = 4
    total_text_height = sum(line_heights) + (len(lines) - 1) * line_spacing + 2 * padding
    total_text_width = max_line_w + 2 * padding

    combined_h = max(render_h, total_text_height)
    combined_w = total_text_width + render_w

    # Create new combined image with white background
    combined_img = Image.new("RGB", (combined_w, combined_h), color=(255, 255, 255))
    draw_comb = ImageDraw.Draw(combined_img)

    # Draw text lines on the left side
    y = padding
    for line, h in zip(lines, line_heights):
        draw_comb.text((padding, y), line, fill=(0, 0, 0), font=font)
        y += h + line_spacing

    # Paste the 3D render image on the right (centered vertically)
    paste_y = (combined_h - render_h) // 2
    combined_img.paste(render_img, (total_text_width, paste_y))

    # Ensure final output path has a valid image extension
    _, ext = os.path.splitext(output_path)
    if ext.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".tiff"):
        output_path = output_path + ".png"

    # Save to the final output path (ensuring parent directories exist)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    combined_img.save(output_path)
    return output_path

def format_dual_seq_representations(dual_seq, metadata=None) -> str:
    lines = []
    lines.append("[1] BASIC REPRESENTATION")
    lines.append(f"Commands: {dual_seq.cmds}")
    lines.append("Args:")
    for cmd, arg in zip(dual_seq.cmds, dual_seq.args):
        arg_str = ", ".join(f"{k}={v:.4f}" for k, v in arg.items())
        lines.append(f"  {cmd}: {arg_str}")
    lines.append("")
    
    lines.append("[2] ONE-HEAD TOKENIZED REPRESENTATION")
    try:
        from utils.representations.converter import basic_to_one_head_tokenized
        from utils.dual_seq import get_dualseq_schema
        schema = get_dualseq_schema()
        tokens = basic_to_one_head_tokenized(dual_seq.cmds, dual_seq.args, schema)
        lines.append(f"Tokens: {tokens[:25]} ...")
        id_to_arg = schema["id_to_arg"]
        token_strs = [id_to_arg.get(t, str(t)) for t in tokens]
        lines.append(f"Token Streams: {' '.join(token_strs[:20])} ...")
    except Exception as e:
        lines.append(f"Error: {e}")
    lines.append("")
    
    lines.append("[3] EIGHT-BIT BINARIZED REPRESENTATION")
    if metadata is not None:
        try:
            from utils.representations.converter import basic_to_eight_bit_binarized
            binned = basic_to_eight_bit_binarized(dual_seq.cmds, dual_seq.args, metadata)
            lines.append("Binned Args:")
            for cmd, arg in zip(dual_seq.cmds, binned):
                if arg:
                    arg_str = ", ".join(f"{k}={v}" for k, v in arg.items())
                    lines.append(f"  {cmd}: {arg_str}")
        except Exception as e:
            lines.append(f"Error: {e}")
    else:
        lines.append("No DualSeqMetadata provided for 8-bit representation.")
    lines.append("")
    
    return "\n".join(lines)


def render_debug_instance_image(
    uid: str,
    gt_cmds: list[str],
    gt_args: list[dict],
    pred_cmds: list[str],
    pred_args: list[dict],
    cd_score: float,
    reward: float,
    valid: bool,
    output_path: str,
    size=(512, 512)
) -> str:
    """
    Renders a composite debug image for a DualSeq instance:
    - Text panel: UID, GRPO reward, Chamfer Distance, GT sequence, Pred sequence.
    - Top render: Ground Truth 3D CAD shape.
    - Bottom render: Predicted 3D CAD shape.
    """
    from .__init__ import render_dual_seq_to_shape
    tmp_gt_path = None
    tmp_pred_path = None
    tmp_panel_path = None

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_gt, \
         tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_pred:
        tmp_gt_path = tmp_gt.name
        tmp_pred_path = tmp_pred.name

    try:
        # 1. Ground Truth Render
        gt_shape = render_dual_seq_to_shape(gt_cmds, gt_args)
        if gt_shape is not None:
            render_to_image(gt_shape, tmp_gt_path, size=size)
            gt_img = Image.open(tmp_gt_path)
        else:
            gt_img = Image.new("RGB", size, color=(245, 245, 245))
            d = ImageDraw.Draw(gt_img)
            d.text((40, size[1] // 2), "GT Render Failed", fill=(180, 0, 0))

        # 2. Predicted Render
        pred_shape = render_dual_seq_to_shape(pred_cmds, pred_args) if valid else None
        if pred_shape is not None:
            render_to_image(pred_shape, tmp_pred_path, size=size)
            pred_img = Image.open(tmp_pred_path)
        else:
            pred_img = Image.new("RGB", size, color=(245, 245, 245))
            d = ImageDraw.Draw(pred_img)
            d.text((40, size[1] // 2), "Pred Render Failed / Invalid", fill=(180, 0, 0))

        # Add Titles
        d_gt = ImageDraw.Draw(gt_img)
        d_gt.rectangle([0, 0, size[0], 25], fill=(220, 220, 220))
        d_gt.text((10, 5), "Ground Truth Shape", fill=(0, 0, 0))

        d_pred = ImageDraw.Draw(pred_img)
        d_pred.rectangle([0, 0, size[0], 25], fill=(220, 220, 220))
        d_pred.text((10, 5), "Predicted Shape", fill=(0, 0, 0))

        # Combined 3D Render Panel (GT on top, Pred on bottom)
        render_panel = Image.new("RGB", (size[0], size[1] * 2), color=(255, 255, 255))
        render_panel.paste(gt_img, (0, 0))
        render_panel.paste(pred_img, (0, size[1]))

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_panel:
            tmp_panel_path = tmp_panel.name
        render_panel.save(tmp_panel_path)

        # Build Monospace Text Panel
        lines = [
            f"UID: {uid}",
            f"GRPO Reward: {reward:.4f}",
            f"Chamfer Dist: {cd_score:.5f}",
            f"Valid Model: {valid}",
            "=" * 55,
            "=== GROUND TRUTH SEQUENCE ===",
            f"Cmds ({len(gt_cmds)}): {gt_cmds}",
            "Args:"
        ]
        for c, a in zip(gt_cmds, gt_args):
            a_str = ", ".join(f"{k}={v}" for k, v in a.items()) if isinstance(a, dict) else str(a)
            lines.append(f"  {c}: {a_str[:55]}")

        lines.extend([
            "",
            "=== PREDICTED SEQUENCE ===",
            f"Cmds ({len(pred_cmds)}): {pred_cmds}",
            "Args:"
        ])
        for c, a in zip(pred_cmds, pred_args):
            a_str = ", ".join(f"{k}={v}" for k, v in a.items()) if isinstance(a, dict) else str(a)
            lines.append(f"  {c}: {a_str[:55]}")

        combined_text = "\n".join(lines)
        render_with_text_side_by_side(combined_text, tmp_panel_path, output_path)

    finally:
        for p in [tmp_gt_path, tmp_pred_path, tmp_panel_path]:
            if p is not None and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

    return output_path


