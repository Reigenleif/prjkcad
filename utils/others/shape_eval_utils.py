import os
import matplotlib.pyplot as plt
from PIL import Image

from utils.render import render_to_image, render_dual_seq_to_shape
from utils.evaluate import chamfer_distance_from_shapes
from utils.dual_seq import DualSeq

# Example Shapes
def make_cube(size=1.0):
    cmds = [
        "COOR", "FACE", "LOOP",
        "LINE", "LINE", "LINE", "LINE",
        "EXTRUDE_NEW"
    ]
    args = [
        {'coor_euax': 0.0, 'coor_euay': 0.0, 'coor_euaz': 0.0, 'coor_tx': 0.0, 'coor_ty': 0.0, 'coor_tz': 0.0},
        {}, {},
        {'line_sx': 0.0, 'line_sy': 0.0, 'line_ex': size, 'line_ey': 0.0},
        {'line_sx': size, 'line_sy': 0.0, 'line_ex': size, 'line_ey': size},
        {'line_sx': size, 'line_sy': size, 'line_ex': 0.0, 'line_ey': size},
        {'line_sx': 0.0, 'line_sy': size, 'line_ex': 0.0, 'line_ey': 0.0},
        {'extrude_new_dtn': size, 'extrude_new_don': 0.0, 'extrude_new_scale': 1.0}
    ]
    return render_dual_seq_to_shape(cmds, args)

def make_cylinder_on_top(cube_size=1.0, cyl_r=0.3, cyl_h=0.5):
    cmds = [
        "COOR", "FACE", "LOOP",
        "LINE", "LINE", "LINE", "LINE",
        "EXTRUDE_NEW",
        "COOR", "FACE", "LOOP",
        "CIRCLE",
        "EXTRUDE_JOIN"
    ]
    args = [
        {'coor_euax': 0.0, 'coor_euay': 0.0, 'coor_euaz': 0.0, 'coor_tx': 0.0, 'coor_ty': 0.0, 'coor_tz': 0.0},
        {}, {},
        {'line_sx': 0.0, 'line_sy': 0.0, 'line_ex': cube_size, 'line_ey': 0.0},
        {'line_sx': cube_size, 'line_sy': 0.0, 'line_ex': cube_size, 'line_ey': cube_size},
        {'line_sx': cube_size, 'line_sy': cube_size, 'line_ex': 0.0, 'line_ey': cube_size},
        {'line_sx': 0.0, 'line_sy': cube_size, 'line_ex': 0.0, 'line_ey': 0.0},
        {'extrude_new_dtn': cube_size, 'extrude_new_don': 0.0, 'extrude_new_scale': 1.0},
        {'coor_euax': 0.0, 'coor_euay': 0.0, 'coor_euaz': 0.0, 'coor_tx': 0.0, 'coor_ty': 0.0, 'coor_tz': cube_size},
        {}, {},
        {'circle_cx': cube_size / 2.0, 'circle_cy': cube_size / 2.0, 'circle_r': cyl_r},
        {'extrude_join_dtn': cyl_h, 'extrude_join_don': 0.0, 'extrude_join_scale': 1.0}
    ]
    return render_dual_seq_to_shape(cmds, args)

def make_cylinder(r=0.5, h=1.0):
    cmds = [
        "COOR", "FACE", "LOOP",
        "CIRCLE",
        "EXTRUDE_NEW"
    ]
    args = [
        {'coor_euax': 0.0, 'coor_euay': 0.0, 'coor_euaz': 0.0, 'coor_tx': 0.0, 'coor_ty': 0.0, 'coor_tz': 0.0},
        {}, {},
        {'circle_cx': 0.0, 'circle_cy': 0.0, 'circle_r': r},
        {'extrude_new_dtn': h, 'extrude_new_don': 0.0, 'extrude_new_scale': 1.0}
    ]
    return render_dual_seq_to_shape(cmds, args)


# Viz tools
def visualize_pair(shape1, shape2, name1, name2, out_dir="out/visualizations"):
    os.makedirs(out_dir, exist_ok=True)
    path1 = os.path.join(out_dir, f"{name1}.png")
    path2 = os.path.join(out_dir, f"{name2}.png")
    
    render_to_image(shape1, path1)
    render_to_image(shape2, path2)
    
    cd = chamfer_distance_from_shapes(shape1, shape2, n_u=20, n_v=20)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    img1 = Image.open(path1)
    img2 = Image.open(path2)
    
    axes[0].imshow(img1)
    axes[0].set_title(name1, fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(img2)
    axes[1].set_title(name2, fontsize=12)
    axes[1].axis('off')
    
    plt.suptitle(f"Chamfer Distance: {cd:.6f}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()