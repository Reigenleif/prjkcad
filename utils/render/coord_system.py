import numpy as np
from scipy.spatial.transform import Rotation as R
from OCC.Core.gp import gp_Ax3, gp_Pnt, gp_Dir


def make_coord_system(coor_args: dict) -> gp_Ax3:
    """Build an OCC gp_Ax3 coordinate frame from COOR token args."""
    euax = float(coor_args.get("coor_euax", 0.0))
    euay = float(coor_args.get("coor_euay", 0.0))
    euaz = float(coor_args.get("coor_euaz", 0.0))
    tx = float(coor_args.get("coor_tx", 0.0))
    ty = float(coor_args.get("coor_ty", 0.0))
    tz = float(coor_args.get("coor_tz", 0.0))

    rot_matrix = R.from_euler(
        seq="zyx", angles=[euax, euay, euaz], degrees=True
    ).as_matrix()
    x_axis = rot_matrix[0]
    z_axis = rot_matrix[2]

    origin = gp_Pnt(tx, ty, tz)
    z_dir = gp_Dir(float(z_axis[0]), float(z_axis[1]), float(z_axis[2]))
    x_dir = gp_Dir(float(x_axis[0]), float(x_axis[1]), float(x_axis[2]))
    return gp_Ax3(origin, z_dir, x_dir)

