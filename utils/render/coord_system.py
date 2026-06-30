import numpy as np
from OCC.Core.gp import gp_Ax3, gp_Pnt, gp_Dir


def make_coord_system(coor_args: dict) -> gp_Ax3:
    """
    Build an OCC gp_Ax3 (local sketch plane) from COOR token args.

    Steps:
    1. Extract Euler angles (roll=euax, pitch=euay, yaw=euaz) + translation (tx,ty,tz)
    2. Build Rx, Ry, Rz rotation matrices using ZYX Euler convention
    3. Combine: R = Rz @ Ry @ Rx — columns are local X,Y,Z axes in world frame
    4. Extract local X-axis (col 0) and Z-axis (col 2 = sketch normal / extrude dir)
    5. Construct gp_Ax3(origin, Z-normal, X-direction) → right-handed coordinate frame
    """
    euax = coor_args["coor_euax"]   # roll  (rotation about world X)
    euay = coor_args["coor_euay"]   # pitch (rotation about world Y)
    euaz = coor_args["coor_euaz"]   # yaw   (rotation about world Z)
    tx, ty, tz = coor_args["coor_tx"], coor_args["coor_ty"], coor_args["coor_tz"]

    # Step 2: Individual rotation matrices
    cx, sx = np.cos(euax), np.sin(euax)
    cy, sy = np.cos(euay), np.sin(euay)
    cz, sz = np.cos(euaz), np.sin(euaz)

    Rx = np.array([[1,   0,   0],
                   [0,  cx, -sx],
                   [0,  sx,  cx]])
    Ry = np.array([[ cy, 0, sy],
                   [  0, 1,  0],
                   [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0],
                   [sz,  cz, 0],
                   [ 0,   0, 1]])

    # Step 3: Combined rotation — apply Rx first, then Ry, then Rz (ZYX order)
    R = Rz @ Ry @ Rx

    # Step 4: Extract local axes from rotation matrix columns
    x_axis = R[:, 0]   # local X: horizontal direction in the sketch plane
    z_axis = R[:, 2]   # local Z: normal to the sketch plane = extrude direction

    # Step 5: Build OCC coordinate system
    origin = gp_Pnt(float(tx), float(ty), float(tz))
    z_dir  = gp_Dir(float(z_axis[0]), float(z_axis[1]), float(z_axis[2]))
    x_dir  = gp_Dir(float(x_axis[0]), float(x_axis[1]), float(x_axis[2]))
    return gp_Ax3(origin, z_dir, x_dir)
