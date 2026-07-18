import numpy as np

def R_to_euler(transform: dict[str, dict[str, float]]):
    x_axis = np.array([transform["x_axis"]["x"], transform["x_axis"]["y"], transform["x_axis"]["z"]])
    y_axis = np.array([transform["y_axis"]["x"], transform["y_axis"]["y"], transform["y_axis"]["z"]])
    z_axis = np.array([transform["z_axis"]["x"], transform["z_axis"]["y"], transform["z_axis"]["z"]])
    R = np.column_stack((x_axis, y_axis, z_axis))
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0
    return {"euax": roll, "euay": pitch, "euaz": yaw}

def line_3d_to_line_2d(line_3d: dict[str, dict[str, float]], coor_args: dict[str, float]):
    start_3d = line_3d["start_point"]
    end_3d = line_3d["end_point"]
    tx = coor_args["coor_tx"]
    ty = coor_args["coor_ty"]
    tz = coor_args["coor_tz"]
    euax = coor_args["coor_euax"]
    euay = coor_args["coor_euay"]
    euaz = coor_args["coor_euaz"]
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(euax), -np.sin(euax)],
                    [0, np.sin(euax), np.cos(euax)]])
    Ry = np.array([[np.cos(euay), 0, np.sin(euay)],
                    [0, 1, 0],
                    [-np.sin(euay), 0, np.cos(euay)]])
    Rz = np.array([[np.cos(euaz), -np.sin(euaz), 0],
                    [np.sin(euaz), np.cos(euaz), 0],
                    [0, 0, 1]])
    R = Rz @ Ry @ Rx
    start_3d_vec = np.array([start_3d["x"], start_3d["y"], start_3d["z"]]) - np.array([tx, ty, tz])
    end_3d_vec = np.array([end_3d["x"], end_3d["y"], end_3d["z"]]) - np.array([tx, ty, tz])
    start_2d_vec = R.T @ start_3d_vec
    end_2d_vec = R.T @ end_3d_vec
    return {
        "sx": start_2d_vec[0],
        "sy": start_2d_vec[1],
        "ex": end_2d_vec[0],
        "ey": end_2d_vec[1]
    }

def circle_3d_to_circle_2d(circle_3d: dict[str, dict[str, float]], coor_args: dict[str, float]):
    center_3d = circle_3d["center_point"]
    radius = circle_3d["radius"]
    tx = coor_args["coor_tx"]
    ty = coor_args["coor_ty"]
    tz = coor_args["coor_tz"]
    euax = coor_args["coor_euax"]
    euay = coor_args["coor_euay"]
    euaz = coor_args["coor_euaz"]
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(euax), -np.sin(euax)],
                    [0, np.sin(euax), np.cos(euax)]])
    Ry = np.array([[np.cos(euay), 0, np.sin(euay)],
                    [0, 1, 0],
                    [-np.sin(euay), 0, np.cos(euay)]])
    Rz = np.array([[np.cos(euaz), -np.sin(euaz), 0],
                    [np.sin(euaz), np.cos(euaz), 0],
                    [0, 0, 1]])
    R = Rz @ Ry @ Rx
    center_3d_vec = np.array([center_3d["x"], center_3d["y"], center_3d["z"]]) - np.array([tx, ty, tz])
    center_2d_vec = R.T @ center_3d_vec
    return {
        "cx": center_2d_vec[0],
        "cy": center_2d_vec[1],
        "r": radius
    }

def arc_3d_to_arc_2d(arc_3d: dict[str, dict[str, float]], coor_args: dict[str, float]):
    center_3d = arc_3d["center_point"]
    radius = arc_3d["radius"]
    start_point = arc_3d["start_point"]
    end_point = arc_3d["end_point"]
    tx = coor_args["coor_tx"]
    ty = coor_args["coor_ty"]
    tz = coor_args["coor_tz"]
    euax = coor_args["coor_euax"]
    euay = coor_args["coor_euay"]
    euaz = coor_args["coor_euaz"]
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(euax), -np.sin(euax)],
                    [0, np.sin(euax), np.cos(euax)]])
    Ry = np.array([[np.cos(euay), 0, np.sin(euay)],
                    [0, 1, 0],
                    [-np.sin(euay), 0, np.cos(euay)]])
    Rz = np.array([[np.cos(euaz), -np.sin(euaz), 0],
                    [np.sin(euaz), np.cos(euaz), 0],
                    [0, 0, 1]])
    R = Rz @ Ry @ Rx
    center_3d_vec = np.array([center_3d["x"], center_3d["y"], center_3d["z"]]) - np.array([tx, ty, tz])
    start_3d_vec = np.array([start_point["x"], start_point["y"], start_point["z"]]) - np.array([tx, ty, tz])
    end_3d_vec = np.array([end_point["x"], end_point["y"], end_point["z"]]) - np.array([tx, ty, tz])
    center_2d_vec = R.T @ center_3d_vec
    start_2d_vec = R.T @ start_3d_vec
    end_2d_vec = R.T @ end_3d_vec
    angle_between_start_end = np.arctan2(end_2d_vec[1] - center_2d_vec[1], end_2d_vec[0] - center_2d_vec[0]) - np.arctan2(start_2d_vec[1] - center_2d_vec[1], start_2d_vec[0] - center_2d_vec[0])
    half_angle = angle_between_start_end / 2
    mid_2d_vec = center_2d_vec + radius * np.array([np.cos(np.arctan2(start_2d_vec[1] - center_2d_vec[1], start_2d_vec[0] - center_2d_vec[0]) + half_angle), np.sin(np.arctan2(start_2d_vec[1] - center_2d_vec[1], start_2d_vec[0] - center_2d_vec[0]) + half_angle)])    
    return {
        "cx": center_2d_vec[0],
        "cy": center_2d_vec[1],
        "r": radius,
        "start_angle": np.arctan2(start_2d_vec[1] - center_2d_vec[1], start_2d_vec[0] - center_2d_vec[0]),
        "end_angle": np.arctan2(end_2d_vec[1] - center_2d_vec[1], end_2d_vec[0] - center_2d_vec[0]),
        "mid_x": mid_2d_vec[0],
        "mid_y": mid_2d_vec[1]
    }
