DEFAULT_COMMANDS = {
    "COOR": ["coor_euax", "coor_euay", "coor_euaz", "coor_tx", "coor_ty", "coor_tz"],
    "FACE": [],
    "LOOP": [],
    "LINE": ["line_sx", "line_sy", "line_ex", "line_ey"],
    "CIRCLE": ["circle_cx", "circle_cy", "circle_r"],
    "EXTRUDE_NEW": ["extrude_new_dtn", "extrude_new_don", "extrude_new_scale"],
    "EXTRUDE_JOIN": ["extrude_join_dtn", "extrude_join_don", "extrude_join_scale"],
    "EXTRUDE_CUT": ["extrude_cut_dtn", "extrude_cut_don", "extrude_cut_scale"],
    "EXTRUDE_INTERSECT": ["extrude_intersect_dtn", "extrude_intersect_don", "extrude_intersect_scale"],
}

class DualSeq:
    EXTRUSION_OPERATORS = {'NewBodyFeatureOperation', 
                           'JoinFeatureOperation', 
                           'CutFeatureOperation', 
                           'IntersectFeatureOperation'
                           }
    
    
    def __init__(self, json_object: dict, uid: str):
        self.json_to_dual_seq(json_object, uid)
        
    def json_to_dual_seq(self, json_object: dict, uid: str) -> list[dict]:
        self.uid = uid
        
        # The body of dual_seq
        self.cmds = []
        self.args = []
        
        self.json_object = json_object
        self.extrusion_kinds = set()
        
        for part_key, part in json_object["parts"].items():
            # COORDINATE COMMAND
            coordinate_system = part["coordinate_system"]
            euler_angles = coordinate_system["Euler Angles"]
            translation_vector = coordinate_system["Translation Vector"]
            
            self.cmds.append("COOR")
            self.args.append({
                "coor_euax": euler_angles[0],
                "coor_euay": euler_angles[1],
                "coor_euaz": euler_angles[2],
                "coor_tx": translation_vector[0],
                "coor_ty": translation_vector[1],
                "coor_tz": translation_vector[2],
            })
            
            # SKETCH COMMANDS
            for face_key, face in part["sketch"].items():
                self.cmds.append("FACE")
                self.args.append({})
                
                more_cmds, more_args = self.face_to_cmd(face)
                self.cmds.extend(more_cmds)
                self.args.extend(more_args)\
            
            # EXTRUDE COMMAND
            extrusion = part["extrusion"]
            self.extrusion_kinds.add(extrusion["operation"]) # For exploration, remove at release
            
            more_cmds, more_args = self.extrusion_to_cmd(extrusion)

            self.cmds.extend(more_cmds)
            self.args.extend(more_args)

    
    def face_to_cmd(self, face: dict) -> tuple[list[str], list[dict]]:
        cmds = []
        args = []
        
        for loop_key, loop in face.items():
            # LOOP COMMAND
            cmds.append("LOOP")
            args.append({})

            for segment_key, segment in loop.items():
                
                
                # EXTRACT LINES
                if segment_key.startswith("line"):
                    start_point = segment["Start Point"]
                    end_point = segment["End Point"]
                    cmds.append("LINE")
                    args.append({
                        "line_sx": start_point[0],
                        "line_sy": start_point[1],
                        "line_ex": end_point[0],
                        "line_ey": end_point[1],
                    })
                    
                # EXTRACT CIRCLES
                elif segment_key.startswith("circle"):
                    center = segment["Center"]
                    radius = segment["Radius"]
                    cmds.append("CIRCLE")
                    args.append({
                        "circle_cx": center[0],
                        "circle_cy": center[1],
                        "circle_r": radius,
                    })
                    
                    
                
        return cmds, args
    
    def extrusion_to_cmd(self, extrusion: dict) -> tuple[list[str], list[dict]]: 
        operation = extrusion["operation"]
        
        if operation not in self.EXTRUSION_OPERATORS:
            raise ValueError(f"Unsupported extrusion operation: {operation}")
        
        if operation == "NewBodyFeatureOperation":
            cmd = "EXTRUDE_NEW"
            args = {
                "extrude_new_dtn": extrusion["extrude_depth_towards_normal"],
                "extrude_new_don": extrusion["extrude_depth_opposite_normal"],
                "extrude_new_scale": extrusion["sketch_scale"]
            }
            
            
        elif operation == "JoinFeatureOperation":
            cmd = "EXTRUDE_JOIN"
            args = {
                "extrude_join_dtn": extrusion["extrude_depth_towards_normal"],
                "extrude_join_don": extrusion["extrude_depth_opposite_normal"],
                "extrude_join_scale": extrusion["sketch_scale"]
            }
        elif operation == "CutFeatureOperation":
            cmd = "EXTRUDE_CUT"
            args = {
                "extrude_cut_dtn": extrusion["extrude_depth_towards_normal"],
                "extrude_cut_don": extrusion["extrude_depth_opposite_normal"],
                "extrude_cut_scale": extrusion["sketch_scale"]
            }
        elif operation == "IntersectFeatureOperation":
            cmd = "EXTRUDE_INTERSECT"
            args = {
                "extrude_intersect_dtn": extrusion["extrude_depth_towards_normal"],
                "extrude_intersect_don": extrusion["extrude_depth_opposite_normal"],
                "extrude_intersect_scale": extrusion["sketch_scale"]
            }
  
        return [cmd], [args]
    
    def __str__(self) -> str:
        # Build a simple table with columns: cmd, args (each cmd corresponds to args)
        cmds  = self.cmds
        args = self.args
        rows = []
        rows.append(("cmd", "args"))
        for c, a in zip(cmds, args):
            rows.append((str(c), str(a)))

        # compute column widths
        col1_w = max(len(r[0]) for r in rows)
        col2_w = max(len(r[1]) for r in rows)

        lines = []
        # header
        lines.append(f"{rows[0][0].ljust(col1_w)} | {rows[0][1].ljust(col2_w)}")
        lines.append(f"{'-'*col1_w}-+-{'-'*col2_w}")
        # data
        for r in rows[1:]:
            lines.append(f"{r[0].ljust(col1_w)} | {r[1].ljust(col2_w)}")

        return "\n".join(lines)
    
    @property
    def part_count(self) -> int:
        return len(self.json_object["parts"])