import json
import tempfile

class DualSeq:
    EXTRUSION_OPERATORS = {"NewBodyFeatureOperation", 
                           "JoinFeatureOperation", 
                           "CutFeatureOperation", 
                           "IntersectFeatureOperation"}

    def __init__(self, 
                 json_object: dict, 
                 uid: str, 
                 descriptions: dict[str, str] | None = None):
        
        # Default input format : text2cad
        self.from_text2cad(json_object, uid, descriptions)

    def from_text2cad(self, 
                      json_object: dict, 
                      uid: str,
                      descriptions: dict[str, str] | None):
        self.uid = uid
        self.json_object = json_object
        self.cmds: list[str] = []
        self.args: list[dict] = []
        self.extrusion_kinds: set[str] = set()
        
        self.descriptions = descriptions or {}
        
        # DECODE EACH PART
        for part_name, part in json_object["parts"].items():
            # COORDINATE SYSTEM (Also the indicates the start of a new part so we dont need a token for PART)
            coordinate_system = part["coordinate_system"]
            euler_angles = coordinate_system["Euler Angles"]
            translation_vector = coordinate_system["Translation Vector"]
            self.cmds.append("COOR")
            self.args.append({"coor_euax": euler_angles[0], 
                              "coor_euay": euler_angles[1], 
                              "coor_euaz": euler_angles[2], 
                              "coor_tx": translation_vector[0], 
                              "coor_ty": translation_vector[1], 
                              "coor_tz": translation_vector[2]})
            
            
            # FACES
            # DECODE FOR EACH FACE
            for face_name, face in part["sketch"].items():
                self.cmds.append("FACE")
                self.args.append({})
                more_cmds, more_args = self.face_to_cmd(face)
                self.cmds.extend(more_cmds)
                self.args.extend(more_args)
                
            # EXTRUSION
            extrusion = part["extrusion"]
            self.extrusion_kinds.add(extrusion["operation"])
            more_cmds, more_args = self.extrusion_to_cmd(extrusion)
            self.cmds.extend(more_cmds)
            self.args.extend(more_args)
            
        return self.args

    def face_to_cmd(self, face: dict) -> tuple[list[str], list[dict]]:
        cmds: list[str] = []
        args: list[dict] = []
        
        # DECODE EACH LOOP
        for loop_name, loop in face.items():
            cmds.append("LOOP")
            args.append({})
            
            # DECODE EACH SEGMENT
            for segment_name, segment in loop.items():
                # LINE
                if str(segment_name).startswith("line"):
                    start_point = segment["Start Point"]
                    end_point = segment["End Point"]
                    cmds.append("LINE")
                    args.append({"line_sx": start_point[0], "line_sy": start_point[1], "line_ex": end_point[0], "line_ey": end_point[1]})
                
                # CIRCLE
                elif str(segment_name).startswith("circle"):
                    center = segment["Center"]
                    cmds.append("CIRCLE")
                    args.append({"circle_cx": center[0], "circle_cy": center[1], "circle_r": segment["Radius"]})
        return cmds, args

    def extrusion_to_cmd(self, extrusion: dict) -> tuple[list[str], list[dict]]:
        operation = extrusion["operation"]
        if operation not in self.EXTRUSION_OPERATORS:
            raise ValueError(f"Unsupported extrusion operation: {operation}")
        if operation == "NewBodyFeatureOperation":
            cmd = "EXTRUDE_NEW"
            args = {"extrude_new_dtn": extrusion["extrude_depth_towards_normal"], 
                    "extrude_new_don": extrusion["extrude_depth_opposite_normal"], 
                    "extrude_new_scale": extrusion["sketch_scale"]}
            
        elif operation == "JoinFeatureOperation":
            cmd = "EXTRUDE_JOIN"
            args = {"extrude_join_dtn": extrusion["extrude_depth_towards_normal"], 
                    "extrude_join_don": extrusion["extrude_depth_opposite_normal"], 
                    "extrude_join_scale": extrusion["sketch_scale"]}
            
        elif operation == "CutFeatureOperation":
            cmd = "EXTRUDE_CUT"
            args = {"extrude_cut_dtn": extrusion["extrude_depth_towards_normal"], 
                    "extrude_cut_don": extrusion["extrude_depth_opposite_normal"], 
                    "extrude_cut_scale": extrusion["sketch_scale"]}
        else:
            cmd = "EXTRUDE_INTERSECT"
            args = {"extrude_intersect_dtn": extrusion["extrude_depth_towards_normal"], 
                    "extrude_intersect_don": extrusion["extrude_depth_opposite_normal"], 
                    "extrude_intersect_scale": extrusion["sketch_scale"]}
            
        return [cmd], [args]
    
    def from_deepcad(self, json_object: dict, uid: str):
        # TODO : Implement this for DeepCAD data format
        pass
    
    def from_cadfusion(self, cadfusion_string: str, uid: str):
        # TODO : Implement this for CADFusion data format
        pass

    def __str__(self) -> str:
        rows = [("cmd", "args"), *[(str(command), str(arg)) for command, arg in zip(self.cmds, self.args)]]
        col1_w = max(len(row[0]) for row in rows)
        col2_w = max(len(row[1]) for row in rows)
        lines = [f"{rows[0][0].ljust(col1_w)} | {rows[0][1].ljust(col2_w)}", f"{'-' * col1_w}-+-{'-' * col2_w}"]
        lines.extend(f"{command.ljust(col1_w)} | {arg.ljust(col2_w)}" for command, arg in rows[1:])
        return "\n".join(lines)

    def render_stl(self, img_path: str) -> None:
        from utils.refs.CADSeqProc.json2stl_skt3d import process_one

        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as tmp_file:
            json.dump(self.json_object, tmp_file)
            tmp_path = tmp_file.name
        try:
            process_one(tmp_path, {"output_dir": img_path})
        except Exception as error:
            print(f"Error occurred while rendering STL: {error}")

    @property
    def part_count(self) -> int:
        return len(self.json_object["parts"])