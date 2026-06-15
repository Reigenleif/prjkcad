import json
import tempfile
from tqdm import tqdm
import pandas as pd


from .schema import get_dualseq_schema
from .geom_utils import *

class DualSeq:
    """
    Dual Sequence class for representing CAD 3D modelling.

    Args:
        json_object (dict): Original JSON containing CAD data.
        uid (str): Unique identifier for the CAD model.
        format (str): Input JSON format.
        descriptions (dict[str, str] | None):
            Descriptions at different expertise levels.
            Keys: "abstract", "beginner", "intermediate", "expert".

    Supported formats:
        text2cad: Text2CAD description + Text2CAD JSON format
        text2caddeepcad: Text2CAD description + DeepCAD JSON format
    """
        
    EXTRUSION_OPERATORS = {"NewBodyFeatureOperation", 
                           "JoinFeatureOperation", 
                           "CutFeatureOperation", 
                           "IntersectFeatureOperation"}
    
    EXTEND_TYPES = {"AlongDistance", "AgainstDistance"}
    

    def __init__(self, 
                 json_object: dict, 
                 uid: str,
                 format: str = "text2cad",
                 descriptions: dict[str, str] | None = None):
       
        
        # Default input format : text2cad
        if format == "text2cad":
            self.init_text2cad(json_object, uid, descriptions)
            
        elif format == "text2caddeepcad":
            self.init_text2caddeepcad(json_object, uid, descriptions)
            
        else:
            raise ValueError(f"Unsupported format: {format}")

    def init_text2cad(self, 
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

                elif str(segment_name).startswith("arc"):
                    start_point = segment["Start Point"]
                    mid_point = segment["Mid Point"]
                    end_point = segment["End Point"]
                    cmds.append("ARC")
                    args.append({"arc_sx": start_point[0], "arc_sy": start_point[1], 
                                "arc_mx": mid_point[0], "arc_my": mid_point[1],
                                "arc_ex": end_point[0], "arc_ey": end_point[1]})
                else:
                    raise ValueError(f"Unsupported segment type: {segment_name}")
                
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
    
    def init_text2caddeepcad(self, 
                             json_object: dict, 
                             uid: str,
                             descriptions: dict[str, str] | None):
        
        self.uid = uid
        self.descriptions = descriptions or {}
        
        entities = json_object["entities"]
        sequences = json_object["sequence"] # Unordered list of sequence groups
        sequence_list = [None] * len(sequences)
        
        cmds: list[str] = []
        args: list[dict] = []
        for seq in sequences:
            if seq["index"] >= len(sequences):
                raise ValueError(f"Sequence index {seq['index']} is out of bounds for sequence length {len(sequences)}")
            sequence_list[seq["index"]] = seq
        
        for seq in sequence_list:
            if seq["type"] != "ExtrudeFeature" :
                continue
            
            new_cmds, new_args = self.process_extrusion(json_object, seq)
            
            cmds.extend(new_cmds)
            args.extend(new_args)
        
        pass
    
    def process_extrusion(self, json_object: dict, seq: dict) -> list[tuple[str, dict]]:
        cmds = []
        args = []

        extrude_id = seq["entity"]
        extrude_entity = json_object["entities"][extrude_id]
        
        # SKETCHES
        sketches = extrude_entity["profiles"] # the key is profiles
        
        for sketch in sketches:
            sketch_id = sketch["sketch"]
            assert sketch_id in json_object["entities"], f"Sketch ID {sketch_id} from extrusion {extrude_id} not found in entities"
            
            sketch_entity = json_object["entities"][sketch_id]    
            
            # add COOR
            eua = R_to_euler(sketch_entity["transform"])
            coor_args = {"coor_tx": sketch_entity["transform"]["origin"]["x"],
                            "coor_ty": sketch_entity["transform"]["origin"]["y"],
                            "coor_tz": sketch_entity["transform"]["origin"]["z"],
                            "coor_euax": eua["euax"],
                            "coor_euay": eua["euay"],
                            "coor_euaz": eua["euaz"]}
            cmds.append("COOR")
            args.append(coor_args)
            
            # add FACE
            face_keys = sketch_entity["profiles"] # the key is profiles
            for face_key in face_keys:
                face = sketch_entity["profiles"][face_key]
                
                cmds.append("FACE")
                args.append({})
                
                loops = face["loops"]
                for loop in loops:
                    cmds.append("LOOP")
                    args.append({})
                    
                    curves = loop["profile_curves"]
                    for curve in curves:
                        # LINE
                        if curve["type"] == "Line3D":
                            line_2d_args = line_3d_to_line_2d(curve, 
                                                              coor_args)
                            
                            cmds.append("LINE")
                            args.append(line_2d_args)
                            
                        # CIRCLE
                        elif curve["type"] == "Circle3D":
                            circle_2d_args = circle_3d_to_circle_2d(curve, 
                                                                    coor_args)
                            cmds.append("CIRCLE")
                            args.append(circle_2d_args)
                            
                        # ARC
                        elif curve["type"] == "Arc3D":
                            arc_2d_args = arc_3d_to_arc_2d(curve, 
                                                          coor_args)
                            cmds.append("ARC")
                            args.append(arc_2d_args)
                            
                        else:
                            raise ValueError(f"Unsupported curve type: {curve['type']} in sketch {sketch_id} in extrusion {extrude_id}}}")
            
            # EXTRUSION definition
            operation = extrude_entity["operation"]
            extend_one = extrude_entity["extend_one"]
            extend_one_type = extend_one["type"]
            extend_one_value = extend_one["value"]
            extend_two = extrude_entity["extend_two"] or None
            extend_two_type = extend_two["type"] if extend_two else None
            extend_two_value = extend_two["value"] if extend_two else None

            assert not (extend_one_type == extend_two_type), f"Both extend_one and extend_two cannot have the same type in extrusion {extrude_id}"
            assert extend_one_type in self.EXTEND_TYPES, f"Unsupported extend_one type: {extend_one_type} in extrusion {extrude_id}"
            if extend_two_type:
                assert extend_two_type in self.EXTEND_TYPES, f"Unsupported extend_two type: {extend_two_type} in extrusion {extrude_id}"
            
            extrusion_args = {"extrude_dtn": None,
                              "extrude_don": None,
                              "extrude_scale": 1.0}
            
            if extend_one["type"] == "AlongDistance":
                extrusion_args["extrude_dtn"] = extend_one_value
            else:
                extrusion_args["extrude_don"] = extend_one_value
                
            if extend_two_type:
                if extend_two_type == "AlongDistance":
                    extrusion_args["extrude_dtn"] = extend_two_value
                else:
                    extrusion_args["extrude_don"] = extend_two_value
            if operation == "NewBodyFeatureOperation":
                cmds.append("EXTRUDE_NEW")
                args.append(extrusion_args)
                
            elif operation == "JoinFeatureOperation":
                cmds.append("EXTRUDE_JOIN")
                args.append(extrusion_args)
                
            elif operation == "CutFeatureOperation":
                cmds.append("EXTRUDE_CUT")
                args.append(extrusion_args)
                
            elif operation == "IntersectFeatureOperation":
                cmds.append("EXTRUDE_INTERSECT")
                args.append(extrusion_args)
            else:
                raise ValueError(f"Unsupported extrusion operation: {operation} in extrusion {extrude_id}")
            
        return cmds, args
        
            
        
        
    @staticmethod
    def from_cadfusion(cadfusion_string: str, uid: str):
        # TODO : Implement this for CADFusion data format
        pass
    
    
    @staticmethod
    def from_text2cad_df(df:pd.DataFrame, max_len: int = None) :
        dual_seqs = []
        success_count = 0
        for i, row in tqdm(df.iterrows(), total=len(df)):
            try :
                json_object = row["json_target"]
                # Skip this row if any description field is empty or missing
                desc_keys = ["abstract", "beginner", "intermediate", "expert"]
                try:
                    desc_vals = {k: row[k] for k in desc_keys}
                except Exception:
                    continue
                if any(pd.isna(v) or str(v).strip() == "" for v in desc_vals.values()):
                    continue
                descriptions = desc_vals
                dual_seq = DualSeq(json_object,
                                format="text2cad",
                                uid=row["uid"],
                                descriptions=descriptions)
                if max_len is not None and len(dual_seq) > max_len:
                    continue
                success_count += 1
                dual_seqs.append(dual_seq)
                
            except Exception as e:
                print(f"Error at index {i}: {e}")

        if success_count < len(df):
            print(f"Successfully created {success_count}/{len(df)} DualSeq instances.")
        return dual_seqs
    
    @staticmethod
    def from_text2caddeepcad_df(df: pd.DataFrame, max_len: int = None):
        dual_seqs = []
        success_count = 0
        for i, row in tqdm(df.iterrows(), total=len(df)):
            try :
                json_object = row["json_target"]
                # Skip this row if any description field is empty or missing
                desc_keys = ["abstract", "beginner", "intermediate", "expert"]
                try:
                    desc_vals = {k: row[k] for k in desc_keys}
                except Exception:
                    continue
                if any(pd.isna(v) or str(v).strip() == "" for v in desc_vals.values()):
                    continue
                descriptions = desc_vals
                dual_seq = DualSeq(json_object,
                                   format="text2caddeepcad",
                                uid=row["uid"],
                                descriptions=descriptions)
                if max_len is not None and len(dual_seq) > max_len:
                    continue
                success_count += 1
                dual_seqs.append(dual_seq)
                
            except Exception as e:
                print(f"Error at index {i}: {e}")

        if success_count < len(df):
            print(f"Successfully created {success_count}/{len(df)} DualSeq instances.")
        return dual_seqs
        

    def __str__(self) -> str:
        rows = [("cmd", "args"), *[(str(command), str(arg)) for command, arg in zip(self.cmds, self.args)]]
        col1_w = max(len(row[0]) for row in rows)
        col2_w = max(len(row[1]) for row in rows)
        lines = [f"{rows[0][0].ljust(col1_w)} | {rows[0][1].ljust(col2_w)}", f"{'-' * col1_w}-+-{'-' * col2_w}"]
        lines.extend(f"{command.ljust(col1_w)} | {arg.ljust(col2_w)}" for command, arg in rows[1:])
        return "\n".join(lines)
    
    def __len__(self) -> int:
        return len(self.cmds)

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
    
    @staticmethod
    def id_to_cmds(id_seq: list[int]) -> list[str]:
        id_to_command = get_dualseq_schema()["id_to_command"]
        return [id_to_command.get(id, "<UNK>") for id in id_seq]