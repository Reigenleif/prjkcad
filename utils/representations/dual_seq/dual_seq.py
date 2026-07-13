import json
import tempfile
from tqdm import tqdm
import pandas as pd
from typing import Optional

from .schema import get_dualseq_schema, DEFAULT_COMMANDS
from utils.representations.legacy_dual_seq.geom_utils import *

def encode_int_part(val_str: str, schema: dict) -> list[int]:
    if val_str == "0" or val_str == "":
        return [schema["arg_to_id"]["0"]]
    
    val_int = int(val_str)
    chunks = []
    while val_int > 0:
        chunks.append(val_int % 1000)
        val_int //= 1000
    if not chunks:
        chunks = [0]
    chunks = chunks[::-1]
    return [schema["arg_to_id"][str(c)] for c in chunks]

def encode_frac_part(val_str: str, schema: dict) -> list[int]:
    if not val_str:
        return []
    pad_len = (3 - len(val_str) % 3) % 3
    val_str += "0" * pad_len
    
    tokens = []
    for i in range(0, len(val_str), 3):
        chunk_int = int(val_str[i:i+3])
        tokens.append(schema["arg_to_id"][str(chunk_int)])
        
    while tokens and tokens[-1] == schema["arg_to_id"]["0"]:
        tokens.pop()
    
    return tokens

def float_to_tokens(val: float | int | None, schema: dict) -> list[int]:
    if val is None:
        val = 0.0
        
    tokens = []
    if val < 0:
        tokens.append(schema["arg_neg_id"])
        val = abs(val)
        
    val_str = f"{val:.6f}".rstrip('0').rstrip('.')
    if not val_str:
        val_str = "0"
        
    if "." in val_str:
        int_part, frac_part = val_str.split(".")
        tokens.extend(encode_int_part(int_part, schema))
        frac_tokens = encode_frac_part(frac_part, schema)
        if frac_tokens:
            tokens.append(schema["arg_dec_id"])
            tokens.extend(frac_tokens)
    else:
        tokens.extend(encode_int_part(val_str, schema))
        
    return tokens

def tokens_to_float(tokens: list[int], schema: dict) -> float:
    if not tokens:
        return 0.0
        
    is_neg = False
    if tokens[0] == schema["arg_neg_id"]:
        is_neg = True
        tokens = tokens[1:]
        
    if not tokens:
        return 0.0
        
    dec_id = schema["arg_dec_id"]
    try:
        dec_idx = tokens.index(dec_id)
        int_tokens = tokens[:dec_idx]
        frac_tokens = tokens[dec_idx+1:]
    except ValueError:
        int_tokens = tokens
        frac_tokens = []
        
    int_val = 0
    for t in int_tokens:
        val = int(schema["id_to_arg"][t])
        int_val = int_val * 1000 + val
        
    frac_val = 0.0
    divisor = 1000.0
    for t in frac_tokens:
        val = int(schema["id_to_arg"][t])
        frac_val += val / divisor
        divisor *= 1000.0
        
    total = int_val + frac_val
    if is_neg:
        total = -total
    return total

class DualSeq:
    EXTRUSION_OPERATORS = {"NewBodyFeatureOperation", 
                           "JoinFeatureOperation", 
                           "CutFeatureOperation", 
                           "IntersectFeatureOperation"}
    
    EXTEND_TYPES = {"AlongDistance", "AgainstDistance"}

    def __init__(self, 
                 json_object: dict | None = None, 
                 uid: str = "mock",
                 format: str = "text2cad",
                 descriptions: dict[str, str] | None = None,
                 cmds: list[str] | None = None,
                 args: list[dict] | None = None,
                 cmd_args_tuples: list[tuple[str, dict]] | None = None):
       
        self.schema = get_dualseq_schema()
        self.uid = uid
        self.json_object = json_object
        self.cmds: list[str] = []
        self.args: list[int] = [] 
        self.args_dict: list[dict] = [] # Legacy preservation for str output and debug
        self.extrusion_kinds: set[str] = set()
        self.descriptions = descriptions or {}
        
        if cmd_args_tuples is not None:
            self.cmds = [c for c, _ in cmd_args_tuples]
            self.args_dict = [a for _, a in cmd_args_tuples]
        elif cmds is not None and args is not None:
            self.cmds = cmds
            self.args_dict = args
        elif json_object is not None:
            if format == "text2cad":
                self.init_text2cad(json_object, uid, descriptions)
            elif format == "text2caddeepcad":
                self.init_text2caddeepcad(json_object, uid, descriptions)
            else:
                raise ValueError(f"Unsupported format: {format}")
        else:
            pass # Empty initialization
            
        self.build_args_tokens()

    def build_args_tokens(self):
        """Converts self.args_dict into a flat list of token IDs in self.args"""
        all_tokens = []
        for i, cmd in enumerate(self.cmds):
            cmd_arg_names = DEFAULT_COMMANDS.get(cmd, [])
            if not cmd_arg_names:
                continue
                
            arg_dict = self.args_dict[i]
            if not arg_dict:
                continue
            
            # Follow the schema order
            for j, arg_name in enumerate(cmd_arg_names):
                val = arg_dict.get(arg_name, 0.0)
                all_tokens.extend(float_to_tokens(val, self.schema))
                # Add SEP after every argument
                all_tokens.append(self.schema["arg_sep_id"])
                
        self.args = all_tokens

    def init_text2cad(self, 
                      json_object: dict, 
                      uid: str,
                      descriptions: dict[str, str] | None):
        for part_name, part in json_object["parts"].items():
            coordinate_system = part["coordinate_system"]
            euler_angles = coordinate_system["Euler Angles"]
            translation_vector = coordinate_system["Translation Vector"]
            self.cmds.append("COOR")
            self.args_dict.append({"coor_euax": euler_angles[0], 
                              "coor_euay": euler_angles[1], 
                              "coor_euaz": euler_angles[2], 
                              "coor_tx": translation_vector[0], 
                              "coor_ty": translation_vector[1], 
                              "coor_tz": translation_vector[2]})
        
            for face_name, face in part["sketch"].items():
                self.cmds.append("FACE")
                self.args_dict.append({})
                more_cmds, more_args = self.face_to_cmd(face)
                self.cmds.extend(more_cmds)
                self.args_dict.extend(more_args)
                
            extrusion = part["extrusion"]
            self.extrusion_kinds.add(extrusion["operation"])
            more_cmds, more_args = self.extrusion_to_cmd(extrusion)
            self.cmds.extend(more_cmds)
            self.args_dict.extend(more_args)

    def face_to_cmd(self, face: dict) -> tuple[list[str], list[dict]]:
        cmds: list[str] = []
        args: list[dict] = []
        
        for loop_name, loop in face.items():
            cmds.append("LOOP")
            args.append({})
            
            for segment_name, segment in loop.items():
                if str(segment_name).startswith("line"):
                    start_point = segment["Start Point"]
                    end_point = segment["End Point"]
                    cmds.append("LINE")
                    args.append({"line_sx": start_point[0], "line_sy": start_point[1], "line_ex": end_point[0], "line_ey": end_point[1]})
                
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
        
        entities = json_object["entities"]
        sequences = json_object["sequence"]
        sequence_list = [None] * len(sequences)
        
        for seq in sequences:
            if seq["index"] >= len(sequences):
                raise ValueError(f"Sequence index {seq['index']} is out of bounds for sequence length {len(sequences)}")
            sequence_list[seq["index"]] = seq
        
        for seq in sequence_list:
            if seq["type"] != "ExtrudeFeature" :
                continue
            
            new_cmds, new_args = self.process_extrusion(json_object, seq)
            
            self.cmds.extend(new_cmds)
            self.args_dict.extend(new_args)
    
    def process_extrusion(self, json_object: dict, seq: dict) -> tuple[list[str], list[dict]]:
        cmds = []
        args = []

        extrude_id = seq["entity"]
        extrude_entity = json_object["entities"][extrude_id]
        sketches = extrude_entity["profiles"]
        
        for sketch in sketches:
            sketch_id = sketch["sketch"]
            assert sketch_id in json_object["entities"], f"Sketch ID {sketch_id} from extrusion {extrude_id} not found in entities"
            
            sketch_entity = json_object["entities"][sketch_id]    
            
            eua = R_to_euler(sketch_entity["transform"])
            coor_args = {"coor_tx": sketch_entity["transform"]["origin"]["x"],
                            "coor_ty": sketch_entity["transform"]["origin"]["y"],
                            "coor_tz": sketch_entity["transform"]["origin"]["z"],
                            "coor_euax": eua["euax"],
                            "coor_euay": eua["euay"],
                            "coor_euaz": eua["euaz"]}
            cmds.append("COOR")
            args.append(coor_args)
            
            face_keys = sketch_entity["profiles"]
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
                        if curve["type"] == "Line3D":
                            line_2d_args = line_3d_to_line_2d(curve, coor_args)
                            cmds.append("LINE")
                            args.append(line_2d_args)
                            
                        elif curve["type"] == "Circle3D":
                            circle_2d_args = circle_3d_to_circle_2d(curve, coor_args)
                            cmds.append("CIRCLE")
                            args.append(circle_2d_args)
                            
                        elif curve["type"] == "Arc3D":
                            arc_2d_args = arc_3d_to_arc_2d(curve, coor_args)
                            cmds.append("ARC")
                            args.append(arc_2d_args)
                            
                        else:
                            raise ValueError(f"Unsupported curve type: {curve['type']} in sketch {sketch_id} in extrusion {extrude_id}}}")
            
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
    def from_text2cad_df(df: pd.DataFrame, max_len: int = None):
        dual_seqs = []
        success_count = 0
        for i, row in tqdm(df.iterrows(), total=len(df)):
            try :
                json_object = row["json_target"]
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
        # Display legacy args dict alongside cmds
        rows = [("cmd", "args"), *[(str(command), str(arg)) for command, arg in zip(self.cmds, self.args_dict)]]
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

    @classmethod
    def from_sequences(cls, cmds: list[str], args_tokens: list[int], uid: str = "") -> "DualSeq":
        """Create a DualSeq object from raw predicted command strings and arg tokens."""
        schema = get_dualseq_schema()
        sep_id = schema["arg_sep_id"]
        
        # Split tokens by SEP
        arg_groups = []
        current_group = []
        for t in args_tokens:
            if t == sep_id:
                arg_groups.append(current_group)
                current_group = []
            else:
                current_group.append(t)
        if current_group:
            arg_groups.append(current_group)
            
        decoded_args_dict = []
        arg_group_idx = 0
        
        for cmd in cmds:
            arg_dict = {}
            if cmd in DEFAULT_COMMANDS:
                for arg_name in DEFAULT_COMMANDS[cmd]:
                    if arg_group_idx < len(arg_groups):
                        val = tokens_to_float(arg_groups[arg_group_idx], schema)
                        arg_dict[arg_name] = val
                        arg_group_idx += 1
                    else:
                        arg_dict[arg_name] = 0.0
            decoded_args_dict.append(arg_dict)
            
        instance = cls.__new__(cls)
        instance.schema = schema
        instance.uid = uid
        instance.cmds = list(cmds)
        instance.args = list(args_tokens)
        instance.args_dict = decoded_args_dict
        instance.descriptions = {}
        instance.json_object = {"parts": {}} # Mocked for visualization
        return instance
