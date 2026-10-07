import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import pickle
import random
import struct
import uuid
from typing import Dict, Any, Optional, Tuple, List

try:
    from OCC.Core.StlAPI import StlAPI_Writer
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from utils.render import render_dual_seq_to_shape, render_to_image, render_model_comparison_grid
except (ImportError, Exception):
    StlAPI_Writer = None
    BRepMesh_IncrementalMesh = None
    render_dual_seq_to_shape = None
    render_to_image = None
    render_model_comparison_grid = None

from utils.representations.dual_seq.dual_seq import DualSeq

MODEL_FOLDERS = {
    "Baseline": "baseline_text2cad",
    "M1": "m1",
    "M2": "m2",
    "M3": "m3",
    "M4": "grpo_fuse",
}

DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "out/gui_renders")
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

CURATED_PATHS = [
    os.path.join(PROJECT_ROOT, "gui/assets/curated_recommendations.json"),
    os.path.join(PROJECT_ROOT, "assets/curated_recommendations.json"),
    os.path.join(PROJECT_ROOT, "gui/curated_recommendations.json"),
]
UID_PROMPT_MAP_PATHS = [
    os.path.join(PROJECT_ROOT, "gui/assets/uid_prompt_map.json"),
    os.path.join(PROJECT_ROOT, "assets/uid_prompt_map.json"),
    os.path.join(PROJECT_ROOT, "gui/uid_prompt_map.json"),
]

_PRED_CACHE: Dict[str, Dict[str, Any]] = {}
_CURATED_LIST: List[Dict[str, Any]] = []
_CURATED_MAP: Dict[str, Dict[str, Any]] = {}
_UID_MAP: Dict[str, Dict[str, Any]] = {}

def export_shape_to_stl(shape, stl_path: str, deflection: float = 0.05) -> bool:
    if shape is None or shape.IsNull():
        return False
    try:
        mesh = BRepMesh_IncrementalMesh(shape, deflection)
        mesh.Perform()
        writer = StlAPI_Writer()
        writer.SetASCIIMode(False)
        writer.Write(shape, stl_path)
        if not (os.path.isfile(stl_path) and os.path.getsize(stl_path) > 84):
            if os.path.isfile(stl_path):
                os.remove(stl_path)
            return False
        sz = os.path.getsize(stl_path)
        with open(stl_path, "rb") as f:
            header = f.read(84)
        if len(header) == 84:
            n_faces = struct.unpack("<I", header[80:84])[0]
            if n_faces > 0 and 84 + n_faces * 50 == sz:
                return True
        return sz > 84
    except Exception as e:
        print(f"STL export error for {stl_path}: {e}")
        if os.path.isfile(stl_path):
            try:
                os.remove(stl_path)
            except Exception:
                pass
        return False

def is_valid_binary_stl(stl_path: str) -> bool:
    if not (os.path.isfile(stl_path) and os.path.getsize(stl_path) > 84):
        return False
    sz = os.path.getsize(stl_path)
    try:
        with open(stl_path, "rb") as f:
            header = f.read(84)
        if len(header) == 84:
            n_faces = struct.unpack("<I", header[80:84])[0]
            if n_faces > 0 and (84 + n_faces * 50 == sz):
                return True
    except Exception:
        return False
    return False

def load_curated() -> List[Dict[str, Any]]:
    global _CURATED_LIST, _CURATED_MAP
    if not _CURATED_LIST:
        for p in CURATED_PATHS:
            if os.path.isfile(p):
                with open(p, "r") as f:
                    _CURATED_LIST = json.load(f)
                    _CURATED_MAP = {item["uid"]: item for item in _CURATED_LIST}
                break
    return _CURATED_LIST

def load_uid_map() -> Dict[str, Dict[str, Any]]:
    global _UID_MAP
    if not _UID_MAP:
        for p in UID_PROMPT_MAP_PATHS:
            if os.path.isfile(p):
                with open(p, "r") as f:
                    _UID_MAP = json.load(f)
                break
    return _UID_MAP

def get_prediction_dict(model_name: str, split: str = "test") -> Dict[str, Any]:
    folder = MODEL_FOLDERS.get(model_name, model_name)
    cache_key = f"{folder}_{split}"
    if cache_key not in _PRED_CACHE:
        candidates = [
            os.path.join(PROJECT_ROOT, f"gui/assets/{folder}/{split}_predictions.pkl"),
            os.path.join(PROJECT_ROOT, f"assets/{folder}/{split}_predictions.pkl"),
            os.path.join(PROJECT_ROOT, f"out/merged/{folder}/{split}_predictions.pkl"),
        ]
        found_path = None
        for cand in candidates:
            if os.path.isfile(cand):
                found_path = cand
                break
        if not found_path:
            raise FileNotFoundError(f"Prediction file not found for {model_name} ({split})")
        with open(found_path, "rb") as f:
            _PRED_CACHE[cache_key] = pickle.load(f)
    return _PRED_CACHE[cache_key]

def get_sample_metadata(uid: str) -> Dict[str, Any]:
    clean_uid = uid.strip()
    load_curated()
    if clean_uid in _CURATED_MAP:
        return _CURATED_MAP[clean_uid]

    uid_map = load_uid_map()
    if clean_uid in uid_map:
        entry = uid_map[clean_uid]
        return {
            "uid": clean_uid,
            "split": entry.get("split", "test"),
            "prompt": entry.get("prompt", ""),
            "category": "Custom / Search",
            "note": "",
            "gt_cmds": [],
            "gt_args": [],
            "cd_scores": {},
        }

    return {
        "uid": clean_uid,
        "split": "test",
        "prompt": "",
        "category": "Unknown",
        "note": "",
        "gt_cmds": [],
        "gt_args": [],
        "cd_scores": {},
    }

def truncate_description(text: str, max_chars: int = 180) -> str:
    if not text:
        return ""
    clean = " ".join(text.split())
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rstrip() + "..."

def unify_model_prediction(pred_out, uid: str = "") -> Tuple[Optional[DualSeq], Any]:
    if pred_out is None:
        return None, None
    if isinstance(pred_out, DualSeq):
        return pred_out, None
    if hasattr(pred_out, "_json") or type(pred_out).__name__ == "CADSequence":
        try:
            json_data = pred_out._json() if hasattr(pred_out, "_json") else {}
            if json_data and "parts" in json_data and json_data["parts"]:
                ds_obj = DualSeq(json_object=json_data, uid=uid)
                if ds_obj.cmds:
                    return ds_obj, pred_out
        except Exception:
            pass
        return None, pred_out
    if isinstance(pred_out, dict):
        cmds = pred_out.get("cmds", [])
        args = pred_out.get("args", pred_out.get("args_dict", []))
        return DualSeq(cmds=cmds, args=args, uid=uid), None
    return None, None

def get_random_sample() -> Tuple[str, str, str]:
    uid_map = load_uid_map()
    if uid_map:
        random_uid = random.choice(list(uid_map.keys()))
        meta = get_sample_metadata(random_uid)
        return random_uid, meta.get("split", "test"), truncate_description(meta.get("prompt", ""))
    load_curated()
    if _CURATED_LIST:
        sample = random.choice(_CURATED_LIST)
        return sample["uid"], sample.get("split", "test"), truncate_description(sample.get("prompt", ""))
    return "0009/00097460", "test", ""

def get_ground_truth_for_uid(uid: str, split: str = "test") -> Tuple[Optional[str], Optional[str], str]:
    meta = get_sample_metadata(uid)
    gt_cmds = meta.get("gt_cmds") or []
    gt_args = meta.get("gt_args") or []

    # Dynamic fallback: load from Text2CAD minimal_json if missing in metadata
    if not gt_cmds:
        parts = uid.split("/")
        if len(parts) == 2:
            prefix, sid = parts[0], parts[1]
        else:
            sid = uid
            prefix = sid[:4]
        json_path = os.path.join(PROJECT_ROOT, f"data/text2cad/{prefix}/{sid}/minimal_json/{sid}.json")
        if os.path.isfile(json_path):
            try:
                with open(json_path, "r") as f:
                    d = json.load(f)
                ds_loaded = DualSeq(json_object=d, uid=uid)
                gt_cmds = ds_loaded.cmds
                gt_args = ds_loaded.args_dict
            except Exception as e:
                print(f"Error loading GT json for {uid}: {e}")

    if not gt_cmds:
        return None, None, "Ground Truth commands not available for this sample."

    ds_gt = DualSeq(cmds=gt_cmds, args=gt_args)
    gt_seq_str = str(ds_gt)

    uid_safe = uid.replace("/", "_")
    stl_file = os.path.join(DEFAULT_OUTPUT_DIR, f"gt_{uid_safe}.stl")
    png_file = os.path.join(DEFAULT_OUTPUT_DIR, f"gt_{uid_safe}.png")

    if (
        is_valid_binary_stl(stl_file)
        and os.path.isfile(png_file)
        and os.path.getsize(png_file) > 0
    ):
        return stl_file, png_file, gt_seq_str

    gt_stl_path = None
    gt_png_path = None

    try:
        gt_shape = render_dual_seq_to_shape(ds_gt)
        if gt_shape is not None and not gt_shape.IsNull():
            if export_shape_to_stl(gt_shape, stl_file):
                gt_stl_path = stl_file

            render_to_image(gt_shape, png_file)
            if os.path.isfile(png_file) and os.path.getsize(png_file) > 0:
                gt_png_path = png_file
    except Exception as e:
        print(f"Ground truth rendering error for {uid}: {e}")

    return gt_stl_path, gt_png_path, gt_seq_str

def render_single_model(
    model_name: str,
    uid: str
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], str, str, str, str]:
    clean_uid = uid.strip()
    if not clean_uid:
        return None, None, None, None, "", "", "Split: Unknown", ""

    meta = get_sample_metadata(clean_uid)
    split = meta.get("split", "test")
    prompt = truncate_description(meta.get("prompt", ""))
    split_badge = f"**Dataset Split:** 🏷️ `{split.upper()}`"

    try:
        preds = get_prediction_dict(model_name, split)
    except Exception as e:
        return None, None, None, None, "", "", split_badge, prompt

    if clean_uid not in preds:
        return None, None, None, None, "", "", split_badge, prompt

    raw_pred = preds[clean_uid]
    ds_pred, _ = unify_model_prediction(raw_pred, uid=clean_uid)
    pred_seq_str = str(ds_pred) if ds_pred is not None else (str(raw_pred) if raw_pred else "No sequence data")

    uid_safe = clean_uid.replace("/", "_")
    pred_stl_path = os.path.join(DEFAULT_OUTPUT_DIR, f"{model_name}_{uid_safe}.stl")
    pred_png_path = os.path.join(DEFAULT_OUTPUT_DIR, f"{model_name}_{uid_safe}.png")

    # Reuse if already rendered as valid binary STL and valid PNG
    if not (
        is_valid_binary_stl(pred_stl_path)
        and os.path.isfile(pred_png_path)
        and os.path.getsize(pred_png_path) > 0
    ):
        pred_shape = None
        try:
            if ds_pred is not None and getattr(ds_pred, "cmds", None):
                pred_shape = render_dual_seq_to_shape(ds_pred)
            elif raw_pred is not None:
                pred_shape = render_dual_seq_to_shape(raw_pred)
        except Exception as e:
            print(f"Single model shape synthesis error: {e}")

        if pred_shape is not None and not pred_shape.IsNull():
            if export_shape_to_stl(pred_shape, pred_stl_path):
                pass
            else:
                pred_stl_path = None

            try:
                render_to_image(pred_shape, pred_png_path)
                if not (os.path.isfile(pred_png_path) and os.path.getsize(pred_png_path) > 0):
                    pred_png_path = None
            except Exception as e:
                print(f"2D render error: {e}")
                pred_png_path = None
        else:
            pred_stl_path = None
            pred_png_path = None

    if pred_stl_path is None and pred_png_path is None:
        split_badge += f" | ⚠️ *Model `{model_name}` did not produce a closed 3D solid for UID `{clean_uid}`.*"

    gt_stl_path, gt_png_path, gt_seq_str = get_ground_truth_for_uid(clean_uid, split)

    return (
        pred_stl_path,
        gt_stl_path,
        pred_png_path,
        gt_png_path,
        pred_seq_str,
        gt_seq_str,
        split_badge,
        prompt,
    )


def find_precomputed_grid_image(uid: str, split: str) -> Optional[str]:
    uid_safe = uid.replace("/", "_")
    candidates = [
        os.path.join(PROJECT_ROOT, f"gui/assets/renders/{uid_safe}.png"),
        os.path.join(PROJECT_ROOT, f"scratch/renders/{uid_safe}.png"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c

    # Search in visualizations folders
    vis_dirs = [
        os.path.join(PROJECT_ROOT, f"gui/assets/second-test/{split}/visualizations/top50_least_cd"),
        os.path.join(PROJECT_ROOT, f"gui/assets/second-test/{split}/visualizations/top50_most_cd"),
        os.path.join(PROJECT_ROOT, f"out/merged/second-test/{split}/visualizations/top50_least_cd"),
        os.path.join(PROJECT_ROOT, f"out/merged/second-test/{split}/visualizations/top50_most_cd"),
    ]
    for vd in vis_dirs:
        if os.path.isdir(vd):
            for fn in os.listdir(vd):
                if uid_safe in fn and fn.endswith(".png"):
                    return os.path.join(vd, fn)
    return None

def render_comparison_grid(
    uid: str
) -> Tuple[Optional[str], str, str, str]:
    clean_uid = uid.strip()
    if not clean_uid:
        return None, "Split: Unknown", "", "Please enter or select a valid UID."

    meta = get_sample_metadata(clean_uid)
    split = meta.get("split", "test")
    raw_prompt = meta.get("prompt", "")
    prompt_text = truncate_description(raw_prompt)
    cd_scores = meta.get("cd_scores", {})
    gt_cmds = meta.get("gt_cmds", [])
    gt_args = meta.get("gt_args", [])
    split_badge = f"**Dataset Split:** 🏷️ `{split.upper()}`"

    cd_table_md = "| Model | Chamfer Distance (CD) |\n| :--- | :--- |\n"
    for m in ["Baseline", "M1", "M2", "M3", "M4-xx"]:
        val = cd_scores.get(m, "N/A")
        val_str = f"{val:.6f}" if isinstance(val, (int, float)) else str(val)
        display_m = "M4 (GRPO)" if m == "M4-xx" else m
        cd_table_md += f"| **{display_m}** | `{val_str}` |\n"

    # 1. Check precomputed fixed renders first
    precomputed = find_precomputed_grid_image(clean_uid, split)
    if precomputed is not None:
        return precomputed, split_badge, prompt_text, cd_table_md

    # 2. Dynamic generation using unified predictions and fixed renderer
    model_preds = []
    for model_name in ["Baseline", "M1", "M2", "M3", "M4"]:
        cd_key = model_name if model_name != "M4" else "M4-xx"
        cd_val = cd_scores.get(cd_key, None)
        try:
            preds = get_prediction_dict(model_name, split)
            raw_pred = preds.get(clean_uid, None)
            ds_obj, _ = unify_model_prediction(raw_pred, uid=clean_uid)
            p_cmds = ds_obj.cmds if ds_obj else []
            p_args = ds_obj.args_dict if ds_obj else []
        except Exception:
            p_cmds, p_args = [], []

        model_preds.append((model_name, p_cmds, p_args, cd_val))

    session_id = uuid.uuid4().hex[:8]
    grid_img_path = os.path.join(DEFAULT_OUTPUT_DIR, f"grid_{session_id}.png")

    try:
        render_model_comparison_grid(
            models_predictions=model_preds,
            gt_cmds=gt_cmds,
            gt_args=gt_args,
            output_path=grid_img_path,
            uid=clean_uid,
            prompt_text=prompt_text,
            cell_size=(260, 260),
        )
    except Exception as e:
        grid_img_path = None
        print(f"Comparison grid render failed: {e}")

    return grid_img_path, split_badge, prompt_text, cd_table_md
