import os
import shutil
from fastapi import APIRouter
from gui.backend.cad_engine import parse_solidworks_part_tree, DEFAULT_STL_DIR
from gui.model_runner import (
    load_curated,
    get_sample_metadata,
    get_ground_truth_for_uid,
)

router = APIRouter(prefix="/api/samples", tags=["samples"])


@router.get("/curated")
def get_curated_samples():
    curated = load_curated()
    return {"samples": curated}


@router.get("/{uid:path}")
def get_sample_details(uid: str):
    clean_uid = uid.strip()
    meta = get_sample_metadata(clean_uid)
    stl_path, png_path, seq_str = get_ground_truth_for_uid(clean_uid, meta.get("split", "test"))

    gt_cmds = meta.get("gt_cmds") or []
    gt_args = meta.get("gt_args") or []
    tuples = list(zip(gt_cmds, gt_args)) if gt_cmds else []

    stl_url = None
    if stl_path and os.path.isfile(stl_path):
        stl_filename = os.path.basename(stl_path)
        dest_path = os.path.join(DEFAULT_STL_DIR, stl_filename)
        if not os.path.isfile(dest_path):
            try:
                shutil.copyfile(stl_path, dest_path)
            except Exception:
                pass
        stl_url = f"/api/renders/{stl_filename}"

    tree = parse_solidworks_part_tree(gt_cmds, gt_args) if gt_cmds else {"tree": [], "part_count": 0}

    return {
        "uid": clean_uid,
        "metadata": meta,
        "dualseq_tuples": tuples,
        "sequence_string": seq_str,
        "stl_url": stl_url,
        "tree_data": tree,
    }
