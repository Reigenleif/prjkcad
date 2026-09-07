import os
import pickle
from typing import Any, Dict, List, Optional
import torch
from tqdm import tqdm
from utils.dual_seq import DualSeq
from utils.representations.converter import dualseq_to_cadseq
from utils.representations.CadSeqProc.cad_sequence import CADSequence

def save_predictions_pkl(preds: Dict[str, Any], save_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(preds, f)
    print(f"Saved {len(preds)} prediction cache to: {save_path}")

def load_predictions_pkl(save_path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(save_path):
        return None
    try:
        with open(save_path, "rb") as f:
            data = pickle.load(f)
        print(f"Loaded {len(data)} cached predictions from: {save_path}")
        return data
    except Exception as e:
        print(f"Failed to load cache from {save_path}: {e}")
        return None

def predict_and_cache_validation(
    wrapper: Any,
    val_samples: List[Any],
    cache_path: str,
    force_recompute: bool = False,
    batch_size: int = 32,
    max_new_tokens: int = 50,
) -> Dict[str, Any]:
    if not force_recompute:
        cached = load_predictions_pkl(cache_path)
        if cached is not None:
            return cached

    preds = {}
    uids = [getattr(sample, "uid", str(i)) for i, sample in enumerate(val_samples)]
    texts = [sample.descriptions.get("expert", "") if hasattr(sample, "descriptions") else str(sample) for sample in val_samples]

    if hasattr(wrapper, "infer_batch") and batch_size > 1:
        pbar = tqdm(range(0, len(texts), batch_size), desc=f"Predicting & caching ({os.path.basename(cache_path)})")
        for i in pbar:
            chunk_texts = texts[i : i + batch_size]
            chunk_uids = uids[i : i + batch_size]
            with torch.no_grad():
                chunk_preds = wrapper.infer_batch(chunk_texts, batch_size=len(chunk_texts), max_new_tokens=max_new_tokens)
            for uid, pred_out in zip(chunk_uids, chunk_preds):
                preds[uid] = pred_out
    else:
        pbar = tqdm(val_samples, desc=f"Predicting & caching ({os.path.basename(cache_path)})")
        for sample in pbar:
            uid = getattr(sample, "uid", str(len(preds)))
            text = sample.descriptions.get("expert", "") if hasattr(sample, "descriptions") else str(sample)
            with torch.no_grad():
                pred_out = wrapper.infer(text, max_new_tokens=max_new_tokens)
            preds[uid] = pred_out

    save_predictions_pkl(preds, cache_path)
    return preds
