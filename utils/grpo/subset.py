from __future__ import annotations

import os
import json
import hashlib
from typing import List, Tuple, Any, Optional, TYPE_CHECKING
import numpy as np
import torch
from tqdm import tqdm

from utils.dual_seq import DualSeq
if TYPE_CHECKING:
    from utils.pipeline.grpo_pipeline import GRPOPipeline
from utils.evaluate.t2c.t2c_report import eval_t2c_report
from utils.evaluate.batch_evaluation import dualseq_to_cadseq
from utils.evaluate.eval.cmds import eval_cmd_only
from utils.data_utils import create_dualseq_data_loader


def get_canonical_cache_key(ds: DualSeq, desc_level: str) -> str:
    """Generates a canonical cache key for a DualSeq instance."""
    text = ds.descriptions.get(desc_level, "")
    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
    uid = getattr(ds, "uid", "sample")
    return f"{uid}_{text_hash}"


def subset_grpo_dataset(
    pipeline: Any,
    use_cache: bool = True,
    cache_path: Optional[str] = None
) -> Any:
    """
    Evaluates all training instances using teacher forcing,
    ranks them by teacher forcing F1 score ascending, picks subset_top_p * n_train samples
    with the least F1 score, reinserts them into the pipeline, and updates loaders.
    """
    if pipeline.dual_seqs is None:
        pipeline.load_dataset()
    if pipeline.wrapper is None or pipeline.model is None:
        pipeline.load_tokenizer()
        pipeline.load_model_and_wrapper()

    desc_level = getattr(pipeline.cfg.data, "description_level", "expert")
    train_seqs: List[DualSeq] = pipeline.dual_seqs
    n_train = len(train_seqs)

    subset_top_p = getattr(getattr(pipeline.cfg, "grpo", None), "subset_top_p", 1.0)
    if subset_top_p is None:
        subset_top_p = 1.0

    if cache_path is None:
        save_root = getattr(pipeline, "SAVE_ROOT", "out")
        os.makedirs(save_root, exist_ok=True)
        cache_path = os.path.join(save_root, "subset_tf_cache.json")

    cache_data: dict[str, float] = {}
    if use_cache and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
        except Exception:
            cache_data = {}

    batch_size = getattr(getattr(pipeline.cfg, "data", None), "batch_size", 32)
    loader = create_dualseq_data_loader(
        train_seqs,
        pipeline.text_tokenizer,
        description_level=desc_level,
        batch_size=batch_size,
        num_workers=getattr(getattr(pipeline.cfg, "data", None), "num_workers", 0),
        val_ratio=0.0,
        shuffle=False,
        out_type="EightBitBinarizedArgs",
        metadata=getattr(pipeline, "metadata", None)
    )

    device = next(pipeline.model.parameters()).device
    pipeline.wrapper.eval()

    n_cached = 0
    n_eval = 0
    scored_samples: List[Tuple[float, DualSeq]] = []

    underlying_wrapper = getattr(pipeline.wrapper, "wrapper", pipeline.wrapper)

    pbar = tqdm(loader, desc="Evaluating train set via teacher forcing")
    sample_idx = 0

    with torch.no_grad():
        for batch in pbar:
            batch_seqs = train_seqs[sample_idx: sample_idx + batch_size]
            sample_idx += len(batch_seqs)

            cached_batch_scores = []
            if use_cache:
                for ds in batch_seqs:
                    key = get_canonical_cache_key(ds, desc_level)
                    if key in cache_data:
                        cached_batch_scores.append((float(cache_data[key]), ds))
                    else:
                        break

            if len(cached_batch_scores) == len(batch_seqs):
                n_cached += len(batch_seqs)
                scored_samples.extend(cached_batch_scores)
                continue

            input_ids, attention_mask, cmd_targets, arg_targets = pipeline.wrapper.extract_inputs(batch)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            if cmd_targets is not None:
                cmd_targets = cmd_targets.to(device)
            if arg_targets is not None:
                arg_targets = arg_targets.to(device)

            batch_inputs = (input_ids, cmd_targets, arg_targets, attention_mask)
            out_dict = pipeline.wrapper(batch_inputs, is_teacher_forcing=True)

            cmd_preds = out_dict.get("cmd_preds")
            arg_preds = out_dict.get("arg_preds")

            b_size = input_ids.size(0)
            for b in range(b_size):
                ds = batch_seqs[b]
                key = get_canonical_cache_key(ds, desc_level)

                if use_cache and key in cache_data:
                    f1_score = float(cache_data[key])
                    n_cached += 1
                else:
                    c_toks = cmd_preds[b].cpu().numpy().tolist() if cmd_preds is not None else []
                    a_bins = arg_preds[b].cpu().numpy().tolist() if arg_preds is not None else []

                    if hasattr(underlying_wrapper, "_tokens_to_dualseq"):
                        pred_ds = underlying_wrapper._tokens_to_dualseq(c_toks, a_bins)
                    else:
                        pred_ds = DualSeq(cmds=[], args=[])

                    gt_cad = dualseq_to_cadseq(ds)
                    pred_cad = dualseq_to_cadseq(pred_ds)

                    f1_score = 0.0
                    if gt_cad is not None and pred_cad is not None:
                        t2c_m = eval_t2c_report(gt_cad, pred_cad, uid=ds.uid)
                        f1_score = float(t2c_m.get("t2c/macro_f1", 0.0))
                    else:
                        pred_cmds = getattr(pred_ds, "cmds", [])
                        gt_cmds = getattr(ds, "cmds", [])
                        cmd_m = eval_cmd_only(pred_cmds, gt_cmds)
                        f1_score = float(cmd_m.get("eval/avg_f1", 0.0))

                    cache_data[key] = f1_score
                    n_eval += 1

                scored_samples.append((f1_score, ds))

    if use_cache and cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2)
        except Exception:
            pass

    scored_samples.sort(key=lambda item: item[0])

    n_subset = max(1, int(np.ceil(subset_top_p * n_train)))
    subsetted_seqs = [item[1] for item in scored_samples[:n_subset]]

    pipeline.dual_seqs = subsetted_seqs
    pipeline.load_loaders()

    print(f"successfully subsetted {n_subset} from {n_train} samples via teacher forcing ranking (cached: {n_cached}, evaluated: {n_eval})")
    return pipeline

