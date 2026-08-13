from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

from utils.representations.CadSeqProc.cad_sequence import CADSequence


def eval_t2c_report(gt_cad_seq: Optional[CADSequence], pred_cad_seq: Optional[CADSequence], uid: str = "0000") -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    if gt_cad_seq is None or pred_cad_seq is None:
        metrics["t2c/is_invalid"] = 1.0
        return metrics

    try:
        report_df, _ = gt_cad_seq.generate_report(pred_cad_seq, uid=uid)
        if report_df is not None and not report_df.empty:
            for col in report_df.columns:
                if col in ("uid", "model_type"):
                    continue
                val = report_df[col].iloc[0]
                if isinstance(val, (int, float, np.number)):
                    val_float = float(val)
                    if np.isnan(val_float):
                        val_float = 0.0
                    metrics[f"t2c/{col}"] = val_float
    except Exception:
        metrics["t2c/is_invalid"] = 1.0

    return metrics
