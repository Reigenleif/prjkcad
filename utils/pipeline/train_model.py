from __future__ import annotations

import os
import glob
from typing import Any, Dict, Union
import pandas as pd

from utils.pipeline.config import Config
from utils.pipeline.fine_tune_pipeline import FineTunePipeline
from utils.pipeline.pretrain_pipeline import PretrainPipeline
from utils.pipeline.grpo_pipeline import GRPOPipeline

def load_config(config_path: str) -> Config:
    # <-- Load Config YAML -->
    return Config.from_yaml(config_path)

def merge_best_epochs(out_dir: str = "out", output_path: str = "out/best_merged.csv") -> pd.DataFrame:
    # <-- Merge CSV Results -->
    pattern = os.path.join(out_dir, "*", "best_epoch.csv")
    csv_paths = sorted(glob.glob(pattern))

    if not csv_paths:
        raise FileNotFoundError(f"No best_epoch.csv files found matching: {pattern}")

    frames = []
    for path in csv_paths:
        df = pd.read_csv(path)
        df.insert(0, "run_name", os.path.basename(os.path.dirname(path)))
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged.to_csv(output_path, index=False)
    print(f"Merged {len(frames)} experiment(s) -> {output_path}")
    return merged

class CustomPipeline:
    """Factory router instantiating the matching pipeline based on config type."""

    def __new__(cls, cfg: Union[Config, Dict[str, Any], str]):
        # <-- Config Type Guard Router -->
        if isinstance(cfg, str):
            cfg = load_config(cfg)
        elif isinstance(cfg, dict):
            cfg = Config.from_dict(cfg)

        cfg_type = getattr(cfg, "type", "fine_tune")
        if cfg_type == "pretrain":
            return PretrainPipeline(cfg)
        if cfg_type == "grpo":
            return GRPOPipeline(cfg)
        if cfg_type in ["fine_tune", "fine_tuning"]:
            return FineTunePipeline(cfg)

        raise ValueError(f"Unsupported config type: {cfg_type}")
