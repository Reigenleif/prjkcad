import os
import glob
from typing import Union, Dict, Any, Optional

import pandas as pd

from utils.pipeline.config import Config
from utils.pipeline.fine_tune_pipeline import FineTuningPipeline
from utils.pipeline.tokenized_args_fine_tune_pipeline import TokenizedArgsFineTuningPipeline
from utils.pipeline.pretrain_pipeline import PretrainPipeline
from utils.pipeline.text2cad_pipeline import Text2CADPipeline
from utils.pipeline.grpo_pipeline import GRPOPipeline


def load_config(config_path: str) -> Config:
    """
    Load config from a YAML file.
    
    Args:
        config_path: Path to the config file.
        
    Returns:
        Config object.
    """
    return Config.from_yaml(config_path)


def merge_best_epochs(out_dir: str = "out", output_path: str = "out/best_merged.csv") -> pd.DataFrame:
    """
    Utility for optuna model pruning, concats results into a single table
    """ 

    pattern = os.path.join(out_dir, "*", "best_epoch.csv")
    csv_paths = sorted(glob.glob(pattern))

    if not csv_paths:
        raise FileNotFoundError(f"No best_epoch.csv files found matching: {pattern}")

    frames = []
    for path in csv_paths:
        run_name = os.path.basename(os.path.dirname(path))
        df = pd.read_csv(path)
        df.insert(0, "run_name", run_name)
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged.to_csv(output_path, index=False)
    print(f"Merged {len(frames)} experiment(s) → {output_path}  ({len(merged)} row(s))")
    return merged


class TrainModelPipeline:
    """
    Router Class.
    
    Instantiates FineTuningPipeline, TokenizedArgsFineTuningPipeline, PretrainPipeline or GRPOPipeline based on config type.
    """
    def __new__(cls, cfg: Union[Config, Dict[str, Any], str]):
        if isinstance(cfg, str):
            cfg = load_config(cfg)
        elif isinstance(cfg, dict):
            cfg = Config.from_dict(cfg)

        cfg_type = cfg.type
        if cfg_type == "pretrain":
            return PretrainPipeline(cfg)
        elif cfg_type == "grpo":
            return GRPOPipeline(cfg)
        elif cfg_type == "tokenized_args":
            return TokenizedArgsFineTuningPipeline(cfg)
        elif cfg_type == "text2cad":
            return Text2CADPipeline(cfg)
        else:
            return FineTuningPipeline(cfg)
