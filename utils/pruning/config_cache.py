import json
import os
from typing import Optional

from utils.pipeline.config import ModelConfig, TrainerConfig


def _canonical_key(cfg: ModelConfig, trainer_cfg: Optional[TrainerConfig] = None, batch_size: Optional[int] = None) -> str:
    """
    Compute a canonical string key for a configuration.
    Includes model, trainer, and batch size options.
    """
    use_drop_out = getattr(cfg, "use_drop_out", True)
    drop_out_p   = getattr(cfg, "drop_out_p", 0.1) if use_drop_out else 0.0
    adaptive_layer    = getattr(cfg, "adaptive_layer", "none")
    cmd_embedding_type  = getattr(cfg, "cmd_embedding_type", "standard")
    args_embedding_type = getattr(cfg, "args_embedding_type", "standard")
    use_cmd_args_fusion = getattr(cfg, "use_cmd_args_fusion", False)
    n_dec_blocks     = getattr(cfg, "n_dec_blocks", 6)

    moe_conf = getattr(cfg, "moe_conf", None)
    args_dec = cfg.args_decoder_type if not cfg.is_cmd_only else None

    key_dict = {
        "d_model": getattr(cfg, "d_model", 512) or 512,
        "encoder_type": cfg.encoder_type,
        "cmd_decoder_type": cfg.cmd_decoder_type,
        "args_decoder_type": args_dec,
        "is_cmd_only": cfg.is_cmd_only,
        "moe_conf": moe_conf,
        "adaptive_layer": adaptive_layer,
        "cmd_embedding_type": cmd_embedding_type,
        "args_embedding_type": args_embedding_type if not cfg.is_cmd_only else None,
        "use_drop_out": use_drop_out,
        "drop_out_p": drop_out_p,
        "use_cmd_args_fusion": use_cmd_args_fusion,
        "n_dec_blocks": n_dec_blocks,
    }

    if trainer_cfg is not None:
        opt_kwargs = getattr(trainer_cfg, "optimizer_kwargs", {}) or {}
        key_dict["lr"] = opt_kwargs.get("lr", 1e-4)
        key_dict["optimizer"] = getattr(trainer_cfg, "optimizer", "AdamW")
        key_dict["weight_decay"] = opt_kwargs.get("weight_decay", 0.0)

        sched = getattr(trainer_cfg, "scheduler", None)
        if sched is not None:
            key_dict["scheduler_type"] = getattr(sched, "type", "none")
            key_dict["warmup_ratio"] = getattr(sched, "warmup_ratio", 0.0)
            key_dict["eta_min"] = getattr(sched, "eta_min", 0.0)
        else:
            key_dict["scheduler_type"] = "none"

    if batch_size is not None:
        key_dict["batch_size"] = batch_size

    return json.dumps(key_dict, sort_keys=True)


class ConfigCache:
    """
    Persistent cache mapping canonical config keys to their best val_avg_f1.
    Using JSON to persist cache
    """

    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self._data: dict = {}
        self.load()

    def load(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r") as f:
                self._data = json.load(f)
        else:
            self._data = {}

    def save(self):
        """Persist cache to disk."""
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self._data, f, indent=2)

    def key(self, cfg: ModelConfig, trainer_cfg: Optional[TrainerConfig] = None, batch_size: Optional[int] = None) -> str:
        return _canonical_key(cfg, trainer_cfg, batch_size)


    def get(self, key: str) -> Optional[float]:
        return self._data.get(key, None)

    def set(self, key: str, value: float):
        self._data[key] = value

    def __len__(self) -> int:
        return len(self._data)
