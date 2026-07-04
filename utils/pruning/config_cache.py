import json
import os
from typing import Optional

from utils.pipeline.config import ModelConfig


def _canonical_key(cfg: ModelConfig) -> str:
    """
    Compute a canonical string key for a ModelConfig.

    Normalization rules applied before hashing:
    - If moe_conf is None, ignore moe_type (set it to None too).
    - If is_cmd_only=True, set args_decoder_type=None.
    - Extract only the relevant fields (no run_name, no pretrained_path).
    - Sort fields alphabetically and serialize to a compact JSON string.
    """
    use_drop_out = getattr(cfg, "use_drop_out", True)
    drop_out_p   = getattr(cfg, "drop_out_p", 0.1) if use_drop_out else 0.0
    adaptive_layer    = getattr(cfg, "adaptive_layer", "none")
    cmd_embedding_type  = getattr(cfg, "cmd_embedding_type", "standard")
    args_embedding_type = getattr(cfg, "args_embedding_type", "standard")

    moe_conf = getattr(cfg, "moe_conf", None)
    # If no MoE conf, moe_type is irrelevant
    moe_type = getattr(cfg, "moe_type", None) if moe_conf is not None else None

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
    }

    return json.dumps(key_dict, sort_keys=True)


class ConfigCache:
    """
    Persistent cache mapping canonical ModelConfig keys to their best val_avg_f1.
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

    def key(self, cfg: ModelConfig) -> str:
        return _canonical_key(cfg)

    def get(self, key: str) -> Optional[float]:
        return self._data.get(key, None)

    def set(self, key: str, value: float):
        self._data[key] = value

    def __len__(self) -> int:
        return len(self._data)
