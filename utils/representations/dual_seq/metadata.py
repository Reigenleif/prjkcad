import os
import pickle
import numpy as np
from .schema import DEFAULT_COMMANDS

class DualSeqMetadata:
    def __init__(self):
        self.min_vals = {}
        self.max_vals = {}
        self.num_bins = 256
        
    def fit(self, dual_seqs):
        all_vals = {}
        for cmd, keys in DEFAULT_COMMANDS.items():
            for key in keys:
                all_vals[key] = []
                
        for ds in dual_seqs:
            for i, cmd in enumerate(ds.cmds):
                if cmd in DEFAULT_COMMANDS:
                    arg_dict = ds.args[i]
                    for key in DEFAULT_COMMANDS[cmd]:
                        val = arg_dict.get(key, 0.0)
                        all_vals[key].append(val)
                        
        for key, vals in all_vals.items():
            if vals:
                self.min_vals[key] = float(min(vals))
                self.max_vals[key] = float(max(vals))
            else:
                self.min_vals[key] = 0.0
                self.max_vals[key] = 1.0
                
    def float_to_bin(self, key: str, val: float) -> int:
        min_val = self.min_vals.get(key, 0.0)
        max_val = self.max_vals.get(key, 1.0)
        if max_val <= min_val:
            return 0
        scaled = (val - min_val) / (max_val - min_val)
        bin_idx = int(np.clip(np.floor(scaled * self.num_bins), 0, self.num_bins - 1))
        return bin_idx
        
    def bin_to_float(self, key: str, bin_idx: int) -> float:
        min_val = self.min_vals.get(key, 0.0)
        max_val = self.max_vals.get(key, 1.0)
        if max_val <= min_val:
            return min_val
        val = min_val + ((bin_idx + 0.5) / self.num_bins) * (max_val - min_val)
        return val
        
    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
            
    @classmethod
    def load(cls, filepath: str) -> "DualSeqMetadata":
        with open(filepath, "rb") as f:
            return pickle.load(f)
