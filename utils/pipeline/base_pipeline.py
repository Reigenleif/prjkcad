from __future__ import annotations

import os
from typing import Any, Dict, Optional, Union
import torch
from transformers import AutoTokenizer

from utils.pipeline.config import Config
from utils.set_seed import set_seed
from utils.data_utils import load_split_data
from utils.wrapper.custom_wrapper import CustomWrapper
from utils.criterion.custom_criterion import CustomCriterion
from utils.trainer.custom_trainer import CustomTrainer
from utils.representations.dual_seq.dual_seq import DualSeqMetadata

class BasePipeline:
    """Base class for training & evaluation pipelines."""

    def __init__(self, cfg: Union[Config, Dict[str, Any], str]):
        # <-- Config & Setup Initialization -->
        self.cfg = self._parse_config(cfg)
        self.out_type = getattr(self.cfg.model, "out_type", None) or getattr(self.cfg.data, "out_type", "FloatArgs")
        self.SAVE_ROOT = f"out/{self.cfg.run_name}"
        os.makedirs(self.SAVE_ROOT, exist_ok=True)
        set_seed(getattr(self.cfg, "random_seed", 42))

        # <-- Components Placeholder -->
        self.text_tokenizer = None
        self.dual_seqs = None
        self.val_dual_seqs = None
        self.metadata = None
        self.wrapper = None
        self.criterion = None
        self.trainer = None

    def _parse_config(self, cfg: Union[Config, Dict[str, Any], str]) -> Config:
        # <-- Config Parsing Guard Clause -->
        if isinstance(cfg, str):
            return Config.from_yaml(cfg)
        if isinstance(cfg, dict):
            return Config.from_dict(cfg)
        return cfg

    def load_tokenizer(self) -> None:
        # <-- Load HuggingFace Tokenizer -->
        if self.cfg.tokenizer.source == "huggingface":
            self.text_tokenizer = AutoTokenizer.from_pretrained(self.cfg.tokenizer.model_name)
        else:
            raise ValueError(f"Unsupported tokenizer source: {self.cfg.tokenizer.source}")

    def load_dataset(self) -> None:
        # <-- Load Dataset Split Data -->
        if self.dual_seqs is None:
            self.dual_seqs, self.val_dual_seqs = load_split_data(
                data_folder=self.cfg.data.data_folder,
                metadata_csv=self.cfg.data.metadata_csv,
                source_data_type=self.cfg.data.source_data_type,
                split_json=self.cfg.data.split_json,
                max_samples=self.cfg.data.max_samples,
                sample_ratio=self.cfg.data.sample_ratio
            )
        self._fit_metadata_if_needed()

    def _fit_metadata_if_needed(self) -> None:
        # <-- Metadata Fitting Guard -->
        if self.out_type == "EightBitBinarizedArgs":
            metadata_path = f"{self.SAVE_ROOT}/dual_seq_metadata.pkl"
            if os.path.exists(metadata_path):
                self.metadata = DualSeqMetadata.load(metadata_path)
            else:
                self.metadata = DualSeqMetadata()
                self.metadata.fit(self.dual_seqs)
                self.metadata.save(metadata_path)
