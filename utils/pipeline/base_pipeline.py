from __future__ import annotations

import os
from typing import Any, Dict, Optional, Union
import torch
from transformers import AutoTokenizer

from utils.pipeline.config import Config
from utils.set_seed import set_seed
from utils.data_utils import load_split_data, load_test_data
from utils.wrapper.custom_wrapper import CustomWrapper
from utils.criterion.custom_criterion import CustomCriterion
from utils.trainer import CustomTrainer
from utils.representations.dual_seq.dual_seq import DualSeqMetadata, DualSeq
from utils.wandb import init_wandb

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
        self.test_dual_seqs = None
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

    @property
    def tokenizer(self):
        if self.text_tokenizer is None:
            self.load_tokenizer()
        return self.text_tokenizer

    def load_dataset(self) -> None:
        # <-- Load Dataset Split Data -->
        val_max_sample = getattr(self.cfg.data, "val_max_sample", None)
        val_sample_ratio = getattr(self.cfg.data, "val_sample_ratio", None)
        test_max_sample = getattr(self.cfg.data, "test_max_sample", getattr(self.cfg.data, "test_max_samples", None))
        random_sample = getattr(self.cfg.data, "random_sample", False)
        seed = getattr(self.cfg, "random_seed", 42)
        if self.dual_seqs is None:
            self.dual_seqs, self.val_dual_seqs = load_split_data(
                data_folder=self.cfg.data.data_folder,
                metadata_csv=self.cfg.data.metadata_csv,
                source_data_type=self.cfg.data.source_data_type,
                split_json=self.cfg.data.split_json,
                max_samples=self.cfg.data.max_samples,
                sample_ratio=self.cfg.data.sample_ratio,
                val_max_sample=val_max_sample,
                val_sample_ratio=val_sample_ratio,
                random_train=random_sample,
                seed=seed,
            )
        if val_max_sample is not None and self.val_dual_seqs is not None:
            self.val_dual_seqs = self.val_dual_seqs[:val_max_sample]
        if test_max_sample is not None:
            self.load_test_dataset(test_max_sample=test_max_sample)
        self._fit_metadata_if_needed()

    def load_test_dataset(self, test_max_sample: Optional[int] = None) -> list[DualSeq]:
        t_max = test_max_sample if test_max_sample is not None else getattr(self.cfg.data, "test_max_sample", getattr(self.cfg.data, "test_max_samples", None))
        t_ratio = getattr(self.cfg.data, "test_sample_ratio", None)
        self.test_dual_seqs = load_test_data(
            data_folder=self.cfg.data.data_folder,
            metadata_csv=self.cfg.data.metadata_csv,
            source_data_type=self.cfg.data.source_data_type,
            split_json=self.cfg.data.split_json,
            test_max_sample=t_max,
            test_sample_ratio=t_ratio,
        )
        return self.test_dual_seqs

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

    def load_weights(self, path: Optional[str] = None) -> bool:
        # <-- Automatic Checkpoint Weight Loading -->
        if path is None:
            candidates = [
                os.path.join(self.SAVE_ROOT, "checkpoint.ckpt"),
                os.path.join(self.SAVE_ROOT, "checkpoint.pt"),
            ]
            if hasattr(self.cfg, "pretrained_path") and self.cfg.pretrained_path:
                p = str(self.cfg.pretrained_path)
                candidates.extend([
                    p,
                    os.path.join(p, "checkpoint.ckpt"),
                    os.path.join(p, "checkpoint.pt"),
                    os.path.join("out", p),
                    os.path.join("out", p, "checkpoint.ckpt"),
                    os.path.join("out", p, "checkpoint.pt"),
                ])
            for cand in candidates:
                if cand and os.path.isfile(cand):
                    path = cand
                    break

        if not path or not os.path.isfile(path):
            if hasattr(self.cfg, "pretrained_path") and self.cfg.pretrained_path:
                print(f"[BasePipeline] WARNING: Pretrained checkpoint '{self.cfg.pretrained_path}' NOT FOUND! Starting without pretrained weights.")
            return False

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            state_dict = checkpoint

        if any(isinstance(v, torch.Tensor) and torch.isnan(v).any() for v in state_dict.values()):
            print(f"Skipping corrupted checkpoint with NaN weights: {path}")
            return False

        clean_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            for prefix in ["lightning_module.", "wrapper.wrapper.model.", "wrapper.model.", "model."]:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            clean_state_dict[new_key] = v

        if hasattr(self, "model") and self.model is not None:
            model_state = self.model.state_dict()
            clean_state_dict = {
                k: v for k, v in clean_state_dict.items()
                if k in model_state and model_state[k].shape == v.shape
            }
            self.model.load_state_dict(clean_state_dict, strict=False)
            print(f"Loaded checkpoint weights from: {path}")
            return True
        return False

    def load(self) -> None:
        self.load_tokenizer()
        self.load_dataset()
        self.load_loaders()
        self.load_model_and_wrapper()
        self.load_criterion()
        self.load_trainer()

    def run(self) -> Any:
        if self.trainer is None:
            self.load()
        eval_steps = getattr(getattr(self.cfg, "trainer", None), "eval_steps", 1000)
        if getattr(self.cfg, "use_wandb", True):
            init_wandb(
                run_name=self.cfg.run_name,
                config_dict=self.cfg.to_dict(),
                wrapper=self.wrapper,
                eval_steps=eval_steps
            )
        return self.trainer.fit(self.train_loader, self.val_loader)

    def eval(self, val_loader: Optional[Any] = None) -> Any:
        if self.trainer is None:
            self.load()
        loader = val_loader if val_loader is not None else self.val_loader
        if loader is None:
            raise ValueError("No validation loader available for evaluation.")
        return self.trainer.eval(loader)

    def infer(self, input_text: str, max_new_tokens: int = 50) -> DualSeq:
        # <-- Text-to-CAD DualSeq Inference -->
        if self.wrapper is None:
            if hasattr(self, "load"):
                self.load()
            elif hasattr(self, "load_tokenizer") and hasattr(self, "load_model_and_wrapper"):
                self.load_tokenizer()
                self.load_model_and_wrapper()
        if self.wrapper is None:
            raise RuntimeError("Model wrapper was not initialized for inference.")
        return self.wrapper.infer(input_text, max_new_tokens=max_new_tokens)


