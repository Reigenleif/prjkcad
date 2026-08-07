from __future__ import annotations

import os
from typing import Any, Dict, Optional, Union
import torch
from transformers import AutoTokenizer

from models.base_model import BaseModel
from utils.pipeline.base_pipeline import BasePipeline
from utils.pipeline.config import Config
from utils.data_utils import create_pretrain_data_loader
from utils.data_utils import load_split_data
from utils.dual_seq import get_dualseq_schema
from utils.wrapper.custom_wrapper import CustomWrapper
from utils.criterion.custom_criterion import CustomCriterion
from utils.trainer.custom_trainer import CustomTrainer
from utils.scheduler.scheduler import CustomScheduler

class PretrainPipeline(BasePipeline):
    """Pipeline for Autoencoder Pretraining execution."""

    def load_dataset(self) -> None:
        # <-- Load All Data Without Split JSON for Pretrain -->
        if self.dual_seqs is not None:
            return
        max_samples = self.cfg.data.max_samples
        sample_ratio = None if max_samples is not None else self.cfg.data.sample_ratio
        self.dual_seqs, _ = load_split_data(
            data_folder=self.cfg.data.data_folder,
            metadata_csv=self.cfg.data.metadata_csv,
            source_data_type=self.cfg.data.source_data_type,
            split_json=None,
            max_samples=max_samples,
            sample_ratio=sample_ratio,
        )
        self.val_dual_seqs = None

    def load_tokenizer(self) -> None:
        # <-- Tokenizer Setup -->
        if self.cfg.tokenizer.source == "huggingface":
            self.text_tokenizer = AutoTokenizer.from_pretrained(self.cfg.tokenizer.model_name)
        else:
            raise ValueError(f"Unsupported tokenizer: {self.cfg.tokenizer.source}")

    def load_loaders(self) -> None:
        # <-- DataLoader Setup -->
        if self.text_tokenizer is None:
            self.load_tokenizer()

        use_val = self.cfg.data.eval_split_ratio > 0
        if self.val_dual_seqs is not None:
            self.train_loader = create_pretrain_data_loader(self.dual_seqs, tokenizer=self.text_tokenizer, description_level=self.cfg.data.description_level, batch_size=self.cfg.data.batch_size, num_workers=self.cfg.data.num_workers, val_ratio=0.0, shuffle=True)
            self.val_loader = create_pretrain_data_loader(self.val_dual_seqs, tokenizer=self.text_tokenizer, description_level=self.cfg.data.description_level, batch_size=self.cfg.data.batch_size, num_workers=self.cfg.data.num_workers, val_ratio=0.0, shuffle=False)
            return

        if use_val:
            self.train_loader, self.val_loader = create_pretrain_data_loader(self.dual_seqs, tokenizer=self.text_tokenizer, description_level=self.cfg.data.description_level, batch_size=self.cfg.data.batch_size, num_workers=self.cfg.data.num_workers, val_ratio=self.cfg.data.eval_split_ratio, shuffle=True)
        else:
            self.train_loader = create_pretrain_data_loader(self.dual_seqs, tokenizer=self.text_tokenizer, description_level=self.cfg.data.description_level, batch_size=self.cfg.data.batch_size, num_workers=self.cfg.data.num_workers, val_ratio=0.0, shuffle=True)
            self.val_loader = None

    def load_model_and_wrapper(self) -> None:
        # <-- Model & CustomWrapper Setup -->
        schema = get_dualseq_schema()
        model_kwargs = {"vocab_size": len(self.text_tokenizer), "vocab_size_args": schema["args_n_tokens"], "cfg": self.cfg.model, **self.cfg.model.kwargs}
        self.model = BaseModel(**model_kwargs)
        self.load_weights()
        self.wrapper = CustomWrapper(self.model, self.text_tokenizer, out_type="pretrain")


    def load_criterion(self) -> None:
        # <-- CustomCriterion Setup -->
        self.criterion = CustomCriterion(self.cfg.trainer.criterion, out_type="pretrain")

    def load_trainer(self) -> None:
        # <-- CustomTrainer Setup -->
        optimizer = torch.optim.AdamW(self.wrapper.parameters(), **self.cfg.trainer.optimizer_kwargs)
        
        scheduler = None
        if hasattr(self.cfg.trainer, "scheduler") and self.cfg.trainer.scheduler is not None and self.train_loader is not None:
            total_steps = len(self.train_loader) * self.cfg.trainer.epochs
            scheduler = CustomScheduler(optimizer, self.cfg.trainer.scheduler, total_steps=total_steps)

        self.trainer = CustomTrainer(
            trainer_cfg=self.cfg.trainer,
            wrapper=self.wrapper,
            criterion=self.criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            save_folder=self.SAVE_ROOT,
            trainer_type="gd",
            run_name=getattr(self.cfg, "run_name", None)
        )

    def load(self) -> None:
        self.load_tokenizer()
        self.load_dataset()
        self.load_loaders()
        self.load_model_and_wrapper()
        self.load_criterion()
        self.load_trainer()

    def run(self) -> None:
        if self.trainer is None:
            self.load()
        self.trainer.fit(self.train_loader, self.val_loader)

    def infer(self, input_text: str, max_new_tokens: int = 50) -> Any:
        # <-- Pretrain Pipeline Guard Clause -->
        raise NotImplementedError("PretrainPipeline does not support DualSeq CAD inference.")

