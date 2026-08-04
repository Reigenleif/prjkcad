from __future__ import annotations

import os
from typing import Any, Dict, Optional, Union
import torch
from transformers import AutoTokenizer

from models.base_model import BaseModel
from utils.pipeline.base_pipeline import BasePipeline
from utils.pipeline.config import Config
from utils.data_utils import create_dualseq_data_loader
from utils.dual_seq import get_dualseq_schema
from utils.wrapper.custom_wrapper import CustomWrapper
from utils.criterion.custom_criterion import CustomCriterion
from utils.trainer.custom_trainer import CustomTrainer

class FineTunePipeline(BasePipeline):
    """Pipeline for Supervised Fine-Tuning execution."""

    def load_tokenizer(self) -> None:
        # <-- Tokenizer Initialization -->
        if self.cfg.tokenizer.source == "huggingface":
            self.text_tokenizer = AutoTokenizer.from_pretrained(self.cfg.tokenizer.model_name)
        else:
            raise ValueError(f"Unsupported tokenizer: {self.cfg.tokenizer.source}")

    def load_loaders(self) -> None:
        # <-- DataLoader Initialization -->
        if self.text_tokenizer is None:
            self.load_tokenizer()

        use_val = self.cfg.data.eval_split_ratio > 0
        if self.val_dual_seqs is not None:
            self.train_loader = create_dualseq_data_loader(self.dual_seqs, self.text_tokenizer, description_level=self.cfg.data.description_level, batch_size=self.cfg.data.batch_size, num_workers=self.cfg.data.num_workers, val_ratio=0.0, shuffle=True, out_type=self.out_type, metadata=self.metadata)
            self.val_loader = create_dualseq_data_loader(self.val_dual_seqs, self.text_tokenizer, description_level=self.cfg.data.description_level, batch_size=self.cfg.data.batch_size, num_workers=self.cfg.data.num_workers, val_ratio=0.0, shuffle=False, out_type=self.out_type, metadata=self.metadata)
            return

        if use_val:
            self.train_loader, self.val_loader = create_dualseq_data_loader(self.dual_seqs, self.text_tokenizer, description_level=self.cfg.data.description_level, batch_size=self.cfg.data.batch_size, num_workers=self.cfg.data.num_workers, val_ratio=self.cfg.data.eval_split_ratio, shuffle=True, out_type=self.out_type, metadata=self.metadata)
        else:
            self.train_loader = create_dualseq_data_loader(self.dual_seqs, self.text_tokenizer, description_level=self.cfg.data.description_level, batch_size=self.cfg.data.batch_size, num_workers=self.cfg.data.num_workers, val_ratio=0.0, shuffle=True, out_type=self.out_type, metadata=self.metadata)
            self.val_loader = None

    def load_model_and_wrapper(self) -> None:
        # <-- BaseModel & CustomWrapper Instantiation -->
        schema = get_dualseq_schema()
        model_kwargs = {"vocab_size": schema["cmd_n_tokens"], "vocab_size_args": schema["args_n_tokens"], "cfg": self.cfg.model, **self.cfg.model.kwargs}
        self.model = BaseModel(**model_kwargs)
        self.wrapper = CustomWrapper(self.model, self.text_tokenizer, out_type=self.out_type, metadata=self.metadata)

    def load_criterion(self) -> None:
        # <-- CustomCriterion Instantiation -->
        self.criterion = CustomCriterion(self.cfg.trainer.criterion, out_type=self.out_type)

    def load_trainer(self) -> None:
        # <-- CustomTrainer Instantiation -->
        optimizer = torch.optim.AdamW(self.wrapper.parameters(), **self.cfg.trainer.optimizer_kwargs)
        self.trainer = CustomTrainer(
            trainer_cfg=self.cfg.trainer,
            wrapper=self.wrapper,
            criterion=self.criterion,
            optimizer=optimizer,
            save_folder=self.SAVE_ROOT,
            trainer_type="gd"
        )

    def run(self) -> None:
        # <-- Complete Pipeline Runner -->
        self.load_tokenizer()
        self.load_dataset()
        self.load_loaders()
        self.load_model_and_wrapper()
        self.load_criterion()
        self.load_trainer()
        self.trainer.fit(self.train_loader, self.val_loader)
