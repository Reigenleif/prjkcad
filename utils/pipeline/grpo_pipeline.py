from __future__ import annotations

import os
from typing import Any, Dict, Optional, Union
import torch
from transformers import AutoTokenizer

from models.base_model import BaseModel
from utils.pipeline.base_pipeline import BasePipeline
from utils.pipeline.config import Config, GRPOConfig
from utils.data_utils import create_dualseq_data_loader
from utils.dual_seq import get_dualseq_schema
from utils.wrapper.custom_wrapper import CustomWrapper
from utils.criterion.custom_criterion import CustomCriterion
from utils.trainer.custom_trainer import CustomTrainer

class GRPOPipeline(BasePipeline):
    """Pipeline for GRPO reinforcement learning execution."""

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

        self.cfg.model.out_type = "EightBitBinarizedArgs"
        use_val = self.cfg.data.eval_split_ratio > 0

        if self.val_dual_seqs is not None:
            self.train_loader = create_dualseq_data_loader(self.dual_seqs, self.text_tokenizer, description_level=self.cfg.data.description_level, batch_size=self.cfg.data.batch_size, num_workers=self.cfg.data.num_workers, val_ratio=0.0, shuffle=True, out_type="EightBitBinarizedArgs")
            self.val_loader = create_dualseq_data_loader(self.val_dual_seqs, self.text_tokenizer, description_level=self.cfg.data.description_level, batch_size=self.cfg.data.batch_size, num_workers=self.cfg.data.num_workers, val_ratio=0.0, shuffle=False, out_type="EightBitBinarizedArgs")
            return

        if use_val:
            self.train_loader, self.val_loader = create_dualseq_data_loader(self.dual_seqs, self.text_tokenizer, description_level=self.cfg.data.description_level, batch_size=self.cfg.data.batch_size, num_workers=self.cfg.data.num_workers, val_ratio=self.cfg.data.eval_split_ratio, shuffle=True, out_type="EightBitBinarizedArgs")
        else:
            self.train_loader = create_dualseq_data_loader(self.dual_seqs, self.text_tokenizer, description_level=self.cfg.data.description_level, batch_size=self.cfg.data.batch_size, num_workers=self.cfg.data.num_workers, val_ratio=0.0, shuffle=True, out_type="EightBitBinarizedArgs")
            self.val_loader = None

    def load_model_and_wrapper(self) -> None:
        # <-- Model & CustomWrapper Setup -->
        schema = get_dualseq_schema()
        model_kwargs = {"vocab_size": schema["cmd_n_tokens"], "vocab_size_args": schema["args_n_tokens"], "cfg": self.cfg.model, **self.cfg.model.kwargs}
        self.model = BaseModel(**model_kwargs)
        self.load_weights()
        self.wrapper = CustomWrapper(self.model, self.text_tokenizer, out_type="grpo", metadata=self.metadata)


    def load_criterion(self) -> None:
        # <-- CustomCriterion Setup -->
        self.criterion = CustomCriterion(self.cfg.trainer.criterion, out_type="EightBitBinarizedArgs")

    def load_trainer(self) -> None:
        # <-- CustomTrainer Setup -->
        optimizer = torch.optim.AdamW(self.wrapper.parameters(), **self.cfg.trainer.optimizer_kwargs)
        grpo_kwargs = getattr(self.cfg, "grpo", {})
        if hasattr(grpo_kwargs, "to_dict"):
            grpo_kwargs = grpo_kwargs.to_dict()

        self.trainer = CustomTrainer(
            trainer_cfg=self.cfg.trainer,
            wrapper=self.wrapper,
            criterion=self.criterion,
            optimizer=optimizer,
            save_folder=self.SAVE_ROOT,
            trainer_type="grpo",
            run_name=getattr(self.cfg, "run_name", None),
            **grpo_kwargs
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
