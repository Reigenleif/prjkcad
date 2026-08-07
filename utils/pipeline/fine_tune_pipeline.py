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
from utils.scheduler.scheduler import CustomScheduler

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
        self.load_weights()
        self.wrapper = CustomWrapper(self.model, self.text_tokenizer, out_type=self.out_type, metadata=self.metadata)


    def load_criterion(self) -> None:
        # <-- CustomCriterion Instantiation -->
        self.criterion = CustomCriterion(self.cfg.trainer.criterion, out_type=self.out_type)

    def load_trainer(self) -> None:
        # <-- CustomTrainer Instantiation -->
        optimizer = torch.optim.AdamW(self.wrapper.parameters(), **self.cfg.trainer.optimizer_kwargs)
        
        scheduler = None
        if hasattr(self.cfg.trainer, "scheduler") and self.cfg.trainer.scheduler is not None and self.train_loader is not None:
            total_steps = len(self.train_loader) * self.cfg.trainer.epochs
            scheduler = CustomScheduler(optimizer, self.cfg.trainer.scheduler, total_steps=total_steps)

        eval_steps = getattr(self.cfg.trainer, "eval_steps", 1000)
        t_kwargs = getattr(self.cfg.trainer, "kwargs", {}) or {}
        max_grad_norm = t_kwargs.get("max_grad_norm", getattr(self.cfg.trainer, "max_grad_norm", 1.0))
        quant_type = t_kwargs.get("quant_type", getattr(self.cfg.trainer, "quant_type", None))

        self.trainer = CustomTrainer(
            trainer_cfg=self.cfg.trainer,
            use_wandb=getattr(self.cfg, "use_wandb", True),
            wrapper=self.wrapper,
            criterion=self.criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            save_folder=self.SAVE_ROOT,
            trainer_type="gd",
            epochs=self.cfg.trainer.epochs,
            eval_steps=eval_steps,
            max_grad_norm=max_grad_norm,
            quant_type=quant_type,
            run_name=getattr(self.cfg, "run_name", None),
            out_type=self.out_type,
            metadata=self.metadata
        )

    def load(self) :
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

    def eval(self, val_loader: Optional[Any] = None) -> Any:
        # <-- Complete Pipeline Evaluator -->
        if self.trainer is None:
            self.load()
        loader = val_loader if val_loader is not None else self.val_loader
        if loader is None:
            raise ValueError("No validation loader available for evaluation.")
        return self.trainer.eval(loader)

