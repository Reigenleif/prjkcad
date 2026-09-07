from __future__ import annotations

import copy
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
from utils.trainer import CustomTrainer
from utils.scheduler.scheduler import CustomScheduler
from utils.wandb import init_wandb
from utils.grpo.subset import subset_grpo_dataset

class GRPOPipeline(BasePipeline):
    """Pipeline for GRPO reinforcement learning execution."""

    def __init__(self, cfg: Union[Config, Dict[str, Any], str]):
        super().__init__(cfg)
        self.out_type = "EightBitBinarizedArgs"

    def load_tokenizer(self) -> None:
        if self.cfg.tokenizer.source == "huggingface":
            self.text_tokenizer = AutoTokenizer.from_pretrained(self.cfg.tokenizer.model_name)
        else:
            raise ValueError(f"Unsupported tokenizer: {self.cfg.tokenizer.source}")

    def load_loaders(self) -> None:
        if self.text_tokenizer is None:
            self.load_tokenizer()

        self.cfg.model.out_type = "EightBitBinarizedArgs"
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
        schema = get_dualseq_schema()
        model_kwargs = {"vocab_size": schema["cmd_n_tokens"], "vocab_size_args": schema["args_n_tokens"], "cfg": self.cfg.model, **self.cfg.model.kwargs}
        self.model = BaseModel(**model_kwargs)
        self.load_weights()
        self.wrapper = CustomWrapper(self.model, self.text_tokenizer, out_type="grpo", metadata=self.metadata)

    def load_criterion(self) -> None:
        self.criterion = CustomCriterion(self.cfg.trainer.criterion, out_type="EightBitBinarizedArgs")

    def load_trainer(self) -> None:
        opt_name = getattr(self.cfg.trainer, "optimizer", "AdamW")
        opt_cls = getattr(torch.optim, opt_name, torch.optim.AdamW)
        optimizer = opt_cls(self.wrapper.parameters(), **self.cfg.trainer.optimizer_kwargs)

        scheduler = None
        if hasattr(self.cfg.trainer, "scheduler") and self.cfg.trainer.scheduler is not None and self.train_loader is not None:
            total_steps = len(self.train_loader) * self.cfg.trainer.epochs
            scheduler = CustomScheduler(optimizer, self.cfg.trainer.scheduler, total_steps=total_steps)

        eval_steps = getattr(self.cfg.trainer, "eval_steps", 1000)
        t_kwargs = copy.deepcopy(getattr(self.cfg.trainer, "kwargs", {}) or {})
        max_grad_norm = t_kwargs.pop("max_grad_norm", getattr(self.cfg.trainer, "max_grad_norm", 1.0))
        quant_type = t_kwargs.pop("quant_type", getattr(self.cfg.trainer, "quant_type", None))

        grpo_kwargs = getattr(self.cfg, "grpo", {})
        if hasattr(grpo_kwargs, "to_dict"):
            grpo_kwargs = grpo_kwargs.to_dict()

        merged_kwargs = {**t_kwargs, **grpo_kwargs}

        subset_refresh_every = merged_kwargs.pop("subset_refresh_every", 0)

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
            trainer_type="grpo",
            epochs=self.cfg.trainer.epochs,
            eval_steps=eval_steps,
            max_grad_norm=max_grad_norm,
            quant_type=quant_type,
            run_name=getattr(self.cfg, "run_name", None),
            out_type=self.out_type,
            metadata=self.metadata,
            pipeline=self,
            subset_refresh_every=subset_refresh_every,
            **merged_kwargs
        )


    def load(self) -> None:
        self.load_tokenizer()
        self.load_dataset()
        self.load_model_and_wrapper()
        subset_top_p = getattr(getattr(self.cfg, "grpo", None), "subset_top_p", 1.0)
        if subset_top_p is not None and subset_top_p < 1.0:
            subset_grpo_dataset(self, use_cache=True)
        else:
            self.load_loaders()
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
        progression = self.trainer.fit(self.train_loader, self.val_loader)
        run_data_file = os.path.join(self.SAVE_ROOT, "run_data.csv")
        if os.path.exists(run_data_file):
            print(f"Generated run_data.csv at: {run_data_file}")
        return progression

    def eval(self, val_loader: Optional[Any] = None) -> Any:
        if self.trainer is None:
            self.load()
        loader = val_loader if val_loader is not None else self.val_loader
        if loader is None:
            raise ValueError("No validation loader available for evaluation.")
        return self.trainer.eval(loader)
