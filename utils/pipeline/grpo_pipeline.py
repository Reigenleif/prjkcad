import os
import random
from typing import Union, Dict, Any

import pandas as pd
import torch
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from utils.set_seed import set_seed
from utils.data_utils import RefLoader, load_split_data
from utils.dual_seq import DualSeq, get_dualseq_schema
from utils.data_utils import create_dualseq_data_loader
from utils.wrapper.grpo_wrapper import GRPOWrapper
from utils.criterion import DualSeqCriterion
from utils.trainer.grpo_trainer import GRPOTrainer
from models import BaseModel
from utils.pipeline.config import Config


class GRPOPipeline:
    def __init__(self, cfg: Union[Config, Dict[str, Any], str]):
        if isinstance(cfg, str):
            cfg = Config.from_yaml(cfg)
        elif isinstance(cfg, dict):
            cfg = Config.from_dict(cfg)
        self.cfg = cfg
        self.progression = None
        self.trainer = None
        self.wrapper = None
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.train_loader = None
        self.val_loader = None
        self.dual_seqs = None
        self.val_dual_seqs = None

    def load_things(self):
        config = self.cfg
        
        # Validation checks
        has_split_json = config.data.split_json is not None
        has_val_ratio = config.data.eval_split_ratio > 0.0
        if not (has_split_json or has_val_ratio):
            raise ValueError("GRPO pipeline requires a validation set for reward scoring.")
            
        if config.data.is_cmdonly:
            raise ValueError("GRPO pipeline only supports cmd+args models (is_cmdonly must be False).")

        self.SAVE_ROOT = f"out/{config.run_name}"
        os.makedirs(self.SAVE_ROOT, exist_ok=True)
        self.CHECKPOINT_SAVE_PATH = f"{self.SAVE_ROOT}/checkpoint.pt"
        self.RENDER_RESULTS_PATH = f"{self.SAVE_ROOT}/render_results"
        self.TEST_RESULT_PATH = f"{self.SAVE_ROOT}/test_result.csv"
        
        # Set random seed
        set_seed(config.random_seed)
        
        # Tokenizer loading
        if config.tokenizer.source == "huggingface":
            text_tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.model_name)
        else:
            raise ValueError(f"Unsupported tokenizer source: {config.tokenizer.source}")
            
        # Raw data loading
        if self.dual_seqs is None:
            dual_seqs, val_dual_seqs = load_split_data(
                data_folder=config.data.data_folder,
                metadata_csv=config.data.metadata_csv,
                source_data_type=config.data.source_data_type,
                split_json=config.data.split_json,
                max_samples=config.data.max_samples,
                sample_ratio=config.data.sample_ratio
            )
            self.dual_seqs = dual_seqs
            self.val_dual_seqs = val_dual_seqs

        dual_seqs = self.dual_seqs
        val_dual_seqs = self.val_dual_seqs

        if val_dual_seqs is not None:
            train_loader = create_dualseq_data_loader(dual_seqs,
                                                      text_tokenizer,
                                                      description_level=config.data.description_level,
                                                      batch_size=config.data.batch_size,
                                                      num_workers=config.data.num_workers,
                                                      val_ratio=0.0,
                                                      shuffle=True)
            val_loader = create_dualseq_data_loader(val_dual_seqs,
                                                    text_tokenizer,
                                                    description_level=config.data.description_level,
                                                    batch_size=config.data.batch_size,
                                                    num_workers=config.data.num_workers,
                                                    val_ratio=0.0,
                                                    shuffle=False)
        else:
            train_loader, val_loader = create_dualseq_data_loader(dual_seqs,
                                                                  text_tokenizer,
                                                                  description_level=config.data.description_level,
                                                                  batch_size=config.data.batch_size,
                                                                  num_workers=config.data.num_workers,
                                                                  val_ratio=config.data.eval_split_ratio,
                                                                  shuffle=True)
            
        # Model kwargs setup
        model_kwargs = {
            "vocab_size": get_dualseq_schema()["n_tokens"], 
            "n_args": get_dualseq_schema()["n_args"], 
            **config.model.kwargs
        }
        model_kwargs["cfg"] = config.model
        model = BaseModel(**model_kwargs)
        
        # Load local checkpoint if specified
        ex_input_ids = torch.zeros((1, 10), dtype=torch.int64)
        ex_input_mask = torch.ones((1, 10), dtype=torch.int64)
        
        before_model_out = model(ex_input_ids, ex_input_mask)
        if isinstance(before_model_out, tuple):
            before_model_out = before_model_out[0]

        load_dir = None
        save_root_has_checkpoint = False
        if hasattr(self, "SAVE_ROOT") and self.SAVE_ROOT is not None:
            for fname in ["encoder.pt", "adaptive_layer.pt", "checkpoint.pt"]:
                if os.path.exists(os.path.join(self.SAVE_ROOT, fname)):
                    save_root_has_checkpoint = True
                    break
        
        if save_root_has_checkpoint:
            load_dir = self.SAVE_ROOT
            print(f"Found checkpoint files in current training output directory ({self.SAVE_ROOT}). Loading from here instead of config path.")
        elif config.pretrained_path is not None:
            load_dir = config.pretrained_path

        if load_dir is not None:
            if os.path.isdir(load_dir):
                encoder_path = os.path.join(load_dir, "encoder.pt")
                adaptive_layer_path = os.path.join(load_dir, "adaptive_layer.pt")
                checkpoint_path = os.path.join(load_dir, "checkpoint.pt")
                
                if os.path.exists(encoder_path):
                    model.encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
                    print(f"Loaded encoder from {encoder_path}")
                if os.path.exists(adaptive_layer_path):
                    model.adaptive_layer.load_state_dict(torch.load(adaptive_layer_path, map_location="cpu"))
                    print(f"Loaded adaptive_layer from {adaptive_layer_path}")
                if os.path.exists(checkpoint_path):
                    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
                    print(f"Loaded full model checkpoint from {checkpoint_path}")
            else:
                state_dict = torch.load(load_dir, map_location="cpu")
                model.load_state_dict(state_dict)
                print(f"Loaded model checkpoint file from {load_dir}")
        
        after_model_out = model(ex_input_ids, ex_input_mask)
        if isinstance(after_model_out, tuple):
            after_model_out = after_model_out[0]

        if not torch.equal(before_model_out, after_model_out):
            print("Successfully loaded model checkpoint: diff on model output")
        else:
            print("Not loading model checkpoint: diff on model output is 0")
            
        # Init GRPO wrapper
        wrapper = GRPOWrapper(model, text_tokenizer)
        
        # Criterion loading
        if config.trainer.criterion.source == "local" and config.trainer.criterion.cls == "DualSeqCriterion":
            criterion = DualSeqCriterion(**config.trainer.criterion.kwargs)
        else:
            raise ValueError(f"Unsupported criterion: {config.trainer.criterion.cls}")
            
        # Optimizer loading
        if config.trainer.optimizer == "AdamW":
            optimizer_cls = torch.optim.AdamW
        else:
            raise ValueError(f"Unsupported optimizer: {config.trainer.optimizer}")
        optimizer = optimizer_cls(wrapper.parameters(), **config.trainer.optimizer_kwargs)
        
        # Scheduler loading
        scheduler = None
        if config.trainer.scheduler is not None:
            scheduler_cfg = config.trainer.scheduler
            total_steps = config.trainer.epochs * len(train_loader)
            warmup_steps = scheduler_cfg.warmup_steps
            if scheduler_cfg.warmup_ratio is not None:
                warmup_steps = int(scheduler_cfg.warmup_ratio * total_steps)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )

        # Load optimizer and scheduler state dicts from current training output directory (SAVEPATH root) if they exist
        optimizer_path = os.path.join(self.SAVE_ROOT, "optimizer.pt")
        scheduler_path = os.path.join(self.SAVE_ROOT, "scheduler.pt")
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
            print(f"Loaded optimizer state from {optimizer_path}")
        if scheduler is not None and os.path.exists(scheduler_path):
            scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu"))
            print(f"Loaded scheduler state from {scheduler_path}")
            
        # Initialize GRPOTrainer
        trainer_kwargs = {
            "eval_steps": config.trainer.eval_steps,
            "scheduler": scheduler,
            "n_rollouts": config.grpo.n_rollouts,
            "clip_eps": config.grpo.clip_eps,
            "min_cd": config.grpo.min_cd,
            "max_cd": config.grpo.max_cd,
            "eval_fraction": config.grpo.eval_fraction,
            "temperature": config.grpo.temperature,
            **config.trainer.kwargs
        }
        trainer = GRPOTrainer(
            wrapper,
            criterion,
            optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            save_folder=self.SAVE_ROOT,
            **trainer_kwargs
        )
        
        self.trainer = trainer
        self.wrapper = wrapper
        self.model = wrapper.model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.dual_seqs = dual_seqs

    def train_model(self, verbose_all: bool = False, do_render: bool = True):
        if not self.trainer or not self.train_loader:
            self.load_things()

        progression = self.trainer.train(self.cfg.trainer.epochs, verbose=verbose_all)
        self.progression = progression
        
        # inference test for ten random samples
        rand_idxs = torch.randperm(len(self.dual_seqs))[:10]
        results = []
        for i in rand_idxs.tolist():
            desc = self.dual_seqs[i].descriptions[self.cfg.data.description_level]
            gen_cmds = self.wrapper.generate(desc, max_new_tokens=self.cfg.trainer.max_new_cmds)
            results.append({
                "input": desc,
                "target_cmds": self.dual_seqs[i].cmds,
                "generated_cmds": gen_cmds
            })
            
        # save to csv
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.TEST_RESULT_PATH, index=False)

        # render generated cmds and args to images
        if do_render:
            from utils.render import render_dual_seq_to_img
            from tqdm import tqdm
            os.makedirs(self.RENDER_RESULTS_PATH, exist_ok=True)
            print(f"Rendering generated designs to {self.RENDER_RESULTS_PATH}...")
            for idx, res in enumerate(tqdm(results, desc="Rendering generated shapes")):
                gen_seq = res["generated_cmds"]
                desc = res["input"]
                
                gen_dual_seq = DualSeq.__new__(DualSeq)
                gen_dual_seq.uid = f"gen_{idx}"
                gen_dual_seq.cmds = [item[0] for item in gen_seq]
                gen_dual_seq.args = [item[1] for item in gen_seq]
                gen_dual_seq.descriptions = {self.cfg.data.description_level: desc}
                
                img_path = os.path.join(self.RENDER_RESULTS_PATH, f"sample_{idx}.png")
                try:
                    render_dual_seq_to_img(gen_dual_seq, img_path, with_str=True, with_desc=self.cfg.data.description_level)
                except Exception as e:
                    print(f"Error rendering generated sample {idx}: {e}")
        
        return progression
