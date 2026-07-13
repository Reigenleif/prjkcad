import os
from typing import Union, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from utils.set_seed import set_seed
from utils.data_utils import load_split_data, create_pretrain_data_loader
from utils.dual_seq import get_dualseq_schema
from utils.wrapper import PretrainWrapper
from utils.criterion import PretrainCriterion
from utils.trainer import PretrainTrainer

from models import BaseModel
from utils.pipeline.config import Config


class PretrainPipeline:
    """
    Autoencoder Pretraining Pipeline
    """
    def __init__(self, cfg: Union[Config, Dict[str, Any], str]):
        if isinstance(cfg, str):
            from .train_model import load_config
            cfg = load_config(cfg)
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
        self.SAVE_ROOT = f"out/{config.run_name}"
        os.makedirs(self.SAVE_ROOT, exist_ok=True)

        set_seed(config.random_seed)

        if config.tokenizer.source == "huggingface":
            text_tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.model_name)
        else:
            raise ValueError(f"Unsupported tokenizer source: {config.tokenizer.source}")

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
            train_loader = create_pretrain_data_loader(
                self.dual_seqs,
                tokenizer=text_tokenizer,
                description_level=config.data.description_level,
                batch_size=config.data.batch_size,
                num_workers=config.data.num_workers,
                val_ratio=0.0,
                shuffle=True
            )
            val_loader = create_pretrain_data_loader(
                val_dual_seqs,
                tokenizer=text_tokenizer,
                description_level=config.data.description_level,
                batch_size=config.data.batch_size,
                num_workers=config.data.num_workers,
                val_ratio=0.0,
                shuffle=False
            )
        else:
            if config.data.eval_split_ratio > 0:
                train_loader, val_loader = create_pretrain_data_loader(
                    self.dual_seqs,
                    tokenizer=text_tokenizer,
                    description_level=config.data.description_level,
                    batch_size=config.data.batch_size,
                    num_workers=config.data.num_workers,
                    val_ratio=config.data.eval_split_ratio,
                    shuffle=True
                )
            else:
                train_loader = create_pretrain_data_loader(
                    self.dual_seqs,
                    tokenizer=text_tokenizer,
                    description_level=config.data.description_level,
                    batch_size=config.data.batch_size,
                    num_workers=config.data.num_workers,
                    val_ratio=0.0,
                    shuffle=True
                )
                val_loader = None

        n_args = get_dualseq_schema()["n_args"]
        base_model = BaseModel(
            cfg=config.model,
            vocab_size=len(text_tokenizer),
            n_args=n_args
        )

        wrapper = PretrainWrapper(
            model=base_model,
            text_tokenizer=text_tokenizer,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load local checkpoint if specified
        ex_input_ids = torch.zeros((1, 10), dtype=torch.int64).to(wrapper.device)
        ex_input_mask = torch.ones((1, 10), dtype=torch.int64).to(wrapper.device)
        
        before_model_out = wrapper.reconstructor(ex_input_ids, ex_input_mask)
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
                    base_model.encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
                    print(f"Loaded encoder from {encoder_path}")
                if os.path.exists(adaptive_layer_path):
                    base_model.adaptive_layer.load_state_dict(torch.load(adaptive_layer_path, map_location="cpu"))
                    print(f"Loaded adaptive_layer from {adaptive_layer_path}")
                if os.path.exists(checkpoint_path):
                    wrapper.reconstructor.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
                    print(f"Loaded checkpoint (reconstructor) from {checkpoint_path}")
            else:
                state_dict = torch.load(load_dir, map_location="cpu")
                first_key = next(iter(state_dict.keys()))
                if first_key.startswith("reconstructor."):
                    wrapper.load_state_dict(state_dict)
                else:
                    base_model.load_state_dict(state_dict)
                print(f"Loaded model checkpoint file from {load_dir}")
        
        after_model_out = wrapper.reconstructor(ex_input_ids, ex_input_mask)
        if isinstance(after_model_out, tuple):
            after_model_out = after_model_out[0]

        if not torch.equal(before_model_out, after_model_out):
            print("Successfully loaded model checkpoint: diff on model output")
        else:
            print("Not loading model checkpoint: diff on model output is 0")

        kl_weight = config.trainer.criterion.kwargs.get("kl_weight", 1.0)
        criterion = PretrainCriterion(
            pad_id=text_tokenizer.pad_token_id or 0,
            kl_weight=kl_weight
        )

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

        trainer = PretrainTrainer(
            model_wrapper=wrapper,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=wrapper.device,
            max_grad_norm=config.trainer.kwargs.get("max_grad_norm", 1.0),
            save_folder=self.SAVE_ROOT,
            best_metric_key="val_f1",
            best_metric_mode="max",
            eval_steps=config.trainer.eval_steps,
            scheduler=scheduler
        )

        self.trainer = trainer
        self.wrapper = wrapper
        self.model = base_model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train_model(self, verbose_all: bool = False, do_render: bool = False):
        if not self.trainer or not self.train_loader:
            self.load_things()

        wandb_api_key = os.environ.get("WANDB_API_KEY")
        wandb_project = os.environ.get("WANDB_PROJECT")
        if wandb_api_key and wandb_project:
            import wandb
            wandb.login(key=wandb_api_key)
            config_dict = self.cfg.to_dict() if hasattr(self.cfg, "to_dict") else (self.cfg.__dict__ if hasattr(self.cfg, "__dict__") else {})
            wandb.init(
                project=wandb_project,
                name=self.cfg.run_name,
                config=config_dict,
                reinit=True
            )
            if self.trainer is not None:
                wandb.watch(self.trainer.wrapper, log="gradients", log_freq=self.trainer.eval_steps)

        try:
            progression = self.trainer.train(self.cfg.trainer.epochs, verbose=verbose_all)
            self.progression = progression
            return progression
        finally:
            import wandb
            if wandb.run:
                wandb.finish()

