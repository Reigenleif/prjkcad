import os
import random
from typing import Union, Dict, Any, Optional

import pandas as pd
import matplotlib.pyplot as plt
import torch
import wandb
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from utils.set_seed import set_seed
from utils.data_utils import load_split_data
from utils.dual_seq import DualSeq, get_dualseq_schema
from utils.data_utils import create_cmdonly_data_loader, create_dualseq_data_loader
from utils.wrapper import TokenizedArgsWrapper
from utils.criterion import TokenizedArgsCriterion
from utils.trainer import DualSeqTrainer, DualSeqCMDOnlyTrainer
from utils.render import render_dual_seq_to_img
        

from models import TokenizedArgsBaseModel
from utils.pipeline.config import Config


class TokenizedArgsFineTuningPipeline:
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
        
        USE_VAL = config.data.eval_split_ratio > 0
        self.SAVE_ROOT = f"out/{config.run_name}"
        os.makedirs(self.SAVE_ROOT, exist_ok=True)
        self.CHECKPOINT_SAVE_PATH = f"{self.SAVE_ROOT}/checkpoint.pt"
        self.RENDER_RESULTS_PATH = f"{self.SAVE_ROOT}/render_results"
        self.TEST_RESULT_PATH = f"{self.SAVE_ROOT}/test_result.csv"
        
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
            if config.data.is_cmdonly:
                train_loader = create_cmdonly_data_loader(dual_seqs, text_tokenizer, description_level=config.data.description_level, batch_size=config.data.batch_size, num_workers=config.data.num_workers, val_ratio=0.0, shuffle=True)
                val_loader = create_cmdonly_data_loader(val_dual_seqs, text_tokenizer, description_level=config.data.description_level, batch_size=config.data.batch_size, num_workers=config.data.num_workers, val_ratio=0.0, shuffle=False)
            else:
                train_loader = create_dualseq_data_loader(dual_seqs, text_tokenizer, description_level=config.data.description_level, batch_size=config.data.batch_size, num_workers=config.data.num_workers, val_ratio=0.0, shuffle=True)
                val_loader = create_dualseq_data_loader(val_dual_seqs, text_tokenizer, description_level=config.data.description_level, batch_size=config.data.batch_size, num_workers=config.data.num_workers, val_ratio=0.0, shuffle=False)
        else:
            if USE_VAL:
                if config.data.is_cmdonly:
                    train_loader, val_loader = create_cmdonly_data_loader(dual_seqs, text_tokenizer, description_level=config.data.description_level, batch_size=config.data.batch_size, num_workers=config.data.num_workers, val_ratio=config.data.eval_split_ratio, shuffle=True)
                else:
                    train_loader, val_loader = create_dualseq_data_loader(dual_seqs, text_tokenizer, description_level=config.data.description_level, batch_size=config.data.batch_size, num_workers=config.data.num_workers, val_ratio=config.data.eval_split_ratio, shuffle=True)
            else:
                if config.data.is_cmdonly:
                    train_loader = create_cmdonly_data_loader(dual_seqs, text_tokenizer, description_level=config.data.description_level, batch_size=config.data.batch_size, num_workers=config.data.num_workers, val_ratio=0.0, shuffle=True)
                else:
                    train_loader = create_dualseq_data_loader(dual_seqs, text_tokenizer, description_level=config.data.description_level, batch_size=config.data.batch_size, num_workers=config.data.num_workers, val_ratio=0.0, shuffle=True)
                val_loader = None
        
        model_cls = TokenizedArgsBaseModel
        schema = get_dualseq_schema()
        
        if config.data.is_cmdonly:
            model_kwargs = {"vocab_size": schema["cmd_n_tokens"], **config.model.kwargs}
        else:
            model_kwargs = {"vocab_size": schema["cmd_n_tokens"], "vocab_size_args": schema["args_n_tokens"], **config.model.kwargs}
            
        model_kwargs["cfg"] = config.model
            
        model = model_cls(**model_kwargs)
        
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
                    state_dict = torch.load(encoder_path, map_location="cpu")
                    model_state_dict = model.encoder.state_dict()
                    filtered = {k: v for k, v in state_dict.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}
                    model.encoder.load_state_dict(filtered, strict=False)
                    print(f"Loaded encoder from {encoder_path}")
                if os.path.exists(adaptive_layer_path):
                    state_dict = torch.load(adaptive_layer_path, map_location="cpu")
                    model_state_dict = model.adaptive_layer.state_dict()
                    filtered = {k: v for k, v in state_dict.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}
                    model.adaptive_layer.load_state_dict(filtered, strict=False)
                    print(f"Loaded adaptive_layer from {adaptive_layer_path}")
                
                if load_dir == self.SAVE_ROOT and os.path.exists(checkpoint_path):
                    state_dict = torch.load(checkpoint_path, map_location="cpu")
                    model_state_dict = model.state_dict()
                    filtered = {}
                    for k, v in state_dict.items():
                        if k in model_state_dict:
                            if v.shape == model_state_dict[k].shape:
                                filtered[k] = v
                            else:
                                print(f"Shape mismatch for key {k}: checkpoint shape {v.shape}, model shape {model_state_dict[k].shape}. Skipping.")
                        else:
                            print(f"Key {k} not in model. Skipping.")
                    model.load_state_dict(filtered, strict=False)
                    print(f"Loaded full model checkpoint from {checkpoint_path} (filtered)")
            else:
                state_dict = torch.load(load_dir, map_location="cpu")
                model_state_dict = model.state_dict()
                filtered = {}
                for k, v in state_dict.items():
                    if k in model_state_dict:
                        if v.shape == model_state_dict[k].shape:
                            filtered[k] = v
                        else:
                            print(f"Shape mismatch for key {k}: checkpoint shape {v.shape}, model shape {model_state_dict[k].shape}. Skipping.")
                    else:
                        print(f"Key {k} not in model. Skipping.")
                model.load_state_dict(filtered, strict=False)
                print(f"Loaded model checkpoint file from {load_dir} (filtered)")
        
        after_model_out = model(ex_input_ids, ex_input_mask)
        if isinstance(after_model_out, tuple):
            after_model_out = after_model_out[0]

        if not torch.equal(before_model_out, after_model_out):
            print("Successfully loaded model checkpoint: diff on model output")
        else:
            print("Not loading model checkpoint: diff on model output is 0")
 
        if config.data.is_cmdonly:
            from utils.legacy.wrapper.dual_seq_cmdonly import DualSeqCMDOnlyWrapper
            wrapper = DualSeqCMDOnlyWrapper(model, text_tokenizer)
        else:
            wrapper = TokenizedArgsWrapper(model, text_tokenizer)
            
        if config.trainer.criterion.source == "local":
            if config.trainer.criterion.cls == "DualSeqCMDOOnlyCriterion":
                from utils.legacy.criterion.dual_seq_cmdonly_criterion import DualSeqCMDOnlyCriterion
                criterion_cls = DualSeqCMDOnlyCriterion
            elif config.trainer.criterion.cls == "DualSeqCriterion":
                from utils.criterion import DualSeqCriterion
                criterion_cls = DualSeqCriterion
            elif config.trainer.criterion.cls == "TokenizedArgsCriterion":
                criterion_cls = TokenizedArgsCriterion
            else:
                raise ValueError(f"Unsupported criterion class: {config.trainer.criterion.cls}")
        else:
            raise ValueError(f"Unsupported criterion source: {config.trainer.criterion.source}")
        
        criterion = criterion_cls(**config.trainer.criterion.kwargs)
        
        if config.trainer.optimizer == "AdamW":
            optimizer_cls = torch.optim.AdamW
        else:
            raise ValueError(f"Unsupported optimizer: {config.trainer.optimizer}")
        optimizer = optimizer_cls(wrapper.parameters(), **config.trainer.optimizer_kwargs)
        
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

        optimizer_path = os.path.join(self.SAVE_ROOT, "optimizer.pt")
        scheduler_path = os.path.join(self.SAVE_ROOT, "scheduler.pt")
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
            print(f"Loaded optimizer state from {optimizer_path}")
        if scheduler is not None and os.path.exists(scheduler_path):
            scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu"))
            print(f"Loaded scheduler state from {scheduler_path}")

        trainer_kwargs = {
            "eval_steps": config.trainer.eval_steps,
            "scheduler": scheduler,
            **config.trainer.kwargs
        }
        if config.data.is_cmdonly:
            trainer = DualSeqCMDOnlyTrainer(
                wrapper,
                criterion,
                optimizer,
                train_loader=train_loader,
                val_loader=val_loader,
                save_folder=self.SAVE_ROOT,
                **trainer_kwargs
            )
        else:
            trainer = DualSeqTrainer(
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

    def train_model(self, verbose_all: bool = False, log_artifacts: bool = False):
        if not self.trainer or not self.train_loader:
            self.load_things()

        wandb_api_key = os.environ.get("WANDB_API_KEY")
        wandb_project = os.environ.get("WANDB_PROJECT")
        if wandb_api_key and wandb_project:
            
            wandb.login(key=wandb_api_key)
            config_dict = self.cfg.to_dict() if hasattr(self.cfg, "to_dict") else (self.cfg.__dict__ if hasattr(self.cfg, "__dict__") else {})
            wandb.init(
                project=wandb_project,
                name=self.cfg.run_name,
                config=config_dict,
                reinit=True
            )
            if self.trainer is not None:
                self.trainer.log_artifacts = log_artifacts
                wandb.watch(self.trainer.wrapper, log="gradients", log_freq=self.trainer.eval_steps)

        try:
            progression = self.trainer.train(self.cfg.trainer.epochs, verbose=verbose_all)
            self.progression = progression
        except Exception as e :
            print(f"Err: {e}")
        
        finally:
            if wandb.run:
                wandb.finish()
        
        rand_idxs = torch.randperm(len(self.dual_seqs))[:10]
        results = []
        for i in rand_idxs.tolist():
            desc = self.dual_seqs[i].descriptions[self.cfg.data.description_level]
            gen_cmds_args = self.wrapper.generate(desc, max_new_tokens=self.wrapper.max_new_cmds)
            results.append({
                "input": desc,
                "target_cmds": self.dual_seqs[i].cmds,
                "generated_cmds": gen_cmds_args
            })
            
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.TEST_RESULT_PATH, index=False)
        
        return progression

    def plot_progression(self):
        if not self.progression:
            raise ValueError("No training progression found. Please train the model first.")

        config = self.cfg
        progression = self.progression
        out_path = f"out/{config.run_name}/progression.png"

        vp = {
            "LINE_precision":    [h.get("val_LINE_precision",    0) for h in progression],
            "LINE_recall":       [h.get("val_LINE_recall",       0) for h in progression],
            "LINE_f1":           [h.get("val_LINE_f1",           0) for h in progression],
            "CIRCLE_precision":  [h.get("val_CIRCLE_precision",  0) for h in progression],
            "CIRCLE_recall":     [h.get("val_CIRCLE_recall",     0) for h in progression],
            "CIRCLE_f1":         [h.get("val_CIRCLE_f1",         0) for h in progression],
            "ARC_precision":     [h.get("val_ARC_precision",     0) for h in progression],
            "ARC_recall":        [h.get("val_ARC_recall",        0) for h in progression],
            "ARC_f1":            [h.get("val_ARC_f1",            0) for h in progression],
            "EXTRUDE_precision": [h.get("val_EXTRUDE_precision", 0) for h in progression],
            "EXTRUDE_recall":    [h.get("val_EXTRUDE_recall",    0) for h in progression],
            "EXTRUDE_f1":        [h.get("val_EXTRUDE_f1",        0) for h in progression],
        }
        
        is_full = not config.data.is_cmdonly
        has_shape_metrics = False

        if is_full:
            vp["val_arg_float_r2"] = [h.get("val_arg_float_r2", h.get("val_arg_r2", 0)) for h in progression]
            vp["val_arg_float_mse"] = [h.get("val_arg_float_mse", 0) for h in progression]
            vp["val_arg_token_f1"] = [h.get("val_arg_token_f1", 0) for h in progression]
            vp["val_arg_sep_count_mse"] = [h.get("val_arg_sep_count_mse", 0) for h in progression]
            vp["val_ir"] = [h.get("val_total_invalidity_ratio", h.get("val_ir")) for h in progression]
            vp["val_cd"] = [h.get("val_cd") for h in progression]
            has_ir = any(v is not None for v in vp["val_ir"])
            has_cd = any(v is not None for v in vp["val_cd"])
            has_shape_metrics = has_ir or has_cd

        train_loss  = [h["train_loss"]      for h in progression]
        val_loss    = [h["val_loss"]        for h in progression]
        val_perp    = [h["val_perplexity"]  for h in progression]
        epochs      = list(range(1, len(progression) + 1))
        avg_f1      = [(vp["LINE_f1"][i] + vp["CIRCLE_f1"][i] + vp["ARC_f1"][i]) / 3
                       for i in range(len(epochs))]
        lr_history  = [h.get("lr", 0.0)     for h in progression]
        grad_norm_history = [h.get("grad_norm", 0.0) for h in progression]

        n_rows = 6 + int(is_full)
        fig = plt.figure(figsize=(12, 3.0 * n_rows))
        gs  = fig.add_gridspec(n_rows, 2, hspace=0.50, wspace=0.30)

        C = {
            "train":    "#4C72B0",
            "val":      "#DD8452",
            "prec":     "#1F77B4",
            "rec":      "#FF7F0E",
            "f1":       "#2CA02C",
            "avg_f1":   "#C44E52",
            "ir":       "#9467BD",
            "cd":       "#8C564B",
            "arg_r2":   "#937860",
            "arg_mape": "#DA8BC3",
            "arg_f1":   "#17BECF",
            "arg_mse":  "#BCBD22",
            "arg_sep":  "#E377C2",
        }

        def _style(ax, title, ylabel=None, ylim=None):
            ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
            ax.set_xlabel("Epoch", fontsize=9)
            if ylabel:
                ax.set_ylabel(ylabel, fontsize=9)
            if ylim:
                ax.set_ylim(*ylim)
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
            ax.tick_params(labelsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            leg = ax.legend(fontsize=8, framealpha=0.7, loc="best")
            if leg:
                leg.get_frame().set_linewidth(0.5)

        def _plot_prf(ax, key, title):
            ax.plot(epochs, vp[f"{key}_precision"], color=C["prec"], label="Precision",
                    linewidth=1.2, linestyle="--")
            ax.plot(epochs, vp[f"{key}_recall"],    color=C["rec"],  label="Recall",
                    linewidth=1.2, linestyle=":")
            ax.plot(epochs, vp[f"{key}_f1"],        color=C["f1"],   label="F1",
                    linewidth=1.5)
            _style(ax, title, ylabel="Score", ylim=(0, 1))

        # Row 0 – Loss (full-width)
        ax_loss = fig.add_subplot(gs[0, :])
        ax_loss.plot(epochs, train_loss, color=C["train"], label="Train", linewidth=1.5)
        ax_loss.plot(epochs, val_loss,   color=C["val"],   label="Val",   linewidth=1.5, linestyle="--")
        _style(ax_loss, "Loss", ylabel="Loss")

        # Row 1 – LR and Gradient Norm (full-width)
        ax_lr = fig.add_subplot(gs[1, :])
        ax_lr.plot(epochs, lr_history, color="#1F77B4", label="Learning Rate", linewidth=1.5)
        ax_lr.set_title("Learning Rate & Gradient Norm", fontsize=11, fontweight="bold", pad=6)
        ax_lr.set_xlabel("Epoch", fontsize=9)
        ax_lr.set_ylabel("Learning Rate", fontsize=9, color="#1F77B4")
        ax_lr.tick_params(axis="y", colors="#1F77B4", labelsize=8)
        ax_lr.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
        ax_lr.spines[["top", "right"]].set_visible(False)

        ax_gn = ax_lr.twinx()
        ax_gn.plot(epochs, grad_norm_history, color="#DD8452", label="Gradient Norm", linewidth=1.5, linestyle="--")
        ax_gn.set_ylabel("Gradient Norm", fontsize=9, color="#DD8452")
        ax_gn.tick_params(axis="y", colors="#DD8452", labelsize=8)
        ax_gn.spines[["top"]].set_visible(False)
        
        lines_lr, labels_lr = ax_lr.get_legend_handles_labels()
        lines_gn, labels_gn = ax_gn.get_legend_handles_labels()
        ax_lr.legend(lines_lr + lines_gn, labels_lr + labels_gn, fontsize=8, framealpha=0.7, loc="best")

        # Row 2 – Perplexity | Shape metrics
        if is_full:
            ax_perp  = fig.add_subplot(gs[2, 0])
            ax_shape = fig.add_subplot(gs[2, 1])
            ax_perp.plot(epochs, val_perp, color=C["val"], linewidth=1.5, label="Val Perplexity")
            _style(ax_perp, "Validation Perplexity", ylabel="Perplexity")

            if has_shape_metrics:
                shape_lines, shape_labels = [], []
                if has_ir:
                    ir_vals = [v for v in vp["val_ir"]]
                    ir_plot = [v if v is not None else float("nan") for v in ir_vals]
                    ax_shape.plot(epochs, ir_plot, color=C["ir"], label="IR ↓", linewidth=1.5)
                    ax_shape.set_ylabel("IR", fontsize=9, color=C["ir"])
                    ax_shape.tick_params(axis="y", colors=C["ir"], labelsize=8)
                    shape_lines += ax_shape.get_legend_handles_labels()[0]
                    shape_labels += ax_shape.get_legend_handles_labels()[1]
                if has_cd:
                    cd_vals = [v for v in vp["val_cd"]]
                    cd_plot = [v if v is not None else float("nan") for v in cd_vals]
                    ax_cd = ax_shape.twinx()
                    ax_cd.plot(epochs, cd_plot, color=C["cd"], label="CD ↓", linewidth=1.5, linestyle="--")
                    ax_cd.set_ylabel("CD", fontsize=9, color=C["cd"])
                    ax_cd.tick_params(axis="y", colors=C["cd"], labelsize=8)
                    ax_cd.spines[["top"]].set_visible(False)
                    shape_lines += ax_cd.get_legend_handles_labels()[0]
                    shape_labels += ax_cd.get_legend_handles_labels()[1]
                ax_shape.legend(shape_lines, shape_labels, fontsize=8, framealpha=0.7, loc="best")
                ax_shape.set_title("Shape Metrics (IR / CD)", fontsize=11, fontweight="bold", pad=6)
                ax_shape.set_xlabel("Epoch", fontsize=9)
                ax_shape.set_ylim(0, 1)
                ax_shape.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
                ax_shape.spines[["top", "right"]].set_visible(False)
            else:
                ax_shape.axis("off")
                ax_shape.text(0.5, 0.5, "Shape metrics\n(val_ir / val_cd)\nnot yet available",
                               ha="center", va="center", fontsize=9, color="gray",
                               transform=ax_shape.transAxes)
        else:
            ax_perp = fig.add_subplot(gs[2, :])
            ax_perp.plot(epochs, val_perp, color=C["val"], linewidth=1.5, label="Val Perplexity")
            _style(ax_perp, "Validation Perplexity", ylabel="Perplexity")

        # Row 3 – LINE | CIRCLE
        _plot_prf(fig.add_subplot(gs[3, 0]), "LINE",   "LINE Metrics")
        _plot_prf(fig.add_subplot(gs[3, 1]), "CIRCLE", "CIRCLE Metrics")

        # Row 4 – ARC | EXTRUDE 
        _plot_prf(fig.add_subplot(gs[4, 0]), "ARC",     "ARC Metrics")
        _plot_prf(fig.add_subplot(gs[4, 1]), "EXTRUDE", "EXTRUDE Metrics\n(extrude-only filtered)")

        # Row 5 – Avg F1 (full-width)
        ax_avg = fig.add_subplot(gs[5, :])
        ax_avg.plot(epochs, avg_f1, color=C["avg_f1"], linewidth=2.0,
                    label="Avg F1 (LINE + CIRCLE + ARC)")
        _style(ax_avg, "Average F1 (Sketch Tokens)", ylabel="F1", ylim=(0, 1))

        # Row 6 – Argument Metrics (full-width, cmd+args only) 
        if is_full:
            ax_args = fig.add_subplot(gs[6, :])
            ax_args.plot(epochs, vp["val_arg_float_r2"], color=C["arg_r2"], label="Arg Float R²", linewidth=1.5)
            ax_args.plot(epochs, vp["val_arg_token_f1"], color=C["arg_f1"], label="Arg Token F1", linewidth=1.5)
            ax_args.set_ylabel("R² / F1", fontsize=9)
            ax_args.tick_params(axis="y", labelsize=8)
            ax_args.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
            ax_args.spines[["top", "right"]].set_visible(False)
            ax_args.set_xlabel("Epoch", fontsize=9)
            ax_args.set_title("Argument Metrics", fontsize=11, fontweight="bold", pad=6)
            ax_args.set_ylim(-0.1, 1.1)

            ax_sec = ax_args.twinx()
            ax_sec.plot(epochs, vp["val_arg_float_mse"], color=C["arg_mse"],
                         label="Arg Float MSE", linewidth=1.5, linestyle="--")
            ax_sec.plot(epochs, vp["val_arg_sep_count_mse"], color=C["arg_sep"],
                         label="Arg Sep Count MSE", linewidth=1.5, linestyle=":")
            ax_sec.set_ylabel("MSE", fontsize=9)
            ax_sec.tick_params(axis="y", labelsize=8)
            ax_sec.spines[["top"]].set_visible(False)

            lines1, labels1 = ax_args.get_legend_handles_labels()
            lines2, labels2 = ax_sec.get_legend_handles_labels()
            ax_args.legend(lines1 + lines2, labels1 + labels2, fontsize=8, framealpha=0.7, loc="best")

        fig.suptitle(f"Training Progression: {config.run_name}", fontsize=13, fontweight="bold")
        if out_path is not None:
            fig.savefig(out_path, bbox_inches="tight", dpi=120)
        plt.close(fig)

    def render_val_set(self, num_samples: int = 10, save_dir: str = None):
        if not self.wrapper:
            self.load_things()
            
        self.wrapper.eval()
        device = next(self.wrapper.parameters()).device
        self.wrapper.to(device)
        
        val_seqs = self.val_dual_seqs
        if not val_seqs:
            print("No validation set available to render.")
            return
            
        if save_dir is None:
            save_dir = os.path.join(self.SAVE_ROOT, "val_render_results")
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Rendering {min(num_samples, len(val_seqs))} validation set designs to {save_dir}...")
        
        for idx, dual_seq in enumerate(tqdm(val_seqs[:num_samples], desc="Rendering Validation Set")):
            desc = dual_seq.descriptions[self.cfg.data.description_level]
            
            try:
                pred_seq = self.wrapper.generate(desc, max_new_tokens=self.wrapper.max_new_cmds)
            except Exception as e:
                print(f"Error generating sample {idx}: {e}")
                continue
                
            gen_dual_seq = DualSeq.__new__(DualSeq)
            gen_dual_seq.uid = f"val_gen_{idx}"
            gen_dual_seq.cmds = [item[0] for item in pred_seq]
            gen_dual_seq.args_dict = [item[1] for item in pred_seq]
            gen_dual_seq.descriptions = {self.cfg.data.description_level: desc}
            gen_dual_seq.json_object = {"parts": {}}
            
            img_path = os.path.join(save_dir, f"val_sample_{idx}.png")
            try:
                render_dual_seq_to_img(gen_dual_seq, img_path, with_str=True, with_desc=self.cfg.data.description_level)
            except Exception as e:
                print(f"Error rendering sample {idx}: {e}")
