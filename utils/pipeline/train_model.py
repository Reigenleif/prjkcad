import os
from typing import Union, Dict, Any, Optional
import yaml
import random

import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from utils.set_seed import set_seed
from utils.data_utils import RefLoader
from utils.dual_seq import DualSeq, get_dualseq_schema
from utils.data_utils import create_cmdonly_data_loader, create_dualseq_data_loader
from utils.wrapper import DualSeqCMDOnlyWrapper
from utils.wrapper import DualSeqWrapper
from utils.criterion import DualSeqCMDOnlyCriterion
from utils.criterion import DualSeqCriterion
from utils.trainer import DualSeqCMDOnlyTrainer
from utils.trainer import DualSeqTrainer

from models import BaseModel

from utils.pipeline.config import Config


def load_config(config_path: str) -> Config:
    """
    Load config from a YAML file.
    
    Args:
        config_path: Path to the config file.
        
    Returns:
        Config object.
    """
    return Config.from_yaml(config_path)


class TrainModelPipeline:
    """
    End-to-end training pipeline

    Args:
        cfg: Config object, dict or path to config file.
    """

    def __init__(self, cfg: Union[Config, Dict[str, Any], str]):
        if isinstance(cfg, str):
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
    
    def load_things(self):
        config = self.cfg
        
        USE_VAL = config.data.eval_split_ratio > 0
        self.SAVE_ROOT = f"out/{config.run_name}"
        os.makedirs(self.SAVE_ROOT, exist_ok=True)
        self.CHECKPOINT_SAVE_PATH = f"{self.SAVE_ROOT}/checkpoint.pt"
        self.RENDER_RESULTS_PATH = f"{self.SAVE_ROOT}/render_results"
        self.TEST_RESULT_PATH = f"{self.SAVE_ROOT}/test_result.csv"
        
        # Set random seed for reproducibility
        set_seed(config.random_seed)
        
        # Tokenizer loading
        if config.tokenizer.source == "huggingface":
            text_tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.model_name)
        else:
            raise ValueError(f"Unsupported tokenizer source: {config.tokenizer.source}")
        
        # Raw data loading
        if self.dual_seqs is None :
            loader = RefLoader(config.data.data_root,
                            max_samples=config.data.max_samples,
                            source_data_type=config.data.source_data_type)
            df = loader.load()
            
            dual_seqs = DualSeq.from_text2cad_df(df)
        
            if config.data.sample_ratio:
                sample_size = int(len(dual_seqs) * config.data.sample_ratio)
                dual_seqs = random.sample(dual_seqs, sample_size)
                print(f"Sampled {sample_size} instances from the dataset based on the specified sample ratio of {config.data.sample_ratio}.")

            self.dual_seqs = dual_seqs 
        
        dual_seqs = self.dual_seqs

        if USE_VAL:
            if config.data.is_cmdonly:
                train_loader, val_loader = create_cmdonly_data_loader(dual_seqs,
                                                                    text_tokenizer,
                                                                    description_level=config.data.description_level,
                                                                    batch_size=config.data.batch_size,
                                                                    num_workers=config.data.num_workers,
                                                                    val_ratio=config.data.eval_split_ratio,
                                                                    shuffle=True)
            else:
                train_loader, val_loader = create_dualseq_data_loader(dual_seqs,
                                                            text_tokenizer,
                                                            description_level=config.data.description_level,
                                                            batch_size=config.data.batch_size,
                                                            num_workers=config.data.num_workers,
                                                            val_ratio=config.data.eval_split_ratio,
                                                            shuffle=True)
        else:
            if config.data.is_cmdonly:
                train_loader = create_cmdonly_data_loader(dual_seqs,
                                                        text_tokenizer,
                                                        description_level=config.data.description_level,
                                                        batch_size=config.data.batch_size,
                                                        num_workers=config.data.num_workers,
                                                        val_ratio=0.0,
                                                        shuffle=True)
            else:
                train_loader = create_dualseq_data_loader(dual_seqs,
                                                        text_tokenizer,
                                                        description_level=config.data.description_level,
                                                        batch_size=config.data.batch_size,
                                                        num_workers=config.data.num_workers,
                                                        val_ratio=0.0,
                                                        shuffle=True)
            val_loader = None
        
        # Wrapper and Model loading
        model_cls = BaseModel
        
        if config.data.is_cmdonly:
            model_kwargs = {"vocab_size": get_dualseq_schema()["n_tokens"], 
                            **config.model.kwargs}
        else:
            model_kwargs = {"vocab_size": get_dualseq_schema()["n_tokens"], 
                            "n_args": get_dualseq_schema()["n_args"], 
                            **config.model.kwargs}
            
        model_kwargs["cfg"] = config.model
            
        # Initialize model locally
        model = model_cls(**model_kwargs)
        
        # Load local checkpoint if specified
        if config.pretrained_path is not None:
            state_dict = torch.load(config.pretrained_path, map_location="cpu")
            model.load_state_dict(state_dict)
            print(f"Loaded model checkpoint from {config.pretrained_path}")
            
        # Init wrapper
        if config.data.is_cmdonly:
            wrapper = DualSeqCMDOnlyWrapper(model, text_tokenizer)
        else:
            wrapper = DualSeqWrapper(model, text_tokenizer)
            
        # Criterion loading
        if config.trainer.criterion.source == "local":
            if config.trainer.criterion.cls == "DualSeqCMDOOnlyCriterion":
                criterion_cls = DualSeqCMDOnlyCriterion
            elif config.trainer.criterion.cls == "DualSeqCriterion":
                criterion_cls = DualSeqCriterion
            else:
                raise ValueError(f"Unsupported criterion class: {config.trainer.criterion.cls}")
        else:
            raise ValueError(f"Unsupported criterion source: {config.trainer.criterion.source}")
        
        criterion = criterion_cls(**config.trainer.criterion.kwargs)
        
        # Optimizer loading
        if config.trainer.optimizer == "AdamW":
            optimizer_cls = torch.optim.AdamW
        else:
            raise ValueError(f"Unsupported optimizer: {config.trainer.optimizer}")
        optimizer = optimizer_cls(wrapper.parameters(), **config.trainer.optimizer_kwargs)
        
        # Trainer initialization
        if config.data.is_cmdonly:
            trainer = DualSeqCMDOnlyTrainer(
                wrapper,
                criterion,
                optimizer,
                train_loader=train_loader,
                val_loader=val_loader,
                save_folder=self.SAVE_ROOT,
                **config.trainer.kwargs
            )
        else:
            trainer = DualSeqTrainer(
                wrapper,
                criterion,
                optimizer,
                train_loader=train_loader,
                val_loader=val_loader,
                save_folder=self.SAVE_ROOT,
                **config.trainer.kwargs
            )  
        
        # Keep all training objects in the pipeline
        self.trainer = trainer
        self.wrapper = wrapper
        self.model = wrapper.model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.dual_seqs = dual_seqs

    def train_model(self, verbose_all: bool = False, do_render: bool = True):
        """
        End-to-end function to train the model from model and data loading, training loop, evaluation, and saving.
        """
        
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
        if do_render and not self.cfg.model.is_cmd_only:
            from utils.render import render_dual_seq_to_img
            from tqdm import tqdm
            os.makedirs(self.RENDER_RESULTS_PATH, exist_ok=True)
            print(f"Rendering generated designs to {self.RENDER_RESULTS_PATH}...")
            for idx, res in enumerate(tqdm(results, desc="Rendering generated shapes")):
                gen_seq = res["generated_cmds"]
                desc = res["input"]
                
                # Construct a DualSeq
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

    def plot_progression(self):
        """
        Plot the training progression/history.

        Layout (2-column grid):
            Row 0  [full-width] : Loss (Train & Val)
            Row 1               : Val Perplexity  |  Shape metrics (IR / CD) [if not cmd-only]
            Row 2               : LINE P/R/F1     |  CIRCLE P/R/F1
            Row 3               : ARC P/R/F1      |  EXTRUDE P/R/F1 (extrude-filtered)
            Row 4  [full-width] : Average F1 (sketch tokens)
            Row 5  [full-width] : Arg Regression R² / MAPE  (cmd+args only)
        """
        if not self.progression:
            raise ValueError("No training progression found. Please train the model first.")

        config = self.cfg
        progression = self.progression
        out_path = f"out/{config.run_name}/progression.png"

        # ── Build data arrays ─────────────────────────────────────────────────
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
            vp["val_arg_r2"]   = [h.get("val_arg_r2",   0) for h in progression]
            vp["val_arg_mape"] = [h.get("val_arg_mape", 0) for h in progression]
            vp["val_ir"]       = [h.get("val_ir") for h in progression]
            vp["val_cd"]       = [h.get("val_cd") for h in progression]
            has_ir = any(v is not None for v in vp["val_ir"])
            has_cd = any(v is not None for v in vp["val_cd"])
            has_shape_metrics = has_ir or has_cd

        train_loss  = [h["train_loss"]      for h in progression]
        val_loss    = [h["val_loss"]        for h in progression]
        val_perp    = [h["val_perplexity"]  for h in progression]
        epochs      = list(range(1, len(progression) + 1))
        avg_f1      = [(vp["LINE_f1"][i] + vp["CIRCLE_f1"][i] + vp["ARC_f1"][i]) / 3
                       for i in range(len(epochs))]

        # ── Grid layout ───────────────────────────────────────────────────────
        n_rows = 5 + int(is_full)
        fig = plt.figure(figsize=(12, 3.0 * n_rows))
        gs  = fig.add_gridspec(n_rows, 2, hspace=0.50, wspace=0.30)

        # ── Color palette ─────────────────────────────────────────────────────
        C = {
            "train":    "#4C72B0",
            "val":      "#DD8452",
            "prec":     "#1F77B4",   # blue   – Precision
            "rec":      "#FF7F0E",   # orange – Recall
            "f1":       "#2CA02C",   # green  – F1
            "avg_f1":   "#C44E52",   # red    – Avg F1
            "ir":       "#9467BD",   # purple – Invalidity Rate
            "cd":       "#8C564B",   # brown  – Chamfer Distance
            "arg_r2":   "#937860",
            "arg_mape": "#DA8BC3",
        }

        # ── Helpers ───────────────────────────────────────────────────────────
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
            """Plot Precision (dashed), Recall (dotted), F1 (solid) for `key`."""
            ax.plot(epochs, vp[f"{key}_precision"], color=C["prec"], label="Precision",
                    linewidth=1.2, linestyle="--")
            ax.plot(epochs, vp[f"{key}_recall"],    color=C["rec"],  label="Recall",
                    linewidth=1.2, linestyle=":")
            ax.plot(epochs, vp[f"{key}_f1"],        color=C["f1"],   label="F1",
                    linewidth=1.5)
            _style(ax, title, ylabel="Score", ylim=(0, 1))

        # ── Row 0 – Loss (full-width) ─────────────────────────────────────────
        ax_loss = fig.add_subplot(gs[0, :])
        ax_loss.plot(epochs, train_loss, color=C["train"], label="Train", linewidth=1.5)
        ax_loss.plot(epochs, val_loss,   color=C["val"],   label="Val",   linewidth=1.5, linestyle="--")
        _style(ax_loss, "Loss", ylabel="Loss")

        # ── Row 1 – Perplexity | Shape metrics ───────────────────────────────
        if is_full:
            ax_perp  = fig.add_subplot(gs[1, 0])
            ax_shape = fig.add_subplot(gs[1, 1])
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
            ax_perp = fig.add_subplot(gs[1, :])
            ax_perp.plot(epochs, val_perp, color=C["val"], linewidth=1.5, label="Val Perplexity")
            _style(ax_perp, "Validation Perplexity", ylabel="Perplexity")

        # ── Row 2 – LINE | CIRCLE ─────────────────────────────────────────────
        _plot_prf(fig.add_subplot(gs[2, 0]), "LINE",   "LINE Metrics")
        _plot_prf(fig.add_subplot(gs[2, 1]), "CIRCLE", "CIRCLE Metrics")

        # ── Row 3 – ARC | EXTRUDE ────────────────────────────────────────────
        _plot_prf(fig.add_subplot(gs[3, 0]), "ARC",     "ARC Metrics")
        _plot_prf(fig.add_subplot(gs[3, 1]), "EXTRUDE", "EXTRUDE Metrics\n(extrude-only filtered)")

        # ── Row 4 – Avg F1 (full-width) ───────────────────────────────────────
        ax_avg = fig.add_subplot(gs[4, :])
        ax_avg.plot(epochs, avg_f1, color=C["avg_f1"], linewidth=2.0,
                    label="Avg F1 (LINE + CIRCLE + ARC)")
        _style(ax_avg, "Average F1 (Sketch Tokens)", ylabel="F1", ylim=(0, 1))

        # ── Row 5 – Arg Regression (full-width, cmd+args only) ───────────────
        if is_full:
            ax_args = fig.add_subplot(gs[5, :])
            ax_args.plot(epochs, vp["val_arg_r2"], color=C["arg_r2"], label="Arg R²", linewidth=1.5)
            ax_args.set_ylabel("R²", fontsize=9, color=C["arg_r2"])
            ax_args.tick_params(axis="y", colors=C["arg_r2"], labelsize=8)
            ax_args.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
            ax_args.spines[["top", "right"]].set_visible(False)
            ax_args.set_xlabel("Epoch", fontsize=9)
            ax_args.set_title("Argument Regression Metrics", fontsize=11, fontweight="bold", pad=6)
            ax_mape = ax_args.twinx()
            ax_mape.plot(epochs, vp["val_arg_mape"], color=C["arg_mape"],
                         label="Arg MAPE", linewidth=1.5, linestyle="--")
            ax_mape.set_ylabel("MAPE", fontsize=9, color=C["arg_mape"])
            ax_mape.tick_params(axis="y", colors=C["arg_mape"], labelsize=8)
            ax_mape.spines[["top"]].set_visible(False)
            lines1, labels1 = ax_args.get_legend_handles_labels()
            lines2, labels2 = ax_mape.get_legend_handles_labels()
            ax_args.legend(lines1 + lines2, labels1 + labels2, fontsize=8, framealpha=0.7, loc="best")

        fig.suptitle(f"Training Progression: {config.run_name}", fontsize=13, fontweight="bold")
        plt.show()

        if out_path is not None:
            fig.savefig(out_path, bbox_inches="tight", dpi=120)


def merge_best_epochs(out_dir: str = "out", output_path: str = "out/best_merged.csv") -> pd.DataFrame:
    """
    Merge all ``best_epoch.csv`` files found under ``<out_dir>/*/best_epoch.csv``
    into a single CSV saved at ``output_path``.

    A ``run_name`` column is prepended to each row, derived from the immediate
    parent directory name of each CSV (i.e. the experiment/run folder).
    Experiments that have different column sets (e.g. ``val_arg_r2``,
    ``val_arg_mape``) are handled gracefully via ``pd.concat`` with
    ``sort=False``; missing columns are filled with ``NaN``.

    Args:
        out_dir:     Root output directory to search (default: ``"out"``).
        output_path: Destination path for the merged CSV (default: ``"out/best_merged.csv"``).

    Returns:
        The merged :class:`~pandas.DataFrame`.
    """
    import glob

    pattern = os.path.join(out_dir, "*", "best_epoch.csv")
    csv_paths = sorted(glob.glob(pattern))

    if not csv_paths:
        raise FileNotFoundError(f"No best_epoch.csv files found matching: {pattern}")

    frames = []
    for path in csv_paths:
        run_name = os.path.basename(os.path.dirname(path))
        df = pd.read_csv(path)
        df.insert(0, "run_name", run_name)
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged.to_csv(output_path, index=False)
    print(f"Merged {len(frames)} experiment(s) → {output_path}  ({len(merged)} row(s))")
    return merged
