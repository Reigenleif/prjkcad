import copy
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import optuna
import pandas as pd
import pytorch_lightning as pl
import torch

from utils.data_utils.coreset import CoresetCreator
from utils.data_utils.ref_loader import load_val_data
from utils.pipeline import BasePipeline, CustomPipeline
from utils.pipeline.config import ModelConfig, SchedulerConfig, TrainerConfig
from utils.pruning.config_cache import ConfigCache
from utils.pruning.config_validator import is_valid_config

_ENCODERS       = ["t5-small", "bert"]
_DECODERS       = ["sdpa", "t5-small", "mamba"]
_MOE_CONFS      = ["Switch", "Mixtral", None]
_ADAPTIVE_TYPES = ["none", "linear", "ffn_head", "sdpa"]
_EMBED_TYPES    = ["standard", "rope", "sdpa", "rope_sdpa"]
_DROP_OUT_PS    = [0.1, 0.3, 0.5]


class OptunaPruningCallback(pl.Callback):
    def __init__(self, trial: optuna.Trial, monitor: str = "val_avg_f1"):
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        val = trainer.callback_metrics.get(self.monitor)
        if val is None:
            val_loss = trainer.callback_metrics.get("val_loss")
            if val_loss is None:
                return
            val = -float(val_loss.detach().cpu().item() if isinstance(val_loss, torch.Tensor) else val_loss)
        else:
            val = float(val.detach().cpu().item() if isinstance(val, torch.Tensor) else val)

        epoch = trainer.current_epoch
        self.trial.report(val, step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


class OptunaStudyManager:
    """For Optuna to be object oriented and clean"""

    def __init__(
        self,
        base_cfg: Any,
        coreset: Optional[Any] = None,
        n_trials: int = 50,
        output_dir: str = "out/optuna",
        study_name: str = "prjkcad_arch_search",
        seed: int = 42,
    ):
        self.base_cfg = base_cfg
        self.coreset = coreset
        self.n_trials = n_trials
        self.output_dir = output_dir
        self.study_name = study_name
        self.seed = seed

        os.makedirs(self.output_dir, exist_ok=True)
        self.db_path = os.path.abspath(os.path.join(self.output_dir, "optuna.db"))
        self.cache_path = os.path.abspath(os.path.join(self.output_dir, "config_cache.json"))
        self.metadata_path = os.path.abspath(os.path.join(self.output_dir, "metadata.csv"))
        self.storage = f"sqlite:///{self.db_path}"

        self.cache = ConfigCache(self.cache_path)
        print(f"[Optuna] Config cache loaded: {len(self.cache)} entries from {self.cache_path}")

        self.dual_seqs, self.val_dual_seqs = self._preload_data()

        os.makedirs(self.output_dir, exist_ok=True)
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction="maximize",
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
            sampler=optuna.samplers.TPESampler(seed=self.seed),
        )

    def _preload_data(self) -> Tuple[Any, Any]:
        """Pre-load dataset or coreset dataset once to avoid redundant I/O during trials."""
        if self.coreset is not None:
            if isinstance(self.coreset, tuple):
                print("[Optuna] Using provided coreset tuple for training and validation.")
                return self.coreset[0], self.coreset[1]
            print("[Optuna] Using provided coreset for training.")
            return self.coreset, None

        coreset_kwargs = getattr(self.base_cfg.data, "coreset_kwargs", None)
        if coreset_kwargs is not None:
            c_kwargs = copy.deepcopy(coreset_kwargs)
            selection_method = c_kwargs.pop("selection_method", "closest")
            save_path = c_kwargs.pop("save_path", None)
            force_recreate = c_kwargs.pop("force_recreate", False)

            if "description_level" not in c_kwargs and hasattr(self.base_cfg.data, "description_level"):
                c_kwargs["description_level"] = self.base_cfg.data.description_level

            if save_path is None:
                p = c_kwargs.get("p", 0.01)
                k = c_kwargs.get("k", 100)
                desc_lvl = c_kwargs.get("description_level", "expert")
                model_name = c_kwargs.get("model_name", "bert-base-uncased")
                seed = c_kwargs.get("random_seed", 42)
                safe_model_name = re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)
                filename = (
                    f"coreset_{desc_lvl}_p{p}_k{k}_"
                    f"{safe_model_name}_{selection_method}_seed{seed}.json"
                )
                save_path = os.path.join("out", "coreset", filename)

            if not force_recreate and os.path.exists(save_path):
                print(f"[Optuna] Coreset cache found at: {save_path}. Loading coreset directly without reading raw train dataset...")
                creator = CoresetCreator(dual_seqs=None, data_root=self.base_cfg.data.data_folder, **c_kwargs)
                train_seqs = creator.create_coreset(
                    selection_method=selection_method,
                    save_path=save_path,
                    force_recreate=False
                )
                print("[Optuna] Loading validation split dataset...")
                val_max_sample = getattr(self.base_cfg.data, "val_max_sample", None)
                val_seqs = load_val_data(
                    data_folder=self.base_cfg.data.data_folder,
                    metadata_csv=self.base_cfg.data.metadata_csv,
                    source_data_type=self.base_cfg.data.source_data_type,
                    split_json=self.base_cfg.data.split_json,
                    val_max_sample=val_max_sample
                )
                val_len = len(val_seqs) if val_seqs is not None else 0
                print(f"[Optuna] Coreset cache loaded instantly: {len(train_seqs)} train sequences, {val_len} val sequences.")
                return train_seqs, val_seqs

            print("[Optuna] Coreset cache not found. Pre-loading full raw dataset to compute BERT coreset...")
            base_pipeline = BasePipeline(copy.deepcopy(self.base_cfg))
            base_pipeline.load_dataset()
            val_seqs = base_pipeline.val_dual_seqs
            val_len = len(val_seqs) if val_seqs is not None else 0

            creator = CoresetCreator(dual_seqs=base_pipeline.dual_seqs, **c_kwargs)
            train_seqs = creator.create_coreset(
                selection_method=selection_method,
                save_path=save_path,
                force_recreate=force_recreate
            )
            print(f"[Optuna] Coreset dataset created & saved: {len(train_seqs)} train sequences, {val_len} val sequences.")
            return train_seqs, val_seqs

        print("[Optuna] Pre-loading full dataset...")
        base_pipeline = BasePipeline(copy.deepcopy(self.base_cfg))
        base_pipeline.load_dataset()
        val_seqs = base_pipeline.val_dual_seqs
        val_len = len(val_seqs) if val_seqs is not None else 0
        train_len = len(base_pipeline.dual_seqs) if base_pipeline.dual_seqs is not None else 0
        print(f"[Optuna] Dataset pre-loaded: {train_len} train sequences, {val_len} val sequences.")
        return base_pipeline.dual_seqs, val_seqs

    def _build_model_config(self, trial: optuna.Trial) -> ModelConfig:
        encoder_type        = trial.suggest_categorical("encoder_type",        _ENCODERS)
        cmd_decoder_type    = trial.suggest_categorical("cmd_decoder_type",    _DECODERS)
        args_decoder_type   = trial.suggest_categorical("args_decoder_type",   _DECODERS)
        moe_conf            = trial.suggest_categorical("moe_conf",            _MOE_CONFS)
        adaptive_layer      = trial.suggest_categorical("adaptive_layer",      _ADAPTIVE_TYPES)
        cmd_embedding_type  = trial.suggest_categorical("cmd_embedding_type",  _EMBED_TYPES)
        args_embedding_type = trial.suggest_categorical("args_embedding_type", _EMBED_TYPES)
        use_drop_out        = trial.suggest_categorical("use_drop_out",        [True, False])
        drop_out_p          = trial.suggest_categorical("drop_out_p",          _DROP_OUT_PS) if use_drop_out else 0.1
        use_cmd_args_fusion = trial.suggest_categorical("use_cmd_args_fusion", [True, False])
        n_dec_blocks        = trial.suggest_categorical("n_dec_blocks",        [2, 4, 6, 8])

        d_model = 512
        max_new_cmds = getattr(self.base_cfg.model, "max_new_cmds", 128)

        freeze_encoder = (encoder_type == "bert") or getattr(self.base_cfg.model, "freeze_encoder", False)
        freeze_cmd_decoder = getattr(self.base_cfg.model, "freeze_cmd_decoder", False)
        freeze_args_decoder = getattr(self.base_cfg.model, "freeze_args_decoder", False)

        out_type = getattr(self.base_cfg.model, "out_type", None) or getattr(self.base_cfg.data, "out_type", "FloatArgs")
        base_model_kwargs = copy.deepcopy(getattr(self.base_cfg.model, "kwargs", {}) or {})

        return ModelConfig(
            is_pretrained=False,
            encoder_type=encoder_type,
            cmd_decoder_type=cmd_decoder_type,
            args_decoder_type=args_decoder_type,
            out_type=out_type,
            is_cmd_only=False,
            d_model=d_model,
            moe_conf=moe_conf,
            max_new_cmds=max_new_cmds,
            freeze_encoder=freeze_encoder,
            freeze_cmd_decoder=freeze_cmd_decoder,
            freeze_args_decoder=freeze_args_decoder,
            adaptive_layer=adaptive_layer,
            cmd_embedding_type=cmd_embedding_type,
            args_embedding_type=args_embedding_type,
            use_drop_out=use_drop_out,
            drop_out_p=drop_out_p,
            use_cmd_args_fusion=use_cmd_args_fusion,
            n_dec_blocks=n_dec_blocks,
            kwargs=base_model_kwargs,
        )

    def _build_trainer_config(self, trial: optuna.Trial) -> Tuple[TrainerConfig, int]:
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "Adam"])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)

        scheduler_type = trial.suggest_categorical("scheduler_type", ["cosine", "cosine_with_warmup", "linear", "none"])
        if scheduler_type != "none":
            warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2, step=0.05)
            eta_min = trial.suggest_float("eta_min", 1e-7, 1e-4, log=True)
            scheduler_cfg = SchedulerConfig(
                type=scheduler_type,
                warmup_ratio=warmup_ratio,
                eta_min=eta_min
            )
        else:
            scheduler_cfg = None

        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])

        optimizer_kwargs = copy.deepcopy(getattr(self.base_cfg.trainer, "optimizer_kwargs", {}) or {})
        optimizer_kwargs["lr"] = lr
        optimizer_kwargs["weight_decay"] = weight_decay

        base_eval_steps = getattr(self.base_cfg.trainer, "eval_steps", 100)
        eval_steps = min(base_eval_steps, 200) if base_eval_steps is not None else 100

        trainer_cfg = TrainerConfig(
            optimizer=optimizer_name,
            criterion=self.base_cfg.trainer.criterion,
            epochs=self.base_cfg.trainer.epochs,
            max_new_cmds=getattr(self.base_cfg.trainer, "max_new_cmds", None),
            max_new_args=getattr(self.base_cfg.trainer, "max_new_args", None),
            optimizer_kwargs=optimizer_kwargs,
            kwargs=copy.deepcopy(getattr(self.base_cfg.trainer, "kwargs", {}) or {}),
            eval_steps=eval_steps,
            scheduler=scheduler_cfg
        )
        return trainer_cfg, batch_size

    def _build_metadata_row(
        self,
        trial: optuna.Trial,
        cfg: ModelConfig,
        trainer_cfg: TrainerConfig,
        batch_size: int,
        run_name: str,
        avg_f1: float,
        best_epoch: int,
        best_loss: Optional[float],
        status: str,
        cache_hit: bool,
    ) -> dict:
        opt_kwargs = getattr(trainer_cfg, "optimizer_kwargs", {}) or {}
        sched = getattr(trainer_cfg, "scheduler", None)
        return {
            "trial_number":        trial.number,
            "run_name":            run_name,
            "encoder_type":        cfg.encoder_type,
            "cmd_decoder_type":    cfg.cmd_decoder_type,
            "args_decoder_type":   cfg.args_decoder_type or "None",
            "d_model":             cfg.d_model or 512,
            "moe_conf":            cfg.moe_conf or "None",
            "adaptive_layer":      getattr(cfg, "adaptive_layer", "none"),
            "cmd_embedding_type":  getattr(cfg, "cmd_embedding_type", "standard"),
            "args_embedding_type": getattr(cfg, "args_embedding_type", "standard"),
            "use_drop_out":        getattr(cfg, "use_drop_out", True),
            "drop_out_p":          getattr(cfg, "drop_out_p", 0.1),
            "use_cmd_args_fusion": getattr(cfg, "use_cmd_args_fusion", False),
            "n_dec_blocks":        getattr(cfg, "n_dec_blocks", 6),
            "lr":                  opt_kwargs.get("lr", 1e-4),
            "optimizer":           getattr(trainer_cfg, "optimizer", "AdamW"),
            "weight_decay":        opt_kwargs.get("weight_decay", 0.0),
            "scheduler_type":      getattr(sched, "type", "none") if sched else "none",
            "warmup_ratio":        getattr(sched, "warmup_ratio", 0.0) if sched else 0.0,
            "eta_min":             getattr(sched, "eta_min", 0.0) if sched else 0.0,
            "batch_size":          batch_size,
            "best_epoch":          best_epoch,
            "avg_f1":              avg_f1,
            "best_loss":           best_loss,
            "status":              status,
            "cache_hit":           cache_hit,
        }

    def _append_metadata(self, row: dict) -> None:
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            df_row = pd.DataFrame([row])
            file_exists = os.path.exists(self.metadata_path)
            df_row.to_csv(self.metadata_path, mode="a", header=not file_exists, index=False)
        except Exception as e:
            print(f"[Optuna] Warning: Failed to write metadata row: {e}")

    def _clean_non_best_checkpoints(self) -> None:
        try:
            completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if len(completed_trials) > 0:
                best_trial_num = self.study.best_trial.number
                for t in completed_trials:
                    if t.number != best_trial_num:
                        trial_dir = os.path.join(self.output_dir, f"trial_{t.number + 1:04d}")
                        for fname in ["checkpoint.pt", "checkpoint.ckpt"]:
                            chk_path = os.path.join(trial_dir, fname)
                            if os.path.exists(chk_path):
                                try:
                                    os.remove(chk_path)
                                    print(f"[Optuna] Cleaned up checkpoint for non-best trial {t.number + 1}")
                                except Exception:
                                    pass
        except Exception as e:
            print(f"[Optuna] Checkpoint cleanup warning: {e}")

    def _objective(self, trial: optuna.Trial) -> float:
        print(f"\n[Optuna] Starting Trial {trial.number + 1}...")

        model_cfg = self._build_model_config(trial)
        trainer_cfg, batch_size = self._build_trainer_config(trial)

        rel_output_dir = os.path.relpath(self.output_dir, start="out") if self.output_dir.startswith("out") else self.output_dir
        run_name = os.path.join(rel_output_dir, f"trial_{trial.number + 1:04d}")
        avg_f1 = 0.0
        best_epoch = 0
        best_loss = None
        status = "failed"
        cache_hit = False

        if not is_valid_config(model_cfg):
            print(f"[Optuna] Invalid config sampled. Pruning trial {trial.number + 1}...")
            status = "pruned_invalid"
            row = self._build_metadata_row(
                trial, model_cfg, trainer_cfg, batch_size, run_name, avg_f1, best_epoch, best_loss, status, cache_hit
            )
            self._append_metadata(row)
            raise optuna.TrialPruned()

        cache_key = self.cache.key(model_cfg, trainer_cfg, batch_size)
        cached_result = self.cache.get(cache_key)

        if cached_result is not None:
            print(f"[Optuna] Trial {trial.number + 1} cache hit: F1 = {cached_result:.4f}")
            avg_f1 = cached_result
            status = "cache_hit"
            cache_hit = True
            row = self._build_metadata_row(
                trial, model_cfg, trainer_cfg, batch_size, run_name, avg_f1, best_epoch, best_loss, status, cache_hit
            )
            self._append_metadata(row)
            return cached_result

        cfg = copy.deepcopy(self.base_cfg)
        cfg.model = model_cfg
        cfg.trainer = trainer_cfg
        cfg.data.batch_size = batch_size
        cfg.data.num_workers = 0
        cfg.run_name = run_name

        if model_cfg.encoder_type == "bert":
            cfg.tokenizer.model_name = "bert-base-uncased"
        elif model_cfg.encoder_type.startswith("t5-"):
            cfg.tokenizer.model_name = model_cfg.encoder_type

        if not hasattr(cfg.trainer, "kwargs") or cfg.trainer.kwargs is None:
            cfg.trainer.kwargs = {}
        cfg.trainer.kwargs["callbacks"] = [OptunaPruningCallback(trial)]

        try:
            pipeline = CustomPipeline(cfg)
            pipeline.dual_seqs = self.dual_seqs
            pipeline.val_dual_seqs = self.val_dual_seqs

            progression = pipeline.run()

            trial_folder = os.path.join(self.output_dir, f"trial_{trial.number + 1:04d}")
            os.makedirs(trial_folder, exist_ok=True)

            if hasattr(pipeline, "trainer") and hasattr(pipeline.trainer, "save_progression"):
                pipeline.trainer.save_progression(trial_folder)
            elif isinstance(progression, list) and len(progression) > 0:
                pd.DataFrame(progression).to_csv(os.path.join(trial_folder, "history.csv"), index=False)

            best_epoch = 0
            best_f1 = 0.0
            min_loss = float("inf")

            if isinstance(progression, list) and len(progression) > 0:
                for idx, epoch_metrics in enumerate(progression):
                    if not isinstance(epoch_metrics, dict):
                        continue
                    f1 = float(epoch_metrics.get("val_avg_f1", 0.0))
                    loss = float(epoch_metrics.get("val_loss", float("inf")))

                    if f1 > best_f1 or (f1 == best_f1 and loss < min_loss):
                        best_f1 = f1
                        min_loss = loss
                        best_epoch = idx

            best_loss = min_loss if min_loss != float("inf") else None
            best_info = {
                "best_epoch": best_epoch,
                "best_val_avg_f1": best_f1,
                "best_val_loss": best_loss,
                "total_epochs": len(progression) if isinstance(progression, list) else 0,
            }
            with open(os.path.join(trial_folder, "best_epoch_info.json"), "w") as f:
                json.dump(best_info, f, indent=2)

            avg_f1 = best_f1
            status = "success"
            self.cache.set(cache_key, avg_f1)
            self.cache.save()
            return avg_f1

        except optuna.TrialPruned:
            print(f"[Optuna] Trial {trial.number + 1} pruned during training.")
            avg_f1 = 0.0
            status = "pruned"
            raise

        except Exception as e:
            print(f"[Optuna] Trial {trial.number + 1} failed: {type(e).__name__}: {e}")
            avg_f1 = 0.0
            status = f"failed ({type(e).__name__})"
            return avg_f1

        finally:
            if not cache_hit and status != "pruned_invalid":
                row = self._build_metadata_row(
                    trial, model_cfg, trainer_cfg, batch_size, run_name, avg_f1, best_epoch, best_loss, status, cache_hit
                )
                self._append_metadata(row)

    def run(self) -> optuna.Study:
        already_done = len(self.study.trials)
        remaining = max(0, self.n_trials - already_done)

        if remaining == 0:
            print(f"[Optuna] Study '{self.study_name}' already has {already_done} trials. Nothing to do.")
            return self.study

        print(
            f"[Optuna] Starting study '{self.study_name}' "
            f"(completed={already_done}, remaining={remaining}, total={self.n_trials})"
        )

        self.study.optimize(self._objective, n_trials=remaining)
        self._clean_non_best_checkpoints()

        print("\n[Optuna] Best trial:")
        print(f"  Value : {self.study.best_trial.value:.4f}")
        print(f"  Params: {self.study.best_trial.params}")

        return self.study


def run_optuna_study(
    base_cfg: Any,
    coreset: Optional[Any] = None,
    n_trials: int = 50,
    output_dir: str = "out/optuna",
    study_name: str = "prjkcad_arch_search",
    seed: int = 42,
) -> optuna.Study:
    """Run Optuna study using OptunaStudyManager."""
    manager = OptunaStudyManager(
        base_cfg=base_cfg,
        coreset=coreset,
        n_trials=n_trials,
        output_dir=output_dir,
        study_name=study_name,
        seed=seed,
    )
    return manager.run()
