import copy
import os
import types
from typing import List

import numpy as np
import optuna
import pandas as pd
from tqdm import tqdm

from utils.pipeline import TrainModelPipeline
from utils.pipeline.config import ModelConfig
from utils.pruning.config_validator import is_valid_config
from utils.pruning.config_cache import ConfigCache

try:
    from IPython.display import clear_output
    _has_ipython = True
except ImportError:
    _has_ipython = False


# Search spaces
_ENCODERS       = ["t5-small", "bert"]
_DECODERS       = ["sdpa", "t5-small", "mamba"]
_MOE_CONFS      = ["Switch", "Mixtral", None]
_ADAPTIVE_TYPES = ["none", "linear", "ffn_head", "sdpa"]
_EMBED_TYPES    = ["standard", "rope", "sdpa", "rope_sdpa"]
_DROP_OUT_PS    = [0.1, 0.3, 0.5]


def _build_model_config(trial: optuna.Trial, base_cfg, max_new_cmds: int) -> ModelConfig:
    """Sample a ModelConfig from the Optuna trial."""
    encoder_type     = trial.suggest_categorical("encoder_type",     _ENCODERS)
    cmd_decoder_type = trial.suggest_categorical("cmd_decoder_type", _DECODERS)
    args_decoder_type = trial.suggest_categorical("args_decoder_type", _DECODERS)
    moe_conf         = trial.suggest_categorical("moe_conf",         _MOE_CONFS)
    adaptive_layer   = trial.suggest_categorical("adaptive_layer",   _ADAPTIVE_TYPES)
    cmd_embedding_type  = trial.suggest_categorical("cmd_embedding_type",  _EMBED_TYPES)
    args_embedding_type = trial.suggest_categorical("args_embedding_type", _EMBED_TYPES)
    use_drop_out     = trial.suggest_categorical("use_drop_out",     [True, False])
    drop_out_p       = trial.suggest_categorical("drop_out_p",       _DROP_OUT_PS) if use_drop_out else 0.1

    # d_model is tied to encoder
    d_model = 512  # only t5-small / bert (bert auto-projects, so 512 is fine)

    freeze_encoder = (encoder_type == "bert") or getattr(base_cfg.model, "freeze_encoder", False)
    freeze_cmd_decoder = getattr(base_cfg.model, "freeze_cmd_decoder", False)
    freeze_args_decoder = getattr(base_cfg.model, "freeze_args_decoder", False)

    return ModelConfig(
        is_pretrained=False,
        encoder_type=encoder_type,
        cmd_decoder_type=cmd_decoder_type,
        args_decoder_type=args_decoder_type,
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
        kwargs=copy.deepcopy(base_cfg.model.kwargs),
    )


def _build_metadata_row(
    trial: optuna.Trial,
    cfg: ModelConfig,
    run_name: str,
    avg_f1: float,
    status: str,
    cache_hit: bool,
) -> dict:
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
        "avg_f1":              avg_f1,
        "status":              status,
        "cache_hit":           cache_hit,
    }


def _append_metadata(metadata_path: str, row: dict):
    if os.path.exists(metadata_path):
        df = pd.read_csv(metadata_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(metadata_path, index=False)


def make_patched_fit(trial: optuna.Trial):
    """
    Returns a patched fit function that intercepts the epoch training loop to:
    1. Suppress output logs when not verbose.
    2. Report intermediate val_avg_f1 values to the Optuna trial.
    3. Raise optuna.TrialPruned if the trial should be pruned.
    """
    def patched_fit(self, epochs: int, verbose: bool = True) -> List[dict]:
        init_str = f"Starting training for {epochs} epochs"
        if self.quant_type is not None:
            init_str += f" with quantization: {self.quant_type}"
        
        try:
            model_param_cnt = sum(p.numel() for p in self.wrapper.model.parameters())
            trainable_param_cnt = sum(p.numel() for p in self.wrapper.model.parameters() if p.requires_grad)
            init_str += f". Model parameters: {model_param_cnt:,} (trainable: {trainable_param_cnt:,})"
        except Exception:
            pass    
        
        if verbose:
            print(init_str)
        
        history: List[dict] = []
        best_metric_value = float('-inf') if self.best_metric_mode == "max" else float('inf')
        best_epoch = -1

        for epoch in range(epochs):
            train_ratio = self._scheduled_ratio(epoch)
            train_metrics = []
            
            loader = self.train_loader or []
            if verbose:
                loader = tqdm(loader, desc=f"Train {epoch + 1}/{epochs}")
            for batch in loader:
                train_metrics.append(self.train_step(batch, train_ratio))

            summary = {}
            if train_metrics:
                all_train_keys = set()
                for m in train_metrics:
                    all_train_keys.update(m.keys())
                for key in all_train_keys:
                    vals = [metric[key] for metric in train_metrics if metric.get(key) is not None]
                    if vals:
                        summary[f"train_{key}"] = float(np.mean(vals))

            if self.val_loader is not None:
                eval_metrics = []
                val_loader = self.val_loader
                if verbose:
                    val_loader = tqdm(val_loader, desc=f"Eval {epoch + 1}/{epochs}")
                for batch in val_loader:
                    eval_metrics.append(self.eval_step(batch))
                if eval_metrics:
                    all_val_keys = set()
                    for m in eval_metrics:
                        all_val_keys.update(m.keys())
                    for key in all_val_keys:
                        vals = [metric[key] for metric in eval_metrics if metric.get(key) is not None]
                        if vals:
                            summary[f"val_{key}"] = float(np.mean(vals))

            history.append(summary)
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}: {summary}")
            
            if self.best_metric_key not in summary:
                raise ValueError(f"Best metric '{self.best_metric_key}' not found in summary metrics: {summary.keys()}")
            
            current_metric_value = summary[self.best_metric_key]
            is_best = False
            if self.best_metric_mode == "min":
                if current_metric_value < best_metric_value:
                    best_metric_value = current_metric_value
                    is_best = True
            else:
                if current_metric_value > best_metric_value:
                    best_metric_value = current_metric_value
                    is_best = True
                    
            if is_best and self.save_folder is not None:
                best_epoch = epoch
                self.save_on_best_epoch(self.save_folder, best_epoch, summary)

            # Report intermediate value to Optuna
            val_metric = summary.get(self.best_metric_key, 0.0)
            trial.report(val_metric, epoch)
            if trial.should_prune():
                # Clean up current trial's checkpoint immediately if pruned to save space
                chk_path = os.path.join(self.save_folder, "checkpoint.pt")
                if os.path.exists(chk_path):
                    try:
                        os.remove(chk_path)
                    except Exception:
                        pass
                raise optuna.TrialPruned()

        if self.save_folder is not None:
            self.save_progression(self.save_folder, history)
                
        return history
    return patched_fit


def _clean_non_best_checkpoints(output_dir: str, study: optuna.Study):
    """Removes checkpoint.pt from all trial directories except the best one."""
    try:
        if len(study.trials) > 0:
            best_trial_num = study.best_trial.number
            for t in study.trials:
                if t.state == optuna.trial.TrialState.COMPLETE and t.number != best_trial_num:
                    chk_path = os.path.join(output_dir, f"trial_{t.number + 1:04d}", "checkpoint.pt")
                    if os.path.exists(chk_path):
                        try:
                            os.remove(chk_path)
                            print(f"[Optuna] Cleaned up checkpoint for non-best trial {t.number + 1}")
                        except Exception:
                            pass
    except Exception:
        pass


def _make_objective(base_cfg, coreset, output_dir: str, cache: ConfigCache, max_new_cmds: int, study: optuna.Study):
    """
    Returns the Optuna objective function (closure capturing shared state).
    """
    metadata_path = os.path.join(output_dir, "metadata.csv")

    def objective(trial: optuna.Trial) -> float:
        # Clear output at the start of each trial to keep the notebook/logs clean
        # if _has_ipython:
        #     clear_output(wait=True)
            
        print(f"\n[Optuna] Starting Trial {trial.number + 1}...")

        # Clean up checkpoints from previous trials that are not the best
        _clean_non_best_checkpoints(output_dir, study)

        # 1. Sample configuration
        model_cfg = _build_model_config(trial, base_cfg, max_new_cmds)

        # 2. Validate — prune invalid trials so Optuna learns to avoid them
        if not is_valid_config(model_cfg):
            print(f"[Optuna] Invalid config sampled. Pruning trial {trial.number + 1}...")
            raise optuna.TrialPruned()

        # 3. Check cache
        cache_key = cache.key(model_cfg)
        cached_result = cache.get(cache_key)
        run_name = f"optuna/trial_{trial.number + 1:04d}"

        if cached_result is not None:
            print(f"[Optuna] Trial {trial.number + 1} cache hit: F1 = {cached_result:.4f}")
            row = _build_metadata_row(
                trial, model_cfg, run_name, cached_result, "cache_hit", cache_hit=True
            )
            _append_metadata(metadata_path, row)
            return cached_result

        # 4. Train
        cfg = copy.deepcopy(base_cfg)
        cfg.model = model_cfg
        cfg.run_name = run_name

        # Align tokenizer with the encoder model dynamically for pruning
        if model_cfg.encoder_type == "bert":
            cfg.tokenizer.model_name = "bert-base-uncased"
        elif model_cfg.encoder_type.startswith("t5-"):
            cfg.tokenizer.model_name = model_cfg.encoder_type


        avg_f1 = 0.0
        status = "failed"
        try:
            pipeline = TrainModelPipeline(cfg)
            pipeline.dual_seqs = coreset
            pipeline.load_things()
            
            # Monkeypatch the trainer's fit function for trial reporting and pruning support
            trainer = pipeline.trainer
            patched_fit = make_patched_fit(trial)
            trainer.fit = types.MethodType(patched_fit, trainer)

            progression = pipeline.train_model(verbose_all=True, do_render=False)

            val_avg_f1s = [h.get("val_avg_f1", 0.0) for h in progression]
            avg_f1 = max(val_avg_f1s) if val_avg_f1s else 0.0
            status = "success"

        except optuna.TrialPruned:
            print(f"[Optuna] Trial {trial.number + 1} pruned during training.")
            avg_f1 = 0.0
            status = "pruned"
            raise
        except Exception as e:
            print(f"[Optuna] Trial {trial.number + 1} failed: {type(e).__name__}: {e}")
            avg_f1 = 0.0
            status = f"failed ({type(e).__name__})"

        # 5. Update cache and metadata
        if status == "success":
            cache.set(cache_key, avg_f1)
            cache.save()

        row = _build_metadata_row(
            trial, model_cfg, run_name, avg_f1, status, cache_hit=False
        )
        _append_metadata(metadata_path, row)

        return avg_f1

    return objective


def run_optuna_study(
    base_cfg,
    coreset: List,
    n_trials: int = 50,
    output_dir: str = "out/optuna",
    study_name: str = "prjkcad_arch_search",
) -> optuna.Study:
    """
    Run study

    Args:
        base_cfg:    Base Config object loaded from a YAML file.
        coreset:     List of DualSeq objects used as training data.
        n_trials:    Total number of trials to run (including already-completed ones).
        output_dir:  Directory for all Optuna outputs (db, metadata, cache).
        study_name:  Name used for SQLite checkpointing — reuse to resume.

    Returns:
        The completed optuna.Study object.
    """
    os.makedirs(output_dir, exist_ok=True)

    db_path    = os.path.join(output_dir, "optuna.db")
    cache_path = os.path.join(output_dir, "config_cache.json")
    storage    = f"sqlite:///{db_path}"

    max_new_cmds = base_cfg.model.max_new_cmds

    cache = ConfigCache(cache_path)
    print(f"[Optuna] Config cache loaded: {len(cache)} entries from {cache_path}")

    # Create or load existing study (checkpoint resume via SQLite storage)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,           # resumes if study_name already exists in db
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), # real pruner to cut off bad runs early
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    already_done = len(study.trials)
    remaining = max(0, n_trials - already_done)
    if remaining == 0:
        print(f"[Optuna] Study '{study_name}' already has {already_done} trials. Nothing to do.")
        return study

    print(
        f"[Optuna] Starting study '{study_name}' "
        f"(completed={already_done}, remaining={remaining}, total={n_trials})"
    )

    objective = _make_objective(base_cfg, coreset, output_dir, cache, max_new_cmds, study)
    study.optimize(objective, n_trials=remaining)

    # Perform a final cleanup to remove any remaining non-best trial checkpoints
    _clean_non_best_checkpoints(output_dir, study)

    print("\n[Optuna] Best trial:")
    print(f"  Value : {study.best_trial.value:.4f}")
    print(f"  Params: {study.best_trial.params}")

    return study
