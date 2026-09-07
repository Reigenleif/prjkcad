from __future__ import annotations

import os
import json
from typing import Any, Dict, Optional, Tuple, Union
import torch
import numpy as np
import pandas as pd

from utils.trainer.base_lightning_module import BaseLightningModule
from utils.grpo import compute_advantages, grpo_loss, compute_rewards
from utils.grpo.rollout import generate_rollouts_tokenized
from utils.representations.dual_seq.schema import get_dualseq_schema
from utils.evaluate import eval_batch


class GRPOLightningModule(BaseLightningModule):
    """Group Relative Policy Optimization Lightning module."""

    def __init__(
        self,
        wrapper: Any,
        criterion: Any = None,
        optimizer: Any = None,
        scheduler: Any = None,
        n_rollouts: int = 8,
        clip_eps: float = 0.2,
        temperature: float = 1.0,
        ref_sync_every: int = 50,
        kl_coef: float = 0.01,
        lambda_arg_reward: float = 0.5,
        deviation_limit: int = 2,
        subset_refresh_every: int = 0,
        save_folder: str = None,
        pipeline: Optional[Any] = None,
        *args,
        **kwargs,
    ):
        super().__init__(wrapper, criterion, optimizer, scheduler, *args, **kwargs)
        self.n_rollouts = n_rollouts
        self.clip_eps = clip_eps
        self.temperature = temperature
        self.ref_sync_every = ref_sync_every
        self.kl_coef = kl_coef
        self.lambda_arg_reward = lambda_arg_reward
        self.deviation_limit = deviation_limit
        self.subset_refresh_every = subset_refresh_every
        self.pipeline = pipeline
        self.save_folder = save_folder or kwargs.get("save_folder", "out")
        self.run_data_path = os.path.join(self.save_folder, "run_data.csv") if self.save_folder else "out/run_data.csv"
        self.run_data = []

    def forward(self, *args, **kwargs) -> Any:
        if self.wrapper is None:
            raise NotImplementedError("Module wrapper is not set.")
        return self.wrapper(*args, **kwargs)

    def _save_run_data(self) -> None:
        if self.run_data and self.run_data_path:
            os.makedirs(os.path.dirname(self.run_data_path), exist_ok=True)
            pd.DataFrame(self.run_data).to_csv(self.run_data_path, index=False)

    def on_train_epoch_start(self) -> None:
        if (
            self.subset_refresh_every > 0
            and self.current_epoch > 0
            and self.current_epoch % self.subset_refresh_every == 0
            and self.pipeline is not None
        ):
            from utils.grpo.subset import subset_grpo_dataset
            print(f"[GRPOLightningModule] Refreshing hard subset at epoch {self.current_epoch}")
            subset_grpo_dataset(self.pipeline, use_cache=False)

    def on_train_epoch_end(self) -> None:
        self._save_run_data()

    def on_train_end(self) -> None:
        self._save_run_data()

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        inner = getattr(self.wrapper, "wrapper", self.wrapper)
        if hasattr(inner, "sync_ref_model") and self.global_step % self.ref_sync_every == 0:
            inner.sync_ref_model()
            self.log("grpo/ref_synced_at", float(self.global_step), on_step=True, on_epoch=False, logger=True)

    def training_step(self, batch: Union[Dict[str, Any], Tuple], batch_idx: int) -> torch.Tensor:
        cmd_targets, arg_targets = None, None
        if hasattr(self.wrapper, "extract_inputs"):
            input_ids, attn_mask, cmd_targets, arg_targets = self.wrapper.extract_inputs(batch)
        elif isinstance(batch, dict):
            input_ids = batch.get("input_ids", batch.get("x"))
            attn_mask = batch.get("attention_mask", batch.get("attn_mask"))
            cmd_targets = batch.get("cmd_targets", batch.get("y_cmds"))
            arg_targets = batch.get("arg_targets", batch.get("y_args"))
        elif len(batch) >= 4:
            input_ids, cmd_targets, arg_targets, attn_mask = batch[0], batch[1], batch[2], batch[3]
        else:
            input_ids, attn_mask = batch[0], batch[1]

        if isinstance(batch, (tuple, list)) and len(batch) >= 5:
            batch_uids = batch[4]
        elif isinstance(batch, dict) and "uids" in batch:
            batch_uids = batch["uids"]
        else:
            batch_uids = [f"sample_{i}" for i in range(input_ids.size(0))]

        input_ids = input_ids.to(self.device)
        attn_mask = attn_mask.to(self.device)

        # ── Rollout ──────────────────────────────────────────────────────────
        inner = getattr(self.wrapper, "wrapper", self.wrapper)
        rollout = inner.generate_rollout(input_ids, attn_mask, n_rollouts=self.n_rollouts, temperature=self.temperature)

        cmd_seqs = rollout["sampled_cmds"].cpu().numpy().tolist()
        arg_seqs = rollout["sampled_args"].cpu().numpy().tolist()
        enc_out  = rollout.get("enc_out")

        # ── Reference log-probs ──────────────────────────────────────────────
        max_T = rollout["sampled_cmds"].size(1)
        N = len(cmd_seqs)
        B = input_ids.size(0)
        n_rollouts_actual = max(1, N // B)

        cmd_tensor_ref = torch.tensor(cmd_seqs, dtype=torch.long)
        arg_tensor_ref = torch.tensor(arg_seqs, dtype=torch.long)
        # Pad to max_T
        cmd_pad = torch.full((N, max_T), inner.model.pad_id, dtype=torch.long)
        arg_pad = torch.full((N, max_T, 31), inner.model.arg_pad_id, dtype=torch.long)
        T_c = min(cmd_tensor_ref.size(1), max_T)
        T_a = min(arg_tensor_ref.size(1), max_T)
        cmd_pad[:, :T_c] = cmd_tensor_ref[:, :T_c]
        arg_pad[:, :T_a] = arg_tensor_ref[:, :T_a]

        input_ids_r = input_ids.repeat_interleave(n_rollouts_actual, dim=0)
        attn_mask_r = attn_mask.repeat_interleave(n_rollouts_actual, dim=0)
        ref_log_probs = inner.compute_ref_log_probs(input_ids_r, attn_mask_r, cmd_pad, arg_pad, max_T, enc_out=enc_out)

        # ── Rewards ──────────────────────────────────────────────────────────
        schema = getattr(inner, "schema", None) or get_dualseq_schema()
        metadata = getattr(inner, "metadata", None)

        rollout_uids = []
        for uid in batch_uids:
            rollout_uids.extend([uid] * self.n_rollouts)

        gt_cmds_batch = cmd_targets.cpu().numpy().tolist() if (cmd_targets is not None and hasattr(cmd_targets, "cpu")) else cmd_targets
        gt_args_batch = arg_targets.cpu().numpy().tolist() if (arg_targets is not None and hasattr(arg_targets, "cpu")) else arg_targets

        rewards_np, arg_rewards_np, generated_dualseqs = compute_rewards(
            cmd_seqs, arg_seqs, schema=schema, metadata=metadata,
            gt_cmds_batch=gt_cmds_batch, gt_args_batch=gt_args_batch,
            uids=rollout_uids,
            deviation_limit=self.deviation_limit,
        )
        combined_np = rewards_np + self.lambda_arg_reward * arg_rewards_np
        rewards_tensor     = torch.tensor(rewards_np,    dtype=torch.float32, device=self.device)
        arg_rewards_tensor = torch.tensor(arg_rewards_np, dtype=torch.float32, device=self.device)
        combined_tensor    = torch.tensor(combined_np,   dtype=torch.float32, device=self.device)

        # ── Run data logging (buffered, written at epoch end) ─────────────────
        for i, ds in enumerate(generated_dualseqs):
            self.run_data.append({
                "uid": ds.uid,
                "step": self.global_step,
                "epoch": self.current_epoch,
                "rollout_idx": i % self.n_rollouts,
                "t2c_reward": float(rewards_np[i]),
                "arg_reward": float(arg_rewards_np[i]),
                "combined_reward": float(combined_np[i]),
                "dualseq": json.dumps({"cmds": ds.cmds, "args": ds.args}),
                "dualseq_str": str(ds),
            })

        # ── Advantages ───────────────────────────────────────────────────────
        advantages = compute_advantages(combined_tensor, n_rollouts=self.n_rollouts, B=B)

        # ── Loss ─────────────────────────────────────────────────────────────
        loss, loss_stats = grpo_loss(
            inner,
            input_ids,
            attn_mask,
            cmd_seqs,
            arg_seqs,
            ref_log_probs,
            advantages,
            self.clip_eps,
            max_T=max_T,
            enc_out=enc_out,
            kl_coef=self.kl_coef,
        )

        # ── grpo/ metrics ────────────────────────────────────────────────────
        log_kwargs = dict(on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train/grpo_loss",          loss,                              on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/reward",             combined_tensor.mean(),            on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("grpo/reward_mean",         combined_tensor.mean(),            **log_kwargs)
        self.log("grpo/reward_std",          combined_tensor.std(),             **log_kwargs)
        self.log("grpo/reward_min",          combined_tensor.min(),             **log_kwargs)
        self.log("grpo/reward_max",          combined_tensor.max(),             **log_kwargs)
        self.log("grpo/t2c_reward_mean",     rewards_tensor.mean(),             **log_kwargs)
        self.log("grpo/t2c_reward_std",      rewards_tensor.std(),              **log_kwargs)
        self.log("grpo/arg_reward_mean",     arg_rewards_tensor.mean(),         **log_kwargs)
        self.log("grpo/arg_reward_std",      arg_rewards_tensor.std(),          **log_kwargs)
        self.log("grpo/advantage_mean",      advantages.mean(),                 **log_kwargs)
        self.log("grpo/advantage_std",       advantages.std(),                  **log_kwargs)
        self.log("grpo/kl",                  loss_stats["kl"],                  **log_kwargs)
        self.log("grpo/ratio_mean",          loss_stats["ratio_mean"],          **log_kwargs)
        self.log("grpo/ratio_clip_frac",     loss_stats["ratio_clip_frac"],     **log_kwargs)
        self.log("grpo/loss_ppo",            loss_stats["loss_ppo"],            **log_kwargs)
        self.log("grpo/seq_len_mean",        loss_stats["seq_len_mean"],        **log_kwargs)

        return loss

    def validation_step(self, batch: Union[Dict[str, Any], Tuple], batch_idx: int) -> torch.Tensor:
        outputs = self(batch, is_teacher_forcing=True)
        if self.criterion is not None:
            loss = self.criterion(outputs, batch)
            if isinstance(loss, tuple):
                loss = loss[0]
        else:
            loss = torch.tensor(0.0, device=self.device)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/perplexity", torch.exp(loss.detach()), on_step=False, on_epoch=True, prog_bar=False, logger=True)

        try:
            with torch.no_grad():
                input_ids, attn_mask, cmd_targets, arg_targets = self.wrapper.extract_inputs(batch)
                input_ids = input_ids.to(self.device)
                attn_mask  = attn_mask.to(self.device)

                # Greedy eval metrics (teacher-forced generation)
                gen_outputs = self(batch, is_teacher_forcing=False)
                if isinstance(gen_outputs, dict):
                    cmd_preds = gen_outputs.get("cmd_preds")
                    arg_preds = gen_outputs.get("arg_preds")
                elif isinstance(gen_outputs, (tuple, list)):
                    cmd_preds = gen_outputs[2] if len(gen_outputs) > 2 else gen_outputs[0]
                    arg_preds = gen_outputs[3] if len(gen_outputs) > 3 else gen_outputs[1]
                else:
                    cmd_preds, arg_preds = None, None

                if cmd_preds is not None and cmd_targets is not None:
                    out_type = getattr(self.wrapper, "out_type", None) or getattr(getattr(self.wrapper, "wrapper", None), "out_type", "EightBitBinarizedArgs")
                    metadata = getattr(self.wrapper, "metadata", None) or getattr(getattr(self.wrapper, "wrapper", None), "metadata", None)
                    eval_metrics = eval_batch(cmd_preds, cmd_targets, arg_preds, arg_targets, out_type=out_type, metadata=metadata)
                    for k, v in eval_metrics.items():
                        self.log(k, v, on_step=False, on_epoch=True, prog_bar=(k in ["eval/avg_f1", "t2c/macro_f1", "eval/arg_token_accuracy"]), logger=True)

                # Stochastic reward on val — aligned with training reward metric
                inner = getattr(self.wrapper, "wrapper", self.wrapper)
                schema = getattr(inner, "schema", None) or get_dualseq_schema()
                metadata = getattr(inner, "metadata", None)
                max_steps = getattr(self, "max_new_cmds", 84)

                val_cmd_seqs, val_arg_seqs, _ = generate_rollouts_tokenized(
                    inner, input_ids, attn_mask,
                    n_rollouts=1,
                    max_steps=max_steps,
                    temperature=self.temperature,
                    schema=schema,
                )
                B = input_ids.size(0)
                gt_cmds = cmd_targets.cpu().numpy().tolist() if hasattr(cmd_targets, "cpu") else cmd_targets
                gt_args = arg_targets.cpu().numpy().tolist() if hasattr(arg_targets, "cpu") else arg_targets

                val_rewards_np, val_arg_rewards_np, _ = compute_rewards(
                    val_cmd_seqs, val_arg_seqs,
                    schema=schema, metadata=metadata,
                    gt_cmds_batch=gt_cmds, gt_args_batch=gt_args,
                    deviation_limit=self.deviation_limit,
                )
                val_combined = val_rewards_np + self.lambda_arg_reward * val_arg_rewards_np
                self.log("val/reward",         float(val_combined.mean()),      on_step=False, on_epoch=True, prog_bar=True,  logger=True)
                self.log("val/t2c_reward_mean", float(val_rewards_np.mean()),   on_step=False, on_epoch=True, prog_bar=False, logger=True)
                self.log("val/arg_reward_mean", float(val_arg_rewards_np.mean()), on_step=False, on_epoch=True, prog_bar=False, logger=True)

        except Exception as e:
            print(f"[GRPOLightningModule] Eval error (suppressed): {type(e).__name__}: {e}")

        return loss

    def _compute_dummy_advantages(self, n_samples: int) -> torch.Tensor:
        B = max(1, n_samples // self.n_rollouts)
        rewards = torch.randn(n_samples, device=self.device)
        return compute_advantages(rewards, n_rollouts=self.n_rollouts, B=B)
