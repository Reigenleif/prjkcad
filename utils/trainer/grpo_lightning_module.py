from __future__ import annotations

import os
import json
from typing import Any, Dict, Tuple, Union
import torch
import numpy as np
import pandas as pd

from utils.trainer.base_lightning_module import BaseLightningModule
from utils.grpo import compute_advantages, grpo_loss, compute_rewards
from utils.representations.dual_seq.schema import get_dualseq_schema

class GRPOLightningModule(BaseLightningModule):
    """Data-type agnostic Group Relative Policy Optimization module."""

    def __init__(
        self,
        wrapper: Any,
        criterion: Any = None,
        optimizer: Any = None,
        scheduler: Any = None,
        n_rollouts: int = 8,
        clip_eps: float = 0.2,
        temperature: float = 1.0,
        save_folder: str = None,
        *args,
        **kwargs
    ):
        super().__init__(wrapper, criterion, optimizer, scheduler, *args, **kwargs)
        self.n_rollouts = n_rollouts
        self.clip_eps = clip_eps
        self.temperature = temperature
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

    def on_train_end(self) -> None:
        self._save_run_data()

    def training_step(self, batch: Union[Dict[str, Any], Tuple], batch_idx: int) -> torch.Tensor:
        if hasattr(self.wrapper, "extract_inputs"):
            input_ids, attn_mask, _, _ = self.wrapper.extract_inputs(batch)
        elif isinstance(batch, dict):
            input_ids = batch.get("input_ids", batch.get("x"))
            attn_mask = batch.get("attention_mask", batch.get("attn_mask"))
        elif len(batch) >= 4:
            input_ids, attn_mask = batch[0], batch[3]
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
        rollout = self.wrapper.generate_rollout(input_ids, attn_mask, n_rollouts=self.n_rollouts, temperature=self.temperature)
        ref_log_probs = rollout["log_probs"].detach()

        cmd_seqs = rollout["sampled_cmds"].cpu().numpy().tolist()
        arg_seqs = rollout["sampled_args"].cpu().numpy().tolist()

        schema = getattr(self.wrapper, "schema", None) or get_dualseq_schema()
        metadata = getattr(self.wrapper, "metadata", None) or getattr(getattr(self.wrapper, "wrapper", None), "metadata", None)

        rollout_uids = []
        for uid in batch_uids:
            rollout_uids.extend([uid] * self.n_rollouts)

        rewards_np, generated_dualseqs = compute_rewards(
            cmd_seqs, arg_seqs, schema=schema, metadata=metadata, uids=rollout_uids
        )
        rewards_tensor = torch.tensor(rewards_np, dtype=torch.float32, device=self.device)

        for i, ds in enumerate(generated_dualseqs):
            score = float(rewards_np[i])
            r_idx = i % self.n_rollouts
            self.run_data.append({
                "uid": ds.uid,
                "step": self.global_step,
                "epoch": self.current_epoch,
                "rollout_idx": r_idx,
                "score": score,
                "dualseq": json.dumps({"cmds": ds.cmds, "args": ds.args}),
                "dualseq_str": str(ds),
            })

        self._save_run_data()

        B = max(1, len(cmd_seqs) // self.n_rollouts)
        advantages = compute_advantages(rewards_tensor, n_rollouts=self.n_rollouts, B=B)

        loss = grpo_loss(
            self.wrapper,
            input_ids,
            attn_mask,
            cmd_seqs,
            arg_seqs,
            ref_log_probs,
            advantages,
            self.clip_eps,
            max_T=rollout["sampled_cmds"].size(1)
        )

        mean_reward = rewards_tensor.mean()
        self.log("train/grpo_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/reward", mean_reward, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def _compute_dummy_advantages(self, n_samples: int) -> torch.Tensor:
        B = max(1, n_samples // self.n_rollouts)
        rewards = torch.randn(n_samples, device=self.device)
        return compute_advantages(rewards, n_rollouts=self.n_rollouts, B=B)
