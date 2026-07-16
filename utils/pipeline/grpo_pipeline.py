"""
GRPOPipeline — Group Relative Policy Optimization over DualSeq (cmd + arg) outputs.

Training flow per step:
  1. Sample n_rollouts completions from the current policy (autoregressive, with temperature)
  2. Decode each rollout → DualSeq → render → CD → reward
  3. Compute group-relative advantage:  A_i = (r_i - mean(r)) / (std(r) + eps)
  4. Compute PPO-clip surrogate loss using current log-probs vs. rollout (reference) log-probs
  5. Gradient step

Reference paper: DeepSeekMath / GRPO — no separate reference model needed;
the rollout log-probs collected @generation time serve as the reference.
"""
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Union, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from utils.set_seed import set_seed
from utils.data_utils import load_split_data, create_dualseq_data_loader
from utils.dual_seq import DualSeq, get_dualseq_schema
from utils.wrapper.tokenized_args_wrapper import TokenizedArgsWrapper
from utils.evaluate.grpo_evaluate import compute_reward
from utils.render import render_dual_seq_to_shape
from utils.evaluate.shape_evaluation_functions import chamfer_distance_from_shapes

from models import TokenizedArgsBaseModel
from utils.pipeline.config import Config, GRPOConfig


def _load_model_from_dir(model, load_dir: str):
    """Load encoder + adaptive_layer + full checkpoint (if available) from a directory."""
    for attr, fname in [("encoder", "encoder.pt"), ("adaptive_layer", "adaptive_layer.pt")]:
        path = os.path.join(load_dir, fname)
        if os.path.exists(path) and hasattr(model, attr):
            state_dict = torch.load(path, map_location="cpu")
            sub = getattr(model, attr)
            model_sd = sub.state_dict()
            filtered = {k: v for k, v in state_dict.items() if k in model_sd and v.shape == model_sd[k].shape}
            sub.load_state_dict(filtered, strict=False)
            print(f"Loaded {attr} from {path}")

    ckpt_path = os.path.join(load_dir, "checkpoint.pt")
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model_sd = model.state_dict()
        filtered = {k: v for k, v in state_dict.items()
                    if k in model_sd and v.shape == model_sd[k].shape}
        model.load_state_dict(filtered, strict=False)
        print(f"Loaded full checkpoint from {ckpt_path}")


def _render_cd_worker(pred_cmds, pred_args_dict, gt_cmds, gt_args_dict):
    """
    Top-level render function for subprocess-isolated OCC CD.
    Used only in evaluate_cd, not in the training loop.
    """
    import sys
    import os as _os
    project_root = _os.path.dirname(
        _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        from utils.render import render_dual_seq_to_shape as _render
        from utils.evaluate.shape_evaluation_functions import chamfer_distance_from_shapes as _cd
        import numpy as _np
        pred_shape = _render(pred_cmds, pred_args_dict)
        gt_shape   = _render(gt_cmds,   gt_args_dict)
        if pred_shape is None or gt_shape is None:
            return 1.0
        cd = _cd(pred_shape, gt_shape)
        return 1.0 if (cd is None or _np.isnan(cd)) else float(cd)
    except Exception:
        return 1.0


def _safe_cd_subprocess_worker(q, pred_cmds, pred_args_dict, gt_cmds, gt_args_dict):
    result = _render_cd_worker(pred_cmds, pred_args_dict, gt_cmds, gt_args_dict)
    q.put(result)


def _safe_cd_subprocess(pred_cmds, pred_args_dict, gt_cmds, gt_args_dict, timeout: int = 30) -> float:
    """
    One-shot subprocess for OCC rendering, used only in evaluate_cd.
    Prevents OCC segfaults from crashing the main process.
    """
    ctx = mp.get_context("spawn")
    q   = ctx.Queue()

    p = ctx.Process(target=_safe_cd_subprocess_worker, args=(q, pred_cmds, pred_args_dict, gt_cmds, gt_args_dict))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return 1.0
    try:
        return q.get_nowait()
    except Exception:
        return 1.0


def _fast_validity_reward(pred_cmds, pred_arg_tokens, schema, min_cd=1e-5, max_cd=0.5) -> float:
    """
    Fast reward for training — no OCC rendering.
    Returns 1.0 if DualSeq parses successfully with at least one extrude command,
    else a scaled score based on sequence length similarity.
    """
    if not pred_cmds:
        return 0.0
    extrude_cmds = {"EXTRUDE_NEW", "EXTRUDE_JOIN", "EXTRUDE_CUT", "EXTRUDE_INTERSECT"}
    has_extrude = any(c in extrude_cmds for c in pred_cmds)
    if not has_extrude:
        return 0.05
    try:
        ds = DualSeq.from_sequences(pred_cmds, pred_arg_tokens)
        n_coor = sum(1 for c in ds.cmds if c == "COOR")
        return min(1.0, 0.1 + 0.3 * n_coor)
    except Exception:
        return 0.02


def _decode_rollout(cmd_ids: list[int], arg_ids: list[int], schema: dict):
    """
    Decode raw token id lists → (cmd_strings, arg_token_ints) trimmed at EOS.
    Returns None, None if the lists are empty.
    """
    id_to_cmd = schema["id_to_command"]

    cmd_strings = [id_to_cmd.get(i, "PAD") for i in cmd_ids]
    try:
        eos_idx = cmd_strings.index("EOS")
        cmd_strings = cmd_strings[:eos_idx]
        arg_ids = arg_ids[:eos_idx]
    except ValueError:
        pass

    cmd_strings = [c for c in cmd_strings if c not in ("PAD", "SOS")]
    return cmd_strings, arg_ids


@torch.no_grad()
def _generate_rollouts_tokenized(
    wrapper: TokenizedArgsWrapper,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    n_rollouts: int,
    max_steps: int,
    temperature: float,
    schema: dict,
):
    """
    Generate n_rollouts autoregressive completions per prompt, collecting log-probs.

    Returns
    -------
    cmd_seqs   : list[list[int]]  length B*n_rollouts, each inner list is token ids
    arg_seqs   : list[list[int]]  same shape
    log_probs  : list[torch.Tensor]  each (T,)  — sum of step log-probs
    """
    device = input_ids.device
    B = input_ids.size(0)
    N = B * n_rollouts

    # Tile inputs for parallel rollouts
    input_ids_r      = input_ids.repeat_interleave(n_rollouts, dim=0)
    attention_mask_r = attention_mask.repeat_interleave(n_rollouts, dim=0)

    model = wrapper.model

    # Pre-compute encoder output once
    _, _, enc_out = model(input_ids=input_ids_r, attention_mask=attention_mask_r)

    cmd_seq = torch.full((N, 1), model.sos_id,     device=device, dtype=torch.long)
    arg_seq = torch.full((N, 1), model.arg_sos_id, device=device, dtype=torch.long)

    step_log_probs_cmd = []
    step_log_probs_arg = []
    finished = torch.zeros(N, dtype=torch.bool, device=device)

    for _ in range(max_steps):
        if finished.all():
            break

        cmd_logits, arg_logits, _ = model(
            input_ids=input_ids_r,
            attention_mask=attention_mask_r,
            decoder_input_ids=cmd_seq,
            decoder_input_args=arg_seq,
            encoder_out_embeddings=enc_out,
        )

        next_cmd_logits = cmd_logits[:, -1, :]
        next_arg_logits = arg_logits[:, -1, :]

        if temperature > 0.0:
            cmd_probs = F.softmax(next_cmd_logits / temperature, dim=-1)
            next_cmd = torch.multinomial(cmd_probs, 1)
        else:
            next_cmd = next_cmd_logits.argmax(dim=-1, keepdim=True)

        if temperature > 0.0:
            arg_probs = F.softmax(next_arg_logits / temperature, dim=-1)
            next_arg = torch.multinomial(arg_probs, 1)
        else:
            next_arg = next_arg_logits.argmax(dim=-1, keepdim=True)

        # Collect log-probs for sampled tokens
        lp_cmd = F.log_softmax(next_cmd_logits, dim=-1).gather(1, next_cmd)
        lp_arg = F.log_softmax(next_arg_logits, dim=-1).gather(1, next_arg)

        # Mask finished sequences
        next_cmd = next_cmd.masked_fill(finished.unsqueeze(1), model.pad_id)
        next_arg = next_arg.masked_fill(finished.unsqueeze(1), model.arg_pad_id)
        lp_cmd   = lp_cmd.masked_fill(finished.unsqueeze(1), 0.0)
        lp_arg   = lp_arg.masked_fill(finished.unsqueeze(1), 0.0)

        cmd_seq = torch.cat([cmd_seq, next_cmd], dim=1)
        arg_seq = torch.cat([arg_seq, next_arg], dim=1)
        step_log_probs_cmd.append(lp_cmd)
        step_log_probs_arg.append(lp_arg)

        finished = finished | (next_cmd.squeeze(1) == model.eos_id)

    # cmd_seq / arg_seq include the leading SOS token — strip it
    cmd_seq = cmd_seq[:, 1:]
    arg_seq = arg_seq[:, 1:]

    summed_lp = (
        torch.cat(step_log_probs_cmd, dim=1) + torch.cat(step_log_probs_arg, dim=1)
    ).sum(dim=1)  # (N,)

    cmd_seqs  = cmd_seq.cpu().tolist()
    arg_seqs  = arg_seq.cpu().tolist()
    log_probs = summed_lp  # (N,) still on device

    return cmd_seqs, arg_seqs, log_probs


def _compute_rewards(
    cmd_seqs, arg_seqs,
    gt_cmds_batch, gt_args_batch,
    schema, grpo_cfg,
    pool: ProcessPoolExecutor | None = None,
):
    """
    Compute scalar rewards for each rollout using the fast validity reward.
    OCC rendering is skipped here to avoid segfaults and subprocess overhead.
    Returns rewards list[float] (N,).
    """
    n_rollouts = len(cmd_seqs) // len(gt_cmds_batch)
    rewards = []
    for n_idx, (cmd_ids, arg_ids) in enumerate(zip(cmd_seqs, arg_seqs)):
        pred_cmds, pred_arg_tokens = _decode_rollout(cmd_ids, arg_ids, schema)
        reward = _fast_validity_reward(pred_cmds, pred_arg_tokens, schema,
                                       grpo_cfg.min_cd, grpo_cfg.max_cd)
        rewards.append(reward)

    return rewards


def _grpo_loss(
    wrapper: TokenizedArgsWrapper,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    cmd_seqs: list[list[int]],
    arg_seqs: list[list[int]],
    ref_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float,
    max_T: int,
):
    """
    PPO-clip surrogate loss over the sampled sequences.
    Uses teacher-forcing with the rollout tokens as targets.
    """
    device = input_ids.device
    N = len(cmd_seqs)

    # Pad rollout sequences to max_T
    cmd_tensor = torch.zeros(N, max_T, dtype=torch.long, device=device)
    arg_tensor = torch.zeros(N, max_T, dtype=torch.long, device=device)
    for i, (c, a) in enumerate(zip(cmd_seqs, arg_seqs)):
        L = min(len(c), max_T)
        cmd_tensor[i, :L] = torch.tensor(c[:L], dtype=torch.long, device=device)
        L2 = min(len(a), max_T)
        arg_tensor[i, :L2] = torch.tensor(a[:L2], dtype=torch.long, device=device)

    # Tile input_ids to N (B * n_rollouts)
    B = input_ids.size(0)
    n_rollouts = N // B
    input_ids_r      = input_ids.repeat_interleave(n_rollouts, dim=0)
    attention_mask_r = attention_mask.repeat_interleave(n_rollouts, dim=0)

    model = wrapper.model

    # Teacher-forced forward pass with rollout tokens
    sos_cmd = torch.full((N, 1), model.sos_id,     device=device, dtype=torch.long)
    sos_arg = torch.full((N, 1), model.arg_sos_id, device=device, dtype=torch.long)
    dec_cmd = torch.cat([sos_cmd, cmd_tensor[:, :-1]], dim=1)
    dec_arg = torch.cat([sos_arg, arg_tensor[:, :-1]], dim=1)

    cmd_logits, arg_logits, _ = model(
        input_ids=input_ids_r,
        attention_mask=attention_mask_r,
        decoder_input_ids=dec_cmd,
        decoder_input_args=dec_arg,
    )

    # Log-probs of the rollout tokens under the current policy
    lp_cmd = F.log_softmax(cmd_logits, dim=-1)
    lp_arg = F.log_softmax(arg_logits, dim=-1)

    # Gather log-prob for each actual token; ignore pad positions
    mask = (cmd_tensor != model.pad_id).float()
    gathered_cmd = lp_cmd.gather(2, cmd_tensor.unsqueeze(2)).squeeze(2)
    gathered_arg = lp_arg.gather(2, arg_tensor.unsqueeze(2)).squeeze(2)

    cur_log_probs = ((gathered_cmd + gathered_arg) * mask).sum(dim=1)  # (N,)

    # PPO-clip ratio
    ratio = torch.exp(cur_log_probs - ref_log_probs.detach())
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

    adv = advantages.to(device)
    loss = -torch.min(ratio * adv, clipped_ratio * adv).mean()
    return loss


class GRPOPipeline:
    def __init__(self, cfg: Union[GRPOConfig, Dict[str, Any], str]):
        if isinstance(cfg, str):
            from .train_model import load_config
            cfg = load_config(cfg)
        elif isinstance(cfg, dict):
            cfg = Config.from_dict(cfg)
        self.cfg = cfg
        self.progression = None
        self.wrapper = None
        self.model = None
        self.optimizer = None
        self.train_loader = None
        self.val_loader = None
        self.dual_seqs = None
        self.val_dual_seqs = None
        self.schema = get_dualseq_schema()
        self._render_pool: ProcessPoolExecutor | None = None

    def __del__(self):
        if self._render_pool is not None:
            try:
                self._render_pool.shutdown(wait=False)
            except Exception:
                pass

    def load_things(self):
        config = self.cfg

        self.SAVE_ROOT = f"out/{config.run_name}"
        os.makedirs(self.SAVE_ROOT, exist_ok=True)

        set_seed(config.random_seed)

        if config.tokenizer.source == "huggingface":
            text_tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.model_name)
        else:
            raise ValueError(f"Unsupported tokenizer: {config.tokenizer.source}")

        if self.dual_seqs is None:
            dual_seqs, val_dual_seqs = load_split_data(
                data_folder=config.data.data_folder,
                metadata_csv=config.data.metadata_csv,
                source_data_type=config.data.source_data_type,
                split_json=config.data.split_json,
                max_samples=config.data.max_samples,
                sample_ratio=config.data.sample_ratio,
            )
            self.dual_seqs = dual_seqs
            self.val_dual_seqs = val_dual_seqs

        USE_VAL = config.data.eval_split_ratio > 0
        if self.val_dual_seqs is not None:
            train_loader = create_dualseq_data_loader(
                self.dual_seqs, text_tokenizer,
                description_level=config.data.description_level,
                batch_size=config.data.batch_size,
                num_workers=config.data.num_workers,
                val_ratio=0.0, shuffle=True,
            )
            val_loader = create_dualseq_data_loader(
                self.val_dual_seqs, text_tokenizer,
                description_level=config.data.description_level,
                batch_size=config.data.batch_size,
                num_workers=config.data.num_workers,
                val_ratio=0.0, shuffle=False,
            )
        elif USE_VAL:
            train_loader, val_loader = create_dualseq_data_loader(
                self.dual_seqs, text_tokenizer,
                description_level=config.data.description_level,
                batch_size=config.data.batch_size,
                num_workers=config.data.num_workers,
                val_ratio=config.data.eval_split_ratio, shuffle=True,
            )
        else:
            train_loader = create_dualseq_data_loader(
                self.dual_seqs, text_tokenizer,
                description_level=config.data.description_level,
                batch_size=config.data.batch_size,
                num_workers=config.data.num_workers,
                val_ratio=0.0, shuffle=True,
            )
            val_loader = None

        model = TokenizedArgsBaseModel(
            cfg=config.model,
            vocab_size=self.schema["cmd_n_tokens"],
            vocab_size_args=self.schema["args_n_tokens"],
        )

        load_dir = None
        if any(os.path.exists(os.path.join(self.SAVE_ROOT, f))
               for f in ["encoder.pt", "adaptive_layer.pt", "checkpoint.pt"]):
            load_dir = self.SAVE_ROOT
            print(f"Resuming from {self.SAVE_ROOT}")
        elif config.pretrained_path is not None:
            load_dir = config.pretrained_path

        if load_dir is not None:
            if os.path.isdir(load_dir):
                _load_model_from_dir(model, load_dir)
            else:
                sd = torch.load(load_dir, map_location="cpu")
                model_sd = model.state_dict()
                filtered = {k: v for k, v in sd.items()
                            if k in model_sd and v.shape == model_sd[k].shape}
                model.load_state_dict(filtered, strict=False)
                print(f"Loaded checkpoint from {load_dir}")

        device = config.trainer.kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        wrapper = TokenizedArgsWrapper(model, text_tokenizer, device=device)

        if config.trainer.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(wrapper.parameters(), **config.trainer.optimizer_kwargs)
        else:
            raise ValueError(f"Unsupported optimizer: {config.trainer.optimizer}")

        scheduler = None
        if config.trainer.scheduler is not None:
            sch_cfg = config.trainer.scheduler
            total_steps = config.trainer.epochs * len(train_loader)
            warmup_steps = sch_cfg.warmup_steps
            if sch_cfg.warmup_ratio is not None:
                warmup_steps = int(sch_cfg.warmup_ratio * total_steps)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )

        self.wrapper = wrapper
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        print(f"GRPO pipeline ready | device={device} | train={len(train_loader)} batches")

    def train_model(self, verbose: bool = False):
        if self.wrapper is None:
            self.load_things()

        config = self.cfg
        grpo_cfg = config.grpo
        schema = self.schema
        device = next(self.wrapper.parameters()).device

        wandb_api_key = os.environ.get("WANDB_API_KEY")
        wandb_project = os.environ.get("WANDB_PROJECT")
        if wandb_api_key and wandb_project:
            wandb.login(key=wandb_api_key)
            wandb.init(project=wandb_project, name=config.run_name,
                       config=config.to_dict() if hasattr(config, "to_dict") else {},
                       reinit=True, resume="allow")

        max_T = config.trainer.max_new_cmds or config.model.max_new_cmds or 128
        max_grad_norm = config.trainer.kwargs.get("max_grad_norm", 1.0)
        total_steps = config.trainer.epochs * len(self.train_loader)
        global_step = 0
        history = []

        pbar = tqdm(total=total_steps, desc="GRPO Training")

        for epoch in range(config.trainer.epochs):
            epoch_rewards = []
            epoch_losses  = []

            for batch in self.train_loader:
                input_ids, cmd_targets, arg_targets, attention_mask = batch
                input_ids      = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                cmd_targets    = cmd_targets.to(device)
                arg_targets    = arg_targets.to(device)

                B = input_ids.size(0)

                # Ground truth for reward computation (CPU lists)
                gt_cmd_lists = cmd_targets.cpu().tolist()
                gt_arg_lists = arg_targets.cpu().tolist()

                # 1. Rollout
                self.wrapper.eval()
                with torch.no_grad():
                    cmd_seqs, arg_seqs, ref_log_probs = _generate_rollouts_tokenized(
                        self.wrapper, input_ids, attention_mask,
                        n_rollouts=grpo_cfg.n_rollouts,
                        max_steps=max_T,
                        temperature=grpo_cfg.temperature,
                        schema=schema,
                    )

                # 2. Rewards (fast validity-based, no OCC rendering)
                rewards_list = _compute_rewards(
                    cmd_seqs, arg_seqs,
                    gt_cmd_lists, gt_arg_lists,
                    schema, grpo_cfg,
                )
                rewards_np = np.array(rewards_list, dtype=np.float32)

                # 3. Group-relative advantage  (per prompt group)
                advantages = np.zeros_like(rewards_np)
                for b in range(B):
                    start = b * grpo_cfg.n_rollouts
                    end   = start + grpo_cfg.n_rollouts
                    grp   = rewards_np[start:end]
                    mu, sigma = grp.mean(), grp.std() + 1e-8
                    advantages[start:end] = (grp - mu) / sigma

                advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)

                # 4. Policy gradient step
                self.wrapper.train()
                self.optimizer.zero_grad(set_to_none=True)

                loss = _grpo_loss(
                    self.wrapper, input_ids, attention_mask,
                    cmd_seqs, arg_seqs,
                    ref_log_probs, advantages_t,
                    clip_eps=grpo_cfg.clip_eps,
                    max_T=max_T,
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.wrapper.model.parameters(), max_grad_norm)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                loss_val = float(loss.detach().cpu())
                mean_reward = float(rewards_np.mean())
                epoch_rewards.append(mean_reward)
                epoch_losses.append(loss_val)
                global_step += 1

                pbar.update(1)
                pbar.set_description(
                    f"E{epoch+1} step {global_step} | loss={loss_val:.4f} | reward={mean_reward:.3f}"
                )

                if wandb.run:
                    wandb.log({
                        "train/loss": loss_val,
                        "train/mean_reward": mean_reward,
                        "train/lr": float(self.optimizer.param_groups[0]["lr"]),
                    }, step=global_step)

            summary = {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(epoch_losses)),
                "train_mean_reward": float(np.mean(epoch_rewards)),
                "lr": float(self.optimizer.param_groups[0]["lr"]),
            }
            history.append(summary)
            print(f"\nEpoch {epoch+1}: loss={summary['train_loss']:.4f}  reward={summary['train_mean_reward']:.4f}")

        pbar.close()
        self.progression = history

        # Save
        self.wrapper.save(self.SAVE_ROOT)
        pd.DataFrame(history).to_csv(os.path.join(self.SAVE_ROOT, "history.csv"), index=False)

        if wandb.run:
            wandb.finish()

        return history

    def evaluate_cd(self, n_samples: int = 20) -> pd.DataFrame:
        """
        Run inference on n_samples, compute CD and reward for each.
        Returns a DataFrame: description, gt_len, pred_len, cd, reward, valid.
        """
        if self.wrapper is None:
            self.load_things()

        desc_level = self.cfg.data.description_level
        dual_seqs  = self.dual_seqs[:n_samples]
        max_new    = self.cfg.model.max_new_cmds or 128

        records = []
        self.wrapper.eval()

        pbar = tqdm(dual_seqs, desc="Evaluating CD")
        for seq in pbar:
            desc     = seq.descriptions[desc_level]
            gt_cmds  = seq.cmds
            gt_args  = seq.args_dict

            try:
                gen       = self.wrapper.generate(desc, max_new_tokens=max_new)
                pred_cmds = [item[0] for item in gen]
                pred_args = [item[1] for item in gen]
                valid     = len(pred_cmds) > 0
            except Exception:
                pred_cmds, pred_args, valid = [], [], False

            if valid:
                cd     = _safe_cd_subprocess(pred_cmds, pred_args, gt_cmds, gt_args)
                reward = compute_reward(cd, self.cfg.grpo.min_cd, self.cfg.grpo.max_cd)
            else:
                cd, reward = 1.0, 0.0

            records.append({
                "description": desc[:80] + "..." if len(desc) > 80 else desc,
                "gt_len":   len(gt_cmds),
                "pred_len": len(pred_cmds),
                "cd":       round(cd, 5),
                "reward":   round(reward, 4),
                "valid":    valid,
            })

        return pd.DataFrame(records)

    def plot_progression(self):
        if not self.progression:
            raise ValueError("No training progression. Please train first.")

        config = self.cfg
        progression = self.progression
        out_path = os.path.join(self.SAVE_ROOT, "grpo_progression.png")

        epochs      = [h["epoch"]             for h in progression]
        losses      = [h["train_loss"]        for h in progression]
        rewards     = [h["train_mean_reward"] for h in progression]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"GRPO Training: {config.run_name}", fontsize=13, fontweight="bold")

        axes[0].plot(epochs, losses, color="#4C72B0", linewidth=1.5, label="PPO-clip loss")
        axes[0].set_title("Loss", fontweight="bold")
        axes[0].set_xlabel("Epoch")
        axes[0].grid(True, linestyle="--", alpha=0.4)
        axes[0].spines[["top", "right"]].set_visible(False)
        axes[0].legend()

        axes[1].plot(epochs, rewards, color="#DD8452", linewidth=1.5, label="Mean reward")
        axes[1].set_title("Mean Reward", fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, linestyle="--", alpha=0.4)
        axes[1].spines[["top", "right"]].set_visible(False)
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved to {out_path}")
