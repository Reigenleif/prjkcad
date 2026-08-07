import os
import torch
import wandb

def init_wandb(run_name: str, config_dict: dict = None, wrapper: torch.nn.Module = None, eval_steps: int = 1000, project_name: str = None):
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if wandb_api_key:
        try:
            wandb.login(key=wandb_api_key)
        except Exception:
            pass

    cfg_proj = config_dict.get("wandb_project") if isinstance(config_dict, dict) else None
    wandb_project = os.environ.get("WANDB_PROJECT") or project_name or cfg_proj or "prjkcad"

    if wandb.run is None:
        try:
            wandb.init(
                project=wandb_project,
                name=run_name,
                config=config_dict,
                reinit=True
            )
            print(f"Initialized WandB run '{run_name}' in project '{wandb_project}'")
        except Exception as e:
            print(f"Warning: wandb.init failed: {e}")

    if wandb.run is not None and wrapper is not None:
        try:
            wandb.watch(wrapper, log="gradients", log_freq=eval_steps)
        except Exception:
            pass

def log_wandb_train(log_dict: dict, step: int):
    if wandb.run:
        wandb.log(log_dict, step=step)

def log_wandb_eval(summary: dict, step: int, wrapper: torch.nn.Module = None, save_folder: str = None, log_artifacts: bool = False):
    if wandb.run:
        val_log = {k.replace("val_", "val/"): v for k, v in summary.items() if k.startswith("val_")}
        grad_dict = {}
        if wrapper is not None:
            for name, param in wrapper.named_parameters():
                if param.grad is not None:
                    grad_dict[f"gradients/{name}"] = param.grad.norm().item()
        wandb.log({**val_log, **grad_dict}, step=step)
        
        if log_artifacts and save_folder is not None:
            artifact = wandb.Artifact(name=f"best_model_{wandb.run.id}", type="model")
            for fname in ["encoder.pt", "adaptive_layer.pt", "checkpoint.pt"]:
                fpath = os.path.join(save_folder, fname)
                if os.path.exists(fpath):
                    artifact.add_file(fpath)
            wandb.log_artifact(artifact)
