import os
import torch
import wandb

def init_wandb(run_name: str, config_dict: dict, wrapper: torch.nn.Module = None, eval_steps: int = 1000):
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    wandb_project = os.environ.get("WANDB_PROJECT")
    if wandb_api_key and wandb_project:
        wandb.login(key=wandb_api_key)
        wandb.init(
            project=wandb_project,
            name=run_name,
            config=config_dict,
            reinit=True
        )
        if wrapper is not None:
            wandb.watch(wrapper, log="gradients", log_freq=eval_steps)

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
