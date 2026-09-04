import wandb
import torch
import os
import time
from typing import Dict, Any, Optional, List


def init_wandb(
    project_name: str,
    config: Dict[str, Any],
    run_name: Optional[str] = None,
    *,
    group: Optional[str] = None,
    job_type: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> wandb.run:
    """Initialize a W&B run. Requires WANDB_API_KEY (or prior `wandb login`)."""

    if run_name is None:
        run_name = f"{project_name}_{time.strftime('%Y%m%d_%H%M%S')}"

    if not os.environ.get("WANDB_API_KEY") and wandb.api.api_key is None:
        raise RuntimeError(
            "WANDB_API_KEY is not set and no wandb login found. "
            "Export WANDB_API_KEY from https://wandb.ai/authorize before training."
        )

    entity = os.environ.get("WANDB_ENTITY") or None
    group = group or os.environ.get("WANDB_RUN_GROUP") or None
    run_tags = tags or [project_name, "training", "tinyworlds"]

    run = wandb.init(
        project=project_name,
        entity=entity,
        config=config,
        name=run_name,
        group=group,
        job_type=job_type,
        tags=run_tags,
        resume="allow",
    )

    # Keep train/val on the same x-axis in the UI.
    wandb.define_metric("step")
    wandb.define_metric("train/*", step_metric="step")
    wandb.define_metric("val/*", step_metric="step")
    wandb.define_metric("system/*", step_metric="step")
    wandb.define_metric("learning_rate/*", step_metric="step")

    print(f"W&B run initialized: {run.name}")
    print(f"Project: {project_name}" + (f" (entity={entity})" if entity else ""))
    print(f"View at: {run.url}")

    return run


def log_training_metrics(step: int, metrics: Dict[str, float], prefix: str = "train"):
    # Add prefix to metric names
    prefixed_metrics = {f"{prefix}/{k}": (v.item() if hasattr(v, "item") else float(v)) for k, v in metrics.items()}
    wandb.log(prefixed_metrics, step=step)


def log_learning_rate(optimizer: torch.optim.Optimizer, step: int):
    for i, param_group in enumerate(optimizer.param_groups):
        wandb.log({
            f"learning_rate/group_{i}": float(param_group['lr']),
        }, step=step)

def log_codebook_usage(codebook_usage: float, step: int, model_name: str = "model"):
    wandb.log({
        f"{model_name}/codebook_usage": float(codebook_usage),
    }, step=step)


def log_action_distribution(action_indices: torch.Tensor, step: int, n_actions: int):
    # Convert to CPU and numpy
    flat = action_indices.detach().cpu().reshape(-1)
    # Compute distribution
    counts = torch.bincount(flat.long(), minlength=n_actions).float()
    probs = counts / counts.sum().clamp_min(1)
    
    wandb.log({
        "action_distribution": wandb.Histogram(flat.numpy()),
        "action_entropy": float(-(probs * (probs + 1e-8).log()).sum()),
        "unique_actions": int((counts > 0).sum().item()),
    }, step=step)


def log_system_metrics(step: int):
    if torch.cuda.is_available():
        wandb.log({
            "system/gpu_memory_allocated": float(torch.cuda.memory_allocated() / 1024**3),  # GB
            "system/gpu_memory_reserved": float(torch.cuda.memory_reserved() / 1024**3),    # GB
        }, step=step)


def finish_wandb():
    """Finish the W&B run"""
    if wandb.run is not None:
        wandb.finish()


def create_wandb_config(args, model_config: Dict[str, Any]) -> Dict[str, Any]:
    config = {
        # Training parameters
        "batch_size": args.batch_size,
        "n_updates": args.n_updates,
        "learning_rate": args.learning_rate,
        "log_interval": getattr(args, 'log_interval', 100),
        
        # Dataset parameters
        "dataset": getattr(args, 'dataset', 'SONIC'),
        "context_length": getattr(args, 'context_length', 4),
        
        # Model architecture
        "model_architecture": model_config,
        
        # System parameters
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    }
    
    # Add any additional args that might exist
    for attr in dir(args):
        if not attr.startswith('_') and not callable(getattr(args, attr)):
            if attr not in config:
                config[attr] = getattr(args, attr)
    
    return config
