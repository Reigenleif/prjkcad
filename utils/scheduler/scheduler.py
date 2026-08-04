from __future__ import annotations

import math
import torch
from torch.optim.lr_scheduler import LambdaLR
from utils.pipeline.config import SchedulerConfig


class CustomScheduler(LambdaLR):
    def __init__(self, optimizer: torch.optim.Optimizer, config: SchedulerConfig, total_steps: int):
        self.config = config
        self.total_steps = total_steps

        warmup_steps = config.warmup_steps
        if config.warmup_ratio is not None:
            warmup_steps = int(config.warmup_ratio * total_steps)
        self.warmup_steps = warmup_steps

        scheduler_type = config.type.lower()
        if scheduler_type not in ("cosine", "linear"):
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

        eta_min = getattr(config, "eta_min", 0.0)

        def lr_lambda(current_step: int) -> float:
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))

            progress = float(current_step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            progress = min(1.0, max(0.0, progress))

            if scheduler_type == "cosine":
                return eta_min + (1.0 - eta_min) * 0.5 * (1.0 + math.cos(math.pi * progress))
            elif scheduler_type == "linear":
                return eta_min + (1.0 - eta_min) * (1.0 - progress)
            else:
                raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

        super().__init__(optimizer, lr_lambda)
