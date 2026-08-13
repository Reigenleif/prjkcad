from __future__ import annotations

from pytorch_lightning.callbacks import Callback
from tqdm import tqdm

class GlobalStepProgressBar(Callback):
    def __init__(self, total_steps: int):
        super().__init__()
        self.total_steps = total_steps
        self.pbar = None

    def on_fit_start(self, trainer, pl_module):
        self.pbar = tqdm(total=self.total_steps, desc="Training")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.pbar is not None:
            self.pbar.update(1)
            self.pbar.set_postfix({
                "global_step": trainer.global_step,
                "epoch": trainer.current_epoch
            })

    def on_fit_end(self, trainer, pl_module):
        if self.pbar is not None:
            self.pbar.close()
