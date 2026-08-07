import random
import numpy as np
import torch

def setup_tensor_cores(precision: str = "high"):
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    setup_tensor_cores()

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True