import torch
from fastapi import APIRouter
from gui.backend.deps import get_harness

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
def health():
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if cuda_avail else 0.0
    return {
        "status": "healthy",
        "device": "cuda" if cuda_avail else "cpu",
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "model_id": get_harness().model_name,
        "is_mock": get_harness().is_mock,
    }
