"""
Device and hardware management utility for PyTorch SDXL LoRA training.
Provides auto-detection of CUDA / MPS / CPU, VRAM querying, and mixed-precision dtypes.
"""

import os
import torch
from typing import Tuple, Dict, Any


def get_optimal_device() -> torch.device:
    """Detect and return the optimal compute device available."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_torch_dtype(precision_str: str) -> torch.dtype:
    """Map string precision ('no', 'fp16', 'bf16') to torch.dtype."""
    precision_str = precision_str.lower()
    if precision_str == "fp16":
        return torch.float16
    elif precision_str in ("bf16", "bfloat16"):
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        # Fallback to float32 or float16 if bfloat16 not supported
        return torch.float32
    return torch.float32


def get_device_info() -> Dict[str, Any]:
    """Retrieve detailed hardware information for the current execution."""
    device = get_optimal_device()
    info: Dict[str, Any] = {
        "device_type": device.type,
        "device_name": "CPU",
        "total_memory_gb": 0.0,
        "available_memory_gb": 0.0,
        "cuda_available": torch.cuda.is_available(),
        "bf16_supported": False
    }

    if device.type == "cuda":
        idx = torch.cuda.current_device()
        info["device_name"] = torch.cuda.get_device_name(idx)
        total = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(idx) / (1024 ** 3)
        info["total_memory_gb"] = round(total, 2)
        info["available_memory_gb"] = round(total - reserved, 2)
        info["bf16_supported"] = torch.cuda.is_bf16_supported()

    return info


def clear_memory() -> None:
    """Flush CUDA / MPS memory caches."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
