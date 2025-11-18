"""
Utilities for working with torch devices across CUDA, MPS, and CPU.
"""

from __future__ import annotations

import torch


def get_best_device(*, prefer_cuda: bool = True, prefer_mps: bool = True) -> torch.device:
    """Return the most capable available torch.device.

    Args:
        prefer_cuda: Try CUDA before other accelerators.
        prefer_mps: Try Apple Metal (MPS) if CUDA is unavailable.

    Returns:
        torch.device pointing to CUDA, MPS, or CPU (fallback).
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")

    has_mps_backend = hasattr(torch.backends, "mps")
    if prefer_mps and has_mps_backend and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    """Provide a human-readable description of the selected device."""
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        try:
            name = torch.cuda.get_device_name(index)
        except torch.cuda.CudaError:
            name = "CUDA device"
        return f"cuda:{index} ({name})"

    if device.type == "mps":
        return "mps (Apple Silicon)"

    return "cpu"
