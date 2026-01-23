from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        # torch not installed / no GPU, ignore
        pass


@dataclass
class DeviceInfo:
    device: str
    dtype: str
    is_cuda: bool


def get_device(prefer_cuda: bool = True) -> DeviceInfo:
    """Return best device info for inference/training."""
    try:
        import torch

        if prefer_cuda and torch.cuda.is_available():
            # bfloat16 if supported, else float16
            if torch.cuda.is_bf16_supported():
                return DeviceInfo(device="cuda", dtype="bfloat16", is_cuda=True)
            return DeviceInfo(device="cuda", dtype="float16", is_cuda=True)
        return DeviceInfo(device="cpu", dtype="float32", is_cuda=False)
    except Exception:
        return DeviceInfo(device="cpu", dtype="float32", is_cuda=False)
