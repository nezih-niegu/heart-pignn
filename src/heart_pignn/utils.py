"""Shared helpers: seeding, device selection, filtering, JSON I/O."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.signal import butter, filtfilt


def set_seed(seed: int = 42) -> None:
    """Seed random, numpy and torch (including CUDA if present)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(preferred: str = "auto") -> torch.device:
    """Resolve the device. 'auto' takes CUDA when available, otherwise CPU."""
    if preferred != "auto":
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_device(device: torch.device) -> str:
    """Human-readable device summary, printed at the top of every run."""
    if device.type != "cuda":
        return f"{device} (CPU)"
    idx = device.index or 0
    props = torch.cuda.get_device_properties(idx)
    total_gb = props.total_memory / 1024**3
    return (
        f"{device} -> {props.name}, {total_gb:.1f} GB, "
        f"compute {props.major}.{props.minor}, torch {torch.__version__}"
    )


def cuda_available_but_unused(device: torch.device) -> bool:
    """True when a GPU exists but the run is on CPU anyway."""
    return device.type != "cuda" and torch.cuda.is_available()


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4):
    nyq = fs / 2.0
    return butter(order, [lowcut / nyq, highcut / nyq], btype="band")


def bandpass_filter(
    signal: np.ndarray, lowcut: float = 0.5, highcut: float = 40.0, fs: float = 360.0
) -> np.ndarray:
    """Band-pass ECG at 0.5-40 Hz: removes baseline drift and high-frequency noise."""
    b, a = butter_bandpass(lowcut, highcut, fs)
    return filtfilt(b, a, signal, axis=0)


def zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (x - x.mean(axis=0)) / (x.std(axis=0) + eps)


def to_native(obj: Any) -> Any:
    """Convert numpy/torch types into JSON-serializable natives."""
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def write_json(path: Path | str, payload: Any) -> None:
    """Write JSON with an explicit encoding (Windows does not default to UTF-8)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(to_native(payload), fh, indent=2, ensure_ascii=False)


def read_json(path: Path | str, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
