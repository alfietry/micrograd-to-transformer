"""Shared helpers: config loading and reproducible seeding."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import yaml


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
