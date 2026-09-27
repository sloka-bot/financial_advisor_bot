"""Seed stochastic libraries; exact results still depend on hardware and versions."""

import random

import numpy as np


def set_seed(seed=42):
    """Seed Python, NumPy and available PyTorch generators for repeatable runs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
