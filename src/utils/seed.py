"""
Reproducibility and deterministic random seed utilities.
Ensures identical initializations and data batch permutations across baseline
and continual learning training phases.
"""

import os
import random
from typing import Optional
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Sets random seeds across Python, NumPy, PyTorch CPU, and PyTorch CUDA.
    Configures deterministic backend settings for complete reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """
    Worker init function to ensure deterministic DataLoader multi-processing.
    Assigns each worker an independent, deterministic sub-seed.
    """
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
