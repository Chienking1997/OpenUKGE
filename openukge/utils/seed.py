"""
Utility function to set random seeds for reproducibility across Python,
NumPy (if available), and PyTorch (if available).
"""

import os
import random
from typing import Optional

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None


def seed_everything(
    seed: int = 888,
    deterministic: bool = True,
    env_var: Optional[str] = "PYTHONHASHSEED",
) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): Random seed. Default: 888.
        deterministic (bool): Enables deterministic PyTorch backend. Default: True.
        env_var (Optional[str]): Env variable name for hash seed. Default: PYTHONHASHSEED.
    """
    # Python random
    random.seed(seed)

    # NumPy random
    if np is not None:
        np.random.seed(seed)

    # Python hash randomization
    if env_var is not None:
        os.environ[env_var] = str(seed)

    # PyTorch seeds
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
