"""
Utility function to set random seeds for reproducibility across Python,
NumPy, and PyTorch (if available).
"""

import os
import random
from typing import Optional

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def seed_everything(
    seed: int = 888,
    deterministic: bool = True,
    env_var: Optional[str] = "PYTHONHASHSEED"
) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch to ensure
    reproducible results.

    Args:
        seed (int): The seed value to use. Defaults to 42.
        deterministic (bool): If True, enables deterministic mode in
            PyTorch (if installed). This may reduce performance but
            ensures reproducibility. Defaults to True.
        env_var (Optional[str]): The environment variable used to set
            Python's hash seed. Defaults to "PYTHONHASHSEED".

    Example:
        >>> seed_everything(1234)
        >>> # After this, all random operations will be reproducible.
    """
    # Python's built-in random module
    random.seed(seed)

    # NumPy random seed
    np.random.seed(seed)

    # Control Python hash randomization
    if env_var is not None:
        os.environ[env_var] = str(seed)

    # PyTorch seeds (if torch is installed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    seed_everything(2025)
    print(random.randint(0, 100))
    print(np.random.rand(3))
    if torch is not None:
        print(torch.randn(2, 2))
