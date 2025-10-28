import torch
from typing import Optional


def prepare_device(device_index: Optional[int] = None) -> torch.device:
    """
    Setup training device: CPU or a specific GPU (if available).

    Args:
        device_index (Optional[int]):
            - None: auto choose GPU:0 if available, otherwise CPU
            - >=0 : manually specify GPU index
            Default: None

    Returns:
        torch.device: the selected device

    Example:
        device = prepare_device()        # auto-select
        device = prepare_device(1)       # use cuda:1
    """
    if torch.cuda.is_available():
        if device_index is not None:
            # 指定某一张 GPU
            if device_index < torch.cuda.device_count():
                device = torch.device(f"cuda:{device_index}")
            else:
                print(f"[Warning] GPU index {device_index} is out of range "
                      f"(available: 0 ~ {torch.cuda.device_count() - 1}), using cuda:0")
                device = torch.device("cuda:0")
        else:
            # 自动选择第一张 GPU
            device = torch.device("cuda:0")

        print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")

    else:
        device = torch.device("cpu")
        print("Using device: CPU (CUDA not available)")

    return device
