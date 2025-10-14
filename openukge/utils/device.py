import torch


def prepare_device():
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # n_gpu = torch.cuda.device_count()
    # if n_gpu > 1:
    #     print(f"Using {n_gpu} GPUs!")
    return device
