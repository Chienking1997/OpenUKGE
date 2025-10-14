import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from typing import Optional, Dict


class OptimBuilder:
    """
    A universal optimizer and scheduler builder for PyTorch.

    This class supports all optimizers and learning rate schedulers
    provided by the ``torch.optim`` and ``torch.optim.lr_scheduler`` modules.
    """

    def __init__(self, config: Dict):
        """
        Initialize the OptimBuilder.

        Args:
            config (dict): A configuration dictionary. Example:
                {
                    "optimizer": {
                        "type": "AdamW",
                        "lr": 1e-3,
                        "weight_decay": 1e-4
                    },
                    "scheduler": {
                        "type": "StepLR",
                        "step_size": 5,
                        "gamma": 0.5
                    }
                }
        """
        self.config = config

    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        """
        Build and return the optimizer.

        Args:
            model (torch.nn.Module): The model whose parameters will be optimized.

        Returns:
            torch.optim.Optimizer: The instantiated optimizer.
        """
        opt_cfg = self.config.get("optimizer", {})
        opt_type = opt_cfg.get("type", "Adam")

        if not hasattr(optim, opt_type):
            raise ValueError(f"Optimizer '{opt_type}' is not found in torch.optim.")

        # Dynamically get optimizer class from torch.optim
        opt_class = getattr(optim, opt_type)
        kwargs = {k: v for k, v in opt_cfg.items() if k != "type"}

        optimizer = opt_class(model.parameters(), **kwargs)
        return optimizer

    def build_scheduler(self, optimizer: torch.optim.Optimizer) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Build and return the learning rate scheduler if specified.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer to which the scheduler will be attached.

        Returns:
            torch.optim.lr_scheduler._LRScheduler or None: The instantiated scheduler, or None if not configured.
        """
        sch_cfg = self.config.get("scheduler", {})
        sch_type = sch_cfg.get("type", "")

        if not sch_type:
            return None

        if not hasattr(lr_scheduler, sch_type):
            raise ValueError(f"Scheduler '{sch_type}' is not found in torch.optim.lr_scheduler.")

        # Dynamically get scheduler class from torch.optim.lr_scheduler
        sch_class = getattr(lr_scheduler, sch_type)
        kwargs = {k: v for k, v in sch_cfg.items() if k != "type"}

        scheduler = sch_class(optimizer, **kwargs)
        return scheduler
