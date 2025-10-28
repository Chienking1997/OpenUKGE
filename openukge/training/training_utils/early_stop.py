import os
import torch


class EarlyStop:
    """
    Early stopping utility to prevent overfitting by stopping training
    when a monitored metric has stopped improving.
    """

    def __init__(self, patience=5, min_delta=0.0, monitor="mse", mode="min", monitor_mode=""):
        """
        Args:
            patience (int): Number of epochs with no improvement
                            after which training will be stopped.
            min_delta (float): Minimum performance improvement required
                               to reset the patience counter.
            monitor (str): Metric name to monitor (e.g., "val_loss", "mse").
            mode (str): "min" means lower metric value is better,
                        "max" means higher metric value is better.
            monitor_mode (str): Additional label for tracking which metric
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.mode = mode
        self.best_value = None
        self.counter = 0
        self.early_stop = False

        # Determine improvement direction
        if mode == "min":
            self.is_improvement = lambda current, best: current < best - self.min_delta
        elif mode == "max":
            self.is_improvement = lambda current, best: current > best + self.min_delta
        else:
            raise ValueError("mode must be 'min' or 'max'")

    def __call__(self, current_value, model=None, save_path=None):
        """
        Evaluate metric after each epoch and check early stopping condition.

        Args:
            current_value (float): Current monitored metric.
            model (torch.nn.Module, optional): Model to save when improved.
            save_path (str, optional): Path to save the best model.

        Returns:
            (bool, float): early_stop_flag, best_value
        """
        if self.best_value is None:
            # First epoch → initialize best value
            self.best_value = current_value
            self.save_best_model(model, save_path)

        elif self.is_improvement(current_value, self.best_value):
            # Reset counter upon improvement
            self.best_value = current_value
            self.counter = 0
            if model is not None and save_path is not None:
                self.save_best_model(model, save_path)

        else:
            # No improvement → increase counter
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop, self.best_value

    def get_monitor(self):
        """Return monitored metric name"""
        return self.monitor

    def get_monitor_mode(self):
        """Return monitored metric mode"""
        return self.monitor_mode

    @staticmethod
    def save_best_model(model, save_path):
        """
        Save model state dict to file. Automatically creates directory if missing.
        """
        if save_path is None or model is None:
            return

        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        torch.save(model.state_dict(), save_path)
        # print(f"[EarlyStop] Best model saved at: {save_path}")
