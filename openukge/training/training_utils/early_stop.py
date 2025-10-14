import torch


class EarlyStop:
    def __init__(self, patience=5, min_delta=0, monitor="mse", mode="min", monitor_mode=""):
        """
        :param patience: 模型允许在多少个epoch没有提升的情况下继续训练
        :param min_delta: 性能提升的最小变化，如果小于该值则认为性能没有提升
        :param monitor: 要监控的指标，比如 "val_loss" 或 "val_acc"
        :param mode: 'min' 表示最小化指标, 'max' 表示最大化指标
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.mode = mode
        self.best_value = None
        self.counter = 0
        self.early_stop = False

        # 如果模式是 'min'，则希望指标尽量小；如果模式是 'max'，则希望指标尽量大
        if self.mode == 'min':
            self.is_improvement = lambda current, best: current < best - self.min_delta
        elif self.mode == 'max':
            self.is_improvement = lambda current, best: current > best + self.min_delta
        else:
            raise ValueError("mode should be either 'min' or 'max'")

    def __call__(self, current_value, model=None, save_path=None):
        """
        在每次训练结束后调用该方法以检查是否应该停止训练
        :param current_value: 当前epoch中的监控指标的值
        :return: 是否需要停止训练
        """
        if self.best_value is None:
            # 初始化时将当前值作为最佳值
            self.best_value = current_value
            self.save_best_model(model, save_path)
        elif self.is_improvement(current_value, self.best_value):
            # 如果监控指标有显著提升，更新最佳值，并重置计数器
            self.best_value = current_value
            self.counter = 0
            # 保存最佳模型参数
            if model is not None and save_path is not None:
                self.save_best_model(model, save_path)
        else:
            # 否则计数器加一
            self.counter += 1
            if self.counter >= self.patience:
                # 如果计数器超过设定的耐心值，触发早停
                self.early_stop = True
        return self.early_stop, self.best_value

    def get_monitor(self):
        return self.monitor

    def get_monitor_mode(self):
        return self.monitor_mode

    @staticmethod
    def save_best_model(model, save_path):
        """
        保存最佳模型参数
        :param model: 当前模型实例
        :param save_path: 保存模型的路径
        """
        torch.save(model.state_dict(), save_path)
        # print(f"Best model saved to {save_path}")
