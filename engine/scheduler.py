# engine/scheduler.py
import math


class _LRScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.last_epoch = -1

    def step(self):
        self.last_epoch += 1
        self._update_lr()

    def _update_lr(self):
        raise NotImplementedError


class StepLR(_LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def _update_lr(self):
        if (self.last_epoch + 1) % self.step_size == 0:
            self.optimizer.lr *= self.gamma


class ExponentialLR(_LRScheduler):
    def __init__(self, optimizer, gamma):
        super().__init__(optimizer)
        self.gamma = gamma

    def _update_lr(self):
        self.optimizer.lr *= self.gamma


class ReduceLROnPlateau:
    """当验证损失停止下降时降低学习率"""
    def __init__(self, optimizer, patience=5, factor=0.5, min_lr=1e-6):
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.best_loss = float('inf')
        self.counter = 0

    def step(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                new_lr = max(self.optimizer.lr * self.factor, self.min_lr)
                if new_lr != self.optimizer.lr:
                    self.optimizer.lr = new_lr
                    self.counter = 0
                return True   # 表示学习率已降低
        return False