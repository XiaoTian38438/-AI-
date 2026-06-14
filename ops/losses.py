# ops/losses.py
import numpy as np
from abc import ABC, abstractmethod

# 检测 CuPy
try:
    import cupy as _cp
    _test = _cp.array([1.0])
    _ = _test + _test
    _HAS_CP = True
except Exception:
    _cp = None
    _HAS_CP = False


class Loss(ABC):
    @abstractmethod
    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        pass

    @abstractmethod
    def backward(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        """返回 dL/da_output, shape (output_size,)"""
        pass

    def forward_batch(self, preds, targets):
        """向量化批量前向，返回 (total_loss, grad_array)。
        preds/targets 可以是 numpy 或 cupy 数组。
        子类可覆写此方法以获得更好的性能。"""
        xp = _cp if _HAS_CP and type(preds).__module__.startswith('cupy') else np
        preds_cpu = _cp.asnumpy(preds) if xp is _cp else np.asarray(preds)
        targets_cpu = np.asarray(targets)
        bs = len(preds)
        total_loss = 0.0
        grad_cpu = np.zeros_like(preds_cpu)
        for i in range(bs):
            total_loss += self.forward(preds_cpu[i], targets_cpu[i])
            grad_cpu[i] = self.backward(preds_cpu[i], targets_cpu[i])
        grads = xp.asarray(grad_cpu) if xp is _cp else grad_cpu
        return total_loss, grads


class MSELoss(Loss):
    def forward(self, pred, target):
        return float(np.mean((pred - target) ** 2))

    def backward(self, pred, target):
        n = len(pred)
        return 2.0 * (pred - target) / n

    def forward_batch(self, preds, targets):
        xp = _cp if _HAS_CP and type(preds).__module__.startswith('cupy') else np
        diff = preds - xp.asarray(targets, dtype=preds.dtype)
        total_loss = float(xp.mean(diff ** 2)) * len(preds)
        n = preds.shape[-1] if preds.ndim > 1 else len(preds[0])
        grad = 2.0 * diff / n
        return total_loss, grad


class CrossEntropyLoss(Loss):
    EPS = 1e-7

    def forward(self, pred, target):
        p = np.clip(pred, self.EPS, 1 - self.EPS)
        return float(-np.mean(target * np.log(p) + (1 - target) * np.log(1 - p)))

    def backward(self, pred, target):
        p = np.clip(pred, self.EPS, 1 - self.EPS)
        n = len(pred)
        return (-target / p + (1 - target) / (1 - p)) / n

    def forward_batch(self, preds, targets):
        xp = _cp if _HAS_CP and type(preds).__module__.startswith('cupy') else np
        p = xp.clip(preds, self.EPS, 1 - self.EPS)
        t = xp.asarray(targets, dtype=preds.dtype)
        total_loss = -float(xp.sum(t * xp.log(p) + (1 - t) * xp.log(1 - p))) / preds.shape[-1]
        n = preds.shape[-1]
        grad = (-t / p + (1 - t) / (1 - p)) / n
        return total_loss, grad


class SoftmaxCrossEntropyLoss(Loss):
    EPS = 1e-7

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        pred = pred - np.max(pred)
        exp_pred = np.exp(pred)
        softmax = exp_pred / np.sum(exp_pred)

        if isinstance(target, (int, np.integer)):
            return -np.log(softmax[int(target)] + self.EPS)
        else:
            return -np.sum(target * np.log(softmax + self.EPS))

    def backward(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        pred = pred - np.max(pred)
        exp_pred = np.exp(pred)
        softmax = exp_pred / np.sum(exp_pred)

        if isinstance(target, (int, np.integer)):
            grad = softmax.copy()
            grad[int(target)] -= 1.0
        else:
            grad = softmax - target
        return grad

    def forward_batch(self, preds, targets):
        """GPU 加速的批量 softmax-cross-entropy"""
        xp = _cp if _HAS_CP and type(preds).__module__.startswith('cupy') else np

        # 数值稳定 softmax
        logits = preds - xp.max(preds, axis=1, keepdims=True)
        exp_logits = xp.exp(logits)
        softmax = exp_logits / xp.sum(exp_logits, axis=1, keepdims=True)

        bs = preds.shape[0]

        # 计算 loss
        if isinstance(targets, (int, np.integer)) or (isinstance(targets, np.ndarray) and targets.ndim == 1):
            # targets 是类别索引
            targets_xp = xp.asarray(targets, dtype=xp.int32) if not type(targets).__module__.startswith('cupy') else targets
            log_probs = xp.log(softmax + self.EPS)
            selected = log_probs[xp.arange(bs), targets_xp]
            total_loss = -float(xp.sum(selected))
            # 梯度: softmax - one_hot
            grad = softmax.copy()
            grad[xp.arange(bs), targets_xp] -= 1.0
        else:
            # one-hot targets
            targets_xp = xp.asarray(targets, dtype=xp.float64) if not type(targets).__module__.startswith('cupy') else targets
            total_loss = -float(xp.sum(targets_xp * xp.log(softmax + self.EPS)))
            grad = softmax - targets_xp

        return total_loss, grad
