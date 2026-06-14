# ops/activations.py
import numpy as np
from abc import ABC, abstractmethod


class Activation(ABC):
    @abstractmethod
    def forward(self, z: float) -> float:
        pass

    def derivative(self, z: float) -> float:
        """默认使用 backward 接口，但为统一保留导数方法"""
        pass


class Sigmoid(Activation):
    def forward(self, z: float) -> float:
        if z >= 0:
            return 1.0 / (1.0 + np.exp(-z))
        else:
            ez = np.exp(z)
            return ez / (1.0 + ez)

    def derivative(self, z: float) -> float:
        s = self.forward(z)
        return s * (1.0 - s)


class ReLU(Activation):
    def forward(self, z: float) -> float:
        return max(0.0, z)

    def derivative(self, z: float) -> float:
        return 1.0 if z > 0 else 0.0


class Tanh(Activation):
    def forward(self, z: float) -> float:
        return float(np.tanh(z))

    def derivative(self, z: float) -> float:
        t = np.tanh(z)
        return 1.0 - t * t


class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, z: float) -> float:
        return z if z > 0 else self.alpha * z

    def derivative(self, z: float) -> float:
        return 1.0 if z > 0 else self.alpha


class Linear(Activation):
    def forward(self, z: float) -> float:
        return z

    def derivative(self, z: float) -> float:
        return 1.0

# Identity 是 Linear 的别名，两者功能完全相同
Identity = Linear