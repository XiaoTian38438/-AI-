# core/edge.py
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.node import Node


class Edge:
    def __init__(self, source: 'Node', target: 'Node', weight: float = None):
        self.source = source
        self.target = target

        if weight is None:
            fan_in = max(len(source.output_edges), 1)
            # 根据目标节点的激活函数选择合适的初始化
            act_name = target.activation_fn.__class__.__name__ if target.activation_fn else "Linear"
            if act_name in ("ReLU", "LeakyReLU"):
                scale = np.sqrt(2.0 / fan_in)   # He初始化
            else:
                scale = np.sqrt(1.0 / fan_in)   # Xavier初始化
            self.weight = float(np.random.randn() * scale)
        else:
            self.weight = float(weight)

        self.weight_grad = 0.0
        source.add_output(self)
        target.add_input(self)

    def forward_contribute(self) -> float:
        return self.weight * self.source.activation

    def accumulate_weight_grad(self):
        # 保留此方法以兼容旧代码，实际反向传播中已直接累加
        self.weight_grad += self.target.gradient * self.source.activation

    def reset_grad(self):
        self.weight_grad = 0.0

    def __repr__(self):
        return f"Edge({self.source.name}→{self.target.name}, w={self.weight:.4f})"