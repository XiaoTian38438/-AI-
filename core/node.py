# core/node.py
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.edge import Edge
    from ops.activations import Activation


class Node:
    def __init__(self, node_id: int, name: str = "", activation_fn=None):
        self.id = node_id
        self.name = name or f"node_{node_id}"
        self.activation_fn = activation_fn

        self.input_edges: List['Edge'] = []
        self.output_edges: List['Edge'] = []

        self.z: float = 0.0
        self.activation: float = 0.0
        self.bias: float = 0.0
        self.gradient: float = 0.0
        self.bias_grad: float = 0.0   # 偏置梯度累积器

    def add_input(self, edge: 'Edge'):
        self.input_edges.append(edge)

    def add_output(self, edge: 'Edge'):
        self.output_edges.append(edge)

    def reset_gradient(self):
        self.gradient = 0.0
        self.bias_grad = 0.0

    def forward(self):
        z = self.bias
        for edge in self.input_edges:
            z += edge.forward_contribute()
        self.z = z
        if self.activation_fn is not None:
            self.activation = self.activation_fn.forward(z)
        else:
            self.activation = z

    def backward(self):
        if self.activation_fn is not None:
            delta = self.gradient * self.activation_fn.derivative(self.z)
        else:
            delta = self.gradient

        self.bias_grad += delta

        for edge in self.input_edges:
            src = edge.source
            edge.weight_grad += delta * src.activation
            src.gradient += delta * edge.weight

    def __repr__(self):
        return f"Node({self.name}, a={self.activation:.4f}, grad={self.gradient:.6f})"