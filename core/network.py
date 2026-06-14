# core/network.py
from collections import deque
from typing import List, Optional, Type
import numpy as np

from core.node import Node
from core.edge import Edge
from ops.activations import Activation, Sigmoid

# 自动检测 CuPy GPU 加速
_GPU = False
cp = None
try:
    import cupy as _cp
    # 验证 CuPy 真能用（不只是 import 成功）
    _test = _cp.array([1.0, 2.0])
    _ = _test @ _test
    cp = _cp
    _GPU = True
except Exception:
    _GPU = False


def gpu_available():
    """检查 GPU 是否可用"""
    return _GPU


class Network:
    def __init__(self, use_gpu=None):
        self.input_nodes: List[Node] = []
        self.output_nodes: List[Node] = []
        self._all_nodes: List[Node] = []
        self._all_edges: List[Edge] = []
        self._topo_order: List[Node] = []
        self._node_counter = 0

        # 向量化加速
        self._vm = None
        self._vm_dirty = True

        # 每层 dropout 配置（在 add_dense_layer 之后设置）
        self._layer_dropout: List[float] = []

        # GPU 设置
        if use_gpu is None:
            use_gpu = _GPU
        self.use_gpu = use_gpu and _GPU
        if self.use_gpu:
            print(f"[GPU] CuPy 加速已启用 ({cp.cuda.runtime.getDeviceProperties(0)['name'].decode()})")
        else:
            print("[CPU] 使用 NumPy 计算（安装 cupy 可启用 GPU 加速：pip install cupy-cuda12x）")

    def _new_node(self, name: str = "", activation_fn: Optional[Activation] = None) -> Node:
        node = Node(self._node_counter, name, activation_fn)
        self._node_counter += 1
        self._all_nodes.append(node)
        return node

    def add_input_layer(self, size: int) -> List[Node]:
        nodes = [self._new_node(f"in_{i}") for i in range(size)]
        self.input_nodes = nodes
        return nodes

    def add_dense_layer(self, prev_nodes: List[Node], size: int,
                        activation_fn: Type[Activation] = Sigmoid,
                        bias_init: float = 0.0) -> List[Node]:
        act_instance = activation_fn()
        new_nodes = []
        for i in range(size):
            node = self._new_node(f"h{len(self._all_nodes)}_{i}", act_instance)
            node.bias = bias_init
            new_nodes.append(node)

        for src in prev_nodes:
            for tgt in new_nodes:
                edge = Edge(src, tgt)
                self._all_edges.append(edge)
        return new_nodes

    def set_output_nodes(self, nodes: List[Node]):
        self.output_nodes = nodes
        self._build_topo_order()
        self._vm_dirty = True

    def _build_topo_order(self):
        in_degree = {n.id: len(n.input_edges) for n in self._all_nodes}
        queue = deque([n for n in self.input_nodes])
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for edge in node.output_edges:
                target = edge.target
                in_degree[target.id] -= 1
                if in_degree[target.id] == 0:
                    queue.append(target)
        if len(order) != len(self._all_nodes):
            raise ValueError("图中存在环或孤立节点，无法完成拓扑排序！")
        self._topo_order = order

    # ===================== 原始逐节点接口（兼容） =====================

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self._topo_order:
            raise RuntimeError("请先调用 set_output_nodes() 构建网络")
        if len(x) != len(self.input_nodes):
            raise ValueError(f"输入维度不匹配: 期望{len(self.input_nodes)}, 得到{len(x)}")
        for node, val in zip(self.input_nodes, x):
            node.activation = float(val)
        for node in self._topo_order:
            if node in self.input_nodes:
                continue
            node.forward()
        return np.array([n.activation for n in self.output_nodes])

    def backward(self, loss_gradient: np.ndarray):
        if len(loss_gradient) != len(self.output_nodes):
            raise ValueError(f"损失梯度维度不匹配")
        for node, grad in zip(self.output_nodes, loss_gradient):
            node.gradient = float(grad)
        for node in reversed(self._topo_order):
            if node in self.input_nodes:
                continue
            node.backward()

    def reset_gradients(self):
        for node in self._all_nodes:
            node.reset_gradient()
        for edge in self._all_edges:
            edge.reset_grad()

    def get_parameters(self):
        params = []
        for edge in self._all_edges:
            params.append(('weight', edge))
        for node in self._all_nodes:
            if node not in self.input_nodes:
                params.append(('bias', node))
        return params

    # ===================== 向量化加速接口 =====================

    def _ensure_vm(self):
        if self._vm is None or self._vm_dirty:
            self._vm = _VectorizedModel(self)
            self._vm_dirty = False

    def forward_batch(self, X: np.ndarray) -> np.ndarray:
        """向量化前向传播 X: (batch, input_dim) -> (batch, output_dim)"""
        self._ensure_vm()
        return self._vm.forward(X)

    def train_batch_vm(self, X_batch, y_batch, loss_fn):
        """向量化训练一个 batch，返回平均 loss"""
        self._ensure_vm()
        vm = self._vm

        # 前向传播（矩阵运算，GPU/CPU）
        preds = vm.forward(X_batch)  # (bs, output_dim)

        # 批量计算损失和梯度（尽量在 GPU 上）
        if hasattr(loss_fn, 'forward_batch'):
            batch_loss, grad = loss_fn.forward_batch(preds, y_batch)
        else:
            xp = vm.xp
            preds_cpu = cp.asnumpy(preds) if self.use_gpu else preds
            bs = len(X_batch)
            batch_loss = 0.0
            grad_cpu = np.zeros_like(preds_cpu)
            for i in range(bs):
                batch_loss += loss_fn.forward(preds_cpu[i], y_batch[i])
                grad_cpu[i] = loss_fn.backward(preds_cpu[i], y_batch[i])
            grad = xp.asarray(grad_cpu) if self.use_gpu else grad_cpu

        # 向量化反向传播
        vm.backward(grad)

        # 将梯度同步到边和节点
        vm.sync_grads_to_nodes()

        return batch_loss / len(X_batch)

    def update_vm_weights(self):
        """优化器更新权重后，将节点/边的权重复制到向量化模型"""
        if self._vm is not None:
            self._vm.sync_weights_from_nodes()

    def summary(self):
        print(f"\n{'='*50}")
        print(f"Network Summary")
        print(f"{'='*50}")
        print(f"  Input nodes:  {len(self.input_nodes)}")
        print(f"  Output nodes: {len(self.output_nodes)}")
        print(f"  Total nodes:  {len(self._all_nodes)}")
        print(f"  Total edges:  {len(self._all_edges)}")
        total_params = len(self._all_edges) + sum(1 for n in self._all_nodes if n not in self.input_nodes)
        print(f"  Total params: {total_params} (weights={len(self._all_edges)}, biases={total_params - len(self._all_edges)})")
        print(f"  Compute:      {'GPU (CuPy)' if self.use_gpu else 'CPU (NumPy)'}")
        print(f"{'='*50}\n")


class _VectorizedModel:
    """将图网络转为矩阵运算，加速批量前向/反向传播"""

    def __init__(self, net: Network):
        self.net = net
        self.use_gpu = net.use_gpu
        self.xp = cp if self.use_gpu else np
        self.layers = []
        self._training = True
        self._build()

    def _build(self):
        net = self.net
        xp = self.xp
        non_input = [n for n in net._topo_order if n not in net.input_nodes]

        # 按拓扑序分组
        layers_nodes = []
        input_id_set = frozenset(id(n) for n in net.input_nodes)
        remaining = list(non_input)
        prev_layer_ids = input_id_set

        while remaining:
            layer = []
            new_remaining = []
            for node in remaining:
                src_ids = frozenset(id(e.source) for e in node.input_edges)
                if src_ids == prev_layer_ids:
                    layer.append(node)
                else:
                    new_remaining.append(node)
            if not layer:
                layer = new_remaining
                new_remaining = []
            layers_nodes.append(layer)
            prev_layer_ids = frozenset(id(n) for n in layer)
            remaining = new_remaining

        # 获取每层 dropout 配置
        dropout_config = getattr(net, '_layer_dropout', [])

        # 判断是否需要从 edge/node 读取已有权重
        # 当 _vm_dirty=True（load_trained_weights 后）或 _vm 非空（重建时）时，读取已有权重
        has_existing_weights = net._vm_dirty or net._vm is not None

        # 构建权重矩阵
        prev_nodes = list(net.input_nodes)
        prev_id_to_idx = {id(n): i for i, n in enumerate(prev_nodes)}

        for layer_idx, layer_nodes in enumerate(layers_nodes):
            n_in = len(prev_nodes)
            n_out = len(layer_nodes)

            act = layer_nodes[0].activation_fn
            act_name = act.__class__.__name__ if act else "Linear"

            if has_existing_weights:
                # 从已有的 edge 权重构建（load_trained_weights 后重建）
                W_np = np.zeros((n_out, n_in), dtype=np.float64)
            else:
                # 矩阵级 He/Xavier 初始化（比逐边随机更均匀）
                if act_name in ("ReLU", "LeakyReLU"):
                    scale = np.sqrt(2.0 / n_in)   # He 初始化
                else:
                    scale = np.sqrt(2.0 / (n_in + n_out))  # Xavier 初始化
                W_np = np.random.randn(n_out, n_in).astype(np.float64) * scale

            edge_map = {}
            for j, tgt in enumerate(layer_nodes):
                for edge in tgt.input_edges:
                    i = prev_id_to_idx[id(edge.source)]
                    edge_map[id(edge)] = (j, i)
                    if has_existing_weights:
                        W_np[j, i] = edge.weight
                    else:
                        edge.weight = float(W_np[j, i])  # 同步到 edge

            b_np = np.zeros(n_out, dtype=np.float64)
            for j, tgt in enumerate(layer_nodes):
                if has_existing_weights:
                    b_np[j] = tgt.bias
                else:
                    tgt.bias = 0.0

            W = xp.asarray(W_np) if self.use_gpu else W_np
            b = xp.asarray(b_np) if self.use_gpu else b_np

            # 最后一个隐藏层不用 dropout
            is_output = (layer_idx == len(layers_nodes) - 1)
            dropout = dropout_config[layer_idx] if layer_idx < len(dropout_config) else 0.0
            if is_output:
                dropout = 0.0

            self.layers.append(_VecLayer(W, b, act, prev_nodes, layer_nodes, edge_map, xp, dropout=dropout))
            prev_nodes = layer_nodes
            prev_id_to_idx = {id(n): i for i, n in enumerate(prev_nodes)}

    def forward(self, X, training=True):
        xp = self.xp
        self._cache = []
        self._training = training
        if self.use_gpu:
            # 检测是否已经在 GPU 上
            is_on_gpu = type(X).__module__.startswith('cupy')
            h = X if is_on_gpu else cp.asarray(X, dtype=np.float64)
            if h.dtype != np.float64:
                h = h.astype(np.float64)
        else:
            h = np.asarray(X, dtype=np.float64)
        for layer in self.layers:
            z = h @ layer.W.T + layer.b  # (batch, n_out)
            self._cache.append((h, z))
            h = layer.activate(z, training=training)
        return h

    def backward(self, grad_output):
        """grad_output: xp array (batch, output_dim)，累加梯度"""
        delta = grad_output.astype(np.float64) if not self.use_gpu else grad_output.astype(np.float64)
        xp = self.xp
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            h_prev, z = self._cache[i]

            delta = delta * layer.activate_deriv(z)
            layer.W_grad += delta.T @ h_prev
            layer.b_grad += delta.sum(axis=0)
            delta = delta @ layer.W

    def sync_grads_to_nodes(self):
        """将向量化梯度同步到节点/边"""
        for layer in self.layers:
            W_grad = layer.W_grad
            b_grad = layer.b_grad
            if self.use_gpu:
                W_grad = cp.asnumpy(W_grad)
                b_grad = cp.asnumpy(b_grad)
            for eid, (r, c) in layer.edge_map.items():
                edge = _find_edge_by_id(self.net, eid)
                if edge is not None:
                    edge.weight_grad += float(W_grad[r, c])
            for j, tgt in enumerate(layer.tgt_nodes):
                tgt.bias_grad += float(b_grad[j])

    def sync_weights_from_nodes(self):
        """将节点/边的权重同步到向量化矩阵"""
        xp = self.xp
        for layer in self.layers:
            W_np = np.zeros(layer.W.shape, dtype=np.float64)
            for eid, (r, c) in layer.edge_map.items():
                edge = _find_edge_by_id(self.net, eid)
                if edge is not None:
                    W_np[r, c] = edge.weight
            b_np = np.array([n.bias for n in layer.tgt_nodes], dtype=np.float64)
            if self.use_gpu:
                layer.W = cp.asarray(W_np)
                layer.b = cp.asarray(b_np)
            else:
                layer.W = W_np
                layer.b = b_np

    def sync_weights_to_nodes_full(self):
        """训练结束后将矩阵权重复制回节点/边（用于 save/load）"""
        for layer in self.layers:
            W = layer.W
            b = layer.b
            if self.use_gpu:
                W = cp.asnumpy(W)
                b = cp.asnumpy(b)
            for eid, (r, c) in layer.edge_map.items():
                edge = _find_edge_by_id(self.net, eid)
                if edge is not None:
                    edge.weight = float(W[r, c])
            for j, tgt in enumerate(layer.tgt_nodes):
                tgt.bias = float(b[j])


# 全局边缓存
_edge_cache = {}

def _find_edge_by_id(net, edge_id):
    if not _edge_cache:
        for edge in net._all_edges:
            _edge_cache[id(edge)] = edge
    return _edge_cache.get(edge_id)

# 清除缓存（当网络结构变化时调用）
def _clear_edge_cache():
    global _edge_cache
    _edge_cache = {}


class _VecLayer:
    def __init__(self, W, b, act_fn, src_nodes, tgt_nodes, edge_map, xp, dropout=0.0):
        self.W = W
        self.b = b
        self.act_fn = act_fn
        self.src_nodes = src_nodes
        self.tgt_nodes = tgt_nodes
        self.edge_map = edge_map
        self.xp = xp
        self.W_grad = xp.zeros_like(W)
        self.b_grad = xp.zeros_like(b)
        self.dropout = dropout  # dropout 概率 (训练时使用)
        self._dropout_mask = None

    def activate(self, z, training=True):
        xp = self.xp
        name = self.act_fn.__class__.__name__ if self.act_fn else "Linear"
        if name == "ReLU":
            h = xp.maximum(0, z)
        elif name == "LeakyReLU":
            alpha = getattr(self.act_fn, 'alpha', 0.01)
            h = xp.where(z > 0, z, alpha * z)
        elif name == "Sigmoid":
            h = xp.where(z >= 0,
                            1.0 / (1.0 + xp.exp(-z)),
                            xp.exp(z) / (1.0 + xp.exp(z)))
        elif name == "Tanh":
            h = xp.tanh(z)
        else:
            h = z
        # Dropout (use numpy random to avoid curand dependency)
        if training and self.dropout > 0:
            mask = (np.random.rand(*h.shape) > self.dropout).astype(np.float32) / (1.0 - self.dropout)
            if _GPU and type(h).__module__.startswith('cupy'):
                mask = cp.asarray(mask)
            self._dropout_mask = mask
            h = h * self._dropout_mask
        else:
            self._dropout_mask = None
        return h

    def activate_deriv(self, z):
        xp = self.xp
        name = self.act_fn.__class__.__name__ if self.act_fn else "Linear"
        if name == "ReLU":
            d = (z > 0).astype(z.dtype)
        elif name == "LeakyReLU":
            alpha = getattr(self.act_fn, 'alpha', 0.01)
            d = xp.where(z > 0, 1.0, alpha)
        elif name == "Sigmoid":
            s = self.activate(z, training=False)
            d = s * (1.0 - s)
        elif name == "Tanh":
            t = xp.tanh(z)
            d = 1.0 - t * t
        else:
            d = xp.ones_like(z)
        # 反向传播时也要乘 dropout mask
        if self._dropout_mask is not None:
            d = d * self._dropout_mask
        return d
