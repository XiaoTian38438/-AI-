# engine/optimizers.py
import numpy as np

# 检测 CuPy
try:
    import cupy as _cp
    _test = _cp.array([1.0])
    _ = _test + _test
    _HAS_CP = True
except Exception:
    _cp = None
    _HAS_CP = False


class Optimizer:
    def __init__(self, params):
        self.params = params   # list of ('weight', edge) or ('bias', node)

    def zero_grad(self):
        for ptype, obj in self.params:
            if ptype == 'weight':
                obj.weight_grad = 0.0
            else:
                obj.bias_grad = 0.0

    def step(self):
        raise NotImplementedError

    def step_vm(self, vm_layers):
        """直接在向量化矩阵上执行优化器更新，跳过逐边循环"""
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.state = {}

    def step(self):
        for idx, (ptype, obj) in enumerate(self.params):
            grad = obj.weight_grad if ptype == 'weight' else obj.bias_grad
            if self.weight_decay != 0:
                param = obj.weight if ptype == 'weight' else obj.bias
                grad += self.weight_decay * param

            if self.momentum != 0:
                if idx not in self.state:
                    self.state[idx] = 0.0
                self.state[idx] = self.momentum * self.state[idx] + grad
                grad = self.state[idx]

            if ptype == 'weight':
                obj.weight -= self.lr * grad
            else:
                obj.bias -= self.lr * grad

    def step_vm(self, vm_layers):
        """向量化 SGD 更新"""
        xp = _cp if _HAS_CP and type(vm_layers[0].W).__module__.startswith('cupy') else np
        for layer in vm_layers:
            # 权重更新
            grad = layer.W_grad
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * layer.W
            layer.W -= self.lr * grad
            # 偏置更新
            grad = layer.b_grad
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * layer.b
            layer.b -= self.lr * grad


class Adam(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.state = {'m': {}, 'v': {}, 't': 0}
        # 向量化 Adam 状态
        self._vm_state = {'m_W': [], 'v_W': [], 'm_b': [], 'v_b': [], 't': 0}

    def step(self):
        self.state['t'] += 1
        t = self.state['t']

        for idx, (ptype, obj) in enumerate(self.params):
            grad = obj.weight_grad if ptype == 'weight' else obj.bias_grad
            if self.weight_decay != 0:
                param = obj.weight if ptype == 'weight' else obj.bias
                grad += self.weight_decay * param

            if idx not in self.state['m']:
                self.state['m'][idx] = 0.0
                self.state['v'][idx] = 0.0

            m = self.state['m'][idx]
            v = self.state['v'][idx]

            m = self.betas[0] * m + (1 - self.betas[0]) * grad
            v = self.betas[1] * v + (1 - self.betas[1]) * (grad ** 2)
            self.state['m'][idx] = m
            self.state['v'][idx] = v

            m_hat = m / (1 - self.betas[0] ** t)
            v_hat = v / (1 - self.betas[1] ** t)
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            if ptype == 'weight':
                obj.weight -= update
            else:
                obj.bias -= update

    def step_vm(self, vm_layers):
        """向量化 Adam 更新 - 直接在矩阵上操作，比逐边快 1000x"""
        xp = _cp if _HAS_CP and type(vm_layers[0].W).__module__.startswith('cupy') else np

        self._vm_state['t'] += 1
        t = self._vm_state['t']
        b1, b2 = self.betas

        # 首次调用时初始化
        if not self._vm_state['m_W']:
            for layer in vm_layers:
                self._vm_state['m_W'].append(xp.zeros_like(layer.W))
                self._vm_state['v_W'].append(xp.zeros_like(layer.W))
                self._vm_state['m_b'].append(xp.zeros_like(layer.b))
                self._vm_state['v_b'].append(xp.zeros_like(layer.b))

        for i, layer in enumerate(vm_layers):
            # 权重梯度
            grad_W = layer.W_grad
            if self.weight_decay != 0:
                grad_W = grad_W + self.weight_decay * layer.W

            m_W = b1 * self._vm_state['m_W'][i] + (1 - b1) * grad_W
            v_W = b2 * self._vm_state['v_W'][i] + (1 - b2) * (grad_W ** 2)
            self._vm_state['m_W'][i] = m_W
            self._vm_state['v_W'][i] = v_W

            m_hat_W = m_W / (1 - b1 ** t)
            v_hat_W = v_W / (1 - b2 ** t)
            layer.W -= self.lr * m_hat_W / (xp.sqrt(v_hat_W) + self.eps)

            # 偏置梯度
            grad_b = layer.b_grad
            if self.weight_decay != 0:
                grad_b = grad_b + self.weight_decay * layer.b

            m_b = b1 * self._vm_state['m_b'][i] + (1 - b1) * grad_b
            v_b = b2 * self._vm_state['v_b'][i] + (1 - b2) * (grad_b ** 2)
            self._vm_state['m_b'][i] = m_b
            self._vm_state['v_b'][i] = v_b

            m_hat_b = m_b / (1 - b1 ** t)
            v_hat_b = v_b / (1 - b2 ** t)
            layer.b -= self.lr * m_hat_b / (xp.sqrt(v_hat_b) + self.eps)


# ---------------- 梯度裁剪 ----------------
def clip_grad_norm(network, max_norm=1.0):
    """梯度裁剪 - 支持向量化模式"""
    vm = getattr(network, '_vm', None)
    if vm is not None and vm.layers:
        xp = vm.xp
        total_norm = 0.0
        for layer in vm.layers:
            total_norm += float(xp.sum(layer.W_grad ** 2))
            total_norm += float(xp.sum(layer.b_grad ** 2))
        total_norm = total_norm ** 0.5
        if total_norm > max_norm:
            scale = max_norm / total_norm
            for layer in vm.layers:
                layer.W_grad *= scale
                layer.b_grad *= scale
    else:
        total_norm = 0.0
        for edge in network._all_edges:
            total_norm += edge.weight_grad ** 2
        for node in network._all_nodes:
            if node not in network.input_nodes:
                total_norm += node.bias_grad ** 2
        total_norm = np.sqrt(total_norm)

        if total_norm > max_norm:
            scale = max_norm / total_norm
            for edge in network._all_edges:
                edge.weight_grad *= scale
            for node in network._all_nodes:
                if node not in network.input_nodes:
                    node.bias_grad *= scale
