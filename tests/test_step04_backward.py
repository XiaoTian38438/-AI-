# tests/test_step04_backward.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.network import Network
from ops.activations import Sigmoid, ReLU, Tanh
from ops.losses import MSELoss


def build_small_net():
    """用极小网络方便手动验算"""
    net = Network()
    inp = net.add_input_layer(2)
    h = net.add_dense_layer(inp, 2, Sigmoid)
    out = net.add_dense_layer(h, 1, Sigmoid)
    net.set_output_nodes(out)
    return net


def numerical_gradient_check():
    """
    有限差分法验证解析梯度
    |analytic - numeric| / max(|analytic|, |numeric|) < tol
    """
    np.random.seed(42)
    net = build_small_net()
    loss_fn = MSELoss()
    x = np.array([0.5, -0.3])
    y = np.array([0.8])
    eps = 1e-5
    tol = 1e-5

    # 解析梯度
    pred = net.forward(x)
    loss_val = loss_fn.forward(pred, y)
    loss_grad = loss_fn.backward(pred, y)
    net.reset_gradients()
    net.backward(loss_grad)

    # 收集所有解析梯度
    analytic_grads = []
    params = []
    for edge in net._all_edges:
        analytic_grads.append(edge.weight_grad)
        params.append(('weight', edge))
    for node in net._all_nodes:
        if node not in net.input_nodes:
            analytic_grads.append(node.bias_grad)  # bias梯度 = delta = grad * act'(z)
            params.append(('bias', node))

    # 数值梯度
    max_rel_error = 0.0
    errors = []
    for idx, (ptype, obj) in enumerate(params):
        if ptype == 'weight':
            original = obj.weight
            obj.weight = original + eps
            loss_plus = loss_fn.forward(net.forward(x), y)
            obj.weight = original - eps
            loss_minus = loss_fn.forward(net.forward(x), y)
            obj.weight = original
        else:  # bias
            original = obj.bias
            obj.bias = original + eps
            loss_plus = loss_fn.forward(net.forward(x), y)
            obj.bias = original - eps
            loss_minus = loss_fn.forward(net.forward(x), y)
            obj.bias = original

        num_grad = (loss_plus - loss_minus) / (2 * eps)
        ana_grad = analytic_grads[idx]
        denom = max(abs(ana_grad), abs(num_grad), 1e-8)
        rel_err = abs(ana_grad - num_grad) / denom
        max_rel_error = max(max_rel_error, rel_err)
        errors.append((ptype, obj, ana_grad, num_grad, rel_err))

    print(f"\n{'='*60}")
    print(f"梯度数值验证结果")
    print(f"{'='*60}")
    for ptype, obj, ana, num, err in errors:
        status = "✅" if err < tol else "❌"
        name = f"{obj.source.name}→{obj.target.name}" if ptype == 'weight' else obj.name
        print(f"  {status} {ptype:6s} {name:15s} | "
              f"analytic={ana:+.8f} numeric={num:+.8f} rel_err={err:.2e}")
    print(f"{'='*60}")
    print(f"最大相对误差: {max_rel_error:.2e} (阈值: {tol:.2e})")

    assert max_rel_error < tol, \
        f"梯度验证失败! 最大相对误差 {max_rel_error:.2e} > {tol:.2e}"
    print("🎉 反向传播梯度完全正确！")


if __name__ == "__main__":
    numerical_gradient_check()