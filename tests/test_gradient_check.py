import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.network import Network
from ops.activations import Sigmoid
from ops.losses import MSELoss

def numerical_gradient_check():
    np.random.seed(42)
    net = Network()
    inp = net.add_input_layer(2)
    h = net.add_dense_layer(inp, 2, Sigmoid)
    out = net.add_dense_layer(h, 1, Sigmoid)
    net.set_output_nodes(out)

    loss_fn = MSELoss()
    x = np.array([0.3, -0.2])
    y = np.array([0.7])
    eps = 1e-6
    tol = 1e-6

    # 解析梯度
    pred = net.forward(x)
    loss_grad = loss_fn.backward(pred, y)
    net.reset_gradients()
    net.backward(loss_grad)

    params = net.get_parameters()
    # 修正点：obj 而不是 edge
    analytic_grads = [obj.weight_grad if ptype=='weight' else obj.bias_grad for ptype, obj in params]

    # 数值梯度
    numeric_grads = []
    for idx, (ptype, obj) in enumerate(params):
        original = obj.weight if ptype=='weight' else obj.bias
        obj_val = lambda val: setattr(obj, 'weight' if ptype=='weight' else 'bias', val)
        obj_val(original + eps)
        loss_plus = loss_fn.forward(net.forward(x), y)
        obj_val(original - eps)
        loss_minus = loss_fn.forward(net.forward(x), y)
        obj_val(original)
        num_grad = (loss_plus - loss_minus) / (2 * eps)
        numeric_grads.append(num_grad)

    max_rel_err = 0.0
    for ana, num in zip(analytic_grads, numeric_grads):
        denom = max(abs(ana), abs(num), 1e-8)
        rel_err = abs(ana - num) / denom
        max_rel_err = max(max_rel_err, rel_err)
    print(f"最大相对误差: {max_rel_err:.2e}")
    assert max_rel_err < tol, "梯度验证失败"
    print("🎉 梯度数值验证通过！")

if __name__ == "__main__":
    numerical_gradient_check()