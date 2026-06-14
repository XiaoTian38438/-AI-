import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from core.network import Network
from ops.activations import Sigmoid, ReLU

def test_simple_network():
    net = Network()
    inp = net.add_input_layer(2)
    h = net.add_dense_layer(inp, 2, Sigmoid)
    out = net.add_dense_layer(h, 1, Sigmoid)
    net.set_output_nodes(out)

    # 手动设置权重偏置以便验证
    edges = net._all_edges
    # 输入->隐藏: w11=0.5, w21=0.5, w12=0.5, w22=0.5
    edges[0].weight = 0.5; edges[1].weight = 0.5
    edges[2].weight = 0.5; edges[3].weight = 0.5
    # 隐藏偏置
    h[0].bias = 0.0; h[1].bias = 0.0
    # 隐藏->输出
    edges[4].weight = 1.0; edges[5].weight = 1.0
    out[0].bias = 0.0

    x = np.array([1.0, 0.0])
    y = net.forward(x)
    # 隐藏层: z1 = 0 + 0.5*1 + 0.5*0 = 0.5, a1 = sigmoid(0.5)=0.6225
    #        z2 = 0 + 0.5*1 + 0.5*0 = 0.5, a2 = 0.6225
    # 输出: z = 0 + 1*0.6225 + 1*0.6225 = 1.245, a = sigmoid(1.245)=0.7764
    expected = 1 / (1 + np.exp(-1.245))
    assert abs(y[0] - expected) < 1e-4
    print("✅ 前向传播测试通过")

if __name__ == "__main__":
    test_simple_network()