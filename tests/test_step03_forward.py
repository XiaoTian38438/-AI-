# tests/test_step03_forward.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.network import Network
from ops.activations import Sigmoid, ReLU, Tanh


def build_test_network():
    """构建 8→16→16→4 测试网络"""
    net = Network()
    inp = net.add_input_layer(8)
    h1 = net.add_dense_layer(inp, 16, ReLU)
    h2 = net.add_dense_layer(h1, 16, Tanh)
    out = net.add_dense_layer(h2, 4, Sigmoid)
    net.set_output_nodes(out)
    return net


def test_topology_order():
    net = build_test_network()
    # 验证拓扑序长度 = 总节点数
    assert len(net._topo_order) == 8 + 16 + 16 + 4
    # 验证输入节点在最前面
    for n in net.input_nodes:
        assert n in net._topo_order[:8]
    # 验证输出节点在最后面
    for n in net.output_nodes:
        assert n in net._topo_order[-4:]
    print("✅ 拓扑排序验证通过")


def test_forward_shape():
    net = build_test_network()
    x = np.random.randn(8)
    y = net.forward(x)
    assert y.shape == (4,), f"输出形状错误: {y.shape}"
    print(f"✅ 前向传播形状验证通过: {y}")


def test_sigmoid_output_range():
    """Sigmoid 输出应在 (0, 1) 之间"""
    net = build_test_network()
    for _ in range(10):
        x = np.random.randn(8) * 10  # 大输入测试数值稳定性
        y = net.forward(x)
        assert np.all(y > 0) and np.all(y < 1), \
            f"Sigmoid 输出越界: {y}"
    print("✅ Sigmoid 数值稳定性验证通过")


def test_deterministic_forward():
    """相同输入应产生相同输出"""
    net = build_test_network()
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    y1 = net.forward(x)
    y2 = net.forward(x)
    assert np.allclose(y1, y2), "前向传播不确定!"
    print("✅ 前向传播确定性验证通过")


def test_network_summary():
    net = build_test_network()
    net.summary()
    # 8*16 + 16*16 + 16*4 = 128+256+64 = 448 weights
    assert len(net._all_edges) == 448
    # biases: 16+16+4 = 36
    print("✅ 网络结构统计验证通过")


if __name__ == "__main__":
    test_topology_order()
    test_forward_shape()
    test_sigmoid_output_range()
    test_deterministic_forward()
    test_network_summary()
    print("\n🎉 第三步所有测试通过！前向传播已就绪！")