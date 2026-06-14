# tests/test_step01_architecture.py
"""第一步：验证网络架构 —— 节点/边/层的创建与连接关系"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.network import Network
from core.node import Node
from core.edge import Edge
from ops.activations import Sigmoid, ReLU


def test_create_input_layer():
    """输入层应创建正确数量的节点"""
    net = Network()
    inp = net.add_input_layer(3)
    assert len(inp) == 3, f"输入层节点数错误: {len(inp)}"
    for node in inp:
        assert isinstance(node, Node)
    print("✅ 输入层创建验证通过")


def test_create_dense_layer():
    """全连接层应正确创建节点和边"""
    net = Network()
    inp = net.add_input_layer(2)
    hidden = net.add_dense_layer(inp, 3, Sigmoid)
    assert len(hidden) == 3, f"隐藏层节点数错误: {len(hidden)}"
    # 每个隐藏节点应有 2 条输入边（来自输入层的2个节点）
    for node in hidden:
        assert len(node.input_edges) == 2, \
            f"隐藏节点输入边数错误: {len(node.input_edges)}"
    print("✅ 全连接层创建验证通过")


def test_full_connectivity():
    """全连接：前一层的每个节点都应连接到后一层的每个节点"""
    net = Network()
    inp = net.add_input_layer(4)
    out = net.add_dense_layer(inp, 2, Sigmoid)
    # 输入节点 → 输出节点: 4*2 = 8 条边
    total_edges = 0
    for node in inp:
        total_edges += len(node.output_edges)
    assert total_edges == 8, f"边数错误: {total_edges}, 期望 8"
    print("✅ 全连接性验证通过")


def test_multi_layer_architecture():
    """多层网络架构验证"""
    net = Network()
    inp = net.add_input_layer(2)
    h1 = net.add_dense_layer(inp, 4, ReLU)
    h2 = net.add_dense_layer(h1, 3, Sigmoid)
    out = net.add_dense_layer(h2, 1, Sigmoid)
    net.set_output_nodes(out)

    # 验证节点总数
    all_nodes = set(inp + h1 + h2 + out)
    assert len(net._all_nodes) == len(all_nodes), "节点数不一致"

    # 验证边数: 2*4 + 4*3 + 3*1 = 8+12+3 = 23
    assert len(net._all_edges) == 23, f"边数错误: {len(net._all_edges)}, 期望 23"
    print("✅ 多层网络架构验证通过")


def test_node_activation_fn():
    """节点应绑定正确的激活函数"""
    net = Network()
    inp = net.add_input_layer(2)
    hidden = net.add_dense_layer(inp, 3, ReLU)
    for node in hidden:
        assert isinstance(node.activation_fn, ReLU), "激活函数类型错误"
    print("✅ 激活函数绑定验证通过")


def test_edge_weight_initialization():
    """边的权重应已初始化且非零"""
    net = Network()
    inp = net.add_input_layer(3)
    out = net.add_dense_layer(inp, 2, Sigmoid)
    net.set_output_nodes(out)
    for edge in net._all_edges:
        assert edge.weight != 0.0, "边权重未初始化或为零"
    print("✅ 权重初始化验证通过")


if __name__ == "__main__":
    test_create_input_layer()
    test_create_dense_layer()
    test_full_connectivity()
    test_multi_layer_architecture()
    test_node_activation_fn()
    test_edge_weight_initialization()
    print("\n🎉 第一步所有测试通过！网络架构已就绪！")
