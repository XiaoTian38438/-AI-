# tests/test_step02_data_structure.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.node import Node
from core.edge import Edge


def test_node_edge_connection():
    """验证节点和边的双向连接是否正确建立"""
    n1 = Node(0, "input_0")
    n2 = Node(1, "hidden_0")

    e = Edge(n1, n2, weight=0.5)

    # 验证拓扑注册
    assert len(n1.output_edges) == 1
    assert len(n2.input_edges) == 1
    assert n1.output_edges[0] is e
    assert n2.input_edges[0] is e

    # 验证边持有正确的节点引用
    assert e.source is n1
    assert e.target is n2
    print("✅ 拓扑连接测试通过")


def test_edge_forward_contribute():
    """验证边的前向贡献计算"""
    n1 = Node(0, "input_0")
    n2 = Node(1, "hidden_0")
    e = Edge(n1, n2, weight=0.3)

    n1.activation = 2.0
    contribution = e.forward_contribute()

    assert abs(contribution - 0.6) < 1e-7
    print("✅ 边前向贡献测试通过")


def test_gradient_locality():
    """
    核心测试：验证梯度本地化
    模拟手动设置梯度后，边能否仅通过本地信息计算权重梯度
    """
    n1 = Node(0, "input_0")
    n2 = Node(1, "hidden_0")
    e = Edge(n1, n2, weight=0.4)

    # 模拟前向传播后的状态
    n1.activation = 1.5
    # 模拟反向传播：假设 dL/da2 = 0.8
    n2.gradient = 0.8

    # 边仅通过本地信息计算 dL/dw
    e.accumulate_weight_grad()

    # dL/dw = dL/da2 * a1 = 0.8 * 1.5 = 1.2
    expected = 0.8 * 1.5
    assert abs(e.weight_grad - expected) < 1e-7
    print(f"✅ 梯度本地化测试通过 (dL/dw={e.weight_grad:.4f}, 期望={expected:.4f})")


def test_batch_gradient_accumulation():
    """验证多次 accumulate 能正确累加（mini-batch 基础）"""
    n1 = Node(0, "in")
    n2 = Node(1, "out")
    e = Edge(n1, n2, weight=0.5)

    n1.activation = 1.0
    n2.gradient = 0.3
    e.accumulate_weight_grad()  # 第一个样本

    n2.gradient = 0.7
    e.accumulate_weight_grad()  # 第二个样本

    # 累加: 0.3*1.0 + 0.7*1.0 = 1.0
    assert abs(e.weight_grad - 1.0) < 1e-7
    print("✅ Batch梯度累加测试通过")


if __name__ == "__main__":
    test_node_edge_connection()
    test_edge_forward_contribute()
    test_gradient_locality()
    test_batch_gradient_accumulation()
    print("\n🎉 第二步所有测试通过！")