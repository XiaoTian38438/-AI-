# utils/visualizer.py
import matplotlib.pyplot as plt
import numpy as np
from core.network import Network


def draw_network(net: Network, show_weights=True, node_size=600, figsize=(12, 8)):
    """可视化神经网络结构，节点颜色表示激活值，边宽表示权重大小"""
    plt.figure(figsize=figsize)

    # 分层
    layers = {}
    for node in net.input_nodes:
        layers[node.id] = 0
    changed = True
    while changed:
        changed = False
        for edge in net._all_edges:
            src_id, tgt_id = edge.source.id, edge.target.id
            if src_id in layers and tgt_id not in layers:
                layers[tgt_id] = layers[src_id] + 1
                changed = True
    max_layer = max(layers.values()) if layers else 0

    nodes_by_layer = {i: [] for i in range(max_layer+1)}
    for n in net._all_nodes:
        nodes_by_layer[layers[n.id]].append(n)

    pos = {}
    for layer, nodes in nodes_by_layer.items():
        y_vals = np.linspace(0, 1, len(nodes)+2)[1:-1]
        for idx, node in enumerate(nodes):
            pos[node.id] = (layer, y_vals[idx])

    # 边
    for edge in net._all_edges:
        src_pos = pos[edge.source.id]
        tgt_pos = pos[edge.target.id]
        weight = edge.weight
        linewidth = 1 + 2 * abs(weight) / (max(abs(e.weight) for e in net._all_edges) + 1e-8)
        plt.plot([src_pos[0], tgt_pos[0]], [src_pos[1], tgt_pos[1]],
                 'gray', linewidth=linewidth, alpha=0.7)
        if show_weights:
            mid_x = (src_pos[0] + tgt_pos[0]) / 2
            mid_y = (src_pos[1] + tgt_pos[1]) / 2
            plt.text(mid_x, mid_y, f"{weight:.2f}", fontsize=7, ha='center', va='center',
                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.6))

    # 节点
    for node in net._all_nodes:
        x, y = pos[node.id]
        act = node.activation
        color = plt.cm.RdBu(act if 0 <= act <= 1 else 0.5)
        plt.scatter(x, y, s=node_size, c=[color], edgecolors='black', zorder=3)
        label = node.name
        if node in net.input_nodes:
            label = f"IN:{label}"
        elif node in net.output_nodes:
            label = f"OUT:{label}"
        plt.text(x, y, label, ha='center', va='center', fontsize=8, fontweight='bold')

    plt.title("Neural Network Structure")
    plt.axis('off')
    plt.tight_layout()
    plt.show()