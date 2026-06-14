"""智能决策助手 - AI智能体系统

第九届全国青少年人工智能创新挑战赛 - AI智能体开发专项赛

使用方式:
  python main.py                  # 启动Web应用
  python main.py --demo           # 运行XOR/MNIST演示
  python main.py --workflow       # 仅展示工作流结构
"""
import sys
import os


def run_web_app():
    """启动智能决策助手 Web 应用"""
    from examples.smart_assistant.app import app
    from examples.smart_assistant.workflow import build_workflow
    from core.llm import is_available

    wf = build_workflow()
    print("=" * 50)
    print("智能决策助手 - AI智能体系统")
    print("第九届全国青少年人工智能创新挑战赛")
    print("=" * 50)
    print(f"  LLM 可用: {is_available()}")
    print(f"  工作流节点数: {len(wf.nodes)}")
    print(f"  访问地址: http://localhost:5000")
    print("=" * 50)

    # 打印工作流结构
    print("\n工作流节点:")
    for name, node in wf.nodes.items():
        branches = ''
        if node.branches:
            branches = ' → ' + ', '.join(f'{t}({l})' for _, t, l in node.branches)
        default = f' → {node.default_next}' if node.default_next else ''
        print(f"  [{node.node_type.value:12s}] {name}{branches}{default}")

    print()
    app.run(host='0.0.0.0', port=5000, debug=False)


def run_demo():
    """运行神经网络框架基础演示"""
    import numpy as np
    from core.network import Network
    from ops.activations import Sigmoid, ReLU
    from ops.losses import MSELoss, CrossEntropyLoss
    from engine.optimizers import Adam
    from engine.scheduler import ReduceLROnPlateau
    from engine.trainer import Trainer

    print("=" * 50)
    print("XOR Problem")
    print("=" * 50)

    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y_xor = np.array([[0], [1], [1], [0]], dtype=float)

    net = Network()
    inp = net.add_input_layer(2)
    hidden = net.add_dense_layer(inp, 4, Sigmoid)
    out = net.add_dense_layer(hidden, 1, Sigmoid)
    net.set_output_nodes(out)

    loss_fn = MSELoss()
    optimizer = Adam(net.get_parameters(), lr=0.1, weight_decay=1e-4)
    trainer = Trainer(net, loss_fn, optimizer, grad_clip=1.0)
    trainer.train(X_xor, y_xor, epochs=300, batch_size=4, verbose=True)

    print("\nXOR Results:")
    for xi, yi in zip(X_xor, y_xor):
        pred = net.forward(xi)
        print(f"Input {xi} -> Pred {pred[0]:.4f} (True {yi[0]})")

    # MNIST 子集演示
    try:
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        print("\n" + "=" * 50)
        print("MNIST (digits) Subset Test")
        print("=" * 50)

        digits = load_digits()
        X, y = digits.data, digits.target
        mask = (y == 0) | (y == 1)
        X, y = X[mask], y[mask]
        y = y.reshape(-1, 1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        net_mnist = Network()
        inp = net_mnist.add_input_layer(64)
        h1 = net_mnist.add_input_layer(inp, 32, ReLU)
        out = net_mnist.add_dense_layer(h1, 1, Sigmoid)
        net_mnist.set_output_nodes(out)

        trainer = Trainer(net_mnist, CrossEntropyLoss(),
                          Adam(net_mnist.get_parameters(), lr=0.01),
                          scheduler=ReduceLROnPlateau(Adam(net_mnist.get_parameters(), lr=0.01), patience=10),
                          grad_clip=1.0)
        trainer.train(X_train, y_train, epochs=100, batch_size=16,
                      X_val=X_test, y_val=y_test, early_stopping_patience=15, verbose=True)

        preds = trainer.predict(X_test)
        acc = np.mean((preds > 0.5).astype(int).flatten() == y_test.flatten())
        print(f"\nTest Accuracy: {acc * 100:.2f}%")
    except ImportError:
        print("\n[Info] sklearn not installed, skipping MNIST demo.")


def show_workflow():
    """展示工作流结构"""
    from examples.smart_assistant.workflow import build_workflow

    wf = build_workflow()
    print("=" * 50)
    print(f"工作流: {wf.name}")
    print(f"描述: {wf.description}")
    print(f"节点数: {len(wf.nodes)}")
    print("=" * 50)

    for name, node in wf.nodes.items():
        print(f"\n  [{node.node_type.value:12s}] {name}")
        print(f"    描述: {node.description}")
        if node.branches:
            for _, target, label in node.branches:
                print(f"    → {target} (条件: {label})")
        if node.default_next:
            print(f"    → {node.default_next} (默认)")

    print("\n" + "=" * 50)
    print("Mermaid 流程图:")
    print("=" * 50)
    print(wf.get_mermaid())


if __name__ == '__main__':
    if '--demo' in sys.argv:
        run_demo()
    elif '--workflow' in sys.argv:
        show_workflow()
    else:
        run_web_app()
