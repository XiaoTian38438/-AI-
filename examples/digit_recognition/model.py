import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pickle
import numpy as np
from core.network import Network
from ops.activations import LeakyReLU, Linear
from ops.losses import SoftmaxCrossEntropyLoss
from engine.optimizers import Adam
from engine.scheduler import StepLR
from engine.trainer import Trainer
from examples.digit_recognition.data_loader import load_mnist_csv

INPUT_DIM = 784  # MNIST 28x28


def build_digit_net():
    """构建数字识别网络：784 → 256 → 128 → 64 → 10 (带 Dropout)"""
    net = Network(use_gpu=True)
    inp = net.add_input_layer(INPUT_DIM)
    h1 = net.add_dense_layer(inp, 256, LeakyReLU)
    h2 = net.add_dense_layer(h1, 128, LeakyReLU)
    h3 = net.add_dense_layer(h2, 64, LeakyReLU)
    out = net.add_dense_layer(h3, 10, Linear)
    net.set_output_nodes(out)
    # 每层 dropout: 隐藏层 0.2~0.3, 输出层 0
    net._layer_dropout = [0.2, 0.3, 0.3, 0.0]
    return net


def save_trained_weights(net, path):
    data = {
        'edges': [(e.weight, e.weight_grad) for e in net._all_edges],
        'biases': [(n.bias, n.bias_grad) for n in net._all_nodes if n not in net.input_nodes],
    }
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"权重已保存到 {path}")


def load_trained_weights(net, path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    for edge, (w, _) in zip(net._all_edges, data['edges']):
        edge.weight = w
    for node, (b, _) in zip(
        [n for n in net._all_nodes if n not in net.input_nodes],
        data['biases']
    ):
        node.bias = b
    # 标记需要重建向量化缓存
    net._vm_dirty = True
    print(f"权重已从 {path} 加载")


def train_digit_model(n_train=0, n_test=0, epochs=60, batch_size=256, lr=0.001):
    """训练数字识别模型（使用真实 MNIST 数据）
    n_train=0 表示使用全部训练数据
    """
    print("=" * 50)
    print("数字识别模型训练 (MNIST)")
    print("=" * 50)

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    train_csv = os.path.join(data_dir, 'mnist_train.csv')
    test_csv = os.path.join(data_dir, 'mnist_test.csv')

    if not os.path.exists(train_csv):
        print(f"错误: 找不到 {train_csv}")
        print("请将 MNIST CSV 文件放入 data/ 目录")
        return None, None

    # 加载数据
    print("加载训练数据...")
    X_all, y_all = load_mnist_csv(train_csv)
    print(f"  总样本数: {len(y_all)}")

    # 限制样本数量（0=全部使用）
    if n_train and n_train < len(y_all):
        idx = np.random.choice(len(y_all), n_train, replace=False)
        X_all, y_all = X_all[idx], y_all[idx]

    # 划分训练/验证
    n_val = min(1000, len(y_all) // 10)
    X_train, y_train = X_all[n_val:], y_all[n_val:]
    X_val, y_val = X_all[:n_val], y_all[:n_val]
    print(f"  训练集: {len(y_train)}, 验证集: {len(y_val)}")

    # 测试集
    if os.path.exists(test_csv):
        X_test, y_test = load_mnist_csv(test_csv)
        if n_test and n_test < len(y_test):
            idx = np.random.choice(len(y_test), n_test, replace=False)
            X_test, y_test = X_test[idx], y_test[idx]
        print(f"  测试集: {len(y_test)}")
    else:
        X_test, y_test = X_val, y_val

    # 构建网络
    net = build_digit_net()
    net.summary()

    loss_fn = SoftmaxCrossEntropyLoss()
    optimizer = Adam(net.get_parameters(), lr=lr, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
    trainer = Trainer(net, loss_fn, optimizer, scheduler=scheduler, grad_clip=5.0)

    print(f"\n开始训练: epochs={epochs}, batch_size={batch_size}, lr={lr}")
    trainer.train(X_train, y_train, epochs=epochs, batch_size=batch_size,
                  X_val=X_val, y_val=y_val, verbose=True)

    # 评估测试集
    preds = trainer.predict(X_test)
    pred_labels = np.array([np.argmax(p) for p in preds])
    accuracy = np.mean(pred_labels == y_test)
    print(f"\n测试集准确率: {accuracy * 100:.1f}%")

    # 每类准确率
    for d in range(10):
        mask = y_test == d
        if mask.sum() > 0:
            acc_d = np.mean(pred_labels[mask] == d)
            print(f"  数字 {d}: {acc_d * 100:.0f}%")

    # 保存权重
    save_trained_weights(net, os.path.join(data_dir, 'digit_weights.pkl'))
    return net, trainer


def get_or_train_net():
    """获取已训练的网络，若权重不存在则自动训练"""
    net = build_digit_net()
    weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'digit_weights.pkl')
    if os.path.exists(weights_path):
        load_trained_weights(net, weights_path)
    else:
        print("[自动训练] 未找到权重文件，开始训练...")
        train_digit_model()
    return net


if __name__ == "__main__":
    train_digit_model()
