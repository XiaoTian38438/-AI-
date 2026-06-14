import numpy as np
import csv

def load_mnist_csv(data_path: str, normalize: bool = True):
    """加载 MNIST CSV，返回 (X, y)，X 形状为 (n_samples, 784)"""
    X, y = [], []
    with open(data_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            label = int(row[0])
            pixels = np.array([int(x) for x in row[1:]], dtype=np.float32)
            if normalize:
                pixels = pixels / 255.0   # 归一化到 [0,1]
            X.append(pixels)
            y.append(label)
    return np.array(X), np.array(y)