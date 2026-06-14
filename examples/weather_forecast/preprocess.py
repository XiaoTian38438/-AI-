import numpy as np


def create_sequences(df, feature_cols, target_col, seq_len=7):
    """
    滑动窗口构造样本
    用过去 seq_len 天的特征预测下一天
    返回: X (n_samples, seq_len * n_features), y (n_samples, 1)
    """
    features = df[feature_cols].values.astype(np.float64)
    targets = df[target_col].values.astype(np.float64)

    X_list, y_list = [], []
    for i in range(len(features) - seq_len):
        X_list.append(features[i:i + seq_len].flatten())
        y_list.append(targets[i + seq_len])

    X = np.array(X_list)
    y = np.array(y_list).reshape(-1, 1)
    return X, y


def normalize_train_test(X_train, X_test):
    """Z-score 标准化"""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std


def split_chronological(X, y, test_ratio=0.2):
    """按时间顺序切分"""
    n = len(X)
    split = int(n * (1 - test_ratio))
    return X[:split], y[:split], X[split:], y[split:]
