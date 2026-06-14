"""股票/货币涨跌预测训练脚本"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from core.network import Network
from ops.activations import LeakyReLU, Sigmoid, Linear
from ops.losses import CrossEntropyLoss
from engine.optimizers import Adam
from engine.scheduler import ReduceLROnPlateau
from engine.trainer import Trainer
from utils.chart import save_stock_chart
from utils.time_utils import get_date_range_around_now

from examples.stock_forecast.data_fetcher import (
    fetch_stock_history, fetch_currency_history,
    add_stock_features, add_currency_features,
    extrapolate_stock_future, extrapolate_currency_future
)
from examples.stock_forecast.preprocess import create_sequences, split_chronological


STOCK_FEATURE_COLS = [
    '收盘', '收盘_lag1', '收盘_lag2', '成交量', '成交量_lag1', '换手率',
    'MA5', 'MA10', 'MA20', '偏离MA5', '偏离MA10',
    'RSI14', 'MACD', 'MACD_signal', '量比'
]


def _get_currency_feature_cols(df):
    price_col = '中间价' if '中间价' in df.columns else df.columns[1]
    return [
        price_col, f'{price_col}_lag1', f'{price_col}_lag2', f'{price_col}_lag3',
        'MA5', 'MA10', 'MA20', '偏离MA5', 'RSI14', '波动率5'
    ]


def _build_classifier(input_dim):
    """构建分类网络: input → 16 → 1 (极轻量防过拟合)"""
    net = Network(use_gpu=True)
    inp = net.add_input_layer(input_dim)
    h1 = net.add_dense_layer(inp, 16, LeakyReLU)
    out = net.add_dense_layer(h1, 1, Sigmoid)
    net.set_output_nodes(out)
    net._layer_dropout = [0.2, 0.0]
    return net


def train_stock_model(symbol='000001', start_date=None, end_date=None,
                      seq_len=1, epochs=80, batch_size=32, future_days=180):
    """训练股票涨跌预测模型
    seq_len=1: lag/MA/RSI 等特征已编码历史，无需滑动窗口重复
    仅用真实数据训练，推算数据用于未来预测可视化
    """
    if start_date is None or end_date is None:
        start_date, end_date, today = get_date_range_around_now(past_years=2, future_days=future_days)
    else:
        today = get_date_range_around_now()[2]

    print(f"{'='*50}")
    print(f"股票涨跌预测: {symbol}")
    print(f"时间范围: {start_date} ~ {end_date} (今天: {today})")
    print(f"{'='*50}")

    # 1. 获取真实历史数据
    df_raw = fetch_stock_history(symbol, start_date, today)
    print(f"真实历史数据: {len(df_raw)} 条")

    # 2. 仅用真实数据训练和验证
    df_real = add_stock_features(df_raw.copy())
    X, y = create_sequences(df_real, STOCK_FEATURE_COLS, '涨跌方向', seq_len=seq_len)
    print(f"训练样本: {len(X)}, 特征维度: {X.shape[1]}")

    X_train, y_train, X_test, y_test = split_chronological(X, y, test_ratio=0.2)

    up_ratio = y_train.mean()
    print(f"朴素基线 (全预测涨): 准确率 = {max(up_ratio, 1-up_ratio) * 100:.1f}%")

    # Z-score 归一化
    feat_mean = X_train.mean(axis=0)
    feat_std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - feat_mean) / feat_std
    X_test = (X_test - feat_mean) / feat_std

    # 标签平滑 + 固定种子确保可复现
    np.random.seed(42)
    y_train_smooth = y_train * 0.8 + 0.1

    net = _build_classifier(X.shape[1])
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(net.get_parameters(), lr=0.0005, weight_decay=5e-3)
    scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    trainer = Trainer(net, loss_fn, optimizer, scheduler=scheduler, grad_clip=1.0)

    trainer.train(X_train, y_train_smooth, epochs=epochs, batch_size=batch_size,
                  X_val=X_test, y_val=y_test, early_stopping_patience=20, verbose=True)

    # 3. 评估
    preds_test = trainer.predict(X_test)
    acc = np.mean((preds_test > 0.5).astype(int).flatten() == y_test.flatten())
    print(f"\n测试集准确率: {acc * 100:.2f}%")

    # 4. 推算未来数据用于图表
    df_full = extrapolate_stock_future(df_raw.copy(), future_days=future_days)
    df_full = add_stock_features(df_full)
    print(f"含推算总数据: {len(df_full)} 条")

    predicted_flags = df_full['_predicted'].values
    X_full, y_full = create_sequences(df_full, STOCK_FEATURE_COLS, '涨跌方向', seq_len=seq_len)
    seq_predicted = predicted_flags[seq_len:seq_len + len(X_full)]
    future_mask = seq_predicted

    X_future = X_full[future_mask]
    if len(X_future) > 0:
        X_future_norm = (X_future - feat_mean) / feat_std
        future_preds = trainer.predict(X_future_norm)
    else:
        future_preds = np.array([])

    # 5. 生成图表
    _save_stock_charts(df_full, future_preds, y_full[future_mask],
                       seq_len, trainer, symbol)
    return net, trainer


def _save_stock_charts(df_full, future_preds, future_labels, seq_len, trainer, symbol):
    """生成股票预测可视化图表"""
    n_future = len(future_preds)
    n_show = min(len(df_full), n_future + seq_len + 60)

    dates = df_full['日期'].values[-n_show:].tolist()
    closes = df_full['收盘'].values[-n_show:].tolist()
    is_predicted = df_full['_predicted'].values[-n_show:].tolist()

    predictions = [None] * len(dates)
    labels = [None] * len(dates)
    for i in range(n_future):
        idx = len(dates) - n_future + i
        predictions[idx] = float(future_preds.flatten()[i])
        labels[idx] = float(future_labels.flatten()[i])

    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    save_stock_chart(
        os.path.join(output_dir, f'stock_{symbol}.html'),
        dates=dates,
        closes=closes,
        predictions=predictions,
        labels=labels,
        train_losses=trainer.train_losses,
        val_losses=trainer.val_losses if trainer.val_losses else None,
        is_predicted=is_predicted,
    )


def train_currency_model(symbol='美元', start_date=None, end_date=None,
                         seq_len=1, epochs=80, batch_size=32, future_days=180):
    """训练货币汇率涨跌预测模型"""
    if start_date is None or end_date is None:
        start_date, end_date, today = get_date_range_around_now(past_years=2, future_days=future_days)
    else:
        today = get_date_range_around_now()[2]

    print(f"{'='*50}")
    print(f"汇率涨跌预测: {symbol}")
    print(f"时间范围: {start_date} ~ {end_date} (今天: {today})")
    print(f"{'='*50}")

    df_raw = fetch_currency_history(symbol, start_date, today)
    print(f"真实历史数据: {len(df_raw)} 条")

    df_real = add_currency_features(df_raw.copy())
    feature_cols = _get_currency_feature_cols(df_real)
    X, y = create_sequences(df_real, feature_cols, '涨跌方向', seq_len=seq_len)
    print(f"训练样本: {len(X)}, 特征维度: {X.shape[1]}")

    X_train, y_train, X_test, y_test = split_chronological(X, y, test_ratio=0.2)

    up_ratio = y_train.mean()
    print(f"朴素基线 (全预测涨): 准确率 = {max(up_ratio, 1-up_ratio) * 100:.1f}%")

    feat_mean = X_train.mean(axis=0)
    feat_std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - feat_mean) / feat_std
    X_test = (X_test - feat_mean) / feat_std

    y_train_smooth = y_train * 0.8 + 0.1

    net = _build_classifier(X.shape[1])
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(net.get_parameters(), lr=0.0005, weight_decay=5e-3)
    scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    trainer = Trainer(net, loss_fn, optimizer, scheduler=scheduler, grad_clip=1.0)

    trainer.train(X_train, y_train_smooth, epochs=epochs, batch_size=batch_size,
                  X_val=X_test, y_val=y_test, early_stopping_patience=20, verbose=True)

    preds_test = trainer.predict(X_test)
    acc = np.mean((preds_test > 0.5).astype(int).flatten() == y_test.flatten())
    print(f"\n测试集准确率: {acc * 100:.2f}%")

    # 推算未来数据
    df_full = extrapolate_currency_future(df_raw.copy(), future_days=future_days)
    df_full = add_currency_features(df_full)
    print(f"含推算总数据: {len(df_full)} 条")

    predicted_flags = df_full['_predicted'].values
    feature_cols_full = _get_currency_feature_cols(df_full)
    X_full, y_full = create_sequences(df_full, feature_cols_full, '涨跌方向', seq_len=seq_len)
    seq_predicted = predicted_flags[seq_len:seq_len + len(X_full)]
    future_mask = seq_predicted

    X_future = X_full[future_mask]
    if len(X_future) > 0:
        X_future_norm = (X_future - feat_mean) / feat_std
        future_preds = trainer.predict(X_future_norm)
    else:
        future_preds = np.array([])

    _save_currency_charts(df_full, future_preds, y_full[future_mask],
                          seq_len, trainer, symbol)
    return net, trainer


def _save_currency_charts(df_full, future_preds, future_labels, seq_len, trainer, symbol):
    """生成汇率预测可视化图表"""
    n_future = len(future_preds)
    n_show = min(len(df_full), n_future + seq_len + 60)

    dates = df_full['日期'].values[-n_show:].tolist()
    price_col = '中间价' if '中间价' in df_full.columns else df_full.columns[1]
    prices = df_full[price_col].values[-n_show:].tolist()
    is_predicted = df_full['_predicted'].values[-n_show:].tolist()

    predictions = [None] * len(dates)
    labels = [None] * len(dates)
    for i in range(n_future):
        idx = len(dates) - n_future + i
        predictions[idx] = float(future_preds.flatten()[i])
        labels[idx] = float(future_labels.flatten()[i])

    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    save_stock_chart(
        os.path.join(output_dir, f'currency_{symbol}.html'),
        dates=dates,
        closes=prices,
        predictions=predictions,
        labels=labels,
        train_losses=trainer.train_losses,
        val_losses=trainer.val_losses if trainer.val_losses else None,
        is_predicted=is_predicted,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='股票/货币涨跌预测')
    parser.add_argument('--type', choices=['stock', 'currency'], default='stock')
    parser.add_argument('--symbol', default='000001')
    parser.add_argument('--start', default=None, help='起始日期 (默认自动)')
    parser.add_argument('--end', default=None, help='结束日期 (默认自动)')
    parser.add_argument('--epochs', type=int, default=80)
    args = parser.parse_args()

    if args.type == 'stock':
        train_stock_model(symbol=args.symbol, start_date=args.start,
                          end_date=args.end, epochs=args.epochs)
    else:
        train_currency_model(symbol=args.symbol, start_date=args.start,
                             end_date=args.end, epochs=args.epochs)
