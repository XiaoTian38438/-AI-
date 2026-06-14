"""天气预测训练脚本 - 默认回归模式预测明日温度"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from core.network import Network
from ops.activations import LeakyReLU, Sigmoid, Linear
from ops.losses import CrossEntropyLoss, MSELoss
from engine.optimizers import Adam
from engine.scheduler import ReduceLROnPlateau
from engine.trainer import Trainer
from utils.chart import save_weather_chart
from utils.time_utils import get_date_range_around_now

from examples.weather_forecast.weather_spider import (
    fetch_weather_history, add_weather_features, CITY_COORDS,
    extrapolate_weather_future
)
from examples.weather_forecast.preprocess import create_sequences, split_chronological


WEATHER_FEATURE_COLS = [
    '平均温', '平均温_lag1', '平均温_lag2', '平均温_lag3',
    '降水量', '降水量_lag1', '最大风速', '最大风速_lag1',
    '温差', '温差_lag1', '温度变化',
    '平均温_ma3', '平均温_ma7',
    'sin_doy', 'cos_doy',
]


def train_weather_model(city='北京', start_date=None, end_date=None,
                        seq_len=1, epochs=150, batch_size=16, task='regression',
                        future_days=180):
    """训练天气预测模型（仅用真实数据训练，推算数据用于未来预测可视化）
    seq_len=1: lag 特征已编码历史，无需滑动窗口重复
    """
    lat, lon = CITY_COORDS.get(city, (39.9, 116.4))

    if start_date is None or end_date is None:
        start_date, end_date, today = get_date_range_around_now(past_years=2, future_days=future_days)
    else:
        today = get_date_range_around_now()[2]

    print(f"城市: {city}, 时间范围: {start_date} ~ {end_date} (今天: {today})")

    # 1. 获取真实历史数据
    df_raw = fetch_weather_history(lat, lon, start_date, today)
    print(f"真实历史数据: {len(df_raw)} 天")

    # 2. 仅用真实数据训练和验证
    df_real = add_weather_features(df_raw.copy())

    if task == 'classification':
        target_col = '温度升'
    else:
        target_col = '平均温'

    X, y = create_sequences(df_real, WEATHER_FEATURE_COLS, target_col, seq_len=seq_len)
    print(f"训练样本: {len(X)}, 特征维度: {X.shape[1]}")

    X_train, y_train, X_test, y_test = split_chronological(X, y, test_ratio=0.2)

    # 朴素基线
    temps = df_real['平均温'].values
    naive_mae = np.mean(np.abs(np.diff(temps)))
    if task == 'regression':
        print(f"朴素基线 (明天=今天): MAE = {naive_mae:.2f}°C")

    # Z-score 归一化特征
    feat_mean = X_train.mean(axis=0)
    feat_std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - feat_mean) / feat_std
    X_test = (X_test - feat_mean) / feat_std

    # 固定种子确保可复现
    np.random.seed(42)

    # 回归: 目标也归一化
    if task == 'regression':
        y_mean = float(y_train.mean())
        y_std = float(y_train.std()) + 1e-8
        y_train_norm = (y_train - y_mean) / y_std
        y_test_norm = (y_test - y_mean) / y_std
    else:
        y_mean, y_std = 0.0, 1.0
        y_train_norm = y_train * 0.8 + 0.1
        y_test_norm = y_test

    # 轻量网络: input → 16 → 1
    input_dim = X.shape[1]
    net = Network(use_gpu=True)
    inp = net.add_input_layer(input_dim)
    h1 = net.add_dense_layer(inp, 16, LeakyReLU)

    if task == 'classification':
        out = net.add_dense_layer(h1, 1, Sigmoid)
        loss_fn = CrossEntropyLoss()
        net._layer_dropout = [0.1, 0.0]
    else:
        out = net.add_dense_layer(h1, 1, Linear)
        loss_fn = MSELoss()
        net._layer_dropout = [0.05, 0.0]

    net.set_output_nodes(out)

    optimizer = Adam(net.get_parameters(), lr=0.001, weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    trainer = Trainer(net, loss_fn, optimizer, scheduler=scheduler, grad_clip=1.0)

    trainer.train(X_train, y_train_norm, epochs=epochs, batch_size=batch_size,
                  X_val=X_test, y_val=y_test_norm, early_stopping_patience=20, verbose=True)

    # 3. 评估
    preds_test = trainer.predict(X_test)
    if task == 'classification':
        acc = np.mean((preds_test > 0.5).astype(int).flatten() == y_test.flatten())
        print(f"\n测试集准确率: {acc * 100:.2f}%")
    else:
        preds_orig = preds_test * y_std + y_mean
        mae = np.mean(np.abs(preds_orig.flatten() - y_test.flatten()))
        print(f"\n测试集 MAE: {mae:.2f}°C (朴素基线: {naive_mae:.2f}°C)")

    # 4. 推算未来数据用于图表
    df_full = extrapolate_weather_future(df_raw.copy(), latitude=lat, future_days=future_days)
    df_full = add_weather_features(df_full)
    print(f"含推算总数据: {len(df_full)} 天")

    predicted_flags = df_full['_predicted'].values
    X_full, y_full = create_sequences(df_full, WEATHER_FEATURE_COLS, target_col, seq_len=seq_len)
    seq_predicted = predicted_flags[seq_len:seq_len + len(X_full)]
    future_mask = seq_predicted

    X_future = X_full[future_mask]
    if len(X_future) > 0:
        X_future_norm = (X_future - feat_mean) / feat_std
        future_preds = trainer.predict(X_future_norm)
        if task == 'regression':
            future_preds = future_preds * y_std + y_mean
    else:
        future_preds = np.array([])

    # 5. 生成图表
    _save_weather_charts(df_full, future_preds, y_full[future_mask],
                         seq_len, trainer, city, task)
    return net, trainer


def _save_weather_charts(df_full, future_preds, future_labels, seq_len, trainer, city, task):
    """生成天气预测可视化图表"""
    n_future = len(future_preds)
    n_show = min(len(df_full), n_future + seq_len + 60)

    dates = df_full['日期'].values[-n_show:].tolist()
    temps = df_full['平均温'].values[-n_show:].tolist()
    is_predicted = df_full['_predicted'].values[-n_show:].tolist()

    predictions = [None] * len(dates)
    labels = [None] * len(dates)
    for i in range(n_future):
        idx = len(dates) - n_future + i
        predictions[idx] = float(future_preds.flatten()[i])
        labels[idx] = float(future_labels.flatten()[i])

    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    save_weather_chart(
        os.path.join(output_dir, f'weather_{city}.html'),
        dates=dates,
        temps=temps,
        predictions=predictions,
        labels=labels,
        train_losses=trainer.train_losses,
        val_losses=trainer.val_losses if trainer.val_losses else None,
        is_predicted=is_predicted,
        task=task,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='天气预测训练')
    parser.add_argument('--city', default='北京',
                        help='城市名称 (预设: 北京, 上海, 广州, 深圳, 成都, 杭州, 武汉, 南京, 重庆, 西安)')
    parser.add_argument('--lat', type=float, default=None, help='自定义纬度 (覆盖预设城市坐标)')
    parser.add_argument('--lon', type=float, default=None, help='自定义经度 (覆盖预设城市坐标)')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--task', choices=['regression', 'classification'], default='regression')
    args = parser.parse_args()

    kwargs = dict(city=args.city, epochs=args.epochs, task=args.task)
    if args.lat is not None and args.lon is not None:
        # 自定义城市坐标：临时加入 CITY_COORDS
        from examples.weather_forecast.weather_spider import CITY_COORDS
        CITY_COORDS[args.city] = (args.lat, args.lon)
    train_weather_model(**kwargs)
