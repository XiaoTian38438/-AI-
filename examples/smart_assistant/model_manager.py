"""ML模型管理器
- 服务器启动时自动训练（后台线程）
- 训练期间用户查询使用回退值，前端显示"模型准备中"
- 每天定时重训（凌晨3点）
- 支持手动触发立即训练
"""
import threading
import time
import numpy as np
from datetime import datetime

_lock = threading.Lock()
_last_daily_train_date = None

_models = {
    'stock': {
        'ready': False, 'training': False,
        'trainer': None, 'mean': None, 'std': None,
        'feature_cols': None, 'df_feat': None,
        'target': None, 'ml_prob': None, 'last_trained': None,
    },
    'weather': {
        'ready': False, 'training': False,
        'trainer': None, 'mean': None, 'std': None,
        'y_mean': None, 'y_std': None,
        'feature_cols': None, 'df_feat': None,
        'target': None, 'ml_pred': None, 'last_trained': None,
    },
    'currency': {
        'ready': False, 'training': False,
        'trainer': None, 'mean': None, 'std': None,
        'feature_cols': None, 'df_feat': None,
        'target': None, 'ml_prob': None, 'last_trained': None,
    },
}


def is_ready(model_type=None):
    """检查模型是否就绪"""
    if model_type:
        return _models.get(model_type, {}).get('ready', False)
    return all(m.get('ready', False) for m in _models.values())


def is_training(model_type=None):
    """检查模型是否正在训练"""
    if model_type:
        return _models.get(model_type, {}).get('training', False)
    return any(m.get('training', False) for m in _models.values())


def get_status():
    """获取所有模型状态（含训练指标）"""
    return {
        k: {
            'ready': v['ready'],
            'training': v['training'],
            'last_trained': v.get('last_trained'),
            'target': v.get('target'),
            'train_acc': v.get('train_acc'),
            'train_mse': v.get('train_mse'),
        } for k, v in _models.items()
    }


def predict_stock(symbol=None):
    """获取股票ML预测结果"""
    with _lock:
        m = _models['stock']
        if not m['ready']:
            return None
        result = {'ml_prob': m.get('ml_prob', 0.5)}
        if symbol and m.get('target') == symbol:
            result['df_cached'] = True
        return result


def predict_weather(city=None):
    """获取天气ML预测结果"""
    with _lock:
        m = _models['weather']
        if not m['ready']:
            return None
        result = {'ml_pred': m.get('ml_pred', 20)}
        if city and m.get('target') == city:
            result['df_cached'] = True
        return result


def predict_currency(currency=None):
    """获取汇率ML预测结果"""
    with _lock:
        m = _models['currency']
        if not m['ready']:
            return None
        result = {'ml_prob': m.get('ml_prob', 0.5)}
        if currency and m.get('target') == currency:
            result['df_cached'] = True
        return result


def get_chart_df(model_type):
    """获取预缓存的图表数据"""
    with _lock:
        return _models.get(model_type, {}).get('df_feat')


# ======================== 训练函数 ========================

def train_stock_model(symbol='000001'):
    """训练股票模型"""
    with _lock:
        if _models['stock']['training']:
            return False
        _models['stock']['training'] = True
        _models['stock']['ready'] = False

    try:
        from examples.stock_forecast.data_fetcher import fetch_stock_history, add_stock_features
        from examples.stock_forecast.preprocess import create_sequences, split_chronological
        from core.network import Network
        from ops.activations import LeakyReLU, Sigmoid
        from ops.losses import CrossEntropyLoss
        from engine.optimizers import Adam
        from engine.trainer import Trainer
        from utils.time_utils import get_date_range_around_now

        start, end, today = get_date_range_around_now(past_years=2, future_days=0)
        df = fetch_stock_history(symbol, start, today)
        if df is not None and len(df) > 30:
            df_feat = add_stock_features(df.copy())
            FEATURE_COLS = ['收盘', '收盘_lag1', '收盘_lag2', '成交量', '成交量_lag1', '换手率',
                            'MA5', 'MA10', 'MA20', '偏离MA5', '偏离MA10', 'RSI14', 'MACD', 'MACD_signal', '量比']
            X, y = create_sequences(df_feat, FEATURE_COLS, '涨跌方向', seq_len=1)
            if len(X) > 20:
                X_train, y_train, X_test, y_test = split_chronological(X, y, test_ratio=0.2)
                mean, std = X_train.mean(0), X_train.std(0) + 1e-8
                X_train_n = (X_train - mean) / std
                y_smooth = y_train * 0.8 + 0.1

                np.random.seed(42)
                net = Network(use_gpu=False)
                inp = net.add_input_layer(X.shape[1])
                h = net.add_dense_layer(inp, 16, LeakyReLU)
                out = net.add_dense_layer(h, 1, Sigmoid)
                net.set_output_nodes(out)
                trainer = Trainer(net, CrossEntropyLoss(), Adam(net.get_parameters(), lr=0.001, weight_decay=1e-3))
                trainer.train(X_train_n, y_smooth, epochs=30, batch_size=32, verbose=False)

                # 训练集准确率评估
                train_pred = (trainer.predict(X_train_n).flatten() > 0.5).astype(int)
                train_acc = float((train_pred == (y_train > 0.5).astype(int)).mean())

                last_row = df_feat.iloc[[-1]]
                x_last = np.array([[last_row[c].values[0] for c in FEATURE_COLS]], dtype=np.float64)
                x_last = (x_last - mean) / std
                prob = float(trainer.predict(x_last).flatten()[0])

                with _lock:
                    _models['stock'].update({
                        'ready': True, 'training': False,
                        'trainer': trainer, 'mean': mean, 'std': std,
                        'feature_cols': FEATURE_COLS, 'df_feat': df_feat,
                        'target': symbol, 'ml_prob': prob,
                        'train_acc': train_acc,
                        'last_trained': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    })
                print(f"[ModelManager] 股票模型训练完成 (symbol={symbol}, ml_prob={prob:.2%}, acc={train_acc:.2%})")
                return True
    except Exception as e:
        print(f"[ModelManager] 股票模型训练失败: {e}")
        import traceback
        traceback.print_exc()

    with _lock:
        _models['stock']['training'] = False
    return False


def train_weather_model(city='北京'):
    """训练天气模型"""
    with _lock:
        if _models['weather']['training']:
            return False
        _models['weather']['training'] = True
        _models['weather']['ready'] = False

    try:
        from examples.weather_forecast.weather_spider import fetch_weather_history, add_weather_features, CITY_COORDS
        from examples.weather_forecast.preprocess import create_sequences, split_chronological
        from core.network import Network
        from ops.activations import LeakyReLU, Linear
        from ops.losses import MSELoss
        from engine.optimizers import Adam
        from engine.trainer import Trainer
        from utils.time_utils import get_date_range_around_now

        lat, lon = CITY_COORDS.get(city, (39.9, 116.4))
        start, end, today = get_date_range_around_now(past_years=2, future_days=0)
        df = fetch_weather_history(lat, lon, start, today)
        if df is not None and len(df) > 30:
            df_feat = add_weather_features(df.copy())
            FEATURE_COLS = ['平均温', '平均温_lag1', '平均温_lag2', '平均温_lag3',
                            '降水量', '降水量_lag1', '最大风速', '最大风速_lag1',
                            '温差', '温差_lag1', '温度变化', '平均温_ma3', '平均温_ma7',
                            'sin_doy', 'cos_doy']
            X, y = create_sequences(df_feat, FEATURE_COLS, '平均温', seq_len=1)
            if len(X) > 20:
                X_train, y_train, X_test, y_test = split_chronological(X, y, test_ratio=0.2)
                mean, std = X_train.mean(0), X_train.std(0) + 1e-8
                X_train_n = (X_train - mean) / std
                y_m, y_s = float(y_train.mean()), float(y_train.std()) + 1e-8
                y_train_n = (y_train - y_m) / y_s

                np.random.seed(42)
                net = Network(use_gpu=False)
                inp = net.add_input_layer(X.shape[1])
                h = net.add_dense_layer(inp, 16, LeakyReLU)
                out = net.add_dense_layer(h, 1, Linear)
                net.set_output_nodes(out)
                trainer = Trainer(net, MSELoss(), Adam(net.get_parameters(), lr=0.001, weight_decay=1e-3))
                trainer.train(X_train_n, y_train_n, epochs=30, batch_size=16, verbose=False)

                # 训练集MSE评估
                train_pred_n = trainer.predict(X_train_n).flatten()
                train_mse = float(((train_pred_n - y_train_n) ** 2).mean())

                last_row = df_feat.iloc[[-1]]
                x_last = np.array([[last_row[c].values[0] for c in FEATURE_COLS]], dtype=np.float64)
                x_last = (x_last - mean) / std
                pred = float(trainer.predict(x_last).flatten()[0]) * y_s + y_m

                with _lock:
                    _models['weather'].update({
                        'ready': True, 'training': False,
                        'trainer': trainer, 'mean': mean, 'std': std,
                        'y_mean': y_m, 'y_std': y_s,
                        'feature_cols': FEATURE_COLS, 'df_feat': df_feat,
                        'target': city, 'ml_pred': round(pred, 1),
                        'train_mse': train_mse,
                        'last_trained': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    })
                print(f"[ModelManager] 天气模型训练完成 (city={city}, ml_pred={pred:.1f}°C, mse={train_mse:.4f})")
                return True
    except Exception as e:
        print(f"[ModelManager] 天气模型训练失败: {e}")
        import traceback
        traceback.print_exc()

    with _lock:
        _models['weather']['training'] = False
    return False


def train_currency_model(currency='美元'):
    """训练汇率模型"""
    with _lock:
        if _models['currency']['training']:
            return False
        _models['currency']['training'] = True
        _models['currency']['ready'] = False

    try:
        from examples.stock_forecast.data_fetcher import fetch_currency_history, add_currency_features
        from examples.stock_forecast.preprocess import create_sequences, split_chronological
        from core.network import Network
        from ops.activations import LeakyReLU, Sigmoid
        from ops.losses import CrossEntropyLoss
        from engine.optimizers import Adam
        from engine.trainer import Trainer
        from utils.time_utils import get_date_range_around_now

        start, end, today = get_date_range_around_now(past_years=2, future_days=0)
        df = fetch_currency_history(currency, start, today)
        if df is not None and len(df) > 20:
            df_feat = add_currency_features(df.copy())
            price_col = '中间价' if '中间价' in df_feat.columns else df_feat.columns[1]
            feature_cols = [price_col, f'{price_col}_lag1', f'{price_col}_lag2', f'{price_col}_lag3',
                            'MA5', 'MA10', 'MA20', '偏离MA5', 'RSI14', '波动率5']
            X, y = create_sequences(df_feat, feature_cols, '涨跌方向', seq_len=1)
            if len(X) > 20:
                X_train, y_train, X_test, y_test = split_chronological(X, y, test_ratio=0.2)
                mean, std = X_train.mean(0), X_train.std(0) + 1e-8
                X_train_n = (X_train - mean) / std
                y_smooth = y_train * 0.8 + 0.1

                np.random.seed(42)
                net = Network(use_gpu=False)
                inp = net.add_input_layer(X.shape[1])
                h = net.add_dense_layer(inp, 16, LeakyReLU)
                out = net.add_dense_layer(h, 1, Sigmoid)
                net.set_output_nodes(out)
                trainer = Trainer(net, CrossEntropyLoss(), Adam(net.get_parameters(), lr=0.001, weight_decay=1e-3))
                trainer.train(X_train_n, y_smooth, epochs=30, batch_size=32, verbose=False)

                # 训练集准确率评估
                train_pred = (trainer.predict(X_train_n).flatten() > 0.5).astype(int)
                train_acc = float((train_pred == (y_train > 0.5).astype(int)).mean())

                last_row = df_feat.iloc[[-1]]
                x_last = np.array([[last_row[c].values[0] for c in feature_cols]], dtype=np.float64)
                x_last = (x_last - mean) / std
                prob = float(trainer.predict(x_last).flatten()[0])

                with _lock:
                    _models['currency'].update({
                        'ready': True, 'training': False,
                        'trainer': trainer, 'mean': mean, 'std': std,
                        'feature_cols': feature_cols, 'df_feat': df_feat,
                        'target': currency, 'ml_prob': prob,
                        'train_acc': train_acc,
                        'last_trained': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    })
                print(f"[ModelManager] 汇率模型训练完成 (currency={currency}, ml_prob={prob:.2%}, acc={train_acc:.2%})")
                return True
    except Exception as e:
        print(f"[ModelManager] 汇率模型训练失败: {e}")
        import traceback
        traceback.print_exc()

    with _lock:
        _models['currency']['training'] = False
    return False


# ======================== 批量训练与调度 ========================

def train_all():
    """同步训练所有模型"""
    print("[ModelManager] 开始训练所有模型...")
    train_stock_model('000001')
    train_weather_model('北京')
    train_currency_model('美元')
    print("[ModelManager] 所有模型训练完成")


def train_all_async():
    """异步训练所有模型（后台线程）"""
    t = threading.Thread(target=train_all, daemon=True)
    t.start()


def retrain_specific(model_type, target=None):
    """手动重训指定模型"""
    if model_type == 'stock':
        return train_stock_model(target or '000001')
    elif model_type == 'weather':
        return train_weather_model(target or '北京')
    elif model_type == 'currency':
        return train_currency_model(target or '美元')
    return False


def retrain_specific_async(model_type, target=None):
    """异步手动重训指定模型"""
    def _train():
        retrain_specific(model_type, target)
    t = threading.Thread(target=_train, daemon=True)
    t.start()


def start_daily_scheduler():
    """启动每日重训调度器（每天凌晨3点）"""
    def _scheduler():
        global _last_daily_train_date
        while True:
            now = datetime.now()
            if now.hour == 3 and (_last_daily_train_date is None or _last_daily_train_date != now.date()):
                _last_daily_train_date = now.date()
                print("[ModelManager] 开始每日模型重训...")
                train_all()
                print("[ModelManager] 每日模型重训完成")
            time.sleep(3600)

    t = threading.Thread(target=_scheduler, daemon=True)
    t.start()
    print("[ModelManager] 每日重训调度器已启动 (每天3:00)")
