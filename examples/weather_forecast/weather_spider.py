"""天气数据获取 - 使用 Open-Meteo 免费 API（无需 API Key）"""
import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.time_utils import get_current_time


def _format_date(date_str):
    """将 YYYYMMDD 或 YYYY-MM-DD 统一转为 YYYY-MM-DD"""
    date_str = date_str.replace('-', '')
    if len(date_str) == 8:
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    return date_str


def fetch_weather_history(latitude=39.9, longitude=116.4, start_date='2020-01-01', end_date=None):
    """
    获取历史天气数据（默认北京）
    Open-Meteo Archive API: https://open-meteo.com/
    无需 API Key，完全免费
    """
    start_date = _format_date(start_date)
    # end_date 不能超过昨天（归档 API 无未来数据）
    if end_date is None:
        end_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
    else:
        end_date = _format_date(end_date)
        yesterday = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
        if end_date > yesterday:
            end_date = yesterday

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'start_date': start_date,
        'end_date': end_date,
        'daily': 'temperature_2m_max,temperature_2m_min,temperature_2m_mean,'
                 'precipitation_sum,wind_speed_10m_max',
        'timezone': 'auto',
    }

    # 清除代理设置，避免本地代理拦截
    session = requests.Session()
    session.trust_env = False  # 忽略环境变量中的代理

    print(f"请求天气数据: lat={latitude}, lon={longitude}, {start_date} ~ {end_date}")
    try:
        resp = session.get(url, params=params, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[网络错误] 无法获取天气数据: {e}")
        print("正在使用模拟数据...")
        return _generate_synthetic_weather(start_date, end_date)

    data = resp.json()
    if 'daily' not in data:
        print("[API 错误] 返回数据中无 daily 字段，使用模拟数据")
        return _generate_synthetic_weather(start_date, end_date)

    daily = data['daily']
    df = pd.DataFrame({
        '日期': daily['time'],
        '最高温': daily['temperature_2m_max'],
        '最低温': daily['temperature_2m_min'],
        '平均温': daily['temperature_2m_mean'],
        '降水量': daily['precipitation_sum'],
        '最大风速': daily['wind_speed_10m_max'],
    })
    df = df.dropna()
    print(f"获取到 {len(df)} 天天气数据")
    return df


def _generate_synthetic_weather(start_date, end_date):
    """生成模拟天气数据（网络不可用时使用）"""
    n_days = (datetime.strptime(end_date, '%Y-%m-%d') -
              datetime.strptime(start_date, '%Y-%m-%d')).days + 1
    dates = pd.date_range(start=start_date, periods=n_days, freq='D')

    np.random.seed(42)
    # 模拟北京气候：冬冷夏热
    day_of_year = dates.dayofyear
    base_temp = 13 + 18 * np.sin((day_of_year - 80) * 2 * np.pi / 365)  # 年均温约13C
    noise = np.random.randn(n_days) * 4

    df = pd.DataFrame({
        '日期': dates.strftime('%Y-%m-%d'),
        '最高温': (base_temp + 5 + noise).round(1),
        '最低温': (base_temp - 5 + noise * 0.5).round(1),
        '平均温': (base_temp + noise * 0.7).round(1),
        '降水量': np.maximum(0, np.random.exponential(2, n_days) - 1).round(1),
        '最大风速': (8 + np.random.randn(n_days) * 3).round(1),
    })
    print(f"[模拟数据] 生成 {len(df)} 天天气数据")
    return df


def add_weather_features(df):
    """添加天气特征"""
    # 季节性特征 (sin/cos 编码 day-of-year)
    dates = pd.to_datetime(df['日期'])
    doy = dates.dt.dayofyear
    df['sin_doy'] = np.sin(2 * np.pi * doy / 365)
    df['cos_doy'] = np.cos(2 * np.pi * doy / 365)

    # 滞后特征
    for col in ['平均温', '降水量', '最大风速']:
        for lag in [1, 2, 3]:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)

    # 温差
    df['温差'] = df['最高温'] - df['最低温']
    df['温差_lag1'] = df['温差'].shift(1)

    # 温度变化趋势
    df['温度变化'] = df['平均温'] - df['平均温_lag1']

    # 滑动平均
    df['平均温_ma3'] = df['平均温'].rolling(3).mean()
    df['平均温_ma7'] = df['平均温'].rolling(7).mean()

    # 目标: create_sequences 取 targets[i+seq_len]，即特征窗口后一天
    # 所以直接用当天指标作为目标即可正确预测"下一天"
    df['温度升'] = (df['平均温'] > df['平均温'].shift(1)).astype(int)

    return df.dropna()


def extrapolate_weather_future(df_hist, latitude=39.9, future_days=180):
    """
    基于历史数据季节性+统计特征推算未来天气数据
    标注 _predicted=True
    """
    if df_hist.empty or len(df_hist) < 30:
        return df_hist

    last_date = pd.to_datetime(df_hist['日期'].iloc[-1])
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=future_days)

    # 按日序计算历史气候基准（季节性）
    day_of_year = df_hist['日期'].apply(lambda x: pd.to_datetime(x).timetuple().tm_yday)
    hist_doy = day_of_year.values
    hist_mean_temp = df_hist['平均温'].values

    # 计算每个 DOY 的气候均值和标准差
    doy_stats = {}
    for doy in range(1, 367):
        mask = hist_doy == doy
        if mask.sum() > 0:
            doy_stats[doy] = {
                'mean': hist_mean_temp[mask].mean(),
                'std': max(hist_mean_temp[mask].std(), 2.0),
            }

    # 对缺失 DOY 用正弦近似
    if not doy_stats:
        return df_hist

    all_means = [v['mean'] for v in doy_stats.values()]
    avg_mean = np.mean(all_means) if all_means else 13.0

    np.random.seed(42)
    rows = []
    for d in future_dates:
        doy = d.timetuple().tm_yday
        stats = doy_stats.get(doy)
        if stats:
            base_temp = stats['mean']
            noise_std = stats['std']
        else:
            # 正弦近似北京气候
            base_temp = 13 + 18 * np.sin((doy - 80) * 2 * np.pi / 365)
            noise_std = 4.0

        avg_t = base_temp + np.random.randn() * noise_std * 0.5
        high_t = avg_t + 3 + np.random.rand() * 4
        low_t = avg_t - 3 - np.random.rand() * 4
        precip = max(0, np.random.exponential(2) - 1)
        wind = max(1, 8 + np.random.randn() * 3)

        rows.append({
            '日期': d.strftime('%Y-%m-%d'),
            '最高温': round(high_t, 1),
            '最低温': round(low_t, 1),
            '平均温': round(avg_t, 1),
            '降水量': round(precip, 1),
            '最大风速': round(wind, 1),
            '_predicted': True,
        })

    df_future = pd.DataFrame(rows)
    df_hist = df_hist.copy()
    df_hist['_predicted'] = False

    df_all = pd.concat([df_hist, df_future], ignore_index=True)
    print(f"[推算] 未来 {len(df_future)} 天天气已推算 (共 {len(df_all)} 条)")
    return df_all


# 常用城市坐标
CITY_COORDS = {
    '北京': (39.9, 116.4),
    '上海': (31.2, 121.5),
    '广州': (23.1, 113.3),
    '深圳': (22.5, 114.1),
    '成都': (30.6, 104.1),
    '杭州': (30.3, 120.2),
    '武汉': (30.6, 114.3),
    '南京': (32.1, 118.8),
    '重庆': (29.6, 106.5),
    '西安': (34.3, 108.9),
}


if __name__ == "__main__":
    df = fetch_weather_history()
    print(df.head())
    print(f"\n数据量: {len(df)} 天")
