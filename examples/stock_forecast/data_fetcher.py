"""股票 & 货币数据获取（含网络异常自动降级为模拟数据）"""
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from utils.time_utils import get_current_time


def _clear_proxy():
    """清除代理环境变量，避免本地代理拦截请求"""
    for key in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']:
        os.environ.pop(key, None)


def _http_get_json(url, encoding='utf-8', referer=None, timeout=15):
    """简易 HTTP GET 返回 JSON，无需第三方库"""
    from urllib.request import Request, urlopen
    from urllib.error import URLError
    req = Request(url)
    req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    if referer:
        req.add_header('Referer', referer)
    with urlopen(req, timeout=timeout) as resp:
        data = resp.read().decode(encoding)
        # JSONP 回调格式: jQueryxxx(...)
        if data.startswith(('jQuery', 'jquery', 'callback')) and '(' in data:
            data = data[data.index('(') + 1:data.rindex(')')]
        return json.loads(data)


def fetch_stock_history(symbol: str, start_date: str, end_date: str):
    """
    获取 A 股历史日线数据（多源降级：东方财富→腾讯→新浪→akshare→模拟）
    symbol: 股票代码，如 '000001'（平安银行）
    网络不可用时自动降级为模拟数据
    """
    _clear_proxy()
    errors = []

    # ---------- 源1: 东方财富 ----------
    try:
        market = '1' if symbol.startswith(('6', '5')) else '0'
        secid = f'{market}.{symbol}'
        s = start_date.replace('-', '')
        e = end_date.replace('-', '')
        url = (
            f'https://push2his.eastmoney.com/api/qt/stock/kline/get?'
            f'secid={secid}&fields1=f1,f2,f3,f4,f5,f6&'
            f'fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61&'
            f'klt=101&fqt=1&beg={s}&end={e}&lmt=1000000'
        )
        print(f"获取股票 {symbol} 历史数据 (东方财富)...")
        result = _http_get_json(url, referer='https://quote.eastmoney.com/')
        klines = result.get('data', {}).get('klines', [])
        if not klines:
            raise ValueError("无数据")
        rows = []
        for line in klines:
            p = line.split(',')
            rows.append({
                '日期': p[0], '开盘': float(p[1]), '收盘': float(p[2]),
                '最高': float(p[3]), '最低': float(p[4]), '成交量': float(p[5]),
                '成交额': float(p[6]), '振幅': float(p[7]), '涨跌幅': float(p[8]),
                '涨跌额': float(p[9]), '换手率': float(p[10]),
            })
        df = pd.DataFrame(rows)
        print(f"获取到 {len(df)} 条真实数据 (东方财富)")
        return df
    except Exception as e:
        errors.append(f"东方财富: {e}")

    # ---------- 源2: 腾讯财经 ----------
    try:
        prefix = 'sh' if symbol.startswith(('6', '5')) else 'sz'
        code = f'{prefix}{symbol}'
        s_fmt = _parse_date(start_date).strftime('%Y-%m-%d')
        e_fmt = _parse_date(end_date).strftime('%Y-%m-%d')
        url = (f'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?'
               f'param={code},day,{s_fmt},{e_fmt},5000,qfq')
        print(f"获取股票 {symbol} 历史数据 (腾讯财经)...")
        result = _http_get_json(url, timeout=15)
        code_data = result.get('data', {}).get(code, {})
        klines = code_data.get('qfqday', code_data.get('day', []))
        if not klines:
            raise ValueError("无数据")
        rows = []
        for k in klines:
            # [日期, 开盘, 收盘, 最高, 最低, 成交量]
            rows.append({
                '日期': k[0], '开盘': float(k[1]), '收盘': float(k[2]),
                '最高': float(k[3]), '最低': float(k[4]), '成交量': float(k[5]),
                '成交额': 0.0, '振幅': 0.0,
                '涨跌幅': 0.0, '涨跌额': 0.0, '换手率': 0.0,
            })
        # 补充涨跌幅等指标
        df = pd.DataFrame(rows)
        if len(df) > 1:
            df['涨跌额'] = df['收盘'].diff()
            df['涨跌幅'] = df['涨跌额'] / (df['收盘'].shift(1) + 1e-8) * 100
            df['振幅'] = (df['最高'] - df['最低']) / (df['收盘'].shift(1) + 1e-8) * 100
        print(f"获取到 {len(df)} 条真实数据 (腾讯财经)")
        return df
    except Exception as e:
        errors.append(f"腾讯财经: {e}")

    # ---------- 源3: 新浪财经 ----------
    try:
        prefix = 'sz' if symbol.startswith(('0', '3')) else 'sh'
        url = (f'https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/'
               f'CN_MarketData.getKLineData?symbol={prefix}{symbol}&scale=240&ma=no&datalen=2000')
        print(f"获取股票 {symbol} 历史数据 (新浪财经)...")
        from urllib.request import Request, urlopen
        req = Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0')
        req.add_header('Referer', 'https://finance.sina.com.cn/')
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode('gbk'))
        if not data:
            raise ValueError("无数据")
        rows = []
        for d in data:
            rows.append({
                '日期': d['day'], '开盘': float(d['open']), '收盘': float(d['close']),
                '最高': float(d['high']), '最低': float(d['low']), '成交量': float(d['volume']),
                '成交额': 0.0, '振幅': 0.0, '涨跌幅': 0.0, '涨跌额': 0.0, '换手率': 0.0,
            })
        df = pd.DataFrame(rows)
        # 按日期过滤
        s_dt = _parse_date(start_date)
        e_dt = _parse_date(end_date)
        df['_dt'] = pd.to_datetime(df['日期'])
        df = df[(df['_dt'] >= s_dt) & (df['_dt'] <= e_dt)].drop(columns=['_dt'])
        if len(df) > 1:
            df['涨跌额'] = df['收盘'].diff()
            df['涨跌幅'] = df['涨跌额'] / (df['收盘'].shift(1) + 1e-8) * 100
            df['振幅'] = (df['最高'] - df['最低']) / (df['收盘'].shift(1) + 1e-8) * 100
        print(f"获取到 {len(df)} 条真实数据 (新浪财经)")
        return df
    except Exception as e:
        errors.append(f"新浪财经: {e}")

    # ---------- 源4: akshare ----------
    try:
        import akshare as ak
        print(f"获取股票 {symbol} 历史数据 (akshare)...")
        df = ak.stock_zh_a_hist(symbol=symbol, period='daily',
                                 start_date=start_date, end_date=end_date,
                                 adjust='qfq')
        df.columns = ['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额',
                      '振幅', '涨跌幅', '涨跌额', '换手率']
        print(f"获取到 {len(df)} 条真实数据 (akshare)")
        return df
    except Exception as e:
        errors.append(f"akshare: {e}")

    # 全部失败 -> 模拟数据
    print(f"[所有数据源均失败] {'; '.join(errors)}")
    print("正在使用模拟数据...")
    return _generate_synthetic_stock(symbol, start_date, end_date)


def fetch_currency_history(symbol: str = '美元', start_date: str = '2020-01-01',
                           end_date: str = '2025-12-31'):
    """
    获取人民币汇率历史数据（frankfurter.app 免费 API，无需 key）
    symbol: '美元', '欧元', '日元', '英镑', '港元'
    网络不可用时自动降级为模拟数据
    """
    _clear_proxy()
    try:
        currency_code_map = {
            '美元': 'USD', '欧元': 'EUR', '日元': 'JPY', '英镑': 'GBP', '港元': 'HKD',
        }
        code = currency_code_map.get(symbol, 'USD')
        s = start_date.replace('-', '')
        e = end_date.replace('-', '')
        # frankfurter.app: 欧洲央行汇率数据，免费无需 key
        # API 限制单次最多 1 年，按年分段请求
        s_dt = _parse_date(start_date)
        e_dt = _parse_date(end_date)

        all_rates = {}
        from urllib.request import Request, urlopen
        cur_start = s_dt
        while cur_start < e_dt:
            cur_end = min(cur_start + timedelta(days=365), e_dt)
            url = (f'https://api.frankfurter.app/{cur_start.strftime("%Y-%m-%d")}'
                   f'..{cur_end.strftime("%Y-%m-%d")}?from={code}&to=CNY')
            req = Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0')
            print(f"获取 {symbol} 汇率历史数据 ({cur_start.year})...")
            with urlopen(req, timeout=20) as resp:
                result = json.loads(resp.read().decode('utf-8'))
            rates = result.get('rates', {})
            for d, v in rates.items():
                all_rates[d] = v.get('CNY', None)
            cur_start = cur_end + timedelta(days=1)

        if not all_rates:
            raise ValueError(f"未获取到 {symbol} 汇率数据")

        rows = []
        for d in sorted(all_rates.keys()):
            rate = all_rates[d]
            if rate is not None:
                rows.append({
                    '日期': d,
                    '中间价': rate,
                    '买入价': rate - 0.01,
                    '卖出价': rate + 0.01,
                })
        df = pd.DataFrame(rows)
        print(f"获取到 {len(df)} 条真实数据")
        return df
    except Exception as e:
        print(f"[网络错误] 汇率 API 失败: {e}")
        print("正在使用模拟数据...")
        return _generate_synthetic_currency(symbol, start_date, end_date)


# ======================== 模拟数据生成 ========================

def _generate_synthetic_stock(symbol, start_date, end_date):
    """生成模拟股票数据"""
    s = _parse_date(start_date)
    e = _parse_date(end_date)
    n_days = (e - s).days + 1
    dates = pd.date_range(start=s, periods=n_days, freq='B')  # 仅工作日

    np.random.seed(hash(symbol) % 2**31)
    # 模拟股价：随机游走
    price = 15.0
    prices = []
    volumes = []
    for _ in range(len(dates)):
        change = np.random.randn() * 0.3
        price = max(price + change, 1.0)
        prices.append(price)
        volumes.append(int(5e7 + np.random.randn() * 2e7))

    df = pd.DataFrame({
        '日期': dates.strftime('%Y-%m-%d'),
        '开盘': [p - np.random.rand() * 0.2 for p in prices],
        '收盘': prices,
        '最高': [p + abs(np.random.randn()) * 0.3 for p in prices],
        '最低': [p - abs(np.random.randn()) * 0.3 for p in prices],
        '成交量': volumes,
        '成交额': [v * p for v, p in zip(volumes, prices)],
        '振幅': np.random.uniform(1, 5, len(dates)),
        '涨跌幅': np.random.uniform(-3, 3, len(dates)),
        '涨跌额': np.random.uniform(-0.5, 0.5, len(dates)),
        '换手率': np.random.uniform(0.3, 2.0, len(dates)),
    })
    print(f"[模拟数据] 生成 {len(df)} 条股票数据 (代码: {symbol})")
    return df


def _generate_synthetic_currency(symbol, start_date, end_date):
    """生成模拟汇率数据"""
    s = _parse_date(start_date)
    e = _parse_date(end_date)
    n_days = (e - s).days + 1
    dates = pd.date_range(start=s, periods=n_days, freq='B')

    np.random.seed(hash(symbol) % 2**31)
    base_map = {'美元': 700, '欧元': 760, '日元': 4.8, '英镑': 880, '港元': 90}
    base = base_map.get(symbol, 700)

    rate = base / 100.0
    rates = []
    for _ in range(len(dates)):
        rate += np.random.randn() * 0.005
        rates.append(rate)

    df = pd.DataFrame({
        '日期': dates.strftime('%Y-%m-%d'),
        '中间价': rates,
        '买入价': [r - 0.01 for r in rates],
        '卖出价': [r + 0.01 for r in rates],
    })
    print(f"[模拟数据] 生成 {len(df)} 条汇率数据 (币种: {symbol})")
    return df


def _parse_date(date_str):
    """解析日期字符串"""
    date_str = date_str.replace('-', '')
    return datetime.strptime(date_str, '%Y%m%d')


# ======================== 未来数据推算 ========================

def extrapolate_stock_future(df_hist, future_days=180):
    """
    基于历史数据统计特征推算未来股票数据
    使用历史均值+季节性+随机游走，标注 _predicted=True
    """
    if df_hist.empty or len(df_hist) < 30:
        return df_hist

    last_date = pd.to_datetime(df_hist['日期'].iloc[-1])
    last_close = df_hist['收盘'].iloc[-1]
    last_volume = df_hist['成交量'].iloc[-1]

    # 生成未来工作日
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=future_days)

    # 统计历史日收益率分布
    daily_returns = df_hist['收盘'].pct_change().dropna()
    mean_ret = daily_returns.mean()
    std_ret = daily_returns.std()

    # 成交量统计
    vol_mean = df_hist['成交量'].mean()
    vol_std = df_hist['成交量'].std()

    np.random.seed(42)
    rows = []
    price = last_close
    for d in future_dates:
        ret = np.random.normal(mean_ret, std_ret)
        price = max(price * (1 + ret), 0.1)
        volume = max(np.random.normal(vol_mean, vol_std), 1000)
        high = price * (1 + abs(np.random.normal(0, std_ret * 0.5)))
        low = price * (1 - abs(np.random.normal(0, std_ret * 0.5)))
        open_p = low + np.random.rand() * (high - low)
        change = price - open_p
        prev_close = rows[-1]['收盘'] if rows else last_close
        pct_change = change / (prev_close + 1e-8) * 100
        amplitude = (high - low) / (prev_close + 1e-8) * 100

        rows.append({
            '日期': d.strftime('%Y-%m-%d'),
            '开盘': round(open_p, 2),
            '收盘': round(price, 2),
            '最高': round(high, 2),
            '最低': round(low, 2),
            '成交量': int(volume),
            '成交额': round(volume * price, 0),
            '振幅': round(amplitude, 2),
            '涨跌幅': round(pct_change, 2),
            '涨跌额': round(change, 2),
            '换手率': round(np.random.uniform(0.3, 2.0), 2),
            '_predicted': True,
        })

    df_future = pd.DataFrame(rows)
    # 给历史数据也加上 _predicted 标记
    df_hist = df_hist.copy()
    df_hist['_predicted'] = False

    df_all = pd.concat([df_hist, df_future], ignore_index=True)
    print(f"[推算] 未来 {len(df_future)} 个交易日已推算 (共 {len(df_all)} 条)")
    return df_all


def extrapolate_currency_future(df_hist, future_days=180):
    """
    基于历史数据统计特征推算未来汇率数据
    标注 _predicted=True
    """
    if df_hist.empty or len(df_hist) < 20:
        return df_hist

    last_date = pd.to_datetime(df_hist['日期'].iloc[-1])
    last_rate = df_hist['中间价'].iloc[-1]

    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=future_days)

    daily_changes = df_hist['中间价'].diff().dropna()
    mean_chg = daily_changes.mean()
    std_chg = daily_changes.std()

    np.random.seed(42)
    rows = []
    rate = last_rate
    for d in future_dates:
        rate += np.random.normal(mean_chg, std_chg)
        rows.append({
            '日期': d.strftime('%Y-%m-%d'),
            '中间价': round(rate, 4),
            '买入价': round(rate - 0.01, 4),
            '卖出价': round(rate + 0.01, 4),
            '_predicted': True,
        })

    df_future = pd.DataFrame(rows)
    df_hist = df_hist.copy()
    df_hist['_predicted'] = False

    df_all = pd.concat([df_hist, df_future], ignore_index=True)
    print(f"[推算] 未来 {len(df_future)} 个交易日汇率已推算 (共 {len(df_all)} 条)")
    return df_all


# ======================== 特征工程 ========================

def add_stock_features(df):
    """添加股票技术指标特征"""
    # 基础滞后
    df['收盘_lag1'] = df['收盘'].shift(1)
    df['收盘_lag2'] = df['收盘'].shift(2)
    df['成交量_lag1'] = df['成交量'].shift(1)

    # 移动平均
    df['MA5'] = df['收盘'].rolling(5).mean()
    df['MA10'] = df['收盘'].rolling(10).mean()
    df['MA20'] = df['收盘'].rolling(20).mean()

    # 偏离均线
    df['偏离MA5'] = (df['收盘'] - df['MA5']) / (df['MA5'] + 1e-8)
    df['偏离MA10'] = (df['收盘'] - df['MA10']) / (df['MA10'] + 1e-8)

    # RSI (14日)
    delta = df['收盘'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    df['RSI14'] = 100 - 100 / (1 + rs)

    # MACD
    ema12 = df['收盘'].ewm(span=12, adjust=False).mean()
    ema26 = df['收盘'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 成交量比
    df['量比'] = df['成交量'] / (df['成交量'].rolling(5).mean() + 1e-8)

    # 涨跌方向 (目标)
    df['昨日收盘'] = df['收盘'].shift(1)
    df['涨跌方向'] = (df['收盘'] > df['昨日收盘']).astype(int)

    return df.dropna()


def add_currency_features(df):
    """添加汇率技术指标特征"""
    price_col = '中间价' if '中间价' in df.columns else df.columns[1]

    # 滞后
    for lag in [1, 2, 3]:
        df[f'{price_col}_lag{lag}'] = df[price_col].shift(lag)

    # 移动平均
    df['MA5'] = df[price_col].rolling(5).mean()
    df['MA10'] = df[price_col].rolling(10).mean()
    df['MA20'] = df[price_col].rolling(20).mean()

    # 偏离
    df['偏离MA5'] = (df[price_col] - df['MA5']) / (df['MA5'] + 1e-8)

    # RSI
    delta = df[price_col].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    df['RSI14'] = 100 - 100 / (1 + rs)

    # 波动率
    df['波动率5'] = df[price_col].rolling(5).std()

    # 涨跌方向
    df['昨日价格'] = df[price_col].shift(1)
    df['涨跌方向'] = (df[price_col] > df['昨日价格']).astype(int)

    return df.dropna()
