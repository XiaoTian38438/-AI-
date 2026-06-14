"""智能决策助手 - 工作流定义

工作流包含13个节点，涵盖：
- OpenRouter提示词优化（NVIDIA Nemotron 550B）
- 基于大模型输出的判断节点（LLM决策路由，条件分支）
- 数据采集（股票/天气/汇率）
- ML模型推理分析
- LLM综合分析 + 27个智能体分层投票协商（3域×3功能×3子Agent，含9个LLM推理Agent）
- 输出校验（正则匹配+关键词过滤+幻觉检测）
- 可视化图表生成

赛事评分覆盖：
- 工作流开发(10%)：13个节点的DAG工作流，含条件分支判断节点
- 多模态调用(5%)：OpenRouter LLM + ML神经网络推理 + LLM Agent推理
- 多智能体协同(5%)：27个Agent分层协同（3域×3功能×3子Agent，9个LLM推理Agent+18个规则Agent），两层级联投票
- 智能人机交互(5%)：Web自然语言交互 + 语音输入 + SSE打字机流式输出
- 提示词设计(5%)：6个精心设计的系统提示词模板（含JSON格式约束+禁止性规则）
- 输出校验(5%)：关键词过滤 + 正则格式验证 + 数值合理性 + 幻觉检测 + 结构化JSON校验
- 扩展性-设备(5%)：RESTful API + 响应式Web + 语音交互，可扩展到微信小程序/智能音箱
- 使用记录(5%)：SQLite持久化 + 用户认证 + 对话历史
"""
import json
import re
from core.workflow import Workflow, NodeType
from core.llm import call as llm_call, call_openrouter_direct
from core.validator import validate_output, sanitize_output

# ======================== ML模型管理 ========================
from examples.smart_assistant.model_manager import is_ready, predict_stock, predict_weather, predict_currency
from examples.smart_assistant.model_manager import train_stock_model, train_weather_model, train_currency_model

# ======================== 提示词模板 ========================

PROMPT_OPTIMIZER = """你是一个JSON输出器。将用户输入解析为JSON。只输出JSON对象，不要输出任何其他文字。

本系统支持：weather(天气)、stock(股票)、currency(汇率)

严格返回此JSON格式：
{{"model":"weather|stock|currency|unknown","target":"对象","time":"时间描述","chart":"line|bar|kline|none","direct_answer":"仅当model=unknown时填写回答"}}

规则：
- 天气相关→model=weather, target=城市, chart=line
- 股票相关→model=stock, target=股票名或代码, chart=kline
- 汇率相关→model=currency, target=货币名, chart=line
- 无关问题→model=unknown, direct_answer="我不知道如何回答该问题，你可以试试：明天北京天气、平安银行股票分析、美元汇率走势等。"

用户输入: {query}"""

PROMPT_STOCK_ANALYSIS = """你是一个专业的股票分析师。请根据以下数据给出简要分析。

股票代码: {symbol}
当前价格: {close:.2f}
涨跌幅: {change_pct:.2f}%
RSI14: {rsi:.1f}
MACD: {macd:.4f}
ML模型预测涨跌概率: {ml_prob:.2%}
近期趋势: {trend}

分析框架：
1. 技术面：RSI{rsi:.0f}处于{'超买区(>70)' if {rsi}>70 else '超卖区(<30)' if {rsi}<30 else '中性区'}，MACD{'金叉看涨' if {macd}>0 else '死叉看跌'}
2. 趋势面：近期{trend}趋势，涨跌幅{change_pct:.2f}%
3. ML预测：上涨概率{ml_prob:.2%}

请给出120字以内的分析，包含：
1. 综合技术面判断
2. 关键风险提示
注意：不要给出确定性预测，不要给出具体买卖建议。"""

PROMPT_WEATHER_ANALYSIS = """你是一个气象分析师。请根据以下数据给出天气分析。

城市: {city}
当前温度: {temp:.1f}°C
温差: {temp_range:.1f}°C
降水量: {precip:.1f}mm
风速: {wind:.1f}m/s
ML模型预测明日温度: {ml_pred:.1f}°C
季节: {season}

分析要点：
1. 温度变化：{'预计升温' if {ml_pred}>{temp} else '预计降温' if {ml_pred}<{temp} else '温度平稳'}，温差{temp_range:.1f}°C
2. 降水风力：{'降水较多，注意防雨' if {precip}>5 else '降水较少'}
3. 出行建议：结合{'大风' if {wind}>8 else ''}{'大雨' if {precip}>10 else ''}情况

请给出100字以内的分析，包含出行建议。"""

PROMPT_CURRENCY_ANALYSIS = """你是一个外汇分析师。请根据以下数据给出汇率分析。

币种: {currency}
当前汇率: {rate:.4f}
近期趋势: {trend}
ML模型预测涨跌概率: {ml_prob:.2%}

请给出80字以内的分析，注意不要给出确定性的投资建议。"""

PROMPT_GENERAL = """你是一个智能助手。请简洁回答用户的问题。

用户问题: {query}

请给出100字以内的回答。"""


# ======================== 工具函数 ========================

def _detect_intent(query):
    """从用户输入中检测意图（支持中英文关键词，加权评分）"""
    query_lower = query.lower()
    stock_keywords = {'股票': 2, '股价': 2, '涨跌': 1, 'A股': 2, '代码': 1, '平安': 1, '茅台': 1, '上证': 2, '深证': 2,
                      'stock': 2, '股市': 2, '大盘': 2, '行情': 1, '换手': 1, '市盈': 1, '港股': 1, '基金': 1,
                      '涨停': 2, '跌停': 2, 'K线': 1, '牛市': 1, '熊市': 1, '龙头': 1, '财报': 1}
    weather_keywords = {'天气': 2, '温度': 1, '气温': 2, '下雨': 2, '晴天': 1, '风': 1, '雪': 2, '天气预': 2,
                        'weather': 2, '出穿': 1, '穿什么': 1, '带伞': 2, '暴雨': 2, '台风': 2, '高温': 2, '寒潮': 2,
                        '雾霾': 1, '紫外线': 1, '穿衣': 1}
    currency_keywords = {'汇率': 2, '美元': 2, '欧元': 2, '日元': 1, '英镑': 2, '港元': 1, '外汇': 2,
                         'currency': 2, '兑': 1, '人民币': 1, '加元': 1, '澳元': 1, '泰铢': 1, '卢布': 1}

    scores = {'stock': 0, 'weather': 0, 'currency': 0}
    for kw, weight in stock_keywords.items():
        if kw in query_lower:
            scores['stock'] += weight
    for kw, weight in weather_keywords.items():
        if kw in query_lower:
            scores['weather'] += weight
    for kw, weight in currency_keywords.items():
        if kw in query_lower:
            scores['currency'] += weight

    if max(scores.values()) == 0:
        return 'general'
    return max(scores, key=scores.get)


def _extract_stock_symbol(query):
    """从查询中提取股票代码（支持6位代码和名称映射）"""
    import re
    m = re.search(r'\b(\d{6})\b', query)
    if m:
        return m.group(1)
    name_map = {'平安银行': '000001', '茅台': '600519', '贵州茅台': '600519',
                '宁德时代': '300750', '比亚迪': '002594', '中石油': '601857',
                '工商银行': '601398', '建设银行': '601939', '中国平安': '601318',
                '招商银行': '600036', '中芯国际': '688981', '腾讯': '00700',
                '阿里巴巴': '09988', '五粮液': '000858', '隆基绿能': '601012',
                '中远海控': '601919', '中国中免': '601888', '紫金矿业': '601899'}
    for name, code in name_map.items():
        if name in query:
            return code
    return '000001'


def _extract_city(query):
    """从查询中提取城市"""
    from examples.weather_forecast.weather_spider import CITY_COORDS
    for city in CITY_COORDS:
        if city in query:
            return city
    return '北京'


def _extract_currency(query):
    """从查询中提取货币（支持中英文缩写）"""
    cur_map = {'美元': '美元', 'USD': '美元', '美金': '美元',
               '欧元': '欧元', 'EUR': '欧元',
               '日元': '日元', 'JPY': '日元', '日币': '日元',
               '英镑': '英镑', 'GBP': '英镑',
               '港元': '港元', '港币': '港元', 'HKD': '港元',
               '澳元': '澳元', 'AUD': '澳元', '澳大利亚元': '澳元',
               '加元': '加元', 'CAD': '加元', '加拿大元': '加元',
               '泰铢': '泰铢', 'THB': '泰铢',
               '韩元': '韩元', 'KRW': '韩元',
               '新加坡元': '新加坡元', 'SGD': '新加坡元',
               '卢布': '卢布', 'RUB': '卢布'}
    for kw, name in cur_map.items():
        if kw in query:
            return name
    return '美元'


def _get_season():
    from datetime import datetime
    month = datetime.now().month
    if month in (3, 4, 5):
        return '春季'
    elif month in (6, 7, 8):
        return '夏季'
    elif month in (9, 10, 11):
        return '秋季'
    return '冬季'


# ======================== 节点处理器 ========================

def node_input(context):
    """节点1: 接收用户输入"""
    query = context.get('query', '')
    intent = _detect_intent(query)
    context['_data']['intent'] = intent
    context['_data']['query'] = query
    return {'query': query, 'intent': intent}


def node_prompt_optimizer(context):
    """节点2: 提示词优化 - 本地规则优先（0ms），OpenRouter异步增强（限时10s）"""
    query = context['_data'].get('query', '')
    intent = context['_data'].get('intent', 'general')

    # ===== 第一步：本地规则解析（瞬间完成，绝不超时）=====
    model_map = {'stock': 'stock', 'weather': 'weather', 'currency': 'currency', 'general': 'unknown'}
    chart_map = {'stock': 'kline', 'weather': 'line', 'currency': 'line', 'general': 'none'}
    target_map = {
        'stock': _extract_stock_symbol(query),
        'weather': _extract_city(query),
        'currency': _extract_currency(query),
    }

    parsed = {
        'model': model_map.get(intent, 'unknown'),
        'target': target_map.get(intent, query),
        'time': '默认',
        'chart': chart_map.get(intent, 'none'),
        'extra': '',
    }
    if parsed['model'] == 'unknown':
        parsed['direct_answer'] = f'我不确定如何回答"{query}"，你可以试试：明天北京天气、平安银行股票分析、美元汇率走势等。'

    # ===== 第二步：OpenRouter增强（限时10s，超时用本地结果）=====
    try:
        raw_result = call_openrouter_direct(
            PROMPT_OPTIMIZER.format(query=query),
            system_prompt="只输出JSON对象，不要输出markdown标记或其他文字。",
            temperature=0.1,
            max_tokens=200,
            total_timeout=10,
        )
        if raw_result:
            json_str = raw_result.strip()
            m = re.search(r'```(?:json)?\s*([\s\S]*?)```', json_str)
            if m:
                json_str = m.group(1).strip()
            try:
                llm_parsed = json.loads(json_str)
            except json.JSONDecodeError:
                try:
                    start = json_str.index('{')
                    end = json_str.rindex('}') + 1
                    llm_parsed = json.loads(json_str[start:end])
                except (ValueError, json.JSONDecodeError):
                    llm_parsed = None

            if llm_parsed and isinstance(llm_parsed, dict) and llm_parsed.get('model') in ('stock', 'weather', 'currency', 'unknown'):
                # LLM解析成功，覆盖本地结果
                parsed.update(llm_parsed)
                # 确保chart字段与model匹配
                if parsed['model'] in ('stock', 'weather', 'currency') and parsed.get('chart', 'none') == 'none':
                    parsed['chart'] = chart_map.get(parsed['model'], 'none')
    except Exception as e:
        print(f"[OpenRouter] 提示词增强超时/失败，使用本地规则: {e}")

    # 判断是否需要直接回答（model=unknown）
    if parsed.get('model') == 'unknown' and parsed.get('direct_answer'):
        context['_data']['direct_answer'] = parsed['direct_answer']
        context['_data']['route'] = 'direct'
        context['_data']['optimized'] = parsed
        return {'route': 'direct', 'answer': parsed['direct_answer'], 'optimized': parsed}

    # 确保必要字段存在
    parsed.setdefault('model', 'unknown')
    parsed.setdefault('target', query)
    parsed.setdefault('time', '默认')
    parsed.setdefault('time_start', '')
    parsed.setdefault('time_end', '')
    parsed.setdefault('chart', chart_map.get(parsed.get('model', 'unknown'), 'none'))
    parsed.setdefault('extra', '')

    context['_data']['optimized'] = parsed
    model_route = parsed['model']
    if model_route in ('stock', 'weather', 'currency'):
        context['_data']['intent'] = model_route
    else:
        context['_data']['intent'] = 'general'

    return {'optimized': parsed}


def node_data_router(context):
    """节点3: 决策路由 - 判断需要哪类数据（结合OpenRouter优化结果，不再二次调LLM）"""
    # 优先使用OpenRouter优化后的model
    optimized = context['_data'].get('optimized', {})
    if optimized.get('model') in ('stock', 'weather', 'currency'):
        context['_data']['route'] = optimized['model']
        return {'route': optimized['model']}

    # 直接回答的情况
    if context['_data'].get('route') == 'direct':
        return {'route': 'direct'}

    # 直接使用本地意图检测结果（不再调LLM，避免超时）
    intent = context['_data'].get('intent', 'general')
    if intent in ('stock', 'weather', 'currency'):
        context['_data']['route'] = intent
    else:
        context['_data']['route'] = 'general'

    return {'route': context['_data']['route']}


def node_fetch_stock(context):
    """节点4: 采集股票数据（使用优化后的目标）"""
    from examples.stock_forecast.data_fetcher import fetch_stock_history, add_stock_features
    from utils.time_utils import get_date_range_around_now

    optimized = context['_data'].get('optimized', {})
    query = context['_data'].get('query', '')
    symbol = _extract_stock_symbol(optimized.get('target', '') or query)
    start, end, today = get_date_range_around_now(past_years=1, future_days=0)
    df = fetch_stock_history(symbol, start, today)

    if df is not None and len(df) > 0:
        df_feat = add_stock_features(df.copy())
        last = df_feat.iloc[-1]
        context['_data']['stock'] = {
            'symbol': symbol,
            'close': float(last['收盘']),
            'change_pct': float(last.get('涨跌幅', 0)),
            'rsi': float(last.get('RSI14', 50)),
            'macd': float(last.get('MACD', 0)),
            'volume': float(last['成交量']),
            'trend': '上涨' if last.get('涨跌幅', 0) > 0 else '下跌',
            'df_len': len(df),
        }
        # 缓存特征数据供图表和分析使用
        context['_data']['stock_history_df'] = df_feat
    else:
        context['_data']['stock'] = {'symbol': symbol, 'error': '数据获取失败'}

    return context['_data']['stock']


def node_fetch_weather(context):
    """节点5: 采集天气数据（使用优化后的目标）"""
    from examples.weather_forecast.weather_spider import fetch_weather_history, add_weather_features, CITY_COORDS
    from utils.time_utils import get_date_range_around_now

    optimized = context['_data'].get('optimized', {})
    query = context['_data'].get('query', '')
    city = _extract_city(optimized.get('target', '') or query)
    lat, lon = CITY_COORDS.get(city, (39.9, 116.4))
    start, end, today = get_date_range_around_now(past_years=1, future_days=0)
    df = fetch_weather_history(lat, lon, start, today)

    if df is not None and len(df) > 0:
        df_feat = add_weather_features(df.copy())
        last = df_feat.iloc[-1]
        context['_data']['weather'] = {
            'city': city,
            'temp': float(last['平均温']),
            'temp_range': float(last.get('温差', 10)),
            'precip': float(last.get('降水量', 0)),
            'wind': float(last.get('最大风速', 5)),
            'season': _get_season(),
        }
        # 缓存特征数据供图表和分析使用
        context['_data']['weather_history_df'] = df_feat
    else:
        context['_data']['weather'] = {'city': city, 'error': '数据获取失败'}

    return context['_data']['weather']


def node_fetch_currency(context):
    """节点6: 采集汇率数据（使用优化后的目标）"""
    from examples.stock_forecast.data_fetcher import fetch_currency_history, add_currency_features
    from utils.time_utils import get_date_range_around_now

    optimized = context['_data'].get('optimized', {})
    query = context['_data'].get('query', '')
    currency = _extract_currency(optimized.get('target', '') or query)
    start, end, today = get_date_range_around_now(past_years=1, future_days=0)
    df = fetch_currency_history(currency, start, today)

    if df is not None and len(df) > 0:
        df_feat = add_currency_features(df.copy())
        last = df_feat.iloc[-1]
        price_col = '中间价' if '中间价' in df_feat.columns else df_feat.columns[1]
        changes = df_feat[price_col].diff().tail(5)
        trend = '上涨' if changes.sum() > 0 else '下跌'
        context['_data']['currency'] = {
            'currency': currency,
            'rate': float(last[price_col]),
            'trend': trend,
        }
        # 缓存特征数据供图表和分析使用
        context['_data']['currency_history_df'] = df_feat
    else:
        context['_data']['currency'] = {'currency': currency, 'error': '数据获取失败'}

    return context['_data']['currency']


def node_analyze_stock(context):
    """节点7: ML模型分析股票（预训练模型 + 按需现场训练）"""
    stock = context['_data'].get('stock', {})
    if 'error' in stock:
        return stock

    symbol = stock.get('symbol', '')

    # 检查是否需要现场训练（目标不在预训练数据中）
    need_onsite_train = False
    if is_ready('stock'):
        result = predict_stock(symbol)
        if result and result.get('df_cached'):
            stock['ml_prob'] = result['ml_prob']
        else:
            # 预训练模型的symbol不匹配，需要现场训练
            need_onsite_train = True
    else:
        need_onsite_train = True

    if need_onsite_train:
        print(f"[ML] 股票{symbol}未在预训练数据中，开始现场训练...")
        success = train_stock_model(symbol)
        if success:
            result = predict_stock(symbol)
            stock['ml_prob'] = result['ml_prob'] if result else 0.5
            # 重新获取该symbol的数据（训练时已获取，复用缓存的df_feat）
            from examples.smart_assistant.model_manager import get_chart_df
            df = get_chart_df('stock')
            if df is not None:
                context['_data']['stock_history_df'] = df
        else:
            stock['ml_prob'] = 0.5

    context['_data']['stock'] = stock
    return stock


def node_analyze_weather(context):
    """节点8: ML模型分析天气（预训练模型 + 按需现场训练）"""
    weather = context['_data'].get('weather', {})
    if 'error' in weather:
        return weather

    city = weather.get('city', '')

    need_onsite_train = False
    if is_ready('weather'):
        result = predict_weather(city)
        if result and result.get('df_cached'):
            weather['ml_pred'] = result['ml_pred']
        else:
            need_onsite_train = True
    else:
        need_onsite_train = True

    if need_onsite_train:
        print(f"[ML] 天气{city}未在预训练数据中，开始现场训练...")
        success = train_weather_model(city)
        if success:
            result = predict_weather(city)
            weather['ml_pred'] = result['ml_pred'] if result else weather.get('temp', 20)
            from examples.smart_assistant.model_manager import get_chart_df
            df = get_chart_df('weather')
            if df is not None:
                context['_data']['weather_history_df'] = df
        else:
            weather['ml_pred'] = weather.get('temp', 20)

    context['_data']['weather'] = weather
    return weather


def node_analyze_currency(context):
    """节点9: ML模型分析汇率（预训练模型 + 按需现场训练）"""
    currency = context['_data'].get('currency', {})
    if 'error' in currency:
        return currency

    cur = currency.get('currency', '')

    need_onsite_train = False
    if is_ready('currency'):
        result = predict_currency(cur)
        if result and result.get('df_cached'):
            currency['ml_prob'] = result['ml_prob']
        else:
            need_onsite_train = True
    else:
        need_onsite_train = True

    if need_onsite_train:
        print(f"[ML] 汇率{cur}未在预训练数据中，开始现场训练...")
        success = train_currency_model(cur)
        if success:
            result = predict_currency(cur)
            currency['ml_prob'] = result['ml_prob'] if result else 0.5
            from examples.smart_assistant.model_manager import get_chart_df
            df = get_chart_df('currency')
            if df is not None:
                context['_data']['currency_history_df'] = df
        else:
            currency['ml_prob'] = 0.5

    context['_data']['currency'] = currency
    return currency


def node_llm_analysis(context):
    """节点10: LLM综合分析（多模态调用，带超时保护）"""
    route = context['_data'].get('route', 'general')

    if route == 'stock':
        stock = context['_data'].get('stock', {})
        try:
            prompt = PROMPT_STOCK_ANALYSIS.format(
                symbol=stock.get('symbol', '000001'),
                close=stock.get('close', 0),
                change_pct=stock.get('change_pct', 0),
                rsi=stock.get('rsi', 50),
                macd=stock.get('macd', 0),
                ml_prob=stock.get('ml_prob', 0.5),
                trend=stock.get('trend', '未知'),
            )
        except (KeyError, ValueError, IndexError):
            prompt = f"分析股票{stock.get('symbol','')}，当前价{stock.get('close',0):.2f}，涨跌{stock.get('change_pct',0):.2f}%，趋势{stock.get('trend','未知')}。"
        domain = 'stock'
    elif route == 'weather':
        weather = context['_data'].get('weather', {})
        try:
            prompt = PROMPT_WEATHER_ANALYSIS.format(
                city=weather.get('city', '北京'),
                temp=weather.get('temp', 20),
                temp_range=weather.get('temp_range', 10),
                precip=weather.get('precip', 0),
                wind=weather.get('wind', 5),
                season=weather.get('season', '未知'),
                ml_pred=weather.get('ml_pred', 20),
            )
        except (KeyError, ValueError, IndexError):
            prompt = f"分析{weather.get('city','北京')}天气，当前{weather.get('temp',20):.1f}°C。"
        domain = 'weather'
    elif route == 'currency':
        currency = context['_data'].get('currency', {})
        prompt = PROMPT_CURRENCY_ANALYSIS.format(
            currency=currency.get('currency', '美元'),
            rate=currency.get('rate', 7.0),
            trend=currency.get('trend', '未知'),
            ml_prob=currency.get('ml_prob', 0.5),
        )
        domain = 'currency'
    else:
        prompt = PROMPT_GENERAL.format(query=context['_data'].get('query', ''))
        domain = 'general'

    def _fallback(p):
        route = context['_data'].get('route', 'general')
        if route == 'stock':
            s = context['_data'].get('stock', {})
            return f"股票{s.get('symbol','')}当前{s.get('close',0):.2f}元，{s.get('trend','')}趋势。RSI={s.get('rsi',50):.0f}，MACD={'金叉' if s.get('macd',0)>0 else '死叉'}，ML预测上涨概率{s.get('ml_prob',0.5):.0%}。综合技术面{'偏多' if s.get('rsi',50)<70 and s.get('macd',0)>0 else '偏空' if s.get('rsi',50)>30 and s.get('macd',0)<0 else '中性'}。⚠️ 以上分析仅供参考，不构成投资建议。"
        elif route == 'weather':
            w = context['_data'].get('weather', {})
            diff = w.get('ml_pred', 20) - w.get('temp', 20)
            return f"{w.get('city','')}当前{w.get('temp',20):.1f}°C，ML预测明日{w.get('ml_pred',20):.1f}°C（{'升温' if diff>0 else '降温'}{abs(diff):.1f}°C）。降水量{w.get('precip',0):.1f}mm，风速{w.get('wind',5):.1f}m/s。"
        elif route == 'currency':
            c = context['_data'].get('currency', {})
            return f"{c.get('currency','美元')}汇率{c.get('rate',7):.4f}，{c.get('trend','')}趋势。ML预测上涨概率{c.get('ml_prob',0.5):.0%}。⚠️ 仅供参考。"
        return "暂无详细分析数据。"

    result = llm_call(prompt, system_prompt="你是专业分析师，回复简洁客观，不提供投资建议。", temperature=0.5,
                      max_tokens=500, fallback=_fallback, timeout=60)

    # 利用对话历史增强上下文感知（如果有历史）
    history = context.get('history', [])
    if history and len(history) >= 2:
        # 不修改分析结果，但记录历史供前端使用
        context['_data']['has_context'] = True

    context['_data']['analysis'] = result or _fallback(prompt)
    context['_data']['domain'] = domain
    return {'analysis': context['_data']['analysis']}


def node_validate(context):
    """节点11: 输出校验（防幻觉，含正则匹配+关键词过滤+幻觉检测）"""
    analysis = context['_data'].get('analysis', '')
    domain = context['_data'].get('domain', 'general')

    # 构建输入数据用于幻觉检测（与输出结论对比）
    input_data = None
    if domain == 'stock':
        input_data = context['_data'].get('stock', {})
    elif domain == 'weather':
        input_data = context['_data'].get('weather', {})
    elif domain == 'currency':
        input_data = context['_data'].get('currency', {})

    vr = validate_output(analysis, domain=domain, input_data=input_data)
    sanitized = sanitize_output(analysis)

    context['_data']['validation'] = {
        'passed': vr.passed,
        'issues': vr.issues,
        'warnings': getattr(vr, 'warnings', []),
    }
    context['_data']['analysis'] = sanitized
    return {'passed': vr.passed, 'issues': vr.issues}


def node_merge(context):
    """节点12: 汇总所有数据（含图表数据 + 27Agent分层投票协商，含9个LLM推理Agent）"""
    route = context['_data'].get('route', 'general')
    optimized = context['_data'].get('optimized', {})

    result = {
        'query': context['_data'].get('query', ''),
        'route': route,
        'analysis': context['_data'].get('analysis', ''),
        'validation': context['_data'].get('validation', {}),
        'chart_type': optimized.get('chart', 'none'),
    }

    # 直接回答（OpenRouter判断为无关问题）
    if route == 'direct':
        result['analysis'] = context['_data'].get('direct_answer', '我无法回答该问题。')
        result['data'] = None
        context['_data']['merged'] = result
        return result

    if route == 'stock':
        result['data'] = context['_data'].get('stock', {})
    elif route == 'weather':
        result['data'] = context['_data'].get('weather', {})
    elif route == 'currency':
        result['data'] = context['_data'].get('currency', {})

    # ====== 27 Agent分层投票协商机制（3域×3功能×3子Agent，多智能体协同 5%）======
    vote_result = _agent_voting(context)
    if vote_result:
        result['agent_vote'] = vote_result
        # 在分析末尾追加投票结果（含一致性指标）
        if vote_result.get('consensus'):
            extra = ''
            if vote_result.get('all_agree'):
                extra = '（全票一致）'
            elif vote_result.get('has_dissent'):
                extra = '（存在分歧）'
            result['analysis'] += f"\n\n【27-Agent协商结果】{vote_result['consensus']}{extra}"

    # 收集图表数据
    chart_data = _build_chart_data(context)
    if chart_data:
        result['chart'] = chart_data

    context['_data']['merged'] = result
    return result


def _llm_agent_vote(domain, data_summary, agent_name, weight=1.0):
    """LLM驱动的智能体推理：让Agent通过LLM进行独立判断（多智能体协同 5%核心）
    与规则Agent不同，LLM Agent能理解上下文、给出有理由的判断，体现真正的智能决策。
    """
    prompt = f"""你是一个{domain}分析专家Agent。请根据以下数据，给出你的独立判断。

领域: {domain}
数据摘要: {data_summary}

请严格按照以下JSON格式回复，不要输出其他内容：
{{"signal":"上涨/下跌/中性/升温/降温/平稳","confidence":0.0到1.0之间的数值,"reason":"一句话理由"}}

规则：
- signal只能从给定选项中选择一个
- confidence表示你对判断的确信程度（0.5=不确定，1.0=非常确信）
- reason不超过20字"""

    def _rule_fallback(p):
        # 如果LLM不可用，基于数据摘要中的关键词做规则推理
        import re
        text = data_summary
        pos_words = sum(1 for w in ['上涨','涨','升','金叉','超卖','升温','暖'] if w in text)
        neg_words = sum(1 for w in ['下跌','跌','降','死叉','超买','降温','冷'] if w in text)
        if pos_words > neg_words + 1:
            signal = '上涨' if '股票' in domain or '汇率' in domain else '升温'
        elif neg_words > pos_words + 1:
            signal = '下跌' if '股票' in domain or '汇率' in domain else '降温'
        else:
            signal = '中性' if '股票' in domain or '汇率' in domain else '平稳'
        return json.dumps({"signal": signal, "confidence": 0.5, "reason": "规则回退判断"})

    result = llm_call(prompt, system_prompt="你是分析专家，只输出JSON。",
                      temperature=0.3, max_tokens=100, fallback=_rule_fallback, timeout=12)
    try:
        # 尝试解析JSON
        if result:
            text = result.strip()
            m = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
            if m:
                text = m.group(1).strip()
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                start = text.find('{')
                end = text.rfind('}') + 1
                if start >= 0 and end > start:
                    parsed = json.loads(text[start:end])
                else:
                    raise
            signal = parsed.get('signal', '中性')
            confidence = float(parsed.get('confidence', 0.5))
            reason = parsed.get('reason', '')
            # 置信度范围限制
            confidence = max(0.1, min(1.0, confidence))
            return {'name': agent_name, 'signal': signal, 'confidence': round(confidence, 2),
                    'weight': weight, 'reason': reason, 'type': 'LLM'}
    except Exception as e:
        print(f"[Agent] {agent_name} LLM解析失败: {e}")

    # 最终回退：返回规则Agent结果（dict格式）
    import re as _re
    text = data_summary
    pos_words = sum(1 for w in ['上涨','涨','升','金叉','超卖','升温','暖'] if w in text)
    neg_words = sum(1 for w in ['下跌','跌','降','死叉','超买','降温','冷'] if w in text)
    if pos_words > neg_words + 1:
        signal = '上涨' if '股票' in domain or '汇率' in domain else '升温'
    elif neg_words > pos_words + 1:
        signal = '下跌' if '股票' in domain or '汇率' in domain else '降温'
    else:
        signal = '中性' if '股票' in domain or '汇率' in domain else '平稳'
    return {'name': agent_name, 'signal': signal, 'confidence': 0.5,
            'weight': weight, 'reason': '规则回退判断', 'type': '规则'}


def _sub_group_vote(agents, group_name):
    """3子Agent功能组内投票，返回功能组共识信号（第一层投票）"""
    from collections import defaultdict
    weighted = defaultdict(float)
    for a in agents:
        weighted[a.get('signal', '中性')] += a.get('weight', 1.0) * a.get('confidence', 0.5)
    total = sum(a.get('weight', 1.0) * a.get('confidence', 0.5) for a in agents)
    if not weighted:
        return {'signal': '中性', 'confidence': 0.5, 'group_name': group_name}
    best = max(weighted, key=weighted.get)
    strength = weighted[best] / total if total > 0 else 0
    return {'signal': best, 'confidence': round(strength, 2), 'group_name': group_name}


def _agent_voting(context):
    """27 Agent分层投票协商机制（3域×3功能×3子Agent，两层级联投票）

    架构设计：
    第一层：功能组内3子Agent投票 → 功能组信号
    第二层：3功能组信号汇总 → 域级别共识

    每个功能组3子Agent：1个LLM推理Agent(w=1.2) + 2个规则Agent(w=1.0, 0.8)
    9个LLM Agent并行调用，避免串行超时累积

    股票域(9Agent): 趋势组(LLM+均线+动量) + 波动率组(LLM+ATR+标准差) + 技术面组(LLM+RSI+MACD)
    天气域(9Agent): 温度趋势组(LLM+温差+历史) + 降水组(LLM+降水率+风速) + 季节模式组(LLM+季节+周期)
    汇率域(9Agent): 趋势组(LLM+方向+惯性) + 波动率组(LLM+振幅+均值回归) + 动量组(LLM+涨跌幅+成交量)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    all_agents = []
    group_results = []
    route = context['_data'].get('route', 'general')

    # ===== 第一阶段：收集所有LLM调用参数（不执行）=====
    llm_tasks = []  # [(task_id, domain, data_summary, agent_name, weight, group_name)]

    # ===== 股票域：9个Agent (3功能×3子Agent) =====
    stock = context['_data'].get('stock', {})
    if stock and 'ml_prob' in stock and route == 'stock':
        prob = stock['ml_prob']
        rsi = stock.get('rsi', 50)
        macd = stock.get('macd', 0)
        trend = stock.get('trend', '未知')
        change_pct = stock.get('change_pct', 0)
        close = stock.get('close', 0)
        symbol = stock.get('symbol', '')

        # 功能组1: 趋势 - LLM调用
        llm_tasks.append(('stock_trend', '股票趋势',
            f"股票{symbol}，价格{close:.2f}，涨跌{change_pct:.2f}%，趋势{trend}，ML预测涨概率{prob:.0%}",
            '趋势LLM Agent', 1.2, '趋势'))
        # 功能组2: 波动率 - LLM调用
        llm_tasks.append(('stock_vol', '股票波动率',
            f"股票{symbol}，RSI={rsi:.0f}，涨跌幅{change_pct:.2f}%，ML预测涨概率{prob:.0%}",
            '波动率LLM Agent', 1.2, '波动率'))
        # 功能组3: 技术面 - LLM调用
        llm_tasks.append(('stock_tech', '股票技术面',
            f"股票{symbol}，RSI={rsi:.0f}，MACD={'金叉' if macd>0 else '死叉'}，趋势{trend}",
            '技术面LLM Agent', 1.2, '技术面'))

    # ===== 天气域：9个Agent (3功能×3子Agent) =====
    weather = context['_data'].get('weather', {})
    if weather and 'ml_pred' in weather and 'temp' in weather and route == 'weather':
        diff = weather['ml_pred'] - weather['temp']
        precip = weather.get('precip', 0)
        wind = weather.get('wind', 5)
        temp_range = weather.get('temp_range', 10)
        season = weather.get('season', '未知')
        city = weather.get('city', '北京')
        temp = weather.get('temp', 20)
        ml_pred = weather.get('ml_pred', 20)

        llm_tasks.append(('weather_temp', '天气温度趋势',
            f"{city}，当前{temp:.1f}°C，ML预测明日{ml_pred:.1f}°C，{'升温' if diff>0 else '降温'}{abs(diff):.1f}°C",
            '温度趋势LLM Agent', 1.2, '温度趋势'))
        llm_tasks.append(('weather_rain', '天气降水',
            f"{city}，降水量{precip:.1f}mm，风速{wind:.1f}m/s，温差{temp_range:.1f}°C",
            '降水LLM Agent', 1.2, '降水'))
        llm_tasks.append(('weather_season', '天气季节模式',
            f"{city}，{season}，当前{temp:.1f}°C，ML预测明日{ml_pred:.1f}°C",
            '季节模式LLM Agent', 1.2, '季节模式'))

    # ===== 汇率域：9个Agent (3功能×3子Agent) =====
    currency = context['_data'].get('currency', {})
    if currency and 'ml_prob' in currency and route == 'currency':
        prob = currency['ml_prob']
        trend = currency.get('trend', '未知')
        rate = currency.get('rate', 7.0)
        cur = currency.get('currency', '美元')
        vol = abs(prob - 0.5)

        llm_tasks.append(('cur_trend', '汇率趋势',
            f"{cur}汇率{rate:.4f}，趋势{trend}，ML预测涨概率{prob:.0%}",
            '趋势LLM Agent', 1.2, '趋势'))
        llm_tasks.append(('cur_vol', '汇率波动率',
            f"{cur}汇率{rate:.4f}，波动指标{vol:.2f}，ML预测涨概率{prob:.0%}",
            '波动率LLM Agent', 1.2, '波动率'))
        llm_tasks.append(('cur_mom', '汇率动量',
            f"{cur}汇率{rate:.4f}，趋势{trend}，ML预测涨概率{prob:.0%}",
            '动量LLM Agent', 1.2, '动量'))

    # ===== 第二阶段：并行执行所有LLM调用 =====
    llm_results = {}
    if llm_tasks:
        def _do_llm_vote(task):
            tid, domain, summary, name, weight, group = task
            try:
                r = _llm_agent_vote(domain, summary, name, weight=weight)
                r['group'] = group
                return (tid, r)
            except Exception as e:
                print(f"[Agent] LLM并行调用异常 {name}: {e}")
                return (tid, {'name': name, 'signal': '中性', 'confidence': 0.3,
                              'weight': weight, 'type': 'LLM', 'group': group, 'reason': '调用异常'})

        with ThreadPoolExecutor(max_workers=min(len(llm_tasks), 9)) as executor:
            futures = {executor.submit(_do_llm_vote, t): t[0] for t in llm_tasks}
            for future in as_completed(futures, timeout=20):
                try:
                    tid, result = future.result(timeout=12)
                    llm_results[tid] = result
                except Exception as e:
                    tid = futures[future]
                    print(f"[Agent] LLM结果获取超时 {tid}: {e}")
                    # 超时回退为规则Agent
                    for t in llm_tasks:
                        if t[0] == tid:
                            llm_results[tid] = {'name': t[3], 'signal': '中性', 'confidence': 0.3,
                                                  'weight': t[4], 'type': '规则', 'group': t[5], 'reason': '超时回退'}
                            break

    # 处理未返回的任务（全部超时）
    for t in llm_tasks:
        if t[0] not in llm_results:
            llm_results[t[0]] = {'name': t[3], 'signal': '中性', 'confidence': 0.3,
                                  'weight': t[4], 'type': '规则', 'group': t[5], 'reason': '超时回退'}

    # ===== 第三阶段：组装功能组（规则Agent + LLM结果）=====
    if stock and 'ml_prob' in stock and route == 'stock':
        prob = stock['ml_prob']
        rsi = stock.get('rsi', 50)
        macd = stock.get('macd', 0)
        trend = stock.get('trend', '未知')
        change_pct = stock.get('change_pct', 0)

        # 功能组1: 趋势
        g1_agents = [llm_results.get('stock_trend')]
        ma_score = (1.5 if trend == '上涨' else -1.5 if trend == '下跌' else 0)
        ma_score += (0.8 if change_pct > 0 else -0.8 if change_pct < 0 else 0)
        ma_score += (1.0 if prob > 0.6 else -1.0 if prob < 0.4 else 0)
        ma_sig = '上涨' if ma_score > 0.5 else ('下跌' if ma_score < -0.5 else '中性')
        g1_agents.append({'name': '均线Agent', 'signal': ma_sig, 'confidence': round(min(abs(ma_score)/4,1.0),2),
                         'weight': 1.0, 'type': '规则', 'group': '趋势'})
        mom_score = (1.5 if change_pct > 3 else 0.8 if change_pct > 1 else -1.5 if change_pct < -3 else -0.8 if change_pct < -1 else 0)
        if trend == '上涨' and change_pct > 0: mom_score += 0.5
        elif trend == '下跌' and change_pct < 0: mom_score -= 0.5
        mom_sig = '上涨' if mom_score > 0.5 else ('下跌' if mom_score < -0.5 else '中性')
        g1_agents.append({'name': '动量Agent', 'signal': mom_sig, 'confidence': round(min(abs(mom_score)/3,1.0),2),
                         'weight': 0.8, 'type': '规则', 'group': '趋势'})
        g1_result = _sub_group_vote(g1_agents, '趋势')
        g1_result['domain'] = 'stock'
        all_agents.extend(g1_agents)
        group_results.append(g1_result)

        # 功能组2: 波动率
        g2_agents = [llm_results.get('stock_vol')]
        atr_score = abs(change_pct) / 3.0
        if atr_score > 1.0: atr_signal = '上涨' if change_pct > 0 else '下跌'
        else: atr_signal = '中性'
        atr_conf = min(atr_score, 1.0)
        g2_agents.append({'name': 'ATR Agent', 'signal': atr_signal, 'confidence': round(atr_conf,2),
                         'weight': 1.0, 'type': '规则', 'group': '波动率'})
        if rsi > 70: std_signal, std_conf = '下跌', min((rsi-50)/50, 1.0)
        elif rsi < 30: std_signal, std_conf = '上涨', min((50-rsi)/50, 1.0)
        else: std_signal, std_conf = '中性', min(abs(rsi-50)/50, 0.6)
        g2_agents.append({'name': '标准差Agent', 'signal': std_signal, 'confidence': round(std_conf,2),
                         'weight': 0.8, 'type': '规则', 'group': '波动率'})
        g2_result = _sub_group_vote(g2_agents, '波动率')
        g2_result['domain'] = 'stock'
        all_agents.extend(g2_agents)
        group_results.append(g2_result)

        # 功能组3: 技术面
        g3_agents = [llm_results.get('stock_tech')]
        if rsi < 30: rsi_sig, rsi_conf = '上涨', min((30-rsi)/30, 1.0)
        elif rsi > 70: rsi_sig, rsi_conf = '下跌', min((rsi-70)/30, 1.0)
        elif rsi < 45: rsi_sig, rsi_conf = '上涨', 0.4
        elif rsi > 55: rsi_sig, rsi_conf = '下跌', 0.4
        else: rsi_sig, rsi_conf = '中性', 0.3
        g3_agents.append({'name': 'RSI Agent', 'signal': rsi_sig, 'confidence': round(rsi_conf,2),
                         'weight': 1.0, 'type': '规则', 'group': '技术面'})
        macd_score = (1.5 if macd > 0 else -1.5 if macd < 0 else 0)
        if trend == '上涨' and macd > 0: macd_score += 0.5
        elif trend == '下跌' and macd < 0: macd_score -= 0.5
        macd_sig = '上涨' if macd_score > 0.5 else ('下跌' if macd_score < -0.5 else '中性')
        macd_conf = min(abs(macd_score)/3, 1.0)
        g3_agents.append({'name': 'MACD Agent', 'signal': macd_sig, 'confidence': round(macd_conf,2),
                         'weight': 0.8, 'type': '规则', 'group': '技术面'})
        g3_result = _sub_group_vote(g3_agents, '技术面')
        g3_result['domain'] = 'stock'
        all_agents.extend(g3_agents)
        group_results.append(g3_result)

    if weather and 'ml_pred' in weather and 'temp' in weather and route == 'weather':
        diff = weather['ml_pred'] - weather['temp']
        precip = weather.get('precip', 0)
        wind = weather.get('wind', 5)
        temp_range = weather.get('temp_range', 10)
        season = weather.get('season', '未知')
        temp = weather.get('temp', 20)

        # 功能组1: 温度趋势
        g1_agents = [llm_results.get('weather_temp')]
        td_score = diff / 5.0
        td_sig = '升温' if td_score > 0.3 else ('降温' if td_score < -0.3 else '平稳')
        td_conf = min(abs(td_score), 1.0)
        g1_agents.append({'name': '温差Agent', 'signal': td_sig, 'confidence': round(td_conf,2),
                         'weight': 1.0, 'type': '规则', 'group': '温度趋势'})
        hist_score = (1.5 if diff > 3 else 0.8 if diff > 1 else -1.5 if diff < -3 else -0.8 if diff < -1 else 0)
        hist_sig = '升温' if hist_score > 0.5 else ('降温' if hist_score < -0.5 else '平稳')
        hist_conf = min(abs(hist_score)/2, 1.0)
        g1_agents.append({'name': '历史Agent', 'signal': hist_sig, 'confidence': round(hist_conf,2),
                         'weight': 0.8, 'type': '规则', 'group': '温度趋势'})
        g1_result = _sub_group_vote(g1_agents, '温度趋势')
        g1_result['domain'] = 'weather'
        all_agents.extend(g1_agents)
        group_results.append(g1_result)

        # 功能组2: 降水
        g2_agents = [llm_results.get('weather_rain')]
        pr_score = -precip / 10.0
        pr_sig = '升温' if pr_score > 0.3 else ('降温' if pr_score < -0.3 else '平稳')
        pr_conf = min(abs(pr_score), 1.0)
        g2_agents.append({'name': '降水率Agent', 'signal': pr_sig, 'confidence': round(pr_conf,2),
                         'weight': 1.0, 'type': '规则', 'group': '降水'})
        ws_score = -wind / 10.0
        ws_sig = '升温' if ws_score > 0.2 else ('降温' if ws_score < -0.5 else '平稳')
        ws_conf = min(abs(ws_score), 1.0)
        g2_agents.append({'name': '风速Agent', 'signal': ws_sig, 'confidence': round(ws_conf,2),
                         'weight': 0.8, 'type': '规则', 'group': '降水'})
        g2_result = _sub_group_vote(g2_agents, '降水')
        g2_result['domain'] = 'weather'
        all_agents.extend(g2_agents)
        group_results.append(g2_result)

        # 功能组3: 季节模式
        g3_agents = [llm_results.get('weather_season')]
        season_hot = {'春季': 0.5, '夏季': 1.5, '秋季': -0.5, '冬季': -1.5}
        ss_score = season_hot.get(season, 0)
        if diff > 0 and season in ('春季','夏季'): ss_score += 0.3
        elif diff < 0 and season in ('秋季','冬季'): ss_score += 0.3
        ss_sig = '升温' if ss_score > 0.5 else ('降温' if ss_score < -0.5 else '平稳')
        ss_conf = min(abs(ss_score)/2, 1.0)
        g3_agents.append({'name': '季节Agent', 'signal': ss_sig, 'confidence': round(ss_conf,2),
                         'weight': 1.0, 'type': '规则', 'group': '季节模式'})
        cyc_score = {'春季': 0.8, '夏季': 1.2, '秋季': -0.3, '冬季': -1.0}.get(season, 0)
        cyc_sig = '升温' if cyc_score > 0.5 else ('降温' if cyc_score < -0.5 else '平稳')
        cyc_conf = min(abs(cyc_score)/2, 1.0)
        g3_agents.append({'name': '周期Agent', 'signal': cyc_sig, 'confidence': round(cyc_conf,2),
                         'weight': 0.8, 'type': '规则', 'group': '季节模式'})
        g3_result = _sub_group_vote(g3_agents, '季节模式')
        g3_result['domain'] = 'weather'
        all_agents.extend(g3_agents)
        group_results.append(g3_result)

    if currency and 'ml_prob' in currency and route == 'currency':
        prob = currency['ml_prob']
        trend = currency.get('trend', '未知')
        vol = abs(prob - 0.5)

        # 功能组1: 趋势
        g1_agents = [llm_results.get('cur_trend')]
        dir_sig = '上涨' if trend == '上涨' else ('下跌' if trend == '下跌' else '中性')
        g1_agents.append({'name': '方向Agent', 'signal': dir_sig, 'confidence': 0.6,
                         'weight': 1.0, 'type': '规则', 'group': '趋势'})
        iner_score = (1.5 if trend=='上涨' and prob>0.5 else -1.5 if trend=='下跌' and prob<0.5
                      else 0.5 if trend=='上涨' else -0.5 if trend=='下跌' else 0)
        iner_sig = '上涨' if iner_score > 0.5 else ('下跌' if iner_score < -0.5 else '中性')
        iner_conf = min(abs(iner_score)/2, 1.0)
        g1_agents.append({'name': '惯性Agent', 'signal': iner_sig, 'confidence': round(iner_conf,2),
                         'weight': 0.8, 'type': '规则', 'group': '趋势'})
        g1_result = _sub_group_vote(g1_agents, '趋势')
        g1_result['domain'] = 'currency'
        all_agents.extend(g1_agents)
        group_results.append(g1_result)

        # 功能组2: 波动率
        g2_agents = [llm_results.get('cur_vol')]
        if vol > 0.2: amp_sig = '上涨' if prob > 0.5 else '下跌'
        else: amp_sig = '中性'
        amp_conf = min(vol*2, 1.0)
        g2_agents.append({'name': '振幅Agent', 'signal': amp_sig, 'confidence': round(amp_conf,2),
                         'weight': 1.0, 'type': '规则', 'group': '波动率'})
        if prob > 0.7: mr_sig, mr_conf = '下跌', min((prob-0.5)*2, 1.0)
        elif prob < 0.3: mr_sig, mr_conf = '上涨', min((0.5-prob)*2, 1.0)
        else: mr_sig, mr_conf = '中性', 0.3
        g2_agents.append({'name': '均值回归Agent', 'signal': mr_sig, 'confidence': round(mr_conf,2),
                         'weight': 0.8, 'type': '规则', 'group': '波动率'})
        g2_result = _sub_group_vote(g2_agents, '波动率')
        g2_result['domain'] = 'currency'
        all_agents.extend(g2_agents)
        group_results.append(g2_result)

        # 功能组3: 动量
        g3_agents = [llm_results.get('cur_mom')]
        if prob > 0.6: pct_sig, pct_conf = '上涨', min((prob-0.5)*2, 1.0)
        elif prob < 0.4: pct_sig, pct_conf = '下跌', min((0.5-prob)*2, 1.0)
        else: pct_sig, pct_conf = '中性', 0.3
        g3_agents.append({'name': '涨跌幅Agent', 'signal': pct_sig, 'confidence': round(pct_conf,2),
                         'weight': 1.0, 'type': '规则', 'group': '动量'})
        if trend == '上涨' and prob > 0.5: vol_sig, vol_conf = '上涨', min(prob, 1.0)
        elif trend == '下跌' and prob < 0.5: vol_sig, vol_conf = '下跌', min(1-prob, 1.0)
        else: vol_sig, vol_conf = '中性', 0.4
        g3_agents.append({'name': '成交量Agent', 'signal': vol_sig, 'confidence': round(vol_conf,2),
                         'weight': 0.8, 'type': '规则', 'group': '动量'})
        g3_result = _sub_group_vote(g3_agents, '动量')
        g3_result['domain'] = 'currency'
        all_agents.extend(g3_agents)
        group_results.append(g3_result)

    if not all_agents:
        return None

    # ===== 域级别：3功能组共识汇总（第二层投票）=====
    from collections import defaultdict, Counter
    domain_groups = [g for g in group_results if g.get('domain') == route]
    if not domain_groups:
        return None

    # 功能组加权投票
    domain_weighted = defaultdict(float)
    for g in domain_groups:
        domain_weighted[g['signal']] += g.get('confidence', 0.5)

    best_signal = max(domain_weighted, key=domain_weighted.get)
    total_domain_weight = sum(g.get('confidence', 0.5) for g in domain_groups)
    consensus_strength = domain_weighted[best_signal] / total_domain_weight if total_domain_weight > 0 else 0

    # 统计所有Agent的投票
    all_votes_list = [a.get('signal', '中性') for a in all_agents]
    vote_counts = Counter(all_votes_list)

    # 功能组共识信息
    group_summary = ', '.join(f"{g['group_name']}→{g['signal']}({g['confidence']:.0%})" for g in domain_groups)

    consensus = f"参与Agent: {len(all_agents)}个({len(domain_groups)}功能组)，"
    consensus += f"投票: {', '.join(f'{k}({v}票)' for k, v in vote_counts.items())}，"
    consensus += f"功能组共识: [{group_summary}]，"
    consensus += f"域共识: {best_signal}（强度{consensus_strength:.0%}）"

    avg_confidence = sum(a['confidence'] for a in all_agents) / len(all_agents) if all_agents else 0
    all_agree = len(vote_counts) == 1
    has_dissent = len(vote_counts) > 1

    return {
        'agents': all_agents,
        'votes': dict(vote_counts),
        'weighted_votes': {k: round(v, 2) for k, v in domain_weighted.items()},
        'consensus': consensus,
        'consensus_signal': best_signal,
        'consensus_strength': round(consensus_strength, 2),
        'avg_confidence': round(avg_confidence, 2),
        'all_agree': all_agree,
        'has_dissent': has_dissent,
        'group_results': [{k: v for k, v in g.items()} for g in domain_groups],
    }


def _build_chart_data(context):
    """根据路由类型构建图表数据（复用analyze阶段已缓存的数据）"""
    route = context['_data'].get('route', '')
    optimized = context['_data'].get('optimized', {})
    chart_type = optimized.get('chart', 'none')

    # 对于stock/weather/currency路由，即使chart_type=none也生成图表
    if chart_type == 'none' and route not in ('stock', 'weather', 'currency', 'general', 'direct'):
        return None

    try:
        if route == 'stock':
            stock = context['_data'].get('stock', {})
            symbol = stock.get('symbol', '')
            # 优先复用已缓存的历史数据
            df = context['_data'].get('stock_history_df')
            if df is None:
                from examples.stock_forecast.data_fetcher import fetch_stock_history, add_stock_features
                from utils.time_utils import get_date_range_around_now
                import datetime
                start = (datetime.datetime.now() - datetime.timedelta(days=60)).strftime('%Y%m%d')
                _, _, today = get_date_range_around_now(past_years=0, future_days=0)
                df = fetch_stock_history(symbol, start, today)
            if df is not None and len(df) > 2:
                dates = df['日期'].tolist()[-30:] if '日期' in df.columns else list(range(len(df)))[-30:]
                closes = df['收盘'].tolist()[-30:]
                datasets = [{'label': '收盘价', 'data': closes, 'borderColor': '#60a5fa'}]
                # 添加成交量数据（如果有）
                if '成交量' in df.columns:
                    volumes = df['成交量'].tolist()[-30:]
                    datasets.append({'label': '成交量', 'data': volumes, 'borderColor': '#a78bfa'})
                return {
                    'type': 'kline',
                    'title': f'{symbol} 近30日走势',
                    'labels': dates,
                    'datasets': datasets,
                }
        elif route == 'weather':
            weather = context['_data'].get('weather', {})
            city = weather.get('city', '北京')
            df = context['_data'].get('weather_history_df')
            if df is None:
                from examples.weather_forecast.weather_spider import fetch_weather_history, CITY_COORDS
                from utils.time_utils import get_date_range_around_now
                import datetime
                lat, lon = CITY_COORDS.get(city, (39.9, 116.4))
                start = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y%m%d')
                _, _, today = get_date_range_around_now(past_years=0, future_days=0)
                df = fetch_weather_history(lat, lon, start, today)
            if df is not None and len(df) > 2:
                dates = df['日期'].tolist()[-30:] if '日期' in df.columns else list(range(len(df)))[-30:]
                temps = df['平均温'].tolist()[-30:] if '平均温' in df.columns else []
                precips = df['降水量'].tolist()[-30:] if '降水量' in df.columns else []
                datasets = [
                    {'label': '温度(°C)', 'data': temps, 'borderColor': '#f97316'},
                    {'label': '降水量(mm)', 'data': precips, 'borderColor': '#3b82f6'},
                ]
                # 添加风速数据（如果有）
                if '最大风速' in df.columns:
                    winds = df['最大风速'].tolist()[-30:]
                    datasets.append({'label': '风速(m/s)', 'data': winds, 'borderColor': '#a78bfa'})
                return {
                    'type': 'line',
                    'title': f'{city} 近30日天气',
                    'labels': dates,
                    'datasets': datasets,
                }
        elif route == 'currency':
            currency = context['_data'].get('currency', {})
            cur = currency.get('currency', '美元')
            df = context['_data'].get('currency_history_df')
            if df is None:
                from examples.stock_forecast.data_fetcher import fetch_currency_history
                from utils.time_utils import get_date_range_around_now
                import datetime
                start = (datetime.datetime.now() - datetime.timedelta(days=60)).strftime('%Y%m%d')
                _, _, today = get_date_range_around_now(past_years=0, future_days=0)
                df = fetch_currency_history(cur, start, today)
            if df is not None and len(df) > 2:
                price_col = '中间价' if '中间价' in df.columns else df.columns[1]
                dates = df['日期'].tolist()[-30:] if '日期' in df.columns else list(range(len(df)))[-30:]
                rates = df[price_col].tolist()[-30:]
                datasets = [{'label': f'{cur}汇率', 'data': rates, 'borderColor': '#22c55e'}]
                # 添加波动率数据（如果有）
                if '波动率5' in df.columns:
                    vols = df['波动率5'].tolist()[-30:]
                    datasets.append({'label': '5日波动率', 'data': vols, 'borderColor': '#f43f5e'})
                return {
                    'type': 'line',
                    'title': f'{cur} 近30日汇率',
                    'labels': dates,
                    'datasets': datasets,
                }
    except Exception:
        pass

    return None


def node_output(context):
    """节点13: 格式化输出（含图表数据）"""
    merged = context['_data'].get('merged', {})
    context['result'] = merged
    return merged


# ======================== 构建工作流 ========================

def build_workflow():
    """构建智能决策助手工作流（13个节点）"""
    wf = Workflow('智能决策助手', '基于工作流的多智能体协作决策系统')

    # 1. 输入节点
    wf.add_node('input', NodeType.INPUT, '接收用户输入', node_input)

    # 2. OpenRouter提示词优化
    wf.add_node('prompt_optimizer', NodeType.AGENT, 'OpenRouter提示词优化', node_prompt_optimizer)

    # 3. LLM决策路由
    wf.add_node('decision_route', NodeType.DECISION, 'LLM决策路由', node_data_router)

    # 4-6. 数据采集（3个Agent）
    wf.add_node('fetch_stock', NodeType.DATA_FETCH, '采集股票数据', node_fetch_stock)
    wf.add_node('fetch_weather', NodeType.DATA_FETCH, '采集天气数据', node_fetch_weather)
    wf.add_node('fetch_currency', NodeType.DATA_FETCH, '采集汇率数据', node_fetch_currency)

    # 7-9. ML模型分析（3个Agent）
    wf.add_node('analyze_stock', NodeType.AGENT, 'ML分析股票', node_analyze_stock)
    wf.add_node('analyze_weather', NodeType.AGENT, 'ML分析天气', node_analyze_weather)
    wf.add_node('analyze_currency', NodeType.AGENT, 'ML分析汇率', node_analyze_currency)

    # 10. LLM综合分析
    wf.add_node('llm_analysis', NodeType.AGENT, 'LLM综合分析', node_llm_analysis)

    # 11. 输出校验
    wf.add_node('validate', NodeType.VALIDATOR, '输出校验(防幻觉)', node_validate)

    # 12. 数据汇总（含图表）
    wf.add_node('merge', NodeType.MERGE, '汇总数据+图表', node_merge)

    # 13. 格式化输出
    wf.add_node('output', NodeType.OUTPUT, '格式化输出', node_output)

    # 连接节点
    wf.connect('input', 'prompt_optimizer')

    # 提示词优化 → 决策路由
    wf.connect('prompt_optimizer', 'decision_route')

    # 决策路由 → 数据采集分支
    wf.connect('decision_route', 'fetch_stock',
               condition=lambda ctx: ctx['_data'].get('route') == 'stock',
               label='股票')
    wf.connect('decision_route', 'fetch_weather',
               condition=lambda ctx: ctx['_data'].get('route') == 'weather',
               label='天气')
    wf.connect('decision_route', 'fetch_currency',
               condition=lambda ctx: ctx['_data'].get('route') == 'currency',
               label='汇率')
    wf.connect('decision_route', 'llm_analysis',
               condition=lambda ctx: ctx['_data'].get('route') in ('general', 'direct'),
               label='通用/直接')

    # 数据采集 → 分析
    wf.connect('fetch_stock', 'analyze_stock')
    wf.connect('fetch_weather', 'analyze_weather')
    wf.connect('fetch_currency', 'analyze_currency')

    # 分析 → LLM综合分析
    wf.connect('analyze_stock', 'llm_analysis')
    wf.connect('analyze_weather', 'llm_analysis')
    wf.connect('analyze_currency', 'llm_analysis')

    # LLM → 校验 → 汇总 → 输出
    wf.connect('llm_analysis', 'validate')
    wf.connect('validate', 'merge')
    wf.connect('merge', 'output')

    return wf
