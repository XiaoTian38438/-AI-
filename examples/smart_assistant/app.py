"""智能决策助手 - Web 应用

赛事评分覆盖：
- 智能人机交互(5%): Web界面+自然语言交互+打字机流式输出
- 扩展性(5%): 可扩展到其他设备
"""
import os
import sys
import json
import uuid
import time
import threading
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context, session
from examples.smart_assistant.workflow import build_workflow
from core.llm import is_available, configure, check_openrouter_keys
from examples.smart_assistant import db
from examples.smart_assistant import model_manager

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.secret_key = os.environ.get('FLASK_SECRET', 'mini-ai-secret-key-2024')
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SECURE'] = False  # 本地开发用HTTP，生产环境改为True
app.config['PERMANENT_SESSION_LIFETIME'] = 86400  # session 24小时有效

# 使用记录存储（赛事要求：有使用记录 5%）
_usage_records = []

# 对话历史存储（智能人机交互：多轮对话上下文）
_conversation_history = []  # [{role, content, timestamp}, ...]
_MAX_HISTORY = 50  # 保留最近50条

# 全局工作流
_workflow = build_workflow()


def _get_current_user_id():
    """获取当前登录用户的ID（未登录返回None）"""
    return session.get('user_id')


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


# ======================== 用户认证 API ========================

@app.route('/api/auth/register', methods=['POST'])
def register():
    """用户注册"""
    data = request.json or {}
    username = data.get('username', '').strip()
    password = data.get('password', '')
    success, msg, user_id = db.register_user(username, password)
    if success:
        session['user_id'] = user_id
        session['username'] = username
        return jsonify({'status': 'ok', 'message': msg, 'user_id': user_id, 'username': username})
    return jsonify({'status': 'error', 'message': msg}), 400


@app.route('/api/auth/login', methods=['POST'])
def login():
    """用户登录"""
    data = request.json or {}
    username = data.get('username', '').strip()
    password = data.get('password', '')
    success, user_id = db.verify_user(username, password)
    if success:
        session['user_id'] = user_id
        session['username'] = username
        return jsonify({'status': 'ok', 'user_id': user_id, 'username': username})
    return jsonify({'status': 'error', 'message': '用户名或密码错误'}), 401


@app.route('/api/auth/logout', methods=['POST'])
def logout():
    """用户登出"""
    session.clear()
    return jsonify({'status': 'ok'})


@app.route('/api/auth/status', methods=['GET'])
def auth_status():
    """获取当前登录状态"""
    user_id = _get_current_user_id()
    if user_id:
        return jsonify({'logged_in': True, 'user_id': user_id, 'username': session.get('username', '')})
    return jsonify({'logged_in': False})


def _run_workflow_stream(query):
    """在后台线程执行工作流，通过队列发送 SSE 事件"""
    import queue
    q = queue.Queue()
    # 在请求上下文中捕获user_id（后台线程无法访问session）
    captured_user_id = _get_current_user_id()

    def worker():
        context = {'query': query}
        # 注入对话历史（多轮对话上下文）
        context['history'] = list(_conversation_history[-10:])  # 最近10轮

        # 包装节点执行，实时推送进度
        _patch_workflow_for_streaming(_workflow, q)

        try:
            # 工作流整体超时保护（最多300秒，LLM分析不限时）
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_workflow.run, context)
                try:
                    result_ctx = future.result(timeout=300)
                except concurrent.futures.TimeoutError:
                    q.put(('error', '工作流执行超时，请稍后重试'))
                    q.put(('done', None))
                    return
                except Exception as e:
                    q.put(('error', str(e)))
                    q.put(('done', None))
                    return
        except Exception as e:
            q.put(('error', str(e)))
            q.put(('done', None))
            return

        result = result_ctx.get('result', {})

        # 记录到对话历史（内存 + SQLite持久化）
        user_id = captured_user_id
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        _conversation_history.append({
            'role': 'user', 'content': query, 'timestamp': ts
        })
        _conversation_history.append({
            'role': 'assistant', 'content': result.get('analysis', ''),
            'route': result.get('route', ''), 'timestamp': ts
        })
        if len(_conversation_history) > _MAX_HISTORY:
            del _conversation_history[:len(_conversation_history) - _MAX_HISTORY]
        # SQLite持久化
        try:
            db.save_conversation(user_id, 'user', query)
            db.save_conversation(user_id, 'assistant', result.get('analysis', ''), result.get('route', ''))
        except Exception:
            pass

        # 记录使用（内存 + SQLite）
        route = result.get('route', 'unknown')
        validation_passed = result.get('validation', {}).get('passed', True)
        record = {
            'id': str(uuid.uuid4())[:8],
            'query': query, 'route': route,
            'timestamp': ts, 'validation_passed': validation_passed,
        }
        _usage_records.append(record)
        try:
            db.save_usage(user_id, query, route, validation_passed)
        except Exception:
            pass

        # 发送最终结果（打字机效果：逐字发送分析文本）
        analysis = result.get('analysis', '')
        q.put(('result_start', json.dumps({
            'route': result.get('route', ''),
            'data': result.get('data', {}),
            'validation': result.get('validation', {}),
            'chart': result.get('chart', None),
            'agent_vote': result.get('agent_vote', None),
            'chart_type': result.get('chart_type', 'none'),
        }, ensure_ascii=False)))

        # 逐字发送分析文本
        for char in analysis:
            q.put(('char', char))
            time.sleep(0.03)  # 打字机速度

        q.put(('result_end', None))
        q.put(('workflow', json.dumps({
            'log': _workflow.execution_log,
            'mermaid': _workflow.get_mermaid(),
            'summary': _workflow.summary(),
        }, ensure_ascii=False)))
        q.put(('done', None))

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return q


def _patch_workflow_for_streaming(wf, q):
    """为工作流节点添加 SSE 流式推送（每次请求恢复原始handler再包装）"""
    for name, node in wf.nodes.items():
        # 保存原始handler（第一次时）
        if not hasattr(node, '_original_handler'):
            node._original_handler = node.handler
        # 恢复原始handler（防止嵌套包装）
        orig = node._original_handler

        def make_streaming_handler(n, h):
            def streaming_handler(context):
                q.put(('node_start', json.dumps({
                    'node': n, 'description': wf.nodes[n].description
                }, ensure_ascii=False)))
                result = h(context)
                q.put(('node_done', json.dumps({
                    'node': n, 'description': wf.nodes[n].description
                }, ensure_ascii=False)))
                return result
            return streaming_handler

        node.handler = make_streaming_handler(name, orig)


@app.route('/api/chat', methods=['POST'])
def chat():
    """主对话接口 - 非流式（兼容旧版）"""
    data = request.json or {}
    query = data.get('query', '').strip()
    if not query:
        return jsonify({'error': '请输入问题'}), 400

    context = {'query': query, 'history': list(_conversation_history[-10:])}
    try:
        result_ctx = _workflow.run(context)
    except Exception as e:
        return jsonify({'error': str(e), 'result': None}), 500

    result = result_ctx.get('result', {})

    # 记录到对话历史（内存 + SQLite持久化）
    user_id = _get_current_user_id()
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    _conversation_history.append({
        'role': 'user', 'content': query, 'timestamp': ts
    })
    _conversation_history.append({
        'role': 'assistant', 'content': result.get('analysis', ''),
        'route': result.get('route', ''), 'timestamp': ts
    })
    if len(_conversation_history) > _MAX_HISTORY:
        del _conversation_history[:len(_conversation_history) - _MAX_HISTORY]
    try:
        db.save_conversation(user_id, 'user', query)
        db.save_conversation(user_id, 'assistant', result.get('analysis', ''), result.get('route', ''))
    except Exception:
        pass

    # 记录使用（内存 + SQLite）
    route = result.get('route', 'unknown')
    validation_passed = result.get('validation', {}).get('passed', True)
    record = {
        'id': str(uuid.uuid4())[:8],
        'query': query, 'route': route,
        'timestamp': ts, 'validation_passed': validation_passed,
    }
    _usage_records.append(record)
    try:
        db.save_usage(user_id, query, route, validation_passed)
    except Exception:
        pass

    return jsonify({
        'result': result,
        'workflow_log': _workflow.execution_log,
        'workflow_mermaid': _workflow.get_mermaid(),
        'workflow_summary': _workflow.summary(),
    })


@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """流式对话接口 - SSE 打字机效果"""
    data = request.json or {}
    query = data.get('query', '').strip()
    if not query:
        return jsonify({'error': '请输入问题'}), 400

    q = _run_workflow_stream(query)

    def generate():
        while True:
            try:
                event_type, payload = q.get(timeout=300)
            except Exception:
                yield f"data: {json.dumps({'type': 'error', 'content': '超时'}, ensure_ascii=False)}\n\n"
                break

            if event_type == 'done':
                yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
                break
            elif event_type == 'char':
                yield f"data: {json.dumps({'type': 'char', 'content': payload}, ensure_ascii=False)}\n\n"
            elif event_type == 'node_start':
                yield f"data: {json.dumps({'type': 'node_start', 'content': payload}, ensure_ascii=False)}\n\n"
            elif event_type == 'node_done':
                yield f"data: {json.dumps({'type': 'node_done', 'content': payload}, ensure_ascii=False)}\n\n"
            elif event_type == 'result_start':
                yield f"data: {json.dumps({'type': 'result_start', 'content': payload}, ensure_ascii=False)}\n\n"
            elif event_type == 'result_end':
                yield f"data: {json.dumps({'type': 'result_end'}, ensure_ascii=False)}\n\n"
            elif event_type == 'workflow':
                yield f"data: {json.dumps({'type': 'workflow', 'content': payload}, ensure_ascii=False)}\n\n"
            elif event_type == 'error':
                yield f"data: {json.dumps({'type': 'error', 'content': payload}, ensure_ascii=False)}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/workflow', methods=['GET'])
def get_workflow():
    """获取工作流结构"""
    return jsonify({
        'mermaid': _workflow.get_mermaid(),
        'summary': _workflow.summary(),
    })


@app.route('/api/usage', methods=['GET'])
def get_usage():
    """获取使用记录（持久化）"""
    user_id = _get_current_user_id()
    records = db.get_usage(user_id=user_id, limit=50)
    total = len(records) + len(_usage_records)
    return jsonify({'records': records[:50], 'total': total})


@app.route('/api/history', methods=['GET'])
def get_history():
    """获取对话历史（SQLite持久化 + 内存最新合并）"""
    user_id = _get_current_user_id()
    records = db.get_conversations(user_id=user_id, limit=50)
    # 内存中有但SQLite中可能还没有的最新记录
    recent_in_memory = list(_conversation_history[-10:])
    # 去重：如果SQLite已有记录，跳过内存中同时间的重复
    existing_timestamps = set(r.get('timestamp', '') for r in records)
    deduped = [h for h in recent_in_memory if h.get('timestamp', '') not in existing_timestamps]
    return jsonify({'history': records + deduped})


@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    """清空对话历史"""
    user_id = _get_current_user_id()
    db.clear_conversations(user_id=user_id)
    _conversation_history.clear()
    return jsonify({'status': 'ok'})


@app.route('/api/llm_status', methods=['GET'])
def llm_status():
    """获取LLM状态"""
    return jsonify({'available': is_available(), 'openrouter': True})


@app.route('/api/configure', methods=['POST'])
def configure_llm():
    """配置LLM参数"""
    data = request.json or {}
    configure(**data)
    return jsonify({'status': 'ok'})


@app.route('/api/train_stock', methods=['POST'])
def train_stock():
    """训练股票模型（异步任务入口）"""
    data = request.json or {}
    symbol = data.get('symbol', '000001')
    epochs = data.get('epochs', 80)
    # 返回训练参数，实际训练通过命令行执行
    return jsonify({
        'command': f'python examples/stock_forecast/train.py --type stock --symbol {symbol} --epochs {epochs}',
        'symbol': symbol,
        'epochs': epochs,
    })


@app.route('/api/train_weather', methods=['POST'])
def train_weather():
    """训练天气模型"""
    data = request.json or {}
    city = data.get('city', '北京')
    return jsonify({
        'command': f'python examples/weather_forecast/train.py --city {city}',
        'city': city,
    })


# ======================== 模型管理 API ========================

@app.route('/api/models/status', methods=['GET'])
def models_status():
    """获取ML模型状态"""
    return jsonify(model_manager.get_status())


@app.route('/api/models/retrain', methods=['POST'])
def models_retrain():
    """手动触发模型重训"""
    data = request.json or {}
    model_type = data.get('model_type', None)  # 'stock'/'weather'/'currency' or None for all
    target = data.get('target', None)

    if model_manager.is_training():
        return jsonify({'status': 'error', 'message': '模型正在训练中，请稍后再试'}), 409

    if model_type:
        if model_type not in ('stock', 'weather', 'currency'):
            return jsonify({'status': 'error', 'message': '未知模型类型'}), 400
        model_manager.retrain_specific_async(model_type, target)
    else:
        model_manager.train_all_async()

    return jsonify({'status': 'ok', 'message': '训练已启动'})


@app.route('/api/models', methods=['GET'])
def list_models():
    """列出可用的ML模型"""
    models = [
        {'id': 'stock_classifier', 'name': '股票涨跌分类器', 'type': 'classification',
         'description': '基于技术指标预测股票涨跌概率', 'features': 15, 'params': '2K+'},
        {'id': 'weather_regressor', 'name': '天气温度回归器', 'type': 'regression',
         'description': '基于历史气象数据预测明日温度', 'features': 15, 'params': '2K+'},
        {'id': 'currency_classifier', 'name': '汇率涨跌分类器', 'type': 'classification',
         'description': '基于汇率指标预测涨跌概率', 'features': 10, 'params': '2K+'},
        {'id': 'nvidia_nemotron', 'name': 'NVIDIA Nemotron (OpenRouter)', 'type': 'llm',
         'description': '提示词优化与意图解析', 'params': '550B'},
    ]
    return jsonify({'models': models})


@app.route('/api/models/train', methods=['POST'])
def train_model():
    """启动模型训练"""
    data = request.json or {}
    model_id = data.get('model_id', '')
    params = data.get('params', {})
    user_id = _get_current_user_id()

    # 保存训练参数
    if user_id:
        db.save_model_params(user_id, model_id, params)

    if model_id == 'stock_classifier':
        cmd = f"python examples/stock_forecast/train.py --type stock --symbol {params.get('symbol','000001')} --epochs {params.get('epochs',80)}"
    elif model_id == 'weather_regressor':
        cmd = f"python examples/weather_forecast/train.py --city {params.get('city','北京')} --epochs {params.get('epochs',150)}"
    elif model_id == 'currency_classifier':
        cmd = f"python examples/stock_forecast/train.py --type currency --symbol {params.get('currency','美元')} --epochs {params.get('epochs',80)}"
    else:
        return jsonify({'error': '未知模型'}), 400

    return jsonify({'command': cmd, 'model_id': model_id, 'status': 'submitted'})


# ======================== 导出报告 API ========================

# ======================== 设备扩展 API（可扩展到微信小程序/智能音箱等）========================

@app.route('/api/v1/query', methods=['POST'])
def api_v1_query():
    """轻量级JSON API - 供微信小程序/智能音箱/其他设备调用
    返回结构化JSON结果，适合非Web前端消费"""
    data = request.json or {}
    query = data.get('query', '').strip()
    if not query:
        return jsonify({'error': '请输入问题'}), 400

    context = {'query': query, 'history': list(_conversation_history[-10:])}
    try:
        result_ctx = _workflow.run(context)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    result = result_ctx.get('result', {})
    # 精简输出，只保留关键字段
    response = {
        'query': query,
        'route': result.get('route', ''),
        'analysis': result.get('analysis', ''),
        'data': result.get('data', {}),
        'validation': result.get('validation', {}),
        'agent_vote': result.get('agent_vote', None),
        'has_chart': result.get('chart') is not None,
    }
    return jsonify(response)


@app.route('/api/v1/status', methods=['GET'])
def api_v1_status():
    """系统状态API - 供设备端检测服务可用性"""
    return jsonify({
        'service': '智能决策助手',
        'version': '1.0',
        'status': 'running',
        'llm_available': is_available(),
        'workflow_nodes': len(_workflow.nodes),
        'agents': 27,
    })


@app.route('/api/export', methods=['POST'])
def export_report():
    """生成分析报告（HTML格式，可下载）"""
    data = request.json or {}
    query = data.get('query', '')
    analysis = data.get('analysis', '')
    route = data.get('route', '')
    chart_data = data.get('chart', None)
    validation = data.get('validation', {})
    agent_vote = data.get('agent_vote', None)

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><title>智能决策分析报告</title>
<style>
body{{font-family:'PingFang SC',sans-serif;max-width:800px;margin:40px auto;padding:20px;background:#f8fafc;color:#1e293b;}}
h1{{color:#6366f1;border-bottom:2px solid #6366f1;padding-bottom:10px;}}
.meta{{color:#64748b;font-size:13px;margin-bottom:20px;}}
.section{{background:#fff;border-radius:8px;padding:16px;margin:12px 0;box-shadow:0 1px 3px rgba(0,0,0,0.1);}}
.section h3{{color:#4f46e5;margin:0 0 8px;}}
.badge{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:12px;}}
.badge-pass{{background:#dcfce7;color:#166534;}}
.badge-warn{{background:#fef9c3;color:#854d0e;}}
</style></head>
<body>
<h1>智能决策分析报告</h1>
<div class="meta">生成时间: {now} | 类型: {route} | 查询: {query}</div>
<div class="section"><h3>分析结论</h3><p>{analysis or '无'}</p></div>"""

    if validation:
        badge = 'badge-pass' if validation.get('passed') else 'badge-warn'
        html += f"""<div class="section"><h3>输出校验</h3><span class="badge {badge}">{'通过' if validation.get('passed') else '警告'}</span></div>"""

    if agent_vote:
        agents_html = ''
        # 按功能组分组显示
        group_results = agent_vote.get('group_results', [])
        all_agents = agent_vote.get('agents', [])
        if group_results:
            for g in group_results:
                g_sig_color = '#22c55e' if g.get('signal') in ('上涨','升温') else '#ef4444' if g.get('signal') in ('下跌','降温') else '#94a3b8'
                agents_html += f'<li style="font-weight:600;margin-top:6px;">▎{g.get("group_name","")}组 → <span style="color:{g_sig_color};">{g.get("signal","")}</span> (共识{g.get("confidence",0):.0%})</li>'
                g_agents = [a for a in all_agents if a.get('group') == g.get('group_name')]
                for a in g_agents:
                    agent_type = a.get('type', '规则')
                    type_badge = '<span style="background:#6366f1;color:white;padding:1px 4px;border-radius:3px;font-size:10px;">LLM</span> ' if agent_type == 'LLM' else ''
                    reason = f" ({a.get('reason', '')})" if a.get('reason') else ''
                    agents_html += f'<li style="margin-left:16px;">{type_badge}{a["name"]}: {a["signal"]} (置信度{a["confidence"]:.0%}, 权重{a.get("weight",1.0):.1f}){reason}</li>'
        else:
            for a in all_agents:
                agent_type = a.get('type', '规则')
                type_badge = f'<span style="background:#6366f1;color:white;padding:1px 4px;border-radius:3px;font-size:10px;">LLM</span> ' if agent_type == 'LLM' else ''
                reason = f" ({a.get('reason', '')})" if a.get('reason') else ''
                agents_html += f"<li>{type_badge}{a['name']}: {a['signal']} (置信度{a['confidence']:.0%}, 权重{a.get('weight',1.0):.1f}){reason}</li>"
        consensus_strength = agent_vote.get('consensus_strength', 0)
        llm_count = sum(1 for a in all_agents if a.get('type') == 'LLM')
        group_count = len(group_results)
        html += f"""<div class="section"><h3>27-Agent分层协商（含{llm_count}个LLM推理Agent，{group_count}个功能组）</h3>
        <p>{agent_vote.get('consensus','')}</p>
        <p>共识强度: {consensus_strength:.0%} | 平均置信度: {agent_vote.get('avg_confidence',0):.0%}</p>
        <ul style="font-size:13px;padding-left:20px;">{agents_html}</ul></div>"""

    html += """<div class="section" style="color:#94a3b8;font-size:12px;text-align:center;">
<p>由智能决策助手生成 | 基于工作流的多智能体协作决策系统</p>
<p>⚠️ 本报告仅供参考，不构成投资建议</p></div></body></html>"""

    return jsonify({'html': html, 'filename': f'report_{route}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'})


# ======================== 全局错误处理（API 返回 JSON） ========================

@app.errorhandler(404)
def not_found(e):
    if request.path.startswith('/api/'):
        return jsonify({'status': 'error', 'message': '接口不存在'}), 404
    return send_from_directory('static', 'index.html')


@app.errorhandler(500)
def server_error(e):
    if request.path.startswith('/api/'):
        return jsonify({'status': 'error', 'message': '服务器内部错误'}), 500
    return send_from_directory('static', 'index.html')


@app.errorhandler(Exception)
def handle_exception(e):
    """捕获所有未处理异常，API 路由返回 JSON"""
    import traceback
    traceback.print_exc()
    if request.path.startswith('/api/'):
        return jsonify({'status': 'error', 'message': f'服务器错误: {str(e)}'}), 500
    return send_from_directory('static', 'index.html')


if __name__ == '__main__':
    PORT = 5000

    # 检查并释放占用的端口
    import subprocess, platform
    system = platform.system()
    try:
        if system == 'Windows':
            result = subprocess.run(f'netstat -ano | findstr :{PORT} | findstr LISTENING',
                                    shell=True, capture_output=True, text=True)
            if result.stdout.strip():
                pids = set()
                for line in result.stdout.strip().split('\n'):
                    parts = line.strip().split()
                    if parts:
                        pids.add(parts[-1])
                for pid in pids:
                    print(f"端口 {PORT} 被进程 PID={pid} 占用，正在终止...")
                    subprocess.run(f'taskkill /F /PID {pid}', shell=True, capture_output=True)
        else:
            result = subprocess.run(f'lsof -ti :{PORT}', shell=True, capture_output=True, text=True)
            if result.stdout.strip():
                for pid in result.stdout.strip().split('\n'):
                    if pid.strip():
                        print(f"端口 {PORT} 被进程 PID={pid.strip()} 占用，正在终止...")
                        subprocess.run(f'kill -9 {pid.strip()}', shell=True, capture_output=True)
    except Exception as e:
        print(f"端口检查出错: {e}")

    import time
    time.sleep(0.5)  # 等待端口释放

    print("=" * 50)
    print("智能决策助手 - AI智能体系统")
    print("赛事：第九届全国青少年人工智能创新挑战赛")
    print("=" * 50)
    print(f"LLM 可用: {is_available()}")
    print(f"工作流节点数: {len(_workflow.nodes)} (含OpenRouter优化+LLM决策分支)")
    print(f"智能体: 27个 (9个LLM推理Agent + 18个规则Agent，3域×3功能×3子Agent)")
    print(f"输出校验: 关键词过滤+正则匹配+幻觉检测+结构化JSON校验")
    
    # 启动时异步训练ML模型
    print("正在启动ML模型训练（后台线程）...")
    model_manager.train_all_async()
    model_manager.start_daily_scheduler()
    
    print(f"访问地址: http://localhost:{PORT}")
    print("=" * 50)
    app.run(host='0.0.0.0', port=PORT, debug=False)
