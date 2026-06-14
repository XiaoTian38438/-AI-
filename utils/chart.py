"""HTML 图表生成工具 - 使用 Chart.js 生成自包含的交互式图表"""
import os
import json


def _escape_js(data):
    """安全转义 Python 数据为 JS 字面量"""
    return json.dumps(data, ensure_ascii=False, default=float)


def save_chart(filename, title, chart_configs, width='100%', height=500):
    """
    生成自包含的 HTML 图表文件
    filename: 输出路径
    title: 页面标题
    chart_configs: list of dict, 每个包含:
        - 'label': 图表标题
        - 'type': 'line' | 'bar'
        - 'labels': x 轴标签列表
        - 'datasets': list of {'name': str, 'data': list, 'color': str, 'fill': bool}
    """
    charts_html = ""
    for i, cfg in enumerate(chart_configs):
        datasets_js = ""
        for ds in cfg['datasets']:
            fill = 'true' if ds.get('fill') else 'false'
            color = ds.get('color', '#e94560')
            dash = '[]' if not ds.get('dashed') else '[8, 4]'
            datasets_js += f"""{{
                label: {json.dumps(ds['name'])},
                data: {_escape_js(ds['data'])},
                borderColor: '{color}',
                backgroundColor: '{color}22',
                fill: {fill},
                tension: 0.3,
                pointRadius: {1 if len(ds['data']) > 50 else 3},
                borderWidth: 2,
                borderDash: {dash},
                spanGaps: false,
            }},"""

        charts_html += f"""
        <div class="chart-box">
            <h3>{cfg['label']}</h3>
            <canvas id="chart_{i}"></canvas>
        </div>
        <script>
        new Chart(document.getElementById('chart_{i}'), {{
            type: '{cfg.get('type', 'line')}',
            data: {{
                labels: {_escape_js(cfg['labels'])},
                datasets: [{datasets_js}]
            }},
            options: {{
                responsive: true,
                interaction: {{ mode: 'index', intersect: false }},
                plugins: {{ legend: {{ position: 'top' }} }},
                scales: {{
                    x: {{ grid: {{ color: 'rgba(255,255,255,0.05)' }} }},
                    y: {{ grid: {{ color: 'rgba(255,255,255,0.05)' }} }}
                }}
            }}
        }});
        </script>
        """

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    min-height: 100vh; padding: 30px; color: #e0e0e0;
}}
h1 {{ text-align: center; margin-bottom: 30px; font-weight: 300; letter-spacing: 2px; }}
.chart-box {{
    background: rgba(255,255,255,0.05); backdrop-filter: blur(10px);
    border-radius: 16px; padding: 24px; margin-bottom: 24px;
    border: 1px solid rgba(255,255,255,0.1);
    max-width: 960px; margin-left: auto; margin-right: auto;
}}
.chart-box h3 {{ margin-bottom: 12px; font-weight: 500; color: #a0a0c0; }}
</style>
</head>
<body>
<h1>{title}</h1>
{charts_html}
</body>
</html>"""

    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"图表已保存: {filename}")


def save_stock_chart(filename, dates, closes, predictions, labels,
                     train_losses=None, val_losses=None, is_predicted=None):
    """生成股票预测可视化图表"""
    configs = []

    # 图1: 价格走势 + 预测标记
    buy_signal = [c if p is not None and p > 0.5 else None for c, p in zip(closes, predictions)]
    sell_signal = [c if p is not None and p <= 0.5 else None for c, p in zip(closes, predictions)]

    datasets = []
    if is_predicted:
        real_closes = [c if not ip else None for c, ip in zip(closes, is_predicted)]
        pred_closes = [c if ip else None for c, ip in zip(closes, is_predicted)]
        datasets.append({'name': '真实收盘价', 'data': real_closes, 'color': '#3498db', 'fill': False})
        datasets.append({'name': '推算收盘价', 'data': pred_closes, 'color': '#9b59b6', 'fill': False, 'dashed': True})
    else:
        datasets.append({'name': '收盘价', 'data': closes, 'color': '#3498db', 'fill': False})

    datasets.extend([
        {'name': '看涨信号', 'data': buy_signal, 'color': '#2ecc71', 'fill': False},
        {'name': '看跌信号', 'data': sell_signal, 'color': '#e74c3c', 'fill': False},
    ])

    configs.append({
        'label': '价格走势与预测信号',
        'type': 'line',
        'labels': dates,
        'datasets': datasets
    })

    # 图2: 训练损失
    if train_losses is not None:
        epochs = [str(i + 1) for i in range(len(train_losses))]
        ds = [{'name': '训练损失', 'data': train_losses, 'color': '#e94560', 'fill': True}]
        if val_losses:
            ds.append({'name': '验证损失', 'data': val_losses, 'color': '#f39c12', 'fill': False})
        configs.append({
            'label': '训练损失曲线',
            'type': 'line',
            'labels': epochs,
            'datasets': ds
        })

    save_chart(filename, '股票涨跌预测分析', configs)


def save_weather_chart(filename, dates, temps, predictions, labels,
                       train_losses=None, val_losses=None, is_predicted=None,
                       task='classification'):
    """生成天气预测可视化图表"""
    configs = []

    if task == 'regression':
        # 回归模式：显示实际温度和模型预测温度
        datasets = []
        if is_predicted:
            real_temps = [t if not ip else None for t, ip in zip(temps, is_predicted)]
            pred_temps_line = [t if ip else None for t, ip in zip(temps, is_predicted)]
            datasets.append({'name': '真实温度', 'data': real_temps, 'color': '#3498db', 'fill': True})
            datasets.append({'name': '推算温度', 'data': pred_temps_line, 'color': '#9b59b6', 'fill': False, 'dashed': True})
        else:
            datasets.append({'name': '实际温度', 'data': temps, 'color': '#3498db', 'fill': True})

        # 模型预测温度线
        datasets.append({'name': '模型预测温度', 'data': predictions, 'color': '#e94560', 'fill': False})

        configs.append({
            'label': '温度走势与模型预测',
            'type': 'line',
            'labels': dates,
            'datasets': datasets
        })
    else:
        # 分类模式：显示涨跌信号
        up_signal = [t if p is not None and p > 0.5 else None for t, p in zip(temps, predictions)]
        down_signal = [t if p is not None and p <= 0.5 else None for t, p in zip(temps, predictions)]

        datasets = []
        if is_predicted:
            real_temps = [t if not ip else None for t, ip in zip(temps, is_predicted)]
            pred_temps_line = [t if ip else None for t, ip in zip(temps, is_predicted)]
            datasets.append({'name': '真实温度', 'data': real_temps, 'color': '#3498db', 'fill': True})
            datasets.append({'name': '推算温度', 'data': pred_temps_line, 'color': '#9b59b6', 'fill': False, 'dashed': True})
        else:
            datasets.append({'name': '平均温度', 'data': temps, 'color': '#3498db', 'fill': True})

        datasets.extend([
            {'name': '预测升温日', 'data': up_signal, 'color': '#2ecc71', 'fill': False},
            {'name': '预测降温日', 'data': down_signal, 'color': '#e74c3c', 'fill': False},
        ])

        configs.append({
            'label': '温度走势与预测信号（绿=预测升温, 红=预测降温）',
            'type': 'line',
            'labels': dates,
            'datasets': datasets
        })

    # 图2: 训练损失
    if train_losses is not None:
        epochs = [str(i + 1) for i in range(len(train_losses))]
        ds = [{'name': '训练损失', 'data': train_losses, 'color': '#e94560', 'fill': True}]
        if val_losses:
            ds.append({'name': '验证损失', 'data': val_losses, 'color': '#f39c12', 'fill': False})
        configs.append({
            'label': '训练损失曲线',
            'type': 'line',
            'labels': epochs,
            'datasets': ds
        })

    save_chart(filename, '天气温度预测分析', configs)
