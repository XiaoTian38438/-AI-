"""输出校验模块 - 防止大模型幻觉和不当输出

赛事要求：在设计中包含对大模型输出的验证逻辑（如正则表达式匹配、关键词过滤），
防止模型产生幻觉导致系统崩溃。

校验层次：
1. 关键词过滤 - 检测不当投资建议
2. 正则格式校验 - 确保各领域输出包含必要数据
3. 数值合理性检查 - 检测异常数值
4. 结构化JSON校验 - 验证LLM返回的JSON结构完整性
5. 置信度过滤 - 低置信度输出添加警告标签
6. LLM幻觉检测 - 检测与输入数据矛盾的结论
7. 输出清洗 - 替换不当表述为安全措辞
8. 风险提示追加 - 涉及投资建议时自动追加免责声明
"""
import re

# 禁止出现的关键词（不当投资建议、虚假承诺等）
FORBIDDEN_KEYWORDS = [
    '一定涨', '必定跌', '保证盈利', '稳赚不赔', '绝对收益',
    '100%盈利', '零风险', '包赚', '内幕消息', '庄家操控',
    '必涨无疑', '抄底必赚', '全仓买入', '必涨', '必跌',
    '满仓', '杠杆全仓', '融资全买', '绝对涨', '绝对跌',
]

# 各领域必要数据格式（正则表达式）
REQUIRED_PATTERNS = {
    'stock': [r'\d{6}', r'(涨|跌|平|中性)'],
    'weather': [r'-?\d+\.?\d*', r'(升温|降温|平稳|晴|阴|雨|雪|多云)'],
    'currency': [r'\d+\.\d+'],
}

# 数值合理性范围
VALUE_RANGES = {
    'stock_price': (0.01, 100000),
    'temperature_c': (-60, 60),
    'exchange_rate': (0.001, 1000),
}

# 幻觉检测：输入数据与输出结论的矛盾模式
HALLUCINATION_PATTERNS = [
    # 股票：数据下跌但结论说上涨
    (r'涨跌幅[:：]\s*-?\d+\.?\d*%', r'(一定|必然|必将)上涨'),
    # 汇率：无数据支撑的极端预测
    (r'汇率.*?(\d+\.\d+)', r'(暴涨|暴跌|崩盘)'),
    # 天气：温度与建议矛盾
    (r'(\d+\.?\d*)°C.*?降温', r'注意防暑'),
    (r'(\d+\.?\d*)°C.*?升温', r'注意防寒'),
]


class ValidationResult:
    """校验结果"""
    def __init__(self, passed=True, issues=None, sanitized=None, warnings=None):
        self.passed = passed
        self.issues = issues or []
        self.sanitized = sanitized
        self.warnings = warnings or []

    def __bool__(self):
        return self.passed

    def __repr__(self):
        if self.passed:
            return "ValidationResult(通过)"
        return f"ValidationResult(未通过, issues={self.issues})"


def validate_output(content, domain='general', strict=False, input_data=None):
    """
    校验大模型输出，防止幻觉和不当内容

    Args:
        content: 待校验文本
        domain: 领域 (stock/weather/currency/general)
        strict: 严格模式（任何问题都不通过）
        input_data: 输入数据字典，用于幻觉检测（与输出结论对比）
    Returns:
        ValidationResult
    """
    if not content or not content.strip():
        return ValidationResult(passed=False, issues=['输出为空'])

    issues = []
    warnings = []

    # 1. 关键词过滤 - 检查不当建议
    for keyword in FORBIDDEN_KEYWORDS:
        if keyword in content:
            issues.append(f"包含不当表述: '{keyword}'")

    # 2. 数据格式校验（正则表达式匹配）
    if domain in REQUIRED_PATTERNS:
        for pattern in REQUIRED_PATTERNS[domain]:
            if not re.search(pattern, content):
                issues.append(f"缺少必要数据格式: {pattern}")

    # 3. 数值合理性检查
    numbers = re.findall(r'-?\d+\.?\d*', content)
    for num_str in numbers:
        try:
            num = float(num_str)
            if abs(num) > 1e10:
                issues.append(f"数值异常大: {num_str}")
            elif domain == 'stock' and 0 < num < 0.01:
                issues.append(f"股价异常低: {num_str}")
            elif domain == 'weather' and (num > 60 or num < -60):
                issues.append(f"温度异常: {num_str}°C")
            elif domain == 'currency' and num > 0 and (num < 0.001 or num > 1000):
                warnings.append(f"汇率值可能异常: {num_str}")
        except ValueError:
            pass

    # 4. 结构化JSON校验（检查LLM返回中是否包含有效JSON片段）
    json_patterns = [r'\{[^}]*"signal"[^}]*\}', r'\{[^}]*"confidence"[^}]*\}']
    json_found = any(re.search(p, content) for p in json_patterns)
    if json_found:
        # 有JSON片段，校验其结构
        try:
            m = re.search(r'\{[^}]+\}', content)
            if m:
                import json
                parsed = json.loads(m.group())
                if 'signal' in parsed and 'confidence' in parsed:
                    conf = float(parsed['confidence'])
                    if not (0 <= conf <= 1):
                        issues.append(f"置信度超出范围: {conf}")
        except (json.JSONDecodeError, ValueError):
            warnings.append("输出中包含无法解析的JSON片段")

    # 5. 置信度过滤（检测"肯定性"表述，低置信度应有保留）
    certainty_patterns = [r'肯定会', r'必然会', r'毫无疑问', r'百分之百']
    for pat in certainty_patterns:
        if re.search(pat, content):
            warnings.append(f"包含过度肯定表述，可能为幻觉: {pat}")

    # 6. 幻觉检测（与输入数据对比，检查矛盾结论）
    if input_data and isinstance(input_data, dict):
        # 股票：检查涨跌结论与实际数据是否矛盾
        if domain == 'stock':
            change_pct = input_data.get('change_pct', 0)
            if change_pct < -2 and '上涨' in content and '大幅' in content:
                issues.append("幻觉: 数据显示下跌但结论称大幅上涨")
            elif change_pct > 2 and '下跌' in content and '大幅' in content:
                issues.append("幻觉: 数据显示上涨但结论称大幅下跌")
        # 天气：检查温度结论与ML预测是否矛盾
        elif domain == 'weather':
            ml_pred = input_data.get('ml_pred')
            temp = input_data.get('temp')
            if ml_pred and temp:
                diff = ml_pred - temp
                if diff < -3 and '升温' in content and '大幅' in content:
                    issues.append("幻觉: ML预测降温但结论称大幅升温")
                elif diff > 3 and '降温' in content and '大幅' in content:
                    issues.append("幻觉: ML预测升温但结论称大幅降温")

    # 7. 长度检查
    if len(content) < 10:
        issues.append("输出内容过短")

    # 判断是否通过
    critical_issues = [i for i in issues if '不当' in i or '空' in i or '异常' in i or '幻觉' in i]
    passed = len(critical_issues) == 0 if not strict else len(issues) == 0

    return ValidationResult(passed=passed, issues=issues, warnings=warnings)


def sanitize_output(content):
    """
    清理输出中的不当内容，替换为安全表述
    """
    sanitized = content
    replacements = {
        '一定涨': '可能上涨',
        '必定跌': '可能下跌',
        '必涨': '可能上涨',
        '必跌': '可能下跌',
        '保证盈利': '存在盈利可能',
        '稳赚不赔': '存在一定风险',
        '零风险': '存在一定风险',
        '100%盈利': '有较高概率盈利',
        '包赚': '有盈利可能',
        '内幕消息': '市场信息',
        '必涨无疑': '有上涨趋势',
        '全仓买入': '建议适度配置',
        '满仓': '建议适度配置',
        '绝对涨': '可能上涨',
        '绝对跌': '可能下跌',
        '毫无疑问': '有一定可能',
        '肯定会': '有较大概率会',
        '暴跌': '较大幅度下跌',
        '暴涨': '较大幅度上涨',
        '崩盘': '大幅波动',
    }
    for old, new in replacements.items():
        sanitized = sanitized.replace(old, new)

    # 追加风险提示（如果内容涉及投资建议）
    investment_words = ['买入', '卖出', '持仓', '投资', '收益', '加仓', '减仓']
    if any(w in sanitized for w in investment_words):
        if '风险提示' not in sanitized and '不构成投资建议' not in sanitized:
            sanitized += '\n\n⚠️ 风险提示：以上分析仅供参考，不构成投资建议。'

    return sanitized
