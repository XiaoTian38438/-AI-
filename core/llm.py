"""大模型调用接口 - 支持 OpenRouter / OpenAI 兼容 API / Ollama / 规则回退

多 Key 轮换机制：
- 内置 3 个 OpenRouter API Key，自动轮换和故障转移
- 定期健康检测，自动跳过不可用的 Key
"""
import os
import json
import time
import threading

os.environ['http_proxy'] = 'http://127.0.0.1:7897'
os.environ['https_proxy'] = 'http://127.0.0.1:7897'
# ======================== OpenRouter 多 Key 管理 ========================

# 从环境变量获取 OpenRouter API Key，多个 Key 用逗号分隔
_OPENROUTER_KEYS_ENV = os.environ.get('OPENROUTER_API_KEYS', '')
_OPENROUTER_KEYS = [k.strip() for k in _OPENROUTER_KEYS_ENV.split(',') if k.strip()]

# 如果环境变量未设置，使用配置文件
if not _OPENROUTER_KEYS:
    _OPENROUTER_KEYS = ['sk-or-v1-placeholder-key-1', 'sk-or-v1-placeholder-key-2', 'sk-or-v1-placeholder-key-3']

_OPENROUTER_MODEL = 'nvidia/nemotron-3-ultra-550b-a55b:free'
_OPENROUTER_BASE = 'https://openrouter.ai/api/v1'

# Key 健康状态：{key: {'available': bool, 'last_check': float, 'fail_count': int}}
_key_health = {}
_current_key_idx = 0
_key_lock = threading.Lock()

# 全局配置
_config = {
    'api_key': os.environ.get('LLM_API_KEY', ''),
    'base_url': os.environ.get('LLM_BASE_URL', 'https://api.openai.com/v1'),
    'model': os.environ.get('LLM_MODEL', 'gpt-3.5-turbo'),
    'timeout': int(os.environ.get('LLM_TIMEOUT', '60')),
}


def _init_key_health():
    """初始化所有 Key 的健康状态"""
    for key in _OPENROUTER_KEYS:
        if key not in _key_health:
            _key_health[key] = {'available': True, 'last_check': 0, 'fail_count': 0}


_init_key_health()


def _get_available_openrouter_key():
    """获取一个可用的 OpenRouter Key（轮换 + 跳过故障）"""
    global _current_key_idx
    with _key_lock:
        for i in range(len(_OPENROUTER_KEYS)):
            idx = (_current_key_idx + i) % len(_OPENROUTER_KEYS)
            key = _OPENROUTER_KEYS[idx]
            health = _key_health.get(key, {})
            # 如果最近5分钟内失败超过3次，跳过
            if health.get('fail_count', 0) >= 3:
                if time.time() - health.get('last_check', 0) < 300:
                    continue
                # 超过5分钟，给一次重试机会
                _key_health[key]['fail_count'] = 2
            _current_key_idx = (idx + 1) % len(_OPENROUTER_KEYS)
            return key
    return None


def _mark_key_failed(key):
    """标记 Key 失败"""
    with _key_lock:
        if key in _key_health:
            _key_health[key]['fail_count'] += 1
            _key_health[key]['last_check'] = time.time()


def _mark_key_ok(key):
    """标记 Key 正常"""
    with _key_lock:
        if key in _key_health:
            _key_health[key]['fail_count'] = 0
            _key_health[key]['last_check'] = time.time()


def check_openrouter_keys():
    """检测所有 OpenRouter Key 的可用性，返回状态摘要"""
    results = []
    for i, key in enumerate(_OPENROUTER_KEYS):
        health = _key_health.get(key, {})
        try:
            from urllib.request import Request, urlopen
            url = f"{_OPENROUTER_BASE}/models"
            req = Request(url, method='GET')
            req.add_header('Authorization', f"Bearer {key}")
            with urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    _mark_key_ok(key)
                    results.append({'index': i, 'status': 'ok'})
                else:
                    _mark_key_failed(key)
                    results.append({'index': i, 'status': f'http_{resp.status}'})
        except Exception as e:
            _mark_key_failed(key)
            results.append({'index': i, 'status': f'error: {str(e)[:50]}'})
    return results


# ======================== 配置管理 ========================

def configure(**kwargs):
    """配置 LLM 参数"""
    _config.update(kwargs)


def get_config():
    return _config.copy()


def is_available():
    """检查 LLM 是否可用（OpenRouter / 自定义API / Ollama）"""
    key = _get_available_openrouter_key()
    if key:
        return True
    if _config.get('api_key'):
        return True
    return _check_ollama()


# ======================== Ollama ========================

def _check_ollama():
    """检查 Ollama 是否在运行"""
    try:
        from urllib.request import urlopen, Request
        req = Request('http://localhost:11434/api/tags')
        with urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


# ======================== 核心调用 ========================

def call(prompt, system_prompt=None, temperature=0.7, max_tokens=1000, fallback=None, timeout=None):
    """
    调用大语言模型，依次尝试：OpenRouter → 自定义 OpenAI API → Ollama → fallback
    timeout: 整体超时秒数，超时后直接走fallback
    """
    start_time = time.time()

    # 1. OpenRouter (优先，带轮询重试)
    for attempt in range(len(_OPENROUTER_KEYS)):
        if timeout and time.time() - start_time > timeout:
            print(f"[LLM] 整体超时({timeout}s)，跳过剩余尝试")
            break
        key = _get_available_openrouter_key()
        if not key:
            print("[LLM] 无可用OpenRouter Key")
            break
        # 限制单次请求超时不超过剩余时间（防止底层120s阻塞）
        _saved_env = None
        if timeout:
            remaining = timeout - (time.time() - start_time)
            if remaining < 3:
                print(f"[LLM] 剩余时间不足({remaining:.1f}s)，跳过OpenRouter")
                break
            _saved_env = os.environ.get('OPENROUTER_TIMEOUT')
            os.environ['OPENROUTER_TIMEOUT'] = str(max(int(remaining), 8))
        try:
            result = _call_openrouter(prompt, system_prompt, temperature, max_tokens, key)
            if result:
                _mark_key_ok(key)
                return result
        except Exception as e:
            _mark_key_failed(key)
            print(f"[LLM] OpenRouter Key失败 (尝试 {attempt+1}/{len(_OPENROUTER_KEYS)}): {e}")
        finally:
            # 恢复环境变量
            if _saved_env is not None:
                os.environ['OPENROUTER_TIMEOUT'] = _saved_env
            elif timeout:
                os.environ.pop('OPENROUTER_TIMEOUT', None)

    # 2. 自定义 OpenAI-compatible API
    if _config.get('api_key'):
        try:
            result = _call_openai(prompt, system_prompt, temperature, max_tokens)
            if result:
                return result
        except Exception as e:
            print(f"[LLM] API 调用失败: {e}")

    # 3. Ollama local
    try:
        result = _call_ollama(prompt, system_prompt)
        if result:
            return result
    except Exception:
        pass

    # 4. Fallback
    if fallback:
        return fallback(prompt)

    return None


def call_openrouter_direct(prompt, system_prompt=None, temperature=0.7, max_tokens=1000, total_timeout=20):
    """直接调用 OpenRouter（跳过其他通道），用于提示词优化节点
    total_timeout: 整体超时秒数，默认20秒
    """
    start_time = time.time()
    print(f"[OpenRouter] 开始提示词优化, 模型={_OPENROUTER_MODEL}")
    for attempt in range(min(len(_OPENROUTER_KEYS), 3)):  # 最多尝试3个key
        remaining = total_timeout - (time.time() - start_time)
        if remaining < 3:
            print(f"[OpenRouter] 整体超时({total_timeout}s)，停止重试，回退到本地规则")
            break
        key = _get_available_openrouter_key()
        if not key:
            print("[OpenRouter] 无可用Key，回退到本地规则")
            break
        try:
            # 限制单次请求超时不超过剩余时间
            old_env = os.environ.get('OPENROUTER_TIMEOUT')
            os.environ['OPENROUTER_TIMEOUT'] = str(min(int(remaining), 15))
            result = _call_openrouter(prompt, system_prompt, temperature, max_tokens, key)
            if old_env is not None:
                os.environ['OPENROUTER_TIMEOUT'] = old_env
            else:
                os.environ.pop('OPENROUTER_TIMEOUT', None)
            if result:
                _mark_key_ok(key)
                print(f"[OpenRouter] 提示词优化成功, 耗时={time.time()-start_time:.1f}s")
                return result
        except Exception as e:
            if old_env is not None:
                os.environ['OPENROUTER_TIMEOUT'] = old_env
            else:
                os.environ.pop('OPENROUTER_TIMEOUT', None)
            _mark_key_failed(key)
            print(f"[OpenRouter] Key尝试失败 (attempt {attempt+1}): {e}")
    print("[OpenRouter] 所有尝试失败，回退到本地规则解析")
    return None


# ======================== OpenRouter 调用 ========================

def _call_openrouter(prompt, system_prompt, temperature, max_tokens, api_key):
    """OpenRouter API 调用（带详细错误日志和HTTP错误处理）"""
    import socket
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError, URLError

    url = f"{_OPENROUTER_BASE}/chat/completions"
    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    messages.append({'role': 'user', 'content': prompt})

    body = json.dumps({
        'model': _OPENROUTER_MODEL,
        'messages': messages,
        'temperature': temperature,
        'max_tokens': max_tokens,
    }).encode('utf-8')

    req = Request(url, data=body, method='POST')
    req.add_header('Content-Type', 'application/json')
    req.add_header('Authorization', f"Bearer {api_key}")
    req.add_header('HTTP-Referer', 'https://mini-ai.local')
    req.add_header('X-Title', 'Mini AI Smart Assistant')
    req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')

    timeout = int(os.environ.get('OPENROUTER_TIMEOUT', '120'))

    # Windows下urlopen超时可能不生效，用socket超时兜底
    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout)
    try:
        resp = urlopen(req, timeout=timeout)
        raw = resp.read().decode('utf-8')
        result = json.loads(raw)

        # 检查API错误
        if 'error' in result:
            err_msg = result['error'].get('message', str(result['error'])) if isinstance(result['error'], dict) else str(result['error'])
            print(f"[OpenRouter] API错误: {err_msg}")
            raise ValueError(f"API错误: {err_msg[:100]}")

        if 'choices' not in result or not result['choices']:
            print(f"[OpenRouter] 返回异常结构: {raw[:300]}")
            raise ValueError(f"返回异常: {raw[:100]}")

        content = result['choices'][0].get('message', {}).get('content', '')
        if not content:
            raise ValueError("返回空内容")

        print(f"[OpenRouter] 调用成功, 模型={_OPENROUTER_MODEL}, 响应长度={len(content)}")
        return content

    except HTTPError as e:
        # 解析HTTP错误响应体
        err_body = ''
        try:
            err_body = e.read().decode('utf-8', errors='replace')[:300]
        except Exception:
            pass
        print(f"[OpenRouter] HTTP错误 {e.code}: {err_body}")
        raise ValueError(f"HTTP {e.code}: {err_body[:100]}")

    except URLError as e:
        print(f"[OpenRouter] 连接失败: {e.reason}")
        raise ValueError(f"连接失败: {e.reason}")

    except socket.timeout:
        print(f"[OpenRouter] 请求超时({timeout}s)")
        raise ValueError(f"请求超时({timeout}s)")

    finally:
        socket.setdefaulttimeout(old_timeout)


# ======================== OpenAI 兼容调用 ========================

def _call_openai(prompt, system_prompt, temperature, max_tokens):
    """OpenAI-compatible API 调用 (纯 urllib，无需 openai 库)"""
    from urllib.request import Request, urlopen

    url = f"{_config['base_url'].rstrip('/')}/chat/completions"
    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    messages.append({'role': 'user', 'content': prompt})

    body = json.dumps({
        'model': _config['model'],
        'messages': messages,
        'temperature': temperature,
        'max_tokens': max_tokens,
    }).encode('utf-8')

    req = Request(url, data=body, method='POST')
    req.add_header('Content-Type', 'application/json')
    req.add_header('Authorization', f"Bearer {_config['api_key']}")

    with urlopen(req, timeout=_config.get('timeout', 30)) as resp:
        result = json.loads(resp.read().decode('utf-8'))
        return result['choices'][0]['message']['content']


# ======================== Ollama 调用 ========================

def _call_ollama(prompt, system_prompt):
    """Ollama 本地模型调用"""
    from urllib.request import Request, urlopen

    model = _config.get('ollama_model', 'qwen2.5:7b')
    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

    body = json.dumps({
        'model': model,
        'prompt': full_prompt,
        'stream': False,
    }).encode('utf-8')

    req = Request('http://localhost:11434/api/generate', data=body, method='POST')
    req.add_header('Content-Type', 'application/json')

    with urlopen(req, timeout=_config.get('timeout', 30)) as resp:
        result = json.loads(resp.read().decode('utf-8'))
        return result.get('response', '')
