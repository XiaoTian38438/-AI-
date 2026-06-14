"""SQLite 数据持久化模块

功能：
- 用户认证（注册/登录/Session）
- 对话历史存储
- 使用记录存储
- 模型参数存储
"""
import os
import json
import hashlib
import sqlite3
from datetime import datetime

_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'smart_assistant.db')


def _get_conn():
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """初始化数据库表"""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now','localtime')),
            last_login TEXT
        );

        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            route TEXT DEFAULT '',
            timestamp TEXT DEFAULT (datetime('now','localtime')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS usage_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            query TEXT NOT NULL,
            route TEXT DEFAULT 'unknown',
            validation_passed INTEGER DEFAULT 1,
            timestamp TEXT DEFAULT (datetime('now','localtime')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS model_params (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            model_name TEXT DEFAULT '',
            params TEXT DEFAULT '{}',
            updated_at TEXT DEFAULT (datetime('now','localtime')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)
    conn.commit()
    conn.close()


def _hash_password(password):
    return hashlib.sha256(password.encode('utf-8')).hexdigest()


# ======================== 用户认证 ========================

def register_user(username, password):
    """注册用户，返回 (success, message, user_id)"""
    if not username or len(username) < 2:
        return False, '用户名至少2个字符', None
    if not password or len(password) < 4:
        return False, '密码至少4个字符', None
    try:
        conn = _get_conn()
        cursor = conn.execute(
            'INSERT INTO users (username, password_hash) VALUES (?, ?)',
            (username, _hash_password(password))
        )
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return True, '注册成功', user_id
    except sqlite3.IntegrityError:
        return False, '用户名已存在', None


def verify_user(username, password):
    """验证用户登录，返回 (success, user_id)"""
    conn = _get_conn()
    row = conn.execute(
        'SELECT id, password_hash FROM users WHERE username = ?',
        (username,)
    ).fetchone()
    conn.close()
    if row and row['password_hash'] == _hash_password(password):
        # 更新最后登录时间
        conn = _get_conn()
        conn.execute('UPDATE users SET last_login = ? WHERE id = ?',
                     (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), row['id']))
        conn.commit()
        conn.close()
        return True, row['id']
    return False, None


# ======================== 对话历史 ========================

def save_conversation(user_id, role, content, route=''):
    """保存一条对话记录"""
    conn = _get_conn()
    conn.execute(
        'INSERT INTO conversations (user_id, role, content, route) VALUES (?, ?, ?, ?)',
        (user_id, role, content, route)
    )
    conn.commit()
    conn.close()


def get_conversations(user_id=None, limit=50):
    """获取对话历史"""
    conn = _get_conn()
    if user_id:
        rows = conn.execute(
            'SELECT * FROM conversations WHERE user_id = ? ORDER BY id DESC LIMIT ?',
            (user_id, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            'SELECT * FROM conversations ORDER BY id DESC LIMIT ?',
            (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def clear_conversations(user_id=None):
    """清空对话历史"""
    conn = _get_conn()
    if user_id:
        conn.execute('DELETE FROM conversations WHERE user_id = ?', (user_id,))
    else:
        conn.execute('DELETE FROM conversations')
    conn.commit()
    conn.close()


# ======================== 使用记录 ========================

def save_usage(user_id, query, route='unknown', validation_passed=True):
    """保存使用记录"""
    conn = _get_conn()
    conn.execute(
        'INSERT INTO usage_records (user_id, query, route, validation_passed) VALUES (?, ?, ?, ?)',
        (user_id, query, route, 1 if validation_passed else 0)
    )
    conn.commit()
    conn.close()


def get_usage(user_id=None, limit=50):
    """获取使用记录"""
    conn = _get_conn()
    if user_id:
        rows = conn.execute(
            'SELECT * FROM usage_records WHERE user_id = ? ORDER BY id DESC LIMIT ?',
            (user_id, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            'SELECT * FROM usage_records ORDER BY id DESC LIMIT ?',
            (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ======================== 模型参数 ========================

def save_model_params(user_id, model_name, params):
    """保存模型参数"""
    conn = _get_conn()
    conn.execute(
        'INSERT OR REPLACE INTO model_params (user_id, model_name, params, updated_at) VALUES (?, ?, ?, ?)',
        (user_id, model_name, json.dumps(params), datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    )
    conn.commit()
    conn.close()


def get_model_params(user_id):
    """获取模型参数"""
    conn = _get_conn()
    rows = conn.execute(
        'SELECT * FROM model_params WHERE user_id = ?',
        (user_id,)
    ).fetchall()
    conn.close()
    return [{'model_name': r['model_name'], 'params': json.loads(r['params']), 'updated_at': r['updated_at']} for r in rows]


# 初始化数据库
init_db()
