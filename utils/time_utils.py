"""时间工具 - 通过微软 NTP 服务器获取精确当前时间"""
import struct
import socket
from datetime import datetime, timedelta


def get_ntp_time(host='time.windows.com', port=123, timeout=5):
    """
    通过 NTP 协议从微软时间服务器获取当前 UTC 时间
    失败时降级为本地系统时间
    """
    try:
        # NTP 协议报文：48字节，首字节 0x1B (LI=0, VN=3, Mode=3=client)
        ntp_data = b'\x1b' + 47 * b'\0'
        addr = (host, port)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(timeout)
            s.sendto(ntp_data, addr)
            data, _ = s.recvfrom(1024)

        if len(data) < 48:
            raise ValueError("NTP 响应数据不完整")

        # 传输时间戳在第40-47字节（8字节），前4字节秒，后4字节小数秒
        # NTP纪元: 1900-01-01, Unix纪元: 1970-01-01, 差值70年
        secs = struct.unpack('!I', data[40:44])[0]
        frac = struct.unpack('!I', data[44:48])[0]
        ntp_epoch_diff = 2208988800  # 1900→1970 秒差
        unix_secs = secs - ntp_epoch_diff + frac / (2 ** 32)

        utc_time = datetime.utcfromtimestamp(unix_secs)
        # 转为北京时间 UTC+8
        local_time = utc_time + timedelta(hours=8)
        return local_time
    except Exception as e:
        print(f"[NTP] 微软时间服务器不可用 ({e})，使用本地系统时间")
        return datetime.now()


# 模块级缓存：首次调用后缓存结果，避免重复网络请求
_cached_time = None


def get_current_time():
    """获取当前时间（带缓存，同一进程内只请求一次 NTP）"""
    global _cached_time
    if _cached_time is None:
        _cached_time = get_ntp_time()
        print(f"[NTP] 当前时间: {_cached_time.strftime('%Y-%m-%d %H:%M:%S')}")
    return _cached_time


def get_date_range_around_now(past_years=2, future_days=180):
    """
    以当前日期为中心，计算数据时间范围
    返回: (start_date, end_date, today) 格式 'YYYY-MM-DD'
    past_years: 向过去取多少年的真实数据
    future_days: 向未来推算多少天
    """
    now = get_current_time()
    start = (now - timedelta(days=int(past_years * 365))).strftime('%Y-%m-%d')
    today = now.strftime('%Y-%m-%d')
    end = (now + timedelta(days=future_days)).strftime('%Y-%m-%d')
    return start, end, today
