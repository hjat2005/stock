# config/settings.py
"""系统配置"""

import os
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

# 数据目录
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# 数据库配置
DATABASE_URL = f"sqlite:///{DATA_DIR / 'quant.db'}"

# 日志配置
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# API配置
YFINANCE_TIMEOUT = 30

# 默认股票池
DEFAULT_TICKERS = [
    "AAPL", "GOOGL", "MSFT", "AMZN", "META", 
    "NVDA", "TSLA", "NFLX", "AMD", "INTC"
]

# 技术指标配置
TECHNICAL_INDICATORS = {
    'sma_windows': [20, 50, 200],
    'ema_spans': [12, 26],
    'rsi_period': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'bb_window': 20,
    'bb_std': 2,
    'atr_period': 14
}

# 预警配置
ALERT_CONFIG = {
    'price_change_threshold': 0.05,  # 5%价格变动
    'volume_spike_threshold': 2.0,   # 成交量放大2倍
}
