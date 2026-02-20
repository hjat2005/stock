#!/usr/bin/env python3
"""
v12_integrated_trading_system.py
V12集成实时交易系统

架构分层：
1. 数据采集层 (DataCollector) - 实时获取市场数据
2. 策略引擎层 (StrategyEngine) - V12策略逻辑
3. 交易执行层 (TradeExecutor) - 长桥API对接
4. 风险管理层 (RiskManager) - 熔断、止损、仓位控制
5. 主控层 (TradingSystem) - 协调各层运行
"""

import argparse
import os
import sys
import time
import signal
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from collections import deque

import numpy as np
import pandas as pd
import yfinance as yf
import pytz  # 用于时区转换

# ==========================
# 交易时间工具函数
# ==========================
def is_us_market_open() -> bool:
    """判断当前是否为美股开盘时间（9:30-16:00 ET，周一到周五）"""
    # 获取美东时间
    et_tz = pytz.timezone('US/Eastern')
    now_et = datetime.now(et_tz)
    
    # 检查是否为工作日（周一=0, 周五=4）
    if now_et.weekday() >= 5:  # 周六或周日
        return False
    
    # 检查是否在开盘时间内（9:30 - 16:00）
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= now_et <= market_close

def is_near_market_close(minutes_before: int = 5) -> bool:
    """判断是否接近收盘时间（默认收盘前5分钟）"""
    et_tz = pytz.timezone('US/Eastern')
    now_et = datetime.now(et_tz)
    
    # 检查是否为工作日
    if now_et.weekday() >= 5:
        return False
    
    # 计算收盘前的时间窗口
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    time_to_close = (market_close - now_et).total_seconds() / 60  # 转换为分钟
    
    # 如果在收盘前指定分钟内
    return 0 <= time_to_close <= minutes_before

def get_market_status() -> str:
    """获取当前市场状态描述"""
    et_tz = pytz.timezone('US/Eastern')
    now_et = datetime.now(et_tz)
    
    if now_et.weekday() >= 5:
        return f"周末休市 ({now_et.strftime('%A')})"
    
    current_time = now_et.strftime('%H:%M')
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    
    if now_et < market_open:
        return f"盘前 ({current_time} ET, 9:30开盘)"
    elif now_et > market_close:
        return f"盘后 ({current_time} ET, 16:00收盘)"
    else:
        return f"交易中 ({current_time} ET)"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==========================
# 长桥SDK导入
# ==========================
try:
    if '/usr/local/lib/python3.10/dist-packages' not in sys.path:
        sys.path.insert(0, '/usr/local/lib/python3.10/dist-packages')
    from longbridge.openapi import TradeContext, Config, OrderType, OrderSide, TimeInForceType
    LONGBRIDGE_AVAILABLE = True
    logger.info("✅ 长桥SDK导入成功")
except ImportError as e:
    LONGBRIDGE_AVAILABLE = False
    logger.warning(f"⚠️ 长桥SDK未安装: {e}")


# ==========================
# 配置常量
# ==========================
@dataclass
class V12Config:
    """V12策略配置"""
    # 策略参数
    mtm_period: int = 20          # 动量周期
    max_positions: int = 3        # 最大持仓数
    leverage: float = 1.3         # 杠杆倍数
    margin_buffer: float = 0.92   # 保证金缓冲
    max_dd_limit: float = 0.22    # 最大回撤限制22%
    cooldown_days: int = 10       # 冷却天数
    special_cap: float = 0.30     # Special组封顶30%
    
    # 选股阈值
    roc_threshold: float = 5.0    # ROC阈值(%)
    roc_buffer: float = 3.0       # 换仓缓冲(%)
    
    # ATR倍数
    atr_multipliers: Dict[str, float] = None
    
    # Special股票
    specials: set = None
    
    def __post_init__(self):
        if self.atr_multipliers is None:
            self.atr_multipliers = {
                "RKLB": 4.0, "CRWV": 4.0, "TSLA": 3.5, "NVDA": 3.5,
                "MU": 3.0, "VRT": 3.0, "DEFAULT": 3.0
            }
        if self.specials is None:
            self.specials = {"RKLB", "CRWV"}


# ==========================
# 数据模型
# ==========================
@dataclass
class MarketData:
    """市场数据模型"""
    ticker: str
    timestamp: datetime
    price: float
    volume: int
    roc: float
    sma30: float
    atr: float
    atr_stop: float
    trend_ok: bool
    atr_ok: bool
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Position:
    """持仓模型"""
    ticker: str
    quantity: int
    avg_price: float
    entry_time: datetime
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.avg_price
    
    def unrealized_pnl(self, current_price: float) -> Tuple[float, float]:
        """返回(盈亏金额, 盈亏百分比)"""
        avg_price = float(self.avg_price)
        pnl = (current_price - avg_price) * self.quantity
        pnl_pct = (current_price - avg_price) / avg_price * 100 if avg_price > 0 else 0
        return pnl, pnl_pct


@dataclass
class TradeSignal:
    """交易信号模型"""
    ticker: str
    action: str  # BUY, SELL, HOLD
    quantity: int
    price: float
    reason: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


# ==========================
# 1. 数据采集层
# ==========================
class DataCollector:
    """数据采集器 - 负责获取实时和历史数据"""
    
    def __init__(self, cache_size: int = 100):
        self.cache: Dict[str, deque] = {}
        self.cache_size = cache_size
        self.config = V12Config()
    
    def fetch_historical_data(self, ticker: str, days: int = 60) -> Optional[pd.DataFrame]:
        """获取历史数据"""
        try:
            end = datetime.now()
            start = end - timedelta(days=days)
            
            df = yf.download(
                ticker,
                start=start.strftime('%Y-%m-%d'),
                end=end.strftime('%Y-%m-%d'),
                interval="1d",
                auto_adjust=True,
                progress=False
            )
            
            if df.empty:
                return None
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            return df.dropna()
            
        except Exception as e:
            logger.error(f"获取 {ticker} 历史数据失败: {e}")
            return None
    
    def fetch_realtime_price(self, ticker: str) -> Optional[float]:
        """获取实时价格"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            price = info.get('regularMarketPrice') or info.get('currentPrice')
            if price:
                return float(price)
            
            # 备用方案：获取最近收盘价
            df = self.fetch_historical_data(ticker, days=2)
            if df is not None and not df.empty:
                return float(df['Close'].iloc[-1])
            
            return None
            
        except Exception as e:
            logger.error(f"获取 {ticker} 实时价格失败: {e}")
            return None
    
    def calculate_indicators(self, df: pd.DataFrame, ticker: str) -> Optional[MarketData]:
        """计算技术指标"""
        if df is None or len(df) < 30:
            return None
        
        try:
            close = df['Close'].astype(float)
            high = df['High'].astype(float)
            low = df['Low'].astype(float)
            volume = df.get('Volume', pd.Series([0] * len(df)))
            
            # 最新价格
            price = float(close.iloc[-1])
            
            # ROC动量
            base_price = float(close.iloc[-self.config.mtm_period-1])
            roc = ((price / base_price) - 1.0) * 100.0
            
            # SMA30
            sma30 = float(close.rolling(30).mean().iloc[-1])
            
            # ATR和止损线
            high_20d = float(high.iloc[-20:].max())
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs()
            ], axis=1).max(axis=1)
            atr = float(tr.rolling(14).mean().iloc[-1])
            
            m = self.config.atr_multipliers.get(ticker, self.config.atr_multipliers["DEFAULT"])
            atr_stop = high_20d - (m * atr)
            
            return MarketData(
                ticker=ticker,
                timestamp=datetime.now(),
                price=price,
                volume=int(volume.iloc[-1]) if not volume.empty else 0,
                roc=roc,
                sma30=sma30,
                atr=atr,
                atr_stop=atr_stop,
                trend_ok=price > sma30,
                atr_ok=price > atr_stop
            )
            
        except Exception as e:
            logger.error(f"计算 {ticker} 指标失败: {e}")
            return None
    
    def get_market_data(self, ticker: str) -> Optional[MarketData]:
        """获取完整市场数据"""
        df = self.fetch_historical_data(ticker)
        return self.calculate_indicators(df, ticker)


# ==========================
# 2. 策略引擎层
# ==========================
class StrategyEngine:
    """V12策略引擎 - 生成交易信号"""
    
    def __init__(self, config: V12Config = None):
        self.config = config or V12Config()
        self.data_collector = DataCollector()
    
    def select_stocks(self, tickers: List[str]) -> List[Tuple[str, MarketData]]:
        """
        V12选股逻辑：
        1. ROC > 5%
        2. 价格 > SMA30 (趋势OK)
        3. 按ROC排序，取前max_positions
        """
        candidates = []
        
        logger.info(f"\n{'='*80}")
        logger.info("🎯 V12选股分析")
        logger.info(f"{'='*80}")
        logger.info(f"选股条件: ROC > {self.config.roc_threshold}% 且 价格 > SMA30\n")
        
        for ticker in tickers:
            data = self.data_collector.get_market_data(ticker)
            if data is None:
                logger.warning(f"{ticker}: 数据获取失败")
                continue
            
            # 选股条件
            selected = data.roc > self.config.roc_threshold and data.trend_ok
            
            status = "✅ 入选" if selected else "❌ 未入选"
            reason = []
            if data.roc <= self.config.roc_threshold:
                reason.append(f"ROC={data.roc:.1f}%<={self.config.roc_threshold}%")
            if not data.trend_ok:
                reason.append(f"价格{data.price:.2f}<=SMA30({data.sma30:.2f})")
            
            reason_str = f" ({', '.join(reason)})" if reason else ""
            logger.info(f"{ticker:6s}: ROC={data.roc:6.2f}% | 价格=${data.price:7.2f} | "
                       f"SMA30=${data.sma30:7.2f} | {status}{reason_str}")
            
            if selected:
                candidates.append((ticker, data))
        
        # 按ROC排序
        candidates.sort(key=lambda x: x[1].roc, reverse=True)
        
        logger.info(f"\n📊 选股结果（按ROC排序）:")
        for i, (ticker, data) in enumerate(candidates[:self.config.max_positions], 1):
            logger.info(f"   {i}. {ticker}: ROC={data.roc:.2f}%, 价格=${data.price:.2f}")
        
        return candidates[:self.config.max_positions]
    
    def check_trend_stop(self, position: Position, pending_stops: Dict[str, datetime]) -> bool:
        """检查是否需要趋势止损（价格跌破SMA30）
        
        Args:
            position: 持仓信息
            pending_stops: 记录已触发止损但尚未执行的持仓 {ticker: 首次触发时间}
        
        Returns:
            是否应该执行卖出（仅在收盘前5分钟且首次触发后仍满足条件时返回True）
        """
        data = self.data_collector.get_market_data(position.ticker)
        if data is None:
            return False
        
        should_stop = data.price < data.sma30
        ticker = position.ticker
        
        if should_stop:
            if ticker not in pending_stops:
                # 首次触发止损，记录时间
                pending_stops[ticker] = datetime.now()
                logger.warning(f"🔴 {ticker} 首次触发趋势止损: "
                              f"价格${data.price:.2f} < SMA30(${data.sma30:.2f})，"
                              f"等待收盘前5分钟确认...")
                return False
            else:
                # 已触发过，检查是否接近收盘
                if is_near_market_close(minutes_before=5):
                    logger.warning(f"🔴 {ticker} 收盘前确认趋势止损: "
                                  f"价格${data.price:.2f} < SMA30(${data.sma30:.2f})，"
                                  f"执行卖出！")
                    return True
                else:
                    # 未接近收盘，继续等待
                    return False
        else:
            # 不再满足止损条件，从待执行列表中移除
            if ticker in pending_stops:
                del pending_stops[ticker]
                logger.info(f"🟢 {ticker} 价格回升，取消趋势止损")
            return False
    
    def check_atr_stop(self, position: Position, pending_stops: Dict[str, datetime]) -> bool:
        """检查是否需要ATR止损
        
        Args:
            position: 持仓信息
            pending_stops: 记录已触发止损但尚未执行的持仓 {ticker: 首次触发时间}
        
        Returns:
            是否应该执行卖出（仅在收盘前5分钟且首次触发后仍满足条件时返回True）
        """
        data = self.data_collector.get_market_data(position.ticker)
        if data is None:
            return False
        
        should_stop = data.price < data.atr_stop
        ticker = position.ticker
        
        if should_stop:
            if ticker not in pending_stops:
                # 首次触发止损，记录时间
                pending_stops[ticker] = datetime.now()
                logger.warning(f"🔴 {ticker} 首次触发ATR止损: "
                              f"价格${data.price:.2f} < ATR止损(${data.atr_stop:.2f})，"
                              f"等待收盘前5分钟确认...")
                return False
            else:
                # 已触发过，检查是否接近收盘
                if is_near_market_close(minutes_before=5):
                    logger.warning(f"🔴 {ticker} 收盘前确认ATR止损: "
                                  f"价格${data.price:.2f} < ATR止损(${data.atr_stop:.2f})，"
                                  f"执行卖出！")
                    return True
                else:
                    # 未接近收盘，继续等待
                    return False
        else:
            # 不再满足止损条件，从待执行列表中移除
            if ticker in pending_stops:
                del pending_stops[ticker]
                logger.info(f"🟢 {ticker} 价格回升，取消ATR止损")
            return False
    
    def generate_signals(
        self,
        tickers: List[str],
        positions: Dict[str, Position],
        total_equity: float,
        last_selected_tickers: List[str] = None,
        pending_stops: Dict[str, datetime] = None
    ) -> Tuple[List[TradeSignal], List[str], Dict[str, datetime]]:
        """生成交易信号
        
        Args:
            last_selected_tickers: 上次选中的股票列表，用于避免重复交易
            pending_stops: 记录已触发止损但尚未执行的持仓 {ticker: 首次触发时间}
        
        Returns:
            (交易信号列表, 当前选中股票列表, 更新后的pending_stops)
        """
        if pending_stops is None:
            pending_stops = {}
        
        signals = []
        
        # 1. 检查现有持仓的止损（延迟到收盘前5分钟执行）
        for ticker, pos in positions.items():
            trend_stop = self.check_trend_stop(pos, pending_stops)
            atr_stop = self.check_atr_stop(pos, pending_stops)
            
            if trend_stop or atr_stop:
                current_price = self.data_collector.fetch_realtime_price(ticker)
                if current_price:
                    stop_reason = "趋势止损" if trend_stop else "ATR止损"
                    signals.append(TradeSignal(
                        ticker=ticker,
                        action="SELL",
                        quantity=pos.quantity,
                        price=current_price,
                        reason=f"止损: {stop_reason} (收盘前5分钟确认)"
                    ))
                    # 执行后从pending列表中移除
                    if ticker in pending_stops:
                        del pending_stops[ticker]
        
        # 2. 选股
        selected = self.select_stocks(tickers)
        selected_tickers = [s[0] for s in selected]
        
        # 3. 清仓不在选股列表的股票（实时检查）
        for ticker in list(positions.keys()):
            if ticker not in selected_tickers:
                current_price = self.data_collector.fetch_realtime_price(ticker)
                if current_price:
                    signals.append(TradeSignal(
                        ticker=ticker,
                        action="SELL",
                        quantity=positions[ticker].quantity,
                        price=current_price,
                        reason="不在选股列表"
                    ))
        
        # 4. 计算目标仓位并生成买入信号
        # 只在以下情况生成买入信号：
        # - 首次运行（last_selected_tickers为None）
        # - 选股列表发生变化
        should_generate_buy = (
            last_selected_tickers is None or 
            set(selected_tickers) != set(last_selected_tickers)
        )
        
        if selected and should_generate_buy:
            target_shares = self.calculate_position_sizes(
                selected, positions, total_equity
            )
            
            # 生成买入信号
            for ticker, target_qty in target_shares.items():
                current_qty = positions.get(ticker, Position(ticker, 0, 0, datetime.now())).quantity
                diff = target_qty - current_qty
                
                if diff > 0:
                    current_price = self.data_collector.fetch_realtime_price(ticker)
                    if current_price:
                        signals.append(TradeSignal(
                            ticker=ticker,
                            action="BUY",
                            quantity=diff,
                            price=current_price,
                            reason=f"V12选股入选,目标仓位{target_qty}股"
                        ))
        elif selected:
            logger.info("📊 选股列表未变化，跳过买入信号生成")
        
        return signals, selected_tickers, pending_stops
    
    def calculate_position_sizes(
        self,
        selected: List[Tuple[str, MarketData]],
        positions: Dict[str, Position],
        total_equity: float
    ) -> Dict[str, int]:
        """计算目标仓位（V12规则）"""
        target_shares = {}
        
        # 分离Special组
        specials = [(t, d) for t, d in selected if t in self.config.specials]
        others = [(t, d) for t, d in selected if t not in self.config.specials]
        
        used_pct = 0.0
        
        # Special组分配
        if specials:
            cap_per = self.config.special_cap / len(specials)
            logger.info(f"\n🔸 Special组 ({self.config.special_cap*100:.0f}%封顶): "
                       f"{', '.join([t[0] for t in specials])}")
            for ticker, data in specials:
                target_val = total_equity * cap_per
                shares = self._round_shares(target_val / data.price)
                target_shares[ticker] = shares
                logger.info(f"   {ticker}: 目标权重{cap_per*100:.0f}% | "
                           f"目标市值${target_val:,.0f} | {shares}股")
            used_pct += self.config.special_cap
        
        # Others组分配
        if others:
            remain_pct = max(0.0, self.config.leverage * self.config.margin_buffer - used_pct)
            each_pct = remain_pct / len(others)
            logger.info(f"\n🔹 Others组 (剩余{remain_pct*100:.0f}%): "
                       f"{', '.join([t[0] for t in others])}")
            for ticker, data in others:
                target_val = total_equity * each_pct
                shares = self._round_shares(target_val / data.price)
                target_shares[ticker] = shares
                logger.info(f"   {ticker}: 目标权重{each_pct*100:.0f}% | "
                           f"目标市值${target_val:,.0f} | {shares}股")
        
        return target_shares
    
    @staticmethod
    def _round_shares(shares: float) -> int:
        """股数取整（10的倍数）"""
        if shares < 10:
            return 0
        return int((shares // 10) * 10)


# ==========================
# 3. 交易执行层
# ==========================
class TradeExecutor:
    """交易执行器 - 对接长桥API"""
    
    def __init__(self, paper_trading: bool = True):
        self.paper_trading = paper_trading
        self.ctx = None
        self._connect()
    
    def _connect(self):
        """连接长桥API"""
        if not LONGBRIDGE_AVAILABLE:
            logger.warning("🔴 长桥SDK未安装，使用模拟模式")
            return
        
        try:
            config = Config.from_env()
            self.ctx = TradeContext(config)
            logger.info(f"✅ 长桥API连接成功 | 模拟交易: {self.paper_trading}")
        except Exception as e:
            logger.error(f"❌ 长桥API连接失败: {e}")
            self.ctx = None
    
    def get_account_info(self) -> Dict:
        """获取账户信息"""
        if self.ctx is None:
            return {"cash": 110000.0, "equity": 110000.0, "mock": True}
        
        try:
            account_list = self.ctx.account_balance()
            if account_list and len(account_list) > 0:
                account = account_list[0]
                cash_info = account.cash_infos[0] if account.cash_infos else None
                available_cash = float(cash_info.available_cash) if cash_info else float(account.total_cash)
                return {
                    "cash": available_cash,
                    "equity": float(account.net_assets),
                    "mock": False
                }
        except Exception as e:
            logger.error(f"获取账户信息失败: {e}")
        
        return {"cash": 0, "equity": 0, "mock": True}
    
    def execute_signal(self, signal: TradeSignal) -> bool:
        """执行交易信号"""
        ticker = signal.ticker
        side = signal.action
        quantity = signal.quantity
        price = signal.price
        
        # 格式化股票代码
        if "." not in ticker:
            ticker_symbol = f"{ticker}.US"
        else:
            ticker_symbol = ticker
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 执行交易 | {ticker} | {side} | {quantity}股 @ ${price:.2f}")
        logger.info(f"   原因: {signal.reason}")
        logger.info(f"{'='*60}")
        
        # 模拟模式
        if self.ctx is None or self.paper_trading:
            logger.info(f"[模拟交易] {side} {ticker_symbol} | {quantity}股 | ${price:.2f}")
            return True
        
        # 实盘交易
        try:
            order_side = OrderSide.Buy if side == "BUY" else OrderSide.Sell
            
            # 价格精度处理
            if price:
                price = round(float(price), 2)
            
            resp = self.ctx.submit_order(
                symbol=ticker_symbol,
                order_type=OrderType.LO,
                side=order_side,
                submitted_quantity=quantity,
                submitted_price=price,
                time_in_force=TimeInForceType.Day
            )
            
            logger.info(f"✅ 实盘下单成功: {resp.order_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 下单失败: {e}")
            return False
    
    def get_positions(self) -> Dict[str, Dict]:
        """获取当前持仓"""
        if self.ctx is None:
            return {}
        
        try:
            resp = self.ctx.stock_positions()
            # 处理响应对象
            if hasattr(resp, 'channels') and resp.channels:
                positions = []
                for channel in resp.channels:
                    if hasattr(channel, 'positions'):
                        positions.extend(channel.positions)
                
                return {
                    p.symbol.replace(".US", ""): {
                        "quantity": int(p.quantity),
                        "market_value": float(p.market_value)
                    }
                    for p in positions
                }
        except Exception as e:
            logger.error(f"获取持仓失败: {e}")
        
        return {}


# ==========================
# 4. 风险管理层
# ==========================
class RiskManager:
    """风险管理器 - 熔断、回撤控制"""
    
    def __init__(self, config: V12Config):
        self.config = config
        self.max_equity = 0
        self.is_halted = False
        self.halt_start_date = None
        self.peak_equity_file = Path("data/peak_equity.txt")
        self.cooldown_file = Path("data/cooldown_until.txt")
        self._load_state()
    
    def _load_state(self):
        """加载状态"""
        # 加载历史最高资产
        if self.peak_equity_file.exists():
            try:
                with open(self.peak_equity_file, "r") as f:
                    self.max_equity = float(f.read().strip())
            except:
                self.max_equity = 0
        
        # 检查是否在冷却期
        if self.cooldown_file.exists():
            try:
                with open(self.cooldown_file, "r") as f:
                    cooldown_until = datetime.fromisoformat(f.read().strip())
                if datetime.now() < cooldown_until:
                    self.is_halted = True
                    self.halt_start_date = cooldown_until - timedelta(days=self.config.cooldown_days)
            except:
                pass
    
    def _save_state(self):
        """保存状态"""
        self.peak_equity_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.peak_equity_file, "w") as f:
            f.write(str(self.max_equity))
    
    def check_circuit_breaker(self, current_equity: float) -> Tuple[bool, str]:
        """
        检查熔断状态
        返回: (是否熔断, 状态信息)
        """
        # 更新历史最高
        if current_equity > self.max_equity:
            self.max_equity = current_equity
            self._save_state()
        
        # 检查冷却期
        if self.is_halted and self.halt_start_date:
            # 统一转换为date类型
            if isinstance(self.halt_start_date, datetime):
                halt_date = self.halt_start_date.date()
            else:
                halt_date = self.halt_start_date
            days_since = (datetime.now().date() - halt_date).days
            if days_since >= self.config.cooldown_days:
                self.is_halted = False
                self.halt_start_date = None
                self.max_equity = current_equity  # 重置峰值
                self.cooldown_file.unlink(missing_ok=True)
                logger.info(f">>> 冷却结束（{days_since}天），重启系统。")
                return False, "冷却结束"
            else:
                return True, f"冷却中（{days_since}/{self.config.cooldown_days}天）"
        
        # 计算回撤
        drawdown = (current_equity - self.max_equity) / self.max_equity
        
        # 检查是否触发熔断
        if drawdown < -self.config.max_dd_limit:
            self.is_halted = True
            self.halt_start_date = datetime.now().date()
            
            # 保存冷却期
            cooldown_until = datetime.now() + timedelta(days=self.config.cooldown_days)
            with open(self.cooldown_file, "w") as f:
                f.write(cooldown_until.isoformat())
            
            logger.error(f"\n{'!'*60}")
            logger.error(f"💀 触发熔断！回撤 {drawdown:.2%} > {self.config.max_dd_limit:.0%}")
            logger.error(f"🛑 强制清仓并进入 {self.config.cooldown_days} 天冷却期")
            logger.error(f"{'!'*60}\n")
            
            return True, f"触发熔断：回撤 {drawdown:.2%}"
        
        return False, f"正常运行（回撤: {drawdown:.2%}）"
    
    def should_clear_all(self) -> bool:
        """是否应该清仓所有持仓"""
        return self.is_halted


# ==========================
# 5. 主控层
# ==========================
class TradingSystem:
    """交易系统主控"""
    
    def __init__(
        self,
        tickers: List[str],
        paper_trading: bool = True,
        check_interval: int = 60
    ):
        self.tickers = tickers
        self.paper_trading = paper_trading
        self.check_interval = check_interval
        self.running = False
        
        # 初始化各层
        self.config = V12Config()
        self.data_collector = DataCollector()
        self.strategy = StrategyEngine(self.config)
        self.executor = TradeExecutor(paper_trading)
        self.risk_manager = RiskManager(self.config)
        
        # 状态跟踪
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[TradeSignal] = []
        self.last_selected_tickers: List[str] = None  # 上次选中的股票列表
        self.pending_stops: Dict[str, datetime] = {}  # 已触发止损但尚未执行的持仓
        
        # 信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """处理退出信号"""
        logger.info("\n🛑 收到停止信号，正在关闭系统...")
        self.running = False
    
    def update_positions(self):
        """更新持仓状态"""
        # 从交易执行器获取持仓
        api_positions = self.executor.get_positions()
        
        # 更新本地持仓
        for ticker, data in api_positions.items():
            if ticker not in self.positions:
                self.positions[ticker] = Position(
                    ticker=ticker,
                    quantity=data["quantity"],
                    avg_price=data.get("avg_price", 0),
                    entry_time=datetime.now()
                )
            else:
                self.positions[ticker].quantity = data["quantity"]
    
    def get_total_equity(self) -> float:
        """获取总资产"""
        account = self.executor.get_account_info()
        cash = account["cash"]
        
        # 计算持仓市值
        positions_value = 0
        for ticker, pos in self.positions.items():
            current_price = self.data_collector.fetch_realtime_price(ticker)
            if current_price:
                positions_value += pos.quantity * current_price
        
        return cash + positions_value
    
    def run(self):
        """主运行循环"""
        logger.info(f"\n{'='*80}")
        logger.info("🚀 V12集成实时交易系统启动")
        logger.info(f"{'='*80}")
        logger.info(f"监控股票: {', '.join(self.tickers)}")
        logger.info(f"交易模式: {'模拟交易' if self.paper_trading else '实盘交易'}")
        logger.info(f"检查间隔: {self.check_interval}秒")
        logger.info(f"交易时间: 美股开盘时段 (9:30-16:00 ET, 周一至周五)")
        logger.info(f"按 Ctrl+C 停止\n")
        
        self.running = True
        last_market_status = None
        
        while self.running:
            try:
                now = datetime.now()
                
                # 检查是否为美股开盘时间
                if not is_us_market_open():
                    market_status = get_market_status()
                    # 只在状态变化时打印日志，减少日志量
                    if market_status != last_market_status:
                        logger.info(f"⏸️  市场关闭 - {market_status}，等待开盘...")
                        last_market_status = market_status
                    time.sleep(self.check_interval)
                    continue
                
                # 市场开盘，重置状态记录
                if last_market_status != "开盘":
                    logger.info(f"\n{'='*80}")
                    logger.info(f"📈 美股开盘！开始交易监控")
                    logger.info(f"{'='*80}")
                    last_market_status = "开盘"
                
                # 更新持仓
                self.update_positions()
                
                # 获取总资产
                total_equity = self.get_total_equity()
                
                # 检查熔断
                is_halted, status = self.risk_manager.check_circuit_breaker(total_equity)
                logger.info(f"\n💰 总资产: ${total_equity:,.2f} | 峰值: ${self.risk_manager.max_equity:,.2f} | 状态: {status}")
                
                if is_halted:
                    # 熔断模式：清仓所有
                    logger.warning("⚠️ 熔断模式：清仓所有持仓")
                    for ticker in list(self.positions.keys()):
                        current_price = self.data_collector.fetch_realtime_price(ticker)
                        if current_price:
                            signal = TradeSignal(
                                ticker=ticker,
                                action="SELL",
                                quantity=self.positions[ticker].quantity,
                                price=current_price,
                                reason="熔断保护"
                            )
                            if self.executor.execute_signal(signal):
                                del self.positions[ticker]
                                self.trade_history.append(signal)
                
                else:
                    # 正常模式：生成并执行交易信号
                    signals, current_selected, self.pending_stops = self.strategy.generate_signals(
                        self.tickers,
                        self.positions,
                        total_equity,
                        self.last_selected_tickers,
                        self.pending_stops
                    )
                    
                    # 更新上次选中的股票列表
                    self.last_selected_tickers = current_selected
                    
                    if signals:
                        logger.info(f"\n📊 生成 {len(signals)} 个交易信号")
                        for sig in signals:
                            if self.executor.execute_signal(sig):
                                if sig.action == "BUY":
                                    # 更新或创建持仓
                                    if sig.ticker in self.positions:
                                        pos = self.positions[sig.ticker]
                                        total_cost = pos.avg_price * pos.quantity + sig.price * sig.quantity
                                        total_qty = pos.quantity + sig.quantity
                                        pos.avg_price = total_cost / total_qty
                                        pos.quantity = total_qty
                                    else:
                                        self.positions[sig.ticker] = Position(
                                            ticker=sig.ticker,
                                            quantity=sig.quantity,
                                            avg_price=sig.price,
                                            entry_time=datetime.now()
                                        )
                                elif sig.action == "SELL":
                                    # 更新或删除持仓
                                    if sig.ticker in self.positions:
                                        self.positions[sig.ticker].quantity -= sig.quantity
                                        if self.positions[sig.ticker].quantity <= 0:
                                            del self.positions[sig.ticker]
                                
                                self.trade_history.append(sig)
                    else:
                        logger.info("📊 无交易信号")
                
                # 打印状态
                self._print_status()
                
                # 等待下一次检查
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                self.running = False
            except Exception as e:
                logger.error(f"❌ 运行错误: {e}")
                time.sleep(self.check_interval)
        
        self._print_summary()
    
    def _print_status(self):
        """打印当前状态"""
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 持仓状态 | {datetime.now().strftime('%H:%M:%S')}")
        logger.info(f"{'='*80}")
        
        if self.positions:
            for ticker, pos in self.positions.items():
                current_price = self.data_collector.fetch_realtime_price(ticker)
                if current_price:
                    pnl, pnl_pct = pos.unrealized_pnl(current_price)
                    logger.info(f"   {ticker}: {pos.quantity}股 | "
                               f"成本: ${pos.avg_price:.2f} | "
                               f"现价: ${current_price:.2f} | "
                               f"盈亏: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
                else:
                    logger.info(f"   {ticker}: {pos.quantity}股 | 成本: ${pos.avg_price:.2f}")
        else:
            logger.info("   无持仓")
        
        logger.info(f"{'='*80}\n")
    
    def _print_summary(self):
        """打印交易总结"""
        logger.info(f"\n{'='*80}")
        logger.info("📊 交易总结")
        logger.info(f"{'='*80}")
        
        if self.trade_history:
            logger.info(f"\n交易记录 ({len(self.trade_history)}笔):")
            for sig in self.trade_history:
                pnl_str = ""
                if sig.action == "SELL":
                    # 查找对应的买入记录计算盈亏
                    for buy_sig in self.trade_history:
                        if (buy_sig.ticker == sig.ticker and 
                            buy_sig.action == "BUY" and 
                            buy_sig.timestamp < sig.timestamp):
                            pnl = (sig.price - buy_sig.price) * sig.quantity
                            pnl_pct = (sig.price - buy_sig.price) / buy_sig.price * 100
                            pnl_str = f" | 盈亏: ${pnl:,.2f} ({pnl_pct:+.2f}%)"
                            break
                
                logger.info(f"   {sig.timestamp.strftime('%m-%d %H:%M')} | "
                           f"{sig.ticker} | {sig.action} | "
                           f"{sig.quantity}股 @ ${sig.price:.2f}{pnl_str}")
        else:
            logger.info("\n无交易记录")
        
        logger.info(f"{'='*80}\n")


# ==========================
# 主函数
# ==========================
def main():
    ap = argparse.ArgumentParser(description="V12集成实时交易系统")
    ap.add_argument("--tickers", nargs="*",
                    default=['AAPL', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'MU', 'WMT', 'VRT', 'RKLB'],
                    help="监控的股票代码列表")
    ap.add_argument("--interval", type=int, default=60,
                    help="检查间隔（秒），默认60秒")
    ap.add_argument("--live", action="store_true",
                    help="启用实盘交易（默认模拟）")
    args = ap.parse_args()
    
    if args.live and not LONGBRIDGE_AVAILABLE:
        logger.error("❌ 长桥SDK未安装，无法启用实盘交易")
        return
    
    # 创建并运行系统
    system = TradingSystem(
        tickers=args.tickers,
        paper_trading=not args.live,
        check_interval=args.interval
    )
    
    system.run()


if __name__ == "__main__":
    main()
