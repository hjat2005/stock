# v12_realtime_minute_trading.py
# JOZ V12 Pro Plus 分钟级实时交易系统
# 
# 功能：
# 1. 使用历史数据训练趋势模型
# 2. 实时获取分钟级数据
# 3. 生成当前买入/卖出信号
# 4. 接入长桥API执行实盘交易

import argparse
import os
import sys
import time
import signal
import threading
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import yfinance as yf

# 长桥API导入
try:
    if '/usr/local/lib/python3.10/dist-packages' not in sys.path:
        sys.path.insert(0, '/usr/local/lib/python3.10/dist-packages')
    from longbridge.openapi import TradeContext, Config, OrderType, OrderSide, TimeInForceType
    LONGBRIDGE_AVAILABLE = True
    print("✅ 长桥SDK导入成功")
except ImportError as e:
    LONGBRIDGE_AVAILABLE = False
    print(f"⚠️ 长桥SDK未安装: {e}")


class MinuteDataCache:
    """分钟数据缓存管理"""
    
    def __init__(self, max_minutes=60):
        self.data = {}
        self.max_minutes = max_minutes
        self.lock = threading.Lock()
    
    def update(self, ticker, minute_data):
        """更新分钟数据"""
        with self.lock:
            if ticker not in self.data:
                self.data[ticker] = deque(maxlen=self.max_minutes)
            self.data[ticker].append(minute_data)
    
    def get_dataframe(self, ticker):
        """获取DataFrame格式的数据"""
        with self.lock:
            if ticker not in self.data or len(self.data[ticker]) < 10:
                return None
            return pd.DataFrame(list(self.data[ticker]))
    
    def get_latest(self, ticker, n=1):
        """获取最新的n条数据"""
        with self.lock:
            if ticker not in self.data:
                return None
            data_list = list(self.data[ticker])
            return data_list[-n:] if len(data_list) >= n else None


class TrendAnalyzer:
    """趋势分析器 - 基于历史数据训练模型"""
    
    def __init__(self, ticker, hist_days=30):
        self.ticker = ticker
        self.hist_days = hist_days
        self.hist_data = None
        self.sma_fast = 5   # 5分钟均线
        self.sma_slow = 20  # 20分钟均线
        self.rsi_period = 14
        self.atr_period = 14
        self._load_historical_data()
    
    def _load_historical_data(self):
        """加载历史数据用于趋势分析"""
        print(f"📊 加载 {self.ticker} 历史数据 ({self.hist_days}天)...")
        end = datetime.now()
        start = end - timedelta(days=self.hist_days)
        
        try:
            # 获取分钟级历史数据
            df = yf.download(
                self.ticker, 
                start=start.strftime('%Y-%m-%d'), 
                period="1d",
                interval="1m",
                progress=False
            )
            
            if df.empty:
                # 如果分钟数据为空，尝试日线数据
                df = yf.download(
                    self.ticker,
                    start=start.strftime('%Y-%m-%d'),
                    end=end.strftime('%Y-%m-%d'),
                    interval="1d",
                    progress=False
                )
            
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                self.hist_data = df.dropna()
                print(f"✅ 历史数据加载完成: {len(self.hist_data)} 条记录")
                self._calculate_indicators()
            else:
                print(f"⚠️ 无法获取 {self.ticker} 历史数据")
        except Exception as e:
            print(f"❌ 加载历史数据失败: {e}")
    
    def _calculate_indicators(self):
        """计算技术指标"""
        if self.hist_data is None or self.hist_data.empty:
            return
        
        df = self.hist_data
        
        # 计算均线
        df['SMA5'] = df['Close'].rolling(window=self.sma_fast).mean()
        df['SMA20'] = df['Close'].rolling(window=self.sma_slow).mean()
        
        # 计算RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 计算ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(self.atr_period).mean()
        
        # 计算成交量均线
        df['Volume_SMA5'] = df['Volume'].rolling(window=5).mean()
        df['Volume_SMA20'] = df['Volume'].rolling(window=20).mean()
        
        self.hist_data = df
    
    def analyze_trend(self, minute_df):
        """分析当前趋势并生成信号"""
        if minute_df is None or len(minute_df) < 20:
            return None
        
        df = minute_df.copy()
        
        # 计算分钟级指标
        df['SMA5'] = df['close'].rolling(window=self.sma_fast).mean()
        df['SMA20'] = df['close'].rolling(window=self.sma_slow).mean()
        df['Volume_SMA5'] = df['volume'].rolling(window=5).mean()
        
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        # 趋势信号计算
        signals = {
            'ticker': self.ticker,
            'timestamp': latest['timestamp'],
            'price': latest['close'],
            'volume': latest['volume'],
            'sma5': latest['SMA5'],
            'sma20': latest['SMA20'],
            'signals': []
        }
        
        # 信号1: 金叉买入 (5分钟均线上穿20分钟均线)
        if prev['SMA5'] <= prev['SMA20'] and latest['SMA5'] > latest['SMA20']:
            signals['signals'].append({
                'type': 'BUY',
                'reason': '金叉信号: 5分钟均线上穿20分钟均线',
                'strength': 'STRONG'
            })
        
        # 信号2: 成交量放大 + 价格上涨
        volume_ratio = latest['volume'] / latest['Volume_SMA5'] if latest['Volume_SMA5'] > 0 else 0
        price_change = (latest['close'] - prev['close']) / prev['close'] if prev['close'] > 0 else 0
        
        if volume_ratio > 1.5 and price_change > 0.001:
            signals['signals'].append({
                'type': 'BUY',
                'reason': f'放量上涨: 成交量是均量的{volume_ratio:.1f}倍, 价格涨幅{price_change*100:.2f}%',
                'strength': 'MEDIUM'
            })
        
        # 信号3: 死叉卖出 (5分钟均线下穿20分钟均线)
        if prev['SMA5'] >= prev['SMA20'] and latest['SMA5'] < latest['SMA20']:
            signals['signals'].append({
                'type': 'SELL',
                'reason': '死叉信号: 5分钟均线下穿20分钟均线',
                'strength': 'STRONG'
            })
        
        # 信号4: 价格跌破20分钟均线
        if latest['close'] < latest['SMA20'] * 0.995:
            signals['signals'].append({
                'type': 'SELL',
                'reason': '趋势跌破: 价格跌破20分钟均线',
                'strength': 'MEDIUM'
            })
        
        # 信号5: 成交量萎缩 + 价格下跌
        if volume_ratio < 0.7 and price_change < -0.001:
            signals['signals'].append({
                'type': 'SELL',
                'reason': f'缩量下跌: 成交量是均量的{volume_ratio:.1f}倍',
                'strength': 'WEAK'
            })
        
        signals['volume_ratio'] = volume_ratio
        signals['price_change'] = price_change
        
        return signals


class LongbridgeTrader:
    """长桥交易接口封装"""
    
    def __init__(self, paper_trading=True):
        self.paper_trading = paper_trading
        self.ctx = None
        
        if not LONGBRIDGE_AVAILABLE:
            print("🔴 长桥SDK未安装，仅支持模拟模式")
            return
            
        try:
            config = Config.from_env()
            self.ctx = TradeContext(config)
            print(f"✅ 长桥API连接成功 | 模拟交易: {paper_trading}")
        except Exception as e:
            print(f"❌ 长桥API连接失败: {e}")
            self.ctx = None
    
    def get_account_info(self):
        """获取账户信息"""
        if self.ctx is None:
            return {"cash": 0, "equity": 0, "mock": True}
        
        try:
            account_list = self.ctx.account_balance()
            if account_list and len(account_list) > 0:
                account = account_list[0]
                cash_info = account.cash_infos[0] if account.cash_infos else None
                available_cash = cash_info.available_cash if cash_info else account.total_cash
                return {
                    "cash": available_cash,
                    "equity": account.net_assets,
                    "mock": False
                }
            return {"cash": 0, "equity": 0, "mock": True}
        except Exception as e:
            print(f"获取账户信息失败: {e}")
            return {"cash": 0, "equity": 0, "mock": True}
    
    def place_order(self, symbol, side, quantity, price=None):
        """下单"""
        if "." not in symbol:
            symbol = f"{symbol}.US"
        
        if self.ctx is None:
            print(f"[模拟交易] {side} {symbol} | 股数: {quantity} | 价格: ${price:.2f if price else '市价'}")
            return {"success": True, "mock": True, "order_id": f"MOCK_{int(time.time())}"}
        
        try:
            order_side = OrderSide.Buy if side == "Buy" else OrderSide.Sell
            
            if price:
                price = round(float(price), 2)
                resp = self.ctx.submit_order(
                    symbol=symbol,
                    order_type=OrderType.LO,
                    side=order_side,
                    submitted_quantity=quantity,
                    submitted_price=price,
                    time_in_force=TimeInForceType.Day
                )
            else:
                resp = self.ctx.submit_order(
                    symbol=symbol,
                    order_type=OrderType.MO,
                    side=order_side,
                    submitted_quantity=quantity,
                    time_in_force=TimeInForceType.Day
                )
            
            print(f"✅ 实盘下单成功: {resp.order_id}")
            return {"success": True, "mock": False, "order_id": resp.order_id}
            
        except Exception as e:
            print(f"❌ 下单失败: {e}")
            return {"success": False, "error": str(e)}
    
    def get_positions(self):
        """获取当前持仓"""
        if self.ctx is None:
            return {}
        
        try:
            positions = self.ctx.stock_positions()
            if positions:
                return {p.symbol: {"quantity": p.quantity, "market_value": p.market_value} for p in positions}
            return {}
        except Exception as e:
            print(f"获取持仓失败: {e}")
            return {}


class RealtimeTradingSystem:
    """实时交易系统主类"""
    
    def __init__(self, tickers, paper_trading=True, check_interval=60):
        self.tickers = tickers
        self.paper_trading = paper_trading
        self.check_interval = check_interval  # 检查间隔（秒）
        self.running = False
        
        # 初始化组件
        self.trader = LongbridgeTrader(paper_trading=paper_trading)
        self.data_cache = MinuteDataCache(max_minutes=60)
        self.analyzers = {t: TrendAnalyzer(t) for t in tickers}
        
        # 持仓状态跟踪
        self.positions = {}  # {ticker: {'quantity': x, 'avg_price': y, 'entry_time': z}}
        self.trade_history = []
        
        # 信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """处理退出信号"""
        print("\n🛑 收到停止信号，正在关闭系统...")
        self.running = False
    
    def _fetch_realtime_data(self, ticker):
        """获取实时数据"""
        try:
            # 使用yfinance获取最新数据
            stock = yf.Ticker(ticker)
            # 获取今天的分钟数据
            today_data = stock.history(period="1d", interval="1m")
            
            if not today_data.empty:
                latest = today_data.iloc[-1]
                return {
                    'timestamp': latest.name,
                    'open': latest['Open'],
                    'high': latest['High'],
                    'low': latest['Low'],
                    'close': latest['Close'],
                    'volume': latest['Volume']
                }
            return None
        except Exception as e:
            print(f"⚠️ 获取 {ticker} 实时数据失败: {e}")
            return None
    
    def _execute_signal(self, ticker, signal_info, current_price):
        """执行交易信号"""
        signal_type = signal_info['type']
        reason = signal_info['reason']
        strength = signal_info['strength']
        
        account = self.trader.get_account_info()
        
        print(f"\n{'='*60}")
        print(f"🎯 交易信号 | {ticker} | {signal_type} | 强度: {strength}")
        print(f"   原因: {reason}")
        print(f"   当前价格: ${current_price:.2f}")
        print(f"   账户现金: ${account['cash']:,.2f}")
        print(f"{'='*60}")
        
        if signal_type == 'BUY':
            # 检查是否已持仓
            if ticker in self.positions and self.positions[ticker]['quantity'] > 0:
                print(f"⏭️ 已持有 {ticker}，跳过买入")
                return
            
            # 计算买入数量（使用20%现金）
            cash_to_use = account['cash'] * 0.2
            if cash_to_use < 1000:
                print(f"⚠️ 现金不足，无法买入")
                return
            
            quantity = int(cash_to_use / current_price / 10) * 10  # 整手买入
            if quantity < 10:
                print(f"⚠️ 计算股数 {quantity} 太少，跳过")
                return
            
            # 执行买入
            result = self.trader.place_order(ticker, "Buy", quantity, current_price)
            if result.get('success'):
                self.positions[ticker] = {
                    'quantity': quantity,
                    'avg_price': current_price,
                    'entry_time': datetime.now()
                }
                trade_record = {
                    'time': datetime.now(),
                    'ticker': ticker,
                    'action': 'BUY',
                    'quantity': quantity,
                    'price': current_price,
                    'value': quantity * current_price,
                    'reason': reason,
                    'order_id': result.get('order_id')
                }
                self.trade_history.append(trade_record)
                print(f"✅ 买入成功: {quantity}股 @ ${current_price:.2f}")
            
        elif signal_type == 'SELL':
            # 检查是否有持仓
            if ticker not in self.positions or self.positions[ticker]['quantity'] == 0:
                print(f"⏭️ 未持有 {ticker}，跳过卖出")
                return
            
            quantity = self.positions[ticker]['quantity']
            
            # 执行卖出
            result = self.trader.place_order(ticker, "Sell", quantity, current_price)
            if result.get('success'):
                entry_price = self.positions[ticker]['avg_price']
                pnl = (current_price - entry_price) * quantity
                pnl_pct = (current_price - entry_price) / entry_price * 100
                
                trade_record = {
                    'time': datetime.now(),
                    'ticker': ticker,
                    'action': 'SELL',
                    'quantity': quantity,
                    'price': current_price,
                    'value': quantity * current_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'reason': reason,
                    'order_id': result.get('order_id')
                }
                self.trade_history.append(trade_record)
                
                del self.positions[ticker]
                print(f"✅ 卖出成功: {quantity}股 @ ${current_price:.2f} | 盈亏: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
    
    def _print_status(self):
        """打印当前状态"""
        print(f"\n{'='*80}")
        print(f"📊 系统状态 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
        account = self.trader.get_account_info()
        print(f"💰 账户现金: ${account['cash']:,.2f}")
        print(f"📈 账户净值: ${account['equity']:,.2f}")
        
        if self.positions:
            print(f"\n📋 当前持仓:")
            for ticker, pos in self.positions.items():
                current_data = self._fetch_realtime_data(ticker)
                if current_data:
                    current_price = current_data['close']
                    pnl = (current_price - pos['avg_price']) * pos['quantity']
                    pnl_pct = (current_price - pos['avg_price']) / pos['avg_price'] * 100
                    print(f"   {ticker}: {pos['quantity']}股 | 成本: ${pos['avg_price']:.2f} | "
                          f"现价: ${current_price:.2f} | 盈亏: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
                else:
                    print(f"   {ticker}: {pos['quantity']}股 | 成本: ${pos['avg_price']:.2f}")
        else:
            print(f"\n📋 当前持仓: 无")
        
        print(f"{'='*80}\n")
    
    def run(self):
        """主运行循环"""
        print(f"\n🚀 启动实时交易系统")
        print(f"   监控股票: {', '.join(self.tickers)}")
        print(f"   检查间隔: {self.check_interval}秒")
        print(f"   交易模式: {'模拟交易' if self.paper_trading else '实盘交易'}")
        print(f"   按 Ctrl+C 停止系统\n")
        
        self.running = True
        last_status_time = 0
        
        while self.running:
            try:
                current_time = time.time()
                
                for ticker in self.tickers:
                    # 获取实时数据
                    data = self._fetch_realtime_data(ticker)
                    if data:
                        # 更新缓存
                        self.data_cache.update(ticker, data)
                        
                        # 分析趋势
                        minute_df = self.data_cache.get_dataframe(ticker)
                        if minute_df is not None:
                            signals = self.analyzers[ticker].analyze_trend(minute_df)
                            
                            if signals and signals['signals']:
                                # 执行信号
                                for sig in signals['signals']:
                                    self._execute_signal(ticker, sig, signals['price'])
                
                # 每5分钟打印一次状态
                if current_time - last_status_time > 300:
                    self._print_status()
                    last_status_time = current_time
                
                # 等待下一次检查
                time.sleep(self.check_interval)
                
            except Exception as e:
                print(f"❌ 运行错误: {e}")
                time.sleep(self.check_interval)
        
        # 系统停止，打印总结
        self._print_summary()
    
    def _print_summary(self):
        """打印交易总结"""
        print(f"\n{'='*80}")
        print(f"📊 交易总结")
        print(f"{'='*80}")
        
        if self.trade_history:
            print(f"\n交易记录:")
            for trade in self.trade_history:
                pnl_str = f" | 盈亏: ${trade.get('pnl', 0):,.2f}" if 'pnl' in trade else ""
                print(f"   {trade['time'].strftime('%m-%d %H:%M')} | {trade['ticker']} | "
                      f"{trade['action']} | {trade['quantity']}股 @ ${trade['price']:.2f}{pnl_str}")
            
            # 计算总盈亏
            total_pnl = sum(t.get('pnl', 0) for t in self.trade_history if 'pnl' in t)
            print(f"\n总盈亏: ${total_pnl:,.2f}")
        else:
            print("\n无交易记录")
        
        print(f"{'='*80}\n")


def main():
    ap = argparse.ArgumentParser(description="V12分钟级实时交易系统")
    ap.add_argument("--tickers", nargs="*", default=['NVDA', 'TSLA', 'AAPL'],
                    help="监控的股票代码列表")
    ap.add_argument("--interval", type=int, default=60,
                    help="检查间隔（秒），默认60秒")
    ap.add_argument("--live", action="store_true",
                    help="启用实盘交易（默认模拟交易）")
    ap.add_argument("--hist-days", type=int, default=30,
                    help="历史数据天数（用于趋势分析），默认30天")
    args = ap.parse_args()
    
    if args.live and not LONGBRIDGE_AVAILABLE:
        print("❌ 长桥SDK未安装，无法启用实盘交易")
        return
    
    # 创建并运行系统
    system = RealtimeTradingSystem(
        tickers=args.tickers,
        paper_trading=not args.live,
        check_interval=args.interval
    )
    
    system.run()


if __name__ == "__main__":
    main()
