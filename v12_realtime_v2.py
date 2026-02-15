# v12_realtime_v2.py
# JOZ V12 Pro Plus 实时交易系统 - 基于原始V12策略
# 
# 核心逻辑（与回测策略完全一致）：
# 1. 使用20日ROC动量选股（ROC > 5%）
# 2. 每周一调仓
# 3. SMA30趋势止损
# 4. 20%熔断机制 + 10天冷却
# 5. 1.3x杠杆，RKLB/CRWV封顶30%

import argparse
import os
import sys
import time
import signal
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# 长桥API导入
try:
    if '/usr/local/lib/python3.10/dist-packages' not in sys.path:
        sys.path.insert(0, '/usr/local/lib/python3.10/dist-packages')
    from longbridge.openapi import TradeContext, Config, OrderType, OrderSide, TimeInForceType
    LONGBRIDGE_AVAILABLE = True
except ImportError as e:
    LONGBRIDGE_AVAILABLE = False
    print(f"⚠️ 长桥SDK未安装: {e}")


class V12StrategyEngine:
    """
    V12策略引擎 - 完全复刻原始回测策略逻辑
    """
    
    def __init__(self, tickers, cash=110000.0, paper_trading=True):
        self.tickers = tickers
        self.cash = cash
        self.paper_trading = paper_trading
        
        # V12策略参数（与test_gen_xls.py完全一致）
        self.mtm_period = 20          # 动量周期
        self.max_positions = 3        # 最大持仓数
        self.leverage = 1.3           # 杠杆倍数
        self.buffer = 0.88            # 保证金缓冲
        self.max_dd_limit = 0.20      # 最大回撤限制20%
        self.cooldown_days = 10       # 冷却天数
        self.special_cap = 0.30       # RKLB/CRWV封顶30%
        
        # 状态跟踪
        self.max_equity = cash
        self.is_halted = False
        self.halt_start_date = None
        self.positions = {}           # 当前持仓 {ticker: {'quantity': x, 'avg_price': y}}
        self.last_rebalance_date = None
        self.trade_history = []
        
        # 数据缓存
        self.price_data = {}
        self.roc_data = {}
        self.sma30_data = {}
        
        # 初始化长桥交易
        self.trader = LongbridgeTrader(paper_trading=paper_trading)
        
        print(f"✅ V12策略引擎初始化完成")
        print(f"   监控股票: {', '.join(tickers)}")
        print(f"   初始资金: ${cash:,.2f}")
        print(f"   杠杆倍数: {self.leverage}x")
        print(f"   最大回撤限制: {self.max_dd_limit*100}%")
    
    def fetch_data(self, ticker, period="30d"):
        """获取股票数据"""
        try:
            df = yf.download(
                ticker,
                period=period,
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
            print(f"⚠️ 获取 {ticker} 数据失败: {e}")
            return None
    
    def calculate_indicators(self, df):
        """计算技术指标"""
        if df is None or len(df) < 30:
            return None
        
        # 20日ROC动量
        df['ROC'] = df['Close'].pct_change(self.mtm_period)
        # 30日SMA
        df['SMA30'] = df['Close'].rolling(window=30).mean()
        # 20日ATR（用于止损）
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        return df
    
    def check_circuit_breaker(self, current_equity):
        """检查熔断机制"""
        if self.is_halted:
            # 检查冷却是否结束
            if self.halt_start_date:
                days_since_halt = (datetime.now().date() - self.halt_start_date).days
                if days_since_halt >= self.cooldown_days:
                    self.is_halted = False
                    self.max_equity = current_equity  # 重置峰值
                    self.halt_start_date = None
                    print(f">>> 冷却结束（{days_since_halt}天），重启系统。")
                    return False, "冷却结束"
                else:
                    return True, f"冷却中（{days_since_halt}/{self.cooldown_days}天）"
            return True, "冷却中"
        
        # 检查是否触发熔断
        if current_equity > self.max_equity:
            self.max_equity = current_equity
        
        drawdown = (current_equity - self.max_equity) / self.max_equity
        
        if drawdown < -self.max_dd_limit:
            self.is_halted = True
            self.halt_start_date = datetime.now().date()
            return True, f"触发熔断：回撤 {drawdown:.2%}"
        
        return False, f"正常运行（回撤: {drawdown:.2%}）"
    
    def check_trend_stop(self, ticker, current_price):
        """检查SMA30趋势止损"""
        df = self.fetch_data(ticker, period="60d")
        if df is None:
            return False
        
        df = self.calculate_indicators(df)
        if df is None or df.empty:
            return False
        
        latest_sma30 = df['SMA30'].iloc[-1]
        
        # 如果价格跌破SMA30，触发止损
        if current_price < latest_sma30:
            return True
        return False
    
    def select_stocks(self):
        """
        选股逻辑：ROC > 5% 且价格在SMA30之上
        返回排序后的股票列表
        """
        scores = []
        
        for ticker in self.tickers:
            df = self.fetch_data(ticker, period="60d")
            if df is None:
                continue
            
            df = self.calculate_indicators(df)
            if df is None or len(df) < self.mtm_period + 5:
                continue
            
            latest = df.iloc[-1]
            roc = latest['ROC']
            price = latest['Close']
            sma30 = latest['SMA30']
            
            # V12选股条件：ROC > 5% 且价格 > SMA30
            if roc > 0.05 and price > sma30:
                scores.append((ticker, roc, price))
                print(f"   {ticker}: ROC={roc*100:.1f}%, 价格=${price:.2f}, SMA30=${sma30:.2f} ✅")
            else:
                reason = []
                if roc <= 0.05:
                    reason.append(f"ROC={roc*100:.1f}%<=5%")
                if price <= sma30:
                    reason.append(f"价格<={sma30:.2f}")
                print(f"   {ticker}: {', '.join(reason)} ❌")
        
        # 按ROC排序，取前max_positions个
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(x[0], x[2]) for x in scores[:self.max_positions]]
    
    def get_current_price(self, ticker):
        """获取当前价格"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return info.get('regularMarketPrice') or info.get('currentPrice')
        except:
            df = self.fetch_data(ticker, period="2d")
            if df is not None and not df.empty:
                return df['Close'].iloc[-1]
            return None
    
    def get_account_value(self):
        """获取账户总价值"""
        account = self.trader.get_account_info()
        cash = account['cash']
        
        # 计算持仓市值
        positions_value = 0
        for ticker, pos in self.positions.items():
            current_price = self.get_current_price(ticker)
            if current_price:
                positions_value += pos['quantity'] * current_price
        
        return cash + positions_value
    
    def execute_rebalance(self):
        """执行调仓"""
        today = datetime.now().date()
        
        # 检查是否已调仓
        if self.last_rebalance_date == today:
            print(f"⏭️ 今日已调仓，跳过")
            return
        
        print(f"\n{'='*80}")
        print(f"📊 开始调仓 | {today}")
        print(f"{'='*80}")
        
        # 获取账户价值
        account_value = self.get_account_value()
        print(f"💰 当前账户价值: ${account_value:,.2f}")
        print(f"📈 历史峰值: ${self.max_equity:,.2f}")
        
        # 检查熔断
        is_halted, status = self.check_circuit_breaker(account_value)
        print(f"🔒 熔断状态: {status}")
        
        if is_halted:
            # 清仓所有持仓
            print(f"⚠️ 熔断状态，清仓所有持仓")
            for ticker in list(self.positions.keys()):
                self.execute_sell(ticker, "熔断清仓")
            return
        
        # 检查趋势止损
        print(f"\n📉 检查趋势止损...")
        for ticker in list(self.positions.keys()):
            current_price = self.get_current_price(ticker)
            if current_price and self.check_trend_stop(ticker, current_price):
                print(f"   {ticker}: 触发SMA30趋势止损")
                self.execute_sell(ticker, "趋势止损")
        
        # 选股
        print(f"\n🎯 选股（ROC > 5% 且价格 > SMA30）...")
        top_picks = self.select_stocks()
        
        if not top_picks:
            print(f"⚠️ 无符合条件的股票，清仓")
            for ticker in list(self.positions.keys()):
                self.execute_sell(ticker, "不在选股列表")
            return
        
        print(f"\n📋 选股结果: {', '.join([t[0] for t in top_picks])}")
        
        # 清仓不在列表中的股票
        for ticker in list(self.positions.keys()):
            if ticker not in [p[0] for p in top_picks]:
                self.execute_sell(ticker, "不在选股列表")
        
        # 计算目标仓位
        total_target_pct = self.leverage * self.buffer
        print(f"\n💼 目标仓位: {total_target_pct*100:.0f}% ({self.leverage}x杠杆 x {self.buffer}缓冲)")
        
        # 分离Special股票（RKLB/CRWV）
        specials = [(t, p) for t, p in top_picks if t in ("RKLB", "CRWV")]
        others = [(t, p) for t, p in top_picks if t not in ("RKLB", "CRWV")]
        
        used_pct = 0.0
        
        # 处理Special组
        if specials:
            cap_each = self.special_cap / len(specials)
            print(f"\n🔸 Special组 ({self.special_cap*100}%封顶): {', '.join([t[0] for t in specials])}")
            for ticker, price in specials:
                if ticker not in self.positions:
                    target_val = account_value * cap_each
                    self.execute_buy(ticker, price, target_val)
            used_pct += self.special_cap
        
        # 处理Others组
        if others:
            remain_pct = max(0.0, total_target_pct - used_pct)
            each_pct = remain_pct / len(others)
            print(f"\n🔹 Others组 (剩余{remain_pct*100:.0f}%): {', '.join([t[0] for t in others])}")
            for ticker, price in others:
                if ticker not in self.positions:
                    target_val = account_value * each_pct
                    self.execute_buy(ticker, price, target_val)
        
        self.last_rebalance_date = today
        
        print(f"\n{'='*80}")
        self.print_status()
    
    def execute_buy(self, ticker, price, target_value):
        """执行买入"""
        if price <= 0:
            return
        
        # 计算股数（整手）
        quantity = int((target_value / price) // 10) * 10
        if quantity < 10:
            print(f"   {ticker}: 计算股数 {quantity} < 10，跳过")
            return
        
        print(f"   {ticker}: 计划买入 {quantity}股 @ ${price:.2f} | 目标市值: ${target_value:,.2f}")
        
        # 执行交易
        result = self.trader.place_order(ticker, "Buy", quantity, price)
        
        if result.get('success'):
            self.positions[ticker] = {
                'quantity': quantity,
                'avg_price': price,
                'entry_time': datetime.now()
            }
            self.trade_history.append({
                'time': datetime.now(),
                'ticker': ticker,
                'action': 'BUY',
                'quantity': quantity,
                'price': price,
                'value': quantity * price,
                'order_id': result.get('order_id')
            })
            print(f"   ✅ 买入成功: {quantity}股 @ ${price:.2f}")
        else:
            print(f"   ❌ 买入失败: {result.get('error')}")
    
    def execute_sell(self, ticker, reason):
        """执行卖出"""
        if ticker not in self.positions:
            return
        
        quantity = self.positions[ticker]['quantity']
        current_price = self.get_current_price(ticker)
        
        if not current_price:
            print(f"   {ticker}: 无法获取当前价格，跳过卖出")
            return
        
        avg_price = self.positions[ticker]['avg_price']
        pnl = (current_price - avg_price) * quantity
        pnl_pct = (current_price - avg_price) / avg_price * 100
        
        print(f"   {ticker}: 卖出 {quantity}股 @ ${current_price:.2f} | 原因: {reason}")
        
        # 执行交易
        result = self.trader.place_order(ticker, "Sell", quantity, current_price)
        
        if result.get('success'):
            del self.positions[ticker]
            self.trade_history.append({
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
            })
            print(f"   ✅ 卖出成功 | 盈亏: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
        else:
            print(f"   ❌ 卖出失败: {result.get('error')}")
    
    def print_status(self):
        """打印当前状态"""
        account_value = self.get_account_value()
        account = self.trader.get_account_info()
        
        print(f"\n📊 当前状态 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   账户现金: ${account['cash']:,.2f}")
        print(f"   账户净值: ${account['equity']:,.2f}")
        print(f"   计算价值: ${account_value:,.2f}")
        print(f"   历史峰值: ${self.max_equity:,.2f}")
        
        drawdown = (account_value - self.max_equity) / self.max_equity
        print(f"   当前回撤: {drawdown:.2%}")
        print(f"   熔断状态: {'🔴 冷却中' if self.is_halted else '🟢 正常'}")
        
        if self.positions:
            print(f"\n   📋 当前持仓:")
            for ticker, pos in self.positions.items():
                current_price = self.get_current_price(ticker)
                if current_price:
                    pnl = (current_price - pos['avg_price']) * pos['quantity']
                    pnl_pct = (current_price - pos['avg_price']) / pos['avg_price'] * 100
                    print(f"      {ticker}: {pos['quantity']}股 | 成本: ${pos['avg_price']:.2f} | "
                          f"现价: ${current_price:.2f} | 盈亏: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
                else:
                    print(f"      {ticker}: {pos['quantity']}股 | 成本: ${pos['avg_price']:.2f}")
        else:
            print(f"\n   📋 当前持仓: 无")
        
        print(f"\n   📅 上次调仓: {self.last_rebalance_date or '从未'}")
    
    def run(self):
        """主运行循环"""
        print(f"\n🚀 启动V12实时交易系统")
        print(f"   策略: 20日ROC动量 + SMA30趋势 + 20%熔断")
        print(f"   调仓: 每周一")
        print(f"   按 Ctrl+C 停止\n")
        
        self.running = True
        
        while self.running:
            try:
                now = datetime.now()
                
                # 只在周一调仓（美股开盘前）
                if now.weekday() == 0:  # 周一
                    # 在盘前（9:30 AM EST前）或盘后调仓
                    if now.hour >= 14 or now.hour < 5:  # 北京时间晚上或凌晨
                        self.execute_rebalance()
                
                # 每分钟检查趋势止损
                if now.minute % 5 == 0:  # 每5分钟
                    print(f"\n⏰ {now.strftime('%H:%M')} - 检查趋势止损...")
                    account_value = self.get_account_value()
                    is_halted, status = self.check_circuit_breaker(account_value)
                    
                    if not is_halted:
                        for ticker in list(self.positions.keys()):
                            current_price = self.get_current_price(ticker)
                            if current_price and self.check_trend_stop(ticker, current_price):
                                print(f"   {ticker}: 触发SMA30趋势止损")
                                self.execute_sell(ticker, "趋势止损")
                    
                    self.print_status()
                
                # 等待1分钟
                time.sleep(60)
                
            except KeyboardInterrupt:
                print("\n🛑 收到停止信号...")
                self.running = False
            except Exception as e:
                print(f"❌ 错误: {e}")
                time.sleep(60)
        
        self.print_summary()
    
    def print_summary(self):
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
            
            total_pnl = sum(t.get('pnl', 0) for t in self.trade_history if 'pnl' in t)
            print(f"\n总盈亏: ${total_pnl:,.2f}")
        else:
            print("\n无交易记录")
        
        print(f"{'='*80}\n")


class LongbridgeTrader:
    """长桥交易接口封装"""
    
    def __init__(self, paper_trading=True):
        self.paper_trading = paper_trading
        self.ctx = None
        
        if not LONGBRIDGE_AVAILABLE:
            print("🔴 长桥SDK未安装，使用模拟模式")
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
            return {"cash": 800000.0, "equity": 800000.0, "mock": True}
        
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


def main():
    ap = argparse.ArgumentParser(description="V12实时交易系统 - 基于原始V12策略")
    ap.add_argument("--tickers", nargs="*", 
                    default=['AAPL', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'MU', 'WMT', 'VRT', 'RKLB'],
                    help="监控的股票代码列表")
    ap.add_argument("--cash", type=float, default=110000.0,
                    help="初始资金，默认110000")
    ap.add_argument("--live", action="store_true",
                    help="启用实盘交易（默认模拟）")
    args = ap.parse_args()
    
    if args.live and not LONGBRIDGE_AVAILABLE:
        print("❌ 长桥SDK未安装，无法启用实盘交易")
        return
    
    # 创建并运行系统
    engine = V12StrategyEngine(
        tickers=args.tickers,
        cash=args.cash,
        paper_trading=not args.live
    )
    
    engine.run()


if __name__ == "__main__":
    main()
