import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import sys

class StockMonitor:
    def __init__(self, tickers, check_interval=30):
        """
        多股票实时监控系统
        tickers: 股票代码列表，如 ["CRWV", "NVDA", "TSLA"]
        check_interval: 检查间隔（秒），默认30秒
        """
        self.tickers = tickers if isinstance(tickers, list) else [tickers]
        self.check_interval = check_interval
        self.alert_sent = {ticker: False for ticker in self.tickers}
        # 存储历史数据用于对比
        self.history_data = {ticker: {
            'prev_volume': None,
            'prev_price': None,
            'prev_time': None,
            'volume_30s_ago': None,
            'price_30s_ago': None
        } for ticker in self.tickers}
        
    def fetch_intraday_data(self, ticker):
        """获取当日分时数据用于计算实时成交量"""
        try:
            stock = yf.Ticker(ticker)
            intraday = stock.history(period="1d", interval="1m")
            if intraday.empty:
                return None
            return intraday
        except Exception as e:
            return None
    
    def fetch_daily_data(self, ticker):
        """获取日线数据用于计算均线"""
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="35d", interval="1d")
            
            if df.empty:
                return None
                
            # 计算 5 周均线 (25日均线)
            df['MA25'] = df['Close'].rolling(window=25).mean()
            
            return df
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 获取 {ticker} 数据失败: {e}")
            return None
    
    def check_buy_signal(self, df, ticker):
        """
        检查买入信号:
        1. 收盘价上穿 5 周均线 (25日均线)
        2. 连续 3 日成交量递增
        """
        if len(df) < 5:
            return False, "数据不足"
            
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # 条件1: 价格上穿 5 周均线
        price_above_ma = latest['Close'] > latest['MA25']
        prev_below_ma = prev['Close'] <= prev['MA25']
        golden_cross = price_above_ma and prev_below_ma
        
        # 条件2: 连续 3 日成交量递增
        if len(df) >= 4:
            vol_today = df.iloc[-1]['Volume']
            vol_1d = df.iloc[-2]['Volume']
            vol_2d = df.iloc[-3]['Volume']
            vol_3d = df.iloc[-4]['Volume']
            
            vol_increasing = (vol_today > vol_1d) and (vol_1d > vol_2d) and (vol_2d > vol_3d)
            
            vol_info = {
                'today': vol_today,
                'd1': vol_1d,
                'd2': vol_2d,
                'd3': vol_3d,
                'increasing': vol_increasing
            }
        else:
            vol_info = {'increasing': False}
        
        # 综合判断
        if golden_cross and vol_info['increasing']:
            return True, "🟢 买入信号触发！"
        elif golden_cross:
            return False, f"⚠️ 价格上穿5周线，但成交量未连续递增"
        else:
            position = "上方" if latest['Close'] > latest['MA25'] else "下方"
            return False, f"价格在5周线{position}"
    
    def calculate_volume_change(self, current_vol, prev_vol):
        """计算成交量变化量和变化率"""
        if prev_vol is None or prev_vol == 0:
            return 0, 0
        change = current_vol - prev_vol
        change_pct = (change / prev_vol) * 100
        return change, change_pct
    
    def display_single_stock(self, ticker, df_daily, df_intraday, signal_msg, iteration):
        """显示单只股票状态"""
        latest = df_daily.iloc[-1]
        prev = df_daily.iloc[-2] if len(df_daily) > 1 else latest
        
        current_price = latest['Close']
        current_volume = latest['Volume']
        ma25 = latest['MA25']
        
        # 计算涨跌幅（相对昨日）
        price_change = current_price - prev['Close']
        price_change_pct = (price_change / prev['Close']) * 100 if prev['Close'] != 0 else 0
        
        # 计算距离5周线的距离
        distance_to_ma = ((current_price - ma25) / ma25) * 100
        
        # 获取历史数据用于对比
        hist = self.history_data[ticker]
        
        # 计算30秒内的变化
        if hist['prev_volume'] is not None:
            vol_change_30s = current_volume - hist['prev_volume']
            vol_change_30s_pct = (vol_change_30s / hist['prev_volume']) * 100 if hist['prev_volume'] > 0 else 0
            price_change_30s = current_price - hist['prev_price']
        else:
            vol_change_30s = 0
            vol_change_30s_pct = 0
            price_change_30s = 0
        
        # 价格信息 - 显示前30s和当前对比
        change_symbol = "📈" if price_change >= 0 else "📉"
        print(f"\n{change_symbol} {ticker} | 当前: ${current_price:.2f}", end="")
        if hist['prev_price'] is not None:
            price_arrow = "↑" if price_change_30s >= 0 else "↓"
            print(f" | 前30s: ${hist['prev_price']:.2f} {price_arrow} ${abs(price_change_30s):.2f}", end="")
        print(f" | 日涨跌: {price_change_pct:+.2f}%")
        
        # 5周均线信息
        ma_symbol = "✅" if current_price > ma25 else "❌"
        print(f"   {ma_symbol} MA25: ${ma25:.2f} ({distance_to_ma:+.2f}%)")
        
        # 成交量信息 - 显示前30s和当前对比
        vol_arrow = "↑" if vol_change_30s >= 0 else "↓"
        vol_symbol = "📈" if vol_change_30s_pct > 5 else "📉" if vol_change_30s_pct < -5 else "➡️"
        
        if hist['prev_volume'] is not None:
            print(f"   {vol_symbol} 成交量: {hist['prev_volume']:,.0f} -> {current_volume:,.0f}", end="")
            print(f" ({vol_arrow}{abs(vol_change_30s):,.0f}, {vol_change_30s_pct:+.1f}%)")
        else:
            print(f"   📊 成交量: {current_volume:,.0f}")
        
        # 实时成交量数据（日内累计）
        if df_intraday is not None and not df_intraday.empty:
            today_cumulative_vol = df_intraday['Volume'].sum()
            
            # 计算交易时间进度
            now = datetime.now()
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
            
            if now < market_open:
                time_elapsed_ratio = 0
            elif now > market_close:
                time_elapsed_ratio = 1
            else:
                total_trading_seconds = (market_close - market_open).total_seconds()
                elapsed_seconds = (now - market_open).total_seconds()
                time_elapsed_ratio = elapsed_seconds / total_trading_seconds
            
            # 预估全天成交量
            if time_elapsed_ratio > 0:
                estimated_full_day = today_cumulative_vol / time_elapsed_ratio
                print(f"   🔴 日内累计: {today_cumulative_vol:,.0f} | 预估全天: {estimated_full_day:,.0f}")
        
        # 信号状态
        print(f"   🔔 {signal_msg}")
        
        # 买入信号提醒
        if "买入信号触发" in signal_msg and not self.alert_sent[ticker]:
            print(f"   🚨🚨🚨 {ticker} 买入信号！🚨🚨🚨")
            self.alert_sent[ticker] = True
        elif "买入信号" not in signal_msg:
            self.alert_sent[ticker] = False
        
        # 更新历史数据（保存当前值作为下一次的前值）
        self.history_data[ticker]['prev_volume'] = current_volume
        self.history_data[ticker]['prev_price'] = current_price
        self.history_data[ticker]['prev_time'] = datetime.now()
    
    def display_status(self, all_data, iteration):
        """显示所有股票状态"""
        # 清屏（Unix/Linux）
        print("\033[2J\033[H")
        
        # 打印状态面板
        print("=" * 75)
        print(f"📊 多股票实时监控系统 | 监控: {', '.join(self.tickers)}")
        print(f"⏰ 更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 刷新次数: #{iteration}")
        print(f"📈 显示格式: [当前值] | [前30秒值] -> [变化量] | 日涨跌")
        print("=" * 75)
        
        for ticker, data in all_data.items():
            if data['daily'] is not None:
                self.display_single_stock(
                    ticker, 
                    data['daily'], 
                    data.get('intraday'), 
                    data['signal_msg'],
                    iteration
                )
        
        print("\n" + "=" * 75)
        print(f"⏱️  下次检查: {self.check_interval}秒后 (按 Ctrl+C 停止)")
        print("=" * 75)
    
    def run(self):
        """主循环"""
        print("🚀 启动多股票实时监控系统...")
        print(f"   监控股票: {', '.join(self.tickers)}")
        print(f"   检查间隔: {self.check_interval}秒")
        print(f"   显示格式: 当前值 | 前30秒值 -> 变化")
        print("\n正在获取初始数据...\n")
        
        iteration = 0
        
        try:
            while True:
                iteration += 1
                all_data = {}
                
                for ticker in self.tickers:
                    # 获取日线数据
                    df_daily = self.fetch_daily_data(ticker)
                    # 获取分时数据
                    df_intraday = self.fetch_intraday_data(ticker)
                    
                    if df_daily is not None and not df_daily.empty:
                        signal_triggered, signal_msg = self.check_buy_signal(df_daily, ticker)
                        all_data[ticker] = {
                            'daily': df_daily,
                            'intraday': df_intraday,
                            'signal_msg': signal_msg
                        }
                    else:
                        all_data[ticker] = {
                            'daily': None,
                            'intraday': None,
                            'signal_msg': "无法获取数据"
                        }
                
                self.display_status(all_data, iteration)
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            print("\n\n👋 监控系统已停止")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            sys.exit(1)


def quick_check(tickers):
    """快速检查当前状态（非监控模式）"""
    tickers = tickers if isinstance(tickers, list) else [tickers]
    
    print(f"🔍 快速检查 {len(tickers)} 只股票当前状态...\n")
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="35d", interval="1d")
            
            if df.empty:
                print(f"❌ {ticker}: 无法获取数据")
                continue
                
            # 计算 5 周均线
            df['MA25'] = df['Close'].rolling(window=25).mean()
            
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # 检查买入信号
            price_above_ma = latest['Close'] > latest['MA25']
            prev_below_ma = prev['Close'] <= prev['MA25']
            golden_cross = price_above_ma and prev_below_ma
            
            # 成交量检查
            vol_increasing = False
            if len(df) >= 4:
                vol_today = df.iloc[-1]['Volume']
                vol_1d = df.iloc[-2]['Volume']
                vol_2d = df.iloc[-3]['Volume']
                vol_3d = df.iloc[-4]['Volume']
                vol_increasing = (vol_today > vol_1d) and (vol_1d > vol_2d) and (vol_2d > vol_3d)
            
            print("=" * 60)
            print(f"📊 {ticker} | {df.index[-1].strftime('%Y-%m-%d')}")
            print("=" * 60)
            print(f"价格: ${latest['Close']:.2f} | MA25: ${latest['MA25']:.2f}")
            print(f"成交量: {latest['Volume']:,.0f}")
            
            if golden_cross and vol_increasing:
                print("🟢 买入信号: 价格上穿5周线 + 成交量连续3日递增")
            elif golden_cross:
                print("⚠️  价格上穿5周线，但成交量未连续递增")
            elif price_above_ma:
                print("📈 价格在5周线上方，持有观望")
            else:
                print("📉 价格在5周线下方，等待金叉")
            
            print()
            
        except Exception as e:
            print(f"❌ {ticker} 错误: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='多股票实时监控系统')
    parser.add_argument('--tickers', '-t', type=str, default='CRWV',
                        help='监控的股票代码，多个用逗号分隔，如 "CRWV,NVDA,TSLA"')
    parser.add_argument('--interval', '-i', type=int, default=30, 
                        help='检查间隔（秒），默认30秒')
    parser.add_argument('--quick', '-q', action='store_true',
                        help='快速检查当前状态（非监控模式）')
    
    args = parser.parse_args()
    
    # 解析股票代码
    tickers = [t.strip().upper() for t in args.tickers.split(',')]
    
    if args.quick:
        quick_check(tickers)
    else:
        monitor = StockMonitor(tickers=tickers, check_interval=args.interval)
        monitor.run()
