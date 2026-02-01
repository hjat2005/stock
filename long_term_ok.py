# -*- coding: utf-8 -*-
import datetime as dt
import backtrader as bt
import pandas as pd
import yfinance as yf
import math

class Mag7ProOptimal(bt.Strategy):
    """
    【美股M7专业级最优解策略】
    优化思路：
    1. 动能爆发进场：EMA 5 突破 EMA 20 且成交量显著放大（确认主力进场）。
    2. 核心趋势持仓：只要价格维持在 EMA 20 或 EMA 50 以上，就死拿，不被小波动洗出。
    3. 动态利润保护：当获盈超过 20% 后，启动更灵敏的 ATR 追踪止损。
    4. 均线优化：使用 EMA（指数移动平均线）代替 SMA，反应更敏锐。
    """
    params = dict(
        ema_fast=5,
        ema_mid=20,
        ema_trend=50,      # 长期大趋势线（周线50均线是美股长牛生死线）
        atr_period=14,
        atr_mult=3.5,      # 较宽的止损空间，容忍M7的波动
        printlog=True,
    )

    def log(self, txt, dt=None):
        if self.p.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}, {txt}')

    def __init__(self):
        self.inds = {}
        for d in self.datas:
            self.inds[d] = {
                'ema5': bt.indicators.EMA(d.close, period=self.p.ema_fast),
                'ema20': bt.indicators.EMA(d.close, period=self.p.ema_mid),
                'ema50': bt.indicators.EMA(d.close, period=self.p.ema_trend),
                'atr': bt.indicators.ATR(d, period=self.p.atr_period),
                'vol_avg': bt.indicators.SMA(d.volume, period=10),
                'highest': 0.0,
                'entry_price': 0.0
            }

    def next(self):
        for d in self.datas:
            pos = self.getposition(d)
            close = d.close[0]
            ind = self.inds[d]
            
            # 基础条件
            bull_market = close > ind['ema50'][0]
            golden_cross = ind['ema5'][0] > ind['ema20'][0]
            vol_surge = d.volume[0] > ind['vol_avg'][0]
            
            if not pos.size:
                # 【买入逻辑】金叉 + 处于长牛趋势 + 放量
                if golden_cross and bull_market and vol_surge:
                    # 分配更积极：每只票分配 18% 资金（最多持仓5只，保持集中度）
                    cash_per_stock = self.broker.get_cash() * 0.18
                    size = math.floor(cash_per_stock / close)
                    if size > 0:
                        self.log(f'🚀【买入】{d._name} | 价格:{close:.2f} | 确认长牛放量')
                        self.buy(data=d, size=size)
                        ind['highest'] = close
                        ind['entry_price'] = close
            else:
                # 【持仓逻辑】
                ind['highest'] = max(ind['highest'], close)
                profit_pct = (close - ind['entry_price']) / ind['entry_price']
                
                # 动态止损：如果获利超过15%，止损收紧
                current_atr_mult = 2.0 if profit_pct > 0.15 else self.p.atr_mult
                trailing_stop = ind['highest'] - (ind['atr'][0] * current_atr_mult)
                
                # 【卖出逻辑】
                exit_signal = False
                reason = ""
                
                if close < ind['ema20'][0] and close < ind['ema5'][0]:
                    # 短期双均线跌破
                    if profit_pct < 0: # 如果是亏损的，果断止损
                        exit_signal = True
                        reason = "跌破均线止损"
                
                if close < trailing_stop:
                    exit_signal = True
                    reason = f"追踪止损({trailing_stop:.2f})"
                
                if close < ind['ema50'][0]:
                    exit_signal = True
                    reason = "长牛趋势终结"

                if exit_signal:
                    pnl = (close - ind['entry_price']) / ind['entry_price'] * 100
                    self.log(f'📉【卖出】{d._name} | 价格:{close:.2f} | 盈亏:{pnl:.1f}% | 原因:{reason}')
                    self.close(data=d)

def download_m7_data(start_date, end_date):
    tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"]
    data_feeds = {}
    print(f"正在准备 M7 深度回测数据...")
    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date, interval="1wk", auto_adjust=True)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            data_feeds[ticker] = bt.feeds.PandasData(dataname=df, name=ticker)
    return data_feeds

def run_backtest():
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(1000000.0)
    cerebro.broker.setcommission(commission=0.0005)
    
    start = "2025-01-01"
    end = dt.datetime.now().strftime("%Y-%m-%d")
    
    data_feeds = download_m7_data(start, end)
    for name, feed in data_feeds.items():
        cerebro.adddata(feed)

    cerebro.addstrategy(Mag7ProOptimal)

    print(f'\n开始回测 (专业级最优解)...')
    cerebro.run()
    final_val = cerebro.broker.getvalue()
    print(f'回测结束! 最终资产: {final_val:,.2f} | 总收益率: {(final_val-1000000)/10000:.2f}%')

if __name__ == "__main__":
    run_backtest()
