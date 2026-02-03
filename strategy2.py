# -*- coding: utf-8 -*-
import datetime as dt
import backtrader as bt
import pandas as pd
import yfinance as yf
import math

class Mag7HybridStrategy(bt.Strategy):
    """
    【美股M7周/日混合增强策略】
    逻辑：
    1. 买入前置：5周均线 上穿 20周均线。
    2. 买入触发：连续3日成交量递增。
    3. 卖出前置：5周均线 下穿 20周均线。
    4. 卖出触发：连续3日成交量递减。
    5. 判定时间：模拟收盘前半小时（使用当日收盘价）。
    """
    params = dict(
        w_ma_fast=5,     # 5周线
        w_ma_slow=20,    # 20周线
        vol_days=3,      # 连续3日
        printlog=True,
    )

    def log(self, txt, dt=None):
        if self.p.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}, {txt}')

    def __init__(self):
        self.inds = {}
        self.entry_price = {}  # 记录买入价格
        self.entry_date = {}   # 记录买入日期
        self.order_dict = {}   # 记录待执行订单
        
        for d in self.datas:
            self.inds[d] = {}
            self.entry_price[d] = 0.0
            self.entry_date[d] = None
            self.order_dict[d] = None
            
            # 日线数据模拟周线均线 (1周=5交易日)
            self.inds[d]['w_ma5'] = bt.indicators.SMA(d.close, period=self.p.w_ma_fast * 5)
            self.inds[d]['w_ma20'] = bt.indicators.SMA(d.close, period=self.p.w_ma_slow * 5)
            
            # 日线成交量趋势
            self.inds[d]['vol_up'] = bt.indicators.And(
                d.volume(0) > d.volume(-1),
                d.volume(-1) > d.volume(-2),
                d.volume(-2) > d.volume(-3)
            )
            self.inds[d]['vol_down'] = bt.indicators.And(
                d.volume(0) < d.volume(-1),
                d.volume(-1) < d.volume(-2),
                d.volume(-2) < d.volume(-3)
            )

    def notify_order(self, order):
        """订单状态变化通知"""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        d = order.data
        
        if order.status == order.Completed:
            if order.isbuy():
                # 买入成交后，打印实际账户信息
                cash_after = self.broker.get_cash()
                value_after = self.broker.get_value()
                cost = order.executed.size * order.executed.price
                
                self.log(f'✅【买入成交】{d._name} | 成交价:{order.executed.price:.2f} | '
                        f'股数:{int(order.executed.size)} | 成本:${cost:,.2f} | '
                        f'成交后余额:${cash_after:,.2f} | 总资产:${value_after:,.2f}')
            elif order.issell():
                # 卖出成交后的账户信息
                cash_after = self.broker.get_cash()
                value_after = self.broker.get_value()
                
                self.log(f'💰【卖出成交】{d._name} | 成交价:{order.executed.price:.2f} | '
                        f'股数:{int(order.executed.size)} | '
                        f'成交后余额:${cash_after:,.2f} | 总资产:${value_after:,.2f}')
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'⚠️【订单失败】{d._name} | 状态:{order.getstatusname()}')
        
        # 清除订单记录
        self.order_dict[d] = None

    def next(self):
        for d in self.datas:
            pos = self.getposition(d)
            w_fast = self.inds[d]['w_ma5'][0]
            w_slow = self.inds[d]['w_ma20'][0]
            
            # 前置条件判断
            is_golden_zone = w_fast > w_slow
            is_death_zone = w_fast < w_slow

            if not pos.size:
                # 买入：前置(金叉区间) + 触发(3日增量)
                if is_golden_zone and self.inds[d]['vol_up']:
                    # 避免重复下单
                    if self.order_dict[d] is not None:
                        continue
                    
                    target_value = (self.broker.get_value() / len(self.datas)) * 0.90
                    size = math.floor(target_value / d.close[0])
                    if size > 0:
                        # 只打印下单信号，成交信息由 notify_order 打印
                        self.log(f'📊【下单买入】{d._name} | 目标价:{d.close[0]:.2f} | 目标股数:{size} | '
                                f'当前余额:${self.broker.get_cash():,.2f}')
                        
                        self.order_dict[d] = self.buy(data=d, size=size)
                        self.entry_price[d] = d.close[0]
                        self.entry_date[d] = self.datas[0].datetime.date(0)
            else:
                # 卖出：前置(死叉区间) + 触发(3日减量)
                if is_death_zone and self.inds[d]['vol_down']:
                    # 避免重复下单
                    if self.order_dict[d] is not None:
                        continue
                    
                    # 计算收益
                    entry = self.entry_price[d]
                    exit_price = d.close[0]
                    profit_pct = ((exit_price - entry) / entry) * 100 if entry > 0 else 0
                    hold_days = (self.datas[0].datetime.date(0) - self.entry_date[d]).days if self.entry_date[d] else 0
                    
                    self.log(f'📊【下单卖出】{d._name} | 买入价:{entry:.2f} → 目标卖出价:{exit_price:.2f} | '
                            f'预期收益率:{profit_pct:+.2f}% | 持有天数:{hold_days}天')
                    
                    self.order_dict[d] = self.close(data=d)
                    self.entry_price[d] = 0.0
                    self.entry_date[d] = None

def download_m7_data(start_date, end_date):
    # tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"]
    tickers = ["BABA", "CRWV", "SOXS"]
    data_feeds = {}
    print(f"正在准备 M7 数据...")
    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date, interval="1d", auto_adjust=True)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            data_feeds[ticker] = bt.feeds.PandasData(dataname=df, name=ticker)
    return data_feeds

def run_backtest():
    # 使用 cheat_on_close 模拟收盘前半小时判定
    cerebro = bt.Cerebro()
    cerebro.broker.set_coc(True)
    cerebro.broker.setcash(1000000.0)
    cerebro.broker.setcommission(commission=0.0005)

    # 回测时间设置：从2025年初开始，观察近期效果
    start = "2025-01-01"
    end = dt.datetime.now().strftime("%Y-%m-%d")
    
    data_feeds = download_m7_data(start, end)
    for name, feed in data_feeds.items():
        cerebro.adddata(feed)

    strats = cerebro.addstrategy(Mag7HybridStrategy)

    print(f'\n开始执行策略回测 (时间范围: {start} 至 {end})...')
    results = cerebro.run()
    strat = results[0]
    
    final_val = cerebro.broker.getvalue()
    print(f'\n回测结束! 最终资产: {final_val:,.2f} | 收益率: {(final_val-1000000)/10000:.2f}%')
    
    # 打印当前持仓及浮动盈亏
    print(f'\n{"="*60}')
    print(f'📊 当前持仓明细 (截至 {end}):')
    print(f'{"="*60}')
    
    has_position = False
    total_unrealized = 0.0
    
    for d in strat.datas:
        pos = strat.getposition(d)
        if pos.size > 0:
            has_position = True
            current_price = d.close[0]
            entry_price = strat.entry_price[d]
            entry_date = strat.entry_date[d]
            
            if entry_price > 0:
                unrealized_pct = ((current_price - entry_price) / entry_price) * 100
                unrealized_value = (current_price - entry_price) * pos.size
                total_unrealized += unrealized_value
                hold_days = (strat.datas[0].datetime.date(0) - entry_date).days if entry_date else 0
                
                print(f'{d._name:6s} | 持仓:{pos.size:4d}股 | 买入价:{entry_price:7.2f} | '
                      f'现价:{current_price:7.2f} | 浮盈:{unrealized_pct:+6.2f}% '
                      f'(${unrealized_value:+,.2f}) | 已持有:{hold_days}天')
    
    if not has_position:
        print('当前无持仓（已全部清仓）')
    else:
        print(f'{"="*60}')
        print(f'总浮动盈亏: ${total_unrealized:+,.2f}')

if __name__ == "__main__":
    run_backtest()
