# v12_live_trading.py
# JOZ V12 Pro Plus 实盘交易版本（接入长桥API）
# 
# 使用前需要：
# 1. 安装长桥SDK: pip install longbridge
# 2. 在长桥APP中申请API密钥
# 3. 设置环境变量: LONGBRIDGE_APP_KEY, LONGBRIDGE_APP_SECRET, LONGBRIDGE_ACCESS_TOKEN

import argparse
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import backtrader as bt
import numpy as np
import pandas as pd
import yfinance as yf

# 长桥API导入（可选，如果未安装则使用模拟模式）
try:
    import sys
    # 确保能找到已安装的包
    if '/usr/local/lib/python3.10/dist-packages' not in sys.path:
        sys.path.insert(0, '/usr/local/lib/python3.10/dist-packages')
    from longbridge.openapi import TradeContext, Config, OrderType, OrderSide, TimeInForceType
    LONGBRIDGE_AVAILABLE = True
    print("✅ 长桥SDK导入成功")
except ImportError as e:
    LONGBRIDGE_AVAILABLE = False
    print(f"⚠️ 长桥SDK未安装或导入失败: {e}")
    print("安装命令: pip install longbridge")


class LongbridgeTrader:
    """长桥交易接口封装"""
    
    def __init__(self, paper_trading=True):
        """
        paper_trading: True=模拟交易, False=实盘交易
        """
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
        if self.ctx is None or self.paper_trading:
            return {"cash": 0, "equity": 0, "mock": True}
        
        try:
            account_list = self.ctx.account_balance()
            if account_list and len(account_list) > 0:
                account = account_list[0]  # 获取第一个账户
                # 获取港币现金信息
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
    
    def place_order(self, symbol, side, quantity, price=None, order_type="Limit"):
        """
        下单
        symbol: 股票代码，如 "CRWV"
        side: "Buy" 或 "Sell"
        quantity: 股数
        price: 价格（限价单需要）
        """
        # 转换股票代码格式（美股加.US后缀）
        if "." not in symbol:
            symbol = f"{symbol}.US"
        
        # 模拟交易模式
        if self.ctx is None or self.paper_trading:
            print(f"[模拟交易] {side} {symbol} | 股数: {quantity} | 价格: ${price:.2f if price else '市价'}")
            return {"success": True, "mock": True, "order_id": f"MOCK_{int(time.time())}"}
        
        # 实盘交易
        try:
            order_side = OrderSide.Buy if side == "Buy" else OrderSide.Sell
            
            # 价格精度处理（美股保留2位小数）
            if price:
                price = round(float(price), 2)
            
            if order_type == "Market":
                resp = self.ctx.submit_order(
                    symbol=symbol,
                    order_type=OrderType.MO,
                    side=order_side,
                    submitted_quantity=quantity,
                    time_in_force=TimeInForceType.Day
                )
            else:
                resp = self.ctx.submit_order(
                    symbol=symbol,
                    order_type=OrderType.LO,
                    side=order_side,
                    submitted_quantity=quantity,
                    submitted_price=price,
                    time_in_force=TimeInForceType.Day
                )
            
            print(f"✅ 实盘下单成功: {resp.order_id}")
            return {"success": True, "mock": False, "order_id": resp.order_id}
            
        except Exception as e:
            print(f"❌ 下单失败: {e}")
            return {"success": False, "error": str(e)}
    
    def get_positions(self):
        """获取当前持仓"""
        if self.ctx is None or self.paper_trading:
            return {}
        
        try:
            positions = self.ctx.stock_positions()
            if positions:
                return {p.symbol: {"quantity": p.quantity, "market_value": p.market_value} for p in positions}
            return {}
        except Exception as e:
            print(f"获取持仓失败: {e}")
            return {}


class JOZ_V12_Live_Trading(bt.Strategy):
    """V12实盘交易策略"""
    
    params = (
        ("mtm_period", 20),
        ("max_positions", 3),
        ("leverage", 1.3),
        ("margin_buffer", 0.92),
        ("atr_period", 14),
        ("atr_multipliers", {
            "RKLB": 4.0, "CRWV": 4.0, "TSLA": 3.5, "NVDA": 3.5,
            "MU": 3.0, "VRT": 3.0, "DEFAULT": 3.0
        }),
        ("max_dd_limit", 0.22),
        ("cooldown_days", 10),
        ("special_cap", 0.30),
        ("rebalance_days", (0,)),
        ("paper_trading", True),  # 默认模拟交易
    )

    def __init__(self):
        self.max_equity = self.broker.getvalue()
        self.is_halted = False
        self.halt_start_bar = 0

        self.inds = {
            d: {
                "roc": bt.indicators.RateOfChange(d.close, period=self.p.mtm_period),
                "sma30": bt.indicators.SMA(d.close, period=30),
                "atr": bt.indicators.ATR(d, period=self.p.atr_period),
                "high20": bt.indicators.Highest(d.high, period=20),
            } for d in self.datas
        }

        self.equity_history = []
        self.dates_history = []
        self.trade_details_list = []
        self.max_prices = {}
        self.partial_sold = {}
        self.last_rebalance_date = None
        
        # 初始化长桥交易接口
        self.trader = LongbridgeTrader(paper_trading=self.p.paper_trading)

    def log(self, txt):
        print(f"{self.datetime.date(0)} | {txt}")

    def round_size(self, price, target_value):
        if price <= 0:
            return 0
        raw = target_value / price
        size = int(raw // 10) * 10
        return size if size >= 10 else 0

    def notify_order(self, order):
        if order.status == order.Completed:
            d = order.data
            m = self.p.atr_multipliers.get(d._name, self.p.atr_multipliers["DEFAULT"])
            atr_stop_val = float(self.inds[d]["high20"][0]) - (m * float(self.inds[d]["atr"][0]))

            op = "买入" if order.isbuy() else "卖出"
            cash = self.broker.getcash()
            total_value = self.broker.getvalue()
            pos = self.getposition(d)
            pos_value = pos.size * float(d.close[0]) if pos.size > 0 else 0
            
            trade_info = {
                "Date": self.datetime.date(0),
                "Ticker": d._name,
                "Action": op,
                "Price": round(float(order.executed.price), 2),
                "Size": int(order.executed.size),
                "Value": round(float(order.executed.price) * abs(int(order.executed.size)), 2),
                "SMA30_Line": round(float(self.inds[d]["sma30"][0]), 2),
                "ATR_Stop_Line": round(float(atr_stop_val), 2),
                "Cash_After": round(float(cash), 2),
                "Position_Value": round(float(pos_value), 2),
                "Total_Equity": round(float(total_value), 2),
            }
            self.trade_details_list.append(trade_info)
            
            if order.isbuy():
                self.log(f"【建仓买入】{d._name}")
                self.log(f"  ├─ 买入价格: ${order.executed.price:.2f}")
                self.log(f"  ├─ 买入股数: {int(order.executed.size)} 股")
                self.log(f"  ├─ 买入市值: ${float(order.executed.price) * int(order.executed.size):,.2f}")
                self.log(f"  ├─ 账户余额: ${cash:,.2f}")
                self.log(f"  ├─ 持仓市值: ${pos_value:,.2f}")
                self.log(f"  ├─ 总资产: ${total_value:,.2f}")
                self.log(f"  └─ ATR止损线: ${atr_stop_val:.2f}")
                
                # 发送实盘订单（如果是实盘模式）
                if not self.p.paper_trading:
                    result = self.trader.place_order(
                        d._name, "Buy", int(order.executed.size), float(order.executed.price)
                    )
                    if result.get("success"):
                        self.log(f"  ✅ 实盘订单已提交: {result.get('order_id')}")
                    else:
                        self.log(f"  ❌ 实盘订单失败: {result.get('error')}")
                        
            else:
                self.log(f"【卖出】{d._name} | 价格: ${order.executed.price:.2f} | 股数: {int(order.executed.size)} | 余额: ${cash:,.2f}")
                
                # 发送实盘订单（如果是实盘模式）
                if not self.p.paper_trading:
                    result = self.trader.place_order(
                        d._name, "Sell", abs(int(order.executed.size)), float(order.executed.price)
                    )
                    if result.get("success"):
                        self.log(f"  ✅ 实盘订单已提交: {result.get('order_id')}")
                    else:
                        self.log(f"  ❌ 实盘订单失败: {result.get('error')}")

    def next(self):
        val = float(self.broker.getvalue())
        if val > self.max_equity:
            self.max_equity = val
        drawdown = (val - self.max_equity) / (self.max_equity + 1e-9)

        self.equity_history.append(val)
        self.dates_history.append(self.datetime.date(0))

        if self.is_halted:
            if len(self) - self.halt_start_bar >= self.p.cooldown_days:
                self.is_halted = False
                self.log(">>> 冷却结束，重启系统。")
            else:
                return

        if drawdown < -self.p.max_dd_limit:
            self.log("！！！触发熔断强制清仓")
            for d in self.datas:
                self.close(d)
            self.is_halted = True
            self.halt_start_bar = len(self)
            return

        for d in self.datas:
            pos = self.getposition(d)
            name = d._name

            if pos.size > 0:
                self.max_prices[name] = max(self.max_prices.get(name, 0.0), float(d.high[0]))
                m = self.p.atr_multipliers.get(name, self.p.atr_multipliers["DEFAULT"])
                atr_stop = self.max_prices[name] - (m * float(self.inds[d]["atr"][0]))

                if float(d.close[0]) < atr_stop and not self.partial_sold.get(name, False):
                    self.log(f"【分批减仓】{name} 触发 {m}x ATR 止损")
                    sell_size = self.round_size(float(d.close[0]), (pos.size * float(d.close[0])) / 2)
                    if sell_size >= 10:
                        self.sell(data=d, size=sell_size)
                    self.partial_sold[name] = True

                elif float(d.close[0]) < float(self.inds[d]["sma30"][0]):
                    self.log(f"【趋势出场】{name} 跌破 SMA30")
                    self.close(d)
                    self.partial_sold[name] = False
            else:
                self.max_prices.pop(name, None)
                self.partial_sold.pop(name, None)

        wd = self.datetime.date(0).weekday()
        if wd in self.p.rebalance_days:
            today = self.datetime.date(0)
            if self.last_rebalance_date != today:
                self.rebalance()
                self.last_rebalance_date = today

    def rebalance(self):
        cash_before = self.broker.getcash()
        equity_before = self.broker.getvalue()
        self.log(f"【调仓开始】账户余额: ${cash_before:,.2f} | 总资产: ${equity_before:,.2f}")
        
        self.log("【当前持仓】")
        for d in self.datas:
            pos = self.getposition(d)
            if pos.size > 0:
                pos_value = pos.size * float(d.close[0])
                self.log(f"  ├─ {d._name}: {pos.size} 股 | 市值: ${pos_value:,.2f}")
        
        scores = [
            (d, float(self.inds[d]["roc"][0]))
            for d in self.datas
            if float(self.inds[d]["roc"][0]) > 0.05 and float(d.close[0]) > float(self.inds[d]["sma30"][0])
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        top_picks = [x[0] for x in scores[: self.p.max_positions]]
        
        if top_picks:
            roc_values = [f"{d._name}({float(self.inds[d]['roc'][0])*100:.1f}%)" for d in top_picks]
            self.log(f"【选股结果】{', '.join(roc_values)}")
        else:
            self.log("【选股结果】无符合条件的股票")

        for d in self.datas:
            if self.getposition(d).size != 0 and d not in top_picks:
                self.close(d)

        if not top_picks:
            return

        total_equity = float(self.broker.getvalue())
        total_target_pct = self.p.leverage * self.p.margin_buffer

        specials = [d for d in top_picks if d._name in ("RKLB", "CRWV")]
        others = [d for d in top_picks if d._name not in ("RKLB", "CRWV")]

        used_pct = 0.0

        if specials:
            cap_each = self.p.special_cap / len(specials)
            self.log(f"【建仓计划】Special组 (上限30%): {', '.join([d._name for d in specials])}")
            for d in specials:
                if self.getposition(d).size == 0:
                    target_val = total_equity * cap_each
                    size = self.round_size(float(d.close[0]), target_val)
                    if size >= 10:
                        self.log(f"  ├─ {d._name}: 计划买入 {size} 股 | 目标市值: ${target_val:,.2f}")
                        self.order_target_size(d, target=size)
                    else:
                        self.log(f"  ├─ {d._name}: 计算股数 {size} < 10，跳过")
            used_pct += self.p.special_cap

        if others:
            remain_pct = max(0.0, total_target_pct - used_pct)
            each_pct = remain_pct / len(others)
            self.log(f"【建仓计划】Others组 (剩余{remain_pct*100:.0f}%): {', '.join([d._name for d in others])}")

            for d in others:
                if self.getposition(d).size == 0:
                    target_val = total_equity * each_pct
                    size = self.round_size(float(d.close[0]), target_val)
                    if size >= 10:
                        self.log(f"  ├─ {d._name}: 计划买入 {size} 股 | 目标市值: ${target_val:,.2f}")
                        self.order_target_size(d, target=size)
                    else:
                        self.log(f"  ├─ {d._name}: 计算股数 {size} < 10，跳过")

        return


def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    return df


def main():
    ap = argparse.ArgumentParser(description="V12实盘交易 (接入长桥API)")
    ap.add_argument("--start", type=str, default="2024-01-01")
    ap.add_argument("--end", type=str, default="2025-02-14")
    ap.add_argument("--cash", type=float, default=110000.0)
    ap.add_argument("--commission", type=float, default=0.0003)
    ap.add_argument("--rebalance_days", nargs="*", type=int, default=[0])
    ap.add_argument("--tickers", nargs="*", default=['AAPL','GOOGL','AMZN','META','NVDA','TSLA','MU','WMT','VRT','RKLB','CRWV'])
    ap.add_argument("--outdir", type=str, default="output")
    ap.add_argument("--live", action="store_true", help="启用实盘交易（默认模拟）")
    args = ap.parse_args()

    if args.live and not LONGBRIDGE_AVAILABLE:
        print("❌ 长桥SDK未安装，无法启用实盘交易")
        print("安装命令: pip install longbridge")
        return

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(args.cash)
    cerebro.broker.setcommission(commission=args.commission)
    cerebro.broker.set_coc(True)

    print("=" * 60)
    print("🚀 JOZ V12 Pro Plus 实盘交易系统")
    print(f"   模式: {'🔴 实盘交易' if args.live else '🟡 模拟交易'}")
    print("=" * 60)
    print("\n正在拉取数据...")
    
    added = 0
    for t in args.tickers:
        df = download_data(t, args.start, args.end)
        if df.empty:
            print(f"⚠️ 拉取失败/为空：{t}")
            continue
        cerebro.adddata(bt.feeds.PandasData(dataname=df, name=t))
        added += 1

    print(f"data feeds added: {added}")
    if added == 0:
        raise RuntimeError("没有任何数据源被加入回测。")

    cerebro.addstrategy(
        JOZ_V12_Live_Trading, 
        rebalance_days=tuple(args.rebalance_days),
        paper_trading=not args.live
    )

    print(">>> 开始运行...")
    results = cerebro.run()
    strat = results[0]

    if len(strat.equity_history) == 0:
        raise RuntimeError("equity_history 为空")

    perf_df = pd.DataFrame({"equity": strat.equity_history}, index=pd.to_datetime(strat.dates_history))
    perf_df["equity_change"] = perf_df["equity"].diff().fillna(0)
    perf_df["ret"] = perf_df["equity"].pct_change().fillna(0)

    start_eq, end_eq = float(perf_df["equity"].iloc[0]), float(perf_df["equity"].iloc[-1])
    days = max((perf_df.index[-1] - perf_df.index[0]).days, 1)
    cagr = (end_eq / start_eq) ** (365 / days) - 1
    sharpe = np.sqrt(252) * perf_df["ret"].mean() / (perf_df["ret"].std() + 1e-9)
    max_dd = ((perf_df["equity"] - perf_df["equity"].cummax()) / perf_df["equity"].cummax()).min() * 100

    print(f"""
===== JOZ V12 Live Trading Report =====
Mode      : {'LIVE' if args.live else 'PAPER TRADING'}
Rebalance : {args.rebalance_days}
Period    : {perf_df.index.min().date()} -> {perf_df.index.max().date()}
Start Eq  : {start_eq:.2f}
End Eq    : {end_eq:.2f}

CAGR      : {cagr:.2%}
Sharpe    : {sharpe:.2f}
Max DD    : {max_dd:.2f}%
""")

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    perf_path = outdir / f"V12_Live_{'REAL' if args.live else 'PAPER'}_{ts}.csv"
    trades_path = outdir / f"V12_Live_Trades_{'REAL' if args.live else 'PAPER'}_{ts}.csv"

    perf_df.to_csv(perf_path, encoding="utf-8-sig")
    pd.DataFrame(strat.trade_details_list).to_csv(trades_path, index=False, encoding="utf-8-sig")
    print(">>> 导出完成：")
    print(" -", perf_path.resolve())
    print(" -", trades_path.resolve())


if __name__ == "__main__":
    main()
