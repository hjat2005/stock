import backtrader as bt
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class JOZ_V12_Professional(bt.Strategy):
    params = (
        ('mtm_period', 20),
        ('max_positions', 3),
        ('leverage', 1.3),
        ('max_dd_limit', 0.20),
        ('cooldown_days', 10),
        ('buffer', 0.88),
        ('rklb_cap', 0.30),
        ('start_cash', 110000.0),   # 用于基准与报告一致
        ('rolling_window', 20),
    )

    def __init__(self):
        # === 原策略字段 ===
        self.max_equity = self.broker.getvalue()
        self.is_halted = False
        self.halt_start_bar = 0

        self.inds = {}
        for d in self.datas:
            self.inds[d] = {
                'roc': bt.indicators.RateOfChange(d.close, period=self.p.mtm_period),
                'sma30': bt.indicators.SMA(d.close, period=30)
            }

        # === 报告记录字段 ===
        self.equity_history = []
        self.dd_history = []
        self.rolling_dd_history = []
        self.dates_history = []
        self.daily_pnl_history = []
        self.bh_equity_history = []

        self.prev_val = self.broker.getvalue()
        self.bh_init_price = None  # 基准以第一个data

    def log(self, txt):
        print(f'{self.datetime.date(0)} | {txt}')

    def notify_order(self, order):
        if order.status in [order.Completed]:
            op = "买入" if order.isbuy() else "卖出"
            # 获取成交后的状态
            cash = self.broker.get_cash()
            value = self.broker.get_value()
            
            self.log(f"【{op}成交】{order.data._name} | 价格:{order.executed.price:.2f} | 股数:{order.executed.size:.0f} | 剩余现金:${cash:,.2f}")
            
            # 打印当前所有持仓情况
            pos_list = [f"{d._name}({self.getposition(d).size}股)" for d in self.datas if self.getposition(d).size != 0]
            if pos_list:
                self.log(f"  └─ 当前持仓: {', '.join(pos_list)} | 总资产:${value:,.2f}")
            else:
                self.log(f"  └─ 当前无持仓 | 总资产:${value:,.2f}")
                
        elif order.status in [order.Margin, order.Rejected]:
            self.log(f"！！！警告：{order.data._name} 下单被拒，状态:{order.getstatusname()}，当前余额:${self.broker.get_cash():,.2f}！！！")

    def next(self):
        val = self.broker.getvalue()
        if val > self.max_equity:
            self.max_equity = val
        drawdown = (val - self.max_equity) / self.max_equity  # 原版

        # ===== 记录（每天都记录，含熔断期） =====
        self.equity_history.append(val)
        self.dd_history.append(drawdown * 100)
        self.dates_history.append(self.datetime.date(0))
        self.daily_pnl_history.append(val - self.prev_val)
        self.prev_val = val

        # Buy&Hold 基准（以第一个标的 close）
        if len(self.datas) > 0:
            px = float(self.datas[0].close[0])
            if self.bh_init_price is None:
                self.bh_init_price = px
            bh_val = self.p.start_cash * (px / self.bh_init_price)
            self.bh_equity_history.append(bh_val)
        else:
            self.bh_equity_history.append(np.nan)

        # Rolling DD（窗口内峰值回撤）
        w = self.p.rolling_window
        if len(self.equity_history) >= w:
            peak = max(self.equity_history[-w:])
            rdd = (val - peak) / peak * 100
        else:
            rdd = 0.0
        self.rolling_dd_history.append(rdd)

        # ===== 1) 熔断冷却逻辑（严格按你原版） =====
        if self.is_halted:
            if len(self) - self.halt_start_bar >= self.p.cooldown_days:
                self.is_halted = False
                self.max_equity = val  # 关键：重置峰值
                self.log(">>> 冷却结束，重启系统。")
            else:
                return

        if drawdown < -self.p.max_dd_limit:
            self.log(f"！！！触发熔断：回撤 {drawdown:.2%} ！！！强制止损清仓。")
            for d in self.datas:
                self.close(d)
            self.is_halted = True
            self.halt_start_bar = len(self)
            return

        # ===== 2) 个股 SMA30 趋势止损 =====
        for d in self.datas:
            if self.getposition(d).size > 0:
                if d.close[0] < self.inds[d]['sma30'][0]:
                    self.log(f"趋势破位离场: {d._name}")
                    self.close(d)

        # ===== 3) 每周一轮动 =====
        if self.datetime.date(0).weekday() == 0:
            self.rebalance_professional()

    def rebalance_professional(self):
        scores = []
        for d in self.datas:
            if self.inds[d]['roc'][0] > 0.05:
                scores.append((d, self.inds[d]['roc'][0]))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_picks = [x[0] for x in scores[:self.p.max_positions]]

        # 清仓不在榜单的股票
        for d in self.datas:
            if self.getposition(d).size != 0 and d not in top_picks:
                self.close(d)

        if not top_picks:
            return

        # 权重分配（RKLB封顶）
        total_target_pct = self.p.leverage * self.p.buffer
        rklb_data = next((d for d in top_picks if d._name == 'RKLB'), None)

        if rklb_data:
            self.order_target_percent(data=rklb_data, target=self.p.rklb_cap)
            other_picks = [d for d in top_picks if d._name != 'RKLB']
            if other_picks:
                remaining_pct = total_target_pct - self.p.rklb_cap
                target_each = remaining_pct / len(other_picks)
                for d in other_picks:
                    self.order_target_percent(data=d, target=target_each)
        else:
            target_each = total_target_pct / len(top_picks)
            for d in top_picks:
                self.order_target_percent(data=d, target=target_each)

# ======================
# Backtest
# ======================
if __name__ == '__main__':
    print(">>> 脚本启动...")
    cerebro = bt.Cerebro()
    cerebro.broker.set_coc(True)
    cerebro.broker.setcash(110000.0)
    cerebro.broker.setcommission(commission=0.0003)

    tickers = ['AAPL', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'MU', 'WMT', 'VRT', 'RKLB']

    print("正在构建 V12 职业版数据...")
    for t in tickers:
        df = yf.download(
            t,
            start='2024-01-01',
            end='2026-02-04',
            auto_adjust=True,   # ✅ 对齐你当前默认行为，且消除 warning
            progress=True
        )
        if df is None or df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        data = bt.feeds.PandasData(dataname=df, name=t)
        cerebro.adddata(data)

    cerebro.addstrategy(JOZ_V12_Professional)

    print('>>> 开启 1.3x 杠杆（RKLB 权重受限版）回测')
    results = cerebro.run()
    strat = results[0]
    final_val = cerebro.broker.getvalue()
        
    # ======================
    # Report DF
    # ======================
    rep = pd.DataFrame({
        "date": pd.to_datetime(strat.dates_history),
        "equity": strat.equity_history,
        "drawdown_pct": strat.dd_history,
        "rolling_dd_20": strat.rolling_dd_history,
        "daily_pnl": strat.daily_pnl_history,
        "buy_hold": strat.bh_equity_history
    }).set_index("date")
    
    print(f'\n>>> 回测结束 | 账户总价值: {final_val:,.2f}')
        
    # 打印最终持仓汇总
    print(f'{"="*60}')
    print(f'📊 最终持仓汇总 (截至 {rep.index.max().date()}):')
    print(f'{"="*60}')
    final_pos = False
    for d in strat.datas:
        pos = strat.getposition(d)
        if pos.size != 0:
            final_pos = True
            cur_price = d.close[0]
            pos_value = pos.size * cur_price
            print(f'{d._name:6s} | 数量: {pos.size:5.0f} 股 | 现价: ${cur_price:7.2f} | 市值: ${pos_value:10,.2f}')
    
    if not final_pos:
        print("当前账户无任何持仓。")
    print(f'{"="*60}\n')

    rep["ret"] = rep["equity"].pct_change().fillna(0.0)

    # ======================
    # Metrics
    # ======================
    start_eq = rep["equity"].iloc[0]
    end_eq = rep["equity"].iloc[-1]
    n = len(rep)

    cagr = (end_eq / start_eq) ** (252 / max(n, 1)) - 1
    vol = rep["ret"].std()
    sharpe = np.sqrt(252) * rep["ret"].mean() / vol if vol != 0 else np.nan
    max_dd = rep["drawdown_pct"].min()
    calmar = cagr / abs(max_dd / 100) if max_dd != 0 else np.nan

    win_rate = (rep["daily_pnl"] > 0).mean()
    pf = rep.loc[rep["daily_pnl"] > 0, "daily_pnl"].sum() / abs(rep.loc[rep["daily_pnl"] < 0, "daily_pnl"].sum()) if (rep["daily_pnl"] < 0).any() else np.nan

    print(f"""
===== JOZ V12 Professional Report =====
Period    : {rep.index.min().date()} -> {rep.index.max().date()}
Start Eq  : {start_eq:.2f}
End Eq    : {end_eq:.2f}

CAGR      : {cagr:.2%}
Sharpe    : {sharpe:.2f}
Max DD    : {max_dd:.2f}%
Calmar    : {calmar:.2f}

Win Rate  : {win_rate:.2%}
ProfitFac : {pf:.2f}
""")

    # ======================
    # Export CSV + Colab Download
    # ======================
    csv_name = f"JOZ_V12_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    rep.to_csv(csv_name, encoding="utf-8-sig")
    print(f">>> 已导出 CSV: {csv_name}")

    try:
        from google.colab import files
        files.download(csv_name)
    except Exception:
        print(">>> 非 Colab 环境：CSV 已保存在当前目录")

    # ======================
    # Monthly heatmap data
    # ======================
    monthly = rep["ret"].resample("ME").apply(lambda x: (1 + x).prod() - 1)
    heat_df = pd.DataFrame({"year": monthly.index.year, "month": monthly.index.month, "ret": monthly.values})
    heat = heat_df.pivot(index="year", columns="month", values="ret").reindex(columns=range(1, 13))

    # ======================
    # Visualization (4 + heatmap)
    # ======================
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(5, 1, height_ratios=[3, 1.2, 1.2, 1.2, 2])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax5 = fig.add_subplot(gs[4])

    ax1.plot(rep.index, rep["equity"], label="Strategy", linewidth=2)
    ax1.plot(rep.index, rep["buy_hold"], linestyle="--", label="Buy&Hold", alpha=0.8)
    ax1.axhline(110000, linestyle=":", alpha=0.5)
    ax1.set_title("Equity Curve vs Benchmark")
    ax1.grid(alpha=0.2)
    ax1.legend()

    ax2.fill_between(rep.index, rep["drawdown_pct"], 0, alpha=0.35)
    ax2.set_title("Drawdown %")
    ax2.set_ylim(min(-40, rep["drawdown_pct"].min() * 1.1), 0)
    ax2.grid(alpha=0.2)

    ax3.plot(rep.index, rep["rolling_dd_20"], linewidth=1.5)
    ax3.set_title("Rolling 20D Max Drawdown %")
    ax3.set_ylim(min(-40, rep["rolling_dd_20"].min() * 1.1), 5)
    ax3.grid(alpha=0.2)

    ax4.plot(rep.index, rep["daily_pnl"], linewidth=1.0)
    ax4.axhline(0, linestyle="--", alpha=0.5)
    ax4.set_title("Daily PnL ($)")
    ax4.grid(alpha=0.2)

    im = ax5.imshow(heat.values, aspect="auto")
    ax5.set_title("Monthly Returns Heatmap")
    ax5.set_yticks(range(len(heat.index)))
    ax5.set_yticklabels(heat.index.astype(str))
    ax5.set_xticks(range(12))
    ax5.set_xticklabels([str(m) for m in range(1, 13)])

    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat.values[i, j]
            if np.isfinite(v):
                ax5.text(j, i, f"{v*100:.1f}%", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plot_name = f"JOZ_V12_Performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_name)
    print(f">>> 已保存图表: {plot_name}")
    # plt.show()
