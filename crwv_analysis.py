import backtrader as bt
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

class VolatilityAnalyzer(bt.Strategy):
    params = (
        ('vol_period', 5),  # 5日波动率
        ('w_ma_fast', 5),   # 5周均线
    )

    def __init__(self):
        self.crwv = self.datas[0]
        self.nvda = self.datas[1]
        
        # 计算 CRWV 的日收益率
        self.returns = bt.indicators.PctChange(self.crwv.close, period=1)
        # 计算 5 日滚动标准差作为波动率指标
        self.volatility = bt.indicators.StdDev(self.returns, period=self.p.vol_period)
        
        # 5周均线 (1周=5交易日，5周=25日)
        self.w_ma5 = bt.indicators.SMA(self.crwv.close, period=self.p.w_ma_fast * 5)
        
        # 成交量连续递增/递减逻辑
        self.vol_up_3d = bt.indicators.And(
            self.crwv.volume(0) > self.crwv.volume(-1),
            self.crwv.volume(-1) > self.crwv.volume(-2),
            self.crwv.volume(-2) > self.crwv.volume(-3)
        )
        self.vol_down_3d = bt.indicators.And(
            self.crwv.volume(0) < self.crwv.volume(-1),
            self.crwv.volume(-1) < self.crwv.volume(-2),
            self.crwv.volume(-2) < self.crwv.volume(-3)
        )
        
        # 记录数据用于后期分析
        self.my_stats = []

    def next(self):
        # 判定买卖信号
        signal = ""
        # 买入信号：收盘价上穿5周均线 + 连续3日成交量递增
        if self.crwv.close[0] > self.w_ma5[0] and self.crwv.close[-1] <= self.w_ma5[-1] and self.vol_up_3d[0]:
            signal = "BUY"
        # 卖出信号：收盘价下穿5周均线 + 连续3日成交量递减
        elif self.crwv.close[0] < self.w_ma5[0] and self.crwv.close[-1] >= self.w_ma5[-1] and self.vol_down_3d[0]:
            signal = "SELL"

        # 记录每日数据
        self.my_stats.append({
            'date': self.crwv.datetime.date(0),
            'crwv_close': self.crwv.close[0],
            'crwv_vol': self.crwv.volume[0],
            'crwv_volatility': self.volatility[0],
            'crwv_ret': self.returns[0],
            'nvda_close': self.nvda.close[0],
            'nvda_ret': (self.nvda.close[0] - self.nvda.close[-1])/self.nvda.close[-1] if len(self.nvda) > 1 else 0,
            'signal': signal,
            'w_ma5': self.w_ma5[0],
            'vol_0': self.crwv.volume[0],
            'vol_1': self.crwv.volume[-1],
            'vol_2': self.crwv.volume[-2],
            'vol_3': self.crwv.volume[-3]
        })

def run_analysis():
    # 1. 下载数据 (从 IPO 日期 2025-03-28 开始)
    print(">>> 正在下载 CRWV 和 NVDA 的历史数据...")
    # 为了包含 2月8日之前的数据，end 设为 2026-02-09
    crwv_df = yf.download("CRWV", start="2025-03-28", end="2026-02-09", auto_adjust=True)
    nvda_df = yf.download("NVDA", start="2025-03-28", end="2026-02-09", auto_adjust=True)
    
    if crwv_df.empty or nvda_df.empty:
        print("数据下载失败，请检查 Ticker 或网络。")
        return

    # 预处理：修复 yfinance 可能产生的多级列名
    if isinstance(crwv_df.columns, pd.MultiIndex):
        crwv_df.columns = crwv_df.columns.get_level_values(0)
    if isinstance(nvda_df.columns, pd.MultiIndex):
        nvda_df.columns = nvda_df.columns.get_level_values(0)

    # 2. 设置 Backtrader
    cerebro = bt.Cerebro()
    
    data0 = bt.feeds.PandasData(dataname=crwv_df, name="CRWV")
    data1 = bt.feeds.PandasData(dataname=nvda_df, name="NVDA")
    
    cerebro.adddata(data0)
    cerebro.adddata(data1)
    cerebro.addstrategy(VolatilityAnalyzer)
    
    results = cerebro.run()
    strat = results[0]
    
    # 3. 分析统计结果
    df_my_stats = pd.DataFrame(strat.my_stats)
    df_my_stats.dropna(inplace=True)
    
    # 打印买卖信号及成交量详情
    signals = df_my_stats[df_my_stats['signal'] != ""]
    print("\n" + "="*60)
    print("📢 CoreWeave (CRWV) 关键交易信号与成交量标注")
    print("="*60)
    if signals.empty:
        print("在回测期间未检测到符合条件的【均线交叉+连续3日成交量】信号。")
    else:
        for _, row in signals.iterrows():
            type_str = "🟢 买入" if row['signal'] == "BUY" else "🔴 卖出"
            inc_dec = "递增" if row['signal'] == "BUY" else "递减"
            print(f"日期: {row['date']} | 信号: {type_str} | 价格: {row['crwv_close']:.2f}")
            print(f"  └─ 📊 成交量连续{inc_dec}: {row['vol_3']:,.0f} -> {row['vol_2']:,.0f} -> {row['vol_1']:,.0f} -> {row['vol_0']:,.0f}")
    
    # 最近几天的详细数据
    print("\n" + "-"*60)
    print("📅 最近 5 个交易日的详细成交量与波动情况:")
    recent_days = df_my_stats.tail(5)
    for _, row in recent_days.iterrows():
        ma5_val = row['w_ma5']
        crossed = " (已站上5周线)" if row['crwv_close'] > ma5_val else " (在5周线下方)"
        print(f"日期: {row['date']} | 价格: {row['crwv_close']:.2f} | 5周线: {ma5_val:.2f}{crossed}")
        print(f"  └─ 涨跌: {row['crwv_ret']:.2%} | 成交量: {row['crwv_vol']:,.0f}")

    # 计算相关性
    corr_vol_volatility = df_my_stats['crwv_vol'].corr(df_my_stats['crwv_volatility'])
    corr_crwv_nvda_ret = df_my_stats['crwv_ret'].corr(df_my_stats['nvda_ret'])
    
    print("\n" + "="*50)
    print("📈 CoreWeave (CRWV) 波动率相关性分析报告")
    print("="*50)
    print(f"数据范围: {df_my_stats['date'].min()} 到 {df_my_stats['date'].max()}")
    print(f"1. 成交量与波动率相关性: {corr_vol_volatility:.4f}")
    print(f"2. CRWV 与 NVDA 收益率相关性: {corr_crwv_nvda_ret:.4f}")
    
    # 找出波动率最大的前 5 天
    top_vol = df_my_stats.nlargest(5, 'crwv_volatility')
    print("\n🔥 波动率最大的 5 个交易日:")
    for _, row in top_vol.iterrows():
        print(f"日期: {row['date']} | 波动率指数: {row['crwv_volatility']:.4f} | "
              f"当日涨跌: {row['crwv_ret']:.2%} | 成交量: {row['crwv_vol']:,.0f}")

    # 4. 结论与逻辑推导
    print("\n💡 分析结论:")
    if corr_vol_volatility > 0.5:
        print("- 波动与成交量显著正相关：大波动通常伴随着巨量交易，反映了市场分歧或突发消息。")
    else:
        print("- 波动与成交量相关性一般：部分波动可能是缩量下跌或阴跌后的剧烈反弹。")
        
    if corr_crwv_nvda_ret > 0.6:
        print("- 与 NVDA 关联极强：CRWV 表现高度依赖 NVIDIA 的景气度，属于 AI 基础设施共振。")
    elif corr_crwv_nvda_ret > 0.3:
        print("- 与 NVDA 中度关联：受行业大势影响，但也有公司自身的独立逻辑（如财报、订单）。")
    else:
        print("- 与 NVDA 独立性较强：股价更多受自身基本面或 IPO 后锁定期解禁等因素影响。")

    # 5. 可视化
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    ax1.plot(df_my_stats['date'], df_my_stats['crwv_volatility'], label='CRWV Volatility (5D StdDev)', color='orange')
    ax1.set_title("CRWV Daily Volatility")
    ax1.legend()
    
    ax2.scatter(df_my_stats['nvda_ret'], df_my_stats['crwv_ret'], alpha=0.5)
    ax2.set_title(f"Correlation: CRWV vs NVDA Returns (Corr: {corr_crwv_nvda_ret:.2f})")
    ax2.set_xlabel("NVDA Daily Return")
    ax2.set_ylabel("CRWV Daily Return")
    
    plt.tight_layout()
    plt.savefig('crwv_volatility_analysis.png')
    print("\n>>> 分析图表已保存为: crwv_volatility_analysis.png")

if __name__ == "__main__":
    run_analysis()
