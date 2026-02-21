# web/app.py
"""Streamlit Web应用主入口"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import init_db
from services import (
    DataService, TechnicalAnalysisService, PortfolioService, AlertService,
    StockAnalysis
)
from repositories import (
    StockRepository, PortfolioRepository, StockPriceRepository, 
    TradeRecordRepository, AlertRepository
)

# 页面配置
st.set_page_config(
    page_title="Quant Analysis System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化数据库
init_db()

# 初始化服务
@st.cache_resource
def get_services():
    return {
        'data': DataService(),
        'technical': TechnicalAnalysisService(),
        'portfolio': PortfolioService(),
        'alert': AlertService()
    }

services = get_services()

# 侧边栏导航
st.sidebar.title("📊 量化分析系统")

page = st.sidebar.radio(
    "选择功能",
    ["🏠 首页", "📈 股票分析", "💼 投资组合", "🔔 预警系统", "⚙️ 数据管理"]
)

# ==========================
# 首页
# ==========================
if page == "🏠 首页":
    st.title("🚀 量化投资分析系统")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("监控股票数", "10+", "+2")
    with col2:
        st.metric("投资组合", "3", "+1")
    with col3:
        st.metric("活跃预警", "5", "-1")
    
    st.divider()
    
    st.subheader("📋 系统功能")
    
    features = {
        "📈 股票分析": "技术分析、趋势判断、买卖建议",
        "💼 投资组合": "持仓管理、绩效追踪、交易记录",
        "🔔 预警系统": "价格预警、技术指标预警",
        "⚙️ 数据管理": "数据更新、历史数据查询"
    }
    
    for title, desc in features.items():
        st.write(f"**{title}**: {desc}")
    
    st.divider()
    
    # 快速操作
    st.subheader("⚡ 快速操作")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 分析热门股票"):
            st.session_state['quick_analyze'] = True
            st.switch_page("📈 股票分析")
    
    with col2:
        if st.button("💼 查看投资组合"):
            st.switch_page("💼 投资组合")
    
    with col3:
        if st.button("🔔 检查预警"):
            st.switch_page("🔔 预警系统")

# ==========================
# 股票分析
# ==========================
elif page == "📈 股票分析":
    st.title("📈 股票技术分析")
    
    # 股票输入
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ticker = st.text_input("输入股票代码", "AAPL").upper()
    
    with col2:
        period = st.selectbox(
            "时间周期",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3
        )
    
    if st.button("🔍 分析", type="primary"):
        with st.spinner("正在获取数据并分析..."):
            # 获取数据
            df = services['data'].fetch_stock_data(ticker, period)
            
            if df.empty:
                st.error(f"无法获取 {ticker} 的数据")
            else:
                # 保存数据
                services['data'].update_stock_prices(ticker)
                
                # 分析
                analysis = services['technical'].analyze_stock(ticker)
                
                if analysis:
                    # 显示分析结果
                    st.subheader(f"📊 {ticker} 分析结果")
                    
                    # 关键指标
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("当前价格", f"${analysis.current_price}", f"{analysis.change_pct}%")
                    with col2:
                        st.metric("SMA20", f"${analysis.sma_20}")
                    with col3:
                        st.metric("SMA50", f"${analysis.sma_50}")
                    with col4:
                        st.metric("RSI(14)", f"{analysis.rsi_14}")
                    
                    # 趋势和建议
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.info(f"**趋势**: {analysis.trend}")
                    with col2:
                        if "买入" in analysis.recommendation:
                            st.success(f"**建议**: {analysis.recommendation}")
                        elif "卖出" in analysis.recommendation:
                            st.error(f"**建议**: {analysis.recommendation}")
                        else:
                            st.warning(f"**建议**: {analysis.recommendation}")
                    
                    # 价格走势图
                    st.subheader("📈 价格走势")
                    
                    fig = go.Figure()
                    
                    # K线图
                    fig.add_trace(go.Candlestick(
                        x=df.index,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        name="K线"
                    ))
                    
                    # 移动平均线
                    close = df['Close']
                    sma_20 = close.rolling(20).mean()
                    sma_50 = close.rolling(50).mean()
                    
                    fig.add_trace(go.Scatter(
                        x=df.index, y=sma_20,
                        name="SMA20", line=dict(color='orange')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=df.index, y=sma_50,
                        name="SMA50", line=dict(color='blue')
                    ))
                    
                    fig.update_layout(
                        title=f"{ticker} 价格走势",
                        yaxis_title="价格 ($)",
                        xaxis_title="日期",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 成交量
                    st.subheader("📊 成交量")
                    
                    fig_vol = go.Figure()
                    fig_vol.add_trace(go.Bar(
                        x=df.index,
                        y=df['Volume'],
                        name="成交量",
                        marker_color='blue'
                    ))
                    
                    fig_vol.update_layout(
                        title=f"{ticker} 成交量",
                        yaxis_title="成交量",
                        height=300
                    )
                    
                    st.plotly_chart(fig_vol, use_container_width=True)
                    
                    # 技术指标
                    st.subheader("📉 技术指标")
                    
                    # RSI
                    rsi = services['technical'].calculate_rsi(close)
                    
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(
                        x=df.index, y=rsi,
                        name="RSI(14)", line=dict(color='purple')
                    ))
                    
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="超买")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="超卖")
                    
                    fig_rsi.update_layout(
                        title="RSI(14)",
                        yaxis_title="RSI",
                        height=250
                    )
                    
                    st.plotly_chart(fig_rsi, use_container_width=True)

# ==========================
# 投资组合
# ==========================
elif page == "💼 投资组合":
    st.title("💼 投资组合管理")
    
    tab1, tab2, tab3 = st.tabs(["📊 组合概览", "📝 交易操作", "📜 交易记录"])
    
    with tab1:
        st.subheader("组合列表")
        
        with PortfolioRepository() as repo:
            portfolios = repo.get_all()
        
        if not portfolios:
            st.info("暂无投资组合，请先创建")
        else:
            for portfolio in portfolios:
                with st.expander(f"📁 {portfolio.name}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**初始资金**: ${portfolio.initial_value:,.2f}")
                    with col2:
                        st.write(f"**当前价值**: ${portfolio.current_value:,.2f}")
                    with col3:
                        pnl = portfolio.current_value - portfolio.initial_value
                        pnl_pct = (pnl / portfolio.initial_value * 100) if portfolio.initial_value > 0 else 0
                        st.write(f"**盈亏**: ${pnl:,.2f} ({pnl_pct:.2f}%)")
                    
                    # 持仓详情
                    st.write("**持仓详情**:")
                    
                    holdings = services['portfolio'].get_holdings_detail(portfolio.id)
                    
                    if holdings:
                        df_holdings = pd.DataFrame(holdings)
                        st.dataframe(df_holdings, use_container_width=True)
                        
                        # 持仓分布图
                        fig = px.pie(
                            df_holdings, 
                            values='market_value', 
                            names='ticker',
                            title='持仓分布'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.write("暂无持仓")
    
    with tab2:
        st.subheader("📝 交易操作")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**买入股票**")
            
            with PortfolioRepository() as repo:
                portfolios = repo.get_all()
            
            if portfolios:
                portfolio_options = {f"{p.name} (ID: {p.id})": p.id for p in portfolios}
                selected = st.selectbox("选择组合", list(portfolio_options.keys()), key="buy_portfolio")
                portfolio_id = portfolio_options[selected]
                
                ticker_buy = st.text_input("股票代码", "AAPL", key="buy_ticker").upper()
                shares_buy = st.number_input("股数", min_value=1, value=100, key="buy_shares")
                price_buy = st.number_input("价格", min_value=0.01, value=150.0, key="buy_price")
                
                if st.button("🟢 买入", type="primary"):
                    try:
                        services['portfolio'].buy_stock(portfolio_id, ticker_buy, shares_buy, price_buy)
                        st.success(f"成功买入 {shares_buy} 股 {ticker_buy}")
                    except Exception as e:
                        st.error(f"买入失败: {e}")
            else:
                st.warning("请先创建投资组合")
        
        with col2:
            st.write("**卖出股票**")
            
            if portfolios:
                selected_sell = st.selectbox("选择组合", list(portfolio_options.keys()), key="sell_portfolio")
                portfolio_id_sell = portfolio_options[selected_sell]
                
                ticker_sell = st.text_input("股票代码", "AAPL", key="sell_ticker").upper()
                shares_sell = st.number_input("股数", min_value=1, value=100, key="sell_shares")
                price_sell = st.number_input("价格", min_value=0.01, value=150.0, key="sell_price")
                
                if st.button("🔴 卖出", type="primary"):
                    try:
                        services['portfolio'].sell_stock(portfolio_id_sell, ticker_sell, shares_sell, price_sell)
                        st.success(f"成功卖出 {shares_sell} 股 {ticker_sell}")
                    except Exception as e:
                        st.error(f"卖出失败: {e}")
            else:
                st.warning("请先创建投资组合")
        
        # 创建新组合
        st.divider()
        st.write("**创建新组合**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            new_name = st.text_input("组合名称", "我的组合")
        with col2:
            new_desc = st.text_input("描述", "")
        with col3:
            new_value = st.number_input("初始资金", min_value=0.0, value=100000.0)
        
        if st.button("➕ 创建组合"):
            try:
                portfolio_id = services['portfolio'].create_portfolio(new_name, new_desc, new_value)
                st.success(f"组合创建成功! ID: {portfolio_id}")
            except Exception as e:
                st.error(f"创建失败: {e}")
    
    with tab3:
        st.subheader("📜 交易记录")
        
        if portfolios:
            selected_record = st.selectbox("选择组合", list(portfolio_options.keys()), key="record_portfolio")
            portfolio_id_record = portfolio_options[selected_record]
            
            days = st.slider("查看天数", 7, 365, 30)
            
            with TradeRecordRepository() as repo:
                trades = repo.get_trades(portfolio_id_record, days)
            
            if trades:
                trade_data = [{
                    '日期': t.trade_date.strftime('%Y-%m-%d %H:%M'),
                    '股票': t.ticker,
                    '操作': t.action,
                    '股数': t.shares,
                    '价格': f"${t.price:.2f}",
                    '总金额': f"${t.total_amount:.2f}",
                    '备注': t.notes or ""
                } for t in trades]
                
                df_trades = pd.DataFrame(trade_data)
                st.dataframe(df_trades, use_container_width=True)
            else:
                st.info("暂无交易记录")
        else:
            st.info("暂无投资组合")

# ==========================
# 预警系统
# ==========================
elif page == "🔔 预警系统":
    st.title("🔔 预警系统")
    
    tab1, tab2 = st.tabs(["📋 预警列表", "➕ 新建预警"])
    
    with tab1:
        st.subheader("活跃预警")
        
        with AlertRepository() as repo:
            alerts = repo.get_active_alerts()
        
        if alerts:
            for alert in alerts:
                with st.expander(f"🔔 {alert.ticker} - {alert.alert_type}"):
                    st.write(f"**条件**: {alert.condition}")
                    st.write(f"**创建时间**: {alert.created_at}")
                    
                    if st.button(f"删除", key=f"del_{alert.id}"):
                        # TODO: 实现删除功能
                        st.rerun()
        else:
            st.info("暂无活跃预警")
    
    with tab2:
        st.subheader("➕ 新建价格预警")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            alert_ticker = st.text_input("股票代码", "AAPL").upper()
        with col2:
            alert_condition = st.selectbox("条件", [">", "<", ">=", "<="])
        with col3:
            alert_price = st.number_input("价格", min_value=0.01, value=150.0)
        
        if st.button("🔔 创建预警", type="primary"):
            try:
                services['alert'].create_price_alert(alert_ticker, alert_condition, alert_price)
                st.success(f"预警创建成功: {alert_ticker} {alert_condition} ${alert_price}")
            except Exception as e:
                st.error(f"创建失败: {e}")

# ==========================
# 数据管理
# ==========================
elif page == "⚙️ 数据管理":
    st.title("⚙️ 数据管理")
    
    tab1, tab2 = st.tabs(["🔄 数据更新", "📊 数据查询"])
    
    with tab1:
        st.subheader("🔄 批量更新数据")
        
        default_tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "NVDA", "TSLA"]
        
        tickers_input = st.text_area(
            "输入股票代码（每行一个）",
            value="\n".join(default_tickers),
            height=150
        )
        
        tickers = [t.strip().upper() for t in tickers_input.split("\n") if t.strip()]
        
        if st.button("🔄 开始更新", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, ticker in enumerate(tickers):
                status_text.text(f"正在更新 {ticker}... ({i+1}/{len(tickers)})")
                services['data'].update_stock_prices(ticker)
                progress_bar.progress((i + 1) / len(tickers))
            
            status_text.text("✅ 更新完成！")
            st.success(f"成功更新 {len(tickers)} 只股票的数据")
    
    with tab2:
        st.subheader("📊 数据查询")
        
        query_ticker = st.text_input("查询股票代码", "AAPL").upper()
        
        if st.button("🔍 查询"):
            with StockPriceRepository() as repo:
                df = repo.get_prices(query_ticker, days=365)
            
            if not df.empty:
                st.write(f"**{query_ticker}** 历史数据")
                st.dataframe(df.tail(20), use_container_width=True)
                
                # 价格走势图
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['Close'],
                    name="收盘价", line=dict(color='blue')
                ))
                
                fig.update_layout(
                    title=f"{query_ticker} 历史价格",
                    yaxis_title="价格 ($)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"未找到 {query_ticker} 的数据")

# 页脚
st.sidebar.divider()
st.sidebar.caption("© 2024 Quant Analysis System")
