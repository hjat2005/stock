# services/__init__.py
"""业务逻辑层 - Service模式"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from repositories import (
    StockRepository, StockPriceRepository, TechnicalIndicatorRepository,
    PortfolioRepository, PortfolioHoldingRepository, TradeRecordRepository,
    AlertRepository
)


@dataclass
class StockAnalysis:
    """股票分析结果"""
    ticker: str
    current_price: float
    change_pct: float
    sma_20: float
    sma_50: float
    rsi_14: float
    trend: str
    recommendation: str


@dataclass
class PortfolioPerformance:
    """组合绩效"""
    portfolio_id: int
    total_value: float
    total_cost: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    holdings_count: int
    daily_return: float


class DataService:
    """数据服务 - 负责数据获取和ETL"""
    
    def __init__(self):
        self.stock_repo = StockRepository()
        self.price_repo = StockPriceRepository()
    
    def fetch_stock_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """从Yahoo Finance获取股票数据"""
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            if not df.empty:
                # 标准化列名
                df.columns = [col.replace(" ", "_").title() for col in df.columns]
                return df
        except Exception as e:
            print(f"获取 {ticker} 数据失败: {e}")
        return pd.DataFrame()
    
    def update_stock_prices(self, ticker: str):
        """更新股票价格数据"""
        df = self.fetch_stock_data(ticker)
        if not df.empty:
            with self.price_repo:
                self.price_repo.save_prices(ticker, df)
            print(f"✅ {ticker} 价格数据已更新")
    
    def batch_update_prices(self, tickers: List[str]):
        """批量更新价格"""
        for ticker in tickers:
            self.update_stock_prices(ticker)


class TechnicalAnalysisService:
    """技术分析服务"""
    
    def __init__(self):
        self.price_repo = StockPriceRepository()
        self.indicator_repo = TechnicalIndicatorRepository()
    
    def calculate_sma(self, prices: pd.Series, window: int) -> pd.Series:
        """计算简单移动平均线"""
        return prices.rolling(window=window).mean()
    
    def calculate_ema(self, prices: pd.Series, span: int) -> pd.Series:
        """计算指数移动平均线"""
        return prices.ewm(span=span, adjust=False).mean()
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算MACD指标"""
        ema_12 = self.calculate_ema(prices, 12)
        ema_26 = self.calculate_ema(prices, 26)
        macd = ema_12 - ema_26
        signal = self.calculate_ema(macd, 9)
        histogram = macd - signal
        return macd, signal, histogram
    
    def calculate_bollinger_bands(self, prices: pd.Series, window: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算布林带"""
        sma = self.calculate_sma(prices, window)
        std = prices.rolling(window=window).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        return upper, sma, lower
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算ATR指标"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(period).mean()
        return atr
    
    def analyze_stock(self, ticker: str) -> Optional[StockAnalysis]:
        """分析股票"""
        with self.price_repo:
            df = self.price_repo.get_prices(ticker, days=365)
        
        if df.empty or len(df) < 50:
            return None
        
        # 计算指标
        close = df['Close']
        current_price = close.iloc[-1]
        prev_price = close.iloc[-2]
        change_pct = ((current_price - prev_price) / prev_price) * 100
        
        sma_20 = self.calculate_sma(close, 20).iloc[-1]
        sma_50 = self.calculate_sma(close, 50).iloc[-1]
        rsi_14 = self.calculate_rsi(close, 14).iloc[-1]
        
        # 趋势判断
        if current_price > sma_20 > sma_50:
            trend = "上升趋势"
        elif current_price < sma_20 < sma_50:
            trend = "下降趋势"
        else:
            trend = "震荡整理"
        
        # 建议
        if rsi_14 < 30:
            recommendation = "超卖 - 考虑买入"
        elif rsi_14 > 70:
            recommendation = "超买 - 考虑卖出"
        elif current_price > sma_20:
            recommendation = "强势 - 持有"
        else:
            recommendation = "弱势 - 观望"
        
        return StockAnalysis(
            ticker=ticker,
            current_price=round(current_price, 2),
            change_pct=round(change_pct, 2),
            sma_20=round(sma_20, 2),
            sma_50=round(sma_50, 2),
            rsi_14=round(rsi_14, 2),
            trend=trend,
            recommendation=recommendation
        )
    
    def save_indicators(self, ticker: str):
        """计算并保存技术指标"""
        with self.price_repo:
            df = self.price_repo.get_prices(ticker, days=365)
        
        if df.empty:
            return
        
        close = df['Close']
        latest_date = df.index[-1]
        
        indicators = {
            'sma_20': self.calculate_sma(close, 20).iloc[-1],
            'sma_50': self.calculate_sma(close, 50).iloc[-1],
            'sma_200': self.calculate_sma(close, 200).iloc[-1] if len(close) >= 200 else None,
            'ema_12': self.calculate_ema(close, 12).iloc[-1],
            'ema_26': self.calculate_ema(close, 26).iloc[-1],
            'rsi_14': self.calculate_rsi(close, 14).iloc[-1],
        }
        
        macd, signal, _ = self.calculate_macd(close)
        indicators['macd'] = macd.iloc[-1]
        indicators['macd_signal'] = signal.iloc[-1]
        
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
        indicators['bb_upper'] = bb_upper.iloc[-1]
        indicators['bb_middle'] = bb_middle.iloc[-1]
        indicators['bb_lower'] = bb_lower.iloc[-1]
        
        indicators['atr_14'] = self.calculate_atr(df).iloc[-1]
        
        with self.indicator_repo:
            self.indicator_repo.save_indicators(ticker, latest_date, indicators)


class PortfolioService:
    """投资组合服务"""
    
    def __init__(self):
        self.portfolio_repo = PortfolioRepository()
        self.holding_repo = PortfolioHoldingRepository()
        self.trade_repo = TradeRecordRepository()
        self.price_repo = StockPriceRepository()
    
    def create_portfolio(self, name: str, description: str = "", initial_value: float = 0) -> int:
        """创建投资组合"""
        with self.portfolio_repo:
            portfolio = self.portfolio_repo.create(name, description, initial_value)
            return portfolio.id
    
    def buy_stock(self, portfolio_id: int, ticker: str, shares: int, price: float):
        """买入股票"""
        with self.holding_repo:
            holding = self.holding_repo.get_holding(portfolio_id, ticker)
            
            if holding and holding.shares > 0:
                # 计算新的平均成本
                total_cost = (holding.shares * holding.avg_cost) + (shares * price)
                total_shares = holding.shares + shares
                new_avg_cost = total_cost / total_shares
            else:
                total_shares = shares
                new_avg_cost = price
            
            self.holding_repo.update_holding(
                portfolio_id, ticker, total_shares, new_avg_cost, price
            )
        
        # 记录交易
        with self.trade_repo:
            self.trade_repo.record_trade(
                portfolio_id, ticker, "BUY", shares, price, 
                notes=f"买入 {shares} 股 @ ${price}"
            )
        
        self._update_portfolio_value(portfolio_id)
    
    def sell_stock(self, portfolio_id: int, ticker: str, shares: int, price: float):
        """卖出股票"""
        with self.holding_repo:
            holding = self.holding_repo.get_holding(portfolio_id, ticker)
            
            if not holding or holding.shares < shares:
                raise ValueError(f"持仓不足: 当前 {holding.shares if holding else 0} 股, 尝试卖出 {shares} 股")
            
            remaining = holding.shares - shares
            if remaining > 0:
                self.holding_repo.update_holding(
                    portfolio_id, ticker, remaining, holding.avg_cost, price
                )
            else:
                # 清仓
                self.holding_repo.update_holding(
                    portfolio_id, ticker, 0, 0, price
                )
        
        # 记录交易
        with self.trade_repo:
            self.trade_repo.record_trade(
                portfolio_id, ticker, "SELL", shares, price,
                notes=f"卖出 {shares} 股 @ ${price}"
            )
        
        self._update_portfolio_value(portfolio_id)
    
    def _update_portfolio_value(self, portfolio_id: int):
        """更新组合价值"""
        with self.holding_repo:
            holdings = self.holding_repo.get_holdings(portfolio_id)
        
        total_value = 0
        for holding in holdings:
            if holding.shares > 0:
                # 获取最新价格
                latest_price = self._get_latest_price(holding.ticker)
                if latest_price:
                    total_value += holding.shares * latest_price
        
        with self.portfolio_repo:
            portfolio = self.portfolio_repo.get_by_id(portfolio_id)
            if portfolio:
                self.portfolio_repo.update_value(portfolio_id, total_value)
    
    def _get_latest_price(self, ticker: str) -> Optional[float]:
        """获取最新价格"""
        try:
            with self.price_repo:
                price_data = self.price_repo.get_latest_price(ticker)
                if price_data:
                    return price_data.close
            
            # 从Yahoo获取
            stock = yf.Ticker(ticker)
            return stock.info.get('regularMarketPrice')
        except:
            return None
    
    def get_performance(self, portfolio_id: int) -> Optional[PortfolioPerformance]:
        """获取组合绩效"""
        with self.portfolio_repo:
            portfolio = self.portfolio_repo.get_by_id(portfolio_id)
        
        if not portfolio:
            return None
        
        with self.holding_repo:
            holdings = self.holding_repo.get_holdings(portfolio_id)
        
        total_value = 0
        total_cost = 0
        
        for holding in holdings:
            if holding.shares > 0:
                latest_price = self._get_latest_price(holding.ticker)
                if latest_price:
                    total_value += holding.shares * latest_price
                    total_cost += holding.shares * holding.avg_cost
        
        unrealized_pnl = total_value - total_cost
        unrealized_pnl_pct = (unrealized_pnl / total_cost * 100) if total_cost > 0 else 0
        
        return PortfolioPerformance(
            portfolio_id=portfolio_id,
            total_value=round(total_value, 2),
            total_cost=round(total_cost, 2),
            unrealized_pnl=round(unrealized_pnl, 2),
            unrealized_pnl_pct=round(unrealized_pnl_pct, 2),
            holdings_count=len([h for h in holdings if h.shares > 0]),
            daily_return=0  # TODO: 计算日收益
        )
    
    def get_holdings_detail(self, portfolio_id: int) -> List[Dict]:
        """获取持仓详情"""
        with self.holding_repo:
            holdings = self.holding_repo.get_holdings(portfolio_id)
        
        details = []
        for holding in holdings:
            if holding.shares > 0:
                current_price = self._get_latest_price(holding.ticker)
                if current_price:
                    market_value = holding.shares * current_price
                    cost_basis = holding.shares * holding.avg_cost
                    pnl = market_value - cost_basis
                    pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0
                    
                    details.append({
                        'ticker': holding.ticker,
                        'shares': holding.shares,
                        'avg_cost': round(holding.avg_cost, 2),
                        'current_price': round(current_price, 2),
                        'market_value': round(market_value, 2),
                        'pnl': round(pnl, 2),
                        'pnl_pct': round(pnl_pct, 2)
                    })
        
        return details


class AlertService:
    """预警服务"""
    
    def __init__(self):
        self.alert_repo = AlertRepository()
        self.price_repo = StockPriceRepository()
    
    def create_price_alert(self, ticker: str, condition: str, threshold: float):
        """创建价格预警"""
        condition_str = f"{condition} ${threshold}"
        with self.alert_repo:
            self.alert_repo.create_alert(ticker, "PRICE", condition_str)
    
    def check_alerts(self):
        """检查预警条件"""
        with self.alert_repo:
            alerts = self.alert_repo.get_active_alerts()
        
        triggered = []
        for alert in alerts:
            # 获取最新价格
            with self.price_repo:
                price_data = self.price_repo.get_latest_price(alert.ticker)
            
            if price_data:
                # 简单条件检查（实际应解析condition）
                triggered.append(alert)
        
        return triggered
