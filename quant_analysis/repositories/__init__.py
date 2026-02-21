# repositories/__init__.py
"""数据访问层 - Repository模式"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import desc, and_
from models import SessionLocal, Stock, StockPrice, TechnicalIndicator, FinancialMetric, Portfolio, PortfolioHolding, TradeRecord, Alert


class BaseRepository:
    """基础仓库类"""
    
    def __init__(self):
        self.db = SessionLocal()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.db.rollback()
        else:
            self.db.commit()
        self.db.close()


class StockRepository(BaseRepository):
    """股票数据仓库"""
    
    def get_or_create(self, ticker: str, **kwargs) -> Stock:
        """获取或创建股票"""
        stock = self.db.query(Stock).filter(Stock.ticker == ticker).first()
        if not stock:
            stock = Stock(ticker=ticker, **kwargs)
            self.db.add(stock)
            self.db.commit()
        return stock
    
    def get_by_ticker(self, ticker: str) -> Optional[Stock]:
        """根据代码获取股票"""
        return self.db.query(Stock).filter(Stock.ticker == ticker).first()
    
    def get_all(self) -> List[Stock]:
        """获取所有股票"""
        return self.db.query(Stock).all()
    
    def get_by_sector(self, sector: str) -> List[Stock]:
        """根据行业获取股票"""
        return self.db.query(Stock).filter(Stock.sector == sector).all()


class StockPriceRepository(BaseRepository):
    """股票价格数据仓库"""
    
    def save_prices(self, ticker: str, df: pd.DataFrame):
        """保存价格数据"""
        for _, row in df.iterrows():
            price = StockPrice(
                ticker=ticker,
                date=row.name if isinstance(row.name, datetime) else pd.to_datetime(row.name),
                open=float(row['Open']),
                high=float(row['High']),
                low=float(row['Low']),
                close=float(row['Close']),
                volume=int(row['Volume']),
                adj_close=float(row.get('Adj Close', row['Close']))
            )
            self.db.merge(price)
        self.db.commit()
    
    def get_prices(self, ticker: str, days: int = 365) -> pd.DataFrame:
        """获取价格数据"""
        start_date = datetime.now() - timedelta(days=days)
        prices = self.db.query(StockPrice).filter(
            and_(StockPrice.ticker == ticker, StockPrice.date >= start_date)
        ).order_by(StockPrice.date).all()
        
        if not prices:
            return pd.DataFrame()
        
        data = [{
            'Date': p.date,
            'Open': p.open,
            'High': p.high,
            'Low': p.low,
            'Close': p.close,
            'Volume': p.volume,
            'Adj Close': p.adj_close
        } for p in prices]
        
        df = pd.DataFrame(data)
        df.set_index('Date', inplace=True)
        return df
    
    def get_latest_price(self, ticker: str) -> Optional[StockPrice]:
        """获取最新价格"""
        return self.db.query(StockPrice).filter(
            StockPrice.ticker == ticker
        ).order_by(desc(StockPrice.date)).first()


class TechnicalIndicatorRepository(BaseRepository):
    """技术指标仓库"""
    
    def save_indicators(self, ticker: str, date: datetime, indicators: Dict[str, float]):
        """保存技术指标"""
        indicator = TechnicalIndicator(
            ticker=ticker,
            date=date,
            sma_20=indicators.get('sma_20'),
            sma_50=indicators.get('sma_50'),
            sma_200=indicators.get('sma_200'),
            ema_12=indicators.get('ema_12'),
            ema_26=indicators.get('ema_26'),
            rsi_14=indicators.get('rsi_14'),
            macd=indicators.get('macd'),
            macd_signal=indicators.get('macd_signal'),
            bb_upper=indicators.get('bb_upper'),
            bb_middle=indicators.get('bb_middle'),
            bb_lower=indicators.get('bb_lower'),
            atr_14=indicators.get('atr_14')
        )
        self.db.merge(indicator)
        self.db.commit()
    
    def get_latest(self, ticker: str) -> Optional[TechnicalIndicator]:
        """获取最新技术指标"""
        return self.db.query(TechnicalIndicator).filter(
            TechnicalIndicator.ticker == ticker
        ).order_by(desc(TechnicalIndicator.date)).first()


class PortfolioRepository(BaseRepository):
    """投资组合仓库"""
    
    def create(self, name: str, description: str = "", initial_value: float = 0) -> Portfolio:
        """创建投资组合"""
        portfolio = Portfolio(
            name=name,
            description=description,
            initial_value=initial_value,
            current_value=initial_value
        )
        self.db.add(portfolio)
        self.db.commit()
        return portfolio
    
    def get_by_id(self, portfolio_id: int) -> Optional[Portfolio]:
        """根据ID获取组合"""
        return self.db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
    
    def get_all(self) -> List[Portfolio]:
        """获取所有组合"""
        return self.db.query(Portfolio).all()
    
    def update_value(self, portfolio_id: int, current_value: float):
        """更新组合价值"""
        portfolio = self.get_by_id(portfolio_id)
        if portfolio:
            portfolio.current_value = current_value
            self.db.commit()


class PortfolioHoldingRepository(BaseRepository):
    """持仓仓库"""
    
    def get_holdings(self, portfolio_id: int) -> List[PortfolioHolding]:
        """获取组合持仓"""
        return self.db.query(PortfolioHolding).filter(
            PortfolioHolding.portfolio_id == portfolio_id
        ).all()
    
    def get_holding(self, portfolio_id: int, ticker: str) -> Optional[PortfolioHolding]:
        """获取特定持仓"""
        return self.db.query(PortfolioHolding).filter(
            and_(PortfolioHolding.portfolio_id == portfolio_id, 
                 PortfolioHolding.ticker == ticker)
        ).first()
    
    def update_holding(self, portfolio_id: int, ticker: str, shares: int, avg_cost: float, current_price: float):
        """更新持仓"""
        holding = self.get_holding(portfolio_id, ticker)
        if holding:
            holding.shares = shares
            holding.avg_cost = avg_cost
            holding.current_price = current_price
        else:
            holding = PortfolioHolding(
                portfolio_id=portfolio_id,
                ticker=ticker,
                shares=shares,
                avg_cost=avg_cost,
                current_price=current_price
            )
            self.db.add(holding)
        self.db.commit()


class TradeRecordRepository(BaseRepository):
    """交易记录仓库"""
    
    def record_trade(self, portfolio_id: int, ticker: str, action: str, 
                     shares: int, price: float, notes: str = ""):
        """记录交易"""
        trade = TradeRecord(
            portfolio_id=portfolio_id,
            ticker=ticker,
            action=action,
            shares=shares,
            price=price,
            total_amount=shares * price,
            notes=notes
        )
        self.db.add(trade)
        self.db.commit()
    
    def get_trades(self, portfolio_id: int, days: int = 30) -> List[TradeRecord]:
        """获取交易记录"""
        start_date = datetime.now() - timedelta(days=days)
        return self.db.query(TradeRecord).filter(
            and_(TradeRecord.portfolio_id == portfolio_id,
                 TradeRecord.trade_date >= start_date)
        ).order_by(desc(TradeRecord.trade_date)).all()


class AlertRepository(BaseRepository):
    """预警仓库"""
    
    def create_alert(self, ticker: str, alert_type: str, condition: str):
        """创建预警"""
        alert = Alert(
            ticker=ticker,
            alert_type=alert_type,
            condition=condition
        )
        self.db.add(alert)
        self.db.commit()
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃预警"""
        return self.db.query(Alert).filter(Alert.is_triggered == 0).all()
    
    def trigger_alert(self, alert_id: int):
        """触发预警"""
        alert = self.db.query(Alert).filter(Alert.id == alert_id).first()
        if alert:
            alert.is_triggered = 1
            alert.triggered_at = datetime.now()
            self.db.commit()
