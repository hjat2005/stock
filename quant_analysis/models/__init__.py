# models/__init__.py
"""数据模型层 - SQLAlchemy模型定义"""

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

Base = declarative_base()

# 数据库路径
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'quant.db')
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

engine = create_engine(f'sqlite:///{DB_PATH}', echo=False)
SessionLocal = sessionmaker(bind=engine)


class Stock(Base):
    """股票基础信息模型"""
    __tablename__ = 'stocks'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String(20), unique=True, nullable=False, index=True)
    name = Column(String(100))
    sector = Column(String(50))
    industry = Column(String(50))
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class StockPrice(Base):
    """股票价格数据模型"""
    __tablename__ = 'stock_prices'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String(20), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)
    adj_close = Column(Float)
    created_at = Column(DateTime, default=datetime.now)
    
    __table_args__ = (
        # 联合唯一约束
        {'sqlite_autoincrement': True},
    )


class TechnicalIndicator(Base):
    """技术指标模型"""
    __tablename__ = 'technical_indicators'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String(20), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    sma_20 = Column(Float)
    sma_50 = Column(Float)
    sma_200 = Column(Float)
    ema_12 = Column(Float)
    ema_26 = Column(Float)
    rsi_14 = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    bb_upper = Column(Float)
    bb_middle = Column(Float)
    bb_lower = Column(Float)
    atr_14 = Column(Float)
    created_at = Column(DateTime, default=datetime.now)


class FinancialMetric(Base):
    """财务指标模型"""
    __tablename__ = 'financial_metrics'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String(20), nullable=False, index=True)
    date = Column(DateTime, nullable=False)
    pe_ratio = Column(Float)
    pb_ratio = Column(Float)
    ps_ratio = Column(Float)
    ev_ebitda = Column(Float)
    debt_to_equity = Column(Float)
    current_ratio = Column(Float)
    roe = Column(Float)
    roa = Column(Float)
    gross_margin = Column(Float)
    operating_margin = Column(Float)
    net_margin = Column(Float)
    created_at = Column(DateTime, default=datetime.now)


class Portfolio(Base):
    """投资组合模型"""
    __tablename__ = 'portfolios'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    initial_value = Column(Float, default=0)
    current_value = Column(Float, default=0)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class PortfolioHolding(Base):
    """持仓模型"""
    __tablename__ = 'portfolio_holdings'
    
    id = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, nullable=False, index=True)
    ticker = Column(String(20), nullable=False)
    shares = Column(Integer, default=0)
    avg_cost = Column(Float, default=0)
    current_price = Column(Float, default=0)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class TradeRecord(Base):
    """交易记录模型"""
    __tablename__ = 'trade_records'
    
    id = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, nullable=False, index=True)
    ticker = Column(String(20), nullable=False)
    action = Column(String(10), nullable=False)  # BUY/SELL
    shares = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    total_amount = Column(Float, nullable=False)
    trade_date = Column(DateTime, default=datetime.now)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.now)


class Alert(Base):
    """预警模型"""
    __tablename__ = 'alerts'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String(20), nullable=False, index=True)
    alert_type = Column(String(50), nullable=False)  # PRICE, TECHNICAL, FUNDAMENTAL
    condition = Column(String(200), nullable=False)
    is_triggered = Column(Integer, default=0)
    triggered_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.now)


# 初始化数据库
def init_db():
    """初始化数据库表"""
    Base.metadata.create_all(bind=engine)
    print(f"✅ 数据库初始化完成: {DB_PATH}")


if __name__ == "__main__":
    init_db()
