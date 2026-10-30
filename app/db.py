"""Database connection: SQLite out of the box, PostgreSQL in Docker."""
import os
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./fraud.db")
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite:") else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

class Base(DeclarativeBase):
    pass

class Transaction(Base):
    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String(40), unique=True, index=True, nullable=False)
    amount = Column(Float, nullable=False)
    hour = Column(Integer, nullable=False)
    distance_from_home = Column(Float, nullable=False)
    is_foreign = Column(Boolean, nullable=False)
    is_online = Column(Boolean, nullable=False)
    velocity_24h = Column(Integer, nullable=False)
    account_age_days = Column(Integer, nullable=False)
    merchant_risk = Column(Float, nullable=False)
    risk_score = Column(Float, nullable=False)
    status = Column(String(16), nullable=False)
    source = Column(String(16), nullable=False, default="API")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
