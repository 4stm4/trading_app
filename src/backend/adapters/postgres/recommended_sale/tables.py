from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from ..base import Base


class RecommendedSaleTable(Base):
    __tablename__ = "recommended_sales"
    __table_args__ = (
        Index("ix_recommended_sales_exchange_symbol_tf", "exchange", "symbol", "timeframe"),
        Index("ix_recommended_sales_status", "status"),
        Index("ix_recommended_sales_recommended_at", "recommended_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    exchange: Mapped[str] = mapped_column(String(32), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    timeframe: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="1h",
        server_default="1h",
    )
    model_name: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="balanced",
        server_default="balanced",
    )
    entry: Mapped[float | None] = mapped_column(Float, nullable=True)
    stop: Mapped[float | None] = mapped_column(Float, nullable=True)
    target: Mapped[float | None] = mapped_column(Float, nullable=True)
    rr: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="none",
        server_default="none",
    )
    market_regime: Mapped[str | None] = mapped_column(String(32), nullable=True)
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="new",
        server_default="new",
    )
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    recommended_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
