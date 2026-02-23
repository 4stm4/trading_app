from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from ..base import Base


class TradingSystemSignalTable(Base):
    __tablename__ = "trading_system_signals"
    __table_args__ = (
        Index("ix_trading_system_signals_system_generated", "system_id", "generated_at"),
        Index("ix_trading_system_signals_lookup", "exchange", "symbol", "timeframe", "generated_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    system_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("trading_systems.id", ondelete="CASCADE"),
        nullable=False,
    )
    system_version_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("trading_system_versions.id", ondelete="SET NULL"),
        nullable=True,
    )
    exchange: Mapped[str] = mapped_column(String(32), nullable=False)
    symbol: Mapped[str] = mapped_column(String(64), nullable=False)
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
    signal: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="none",
        server_default="none",
    )
    confidence: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="none",
        server_default="none",
    )
    entry: Mapped[float | None] = mapped_column(Float, nullable=True)
    stop: Mapped[float | None] = mapped_column(Float, nullable=True)
    target: Mapped[float | None] = mapped_column(Float, nullable=True)
    rr: Mapped[float | None] = mapped_column(Float, nullable=True)
    market_regime: Mapped[str | None] = mapped_column(String(32), nullable=True)
    phase: Mapped[str | None] = mapped_column(String(32), nullable=True)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
