from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, UniqueConstraint, func, text
from sqlalchemy.orm import Mapped, mapped_column

from ..base import Base


class TradingSystemScanTable(Base):
    __tablename__ = "trading_system_scans"
    __table_args__ = (
        UniqueConstraint(
            "owner_user_id",
            "scan_key",
            "exchange",
            "symbol",
            "timeframe",
            "model_name",
            name="uq_trading_system_scans_batch_symbol_tf_model",
        ),
        Index("ix_trading_system_scans_owner_created", "owner_user_id", "created_at"),
        Index("ix_trading_system_scans_system_created", "system_id", "created_at"),
        Index(
            "ix_trading_system_scans_lookup",
            "owner_user_id",
            "exchange",
            "symbol",
            "timeframe",
            "model_name",
            "created_at",
        ),
        Index("ix_trading_system_scans_scan_key", "owner_user_id", "scan_key", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    owner_user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("auth_users.id", ondelete="CASCADE"),
        nullable=False,
    )
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
    scan_key: Mapped[str] = mapped_column(String(64), nullable=False)
    exchange: Mapped[str] = mapped_column(String(32), nullable=False)
    engine: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="stock",
        server_default="stock",
    )
    market: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="shares",
        server_default="shares",
    )
    board: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="",
        server_default="",
    )
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
    tradable: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )
    entry: Mapped[float | None] = mapped_column(Float, nullable=True)
    stop: Mapped[float | None] = mapped_column(Float, nullable=True)
    target: Mapped[float | None] = mapped_column(Float, nullable=True)
    rr: Mapped[float | None] = mapped_column(Float, nullable=True)
    market_regime: Mapped[str | None] = mapped_column(String(32), nullable=True)
    phase: Mapped[str | None] = mapped_column(String(32), nullable=True)
    issues_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
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
