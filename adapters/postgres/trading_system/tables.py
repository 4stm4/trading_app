from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from ..base import Base


class TradingSystemTable(Base):
    __tablename__ = "trading_systems"
    __table_args__ = (
        UniqueConstraint("owner_user_id", "name", name="uq_trading_systems_owner_name"),
        Index("ix_trading_systems_owner_active", "owner_user_id", "is_active"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    owner_user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("auth_users.id", ondelete="CASCADE"),
        nullable=False,
    )
    model_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("trading_models.id", ondelete="SET NULL"),
        nullable=True,
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    exchange: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="moex",
        server_default="moex",
    )
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
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default="true",
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
