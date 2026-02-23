from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from ..base import Base


class PortfolioTable(Base):
    __tablename__ = "portfolios"
    __table_args__ = (
        Index("uq_portfolios_owner_user_id", "owner_user_id", unique=True),
        Index("ix_portfolios_owner_active", "owner_user_id", "is_active"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    owner_user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("auth_users.id", ondelete="CASCADE"),
        nullable=False,
    )
    balance: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=100000.0,
        server_default="100000",
    )
    currency: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="RUB",
        server_default="RUB",
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
