from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, JSON, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column

from ..base import Base


class TradingSystemVersionTable(Base):
    __tablename__ = "trading_system_versions"
    __table_args__ = (
        UniqueConstraint("system_id", "version", name="uq_trading_system_versions_system_version"),
        Index("ix_trading_system_versions_system_id", "system_id"),
        Index(
            "uq_trading_system_versions_current",
            "system_id",
            unique=True,
            postgresql_where=text("is_current"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    system_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("trading_systems.id", ondelete="CASCADE"),
        nullable=False,
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    config_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    is_current: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default="true",
    )
    created_by_user_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("auth_users.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("now()"),
    )
