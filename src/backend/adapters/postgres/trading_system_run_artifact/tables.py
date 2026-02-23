from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Index, Integer, JSON, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from ..base import Base


class TradingSystemRunArtifactTable(Base):
    __tablename__ = "trading_system_run_artifacts"
    __table_args__ = (
        UniqueConstraint("run_id", "artifact_type", name="uq_run_artifacts_run_type"),
        Index("ix_run_artifacts_owner_created", "owner_user_id", "created_at"),
        Index("ix_run_artifacts_run", "run_id", "created_at"),
        Index("ix_run_artifacts_system", "system_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    owner_user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("auth_users.id", ondelete="CASCADE"),
        nullable=False,
    )
    run_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("trading_system_runs.id", ondelete="CASCADE"),
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
    artifact_type: Mapped[str] = mapped_column(String(64), nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
