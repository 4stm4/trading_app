"""Add trading system storage tables.

Revision ID: 20260223_000003
Revises: 20260223_000002
Create Date: 2026-02-23 18:30:00
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "20260223_000003"
down_revision: Union[str, None] = "20260223_000002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "trading_systems",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("owner_user_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("exchange", sa.String(length=32), server_default="moex", nullable=False),
        sa.Column("engine", sa.String(length=32), server_default="stock", nullable=False),
        sa.Column("market", sa.String(length=32), server_default="shares", nullable=False),
        sa.Column("board", sa.String(length=32), server_default="", nullable=False),
        sa.Column("timeframe", sa.String(length=16), server_default="1h", nullable=False),
        sa.Column("model_name", sa.String(length=64), server_default="balanced", nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["owner_user_id"], ["auth_users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("owner_user_id", "name", name="uq_trading_systems_owner_name"),
    )
    op.create_index(
        "ix_trading_systems_owner_active",
        "trading_systems",
        ["owner_user_id", "is_active"],
        unique=False,
    )

    op.create_table(
        "trading_system_versions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("system_id", sa.Integer(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("config_json", sa.JSON(), nullable=False),
        sa.Column("is_current", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("created_by_user_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["created_by_user_id"], ["auth_users.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["system_id"], ["trading_systems.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("system_id", "version", name="uq_trading_system_versions_system_version"),
    )
    op.create_index("ix_trading_system_versions_system_id", "trading_system_versions", ["system_id"], unique=False)
    op.create_index(
        "uq_trading_system_versions_current",
        "trading_system_versions",
        ["system_id"],
        unique=True,
        postgresql_where=sa.text("is_current"),
    )

    op.create_table(
        "trading_system_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("system_id", sa.Integer(), nullable=False),
        sa.Column("system_version_id", sa.Integer(), nullable=True),
        sa.Column("run_type", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), server_default="pending", nullable=False),
        sa.Column("request_json", sa.JSON(), nullable=True),
        sa.Column("result_summary_json", sa.JSON(), nullable=True),
        sa.Column("error_text", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["system_id"], ["trading_systems.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["system_version_id"], ["trading_system_versions.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_trading_system_runs_system_created",
        "trading_system_runs",
        ["system_id", "created_at"],
        unique=False,
    )
    op.create_index("ix_trading_system_runs_status", "trading_system_runs", ["status"], unique=False)
    op.create_index(
        "ix_trading_system_runs_type_status",
        "trading_system_runs",
        ["run_type", "status"],
        unique=False,
    )

    op.create_table(
        "trading_system_signals",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("system_id", sa.Integer(), nullable=False),
        sa.Column("system_version_id", sa.Integer(), nullable=True),
        sa.Column("exchange", sa.String(length=32), nullable=False),
        sa.Column("symbol", sa.String(length=64), nullable=False),
        sa.Column("timeframe", sa.String(length=16), server_default="1h", nullable=False),
        sa.Column("model_name", sa.String(length=64), server_default="balanced", nullable=False),
        sa.Column("signal", sa.String(length=16), server_default="none", nullable=False),
        sa.Column("confidence", sa.String(length=16), server_default="none", nullable=False),
        sa.Column("entry", sa.Float(), nullable=True),
        sa.Column("stop", sa.Float(), nullable=True),
        sa.Column("target", sa.Float(), nullable=True),
        sa.Column("rr", sa.Float(), nullable=True),
        sa.Column("market_regime", sa.String(length=32), nullable=True),
        sa.Column("phase", sa.String(length=32), nullable=True),
        sa.Column("generated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["system_id"], ["trading_systems.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["system_version_id"], ["trading_system_versions.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_trading_system_signals_system_generated",
        "trading_system_signals",
        ["system_id", "generated_at"],
        unique=False,
    )
    op.create_index(
        "ix_trading_system_signals_lookup",
        "trading_system_signals",
        ["exchange", "symbol", "timeframe", "generated_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_trading_system_signals_lookup", table_name="trading_system_signals")
    op.drop_index("ix_trading_system_signals_system_generated", table_name="trading_system_signals")
    op.drop_table("trading_system_signals")

    op.drop_index("ix_trading_system_runs_type_status", table_name="trading_system_runs")
    op.drop_index("ix_trading_system_runs_status", table_name="trading_system_runs")
    op.drop_index("ix_trading_system_runs_system_created", table_name="trading_system_runs")
    op.drop_table("trading_system_runs")

    op.drop_index("uq_trading_system_versions_current", table_name="trading_system_versions")
    op.drop_index("ix_trading_system_versions_system_id", table_name="trading_system_versions")
    op.drop_table("trading_system_versions")

    op.drop_index("ix_trading_systems_owner_active", table_name="trading_systems")
    op.drop_table("trading_systems")
