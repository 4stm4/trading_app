"""Add trading system scans storage.

Revision ID: 20260224_000008
Revises: 20260223_000007
Create Date: 2026-02-24 13:20:00
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "20260224_000008"
down_revision: Union[str, None] = "20260223_000007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "trading_system_scans",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("owner_user_id", sa.Integer(), nullable=False),
        sa.Column("system_id", sa.Integer(), nullable=False),
        sa.Column("system_version_id", sa.Integer(), nullable=True),
        sa.Column("scan_key", sa.String(length=64), nullable=False),
        sa.Column("exchange", sa.String(length=32), nullable=False),
        sa.Column("engine", sa.String(length=32), server_default="stock", nullable=False),
        sa.Column("market", sa.String(length=32), server_default="shares", nullable=False),
        sa.Column("board", sa.String(length=32), server_default="", nullable=False),
        sa.Column("symbol", sa.String(length=64), nullable=False),
        sa.Column("timeframe", sa.String(length=16), server_default="1h", nullable=False),
        sa.Column("model_name", sa.String(length=64), server_default="balanced", nullable=False),
        sa.Column("signal", sa.String(length=16), server_default="none", nullable=False),
        sa.Column("confidence", sa.String(length=16), server_default="none", nullable=False),
        sa.Column("tradable", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("entry", sa.Float(), nullable=True),
        sa.Column("stop", sa.Float(), nullable=True),
        sa.Column("target", sa.Float(), nullable=True),
        sa.Column("rr", sa.Float(), nullable=True),
        sa.Column("market_regime", sa.String(length=32), nullable=True),
        sa.Column("phase", sa.String(length=32), nullable=True),
        sa.Column("issues_json", sa.JSON(), nullable=True),
        sa.Column("generated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["owner_user_id"], ["auth_users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["system_id"], ["trading_systems.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["system_version_id"], ["trading_system_versions.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "owner_user_id",
            "scan_key",
            "exchange",
            "symbol",
            "timeframe",
            "model_name",
            name="uq_trading_system_scans_batch_symbol_tf_model",
        ),
    )
    op.create_index(
        "ix_trading_system_scans_owner_created",
        "trading_system_scans",
        ["owner_user_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_trading_system_scans_system_created",
        "trading_system_scans",
        ["system_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_trading_system_scans_lookup",
        "trading_system_scans",
        ["owner_user_id", "exchange", "symbol", "timeframe", "model_name", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_trading_system_scans_scan_key",
        "trading_system_scans",
        ["owner_user_id", "scan_key", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_trading_system_scans_scan_key", table_name="trading_system_scans")
    op.drop_index("ix_trading_system_scans_lookup", table_name="trading_system_scans")
    op.drop_index("ix_trading_system_scans_system_created", table_name="trading_system_scans")
    op.drop_index("ix_trading_system_scans_owner_created", table_name="trading_system_scans")
    op.drop_table("trading_system_scans")
