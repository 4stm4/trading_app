"""Add market candles table.

Revision ID: 20260223_000002
Revises: 20260223_000001
Create Date: 2026-02-23 16:00:00
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "20260223_000002"
down_revision: Union[str, None] = "20260223_000001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "market_candles",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("exchange", sa.String(length=32), nullable=False),
        sa.Column("engine", sa.String(length=32), nullable=False),
        sa.Column("market", sa.String(length=32), nullable=False),
        sa.Column("board", sa.String(length=32), server_default="", nullable=False),
        sa.Column("symbol", sa.String(length=64), nullable=False),
        sa.Column("timeframe", sa.String(length=16), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=False), nullable=False),
        sa.Column("open", sa.Float(), nullable=False),
        sa.Column("high", sa.Float(), nullable=False),
        sa.Column("low", sa.Float(), nullable=False),
        sa.Column("close", sa.Float(), nullable=False),
        sa.Column("volume", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "exchange",
            "engine",
            "market",
            "board",
            "symbol",
            "timeframe",
            "timestamp",
            name="uq_market_candles_key",
        ),
    )
    op.create_index(
        "ix_market_candles_lookup",
        "market_candles",
        ["exchange", "engine", "market", "board", "symbol", "timeframe", "timestamp"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_market_candles_lookup", table_name="market_candles")
    op.drop_table("market_candles")
