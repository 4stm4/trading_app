"""Initial postgres schema.

Revision ID: 20260223_000001
Revises:
Create Date: 2026-02-23 00:00:01
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "20260223_000001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "auth_users",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_auth_users_email", "auth_users", ["email"], unique=True)

    op.create_table(
        "instruments",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("exchange", sa.String(length=32), nullable=False),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("board", sa.String(length=32), nullable=True),
        sa.Column("lot_size", sa.Integer(), nullable=True),
        sa.Column("currency", sa.String(length=16), server_default="RUB", nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("exchange", "symbol", name="uq_instruments_exchange_symbol"),
    )
    op.create_index("ix_instruments_exchange_board", "instruments", ["exchange", "board"], unique=False)

    op.create_table(
        "recommended_sales",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("exchange", sa.String(length=32), nullable=False),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("timeframe", sa.String(length=16), server_default="1h", nullable=False),
        sa.Column("model_name", sa.String(length=64), server_default="balanced", nullable=False),
        sa.Column("entry", sa.Float(), nullable=True),
        sa.Column("stop", sa.Float(), nullable=True),
        sa.Column("target", sa.Float(), nullable=True),
        sa.Column("rr", sa.Float(), nullable=True),
        sa.Column("confidence", sa.String(length=32), server_default="none", nullable=False),
        sa.Column("market_regime", sa.String(length=32), nullable=True),
        sa.Column("status", sa.String(length=32), server_default="new", nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column(
            "recommended_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_recommended_sales_exchange_symbol_tf",
        "recommended_sales",
        ["exchange", "symbol", "timeframe"],
        unique=False,
    )
    op.create_index("ix_recommended_sales_recommended_at", "recommended_sales", ["recommended_at"], unique=False)
    op.create_index("ix_recommended_sales_status", "recommended_sales", ["status"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_recommended_sales_status", table_name="recommended_sales")
    op.drop_index("ix_recommended_sales_recommended_at", table_name="recommended_sales")
    op.drop_index("ix_recommended_sales_exchange_symbol_tf", table_name="recommended_sales")
    op.drop_table("recommended_sales")

    op.drop_index("ix_instruments_exchange_board", table_name="instruments")
    op.drop_table("instruments")

    op.drop_index("ix_auth_users_email", table_name="auth_users")
    op.drop_table("auth_users")
