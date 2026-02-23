"""Add trading models table and link systems to model.

Revision ID: 20260223_000004
Revises: 20260223_000003
Create Date: 2026-02-23 20:10:00
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "20260223_000004"
down_revision: Union[str, None] = "20260223_000003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "trading_models",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("key", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("description", sa.String(length=512), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_trading_models_key", "trading_models", ["key"], unique=True)
    op.create_index("ix_trading_models_active", "trading_models", ["is_active"], unique=False)

    trading_models = sa.table(
        "trading_models",
        sa.column("key", sa.String),
        sa.column("name", sa.String),
        sa.column("description", sa.String),
        sa.column("is_active", sa.Boolean),
    )
    op.bulk_insert(
        trading_models,
        [
            {"key": "balanced", "name": "Balanced", "description": "Balanced risk/reward profile", "is_active": True},
            {"key": "aggressive", "name": "Aggressive", "description": "Higher risk for higher return", "is_active": True},
            {"key": "scalp", "name": "Scalp", "description": "Short-horizon, quick trade model", "is_active": True},
            {"key": "high_rr", "name": "High RR", "description": "Prioritizes high risk/reward setups", "is_active": True},
            {
                "key": "conservative",
                "name": "Conservative",
                "description": "Lower risk with strict filters",
                "is_active": True,
            },
        ],
    )

    op.add_column("trading_systems", sa.Column("model_id", sa.Integer(), nullable=True))
    op.create_index("ix_trading_systems_model_id", "trading_systems", ["model_id"], unique=False)
    op.create_foreign_key(
        "fk_trading_systems_model_id",
        "trading_systems",
        "trading_models",
        ["model_id"],
        ["id"],
        ondelete="SET NULL",
    )

    op.execute(
        sa.text(
            """
            UPDATE trading_systems ts
            SET model_id = tm.id
            FROM trading_models tm
            WHERE lower(ts.model_name) = tm.key
            """
        )
    )


def downgrade() -> None:
    op.drop_constraint("fk_trading_systems_model_id", "trading_systems", type_="foreignkey")
    op.drop_index("ix_trading_systems_model_id", table_name="trading_systems")
    op.drop_column("trading_systems", "model_id")

    op.drop_index("ix_trading_models_active", table_name="trading_models")
    op.drop_index("ix_trading_models_key", table_name="trading_models")
    op.drop_table("trading_models")
