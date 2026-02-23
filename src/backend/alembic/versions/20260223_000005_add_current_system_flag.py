"""Add current-system flag for trading systems.

Revision ID: 20260223_000005
Revises: 20260223_000004
Create Date: 2026-02-23 20:50:00
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "20260223_000005"
down_revision: Union[str, None] = "20260223_000004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "trading_systems",
        sa.Column("is_current", sa.Boolean(), server_default=sa.text("false"), nullable=False),
    )
    op.execute(
        sa.text(
            """
            UPDATE trading_systems ts
            SET is_current = true
            FROM (
                SELECT DISTINCT ON (owner_user_id) id, owner_user_id
                FROM trading_systems
                ORDER BY owner_user_id, updated_at DESC, id DESC
            ) ranked
            WHERE ts.id = ranked.id
            """
        )
    )
    op.create_index(
        "uq_trading_systems_owner_current",
        "trading_systems",
        ["owner_user_id"],
        unique=True,
        postgresql_where=sa.text("is_current"),
    )


def downgrade() -> None:
    op.drop_index("uq_trading_systems_owner_current", table_name="trading_systems")
    op.drop_column("trading_systems", "is_current")
