"""Add portfolios table.

Revision ID: 20260223_000006
Revises: 20260223_000005
Create Date: 2026-02-23 21:40:00
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "20260223_000006"
down_revision: Union[str, None] = "20260223_000005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "portfolios",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("owner_user_id", sa.Integer(), nullable=False),
        sa.Column("balance", sa.Float(), server_default="100000", nullable=False),
        sa.Column("currency", sa.String(length=16), server_default="RUB", nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["owner_user_id"], ["auth_users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("uq_portfolios_owner_user_id", "portfolios", ["owner_user_id"], unique=True)
    op.create_index("ix_portfolios_owner_active", "portfolios", ["owner_user_id", "is_active"], unique=False)
    op.execute(
        sa.text(
            """
            INSERT INTO portfolios (owner_user_id, balance, currency, is_active)
            SELECT u.id, 100000, 'RUB', true
            FROM auth_users u
            LEFT JOIN portfolios p ON p.owner_user_id = u.id
            WHERE p.id IS NULL
            """
        )
    )


def downgrade() -> None:
    op.drop_index("ix_portfolios_owner_active", table_name="portfolios")
    op.drop_index("uq_portfolios_owner_user_id", table_name="portfolios")
    op.drop_table("portfolios")
