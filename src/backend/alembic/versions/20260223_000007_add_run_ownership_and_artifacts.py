"""Add run ownership fields and run artifacts table.

Revision ID: 20260223_000007
Revises: 20260223_000006
Create Date: 2026-02-23 22:45:00
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "20260223_000007"
down_revision: Union[str, None] = "20260223_000006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("trading_system_runs", sa.Column("owner_user_id", sa.Integer(), nullable=True))
    op.add_column("trading_system_runs", sa.Column("portfolio_id", sa.Integer(), nullable=True))
    op.add_column("trading_system_runs", sa.Column("portfolio_balance_snapshot", sa.Float(), nullable=True))

    op.execute(
        sa.text(
            """
            UPDATE trading_system_runs r
            SET owner_user_id = ts.owner_user_id
            FROM trading_systems ts
            WHERE r.system_id = ts.id
            """
        )
    )

    op.execute(
        sa.text(
            """
            UPDATE trading_system_runs r
            SET portfolio_id = p.id,
                portfolio_balance_snapshot = p.balance
            FROM portfolios p
            WHERE r.owner_user_id = p.owner_user_id
            """
        )
    )

    op.alter_column("trading_system_runs", "owner_user_id", nullable=False)
    op.create_foreign_key(
        "fk_trading_system_runs_owner_user_id",
        "trading_system_runs",
        "auth_users",
        ["owner_user_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_trading_system_runs_portfolio_id",
        "trading_system_runs",
        "portfolios",
        ["portfolio_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_trading_system_runs_owner_created",
        "trading_system_runs",
        ["owner_user_id", "created_at"],
        unique=False,
    )

    op.create_table(
        "trading_system_run_artifacts",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("owner_user_id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("system_id", sa.Integer(), nullable=False),
        sa.Column("system_version_id", sa.Integer(), nullable=True),
        sa.Column("artifact_type", sa.String(length=64), nullable=False),
        sa.Column("payload_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["owner_user_id"], ["auth_users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["run_id"], ["trading_system_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["system_id"], ["trading_systems.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["system_version_id"], ["trading_system_versions.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "artifact_type", name="uq_run_artifacts_run_type"),
    )
    op.create_index(
        "ix_run_artifacts_owner_created",
        "trading_system_run_artifacts",
        ["owner_user_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_run_artifacts_run",
        "trading_system_run_artifacts",
        ["run_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_run_artifacts_system",
        "trading_system_run_artifacts",
        ["system_id", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_run_artifacts_system", table_name="trading_system_run_artifacts")
    op.drop_index("ix_run_artifacts_run", table_name="trading_system_run_artifacts")
    op.drop_index("ix_run_artifacts_owner_created", table_name="trading_system_run_artifacts")
    op.drop_table("trading_system_run_artifacts")

    op.drop_index("ix_trading_system_runs_owner_created", table_name="trading_system_runs")
    op.drop_constraint("fk_trading_system_runs_portfolio_id", "trading_system_runs", type_="foreignkey")
    op.drop_constraint("fk_trading_system_runs_owner_user_id", "trading_system_runs", type_="foreignkey")
    op.drop_column("trading_system_runs", "portfolio_balance_snapshot")
    op.drop_column("trading_system_runs", "portfolio_id")
    op.drop_column("trading_system_runs", "owner_user_id")
