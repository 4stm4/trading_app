from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from adapters.postgres.base import Base
from adapters.postgres.instrument.tables import InstrumentTable  # noqa: F401
from adapters.postgres.market_candle.tables import MarketCandleTable  # noqa: F401
from adapters.postgres.portfolio.tables import PortfolioTable  # noqa: F401
from adapters.postgres.recommended_sale.tables import RecommendedSaleTable  # noqa: F401
from adapters.postgres.trading_model.tables import TradingModelTable  # noqa: F401
from adapters.postgres.trading_system.tables import TradingSystemTable  # noqa: F401
from adapters.postgres.trading_system_run.tables import TradingSystemRunTable  # noqa: F401
from adapters.postgres.trading_system_run_artifact.tables import TradingSystemRunArtifactTable  # noqa: F401
from adapters.postgres.trading_system_signal.tables import TradingSystemSignalTable  # noqa: F401
from adapters.postgres.trading_system_version.tables import TradingSystemVersionTable  # noqa: F401
from adapters.postgres.user.tables import UserTable  # noqa: F401


config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _resolve_database_url() -> str:
    url = os.getenv("ALEMBIC_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not url:
        url = os.getenv("AUTH_DB_URL")
    if not url:
        url = config.get_main_option("sqlalchemy.url")

    if not url or url.startswith("driver://"):
        raise RuntimeError(
            "Set ALEMBIC_DATABASE_URL (or DATABASE_URL/AUTH_DB_URL) before running migrations.",
        )

    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url


def run_migrations_offline() -> None:
    url = _resolve_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = _resolve_database_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
