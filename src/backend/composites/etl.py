#!/usr/bin/env python
"""
Run market data ETL: initial backfill + incremental updates on next runs.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adapters.postgres import create_postgres_engine, create_schema, create_session_factory
from services.etl_service import EtlConfig, MarketDataEtlService

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trading ETL runner")
    parser.add_argument("--exchange", default=os.getenv("ETL_EXCHANGE", "moex"))
    parser.add_argument("--engine", default=os.getenv("ETL_ENGINE", "stock"))
    parser.add_argument("--market", default=os.getenv("ETL_MARKET", "shares"))
    parser.add_argument("--board", default=os.getenv("ETL_BOARD"))
    parser.add_argument("--timeframe", default=os.getenv("ETL_TIMEFRAME", "1h"))
    parser.add_argument("--symbols", default=os.getenv("ETL_SYMBOLS"))
    parser.add_argument("--max-symbols", type=int, default=_env_int("ETL_MAX_SYMBOLS"))
    parser.add_argument(
        "--initial-lookback-days",
        type=int,
        default=int(os.getenv("ETL_INITIAL_LOOKBACK_DAYS", "365")),
    )
    parser.add_argument("--limit", type=int, default=_env_int("ETL_LIMIT"))
    parser.add_argument("--start-date", default=os.getenv("ETL_START_DATE"))
    parser.add_argument("--end-date", default=os.getenv("ETL_END_DATE"))
    parser.add_argument("--sync-instruments", action="store_true", default=True)
    parser.add_argument("--no-sync-instruments", action="store_false", dest="sync_instruments")
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL") or os.getenv("ALEMBIC_DATABASE_URL"),
        help="SQLAlchemy URL (default: DATABASE_URL or ALEMBIC_DATABASE_URL)",
    )
    parser.add_argument("--echo-sql", action="store_true")
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()

    if not args.database_url:
        parser.error("database URL is required (--database-url or DATABASE_URL env)")
        return 2

    symbols = _parse_symbols(args.symbols)
    config = EtlConfig(
        exchange=args.exchange,
        engine=args.engine,
        market=args.market,
        board=args.board,
        timeframe=args.timeframe,
        symbols=symbols,
        max_symbols=args.max_symbols,
        initial_lookback_days=args.initial_lookback_days,
        limit=args.limit,
        start_date=args.start_date,
        end_date=args.end_date,
        sync_instruments=bool(args.sync_instruments),
    )

    engine = create_postgres_engine(args.database_url, echo=bool(args.echo_sql))
    create_schema(engine)
    session_factory = create_session_factory(engine)

    service = MarketDataEtlService(session_factory)
    result = service.run(config)

    logger.info("=" * 80)
    logger.info("ETL FINISHED")
    logger.info("=" * 80)
    logger.info("Started at:   %sZ", result.started_at.isoformat())
    logger.info("Finished at:  %sZ", result.finished_at.isoformat())
    logger.info("Exchange:     %s/%s/%s", result.exchange, result.engine, result.market)
    logger.info("Board:        %s", result.board)
    logger.info("Timeframe:    %s", result.timeframe)
    logger.info("Symbols:      %s/%s ok", result.symbols_ok, result.symbols_total)
    logger.info("Inserted:     %s", result.inserted_total)
    logger.info("Updated:      %s", result.updated_total)
    logger.info("Instruments:  %s", result.instruments_synced)
    logger.info("=" * 80)

    failed = [item for item in result.details if item.status != "ok"]
    if failed:
        logger.error("FAILED SYMBOLS:")
        for item in failed[:50]:
            logger.error("- %s: %s", item.symbol, item.error)
        return 1
    return 0


def _parse_symbols(value: str | None) -> list[str] | None:
    if not value:
        return None
    parsed = [token.strip().upper() for token in value.split(",") if token.strip()]
    return parsed or None


def _env_int(name: str) -> int | None:
    value = os.getenv(name)
    if value in (None, ""):
        return None
    try:
        return int(value)
    except ValueError:
        return None


if __name__ == "__main__":
    sys.exit(main())
