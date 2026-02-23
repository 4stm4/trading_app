#!/usr/bin/env python
"""
Run REST API for trading system.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from urllib.parse import urlsplit, urlunsplit

from loguru import logger
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ports.api import create_app


def _resolve_database_url() -> str | None:
    return (
        str(os.getenv("DATABASE_URL") or "").strip()
        or str(os.getenv("ALEMBIC_DATABASE_URL") or "").strip()
        or None
    )


def _mask_db_url(url: str) -> str:
    try:
        parts = urlsplit(url)
    except Exception:  # noqa: BLE001
        return "<invalid-url>"

    netloc = parts.netloc
    if "@" in netloc and ":" in netloc.split("@", 1)[0]:
        auth, host = netloc.split("@", 1)
        user = auth.split(":", 1)[0]
        netloc = f"{user}:***@{host}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def main() -> None:
    database_url = _resolve_database_url()
    if not database_url:
        logger.error("DATABASE_URL is not set. Export DATABASE_URL or ALEMBIC_DATABASE_URL before starting API.")
        raise SystemExit(2)

    os.environ.setdefault("DATABASE_URL", database_url)
    app = create_app()

    logger.info("=" * 80)
    logger.info("Trading System API started")
    logger.info("DB: {}", _mask_db_url(database_url))
    logger.info("=" * 80)
    logger.info("GET  http://localhost:5000/api/health")
    logger.info("GET  http://localhost:5000/api/models")
    logger.info("GET  http://localhost:5000/api/moex/instruments")
    logger.info("POST http://localhost:5000/api/signal")
    logger.info("POST http://localhost:5000/api/backtest")
    logger.info("GET  http://localhost:5000/api/dashboard/market")
    logger.info("POST http://localhost:5000/api/dashboard/backtest")
    logger.info("POST http://localhost:5000/api/dashboard/robustness")
    logger.info("POST http://localhost:5000/api/optimize")
    logger.info("=" * 80)

    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")


if __name__ == "__main__":
    main()
