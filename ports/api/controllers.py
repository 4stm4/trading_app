"""
HTTP controllers for Trading System REST API.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Body, Query
from fastapi.responses import JSONResponse

from services.api_service import (
    ApiServiceError,
    build_backtest_response,
    build_candles_response,
    build_dashboard_backtest_response,
    build_dashboard_market_response,
    build_dashboard_robustness_response,
    build_health_response,
    build_moex_instruments_response,
    build_models_response,
    build_optimize_response,
    build_signal_response,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _service_error(error: ApiServiceError) -> JSONResponse:
    return JSONResponse(status_code=error.status_code, content={"error": str(error)})


@router.get("/api/health")
def health_check():
    return build_health_response()


@router.get("/api/models")
def models():
    return build_models_response()


@router.get("/api/moex/instruments")
def moex_instruments(
    engine: str = Query(default="stock", min_length=1, max_length=32),
    market: str = Query(default="shares", min_length=1, max_length=32),
    board: str | None = Query(default=None, min_length=1, max_length=12),
    limit: int = Query(default=30, ge=1, le=200),
    search: str | None = Query(default=None, max_length=40),
):
    try:
        return build_moex_instruments_response(
            engine=engine,
            market=market,
            board=board,
            limit=limit,
            search=search,
        )
    except ApiServiceError as error:
        return _service_error(error)
    except Exception as error:  # pragma: no cover - defensive fallback
        logger.exception("Error loading MOEX instruments")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal error", "message": str(error)},
        )


@router.get("/api/candles")
def candles(
    ticker: str = Query(min_length=1, max_length=32),
    exchange: str = Query(default="moex", min_length=1, max_length=16),
    timeframe: str = Query(default="1h", min_length=1, max_length=8),
    engine: str = Query(default="stock", min_length=1, max_length=32),
    market: str = Query(default="shares", min_length=1, max_length=32),
    board: str | None = Query(default=None, min_length=1, max_length=12),
    limit: int = Query(default=300, ge=50, le=1000),
):
    try:
        return build_candles_response(
            ticker=ticker,
            exchange=exchange,
            timeframe=timeframe,
            engine=engine,
            market=market,
            board=board,
            limit=limit,
        )
    except ApiServiceError as error:
        return _service_error(error)
    except Exception as error:  # pragma: no cover - defensive fallback
        logger.exception("Error loading candles")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal error", "message": str(error)},
        )


@router.post("/api/signal")
def signal(payload: dict[str, Any] | None = Body(default=None)):
    try:
        return build_signal_response(payload or {})
    except ApiServiceError as error:
        return _service_error(error)
    except Exception as error:  # pragma: no cover - defensive fallback
        logger.exception("Error generating signal")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal error", "message": str(error)},
        )


@router.post("/api/backtest")
def backtest(payload: dict[str, Any] | None = Body(default=None)):
    try:
        return build_backtest_response(payload or {})
    except ApiServiceError as error:
        return _service_error(error)
    except Exception as error:  # pragma: no cover - defensive fallback
        logger.exception("Error running backtest")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal error", "message": str(error)},
        )


@router.get("/api/dashboard/market")
def dashboard_market(
    ticker: str = Query(min_length=1, max_length=32),
    exchange: str = Query(default="moex", min_length=1, max_length=16),
    timeframe: str = Query(default="1h", min_length=1, max_length=8),
    engine: str = Query(default="stock", min_length=1, max_length=32),
    market: str = Query(default="shares", min_length=1, max_length=32),
    board: str | None = Query(default=None, min_length=1, max_length=12),
    model: str = Query(default="balanced", min_length=1, max_length=32),
    deposit: float = Query(default=100000.0, gt=0),
    limit: int = Query(default=300, ge=50, le=1000),
    commission_bps: float = Query(default=4.0, ge=0.0, le=500.0),
    slippage_bps: float = Query(default=6.0, ge=0.0, le=500.0),
    pattern_min_sample: int = Query(default=30, ge=5, le=500),
):
    payload: dict[str, Any] = {
        "ticker": ticker,
        "exchange": exchange,
        "timeframe": timeframe,
        "engine": engine,
        "market": market,
        "board": board,
        "model": model,
        "deposit": deposit,
        "limit": limit,
        "commission_bps": commission_bps,
        "slippage_bps": slippage_bps,
        "pattern_min_sample": pattern_min_sample,
    }
    try:
        return build_dashboard_market_response(payload)
    except ApiServiceError as error:
        return _service_error(error)
    except Exception as error:  # pragma: no cover - defensive fallback
        logger.exception("Error building market dashboard")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal error", "message": str(error)},
        )


@router.post("/api/dashboard/backtest")
def dashboard_backtest(payload: dict[str, Any] | None = Body(default=None)):
    try:
        return build_dashboard_backtest_response(payload or {})
    except ApiServiceError as error:
        return _service_error(error)
    except Exception as error:  # pragma: no cover - defensive fallback
        logger.exception("Error building backtest dashboard")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal error", "message": str(error)},
        )


@router.post("/api/dashboard/robustness")
def dashboard_robustness(payload: dict[str, Any] | None = Body(default=None)):
    try:
        return build_dashboard_robustness_response(payload or {})
    except ApiServiceError as error:
        return _service_error(error)
    except Exception as error:  # pragma: no cover - defensive fallback
        logger.exception("Error building robustness dashboard")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal error", "message": str(error)},
        )


@router.post("/api/optimize")
def optimize(payload: dict[str, Any] | None = Body(default=None)):
    try:
        return build_optimize_response(payload or {})
    except ApiServiceError as error:
        return _service_error(error)
    except Exception as error:  # pragma: no cover - defensive fallback
        logger.exception("Error optimizing models")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal error", "message": str(error)},
        )
