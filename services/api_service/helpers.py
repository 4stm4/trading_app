"""Shared helper functions for API service modules."""

from __future__ import annotations

import logging
import math
import os
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session, sessionmaker

from adapters import (
    build_exchange_adapter,
    load_data_with_indicators_for_exchange,
    resolve_default_board,
)
from adapters.postgres import (
    MarketCandlePostgresRepository,
    create_postgres_engine,
    create_session_factory,
    session_scope,
)
from services.strategy_engine import calculate_atr, get_model
from services.strategy_engine.indicators import add_indicators

from .errors import ApiValidationError


logger = logging.getLogger(__name__)

DEFAULT_GUEST_PORTFOLIO_BALANCE = 100000.0
DEFAULT_GUEST_PORTFOLIO_CURRENCY = "RUB"

_DB_SESSION_FACTORY: sessionmaker[Session] | None = None
_DB_SESSION_FACTORY_INIT = False
_DB_SESSION_FACTORY_URL: str | None = None


def parse_request_params(payload: dict[str, Any], *, require_model: bool) -> dict[str, Any]:
    require_fields(payload, ("ticker", "deposit"))

    data = parse_market_request_params(
        ticker=payload["ticker"],
        exchange=payload.get("exchange", "moex"),
        timeframe=payload.get("timeframe", "1h"),
        engine=payload.get("engine", "stock"),
        market=payload.get("market", "shares"),
        board=payload.get("board"),
    )
    data["deposit"] = payload["deposit"]
    if require_model:
        data["model_name"] = payload.get("model", "balanced")
    return data


def parse_market_request_params(
    *,
    ticker: str,
    exchange: str,
    timeframe: str,
    engine: str,
    market: str,
    board: str | None,
) -> dict[str, Any]:
    normalized_ticker = str(ticker or "").strip().upper()
    if not normalized_ticker:
        raise ApiValidationError("ticker is required")
    normalized_exchange = str(exchange or "moex").strip().lower()
    normalized_timeframe = str(timeframe or "1h").strip().lower()
    normalized_engine = str(engine or "stock").strip().lower()
    normalized_market = str(market or "shares").strip().lower()

    try:
        resolved_board = str(board or resolve_default_board(normalized_exchange, normalized_engine)).strip().upper()
    except NotImplementedError as error:
        raise ApiValidationError(str(error)) from error

    return {
        "ticker": normalized_ticker,
        "exchange": normalized_exchange,
        "timeframe": normalized_timeframe,
        "engine": normalized_engine,
        "market": normalized_market,
        "board": resolved_board,
    }


def load_dataset(
    params: dict[str, Any],
    *,
    limit: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
):
    adapter = build_adapter(params)

    db_df = load_dataset_from_db(
        params,
        limit=limit,
        start_date=start_date,
        end_date=end_date,
    )
    if not db_df.empty:
        logger.info(
            "Dataset source=db ticker=%s tf=%s rows=%d",
            params["ticker"],
            params["timeframe"],
            len(db_df),
        )
        return db_df, adapter

    logger.info(
        "Dataset source=exchange ticker=%s tf=%s (db empty)",
        params["ticker"],
        params["timeframe"],
    )
    try:
        df, _ = load_data_with_indicators_for_exchange(
            exchange=params["exchange"],
            ticker=params["ticker"],
            timeframe=params["timeframe"],
            start_date=start_date,
            end_date=end_date,
            board=params["board"],
            adapter=adapter,
            limit=limit,
        )
    except NotImplementedError as error:
        raise ApiValidationError(str(error)) from error
    return df, adapter


def build_adapter(params: dict[str, Any]):
    try:
        return build_exchange_adapter(params["exchange"], params["engine"], params["market"])
    except NotImplementedError as error:
        raise ApiValidationError(str(error)) from error


def load_dataset_from_db(
    params: dict[str, Any],
    *,
    limit: int | None,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    session_factory = get_db_session_factory()
    if session_factory is None:
        return pd.DataFrame()

    try:
        with session_scope(session_factory) as session:
            repo = MarketCandlePostgresRepository(session)
            frame = repo.get_frame(
                exchange=params["exchange"],
                engine=params["engine"],
                market=params["market"],
                board=params["board"],
                symbol=params["ticker"],
                timeframe=params["timeframe"],
                limit=limit,
                start_date=start_date,
                end_date=end_date,
            )
    except Exception:
        logger.exception("Failed to load dataset from DB")
        return pd.DataFrame()

    if frame.empty:
        return frame

    enriched = add_indicators(frame, ma_periods=[50, 200], rsi_period=14)
    return enriched


def get_db_session_factory() -> sessionmaker[Session] | None:
    global _DB_SESSION_FACTORY, _DB_SESSION_FACTORY_INIT, _DB_SESSION_FACTORY_URL

    database_url = (
        str(os.getenv("DATABASE_URL") or "").strip()
        or str(os.getenv("ALEMBIC_DATABASE_URL") or "").strip()
        or str(os.getenv("AUTH_DB_URL") or "").strip()
    )
    if _DB_SESSION_FACTORY_INIT and database_url == (_DB_SESSION_FACTORY_URL or ""):
        return _DB_SESSION_FACTORY

    if not database_url:
        _DB_SESSION_FACTORY_INIT = True
        _DB_SESSION_FACTORY = None
        _DB_SESSION_FACTORY_URL = None
        return None

    try:
        engine = create_postgres_engine(database_url, echo=False)
        _DB_SESSION_FACTORY = create_session_factory(engine)
        _DB_SESSION_FACTORY_URL = database_url
    except Exception:
        logger.exception("Failed to initialize DB session factory")
        _DB_SESSION_FACTORY = None
        _DB_SESSION_FACTORY_URL = None
    finally:
        _DB_SESSION_FACTORY_INIT = True

    return _DB_SESSION_FACTORY


def get_model_or_raise(model_name: str):
    try:
        return get_model(model_name)
    except ValueError as error:
        raise ApiValidationError(str(error)) from error


def require_fields(payload: dict[str, Any], fields: tuple[str, ...]) -> None:
    for field in fields:
        if field not in payload:
            raise ApiValidationError(f"Missing required field: {field}")


def resolve_contract_risk_params(adapter: Any, ticker: str, board: str) -> dict[str, float]:
    getter = getattr(adapter, "get_security_spec", None)
    if not callable(getter):
        return {}

    try:
        spec = getter(ticker, board=board) or {}
    except Exception:
        return {}

    result: dict[str, float] = {}
    try:
        margin = float(spec.get("initial_margin"))
        if margin > 0:
            result["contract_margin_rub"] = margin
    except (TypeError, ValueError):
        pass

    try:
        multiplier = float(spec.get("contract_multiplier"))
        if multiplier > 0:
            result["contract_multiplier"] = multiplier
    except (TypeError, ValueError):
        pass

    return result


def with_atr(df: pd.DataFrame) -> pd.DataFrame:
    if "atr" in df.columns:
        return df
    local = df.copy()
    local["atr"] = calculate_atr(local)
    return local


def volume_context_series(
    df: pd.DataFrame,
    period: int = 20,
    threshold_ratio: float = 1.5,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    if "volume" not in df.columns or df.empty:
        empty = pd.Series(index=df.index, dtype="float64")
        return empty, empty, pd.Series(index=df.index, dtype="bool")

    avg_volume = df["volume"].rolling(window=period, min_periods=1).mean()
    ratio = df["volume"] / avg_volume.replace(0.0, pd.NA)
    ratio = ratio.fillna(0.0)
    threshold_series = pd.Series(threshold_ratio, index=df.index, dtype="float64")
    impulse = df["volume"] >= (avg_volume * threshold_ratio)
    return ratio, threshold_series, impulse.fillna(False)


def build_equity_and_drawdown_curves(
    trades: list[Any],
    initial_balance: float,
    start_time: Any = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    equity_curve: list[dict[str, Any]] = []
    drawdown_curve: list[dict[str, Any]] = []

    start_ts = to_unix_timestamp(start_time)
    if start_ts is not None:
        equity_curve.append({"time": start_ts, "equity": round(float(initial_balance), 2)})
        drawdown_curve.append({"time": start_ts, "drawdown_percent": 0.0})

    equity = float(initial_balance)
    peak = float(initial_balance)
    for trade in trades:
        exit_ts = to_unix_timestamp(getattr(trade, "exit_time", None))
        if exit_ts is None:
            continue
        pnl = float(getattr(trade, "pnl", 0.0))
        equity += pnl
        peak = max(peak, equity)
        drawdown_pct = ((peak - equity) / peak * 100.0) if peak > 0 else 0.0
        equity_curve.append({"time": exit_ts, "equity": round(equity, 2)})
        drawdown_curve.append({"time": exit_ts, "drawdown_percent": round(drawdown_pct, 4)})

    return equity_curve, drawdown_curve


def to_unix_timestamp(value: Any) -> int | None:
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return int(ts.timestamp())


def coerce_int(value: Any, *, default: int, min_value: int, max_value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(min_value, min(parsed, max_value))


def coerce_float(value: Any, *, default: float, min_value: float, max_value: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(min_value, min(parsed, max_value))


def to_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def validate_trade_plan(signal_payload: dict[str, Any]) -> dict[str, Any]:
    signal_type = str(signal_payload.get("signal", "none")).strip().lower()
    entry = to_float_or_none(signal_payload.get("entry"))
    stop = to_float_or_none(signal_payload.get("stop"))
    target = to_float_or_none(signal_payload.get("target"))
    rr = to_float_or_none(signal_payload.get("rr"))
    confidence = str(signal_payload.get("confidence", "none"))

    issues: list[str] = []
    if signal_type not in {"long", "short"}:
        return {
            "status": "no_signal",
            "tradable": False,
            "signal": signal_type,
            "entry": entry,
            "stop": stop,
            "target": target,
            "rr": rr,
            "confidence": confidence,
            "issues": ["No actionable signal"],
        }

    if entry is None or entry <= 0:
        issues.append("entry_missing_or_non_positive")
    if stop is None or stop <= 0:
        issues.append("stop_missing_or_non_positive")
    if target is None or target <= 0:
        issues.append("target_missing_or_non_positive")

    if not issues and signal_type == "long" and not (stop < entry < target):
        issues.append("invalid_price_order_for_long")
    if not issues and signal_type == "short" and not (target < entry < stop):
        issues.append("invalid_price_order_for_short")
    if rr is None or rr <= 0:
        issues.append("rr_non_positive")

    return {
        "status": "valid" if not issues else "invalid",
        "tradable": len(issues) == 0,
        "signal": signal_type,
        "entry": entry,
        "stop": stop,
        "target": target,
        "rr": rr,
        "confidence": confidence,
        "issues": issues,
    }


def to_json_object(value: Any) -> dict[str, Any]:
    safe = json_safe(value)
    if isinstance(safe, dict):
        return safe
    return {"value": safe}


def to_float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def to_int_or_none(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result


def normalize_system_config(payload: dict[str, Any]) -> dict[str, Any]:
    source = payload or {}
    return {
        "deposit": coerce_float(source.get("deposit"), default=100000.0, min_value=0.0, max_value=10_000_000_000.0),
        "commissionBps": coerce_float(source.get("commissionBps"), default=4.0, min_value=0.0, max_value=500.0),
        "slippageBps": coerce_float(source.get("slippageBps"), default=6.0, min_value=0.0, max_value=500.0),
        "patternMinSample": coerce_int(source.get("patternMinSample"), default=30, min_value=1, max_value=10_000),
        "marketLimit": coerce_int(source.get("marketLimit"), default=300, min_value=50, max_value=10_000),
        "candidateScanLimit": coerce_int(source.get("candidateScanLimit"), default=24, min_value=1, max_value=1000),
        "candidateScanConcurrency": coerce_int(
            source.get("candidateScanConcurrency"),
            default=4,
            min_value=1,
            max_value=128,
        ),
        "backtestLimit": coerce_int(source.get("backtestLimit"), default=1200, min_value=100, max_value=20_000),
        "backtestLookbackWindow": coerce_int(
            source.get("backtestLookbackWindow"),
            default=300,
            min_value=20,
            max_value=10_000,
        ),
        "backtestMaxHoldingCandles": coerce_int(
            source.get("backtestMaxHoldingCandles"),
            default=50,
            min_value=1,
            max_value=5000,
        ),
        "robustnessLimit": coerce_int(source.get("robustnessLimit"), default=1500, min_value=100, max_value=20_000),
        "monteCarloSimulations": coerce_int(
            source.get("monteCarloSimulations"),
            default=300,
            min_value=10,
            max_value=20_000,
        ),
        "trainRatio": coerce_float(source.get("trainRatio"), default=0.7, min_value=0.1, max_value=0.95),
    }
