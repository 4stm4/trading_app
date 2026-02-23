"""Dataset loading, DB session factory and model lookup helpers."""

from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session, sessionmaker

from adapters import build_exchange_adapter, load_data_with_indicators_for_exchange
from adapters.postgres import (
    MarketCandlePostgresRepository,
    create_postgres_engine,
    create_session_factory,
    session_scope,
)
from services.strategy_engine.public import calculate_atr, get_model
from services.strategy_engine.indicators import add_indicators

from .conversions import to_unix_timestamp
from .errors import ApiValidationError

logger = logging.getLogger(__name__)

_DB_SESSION_FACTORY: sessionmaker[Session] | None = None
_DB_SESSION_FACTORY_INIT = False
_DB_SESSION_FACTORY_URL: str | None = None


def build_adapter(params: dict[str, Any]):
    try:
        return build_exchange_adapter(params["exchange"], params["engine"], params["market"])
    except NotImplementedError as error:
        raise ApiValidationError(str(error)) from error


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

    return add_indicators(frame, ma_periods=[50, 200], rsi_period=14)


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
