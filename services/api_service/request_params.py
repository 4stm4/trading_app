"""Request parsing and system-config normalization helpers."""

from __future__ import annotations

from typing import Any

from adapters import resolve_default_board

from .conversions import coerce_float, coerce_int, to_float_or_none
from .errors import ApiValidationError

DEFAULT_GUEST_PORTFOLIO_BALANCE = 100000.0
DEFAULT_GUEST_PORTFOLIO_CURRENCY = "RUB"


def require_fields(payload: dict[str, Any], fields: tuple[str, ...]) -> None:
    for field in fields:
        if field not in payload:
            raise ApiValidationError(f"Missing required field: {field}")


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
