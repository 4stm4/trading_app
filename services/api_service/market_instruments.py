"""Market instruments and candle-loading use-cases."""

from __future__ import annotations

from typing import Any

from adapters import build_exchange_adapter, resolve_default_board

from .errors import ApiNotFoundError, ApiServiceError, ApiValidationError
from .helpers import (
    json_safe,
    load_dataset,
    parse_market_request_params,
    to_float_or_none,
    to_int_or_none,
)


def build_moex_instruments_response(
    *,
    engine: str = "stock",
    market: str = "shares",
    board: str | None = None,
    limit: int = 30,
    search: str | None = None,
) -> dict[str, Any]:
    normalized_engine = str(engine or "stock").strip().lower()
    normalized_market = str(market or "shares").strip().lower()
    board_raw = str(board or "").strip().upper()
    try:
        resolved_board = board_raw or resolve_default_board("moex", normalized_engine)
    except NotImplementedError as error:
        raise ApiValidationError(str(error)) from error

    try:
        resolved_limit = max(1, min(int(limit), 200))
    except (TypeError, ValueError) as error:
        raise ApiValidationError("limit must be an integer") from error
    normalized_search = str(search or "").strip().upper()

    try:
        adapter = build_exchange_adapter("moex", normalized_engine, normalized_market)
    except NotImplementedError as error:
        raise ApiValidationError(str(error)) from error

    getter = getattr(adapter, "get_securities", None)
    if not callable(getter):
        raise ApiValidationError("MOEX adapter does not support securities list")

    try:
        frame = getter(board=resolved_board)
    except Exception as error:
        raise ApiServiceError(f"Failed to load MOEX instruments: {error}") from error

    if frame is None or frame.empty:
        return {
            "exchange": "moex",
            "engine": normalized_engine,
            "market": normalized_market,
            "board": resolved_board,
            "count": 0,
            "instruments": [],
        }

    records = frame.to_dict(orient="records")
    instruments: list[dict[str, Any]] = []

    for row in records:
        symbol = str(row.get("SECID", "")).strip().upper()
        if not symbol:
            continue

        name = str(row.get("NAME", "")).strip()
        short_name = str(row.get("SHORTNAME", "")).strip()
        haystack = f"{symbol} {name} {short_name}".upper()
        if normalized_search and normalized_search not in haystack:
            continue

        instruments.append(
            {
                "symbol": symbol,
                "name": name or short_name or symbol,
                "shortName": short_name or name or symbol,
                "lotSize": to_int_or_none(row.get("LOTSIZE")),
                "prevPrice": to_float_or_none(row.get("PREVPRICE")),
                "currency": str(row.get("CURRENCY", "")).strip() or "RUB",
            }
        )
        if len(instruments) >= resolved_limit:
            break

    return json_safe(
        {
            "exchange": "moex",
            "engine": normalized_engine,
            "market": normalized_market,
            "board": resolved_board,
            "count": len(instruments),
            "instruments": instruments,
        }
    )


def build_candles_response(
    *,
    ticker: str,
    exchange: str = "moex",
    timeframe: str = "1h",
    engine: str = "stock",
    market: str = "shares",
    board: str | None = None,
    limit: int = 300,
) -> dict[str, Any]:
    params = parse_market_request_params(
        ticker=ticker,
        exchange=exchange,
        timeframe=timeframe,
        engine=engine,
        market=market,
        board=board,
    )
    resolved_limit = max(50, min(int(limit), 1000))

    df, _ = load_dataset(params, limit=resolved_limit)
    if df.empty:
        df, _ = load_dataset(
            params,
            limit=resolved_limit,
            start_date="2010-01-01",
        )
    if df.empty:
        raise ApiNotFoundError(f"No data for {params['ticker']}")

    candles: list[dict[str, Any]] = []
    for timestamp, row in df.iterrows():
        ts = timestamp.to_pydatetime()
        candle = {
            "time": int(ts.timestamp()),
            "open": to_float_or_none(row.get("open")),
            "high": to_float_or_none(row.get("high")),
            "low": to_float_or_none(row.get("low")),
            "close": to_float_or_none(row.get("close")),
            "volume": to_float_or_none(row.get("volume")),
        }
        if all(candle[key] is not None for key in ("open", "high", "low", "close", "volume")):
            candles.append(candle)

    if not candles:
        raise ApiNotFoundError(f"No valid candles for {params['ticker']}")

    return json_safe(
        {
            "ticker": params["ticker"],
            "exchange": params["exchange"],
            "timeframe": params["timeframe"],
            "engine": params["engine"],
            "market": params["market"],
            "board": params["board"],
            "count": len(candles),
            "candles": candles,
        }
    )
