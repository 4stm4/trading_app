"""
Application service layer for REST API use-cases.
"""

from __future__ import annotations

import math
from typing import Any

from adapters import (
    build_exchange_adapter,
    load_data_with_indicators_for_exchange,
    resolve_default_board,
)
from services.strategy_engine import MODELS, generate_signal, get_model, run_backtest


class ApiServiceError(Exception):
    status_code = 500


class ApiValidationError(ApiServiceError):
    status_code = 400


class ApiNotFoundError(ApiServiceError):
    status_code = 404


def build_health_response() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "Trading System API",
        "version": "1.0.0",
        "models_count": len(MODELS),
    }


def build_models_response() -> dict[str, Any]:
    models_info: dict[str, dict[str, Any]] = {}
    for name, model in MODELS.items():
        models_info[name] = {
            "name": model.name,
            "description": model.description,
            "min_rr": model.min_rr,
            "max_risk_percent": model.max_risk_percent,
            "min_volume_ratio": model.min_volume_ratio,
            "atr_multiplier_stop": model.atr_multiplier_stop,
            "trend_required": model.trend_required,
            "allow_range": model.allow_range,
        }
    return {"models": models_info, "count": len(models_info)}


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
                "lotSize": _to_int_or_none(row.get("LOTSIZE")),
                "prevPrice": _to_float_or_none(row.get("PREVPRICE")),
                "currency": str(row.get("CURRENCY", "")).strip() or "RUB",
            }
        )
        if len(instruments) >= resolved_limit:
            break

    return {
        "exchange": "moex",
        "engine": normalized_engine,
        "market": normalized_market,
        "board": resolved_board,
        "count": len(instruments),
        "instruments": instruments,
    }


def build_signal_response(payload: dict[str, Any]) -> dict[str, Any]:
    params = _parse_request_params(payload, require_model=True)
    model = _get_model_or_raise(params["model_name"])

    df, adapter = _load_dataset(params)
    if df.empty:
        raise ApiNotFoundError(f"No data for {params['ticker']}")

    risk_params = _resolve_contract_risk_params(adapter, params["ticker"], params["board"])
    signal = generate_signal(df, params["deposit"], model, **risk_params)

    return {
        "ticker": params["ticker"],
        "exchange": params["exchange"],
        "timeframe": params["timeframe"],
        "model": params["model_name"],
        "data_points": len(df),
        "period": {"start": str(df.index[0]), "end": str(df.index[-1])},
        "signal": signal.to_dict(),
    }


def build_backtest_response(payload: dict[str, Any]) -> dict[str, Any]:
    params = _parse_request_params(payload, require_model=True)
    model = _get_model_or_raise(params["model_name"])

    df, adapter = _load_dataset(params)
    if df.empty:
        raise ApiNotFoundError(f"No data for {params['ticker']}")

    risk_params = _resolve_contract_risk_params(adapter, params["ticker"], params["board"])
    risk_params["sl_tp_only"] = True

    results = run_backtest(
        df=df,
        signal_generator=generate_signal,
        deposit=params["deposit"],
        model=model,
        signal_kwargs=risk_params or None,
    )

    return {
        "ticker": params["ticker"],
        "exchange": params["exchange"],
        "timeframe": params["timeframe"],
        "model": params["model_name"],
        "data_points": len(df),
        "period": {"start": str(df.index[0]), "end": str(df.index[-1])},
        "results": results.to_dict(),
    }


def build_optimize_response(payload: dict[str, Any]) -> dict[str, Any]:
    params = _parse_request_params(payload, require_model=False)

    df, adapter = _load_dataset(params)
    if df.empty:
        raise ApiNotFoundError(f"No data for {params['ticker']}")

    risk_params = _resolve_contract_risk_params(adapter, params["ticker"], params["board"])
    risk_params["sl_tp_only"] = True

    results = []
    for model_name in MODELS:
        model = _get_model_or_raise(model_name)
        backtest_result = run_backtest(
            df=df,
            signal_generator=generate_signal,
            deposit=params["deposit"],
            model=model,
            signal_kwargs=risk_params or None,
        )
        results.append(backtest_result)

    best_model = max(results, key=lambda item: item.expectancy)
    return {
        "ticker": params["ticker"],
        "exchange": params["exchange"],
        "timeframe": params["timeframe"],
        "data_points": len(df),
        "period": {"start": str(df.index[0]), "end": str(df.index[-1])},
        "models_tested": len(results),
        "results": [item.to_dict() for item in results],
        "best_model": {
            "name": best_model.model_name,
            "expectancy": best_model.expectancy,
            "winrate": best_model.winrate,
            "profit_factor": best_model.profit_factor,
        },
    }


def _parse_request_params(payload: dict[str, Any], *, require_model: bool) -> dict[str, Any]:
    _require_fields(payload, ("ticker", "deposit"))

    ticker = payload["ticker"]
    exchange = payload.get("exchange", "moex")
    timeframe = payload.get("timeframe", "1h")
    engine = payload.get("engine", "stock")
    market = payload.get("market", "shares")
    board = payload.get("board")

    try:
        resolved_board = board or resolve_default_board(exchange, engine)
    except NotImplementedError as error:
        raise ApiValidationError(str(error)) from error

    data = {
        "ticker": ticker,
        "deposit": payload["deposit"],
        "exchange": exchange,
        "timeframe": timeframe,
        "engine": engine,
        "market": market,
        "board": resolved_board,
    }
    if require_model:
        data["model_name"] = payload.get("model", "balanced")
    return data


def _load_dataset(params: dict[str, Any]):
    try:
        adapter = build_exchange_adapter(params["exchange"], params["engine"], params["market"])
        df, _ = load_data_with_indicators_for_exchange(
            exchange=params["exchange"],
            ticker=params["ticker"],
            timeframe=params["timeframe"],
            start_date=None,
            end_date=None,
            board=params["board"],
            adapter=adapter,
        )
    except NotImplementedError as error:
        raise ApiValidationError(str(error)) from error
    return df, adapter


def _get_model_or_raise(model_name: str):
    try:
        return get_model(model_name)
    except ValueError as error:
        raise ApiValidationError(str(error)) from error


def _require_fields(payload: dict[str, Any], fields: tuple[str, ...]) -> None:
    for field in fields:
        if field not in payload:
            raise ApiValidationError(f"Missing required field: {field}")


def _resolve_contract_risk_params(adapter: Any, ticker: str, board: str) -> dict[str, float]:
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


def _to_float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _to_int_or_none(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result
