"""
Application service layer for REST API use-cases.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from adapters import (
    build_exchange_adapter,
    load_data_with_indicators_for_exchange,
    resolve_default_board,
)
from services.strategy_engine import (
    MODELS,
    calculate_atr,
    calculate_kelly_criterion,
    calculate_structure,
    classify_market_regime,
    evaluate_model,
    generate_signal,
    get_model,
    run_backtest,
)


class ApiServiceError(Exception):
    status_code = 500


class ApiValidationError(ApiServiceError):
    status_code = 400


class ApiNotFoundError(ApiServiceError):
    status_code = 404


def build_health_response() -> dict[str, Any]:
    return _json_safe({
        "status": "ok",
        "service": "Trading System API",
        "version": "1.0.0",
        "models_count": len(MODELS),
    })


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

    return _json_safe({
        "exchange": "moex",
        "engine": normalized_engine,
        "market": normalized_market,
        "board": resolved_board,
        "count": len(instruments),
        "instruments": instruments,
    })


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
    params = _parse_market_request_params(
        ticker=ticker,
        exchange=exchange,
        timeframe=timeframe,
        engine=engine,
        market=market,
        board=board,
    )
    resolved_limit = max(50, min(int(limit), 1000))

    df, _ = _load_dataset(params, limit=resolved_limit)
    if df.empty:
        # Some instruments may have no recent activity; fallback to deeper history.
        df, _ = _load_dataset(
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
            "open": _to_float_or_none(row.get("open")),
            "high": _to_float_or_none(row.get("high")),
            "low": _to_float_or_none(row.get("low")),
            "close": _to_float_or_none(row.get("close")),
            "volume": _to_float_or_none(row.get("volume")),
        }
        if all(candle[key] is not None for key in ("open", "high", "low", "close", "volume")):
            candles.append(candle)

    if not candles:
        raise ApiNotFoundError(f"No valid candles for {params['ticker']}")

    return _json_safe({
        "ticker": params["ticker"],
        "exchange": params["exchange"],
        "timeframe": params["timeframe"],
        "engine": params["engine"],
        "market": params["market"],
        "board": params["board"],
        "count": len(candles),
        "candles": candles,
    })


def build_dashboard_market_response(payload: dict[str, Any]) -> dict[str, Any]:
    params = _parse_request_params(payload, require_model=True)
    model = _get_model_or_raise(params["model_name"])
    resolved_limit = _coerce_int(payload.get("limit", 300), default=300, min_value=50, max_value=1000)
    commission_bps = _coerce_float(payload.get("commission_bps", 4.0), default=4.0, min_value=0.0, max_value=500.0)
    slippage_bps = _coerce_float(payload.get("slippage_bps", 6.0), default=6.0, min_value=0.0, max_value=500.0)
    pattern_min_sample = _coerce_int(payload.get("pattern_min_sample", 30), default=30, min_value=5, max_value=500)
    # bps -> percent and multiply by 2 sides (entry/exit)
    round_trip_cost_percent = ((commission_bps + slippage_bps) * 2.0) / 100.0
    structure_mode = str(payload.get("structure_mode", "strict")).strip().lower()
    if structure_mode not in {"strict", "simple"}:
        structure_mode = "strict"

    df, adapter = _load_dataset(params, limit=resolved_limit)
    if df.empty:
        df, adapter = _load_dataset(
            params,
            limit=resolved_limit,
            start_date="2010-01-01",
        )
    if df.empty:
        raise ApiNotFoundError(f"No data for {params['ticker']}")

    local = _with_atr(df)
    structure_info = calculate_structure(local, mode=structure_mode)
    regime_series = classify_market_regime(local)
    volume_ratio, threshold_ratio, is_impulse = _volume_context_series(local)

    risk_params = _resolve_contract_risk_params(adapter, params["ticker"], params["board"])
    signal = generate_signal(local, params["deposit"], model, **risk_params)

    candles: list[dict[str, Any]] = []
    for timestamp, row in local.iterrows():
        ts = _to_unix_timestamp(timestamp)
        if ts is None:
            continue
        candle = {
            "time": ts,
            "open": _to_float_or_none(row.get("open")),
            "high": _to_float_or_none(row.get("high")),
            "low": _to_float_or_none(row.get("low")),
            "close": _to_float_or_none(row.get("close")),
            "volume": _to_float_or_none(row.get("volume")),
            "ma50": _to_float_or_none(row.get("ma50")),
            "ma200": _to_float_or_none(row.get("ma200")),
            "rsi": _to_float_or_none(row.get("rsi")),
            "atr": _to_float_or_none(row.get("atr")),
            "volume_ratio": _to_float_or_none(volume_ratio.get(timestamp)),
            "volume_threshold_ratio": _to_float_or_none(threshold_ratio.get(timestamp)),
            "is_impulse": bool(is_impulse.get(timestamp)),
            "regime": str(regime_series.get(timestamp, "range")),
        }
        if all(candle[key] is not None for key in ("open", "high", "low", "close", "volume")):
            candles.append(candle)

    if not candles:
        raise ApiNotFoundError(f"No valid candles for {params['ticker']}")

    signal_payload = signal.to_dict()
    trade_plan = _validate_trade_plan(signal_payload)

    return _json_safe({
        "ticker": params["ticker"],
        "exchange": params["exchange"],
        "timeframe": params["timeframe"],
        "engine": params["engine"],
        "market": params["market"],
        "board": params["board"],
        "model": params["model_name"],
        "data_points": len(candles),
        "period": {"start": str(local.index[0]), "end": str(local.index[-1])},
        "signal": signal_payload,
        "trade_plan": trade_plan,
        "execution_assumptions": {
            "commission_bps_per_side": round(float(commission_bps), 4),
            "slippage_bps_per_side": round(float(slippage_bps), 4),
            "round_trip_cost_percent": round(float(round_trip_cost_percent), 6),
            "pattern_min_sample": int(pattern_min_sample),
        },
        "structure": {
            "structure": structure_info.get("structure", "unknown"),
            "phase": structure_info.get("phase", "unknown"),
            "trend_strength": round(float(structure_info.get("trend_strength", 0.0)), 4),
            "breakout": bool(structure_info.get("breakout", False)),
            "last_swing_high": _to_float_or_none(structure_info.get("last_swing_high")),
            "last_swing_low": _to_float_or_none(structure_info.get("last_swing_low")),
            "swing_highs_count": int(structure_info.get("swing_highs_count", 0) or 0),
            "swing_lows_count": int(structure_info.get("swing_lows_count", 0) or 0),
        },
        "indicator_summary": {
            "rsi": _to_float_or_none(local["rsi"].iloc[-1]) if "rsi" in local.columns else None,
            "atr": _to_float_or_none(local["atr"].iloc[-1]) if "atr" in local.columns else None,
            "volume_ratio": _to_float_or_none(volume_ratio.iloc[-1]) if not volume_ratio.empty else None,
            "volume_threshold_ratio": _to_float_or_none(threshold_ratio.iloc[-1]) if not threshold_ratio.empty else None,
            "is_impulse": bool(is_impulse.iloc[-1]) if not is_impulse.empty else False,
            "rsi_oversold": 30.0,
            "rsi_overbought": 70.0,
        },
        "candles": candles,
    })


def build_dashboard_backtest_response(payload: dict[str, Any]) -> dict[str, Any]:
    params = _parse_request_params(payload, require_model=True)
    model = _get_model_or_raise(params["model_name"])
    resolved_limit = _coerce_int(payload.get("limit", 1200), default=1200, min_value=300, max_value=5000)
    lookback_window = _coerce_int(payload.get("lookback_window", 300), default=300, min_value=20, max_value=2000)
    max_holding_candles = _coerce_int(
        payload.get("max_holding_candles", 50),
        default=50,
        min_value=5,
        max_value=1000,
    )
    debug_filters = _to_bool(payload.get("debug_filters"), default=True)

    df, adapter = _load_dataset(params, limit=resolved_limit)
    if df.empty:
        df, adapter = _load_dataset(
            params,
            limit=resolved_limit,
            start_date="2010-01-01",
        )
    if df.empty:
        raise ApiNotFoundError(f"No data for {params['ticker']}")

    risk_params = _resolve_contract_risk_params(adapter, params["ticker"], params["board"])

    results = run_backtest(
        df=df,
        signal_generator=generate_signal,
        deposit=params["deposit"],
        model=model,
        lookback_window=lookback_window,
        max_holding_candles=max_holding_candles,
        signal_kwargs=risk_params or None,
        debug_filters=debug_filters,
        execution_config={"sl_tp_only": True},
    )

    equity_curve, drawdown_curve = _build_equity_and_drawdown_curves(
        trades=results.trades,
        initial_balance=params["deposit"],
        start_time=df.index[0] if len(df.index) > 0 else None,
    )

    return _json_safe({
        "ticker": params["ticker"],
        "exchange": params["exchange"],
        "timeframe": params["timeframe"],
        "engine": params["engine"],
        "market": params["market"],
        "board": params["board"],
        "model": params["model_name"],
        "data_points": len(df),
        "period": {"start": str(df.index[0]), "end": str(df.index[-1])},
        "summary": results.to_dict(),
        "trades": [_serialize_trade(trade) for trade in results.trades],
        "equity_curve": equity_curve,
        "drawdown_curve": drawdown_curve,
        "filter_funnel": results.filter_stats.to_dict() if results.filter_stats else None,
    })


def build_dashboard_robustness_response(payload: dict[str, Any]) -> dict[str, Any]:
    params = _parse_request_params(payload, require_model=True)
    model = _get_model_or_raise(params["model_name"])
    resolved_limit = _coerce_int(payload.get("limit", 1500), default=1500, min_value=300, max_value=5000)
    monte_carlo_simulations = _coerce_int(
        payload.get("monte_carlo_simulations", 300),
        default=300,
        min_value=0,
        max_value=5000,
    )
    adaptive_regime = _to_bool(payload.get("adaptive_regime"), default=False)

    df, adapter = _load_dataset(params, limit=resolved_limit)
    if df.empty:
        df, adapter = _load_dataset(
            params,
            limit=resolved_limit,
            start_date="2010-01-01",
        )
    if df.empty:
        raise ApiNotFoundError(f"No data for {params['ticker']}")

    signal_kwargs: dict[str, Any] = {}
    for key in (
        "volume_mode",
        "structure_mode",
        "disable_rr",
        "disable_volume",
        "disable_trend",
        "rsi_enabled",
        "rsi_trend_confirmation_only",
        "atr_enabled",
        "atr_min_percentile",
        "lookback_window",
        "max_holding_candles",
        "train_ratio",
    ):
        if key in payload:
            signal_kwargs[key] = payload[key]

    risk_params = _resolve_contract_risk_params(adapter, params["ticker"], params["board"])
    signal_kwargs.update(risk_params)

    stats, _ = evaluate_model(
        df=df,
        deposit=params["deposit"],
        model=model,
        signal_kwargs=signal_kwargs or None,
        walk_forward=True,
        monte_carlo_simulations=monte_carlo_simulations,
        debug_filters=False,
        adaptive_regime=adaptive_regime,
    )

    test_stats = stats.get("test", {}) or {}
    kelly_percent = calculate_kelly_criterion(
        winrate=float(test_stats.get("winrate", 0.0) or 0.0),
        avg_win=float(test_stats.get("avg_win", 0.0) or 0.0),
        avg_loss=float(test_stats.get("avg_loss", 0.0) or 0.0),
    )

    regime_series = classify_market_regime(_with_atr(df))
    regime_timeline = _serialize_regime_timeline(regime_series)

    if len(df) < 2:
        boundary = str(df.index[0]) if len(df.index) > 0 else ""
        train_period = {"start": boundary, "end": boundary}
        test_period = {"start": boundary, "end": boundary}
    else:
        train_ratio = _coerce_float(signal_kwargs.get("train_ratio", 0.7), default=0.7, min_value=0.1, max_value=0.9)
        split_idx = int(len(df) * train_ratio)
        split_idx = max(1, min(split_idx, len(df) - 1))
        train_period = {"start": str(df.index[0]), "end": str(df.index[split_idx - 1])}
        test_period = {"start": str(df.index[split_idx]), "end": str(df.index[-1])}

    return _json_safe({
        "ticker": params["ticker"],
        "exchange": params["exchange"],
        "timeframe": params["timeframe"],
        "engine": params["engine"],
        "market": params["market"],
        "board": params["board"],
        "model": params["model_name"],
        "data_points": len(df),
        "period": {"start": str(df.index[0]), "end": str(df.index[-1])},
        "train_period": train_period,
        "test_period": test_period,
        "train": stats.get("train", {}),
        "test": test_stats,
        "robustness": {
            "pf_train": _to_float_or_none(stats.get("pf_train")),
            "pf_test": _to_float_or_none(stats.get("pf_test")),
            "maxdd_test": _to_float_or_none(stats.get("maxdd_test")),
            "stability_ratio": _to_float_or_none(stats.get("stability_ratio")),
            "unstable": bool(stats.get("unstable", False)),
            "unstable_oos": bool(stats.get("unstable_oos", False)),
            "overfit": bool(stats.get("overfit", False)),
            "robustness_score": _to_float_or_none(stats.get("robustness_score")),
        },
        "market_regime_performance": stats.get("market_regime_performance", {}),
        "regime_timeline": regime_timeline,
        "monte_carlo": stats.get("monte_carlo"),
        "risk": {
            "risk_of_ruin": _to_float_or_none(stats.get("risk_of_ruin")),
            "kelly_percent": round(float(kelly_percent), 2),
        },
        "admission": stats.get("admission", {}),
        "edge_found": bool(stats.get("edge_found", False)),
        "enabled_regimes": stats.get("enabled_regimes", []),
        "disabled_regimes": stats.get("disabled_regimes", {}),
        "train_regime_performance": stats.get("train_regime_performance", {}),
    })


def build_signal_response(payload: dict[str, Any]) -> dict[str, Any]:
    params = _parse_request_params(payload, require_model=True)
    model = _get_model_or_raise(params["model_name"])

    df, adapter = _load_dataset(params)
    if df.empty:
        raise ApiNotFoundError(f"No data for {params['ticker']}")

    risk_params = _resolve_contract_risk_params(adapter, params["ticker"], params["board"])
    signal = generate_signal(df, params["deposit"], model, **risk_params)

    return _json_safe({
        "ticker": params["ticker"],
        "exchange": params["exchange"],
        "timeframe": params["timeframe"],
        "model": params["model_name"],
        "data_points": len(df),
        "period": {"start": str(df.index[0]), "end": str(df.index[-1])},
        "signal": signal.to_dict(),
    })


def build_backtest_response(payload: dict[str, Any]) -> dict[str, Any]:
    params = _parse_request_params(payload, require_model=True)
    model = _get_model_or_raise(params["model_name"])

    df, adapter = _load_dataset(params)
    if df.empty:
        raise ApiNotFoundError(f"No data for {params['ticker']}")

    risk_params = _resolve_contract_risk_params(adapter, params["ticker"], params["board"])

    results = run_backtest(
        df=df,
        signal_generator=generate_signal,
        deposit=params["deposit"],
        model=model,
        signal_kwargs=risk_params or None,
        execution_config={"sl_tp_only": True},
    )

    return _json_safe({
        "ticker": params["ticker"],
        "exchange": params["exchange"],
        "timeframe": params["timeframe"],
        "model": params["model_name"],
        "data_points": len(df),
        "period": {"start": str(df.index[0]), "end": str(df.index[-1])},
        "results": results.to_dict(),
    })


def build_optimize_response(payload: dict[str, Any]) -> dict[str, Any]:
    params = _parse_request_params(payload, require_model=False)

    df, adapter = _load_dataset(params)
    if df.empty:
        raise ApiNotFoundError(f"No data for {params['ticker']}")

    risk_params = _resolve_contract_risk_params(adapter, params["ticker"], params["board"])

    results = []
    for model_name in MODELS:
        model = _get_model_or_raise(model_name)
        backtest_result = run_backtest(
            df=df,
            signal_generator=generate_signal,
            deposit=params["deposit"],
            model=model,
            signal_kwargs=risk_params or None,
            execution_config={"sl_tp_only": True},
        )
        results.append(backtest_result)

    best_model = max(results, key=lambda item: item.expectancy)
    return _json_safe({
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
    })


def _parse_request_params(payload: dict[str, Any], *, require_model: bool) -> dict[str, Any]:
    _require_fields(payload, ("ticker", "deposit"))

    data = _parse_market_request_params(
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


def _parse_market_request_params(
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


def _load_dataset(
    params: dict[str, Any],
    *,
    limit: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
):
    try:
        adapter = build_exchange_adapter(params["exchange"], params["engine"], params["market"])
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


def _with_atr(df: pd.DataFrame) -> pd.DataFrame:
    if "atr" in df.columns:
        return df
    local = df.copy()
    local["atr"] = calculate_atr(local)
    return local


def _volume_context_series(
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


def _build_equity_and_drawdown_curves(
    trades: list[Any],
    initial_balance: float,
    start_time: Any = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    equity_curve: list[dict[str, Any]] = []
    drawdown_curve: list[dict[str, Any]] = []

    start_ts = _to_unix_timestamp(start_time)
    if start_ts is not None:
        equity_curve.append({"time": start_ts, "equity": round(float(initial_balance), 2)})
        drawdown_curve.append({"time": start_ts, "drawdown_percent": 0.0})

    equity = float(initial_balance)
    peak = float(initial_balance)
    for trade in trades:
        exit_ts = _to_unix_timestamp(getattr(trade, "exit_time", None))
        if exit_ts is None:
            continue
        pnl = float(getattr(trade, "pnl", 0.0))
        equity += pnl
        peak = max(peak, equity)
        drawdown_pct = ((peak - equity) / peak * 100.0) if peak > 0 else 0.0
        equity_curve.append({"time": exit_ts, "equity": round(equity, 2)})
        drawdown_curve.append({"time": exit_ts, "drawdown_percent": round(drawdown_pct, 4)})

    return equity_curve, drawdown_curve


def _serialize_trade(trade: Any) -> dict[str, Any]:
    return {
        "entry_time": str(getattr(trade, "entry_time", "")),
        "exit_time": str(getattr(trade, "exit_time", "")),
        "entry_ts": _to_unix_timestamp(getattr(trade, "entry_time", None)),
        "exit_ts": _to_unix_timestamp(getattr(trade, "exit_time", None)),
        "entry_price": _to_float_or_none(getattr(trade, "entry_price", None)),
        "exit_price": _to_float_or_none(getattr(trade, "exit_price", None)),
        "stop_price": _to_float_or_none(getattr(trade, "stop_price", None)),
        "target_price": _to_float_or_none(getattr(trade, "target_price", None)),
        "direction": str(getattr(trade, "direction", "")),
        "position_size": _to_float_or_none(getattr(trade, "position_size", None)),
        "pnl": _to_float_or_none(getattr(trade, "pnl", None)),
        "pnl_percent": _to_float_or_none(getattr(trade, "pnl_percent", None)),
        "exit_reason": str(getattr(trade, "exit_reason", "")),
        "rr_planned": _to_float_or_none(getattr(trade, "rr_planned", None)),
        "rr_actual": _to_float_or_none(getattr(trade, "rr_actual", None)),
        "duration_candles": int(getattr(trade, "duration_candles", 0) or 0),
        "regime": getattr(trade, "regime", None),
        "gross_pnl": _to_float_or_none(getattr(trade, "gross_pnl", None)),
        "fees": _to_float_or_none(getattr(trade, "fees", None)),
        "slippage": _to_float_or_none(getattr(trade, "slippage", None)),
    }


def _serialize_regime_timeline(regime_series: pd.Series) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    if regime_series.empty:
        return timeline

    last_value: str | None = None
    for timestamp, value in regime_series.items():
        regime = str(value)
        if regime == last_value:
            continue
        ts = _to_unix_timestamp(timestamp)
        if ts is None:
            continue
        timeline.append({"time": ts, "regime": regime})
        last_value = regime
    return timeline


def _to_unix_timestamp(value: Any) -> int | None:
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return int(ts.timestamp())


def _coerce_int(value: Any, *, default: int, min_value: int, max_value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(min_value, min(parsed, max_value))


def _coerce_float(value: Any, *, default: float, min_value: float, max_value: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(min_value, min(parsed, max_value))


def _to_bool(value: Any, *, default: bool = False) -> bool:
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


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _validate_trade_plan(signal_payload: dict[str, Any]) -> dict[str, Any]:
    signal_type = str(signal_payload.get("signal", "none")).strip().lower()
    entry = _to_float_or_none(signal_payload.get("entry"))
    stop = _to_float_or_none(signal_payload.get("stop"))
    target = _to_float_or_none(signal_payload.get("target"))
    rr = _to_float_or_none(signal_payload.get("rr"))
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
