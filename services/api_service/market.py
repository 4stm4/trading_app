"""Market data, signal, backtest and robustness use-cases."""

from __future__ import annotations

from typing import Any

from adapters import build_exchange_adapter, resolve_default_board
from services.strategy_engine import (
    MODELS,
    calculate_kelly_criterion,
    calculate_structure,
    classify_market_regime,
    evaluate_model,
    generate_signal,
    run_backtest,
)

from .errors import ApiNotFoundError, ApiServiceError, ApiValidationError
from .helpers import (
    build_adapter,
    build_equity_and_drawdown_curves,
    coerce_float,
    coerce_int,
    get_model_or_raise,
    json_safe,
    load_dataset,
    parse_market_request_params,
    parse_request_params,
    resolve_contract_risk_params,
    to_bool,
    to_float_or_none,
    to_int_or_none,
    to_unix_timestamp,
    validate_trade_plan,
    volume_context_series,
    with_atr,
)
from .serializers import serialize_regime_timeline, serialize_trade
from .tracking import (
    finish_system_run_failure,
    finish_system_run_success,
    resolve_system_run_context,
    start_system_run,
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


def build_dashboard_market_response(payload: dict[str, Any]) -> dict[str, Any]:
    params = parse_request_params(payload, require_model=True)
    model = get_model_or_raise(params["model_name"])
    resolved_limit = coerce_int(payload.get("limit", 300), default=300, min_value=50, max_value=1000)
    commission_bps = coerce_float(payload.get("commission_bps", 4.0), default=4.0, min_value=0.0, max_value=500.0)
    slippage_bps = coerce_float(payload.get("slippage_bps", 6.0), default=6.0, min_value=0.0, max_value=500.0)
    pattern_min_sample = coerce_int(payload.get("pattern_min_sample", 30), default=30, min_value=5, max_value=500)
    round_trip_cost_percent = ((commission_bps + slippage_bps) * 2.0) / 100.0
    structure_mode = str(payload.get("structure_mode", "strict")).strip().lower()
    if structure_mode not in {"strict", "simple"}:
        structure_mode = "strict"

    df, adapter = load_dataset(params, limit=resolved_limit)
    if df.empty:
        df, adapter = load_dataset(
            params,
            limit=resolved_limit,
            start_date="2010-01-01",
        )
    if df.empty:
        raise ApiNotFoundError(f"No data for {params['ticker']}")

    local = with_atr(df)
    structure_info = calculate_structure(local, mode=structure_mode)
    regime_series = classify_market_regime(local)
    volume_ratio, threshold_ratio, is_impulse = volume_context_series(local)

    risk_params = resolve_contract_risk_params(adapter, params["ticker"], params["board"])
    signal = generate_signal(local, params["deposit"], model, **risk_params)

    candles: list[dict[str, Any]] = []
    for timestamp, row in local.iterrows():
        ts = to_unix_timestamp(timestamp)
        if ts is None:
            continue
        candle = {
            "time": ts,
            "open": to_float_or_none(row.get("open")),
            "high": to_float_or_none(row.get("high")),
            "low": to_float_or_none(row.get("low")),
            "close": to_float_or_none(row.get("close")),
            "volume": to_float_or_none(row.get("volume")),
            "ma50": to_float_or_none(row.get("ma50")),
            "ma200": to_float_or_none(row.get("ma200")),
            "rsi": to_float_or_none(row.get("rsi")),
            "atr": to_float_or_none(row.get("atr")),
            "volume_ratio": to_float_or_none(volume_ratio.get(timestamp)),
            "volume_threshold_ratio": to_float_or_none(threshold_ratio.get(timestamp)),
            "is_impulse": bool(is_impulse.get(timestamp)),
            "regime": str(regime_series.get(timestamp, "range")),
        }
        if all(candle[key] is not None for key in ("open", "high", "low", "close", "volume")):
            candles.append(candle)

    if not candles:
        raise ApiNotFoundError(f"No valid candles for {params['ticker']}")

    signal_payload = signal.to_dict()
    trade_plan = validate_trade_plan(signal_payload)

    return json_safe(
        {
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
                "last_swing_high": to_float_or_none(structure_info.get("last_swing_high")),
                "last_swing_low": to_float_or_none(structure_info.get("last_swing_low")),
                "swing_highs_count": int(structure_info.get("swing_highs_count", 0) or 0),
                "swing_lows_count": int(structure_info.get("swing_lows_count", 0) or 0),
            },
            "indicator_summary": {
                "rsi": to_float_or_none(local["rsi"].iloc[-1]) if "rsi" in local.columns else None,
                "atr": to_float_or_none(local["atr"].iloc[-1]) if "atr" in local.columns else None,
                "volume_ratio": to_float_or_none(volume_ratio.iloc[-1]) if not volume_ratio.empty else None,
                "volume_threshold_ratio": to_float_or_none(threshold_ratio.iloc[-1]) if not threshold_ratio.empty else None,
                "is_impulse": bool(is_impulse.iloc[-1]) if not is_impulse.empty else False,
                "rsi_oversold": 30.0,
                "rsi_overbought": 70.0,
            },
            "candles": candles,
        }
    )


def build_dashboard_backtest_response(payload: dict[str, Any]) -> dict[str, Any]:
    params = parse_request_params(payload, require_model=True)
    model = get_model_or_raise(params["model_name"])
    resolved_limit = coerce_int(payload.get("limit", 1200), default=1200, min_value=300, max_value=5000)
    lookback_window = coerce_int(payload.get("lookback_window", 300), default=300, min_value=20, max_value=2000)
    max_holding_candles = coerce_int(
        payload.get("max_holding_candles", 50),
        default=50,
        min_value=5,
        max_value=1000,
    )
    debug_filters = to_bool(payload.get("debug_filters"), default=True)
    run_context = resolve_system_run_context(payload, run_type="backtest")
    run_id = start_system_run(run_context, run_type="backtest", request_payload=payload)

    try:
        df, adapter = load_dataset(params, limit=resolved_limit)
        if df.empty:
            df, adapter = load_dataset(
                params,
                limit=resolved_limit,
                start_date="2010-01-01",
            )
        if df.empty:
            raise ApiNotFoundError(f"No data for {params['ticker']}")

        risk_params = resolve_contract_risk_params(adapter, params["ticker"], params["board"])

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

        equity_curve, drawdown_curve = build_equity_and_drawdown_curves(
            trades=results.trades,
            initial_balance=params["deposit"],
            start_time=df.index[0] if len(df.index) > 0 else None,
        )

        response_payload = {
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
            "trades": [serialize_trade(trade) for trade in results.trades],
            "equity_curve": equity_curve,
            "drawdown_curve": drawdown_curve,
            "filter_funnel": results.filter_stats.to_dict() if results.filter_stats else None,
        }
        finish_system_run_success(
            run_context=run_context,
            run_id=run_id,
            result_summary=response_payload["summary"],
            artifacts={
                "summary": response_payload["summary"],
                "trades": {"items": response_payload["trades"]},
                "equity_curve": {"items": response_payload["equity_curve"]},
                "drawdown_curve": {"items": response_payload["drawdown_curve"]},
                "filter_funnel": response_payload["filter_funnel"] or {},
            },
        )
        return json_safe(response_payload)
    except Exception as error:
        finish_system_run_failure(run_context=run_context, run_id=run_id, error_text=str(error))
        raise


def build_dashboard_robustness_response(payload: dict[str, Any]) -> dict[str, Any]:
    params = parse_request_params(payload, require_model=True)
    model = get_model_or_raise(params["model_name"])
    resolved_limit = coerce_int(payload.get("limit", 1500), default=1500, min_value=300, max_value=5000)
    monte_carlo_simulations = coerce_int(
        payload.get("monte_carlo_simulations", 300),
        default=300,
        min_value=0,
        max_value=5000,
    )
    adaptive_regime = to_bool(payload.get("adaptive_regime"), default=False)
    run_context = resolve_system_run_context(payload, run_type="robustness")
    run_id = start_system_run(run_context, run_type="robustness", request_payload=payload)

    try:
        df, adapter = load_dataset(params, limit=resolved_limit)
        if df.empty:
            df, adapter = load_dataset(
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

        risk_params = resolve_contract_risk_params(adapter, params["ticker"], params["board"])
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

        regime_series = classify_market_regime(with_atr(df))
        regime_timeline = serialize_regime_timeline(regime_series)

        if len(df) < 2:
            boundary = str(df.index[0]) if len(df.index) > 0 else ""
            train_period = {"start": boundary, "end": boundary}
            test_period = {"start": boundary, "end": boundary}
        else:
            train_ratio = coerce_float(signal_kwargs.get("train_ratio", 0.7), default=0.7, min_value=0.1, max_value=0.9)
            split_idx = int(len(df) * train_ratio)
            split_idx = max(1, min(split_idx, len(df) - 1))
            train_period = {"start": str(df.index[0]), "end": str(df.index[split_idx - 1])}
            test_period = {"start": str(df.index[split_idx]), "end": str(df.index[-1])}

        response_payload = {
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
                "pf_train": to_float_or_none(stats.get("pf_train")),
                "pf_test": to_float_or_none(stats.get("pf_test")),
                "maxdd_test": to_float_or_none(stats.get("maxdd_test")),
                "stability_ratio": to_float_or_none(stats.get("stability_ratio")),
                "unstable": bool(stats.get("unstable", False)),
                "unstable_oos": bool(stats.get("unstable_oos", False)),
                "overfit": bool(stats.get("overfit", False)),
                "robustness_score": to_float_or_none(stats.get("robustness_score")),
            },
            "market_regime_performance": stats.get("market_regime_performance", {}),
            "regime_timeline": regime_timeline,
            "monte_carlo": stats.get("monte_carlo"),
            "risk": {
                "risk_of_ruin": to_float_or_none(stats.get("risk_of_ruin")),
                "kelly_percent": round(float(kelly_percent), 2),
            },
            "admission": stats.get("admission", {}),
            "edge_found": bool(stats.get("edge_found", False)),
            "enabled_regimes": stats.get("enabled_regimes", []),
            "disabled_regimes": stats.get("disabled_regimes", {}),
            "train_regime_performance": stats.get("train_regime_performance", {}),
        }
        finish_system_run_success(
            run_context=run_context,
            run_id=run_id,
            result_summary={
                "edge_found": bool(response_payload["edge_found"]),
                "robustness_score": to_float_or_none(response_payload["robustness"].get("robustness_score")),
                "risk_of_ruin": to_float_or_none(response_payload["risk"].get("risk_of_ruin")),
            },
            artifacts={
                "robustness": response_payload,
                "regime_timeline": {"items": response_payload["regime_timeline"]},
                "train_stats": response_payload["train"],
                "test_stats": response_payload["test"],
            },
        )
        return json_safe(response_payload)
    except Exception as error:
        finish_system_run_failure(run_context=run_context, run_id=run_id, error_text=str(error))
        raise


def build_signal_response(payload: dict[str, Any]) -> dict[str, Any]:
    params = parse_request_params(payload, require_model=True)
    model = get_model_or_raise(params["model_name"])

    df, adapter = load_dataset(params)
    if df.empty:
        raise ApiNotFoundError(f"No data for {params['ticker']}")

    risk_params = resolve_contract_risk_params(adapter, params["ticker"], params["board"])
    signal = generate_signal(df, params["deposit"], model, **risk_params)

    return json_safe(
        {
            "ticker": params["ticker"],
            "exchange": params["exchange"],
            "timeframe": params["timeframe"],
            "model": params["model_name"],
            "data_points": len(df),
            "period": {"start": str(df.index[0]), "end": str(df.index[-1])},
            "signal": signal.to_dict(),
        }
    )


def build_backtest_response(payload: dict[str, Any]) -> dict[str, Any]:
    params = parse_request_params(payload, require_model=True)
    model = get_model_or_raise(params["model_name"])

    df, adapter = load_dataset(params)
    if df.empty:
        raise ApiNotFoundError(f"No data for {params['ticker']}")

    risk_params = resolve_contract_risk_params(adapter, params["ticker"], params["board"])

    results = run_backtest(
        df=df,
        signal_generator=generate_signal,
        deposit=params["deposit"],
        model=model,
        signal_kwargs=risk_params or None,
        execution_config={"sl_tp_only": True},
    )

    return json_safe(
        {
            "ticker": params["ticker"],
            "exchange": params["exchange"],
            "timeframe": params["timeframe"],
            "model": params["model_name"],
            "data_points": len(df),
            "period": {"start": str(df.index[0]), "end": str(df.index[-1])},
            "results": results.to_dict(),
        }
    )


def build_optimize_response(payload: dict[str, Any]) -> dict[str, Any]:
    params = parse_request_params(payload, require_model=False)

    df, adapter = load_dataset(params)
    if df.empty:
        raise ApiNotFoundError(f"No data for {params['ticker']}")

    risk_params = resolve_contract_risk_params(adapter, params["ticker"], params["board"])

    results = []
    for model_name in MODELS:
        model = get_model_or_raise(model_name)
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
    return json_safe(
        {
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
    )
