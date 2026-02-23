"""Market dashboard use-case."""

from __future__ import annotations

from typing import Any

from services.strategy_engine.public import calculate_structure, classify_market_regime, generate_signal

from .errors import ApiNotFoundError
from .helpers import (
    coerce_float,
    coerce_int,
    get_model_or_raise,
    json_safe,
    load_dataset,
    parse_request_params,
    resolve_contract_risk_params,
    to_float_or_none,
    to_unix_timestamp,
    validate_trade_plan,
    volume_context_series,
    with_atr,
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
