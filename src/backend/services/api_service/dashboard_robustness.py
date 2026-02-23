"""Robustness dashboard use-case."""

from __future__ import annotations

from typing import Any

from services.strategy_engine.public import calculate_kelly_criterion, classify_market_regime, evaluate_model

from .errors import ApiNotFoundError
from .helpers import (
    coerce_float,
    coerce_int,
    get_model_or_raise,
    json_safe,
    load_dataset,
    parse_request_params,
    resolve_contract_risk_params,
    to_bool,
    to_float_or_none,
    with_atr,
)
from .serializers import serialize_regime_timeline
from .tracking import (
    finish_system_run_failure,
    finish_system_run_success,
    resolve_system_run_context,
    start_system_run,
)


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
