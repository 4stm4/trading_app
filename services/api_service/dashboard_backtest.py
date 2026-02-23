"""Backtest dashboard use-case."""

from __future__ import annotations

from typing import Any

from services.strategy_engine import generate_signal, run_backtest

from .errors import ApiNotFoundError
from .helpers import (
    build_equity_and_drawdown_curves,
    coerce_int,
    get_model_or_raise,
    json_safe,
    load_dataset,
    parse_request_params,
    resolve_contract_risk_params,
    to_bool,
)
from .serializers import serialize_trade
from .tracking import (
    finish_system_run_failure,
    finish_system_run_success,
    resolve_system_run_context,
    start_system_run,
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
