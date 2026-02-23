"""Signal, backtest and model-optimization market use-cases."""

from __future__ import annotations

from typing import Any

from services.strategy_engine import MODELS, generate_signal, run_backtest

from .errors import ApiNotFoundError
from .helpers import (
    get_model_or_raise,
    json_safe,
    load_dataset,
    parse_request_params,
    resolve_contract_risk_params,
)


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
