"""Serialization helpers for API payloads."""

from __future__ import annotations

from typing import Any

from adapters.postgres import TradingSystemVersionPostgresRepository
from entities.portfolio import Portfolio
from entities.trading_system import TradingSystem
from entities.user import User

from .helpers import json_safe, normalize_system_config, to_float_or_none, to_json_object


def serialize_trade(trade: Any) -> dict[str, Any]:
    from .helpers import to_unix_timestamp

    return {
        "entry_time": str(getattr(trade, "entry_time", "")),
        "exit_time": str(getattr(trade, "exit_time", "")),
        "entry_ts": to_unix_timestamp(getattr(trade, "entry_time", None)),
        "exit_ts": to_unix_timestamp(getattr(trade, "exit_time", None)),
        "entry_price": to_float_or_none(getattr(trade, "entry_price", None)),
        "exit_price": to_float_or_none(getattr(trade, "exit_price", None)),
        "stop_price": to_float_or_none(getattr(trade, "stop_price", None)),
        "target_price": to_float_or_none(getattr(trade, "target_price", None)),
        "direction": str(getattr(trade, "direction", "")),
        "position_size": to_float_or_none(getattr(trade, "position_size", None)),
        "pnl": to_float_or_none(getattr(trade, "pnl", None)),
        "pnl_percent": to_float_or_none(getattr(trade, "pnl_percent", None)),
        "exit_reason": str(getattr(trade, "exit_reason", "")),
        "rr_planned": to_float_or_none(getattr(trade, "rr_planned", None)),
        "rr_actual": to_float_or_none(getattr(trade, "rr_actual", None)),
        "duration_candles": int(getattr(trade, "duration_candles", 0) or 0),
        "regime": getattr(trade, "regime", None),
        "gross_pnl": to_float_or_none(getattr(trade, "gross_pnl", None)),
        "fees": to_float_or_none(getattr(trade, "fees", None)),
        "slippage": to_float_or_none(getattr(trade, "slippage", None)),
    }


def serialize_regime_timeline(regime_series: Any) -> list[dict[str, Any]]:
    from .helpers import to_unix_timestamp

    timeline: list[dict[str, Any]] = []
    if regime_series.empty:
        return timeline

    last_value: str | None = None
    for timestamp, value in regime_series.items():
        regime = str(value)
        if regime == last_value:
            continue
        ts = to_unix_timestamp(timestamp)
        if ts is None:
            continue
        timeline.append({"time": ts, "regime": regime})
        last_value = regime
    return timeline


def serialize_trading_system(system: TradingSystem, *, version_repo: TradingSystemVersionPostgresRepository) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if system.id is not None:
        current_version = version_repo.get_current(system_id=int(system.id))
        if current_version is not None:
            config = normalize_system_config(current_version.config_json)
    return {
        "id": int(system.id) if system.id is not None else None,
        "owner_user_id": int(system.owner_user_id),
        "name": str(system.name),
        "model_id": int(system.model_id) if system.model_id is not None else None,
        "model": str(system.model_name),
        "exchange": str(system.exchange),
        "engine": str(system.engine),
        "market": str(system.market),
        "board": str(system.board),
        "timeframe": str(system.timeframe),
        "is_active": bool(system.is_active),
        "is_current": bool(system.is_current),
        "config": config,
        "created_at": str(system.created_at) if system.created_at is not None else None,
        "updated_at": str(system.updated_at) if system.updated_at is not None else None,
    }


def serialize_portfolio(portfolio: Portfolio) -> dict[str, Any]:
    return {
        "id": int(portfolio.id) if portfolio.id is not None else None,
        "owner_user_id": int(portfolio.owner_user_id),
        "balance": float(portfolio.balance),
        "currency": str(portfolio.currency),
        "is_active": bool(portfolio.is_active),
        "created_at": str(portfolio.created_at) if portfolio.created_at is not None else None,
        "updated_at": str(portfolio.updated_at) if portfolio.updated_at is not None else None,
    }


def serialize_user(user: User) -> dict[str, Any]:
    return {
        "id": int(user.id) if user.id is not None else None,
        "email": str(user.email),
        "is_active": bool(user.is_active),
        "created_at": str(user.created_at) if user.created_at is not None else None,
        "updated_at": str(user.updated_at) if user.updated_at is not None else None,
    }


def serialize_system_run(run: Any) -> dict[str, Any]:
    request_json = getattr(run, "request_json", None)
    result_summary_json = getattr(run, "result_summary_json", None)
    return {
        "id": int(run.id) if getattr(run, "id", None) is not None else None,
        "owner_user_id": int(run.owner_user_id),
        "system_id": int(run.system_id),
        "portfolio_id": int(run.portfolio_id) if getattr(run, "portfolio_id", None) is not None else None,
        "portfolio_balance_snapshot": to_float_or_none(getattr(run, "portfolio_balance_snapshot", None)),
        "system_version_id": int(run.system_version_id) if getattr(run, "system_version_id", None) is not None else None,
        "run_type": str(getattr(run, "run_type", "")),
        "status": str(getattr(run, "status", "")),
        "request_json": json_safe(request_json) if request_json is not None else None,
        "result_summary_json": json_safe(result_summary_json) if result_summary_json is not None else None,
        "error_text": str(getattr(run, "error_text", "") or ""),
        "started_at": str(getattr(run, "started_at", "") or ""),
        "finished_at": str(getattr(run, "finished_at", "") or ""),
        "created_at": str(getattr(run, "created_at", "") or ""),
    }


def serialize_run_artifact(artifact: Any) -> dict[str, Any]:
    return {
        "id": int(artifact.id) if getattr(artifact, "id", None) is not None else None,
        "owner_user_id": int(artifact.owner_user_id),
        "run_id": int(artifact.run_id),
        "system_id": int(artifact.system_id),
        "system_version_id": int(artifact.system_version_id) if getattr(artifact, "system_version_id", None) is not None else None,
        "artifact_type": str(getattr(artifact, "artifact_type", "")),
        "payload_json": to_json_object(getattr(artifact, "payload_json", None)),
        "created_at": str(getattr(artifact, "created_at", "") or ""),
    }
