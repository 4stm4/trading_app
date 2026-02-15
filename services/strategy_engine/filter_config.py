"""
Конфигурация фильтров стратегии (YAML + safe defaults).
"""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import replace
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from .models import TradingModel


DEFAULT_FILTERS: Dict[str, Any] = {
    "filters": {
        "volume": {
            "enabled": True,
            "mode": "fixed",
            "min_ratio": 1.1,
            "high_vol_ratio": 1.5,
        },
        "trend": {
            "enabled": True,
            "min_strength_pct": 3.0,
            "require_ma_alignment": True,
            "allow_range": False,
        },
        "pullback": {
            "enabled": True,
            "max_ma50_distance_pct": 1.5,
            "require_pullback": True,
        },
        "rsi": {
            "enabled": True,
            "overbought": 70,
            "oversold": 30,
            "trend_confirmation_only": False,
        },
        "atr": {
            "enabled": True,
            "stop_multiplier": 1.0,
            "min_percentile": 10,
        },
        "risk": {
            "risk_per_trade_pct": 1,
            "min_stop_distance_pct": 0.7,
            "max_position_notional_pct": 80.0,
            "position_step": 1.0,
            "futures_margin_safety_factor": 1.4,
            "max_daily_loss_pct": 3.0,
            "equity_dd_stop_pct": 15.0,
            "partial_take_profit": False,
            "partial_tp_rr": 1.0,
            "partial_close_fraction": 0.5,
            "trail_after_partial": False,
            "trailing_atr_multiplier": 1.0,
        },
        "rr": {
            "min_rr": 2.5,
        },
        "regime": {
            "enabled": True,
            "trade_trend": True,
            "trade_range": True,
            "trade_high_vol": True,
            "min_regime_pf": 1.1,
            "high_vol_pf_threshold": 1.05,
            "trend_pf_threshold": 1.1,
            "range_pf_threshold": 1.1,
        },
        "costs": {
            "enabled": True,
            "futures_per_contract_rub": 2.0,
            "futures_fee_mode": "round_trip",
            "securities_bps": 4.9,
            "settlement_bps": 2.0,
            "slippage_bps": 0.0,
        },
        "scoring": {
            "conservative_v2_enabled": True,
            "conservative_min_score": 0.65,
            "max_threshold_relaxation": 0.20,
            "atr_relaxation_bonus": 0.08,
            "trend_relaxation_bonus": 0.05,
            "range_relaxation_bonus": 0.03,
            "min_expected_trades_per_month": 35.0,
            "weights": {
                "trend_alignment": 0.25,
                "trend_strength": 0.25,
                "volume": 0.15,
                "rsi": 0.10,
                "pullback": 0.25,
                "ma_distance": 0.0,
                "structure": 0.0,
            },
        },
    }
}
DEFAULT_CONFIG_FILENAME = "strict.yaml"


def deep_merge(default: dict, override: dict) -> dict:
    """Чистый рекурсивный merge словарей."""
    result = deepcopy(default)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def resolve_config_path(path: Optional[str]) -> Optional[str]:
    """
    Разрешает путь к конфигу:
    1) явный path,
    2) переменная окружения TRADING_FILTER_CONFIG,
    3) ./strict.yaml (если существует),
    4) None -> встроенные fallback defaults.
    """
    if path:
        return path

    env_path = os.getenv("TRADING_FILTER_CONFIG")
    if env_path:
        return env_path

    if os.path.exists(DEFAULT_CONFIG_FILENAME):
        return DEFAULT_CONFIG_FILENAME

    return None


def load_config(path: Optional[str]) -> Dict[str, Any]:
    """
    Загружает YAML-конфиг фильтров.
    Если путь не передан, возвращает DEFAULT_FILTERS.
    """
    config = deepcopy(DEFAULT_FILTERS)
    resolved_path = resolve_config_path(path)

    if not resolved_path:
        validate_filter_config(config)
        return config

    if yaml is None:
        raise ValueError("PyYAML не установлен. Установите зависимость 'PyYAML'.")

    try:
        with open(resolved_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    except FileNotFoundError as exc:
        raise ValueError(f"Config файл не найден: {resolved_path}") from exc
    except Exception as exc:
        raise ValueError(f"Ошибка чтения config YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError("Config должен быть YAML-словарем верхнего уровня")

    merged = deep_merge(config, raw)
    validate_filter_config(merged)
    return merged


def validate_filter_config(config: Dict[str, Any]) -> None:
    """Валидация диапазонов и типов фильтров."""
    filters = config.get("filters")
    if not isinstance(filters, dict):
        raise ValueError("Config должен содержать блок 'filters'")

    volume = _req_dict(filters, "volume")
    trend = _req_dict(filters, "trend")
    pullback = _req_dict(filters, "pullback")
    rsi = _req_dict(filters, "rsi")
    atr = _req_dict(filters, "atr")
    risk = _req_dict(filters, "risk")
    regime = _req_dict(filters, "regime")
    costs = _req_dict(filters, "costs")
    scoring = _req_dict(filters, "scoring")
    rr = filters.get("rr", {})

    _req_bool(volume, "enabled")
    _req_choice(volume, "mode", {"fixed", "adaptive"})
    _req_positive(volume, "min_ratio")
    _req_positive(volume, "high_vol_ratio")

    _req_bool(trend, "enabled")
    _req_non_negative(trend, "min_strength_pct")
    _req_bool(trend, "require_ma_alignment")
    _req_bool(trend, "allow_range")

    _req_bool(pullback, "enabled")
    _req_non_negative(pullback, "max_ma50_distance_pct")
    _req_bool(pullback, "require_pullback")

    _req_bool(rsi, "enabled")
    _req_between(rsi, "overbought", 1, 100)
    _req_between(rsi, "oversold", 0, 99)
    if int(rsi["oversold"]) >= int(rsi["overbought"]):
        raise ValueError("filters.rsi.oversold должен быть меньше filters.rsi.overbought")
    _req_bool(rsi, "trend_confirmation_only")

    _req_bool(atr, "enabled")
    _req_positive(atr, "stop_multiplier")
    _req_between(atr, "min_percentile", 0, 100)

    _req_positive(risk, "risk_per_trade_pct")
    _req_non_negative(risk, "min_stop_distance_pct")
    _req_positive(risk, "max_position_notional_pct")
    _req_non_negative(risk, "position_step")
    _req_positive(risk, "futures_margin_safety_factor")
    _req_non_negative(risk, "max_daily_loss_pct")
    _req_positive(risk, "equity_dd_stop_pct")
    _req_bool(risk, "partial_take_profit")
    _req_positive(risk, "partial_tp_rr")
    _req_between(risk, "partial_close_fraction", 0.0, 1.0)
    _req_bool(risk, "trail_after_partial")
    _req_positive(risk, "trailing_atr_multiplier")

    _req_bool(regime, "enabled")
    _req_bool(regime, "trade_trend")
    _req_bool(regime, "trade_range")
    _req_bool(regime, "trade_high_vol")
    _req_non_negative(regime, "min_regime_pf")
    _req_non_negative(regime, "high_vol_pf_threshold")
    _req_non_negative(regime, "trend_pf_threshold")
    _req_non_negative(regime, "range_pf_threshold")

    _req_bool(costs, "enabled")
    _req_non_negative(costs, "futures_per_contract_rub")
    _req_choice(costs, "futures_fee_mode", {"round_trip", "per_side"})
    _req_non_negative(costs, "securities_bps")
    _req_non_negative(costs, "settlement_bps")
    _req_non_negative(costs, "slippage_bps")

    _req_bool(scoring, "conservative_v2_enabled")
    _req_between(scoring, "conservative_min_score", 0.0, 1.0)
    _req_between(scoring, "max_threshold_relaxation", 0.0, 1.0)
    _req_between(scoring, "atr_relaxation_bonus", 0.0, 1.0)
    _req_between(scoring, "trend_relaxation_bonus", 0.0, 1.0)
    _req_between(scoring, "range_relaxation_bonus", 0.0, 1.0)
    _req_non_negative(scoring, "min_expected_trades_per_month")

    weights = _req_dict(scoring, "weights")
    for key in (
        "trend_alignment",
        "trend_strength",
        "volume",
        "rsi",
        "pullback",
        "ma_distance",
        "structure",
    ):
        _req_non_negative(weights, key)

    if rr and not isinstance(rr, dict):
        raise ValueError("filters.rr должен быть словарем")
    min_rr = rr.get("min_rr", 2.5) if isinstance(rr, dict) else 2.5
    if not isinstance(min_rr, (int, float)) or float(min_rr) <= 0:
        raise ValueError("filters.rr.min_rr должен быть > 0")
    if float(min_rr) < 2.5:
        raise ValueError("filters.rr.min_rr должен быть >= 2.5")


def apply_filters_to_model(
    model: TradingModel,
    config: Dict[str, Any],
    regime: Optional[str] = None,
) -> TradingModel:
    """Применяет конфиг фильтров к модели (чистая функция)."""
    filters = config["filters"]
    volume = filters["volume"]
    trend = filters["trend"]
    rsi = filters["rsi"]
    atr = filters["atr"]
    risk = filters["risk"]
    pullback = filters["pullback"]
    rr = filters.get("rr", {})

    min_volume_ratio = float(volume["min_ratio"])
    if regime == "high_volatility":
        min_volume_ratio = float(volume["high_vol_ratio"])

    if trend["enabled"]:
        trend_required = bool(model.trend_required and trend["require_ma_alignment"])
        allow_range = bool(trend["allow_range"])
    else:
        trend_required = False
        allow_range = True

    if rsi["enabled"]:
        rsi_overbought = int(rsi["overbought"])
        rsi_oversold = int(rsi["oversold"])
    else:
        rsi_overbought = 100
        rsi_oversold = 0

    return replace(
        model,
        min_volume_ratio=min_volume_ratio,
        min_trend_strength=float(trend["min_strength_pct"]),
        trend_required=trend_required,
        allow_range=allow_range,
        rsi_overbought=rsi_overbought,
        rsi_oversold=rsi_oversold,
        atr_multiplier_stop=float(atr["stop_multiplier"]),
        max_risk_percent=float(risk["risk_per_trade_pct"]),
        max_distance_ma50=float(pullback["max_ma50_distance_pct"]),
        min_rr=max(float(rr.get("min_rr", 2.5)), 2.5),
    )


def _req_dict(container: dict, key: str) -> dict:
    value = container.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Ожидался словарь filters.{key}")
    return value


def _req_bool(container: dict, key: str) -> None:
    if not isinstance(container.get(key), bool):
        raise ValueError(f"filters.{key} должен быть bool")


def _req_choice(container: dict, key: str, allowed: set[str]) -> None:
    value = container.get(key)
    if value not in allowed:
        raise ValueError(f"filters.{key} должен быть одним из {sorted(allowed)}")


def _req_positive(container: dict, key: str) -> None:
    value = container.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"filters.{key} должен быть числом")
    if value <= 0:
        raise ValueError(f"filters.{key} должен быть > 0")


def _req_non_negative(container: dict, key: str) -> None:
    value = container.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"filters.{key} должен быть числом")
    if value < 0:
        raise ValueError(f"filters.{key} должен быть >= 0")


def _req_between(container: dict, key: str, low: float, high: float) -> None:
    value = container.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"filters.{key} должен быть числом")
    if value < low or value > high:
        raise ValueError(f"filters.{key} должен быть в диапазоне [{low}, {high}]")
