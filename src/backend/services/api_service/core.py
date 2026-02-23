"""Core metadata endpoints."""

from __future__ import annotations

from typing import Any

from services.strategy_engine.public import MODELS

from .helpers import json_safe


def build_health_response() -> dict[str, Any]:
    return json_safe(
        {
            "status": "ok",
            "service": "Trading System API",
            "version": "1.0.0",
            "models_count": len(MODELS),
        }
    )


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
