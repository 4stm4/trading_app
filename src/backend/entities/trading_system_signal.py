from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class TradingSystemSignal:
    system_id: int
    exchange: str
    symbol: str
    timeframe: str = "1h"
    model_name: str = "balanced"
    signal: str = "none"
    confidence: str = "none"
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    rr: float | None = None
    market_regime: str | None = None
    phase: str | None = None
    system_version_id: int | None = None
    id: int | None = None
    generated_at: datetime | None = None
    created_at: datetime | None = None
