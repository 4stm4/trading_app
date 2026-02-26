from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class TradingSystemScan:
    owner_user_id: int
    system_id: int
    scan_key: str
    exchange: str
    symbol: str
    timeframe: str = "1h"
    model_name: str = "balanced"
    signal: str = "none"
    confidence: str = "none"
    engine: str = "stock"
    market: str = "shares"
    board: str = ""
    tradable: bool = False
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    rr: float | None = None
    market_regime: str | None = None
    phase: str | None = None
    issues_json: dict[str, Any] | None = None
    system_version_id: int | None = None
    id: int | None = None
    generated_at: datetime | None = None
    created_at: datetime | None = None
