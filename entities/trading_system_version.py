from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class TradingSystemVersion:
    system_id: int
    version: int
    config_json: dict[str, Any]
    is_current: bool = True
    created_by_user_id: int | None = None
    id: int | None = None
    created_at: datetime | None = None
