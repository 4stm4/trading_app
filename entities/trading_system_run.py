from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class TradingSystemRun:
    system_id: int
    run_type: str
    status: str = "pending"
    system_version_id: int | None = None
    request_json: dict[str, Any] | None = None
    result_summary_json: dict[str, Any] | None = None
    error_text: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    id: int | None = None
    created_at: datetime | None = None
