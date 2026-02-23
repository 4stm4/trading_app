from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class TradingSystemRunArtifact:
    owner_user_id: int
    run_id: int
    system_id: int
    artifact_type: str
    payload_json: dict[str, Any]
    system_version_id: int | None = None
    id: int | None = None
    created_at: datetime | None = None
