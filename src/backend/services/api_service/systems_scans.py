"""User scan-history use-cases."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from adapters.postgres import (
    TradingSystemPostgresRepository,
    TradingSystemScanPostgresRepository,
    TradingSystemVersionPostgresRepository,
    session_scope,
)
from entities.trading_system_scan import TradingSystemScan

from .auth import ensure_system_user, extract_user_email
from .errors import ApiNotFoundError, ApiServiceError, ApiValidationError
from .helpers import coerce_int, get_db_session_factory, json_safe, to_bool
from .serializers import serialize_system_scan


def build_scans_response(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    session_factory = get_db_session_factory()
    if session_factory is None:
        raise ApiServiceError("Database is not configured")

    body = payload or {}
    user_email = extract_user_email(body, required=True)
    limit = coerce_int(body.get("limit", 200), default=200, min_value=1, max_value=2000)
    tradable_only = to_bool(body.get("tradable_only"), default=False)
    signal = str(body.get("signal") or "").strip().lower() or None
    scan_key = str(body.get("scan_key") or "").strip() or None
    system_id = _coerce_optional_positive_int(body.get("system_id"))

    with session_scope(session_factory) as session:
        user = ensure_system_user(session, user_email=user_email)
        system_repo = TradingSystemPostgresRepository(session)
        scan_repo = TradingSystemScanPostgresRepository(session)

        systems = system_repo.list_by_owner(owner_user_id=int(user.id), only_active=False, limit=500)
        system_name_by_id = {int(item.id): str(item.name) for item in systems if item.id is not None}

        if scan_key:
            scans = scan_repo.list_by_scan_key(
                owner_user_id=int(user.id),
                scan_key=scan_key,
                tradable_only=tradable_only,
                limit=limit,
            )
            if system_id is not None:
                scans = [item for item in scans if int(item.system_id) == int(system_id)]
            if signal:
                scans = [item for item in scans if str(item.signal).strip().lower() == signal]
        else:
            scans = scan_repo.list_by_owner(
                owner_user_id=int(user.id),
                system_id=system_id,
                tradable_only=tradable_only,
                signal=signal,
                limit=limit,
            )

        serialized = [
            serialize_system_scan(item, system_name=system_name_by_id.get(int(item.system_id)))
            for item in scans
        ]
        sessions = _build_scan_sessions(serialized)

        return json_safe(
            {
                "owner_user_id": int(user.id),
                "count": len(serialized),
                "scans": serialized,
                "sessions": sessions,
            }
        )


def build_scans_create_response(payload: dict[str, Any]) -> dict[str, Any]:
    session_factory = get_db_session_factory()
    if session_factory is None:
        raise ApiServiceError("Database is not configured")

    body = payload or {}
    user_email = extract_user_email(body, required=True)
    system_id = _coerce_required_positive_int(body.get("system_id"), field_name="system_id")
    raw_scans = body.get("scans")
    if not isinstance(raw_scans, list) or not raw_scans:
        raise ApiValidationError("scans must be a non-empty array")
    scan_key = str(body.get("scan_key") or "").strip() or _default_scan_key()
    generated_at_default = _parse_optional_datetime(body.get("generated_at")) or datetime.now(timezone.utc)

    with session_scope(session_factory) as session:
        user = ensure_system_user(session, user_email=user_email)
        system_repo = TradingSystemPostgresRepository(session)
        version_repo = TradingSystemVersionPostgresRepository(session)
        scan_repo = TradingSystemScanPostgresRepository(session)

        system = system_repo.get_by_id(int(system_id))
        if system is None or system.id is None or int(system.owner_user_id) != int(user.id):
            raise ApiNotFoundError("System not found")
        current_version = version_repo.get_current(system_id=int(system.id))
        system_version_id = int(current_version.id) if current_version is not None and current_version.id is not None else None

        created: list[TradingSystemScan] = []
        for item in raw_scans:
            if not isinstance(item, dict):
                continue
            scan = TradingSystemScan(
                owner_user_id=int(user.id),
                system_id=int(system.id),
                system_version_id=system_version_id,
                scan_key=scan_key,
                exchange=str(item.get("exchange") or str(system.exchange or "moex")),
                engine=str(item.get("engine") or str(system.engine or "stock")),
                market=str(item.get("market") or str(system.market or "shares")),
                board=str(item.get("board") or str(system.board or "")),
                symbol=str(item.get("symbol") or "").strip().upper(),
                timeframe=str(item.get("timeframe") or str(system.timeframe or "1h")),
                model_name=str(item.get("model_name") or item.get("model") or str(system.model_name or "balanced")),
                signal=str(item.get("signal") or "none"),
                confidence=str(item.get("confidence") or "none"),
                tradable=to_bool(item.get("tradable"), default=False),
                entry=_to_float_or_none(item.get("entry")),
                stop=_to_float_or_none(item.get("stop")),
                target=_to_float_or_none(item.get("target")),
                rr=_to_float_or_none(item.get("rr")),
                market_regime=_to_text_or_none(item.get("market_regime")),
                phase=_to_text_or_none(item.get("phase")),
                issues_json=_to_issues_json(item.get("issues_json"), item.get("issues")),
                generated_at=_parse_optional_datetime(item.get("generated_at")) or generated_at_default,
            )
            if not scan.symbol:
                continue
            created.append(scan_repo.upsert_by_scan_key(scan))

        serialized = [serialize_system_scan(item, system_name=str(system.name)) for item in created]
        return json_safe(
            {
                "owner_user_id": int(user.id),
                "system_id": int(system.id),
                "system_name": str(system.name),
                "scan_key": scan_key,
                "count": len(serialized),
                "scans": serialized,
            }
        )


def _coerce_required_positive_int(value: Any, *, field_name: str) -> int:
    parsed = _coerce_optional_positive_int(value)
    if parsed is None:
        raise ApiValidationError(f"{field_name} must be a positive integer")
    return parsed


def _coerce_optional_positive_int(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _to_float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _to_text_or_none(value: Any) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None


def _to_issues_json(issues_json: Any, issues: Any) -> dict[str, Any] | None:
    if isinstance(issues_json, dict):
        return dict(issues_json)
    if isinstance(issues, list):
        normalized = [str(item) for item in issues]
        return {"issues": normalized}
    return None


def _parse_optional_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    iso_value = raw.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(iso_value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _default_scan_key() -> str:
    return datetime.now(timezone.utc).strftime("scan-%Y%m%dT%H%M%S")


def _build_scan_sessions(scans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sessions: dict[str, dict[str, Any]] = {}
    for item in scans:
        key = str(item.get("scan_key") or "").strip()
        if not key:
            continue
        bucket = sessions.get(key)
        if bucket is None:
            bucket = {
                "scan_key": key,
                "count": 0,
                "tradable_count": 0,
                "created_at": item.get("created_at"),
                "systems": set(),
                "models": set(),
            }
            sessions[key] = bucket
        bucket["count"] += 1
        if bool(item.get("tradable")):
            bucket["tradable_count"] += 1
        created_at = str(item.get("created_at") or "")
        if created_at and str(bucket.get("created_at") or "") < created_at:
            bucket["created_at"] = created_at
        system_name = str(item.get("system_name") or "").strip()
        if system_name:
            bucket["systems"].add(system_name)
        model_name = str(item.get("model_name") or "").strip()
        if model_name:
            bucket["models"].add(model_name)

    result: list[dict[str, Any]] = []
    for value in sessions.values():
        result.append(
            {
                "scan_key": value["scan_key"],
                "count": int(value["count"]),
                "tradable_count": int(value["tradable_count"]),
                "created_at": value.get("created_at"),
                "systems": sorted(value["systems"]),
                "models": sorted(value["models"]),
            }
        )
    result.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return result
