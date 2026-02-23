"""
Мониторинг активных сетапов: entry/stop/tp/expiry.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from pyjobkit import Engine, Worker
from pyjobkit.backends.memory import MemoryBackend
from pyjobkit.contracts import ExecContext, Executor

from services.strategy_engine import (
    MonitorRiskConfig,
    MonitorRunConfig,
    run_monitoring,
)
from services.strategy_engine.setup_generator import export_setups_json

from services.notification.alert_sender import AlertSender


class _JsonSetupPersistence:
    def __init__(self, export_path: str):
        self.export_path = export_path

    def save_active_setups(self, setups: list[dict[str, Any]]) -> None:
        export_setups_json(self.export_path, setups)


class _MoexPriceFeed:
    def __init__(self, adapter, board: str, monitor_workers: int):
        self.adapter = adapter
        self.board = board
        self.monitor_workers = monitor_workers
        self.spec_cache: dict[str, dict[str, float]] = {}

    def collect_prices(
        self, setups: list[dict[str, Any]]
    ) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
        return _collect_prices(
            setups=setups,
            adapter=self.adapter,
            board=self.board,
            monitor_workers=self.monitor_workers,
            spec_cache=self.spec_cache,
        )


def monitor_setups(
    setups: list[dict[str, Any]],
    adapter,
    board: str,
    interval_seconds: int = 15,
    export_path: str | None = None,
    sender: AlertSender | None = None,
    max_cycles: int = 0,
    filter_config: dict[str, Any] | None = None,
    monitor_workers: int = 10,
) -> list[dict[str, Any]]:
    """
    Следит за сетапами до их закрытия/истечения.

    Args:
        setups: список сетапов-словарей
        adapter: MOEXAdapter
        board: board для котировок
        interval_seconds: частота опроса
        export_path: путь для обновления активного списка сетапов
        sender: отправщик уведомлений
        max_cycles: 0 = бесконечно, иначе ограничение итераций
    """
    sender = sender or AlertSender()
    risk_per_trade_pct, max_daily_loss_pct, equity_dd_stop_pct = _extract_monitor_risk(
        filter_config
    )
    risk_cfg = MonitorRiskConfig(
        risk_per_trade_pct=risk_per_trade_pct,
        max_daily_loss_pct=max_daily_loss_pct,
        equity_dd_stop_pct=equity_dd_stop_pct,
    )
    run_config = MonitorRunConfig(
        interval_seconds=interval_seconds,
        max_cycles=max_cycles,
        skip_entry_on_first_cycle=True,
    )
    persistence = _JsonSetupPersistence(export_path) if export_path else None
    price_feed = _MoexPriceFeed(
        adapter=adapter,
        board=board,
        monitor_workers=monitor_workers,
    )
    return run_monitoring(
        setups=setups,
        price_feed=price_feed,
        notifier=sender,
        risk_config=risk_cfg,
        run_config=run_config,
        persistence=persistence,
    )


def _collect_prices(
    setups: list[dict[str, Any]],
    adapter,
    board: str,
    monitor_workers: int,
    spec_cache: dict[str, dict[str, float]],
) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    symbols = sorted({str(s.get("symbol", "")) for s in setups if s.get("symbol")})
    if not symbols:
        return {}, {}
    return asyncio.run(
        _collect_prices_pyjobkit(
            symbols=symbols,
            adapter=adapter,
            board=board,
            monitor_workers=monitor_workers,
            spec_cache=spec_cache,
        )
    )


class _PriceFetchExecutor(Executor):
    kind = "moex.price.fetch"

    def __init__(self, adapter, board: str):
        self.adapter = adapter
        self.board = board

    async def run(self, *, job_id: UUID, payload: dict, ctx: ExecContext) -> dict:
        _ = job_id, ctx
        symbol = str(payload.get("symbol", ""))
        if not symbol:
            return {"ok": False, "error": "empty symbol"}

        try:
            quote = await asyncio.to_thread(self.adapter.get_current_price, symbol, board=self.board)
        except Exception as exc:
            return {"ok": False, "symbol": symbol, "error": str(exc)}

        price = _extract_price_from_quote(quote)
        if price is None or price <= 0:
            return {"ok": False, "symbol": symbol, "error": "Пустая/некорректная котировка"}
        return {
            "ok": True,
            "symbol": symbol,
            "price": float(price),
            "price_step": _to_float_or_none(payload.get("price_step")),
            "contract_multiplier": _to_float_or_none(payload.get("contract_multiplier")) or 1.0,
        }


async def _collect_prices_pyjobkit(
    *,
    symbols: list[str],
    adapter,
    board: str,
    monitor_workers: int,
    spec_cache: dict[str, dict[str, float]],
) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    prices: dict[str, dict[str, float]] = {}
    errors: dict[str, str] = {}

    backend = MemoryBackend(lease_ttl_s=30)
    executor = _PriceFetchExecutor(adapter=adapter, board=board)
    engine = Engine(backend=backend, executors=[executor])

    workers = max(1, min(int(monitor_workers), len(symbols)))
    worker = Worker(
        engine=engine,
        max_concurrency=workers,
        batch=workers,
        poll_interval=0.01,
        lease_ttl=30,
        stop_timeout=5,
    )
    worker_task = asyncio.create_task(worker.run())
    job_map: dict[UUID, str] = {}

    try:
        for symbol in symbols:
            spec = _resolve_symbol_spec(
                adapter=adapter,
                symbol=symbol,
                board=board,
                spec_cache=spec_cache,
            )
            job_id = await engine.enqueue(
                kind=_PriceFetchExecutor.kind,
                payload={
                    "symbol": symbol,
                    "price_step": spec.get("price_step"),
                    "contract_multiplier": spec.get("contract_multiplier", 1.0),
                },
                max_attempts=1,
                timeout_s=15,
                scheduled_for=datetime.now(timezone.utc),
            )
            job_map[job_id] = symbol

        pending: set[UUID] = set(job_map.keys())
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(8.0, len(symbols) * 1.5)

        while pending:
            finished: list[tuple[UUID, dict[str, Any]]] = []
            for job_id in list(pending):
                try:
                    job = await engine.get(job_id)
                except Exception as exc:  # pragma: no cover - defensive
                    finished.append((job_id, {"status": "failed", "result": {"error": str(exc)}}))
                    continue

                status = str(job.get("status", ""))
                if status in {"success", "failed", "timeout", "cancelled"}:
                    finished.append((job_id, job))

            if not finished:
                if loop.time() >= deadline:
                    for job_id in list(pending):
                        symbol = job_map[job_id]
                        errors[symbol] = "timeout waiting pyjobkit result"
                    break
                await asyncio.sleep(0.01)
                continue

            for job_id, job in finished:
                if job_id not in pending:
                    continue
                pending.remove(job_id)
                symbol = job_map[job_id]
                result = job.get("result") or {}
                if str(job.get("status")) == "success" and bool(result.get("ok", False)):
                    price = _to_float_or_none(result.get("price"))
                    if price is not None and price > 0:
                        prices[symbol] = {
                            "price": float(price),
                            "price_step": _to_float_or_none(result.get("price_step")),
                            "contract_multiplier": _to_float_or_none(result.get("contract_multiplier")) or 1.0,
                        }
                        continue
                errors[symbol] = str(result.get("error") or job.get("status") or "unknown error")
    finally:
        worker.request_stop()
        with suppress(Exception):
            await worker.wait_stopped()
        worker_task.cancel()
        with suppress(asyncio.CancelledError):
            await worker_task

    return prices, errors


def _resolve_symbol_spec(
    *,
    adapter,
    symbol: str,
    board: str,
    spec_cache: dict[str, dict[str, float]],
) -> dict[str, float]:
    if symbol in spec_cache:
        return spec_cache[symbol]

    spec: dict[str, float] = {"price_step": 0.0, "contract_multiplier": 1.0}
    getter = getattr(adapter, "get_security_spec", None)
    if callable(getter):
        try:
            raw = getter(symbol, board=board) or {}
            spec["price_step"] = _to_float_or_none(raw.get("price_step")) or 0.0
            spec["contract_multiplier"] = _to_float_or_none(raw.get("contract_multiplier")) or 1.0
        except Exception:
            pass
    spec_cache[symbol] = spec
    return spec


def _extract_price_from_quote(quote: Any) -> float | None:
    if not isinstance(quote, dict):
        return None

    # Основной источник для last trade price.
    price = _to_float_or_none(quote.get("last"))
    if price is not None and price > 0:
        return price

    # Fallback на close/market price.
    for key in ("close", "marketprice", "price"):
        price = _to_float_or_none(quote.get(key))
        if price is not None and price > 0:
            return price
    return None


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_monitor_risk(filter_config: dict[str, Any] | None) -> tuple[float, float, float]:
    if not filter_config:
        return 0.7, 3.0, 15.0

    filters = filter_config.get("filters", {})
    risk = filters.get("risk", {})
    risk_per_trade = _to_float_or_none(risk.get("risk_per_trade_pct"))
    max_daily_loss = _to_float_or_none(risk.get("max_daily_loss_pct"))
    equity_dd_stop = _to_float_or_none(risk.get("equity_dd_stop_pct"))
    return (
        0.7 if risk_per_trade is None else risk_per_trade,
        3.0 if max_daily_loss is None else max_daily_loss,
        15.0 if equity_dd_stop is None else equity_dd_stop,
    )
