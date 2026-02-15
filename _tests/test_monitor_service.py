from __future__ import annotations

import unittest
from datetime import datetime

from services.strategy_engine.monitor_core import MonitorRiskConfig
from services.strategy_engine.monitor_service import MonitorRunConfig, run_monitoring


class _FakePriceFeed:
    def __init__(self) -> None:
        self.calls = 0

    def collect_prices(self, setups):
        self.calls += 1
        return {"SiH6": {"price": 105.0, "price_step": 1.0, "contract_multiplier": 1.0}}, {}


class _FakeNotifier:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def send(self, message: str, level: str = "info") -> None:
        self.messages.append((level, message))


class _FakePersistence:
    def __init__(self) -> None:
        self.saved: list[list[dict]] = []

    def save_active_setups(self, setups):
        self.saved.append(list(setups))


class MonitorServiceTests(unittest.TestCase):
    def test_run_monitoring_uses_ports_and_persists(self) -> None:
        setup = {
            "symbol": "SiH6",
            "direction": "long",
            "entry_price": 100.0,
            "stop_loss": 95.0,
            "take_profit": 110.0,
            "rr": 2.0,
            "status": "WAITING_FOR_ENTRY",
            "last_price": 99.0,
            "expires_at": "2026-12-31 23:59:59",
        }

        feed = _FakePriceFeed()
        notifier = _FakeNotifier()
        persistence = _FakePersistence()
        risk = MonitorRiskConfig(risk_per_trade_pct=0.3, max_daily_loss_pct=100.0, equity_dd_stop_pct=100.0)
        now = datetime(2026, 2, 15, 12, 0, 0)

        remaining = run_monitoring(
            setups=[setup],
            price_feed=feed,
            notifier=notifier,
            risk_config=risk,
            run_config=MonitorRunConfig(interval_seconds=0, max_cycles=1, skip_entry_on_first_cycle=False),
            persistence=persistence,
            now_fn=lambda: now,
            sleep_fn=lambda _: None,
        )

        self.assertEqual(feed.calls, 1)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["status"], "ACTIVE")
        self.assertTrue(len(persistence.saved) >= 1)
        self.assertTrue(any("ENTRY HIT" in msg for _, msg in notifier.messages))


if __name__ == "__main__":
    unittest.main()
