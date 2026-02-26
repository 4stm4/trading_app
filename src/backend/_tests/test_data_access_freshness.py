from __future__ import annotations

from datetime import datetime, timedelta
import unittest

import pandas as pd

from services.api_service.data_access import _is_dataset_stale, _max_dataset_age


class DataAccessFreshnessTests(unittest.TestCase):
    def test_recent_intraday_dataset_is_not_stale(self) -> None:
        now = datetime(2026, 2, 24, 12, 0, 0)
        frame = pd.DataFrame(
            {"close": [100.0]},
            index=pd.to_datetime([now - timedelta(hours=2)]),
        )
        self.assertFalse(_is_dataset_stale(frame, timeframe="10m", now=now))

    def test_old_intraday_dataset_is_stale(self) -> None:
        now = datetime(2026, 2, 24, 12, 0, 0)
        frame = pd.DataFrame(
            {"close": [100.0]},
            index=pd.to_datetime([now - timedelta(days=20)]),
        )
        self.assertTrue(_is_dataset_stale(frame, timeframe="10m", now=now))

    def test_old_daily_dataset_is_stale(self) -> None:
        now = datetime(2026, 2, 24, 12, 0, 0)
        frame = pd.DataFrame(
            {"close": [100.0]},
            index=pd.to_datetime([now - timedelta(days=25)]),
        )
        self.assertTrue(_is_dataset_stale(frame, timeframe="1d", now=now))

    def test_dataset_age_policy_by_timeframe(self) -> None:
        self.assertEqual(_max_dataset_age("10m"), timedelta(days=3))
        self.assertEqual(_max_dataset_age("1h"), timedelta(days=3))
        self.assertEqual(_max_dataset_age("1d"), timedelta(days=14))


if __name__ == "__main__":
    unittest.main()
