"""
Binance market data adapter.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
import time
from typing import Optional

import pandas as pd
from binance.spot import Spot


logger = logging.getLogger(__name__)


class BinanceAdapter:
    """Adapter for Binance Spot market data."""

    _TIMEFRAME_TO_INTERVAL: dict[str, str] = {
        "1m": "1m",
        "5m": "5m",
        "10m": "5m",  # derived by resampling
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
        "1w": "1w",
        "1M": "1M",
    }

    _INTERVAL_MS: dict[str, int] = {
        "1m": 60_000,
        "5m": 300_000,
        "10m": 600_000,
        "15m": 900_000,
        "30m": 1_800_000,
        "1h": 3_600_000,
        "4h": 14_400_000,
        "1d": 86_400_000,
        "1w": 604_800_000,
        "1M": 2_592_000_000,  # 30d approximation
    }

    def __init__(
        self,
        engine: str = "spot",
        market: str = "spot",
        max_retries: int = 4,
        retry_backoff_sec: float = 1.5,
    ):
        normalized_engine = str(engine or "spot").lower()
        normalized_market = str(market or "spot").lower()

        if normalized_engine in {"stock", "spot"}:
            normalized_engine = "spot"
        if normalized_market in {"shares", "spot"}:
            normalized_market = "spot"

        if normalized_engine != "spot":
            raise NotImplementedError(
                f"Binance engine '{engine}' is not implemented yet. Use spot."
            )

        self.engine = normalized_engine
        self.market = normalized_market
        self.max_retries = max(1, int(max_retries))
        self.retry_backoff_sec = max(0.1, float(retry_backoff_sec))
        self.client = Spot()

    def get_candles(
        self,
        ticker: str,
        timeframe: str = "1h",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        board: str = "",
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        del board  # not used by Binance
        interval = self._TIMEFRAME_TO_INTERVAL.get(timeframe)
        if interval is None:
            raise ValueError(
                f"Unsupported timeframe: {timeframe}. Available: {list(self._TIMEFRAME_TO_INTERVAL.keys())}"
            )

        symbol = str(ticker).upper()
        end_dt = self._parse_end_date(end_date)
        end_ms = int(end_dt.timestamp() * 1000)
        source_limit_multiplier = 2 if timeframe == "10m" else 1
        source_target_limit = None if limit is None else max(int(limit) * source_limit_multiplier, 1)
        interval_ms = self._INTERVAL_MS[interval]

        if start_date is None:
            # Для режима "последние N свечей" якорим окно к текущему времени,
            # чтобы tail(limit) действительно отражал самый свежий рынок.
            if source_target_limit is not None:
                start_ms = max(0, end_ms - (source_target_limit * interval_ms))
            else:
                lookback_days = self._estimate_lookback_days(timeframe=timeframe, limit=limit)
                start_dt = end_dt - timedelta(days=lookback_days)
                start_ms = int(start_dt.timestamp() * 1000)
        else:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            start_ms = int(start_dt.timestamp() * 1000)

        rows: list[list] = []
        current_start_ms = start_ms
        loaded_rows = 0
        fetch_until_end = source_target_limit is not None and start_date is None

        while current_start_ms < end_ms:
            if source_target_limit is None or fetch_until_end:
                batch_limit = 1000
            else:
                remaining = source_target_limit - loaded_rows
                if remaining <= 0:
                    break
                batch_limit = min(1000, remaining)

            payload = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start_ms,
                "endTime": end_ms,
                "limit": batch_limit,
            }

            data = self._request_klines(payload, context=f"Binance candles request {symbol}")
            if not data:
                break

            rows.extend(data)
            loaded = len(data)
            loaded_rows += loaded

            last_open_ms = int(data[-1][0])
            current_start_ms = last_open_ms + interval_ms
            if loaded < batch_limit:
                break

        if not rows:
            logger.warning("No data for %s on timeframe %s", symbol, timeframe)
            return pd.DataFrame()

        df = pd.DataFrame(
            rows,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume",
                "ignore",
            ],
        )

        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.set_index("timestamp")
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        ohlcv = df[["open", "high", "low", "close", "volume"]].dropna()
        if timeframe == "10m":
            ohlcv = (
                ohlcv.resample("10min")
                .agg(
                    {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    }
                )
                .dropna()
            )

        ohlcv = self._drop_unfinished_last_candle(ohlcv, timeframe=timeframe)

        if limit is not None and int(limit) > 0:
            ohlcv = ohlcv.tail(int(limit))

        return ohlcv

    def get_current_price(self, ticker: str, board: str = "") -> dict:
        del board  # not used by Binance
        symbol = str(ticker).upper()
        response = self.client.ticker_price(symbol=symbol)
        price = float(response["price"])
        return {
            "symbol": symbol,
            "current_price": price,
            "last_price": price,
            "change_percent": None,
            "volume": None,
        }

    def get_security_spec(self, ticker: str, board: str = "") -> dict:
        del ticker, board
        # Contract spec is not applicable for spot instruments.
        return {}

    def _request_klines(self, payload: dict, context: str) -> list[list]:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self.client.klines(**payload)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                sleep_sec = self.retry_backoff_sec * (2 ** (attempt - 1))
                logger.warning(
                    "%s: attempt %s/%s failed (%s). retry in %.1fs",
                    context,
                    attempt,
                    self.max_retries,
                    exc,
                    sleep_sec,
                )
                time.sleep(sleep_sec)

        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _parse_end_date(end_date: Optional[str]) -> datetime:
        if end_date is None:
            return datetime.now(timezone.utc)
        base = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        # Include the full selected day.
        return base + timedelta(days=1)

    def _estimate_lookback_days(self, timeframe: str, limit: Optional[int]) -> int:
        if limit is None or limit <= 0:
            return 30

        interval = self._TIMEFRAME_TO_INTERVAL.get(timeframe)
        if interval is None:
            return 30

        interval_minutes_map = {
            "1m": 1.0,
            "5m": 5.0,
            "15m": 15.0,
            "30m": 30.0,
            "1h": 60.0,
            "4h": 240.0,
        }
        interval_minutes = interval_minutes_map.get(interval)
        if interval_minutes is None:
            return 30

        candles_per_day = max((24.0 * 60.0) / interval_minutes, 1.0)
        # Add safety margin for data gaps/delisted periods.
        estimated_days = int(((float(limit) / candles_per_day) * 1.2) + 1)
        return max(30, min(estimated_days, 3650))

    def _drop_unfinished_last_candle(self, df: pd.DataFrame, *, timeframe: str) -> pd.DataFrame:
        if df.empty:
            return df

        timeframe_ms = self._INTERVAL_MS.get(timeframe)
        if timeframe_ms is None:
            mapped = self._TIMEFRAME_TO_INTERVAL.get(timeframe)
            timeframe_ms = self._INTERVAL_MS.get(mapped or "")
        if timeframe_ms is None:
            return df

        now_ts = pd.Timestamp.utcnow().tz_localize(None)
        last_open = df.index[-1]
        last_close_expected = last_open + pd.Timedelta(milliseconds=int(timeframe_ms))
        if now_ts < last_close_expected:
            return df.iloc[:-1]
        return df
