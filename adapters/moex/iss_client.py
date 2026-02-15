import pandas as pd
import requests
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import logging
import time
import math

logger = logging.getLogger(__name__)


def _to_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


class MOEXAdapter:
    BASE_URL = "https://iss.moex.com/iss"

    def __init__(
        self,
        engine: str = "stock",
        market: str = "shares",
        max_retries: int = 4,
        retry_backoff_sec: float = 1.5,
        candles_timeout: tuple[float, float] = (10.0, 40.0),
        quote_timeout: tuple[float, float] = (5.0, 20.0),
    ):
        self.engine = engine
        self.market = market
        self.max_retries = max(1, int(max_retries))
        self.retry_backoff_sec = max(0.1, float(retry_backoff_sec))
        self.candles_timeout = candles_timeout
        self.quote_timeout = quote_timeout

    def _request_json(
        self,
        url: str,
        params: dict,
        timeout: tuple[float, float],
        context: str,
    ) -> dict:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(url, params=params, timeout=timeout)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                sleep_sec = self.retry_backoff_sec * (2 ** (attempt - 1))
                logger.warning(
                    f"{context}: attempt {attempt}/{self.max_retries} failed ({exc}). retry in {sleep_sec:.1f}s"
                )
                time.sleep(sleep_sec)

        assert last_exc is not None
        raise last_exc

    def get_securities(self, board: str = "TQBR") -> pd.DataFrame:
        url = f"{self.BASE_URL}/engines/{self.engine}/markets/{self.market}/boards/{board}/securities.json"
        params = {"securities.columns": "SECID,NAME,SHORTNAME,LOTSIZE,PREVPRICE,CURRENCY"}

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            columns = data["securities"]["columns"]
            rows = data["securities"]["data"]

            df = pd.DataFrame(rows, columns=columns)
            return df

        except Exception as e:
            logger.error(f"Ошибка получения списка инструментов: {e}")
            raise

    def get_candles(
        self,
        ticker: str,
        timeframe: str = "1h",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        board: str = "TQBR",
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        tf_map = {
            "1m": "1",
            "5m": "5",
            "10m": "10",
            "15m": "15",
            "30m": "30",
            "1h": "60",
            "4h": "240",
            "1d": "D",
            "1w": "W",
            "1M": "M",
        }

        if timeframe not in tf_map:
            raise ValueError(
                f"Неподдерживаемый таймфрейм: {timeframe}. Доступные: {list(tf_map.keys())}"
            )

        moex_interval = tf_map[timeframe]

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            lookback_days = self._estimate_lookback_days(timeframe=timeframe, limit=limit)
            start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        url = f"{self.BASE_URL}/engines/{self.engine}/markets/{self.market}/boards/{board}/securities/{ticker}/candles.json"
        # MOEX ISS candles endpoint фактически ограничивает страницу ~500 строк.
        # Используем 500, чтобы корректно пройти весь диапазон по start-offset.
        chunk_size = 500
        start = 0
        loaded_rows = 0
        frames = []

        try:
            while True:
                if limit is not None and loaded_rows >= limit:
                    break

                if limit is None:
                    page_limit = chunk_size
                else:
                    page_limit = min(chunk_size, max(limit - loaded_rows, 0))
                    if page_limit <= 0:
                        break

                params = {
                    "from": start_date,
                    "till": end_date,
                    "interval": moex_interval,
                    "sort": "TRADEDATE",
                    "limit": page_limit,
                    "start": start,
                }

                data = self._request_json(
                    url=url,
                    params=params,
                    timeout=self.candles_timeout,
                    context=f"MOEX candles request {ticker}"
                )

                if "candles" not in data or not data["candles"]["data"]:
                    break

                columns = data["candles"]["columns"]
                rows = data["candles"]["data"]
                page_df = pd.DataFrame(rows, columns=columns)
                if page_df.empty:
                    break

                frames.append(page_df)
                loaded = len(page_df)
                loaded_rows += loaded
                start += loaded

                if loaded < page_limit:
                    break

            if not frames:
                logger.warning(f"Нет данных для {ticker} на таймфрейме {timeframe}")
                return pd.DataFrame()

            df = pd.concat(frames, ignore_index=True)

            df = df.rename(
                columns={
                    "open": "open",
                    "close": "close",
                    "high": "high",
                    "low": "low",
                    "volume": "volume",
                    "value": "value",  # оборот в рублях
                    "begin": "timestamp",
                }
            )

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="last")]

            if limit is not None and limit > 0:
                df = df.tail(limit)

            required_cols = ["open", "high", "low", "close", "volume"]
            if not all(col in df.columns for col in required_cols):
                missing = set(required_cols) - set(df.columns)
                raise ValueError(f"Отсутствуют колонки: {missing}")

            return df[required_cols]

        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка запроса к MOEX API: {e}")
            raise
        except Exception as e:
            logger.error(f"Ошибка обработки данных MOEX: {e}")
            raise

    def _estimate_lookback_days(self, timeframe: str, limit: Optional[int]) -> int:
        """
        Оценка глубины истории в днях для загрузки нужного числа свечей.
        Если limit не задан — сохраняем прежнее поведение (30 дней).
        """
        if limit is None or limit <= 0:
            return 30

        # Оцениваем по торговым часам, а затем добавляем поправку на выходные/праздники.
        trading_hours_per_day = 14.0 if self.engine == "futures" else 8.0
        interval_minutes_map = {
            "1m": 1.0,
            "5m": 5.0,
            "10m": 10.0,
            "15m": 15.0,
            "30m": 30.0,
            "1h": 60.0,
            "4h": 240.0,
        }
        interval_minutes = interval_minutes_map.get(timeframe)

        if interval_minutes is None:
            # Для D/W/M не пытаемся агрессивно расширять диапазон.
            return 30

        candles_per_trading_day = max((trading_hours_per_day * 60.0) / interval_minutes, 1.0)
        # 7/5 для календарных дней + буфер на пропуски/праздники.
        calendar_factor = 1.4 * (7.0 / 5.0)
        estimated_days = math.ceil((float(limit) / candles_per_trading_day) * calendar_factor)
        return max(30, min(int(estimated_days), 3650))

    def get_current_price(self, ticker: str, board: str = "TQBR") -> Dict:
        url = f"{self.BASE_URL}/engines/{self.engine}/markets/{self.market}/boards/{board}/securities/{ticker}.json"
        params = {"securities.columns": "SECID,OPEN,LOW,HIGH,LAST,PREVPRICE,VOLTODAY"}

        try:
            data = self._request_json(
                url=url,
                params=params,
                timeout=self.quote_timeout,
                context=f"MOEX quote request {ticker}"
            )

            columns = data["securities"]["columns"]
            rows = data["securities"]["data"]

            if not rows:
                raise ValueError(f"Нет данных для {ticker}")

            quote = dict(zip(columns, rows[0]))
            open_price = _to_float(quote.get("OPEN"))
            high_price = _to_float(quote.get("HIGH"))
            low_price = _to_float(quote.get("LOW"))
            last_price = _to_float(quote.get("LAST"))
            prev_close = _to_float(quote.get("PREVPRICE"))
            volume = _to_float(quote.get("VOLTODAY"))

            # На части инструментов LAST может быть None (например, вне активной сессии).
            if last_price is None:
                for fallback in (open_price, prev_close, high_price, low_price):
                    if fallback is not None:
                        last_price = fallback
                        break

            change_pct = None
            if prev_close not in (None, 0) and last_price is not None:
                change_pct = (last_price - prev_close) / prev_close * 100

            return {
                "ticker": quote.get("SECID"),
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "last": last_price,
                "prev_close": prev_close,
                "volume": volume,
                "change_pct": change_pct,
            }

        except Exception as e:
            logger.error(f"Ошибка получения котировки: {e}")
            raise

    def get_security_spec(self, ticker: str, board: str = "TQBR") -> Dict:
        """
        Возвращает спецификацию инструмента для нормализации цен.
        """
        url = (
            f"{self.BASE_URL}/engines/{self.engine}/markets/{self.market}/boards/"
            f"{board}/securities/{ticker}.json"
        )
        params = {
            "securities.columns": (
                "SECID,LOTSIZE,MINSTEP,STEPPRICE,FACEVALUE,PREVPRICE,INITIALMARGIN"
            )
        }
        try:
            data = self._request_json(
                url=url,
                params=params,
                timeout=self.quote_timeout,
                context=f"MOEX security spec request {ticker}",
            )
            rows = data.get("securities", {}).get("data", [])
            columns = data.get("securities", {}).get("columns", [])
            if not rows:
                return {
                    "ticker": ticker,
                    "price_step": None,
                    "contract_multiplier": 1.0,
                    "lot_size": 1.0,
                    "initial_margin": None,
                }

            raw = dict(zip(columns, rows[0]))
            lot_size = _to_float(raw.get("LOTSIZE")) or 1.0
            min_step = _to_float(raw.get("MINSTEP"))
            step_price = _to_float(raw.get("STEPPRICE"))
            face_value = _to_float(raw.get("FACEVALUE"))
            initial_margin = _to_float(raw.get("INITIALMARGIN"))

            contract_multiplier = 1.0
            if step_price is not None and min_step not in (None, 0):
                contract_multiplier = abs(step_price / min_step)
            elif face_value is not None and face_value > 0:
                contract_multiplier = face_value
            elif lot_size > 0:
                contract_multiplier = lot_size

            return {
                "ticker": raw.get("SECID", ticker),
                "price_step": min_step,
                "contract_multiplier": contract_multiplier if contract_multiplier > 0 else 1.0,
                "lot_size": lot_size,
                "initial_margin": initial_margin,
            }
        except Exception as e:
            logger.warning(f"Ошибка получения спецификации {ticker}: {e}")
            return {
                "ticker": ticker,
                "price_step": None,
                "contract_multiplier": 1.0,
                "lot_size": 1.0,
                "initial_margin": None,
            }


def fetch_multiple_tickers(
    tickers: List[str], timeframe: str = "1h", adapter: Optional[MOEXAdapter] = None
) -> Dict[str, pd.DataFrame]:
    if adapter is None:
        adapter = MOEXAdapter()

    results = {}
    for ticker in tickers:
        try:
            df = adapter.get_candles(ticker, timeframe=timeframe)
            if not df.empty:
                results[ticker] = df
                logger.info(f"Загружено {len(df)} свечей для {ticker}")
            else:
                logger.warning(f"Пустой датасет для {ticker}")
        except Exception as e:
            logger.error(f"Ошибка загрузки {ticker}: {e}")
            continue

    return results
