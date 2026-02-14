import pandas as pd
import requests
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class MOEXAdapter:
    BASE_URL = "https://iss.moex.com/iss"

    def __init__(self, engine: str = "stock", market: str = "shares"):
        self.engine = engine
        self.market = market

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
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        url = f"{self.BASE_URL}/engines/{self.engine}/markets/{self.market}/boards/{board}/securities/{ticker}/candles.json"
        params = {
            "from": start_date,
            "till": end_date,
            "interval": moex_interval,
            "sort": "TRADEDATE",
            "limit": 10000,
        }

        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            if "candles" not in data or not data["candles"]["data"]:
                logger.warning(f"Нет данных для {ticker} на таймфрейме {timeframe}")
                return pd.DataFrame()

            columns = data["candles"]["columns"]
            rows = data["candles"]["data"]

            df = pd.DataFrame(rows, columns=columns)

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

    def get_current_price(self, ticker: str, board: str = "TQBR") -> Dict:
        url = f"{self.BASE_URL}/engines/{self.engine}/markets/{self.market}/boards/{board}/securities/{ticker}.json"
        params = {"securities.columns": "SECID,OPEN,LOW,HIGH,LAST,PREVPRICE,VOLTODAY"}

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            columns = data["securities"]["columns"]
            rows = data["securities"]["data"]

            if not rows:
                raise ValueError(f"Нет данных для {ticker}")

            quote = dict(zip(columns, rows[0]))
            return {
                "ticker": quote.get("SECID"),
                "open": quote.get("OPEN"),
                "high": quote.get("HIGH"),
                "low": quote.get("LOW"),
                "last": quote.get("LAST"),
                "prev_close": quote.get("PREVPRICE"),
                "volume": quote.get("VOLTODAY"),
                "change_pct": (
                    (quote.get("LAST") - quote.get("PREVPRICE")) / quote.get("PREVPRICE") * 100
                )
                if quote.get("PREVPRICE")
                else None,
            }

        except Exception as e:
            logger.error(f"Ошибка получения котировки: {e}")
            raise


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
