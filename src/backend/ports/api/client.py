"""
Python клиент для Trading System API
"""

import requests
from typing import Dict, Optional

from loguru import logger


class TradingSystemClient:
    """
    Клиент для взаимодействия с Trading System API

    Example:
        >>> client = TradingSystemClient('http://localhost:5000')
        >>> signal = client.get_signal('SBER', 100000, model='conservative')
        >>> print(signal['signal']['signal'])
    """

    def __init__(self, base_url: str = "http://localhost:5000"):
        """
        Args:
            base_url: Базовый URL API (по умолчанию http://localhost:5000)
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()

    def health_check(self) -> Dict:
        """
        Проверка работоспособности API

        Returns:
            Dict с информацией о статусе API

        Example:
            >>> client.health_check()
            {'status': 'ok', 'service': 'Trading System API', ...}
        """
        response = self.session.get(f"{self.base_url}/api/health")
        response.raise_for_status()
        return response.json()

    def get_models(self) -> Dict:
        """
        Получить список доступных торговых моделей

        Returns:
            Dict с информацией о всех моделях

        Example:
            >>> models = client.get_models()
            >>> print(models['count'])
            5
        """
        response = self.session.get(f"{self.base_url}/api/models")
        response.raise_for_status()
        return response.json()

    def get_signal(
        self,
        ticker: str,
        deposit: float,
        timeframe: str = "1h",
        model: str = "balanced",
        exchange: str = "moex",
        engine: str = "stock",
        market: str = "shares",
        board: Optional[str] = None
    ) -> Dict:
        """
        Получить торговый сигнал

        Args:
            ticker: Тикер инструмента (например, 'SBER', 'CCH6')
            deposit: Размер депозита в рублях
            timeframe: Таймфрейм ('1m', '5m', '10m', '1h', '1d' и т.д.)
            model: Торговая модель ('conservative', 'high_rr', 'balanced', 'aggressive', 'scalp')
            exchange: Биржа/источник данных ('moex', 'binance')
            engine: Движок ('stock' для акций, 'futures' для фьючерсов)
            market: Рынок ('shares' для акций, 'forts' для фьючерсов)
            board: Режим торгов (опционально, определяется автоматически)

        Returns:
            Dict с торговым сигналом и метаданными

        Example:
            >>> signal = client.get_signal('SBER', 100000, model='conservative')
            >>> print(signal['signal']['signal'])  # 'long', 'short', или 'none'
        """
        payload = {
            "ticker": ticker,
            "deposit": deposit,
            "timeframe": timeframe,
            "model": model,
            "exchange": exchange,
            "engine": engine,
            "market": market
        }
        if board:
            payload["board"] = board

        response = self.session.post(
            f"{self.base_url}/api/signal",
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def run_backtest(
        self,
        ticker: str,
        deposit: float,
        timeframe: str = "1h",
        model: str = "balanced",
        exchange: str = "moex",
        engine: str = "stock",
        market: str = "shares",
        board: Optional[str] = None
    ) -> Dict:
        """
        Запустить бэктест стратегии

        Args:
            ticker: Тикер инструмента
            deposit: Размер депозита в рублях
            timeframe: Таймфрейм
            model: Торговая модель
            exchange: Биржа/источник данных ('moex', 'binance')
            engine: Движок ('stock' или 'futures')
            market: Рынок ('shares' или 'forts')
            board: Режим торгов (опционально)

        Returns:
            Dict с результатами бэктеста

        Example:
            >>> backtest = client.run_backtest('SBER', 100000, model='aggressive')
            >>> print(backtest['results']['winrate'])
            55.3
        """
        payload = {
            "ticker": ticker,
            "deposit": deposit,
            "timeframe": timeframe,
            "model": model,
            "exchange": exchange,
            "engine": engine,
            "market": market
        }
        if board:
            payload["board"] = board

        response = self.session.post(
            f"{self.base_url}/api/backtest",
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def get_dashboard_market(
        self,
        ticker: str,
        timeframe: str = "1h",
        model: str = "balanced",
        deposit: float = 100000.0,
        exchange: str = "moex",
        engine: str = "stock",
        market: str = "shares",
        board: Optional[str] = None,
        limit: int = 300,
    ) -> Dict:
        params = {
            "ticker": ticker,
            "exchange": exchange,
            "timeframe": timeframe,
            "engine": engine,
            "market": market,
            "model": model,
            "deposit": deposit,
            "limit": limit,
        }
        if board:
            params["board"] = board

        response = self.session.get(
            f"{self.base_url}/api/dashboard/market",
            params=params,
        )
        response.raise_for_status()
        return response.json()

    def get_dashboard_backtest(
        self,
        ticker: str,
        deposit: float,
        timeframe: str = "1h",
        model: str = "balanced",
        exchange: str = "moex",
        engine: str = "stock",
        market: str = "shares",
        board: Optional[str] = None,
        limit: int = 1200,
    ) -> Dict:
        payload = {
            "ticker": ticker,
            "deposit": deposit,
            "timeframe": timeframe,
            "model": model,
            "exchange": exchange,
            "engine": engine,
            "market": market,
            "limit": limit,
        }
        if board:
            payload["board"] = board

        response = self.session.post(
            f"{self.base_url}/api/dashboard/backtest",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    def get_dashboard_robustness(
        self,
        ticker: str,
        deposit: float,
        timeframe: str = "1h",
        model: str = "balanced",
        exchange: str = "moex",
        engine: str = "stock",
        market: str = "shares",
        board: Optional[str] = None,
        limit: int = 1500,
        monte_carlo_simulations: int = 300,
    ) -> Dict:
        payload = {
            "ticker": ticker,
            "deposit": deposit,
            "timeframe": timeframe,
            "model": model,
            "exchange": exchange,
            "engine": engine,
            "market": market,
            "limit": limit,
            "monte_carlo_simulations": monte_carlo_simulations,
        }
        if board:
            payload["board"] = board

        response = self.session.post(
            f"{self.base_url}/api/dashboard/robustness",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    def optimize(
        self,
        ticker: str,
        deposit: float,
        timeframe: str = "1h",
        exchange: str = "moex",
        engine: str = "stock",
        market: str = "shares",
        board: Optional[str] = None
    ) -> Dict:
        """
        Сравнить все модели и найти лучшую

        Args:
            ticker: Тикер инструмента
            deposit: Размер депозита в рублях
            timeframe: Таймфрейм
            exchange: Биржа/источник данных ('moex', 'binance')
            engine: Движок ('stock' или 'futures')
            market: Рынок ('shares' или 'forts')
            board: Режим торгов (опционально)

        Returns:
            Dict с результатами всех моделей и информацией о лучшей

        Example:
            >>> optimization = client.optimize('SBER', 100000)
            >>> print(optimization['best_model']['name'])
            'conservative'
            >>> print(optimization['best_model']['expectancy'])
            425.50
        """
        payload = {
            "ticker": ticker,
            "deposit": deposit,
            "timeframe": timeframe,
            "exchange": exchange,
            "engine": engine,
            "market": market
        }
        if board:
            payload["board"] = board

        response = self.session.post(
            f"{self.base_url}/api/optimize",
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def close(self):
        """Закрыть сессию"""
        self.session.close()

    def __enter__(self):
        """Context manager support"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support"""
        self.close()


# Примеры использования
if __name__ == "__main__":
    # Создаем клиента
    client = TradingSystemClient()

    # Проверяем работоспособность
    health = client.health_check()
    logger.info(f"✅ API Status: {health['status']}")
    logger.info(f"   Models available: {health['models_count']}")

    # Получаем список моделей
    models = client.get_models()
    logger.info(f"\n📊 Доступные модели:")
    for name, info in models['models'].items():
        logger.info(f"   {name:15s} - RR {info['min_rr']}, Risk {info['max_risk_percent']}%")

    # Получаем сигнал для Сбербанка
    logger.info(f"\n🎯 Генерация сигнала для SBER...")
    signal = client.get_signal(
        ticker="SBER",
        deposit=100000,
        timeframe="1h",
        model="balanced"
    )

    logger.info(f"   Направление: {signal['signal']['signal']}")
    logger.info(f"   Уверенность: {signal['signal']['confidence']}")
    if signal['signal']['signal'] != 'none':
        logger.info(f"   Вход: {signal['signal']['entry']}")
        logger.info(f"   RR: {signal['signal']['rr']}")

    # Запускаем бэктест
    logger.info(f"\n📈 Запуск бэктеста...")
    backtest = client.run_backtest(
        ticker="GAZP",
        deposit=200000,
        model="aggressive"
    )

    results = backtest['results']
    logger.info(f"   Всего сделок: {results['total_trades']}")
    logger.info(f"   Winrate: {results['winrate']}%")
    logger.info(f"   Profit Factor: {results['profit_factor']}")
    logger.info(f"   Expectancy: {results['expectancy']}")

    # Оптимизация
    logger.info(f"\n🔧 Оптимизация моделей...")
    optimization = client.optimize(
        ticker="LKOH",
        deposit=300000,
        timeframe="1h"
    )

    logger.info(f"   Протестировано моделей: {optimization['models_tested']}")
    logger.info(f"   Лучшая модель: {optimization['best_model']['name']}")
    logger.info(f"   Expectancy: {optimization['best_model']['expectancy']}")

    logger.info("\n✅ Все примеры выполнены!")
