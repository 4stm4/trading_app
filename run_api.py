#!/usr/bin/env python
"""
Запуск REST API торговой системы
"""

from loguru import logger
import uvicorn

from ports.api import create_app

if __name__ == "__main__":
    app = create_app()

    logger.info("=" * 80)
    logger.info("🚀 Trading System API запущен!")
    logger.info("=" * 80)
    logger.info("\n📍 Доступные endpoints:")
    logger.info("   • GET  http://localhost:5000/api/health       - Проверка работоспособности")
    logger.info("   • GET  http://localhost:5000/api/models       - Список моделей")
    logger.info("   • GET  http://localhost:5000/api/moex/instruments - Инструменты MOEX")
    logger.info("   • POST http://localhost:5000/api/signal       - Генерация сигнала")
    logger.info("   • POST http://localhost:5000/api/backtest     - Бэктест стратегии")
    logger.info("   • GET  http://localhost:5000/api/dashboard/market - Market dashboard")
    logger.info("   • POST http://localhost:5000/api/dashboard/backtest - Backtest dashboard")
    logger.info("   • POST http://localhost:5000/api/dashboard/robustness - Robustness dashboard")
    logger.info("   • POST http://localhost:5000/api/optimize     - Оптимизация моделей")
    logger.info("\n📚 Документация: см. API_GUIDE.md")
    logger.info("=" * 80 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")
