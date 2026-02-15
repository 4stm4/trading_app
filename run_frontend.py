#!/usr/bin/env python
"""
Запуск веб-интерфейса торговой системы
"""

from loguru import logger

from frontend.app import app

if __name__ == '__main__':
    logger.info("=" * 80)
    logger.info("🎨 Trading System Dashboard запущен!")
    logger.info("=" * 80)
    logger.info("\n🌐 Откройте в браузере:")
    logger.info("   http://localhost:8050")
    logger.info("\n📊 Возможности:")
    logger.info("   • Интерактивные свечные графики")
    logger.info("   • Отображение индикаторов (MA50, MA200, RSI)")
    logger.info("   • Генерация торговых сигналов")
    logger.info("   • Визуализация точек входа/выхода")
    logger.info("   • Бэктестинг с детальной статистикой")
    logger.info("=" * 80 + "\n")

    app.run(debug=False, host='0.0.0.0', port=8050)
