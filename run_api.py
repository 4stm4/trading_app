#!/usr/bin/env python
"""
Запуск REST API торговой системы
"""

from ports.api import create_app

if __name__ == '__main__':
    app = create_app()

    print("=" * 80)
    print("🚀 Trading System API запущен!")
    print("=" * 80)
    print("\n📍 Доступные endpoints:")
    print("   • GET  http://localhost:5000/api/health       - Проверка работоспособности")
    print("   • GET  http://localhost:5000/api/models       - Список моделей")
    print("   • POST http://localhost:5000/api/signal       - Генерация сигнала")
    print("   • POST http://localhost:5000/api/backtest     - Бэктест стратегии")
    print("   • POST http://localhost:5000/api/optimize     - Оптимизация моделей")
    print("\n📚 Документация: см. API_GUIDE.md")
    print("=" * 80 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=False)
