# 🚀 Быстрый старт - Trading System

## 1️⃣ CLI (Командная строка)

### Установка
```bash
cd /path/to/trading_app
source venv/bin/activate
```

### Базовые команды

```bash
# Анализ акции
python -m ports.cli.trading_cli SBER --deposit 100000

# С выбором модели
python -m ports.cli.trading_cli SBER -d 100000 --model conservative

# Фьючерсы
python -m ports.cli.trading_cli CCH6 -d 100000 -e futures -m forts

# С бэктестом
python -m ports.cli.trading_cli SBER -d 100000 --backtest

# Оптимизация (сравнить все модели)
python -m ports.cli.trading_cli SBER -d 100000 --optimize

# Список моделей
python -m ports.cli.trading_cli --list-models
```

### Конфиг стратегии

- По умолчанию CLI использует `strict.yaml` из корня проекта (если файл существует).
- Явный выбор файла: `python -m ports.cli.trading_cli SBER -d 100000 --config strict.yaml`.
- Если `strict.yaml` отсутствует, используются встроенные безопасные fallback defaults.

## 2️⃣ REST API

### Запуск сервера
```bash
source venv/bin/activate
python run_api.py
```

API доступен на `http://localhost:5000`

### Примеры запросов

**Проверка:**
```bash
curl http://localhost:5000/api/health
```

**Список моделей:**
```bash
curl http://localhost:5000/api/models
```

**Получить сигнал:**
```bash
curl -X POST http://localhost:5000/api/signal \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "SBER",
    "deposit": 100000,
    "model": "conservative"
  }'
```

**Бэктест:**
```bash
curl -X POST http://localhost:5000/api/backtest \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "GAZP",
    "deposit": 200000,
    "model": "aggressive"
  }'
```

### Python клиент

```python
from ports.api.client import TradingSystemClient

client = TradingSystemClient()

# Получить сигнал
signal = client.get_signal('SBER', 100000, model='conservative')
print(signal['signal']['signal'])

# Запустить бэктест
backtest = client.run_backtest('GAZP', 200000, model='aggressive')
print(f"Winrate: {backtest['results']['winrate']}%")

# Оптимизация
optimization = client.optimize('LKOH', 300000)
print(f"Лучшая модель: {optimization['best_model']['name']}")
```

## 3️⃣ Python библиотека

```python
from services.strategy_engine import get_model, generate_signal
from adapters.moex import load_data_with_indicators

# Загрузить данные
df, _ = load_data_with_indicators('SBER', '1h')

# Получить модель
model = get_model('conservative')

# Сгенерировать сигнал
signal = generate_signal(df, deposit=100000, model=model)

print(f"Сигнал: {signal.signal}")
print(f"Вход: {signal.entry}")
print(f"RR: {signal.rr}")
```

## 📚 Документация

- **CLI**: `adapters/moex/README.md`
- **API**: `API_GUIDE.md`
- **Архитектура**: `ARCHITECTURE.md`
- **Strategy Engine**: `services/strategy_engine/README.md`

## 🎯 Торговые модели

| Модель | RR | Риск | Описание |
|--------|-----|------|----------|
| `conservative` | 2.5 | 1.0% | Высокий RR, строгие фильтры |
| `high_rr` | 2.0 | 1.5% | Умеренные фильтры |
| `balanced` | 2.0 | 1.5% | Сбалансированная (по умолчанию) |
| `aggressive` | 1.5 | 2.0% | Мягкие фильтры, больше сделок |
| `scalp` | 1.2 | 1.5% | Скальпинг, работает в рейндже |

## ⚡ Примеры

### Консервативная торговля
```bash
python -m ports.cli.trading_cli SBER -d 500000 --model conservative --backtest
```

### Агрессивный скальпинг
```bash
python -m ports.cli.trading_cli GAZP -d 100000 --model scalp -t 5m
```

### Найти лучшую модель
```bash
python -m ports.cli.trading_cli LKOH -d 300000 --optimize
```

### API оптимизация
```python
from ports.api.client import TradingSystemClient

client = TradingSystemClient()
result = client.optimize('SBER', 100000, timeframe='1h')

# Вывод результатов
for model_result in result['results']:
    print(f"{model_result['model_name']:15s} "
          f"Trades: {model_result['total_trades']:3d} "
          f"WR: {model_result['winrate']:.1f}% "
          f"Exp: {model_result['expectancy']:.2f}")

print(f"\n🏆 Лучшая: {result['best_model']['name']}")
```

## 🆘 Помощь

```bash
# CLI помощь
python -m ports.cli.trading_cli --help

# Список моделей
python -m ports.cli.trading_cli --list-models

# Сравнение моделей
python -m ports.cli.trading_cli --compare-models

# API примеры
python -m ports.api.client
```

---

**Готово к использованию!** 🎉
