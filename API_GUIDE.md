# 🌐 Trading System REST API

REST API для торговой системы с поддержкой генерации сигналов, бэктестинга и оптимизации моделей.

## 🚀 Быстрый старт

### Запуск API

```bash
# Активируем окружение
source venv/bin/activate

# Запускаем API
python run_api.py
```

API будет доступен на `http://localhost:5000`

### Проверка работоспособности

```bash
curl http://localhost:5000/api/health
```

Ответ:
```json
{
  "status": "ok",
  "service": "Trading System API",
  "version": "1.0.0",
  "models_count": 5
}
```

## 📋 Endpoints

### 1. GET /api/health

Проверка работоспособности API.

**Пример (curl):**
```bash
curl http://localhost:5000/api/health
```

**Пример (Python):**
```python
import requests

response = requests.get('http://localhost:5000/api/health')
print(response.json())
```

**Ответ:**
```json
{
  "status": "ok",
  "service": "Trading System API",
  "version": "1.0.0",
  "models_count": 5
}
```

---

### 2. GET /api/models

Получить список доступных торговых моделей.

**Пример (curl):**
```bash
curl http://localhost:5000/api/models
```

**Пример (Python):**
```python
import requests

response = requests.get('http://localhost:5000/api/models')
models = response.json()
print(f"Доступно моделей: {models['count']}")
for name, info in models['models'].items():
    print(f"  {name}: {info['description']}")
```

**Ответ:**
```json
{
  "models": {
    "conservative": {
      "name": "conservative",
      "description": "Консервативная модель с высоким RR и строгими фильтрами",
      "min_rr": 2.5,
      "max_risk_percent": 1.0,
      "min_volume_ratio": 1.5,
      "atr_multiplier_stop": 1.5,
      "trend_required": true,
      "allow_range": false
    },
    "high_rr": { ... },
    "balanced": { ... },
    "aggressive": { ... },
    "scalp": { ... }
  },
  "count": 5
}
```

---

### 3. POST /api/signal

Генерация торгового сигнала.

**Параметры (JSON body):**

| Параметр | Тип | Обязательно | По умолчанию | Описание |
|----------|-----|-------------|--------------|----------|
| `ticker` | string | ✅ | - | Тикер инструмента |
| `deposit` | number | ✅ | - | Размер депозита |
| `timeframe` | string | ❌ | "1h" | Таймфрейм |
| `model` | string | ❌ | "balanced" | Торговая модель |
| `engine` | string | ❌ | "stock" | stock или futures |
| `market` | string | ❌ | "shares" | shares или forts |
| `board` | string | ❌ | auto | Режим торгов |

**Пример (curl):**
```bash
curl -X POST http://localhost:5000/api/signal \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "SBER",
    "deposit": 100000,
    "timeframe": "1h",
    "model": "conservative"
  }'
```

**Пример (Python):**
```python
import requests

payload = {
    "ticker": "SBER",
    "deposit": 100000,
    "timeframe": "1h",
    "model": "conservative"
}

response = requests.post(
    'http://localhost:5000/api/signal',
    json=payload
)

signal = response.json()
print(f"Сигнал: {signal['signal']['signal']}")
print(f"Вход: {signal['signal']['entry']}")
print(f"RR: {signal['signal']['rr']}")
```

**Пример для фьючерса:**
```python
payload = {
    "ticker": "CCH6",
    "deposit": 100000,
    "timeframe": "10m",
    "model": "aggressive",
    "engine": "futures",
    "market": "forts"
}

response = requests.post('http://localhost:5000/api/signal', json=payload)
```

**Ответ:**
```json
{
  "ticker": "SBER",
  "timeframe": "1h",
  "model": "conservative",
  "data_points": 476,
  "period": {
    "start": "2026-01-15 06:00:00",
    "end": "2026-02-13 23:00:00"
  },
  "signal": {
    "signal": "short",
    "entry": 296.4,
    "stop": 302.8,
    "target": 284.0,
    "rr": 1.94,
    "risk_rub": 1500.0,
    "risk_percent": 1.5,
    "position_size": 234.0,
    "structure": "downtrend",
    "phase": "pullback",
    "volume_ratio": 2.8,
    "atr": 5.1,
    "distance_ma50_pct": -2.63,
    "distance_ma200_pct": -8.37,
    "rsi": 37.7,
    "confidence": "high",
    "warnings": ["Цена на откате к MA50"],
    "model_name": "conservative"
  }
}
```

---

### 4. POST /api/backtest

Запуск бэктеста стратегии.

**Параметры:** Те же что и для `/api/signal`

**Пример (curl):**
```bash
curl -X POST http://localhost:5000/api/backtest \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "SBER",
    "deposit": 100000,
    "timeframe": "1h",
    "model": "high_rr"
  }'
```

**Пример (Python):**
```python
import requests

payload = {
    "ticker": "GAZP",
    "deposit": 200000,
    "timeframe": "1h",
    "model": "aggressive"
}

response = requests.post(
    'http://localhost:5000/api/backtest',
    json=payload
)

results = response.json()
print(f"Всего сделок: {results['results']['total_trades']}")
print(f"Winrate: {results['results']['winrate']}%")
print(f"Profit Factor: {results['results']['profit_factor']}")
print(f"Expectancy: {results['results']['expectancy']}")
```

**Ответ:**
```json
{
  "ticker": "SBER",
  "timeframe": "1h",
  "model": "high_rr",
  "data_points": 476,
  "period": {
    "start": "2026-01-15 06:00:00",
    "end": "2026-02-13 23:00:00"
  },
  "results": {
    "model_name": "high_rr",
    "total_trades": 42,
    "winning_trades": 26,
    "losing_trades": 16,
    "winrate": 61.9,
    "avg_win": 892.34,
    "avg_loss": 456.12,
    "best_trade": 2145.67,
    "worst_trade": -867.23,
    "expectancy": 362.73,
    "max_drawdown": 4234.56,
    "max_drawdown_percent": 4.23,
    "total_profit": 15234.50,
    "final_balance": 115234.50,
    "profit_factor": 1.96,
    "return_pct": 15.23,
    "sharpe_ratio": 1.45,
    "avg_trade_duration": 8
  }
}
```

---

### 5. POST /api/optimize

Сравнение всех моделей (оптимизация).

**Параметры:**

| Параметр | Тип | Обязательно | По умолчанию | Описание |
|----------|-----|-------------|--------------|----------|
| `ticker` | string | ✅ | - | Тикер инструмента |
| `deposit` | number | ✅ | - | Размер депозита |
| `timeframe` | string | ❌ | "1h" | Таймфрейм |
| `engine` | string | ❌ | "stock" | stock или futures |
| `market` | string | ❌ | "shares" | shares или forts |

**Пример (curl):**
```bash
curl -X POST http://localhost:5000/api/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "SBER",
    "deposit": 100000,
    "timeframe": "1h"
  }'
```

**Пример (Python):**
```python
import requests
import pandas as pd

payload = {
    "ticker": "SBER",
    "deposit": 100000,
    "timeframe": "1h"
}

response = requests.post(
    'http://localhost:5000/api/optimize',
    json=payload
)

data = response.json()

print(f"Протестировано моделей: {data['models_tested']}")
print(f"\nЛучшая модель: {data['best_model']['name']}")
print(f"  Expectancy: {data['best_model']['expectancy']}")
print(f"  Winrate: {data['best_model']['winrate']}%")

# Создаем таблицу результатов
results_df = pd.DataFrame(data['results'])
print("\nСравнение всех моделей:")
print(results_df[['model_name', 'total_trades', 'winrate', 'expectancy', 'profit_factor']])
```

**Ответ:**
```json
{
  "ticker": "SBER",
  "timeframe": "1h",
  "data_points": 476,
  "period": {
    "start": "2026-01-15 06:00:00",
    "end": "2026-02-13 23:00:00"
  },
  "models_tested": 5,
  "results": [
    {
      "model_name": "conservative",
      "total_trades": 12,
      "winrate": 58.3,
      "expectancy": 425.50,
      ...
    },
    {
      "model_name": "high_rr",
      "total_trades": 26,
      "winrate": 53.8,
      "expectancy": 312.80,
      ...
    },
    ...
  ],
  "best_model": {
    "name": "conservative",
    "expectancy": 425.50,
    "winrate": 58.3,
    "profit_factor": 2.15
  }
}
```

---

## 🐍 Python Client Example

Полный пример клиента на Python:

```python
import requests
from typing import Dict, Optional

class TradingSystemAPI:
    """Клиент для Trading System API"""

    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url

    def health_check(self) -> Dict:
        """Проверка работоспособности"""
        response = requests.get(f"{self.base_url}/api/health")
        return response.json()

    def get_models(self) -> Dict:
        """Получить список моделей"""
        response = requests.get(f"{self.base_url}/api/models")
        return response.json()

    def get_signal(
        self,
        ticker: str,
        deposit: float,
        timeframe: str = "1h",
        model: str = "balanced",
        engine: str = "stock",
        market: str = "shares"
    ) -> Dict:
        """Получить торговый сигнал"""
        payload = {
            "ticker": ticker,
            "deposit": deposit,
            "timeframe": timeframe,
            "model": model,
            "engine": engine,
            "market": market
        }
        response = requests.post(
            f"{self.base_url}/api/signal",
            json=payload
        )
        return response.json()

    def run_backtest(
        self,
        ticker: str,
        deposit: float,
        timeframe: str = "1h",
        model: str = "balanced",
        engine: str = "stock",
        market: str = "shares"
    ) -> Dict:
        """Запустить бэктест"""
        payload = {
            "ticker": ticker,
            "deposit": deposit,
            "timeframe": timeframe,
            "model": model,
            "engine": engine,
            "market": market
        }
        response = requests.post(
            f"{self.base_url}/api/backtest",
            json=payload
        )
        return response.json()

    def optimize(
        self,
        ticker: str,
        deposit: float,
        timeframe: str = "1h",
        engine: str = "stock",
        market: str = "shares"
    ) -> Dict:
        """Оптимизировать модели"""
        payload = {
            "ticker": ticker,
            "deposit": deposit,
            "timeframe": timeframe,
            "engine": engine,
            "market": market
        }
        response = requests.post(
            f"{self.base_url}/api/optimize",
            json=payload
        )
        return response.json()


# Использование
if __name__ == "__main__":
    api = TradingSystemAPI()

    # Проверка
    print("API Status:", api.health_check()['status'])

    # Получение сигнала
    signal = api.get_signal(
        ticker="SBER",
        deposit=100000,
        model="conservative"
    )
    print(f"\nСигнал: {signal['signal']['signal']}")
    print(f"Уверенность: {signal['signal']['confidence']}")

    # Бэктест
    backtest = api.run_backtest(
        ticker="GAZP",
        deposit=200000,
        model="aggressive"
    )
    print(f"\nБэктест:")
    print(f"  Сделок: {backtest['results']['total_trades']}")
    print(f"  Winrate: {backtest['results']['winrate']}%")

    # Оптимизация
    optimization = api.optimize(
        ticker="LKOH",
        deposit=300000
    )
    print(f"\nЛучшая модель: {optimization['best_model']['name']}")
```

---

## ⚠️ Обработка ошибок

### Коды ответов

| Код | Описание |
|-----|----------|
| 200 | Успешно |
| 400 | Неверные параметры |
| 404 | Данные не найдены |
| 500 | Внутренняя ошибка |

### Формат ошибки

```json
{
  "error": "Bad Request",
  "message": "Missing required field: ticker"
}
```

### Обработка ошибок (Python)

```python
import requests

try:
    response = requests.post(
        'http://localhost:5000/api/signal',
        json={"deposit": 100000}  # Забыли ticker
    )
    response.raise_for_status()
    data = response.json()
except requests.exceptions.HTTPError as e:
    print(f"HTTP Error: {e}")
    print(f"Response: {e.response.json()}")
except Exception as e:
    print(f"Error: {e}")
```

---

## 🔒 Безопасность

**Важно:** Данный API не имеет аутентификации и предназначен для локального использования или защищенной сети.

Для production использования рекомендуется:
- Добавить JWT аутентификацию
- Использовать HTTPS
- Добавить rate limiting
- Настроить CORS правильно

---

## 🚀 Развертывание

### Production запуск с Gunicorn

```bash
pip install gunicorn

gunicorn -w 4 -b 0.0.0.0:5000 ports.api.app:create_app()
```

### Docker (пример)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "ports.api.app:create_app()"]
```

---

## 📊 Производительность

- Генерация сигнала: ~1-3 секунды
- Бэктест: ~3-5 секунд
- Оптимизация (5 моделей): ~15-20 секунд

Время зависит от:
- Таймфрейма (минутки медленнее)
- Объема данных
- Сложности модели

---

## 🆘 Troubleshooting

**Проблема:** `Connection refused`
```bash
# Проверьте, что API запущен
curl http://localhost:5000/api/health
```

**Проблема:** `No data for ticker`
```json
{
  "error": "No data for INVALID"
}
```
Решение: Проверьте правильность тикера и параметров engine/market/board

**Проблема:** Медленный ответ
- Используйте более крупные таймфреймы (1h, 1d вместо 1m)
- Уменьшите период данных через start_date/end_date

---

## 📝 Changelog

### v1.0.0 (2026-02-14)
- Первый релиз
- Endpoints: health, models, signal, backtest, optimize
- Поддержка MOEX (акции + фьючерсы)
- 5 торговых моделей

---

**Документация:** см. также README.md и ARCHITECTURE.md
**Поддержка:** https://github.com/4stm4/trading_app
