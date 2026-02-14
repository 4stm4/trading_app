# 🎯 Strategy Engine - Универсальный движок торговых стратегий

Модульный движок для генерации торговых сигналов, управления рисками и бэктестинга. Может использоваться со всеми адаптерами: MOEX, Binance, Alfa Invest и другими.

## 🏗️ Архитектура

```
services/strategy_engine/
├── __init__.py       # Публичный API
├── core.py          # Ядро: ATR, структура рынка, объемы
├── signals.py       # Генерация торговых сигналов
├── models.py        # 5 торговых моделей
├── risk.py          # Риск-менеджмент и расчеты
└── backtest.py      # Бэктестинг стратегий
```

## ✨ Преимущества

- ✅ **Универсальность** - работает с любым источником данных (MOEX, Binance, и т.д.)
- ✅ **Модульность** - легко расширяется и тестируется
- ✅ **Независимость** - не зависит от конкретного адаптера
- ✅ **Переиспользование** - один движок для всех инструментов
- ✅ **Централизация** - вся торговая логика в одном месте

## 📦 Использование

### Базовый пример

```python
from services.strategy_engine import get_model, generate_signal
import pandas as pd

# Получаем модель
model = get_model('conservative')

# DataFrame должен содержать: open, high, low, close, volume, ma50, ma200, rsi
df = load_your_data()  # Любой источник: MOEX, Binance, etc.

# Генерируем сигнал
signal = generate_signal(
    df=df,
    deposit=100000,
    model=model
)

print(f"Сигнал: {signal.signal}")
print(f"Вход: {signal.entry}")
print(f"Стоп: {signal.stop}")
print(f"RR: {signal.rr}")
```

### С MOEX адаптером

```python
from adapters.moex import load_data_with_indicators, MOEXAdapter
from services.strategy_engine import get_model, generate_signal

# Загружаем данные из MOEX
adapter = MOEXAdapter(engine='futures', market='forts')
df, _ = load_data_with_indicators(
    ticker='CCH6',
    timeframe='10m',
    adapter=adapter
)

# Генерируем сигнал
model = get_model('high_rr')
signal = generate_signal(df, deposit=100000, model=model)
```

### С Binance адаптером (будущее)

```python
from adapters.binance import load_data_with_indicators
from services.strategy_engine import get_model, generate_signal

# Загружаем данные из Binance
df = load_data_with_indicators(
    symbol='BTCUSDT',
    interval='1h'
)

# Тот же самый движок!
model = get_model('scalp')
signal = generate_signal(df, deposit=10000, model=model)
```

## 🎯 Торговые модели

### Доступные модели

```python
from services.strategy_engine import MODELS, get_model, list_models

# Список всех моделей
models = list_models()
# {'conservative': 'Консервативная модель...', ...}

# Получить модель
model = get_model('conservative')
```

| Модель | Min RR | Риск | Описание |
|--------|--------|------|----------|
| conservative | 2.5 | 1.0% | Высокий RR, строгие фильтры |
| high_rr | 2.0 | 1.5% | Умеренные фильтры |
| balanced | 2.0 | 1.5% | Сбалансированная (по умолчанию) |
| aggressive | 1.5 | 2.0% | Мягкие фильтры, больше сделок |
| scalp | 1.2 | 1.5% | Скальпинг, работает в рейндже |

## 📊 Компоненты

### 1. Core (Ядро)

```python
from services.strategy_engine import (
    calculate_atr,
    calculate_structure,
    calculate_distance_to_ma,
    calculate_volume_stats
)

# ATR (волатильность)
atr = calculate_atr(df, period=14)

# Структура рынка
structure = calculate_structure(df, lookback=20)
# {'structure': 'uptrend', 'phase': 'pullback', ...}

# Расстояние до MA
distance = calculate_distance_to_ma(price=300.0, ma_value=295.0)
# 1.69% (выше MA)

# Статистика объемов
volume_stats = calculate_volume_stats(df, period=20)
# {'volume_ratio': 2.5, 'is_impulse': True, ...}
```

### 2. Signals (Сигналы)

```python
from services.strategy_engine import generate_signal, TradingSignal

signal = generate_signal(df, deposit=100000, model=model)

# Атрибуты сигнала
signal.signal          # 'long', 'short', 'none'
signal.entry           # Цена входа
signal.stop            # Стоп-лосс
signal.target          # Тейк-профит
signal.rr              # Risk/Reward соотношение
signal.position_size   # Размер позиции
signal.confidence      # 'high', 'medium', 'low'
signal.warnings        # Список предупреждений

# Конвертация в словарь
signal_dict = signal.to_dict()
```

### 3. Risk (Риск-менеджмент)

```python
from services.strategy_engine import (
    calculate_position_risk,
    calculate_stop_by_atr,
    calculate_target_by_rr,
    calculate_kelly_criterion
)

# Расчет риска
risk = calculate_position_risk(
    entry=300.0,
    stop=295.0,
    target=310.0,
    deposit=100000,
    max_risk_percent=1.5,
    min_rr=2.0
)

# Стоп на основе ATR
stop = calculate_stop_by_atr(
    entry=300.0,
    atr=5.0,
    multiplier=1.5,
    direction='long'
)

# Тейк на основе RR
target = calculate_target_by_rr(
    entry=300.0,
    stop=295.0,
    rr=2.0,
    direction='long'
)

# Критерий Келли (оптимальный риск)
kelly = calculate_kelly_criterion(
    winrate=55.0,
    avg_win=1000.0,
    avg_loss=500.0
)
```

### 4. Backtest (Бэктестинг)

```python
from services.strategy_engine import run_backtest, compare_models_results

# Запуск бэктеста
results = run_backtest(
    df=df,
    signal_generator=generate_signal,
    deposit=100000,
    model=model,
    lookback_window=300,
    max_holding_candles=50
)

# Метрики
results.winrate           # Процент прибыльных сделок
results.profit_factor     # Отношение прибыли к убытку
results.expectancy        # Матожидание на сделку
results.sharpe_ratio      # Шарп коэффициент
results.max_drawdown      # Максимальная просадка
results.total_trades      # Всего сделок

# Сравнение моделей
all_results = [results1, results2, results3]
comparison = compare_models_results(all_results)
print(comparison)
```

## 🔧 Требования к данным

DataFrame должен содержать следующие колонки:

| Колонка | Тип | Описание |
|---------|-----|----------|
| `open` | float | Цена открытия |
| `high` | float | Максимальная цена |
| `low` | float | Минимальная цена |
| `close` | float | Цена закрытия |
| `volume` | float | Объем торгов |
| `ma50` | float | Скользящая средняя 50 |
| `ma200` | float | Скользящая средняя 200 |
| `rsi` | float | RSI индикатор |

Index: `pd.DatetimeIndex` (временные метки)

## 🎨 Создание своей модели

```python
from services.strategy_engine import TradingModel

custom_model = TradingModel(
    name="my_ultra_conservative",
    description="Ультра-консервативная модель",
    min_rr=3.0,                      # Минимум 1:3
    max_risk_percent=0.5,            # Риск 0.5%
    min_volume_ratio=2.0,            # Только импульсные свечи
    atr_multiplier_stop=2.0,         # Широкий стоп
    trend_required=True,             # Только в тренде
    allow_range=False,               # Не торговать в рейндже
    min_trend_strength=3.0,          # Сильный тренд
    rsi_overbought=65,               # RSI < 65 для лонга
    rsi_oversold=35,                 # RSI > 35 для шорта
    max_distance_ma50=2.0,           # Близко к MA50
    require_impulse=True,            # Требовать импульс
    min_confidence='high'            # Только высокая уверенность
)

# Использование
signal = generate_signal(df, deposit=100000, model=custom_model)
```

## 📈 Логика стратегии

### Условия для LONG

1. ✅ Структура: `uptrend` (HH/HL)
2. ✅ Фаза: `pullback` к MA50
3. ✅ Цена > MA200
4. ✅ RSI < overbought (по умолчанию 70)
5. ✅ Volume >= min_volume_ratio
6. ✅ Расстояние до MA50 < max_distance_ma50

### Условия для SHORT

1. ✅ Структура: `downtrend` (LH/LL)
2. ✅ Фаза: `pullback` к MA50
3. ✅ Цена < MA200
4. ✅ RSI > oversold (по умолчанию 30)
5. ✅ Volume >= min_volume_ratio
6. ✅ Расстояние до MA50 < max_distance_ma50

### Расчет стопа и тейка

```
stop_distance = ATR × atr_multiplier_stop
stop = entry ± stop_distance

risk = |entry - stop|
profit = risk × min_rr
target = entry ± profit
```

## 🔌 Интеграция с адаптерами

### Пример для MOEX

```python
# adapters/moex/__init__.py
from services.strategy_engine import (
    generate_signal,
    get_model,
    MODELS,
    run_backtest
)

# Теперь можно использовать:
# from adapters.moex import generate_signal, get_model
```

### Пример для Binance (будущее)

```python
# adapters/binance/__init__.py
from services.strategy_engine import (
    generate_signal,
    get_model,
    MODELS
)

# Тот же API!
```

## 🧪 Тестирование

```python
# Простой тест
from services.strategy_engine import get_model, generate_signal
import pandas as pd

# Создаем тестовый DataFrame
df = pd.DataFrame({
    'open': [100, 101, 102],
    'high': [102, 103, 104],
    'low': [99, 100, 101],
    'close': [101, 102, 103],
    'volume': [1000, 1500, 2000],
    'ma50': [100, 100.5, 101],
    'ma200': [95, 95.5, 96],
    'rsi': [50, 55, 60]
}, index=pd.date_range('2026-01-01', periods=3, freq='1H'))

model = get_model('balanced')
signal = generate_signal(df, deposit=100000, model=model)

assert signal is not None
print("✅ Тест пройден")
```

## 📚 API Reference

### Функции

- `get_model(name: str) -> TradingModel`
- `generate_signal(df, deposit, model) -> TradingSignal`
- `calculate_atr(df, period) -> pd.Series`
- `calculate_structure(df, lookback) -> Dict`
- `calculate_position_risk(...) -> RiskParameters`
- `run_backtest(...) -> BacktestResults`

### Классы

- `TradingModel` - Конфигурация торговой модели
- `TradingSignal` - Торговый сигнал с метаданными
- `RiskParameters` - Параметры риска
- `BacktestResults` - Результаты бэктеста
- `Trade` - Информация о сделке

## 🤝 Вклад

Strategy Engine - универсальный движок. При добавлении нового адаптера:

1. Подготовьте данные в нужном формате (OHLCV + индикаторы)
2. Используйте `generate_signal()` для получения сигналов
3. Используйте `run_backtest()` для тестирования
4. Не дублируйте логику - используйте strategy_engine!

## 📄 Лицензия

См. LICENSE в корне проекта.

---

**Версия:** 1.0.0
**Статус:** Production Ready
**Совместимость:** MOEX ✅, Binance (planned), Alfa Invest (planned)
