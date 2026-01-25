# Trading App

Минимальный каркас для фронта на Dash и адаптеров к биржам/потокам данных.

## Что внутри
- `adapters/moex/` — WebSocket-клиент MOEX (`MoexClient`) + pydantic-модели `AssetInfo` и связки сущностей.
- `adapters/binance/` — пример получения klines через Binance Spot API и сборки их в модели `KLine`/`KLines`.
- `entities/` — pydantic-модели для данных бирж и каркас пользователя на Flask-SQLAlchemy.
- `frontend/` — страницы Dash (авторизация, регистрация, заглушка домашней страницы) и компоненты UI.
- `settings.py` — конфигурация на `pydantic-settings` (env-файл `dev.env`).
- `pyproject.toml` — настройки Ruff (линтер/форматер).

## Требования
- Python 3.10+ (Ruff настроен на `py310`).
- Виртуальное окружение (рекомендуется `python -m venv .venv`).

## Установка
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Создайте `dev.env` в корне (или задайте переменные среды):
```
APP_DB_USER=...
APP_DB_PASS=...
APP_DB_HOST=...
APP_DB_NAME=...
APP_DB_PORT=...
APIKEY=...       # Binance API key
SECRETKEY=...    # Binance secret
LOG_LEVEL=DEBUG
```

## Линт/формат
- Форматирование: `ruff format .`
- Линт и автофикс: `ruff check . --fix`

## Примечания
- `.venv` уже в `.gitignore`.
- Подключение к БД использует URL `mssql+pymssql`; добавьте драйвер/конфигурацию при необходимости.
- В `frontend/components/klines.py` пока заглушка — график можно добавить позже.
