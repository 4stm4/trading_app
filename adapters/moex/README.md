# MOEX WebSocket client (low-level)

`MoexClient` — небольшой асинхронный менеджер соединения с MOEX по WebSocket. Он управляет подключением, пингами и корректным закрытием сокета, не содержит торговой логики и готов к расширению.

## Ключевые идеи
- **Стейт-машина**: `DISCONNECTED → CONNECTING → CONNECTED → CLOSING → DISCONNECTED`. Все проверки строятся на `_state`, а не на флагах.
- **Неблокирующее подключение**: `connect()` асинхронно открывает сокет и поднимает фоновые задачи (reader, heartbeat). Повторные вызовы во время подключения/закрытия игнорируются.
- **Фоновый reader**: бесконечно читает `self._ws` и логирует приём сообщений. При закрытии/ошибках ставит `reconnect_event` (если задан) и инициирует централизованное закрытие.
- **Heartbeat/ping**: периодически шлёт `ping()` без авто-дисконнекта на сбой (чтобы не конфликтовать с reader). Сам `ping()` при нужде может инициировать закрытие.
- **Централизованное закрытие**: `_do_close()` отменяет heartbeat/reader (опционально), закрывает сокет, сбрасывает состояние через `_cleanup()`, переводит в `DISCONNECTED`.

## Как использовать
```python
import asyncio
from trading_app.adapters.moex.moex_client import MoexClient

async def main():
    client = MoexClient(heartbeat_interval=30.0)
    await client.connect()
    await asyncio.sleep(5)
    await client.ping()
    await client.disconnect()

asyncio.run(main())
```

## Расширения (идеи)
- Передавать очередь `asyncio.Queue` для дальнейшей обработки сообщений (вместо логирования в reader).
- Добавить внешнюю политику реконнекта, слушая `reconnect_event`.
- Вынести URL и heartbeat в настройки окружения. 
