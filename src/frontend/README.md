# Frontend (React + TradingView Lightweight Charts)

## Требования
- Node.js 18+
- npm 9+

## Установка
```bash
cd src/frontend
npm install
```

## Запуск в режиме разработки
Перед запуском frontend поднимите API (для загрузки инструментов MOEX):
```bash
cd src/backend
python composites/api.py
```

В отдельном терминале запустите frontend:
```bash
cd src/frontend
npm run dev
```

После запуска откройте адрес из вывода Vite (обычно `http://localhost:5173`).
Запросы `frontend -> /api/*` автоматически проксируются на `http://localhost:5000`.

## Сборка production
```bash
cd src/frontend
npm run build
```

Собранные файлы будут в папке `src/frontend/dist`.

## Локальный просмотр production-сборки
```bash
cd src/frontend
npm run preview
```
