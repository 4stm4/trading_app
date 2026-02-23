## Docker Compose (Portainer)

Структура исходников:
- `src/backend` — backend (FastAPI, ETL, доменная логика)
- `src/backend/_tests` — backend unit-тесты
- `src/frontend` — frontend (React + Vite)

В репозитории добавлен `docker-compose.yml` для запуска:
- `postgres` (PostgreSQL 16)
- `backend` (FastAPI, внешний порт `5001`)
- `frontend` (React + Nginx, порт `8050`)

### Локально
```bash
cp .env.docker.example .env
docker compose up -d --build
```

Для подключения к внешней БД (ваш случай):
- `DB_HOST=192.168.88.3`
- `DB_PORT=5432`
- `DB_NAME=trading`
- `DB_USER=alex`
- `DB_PASSWORD=<your_password>`

Backend использует эти параметры для `DATABASE_URL` и `ALEMBIC_DATABASE_URL`.

После запуска:
- Frontend: `http://localhost:8050`
- Backend API: `http://localhost:5001/api/health`
- Postgres: `localhost:5432`

### Portainer Stack
1. `Stacks -> Add stack`.
2. Вставьте содержимое `docker-compose.yml`.
3. В `Environment variables` задайте при необходимости:
   - `POSTGRES_DB`
   - `POSTGRES_USER`
   - `POSTGRES_PASSWORD`
   - `POSTGRES_PORT`
   - `DB_HOST`
   - `DB_PORT`
   - `DB_NAME`
   - `DB_USER`
   - `DB_PASSWORD`
   - `BACKEND_PORT`
   - `FRONTEND_PORT`
4. Нажмите `Deploy the stack`.

### Свой Docker Registry
Если у вас свой registry, можно деплоить в Portainer без `build`, только через `image`.

Сборка и публикация образов:
```bash
export REGISTRY=registry.example.com/my-team
export TAG=latest

docker build -t ${REGISTRY}/trading-backend:${TAG} ./src/backend
docker build -t ${REGISTRY}/trading-frontend:${TAG} ./src/frontend

docker push ${REGISTRY}/trading-backend:${TAG}
docker push ${REGISTRY}/trading-frontend:${TAG}
```

Автоматизированный деплой в `registry2` и перезапуск compose:
```bash
./scripts/deploy_registry2.sh
```

Переопределяемые переменные:
```bash
REGISTRY2_URL=localhost:5000 \
IMAGE_TAG=latest \
SKIP_GIT_PULL=0 \
./scripts/deploy_registry2.sh
```

Пример stack-файла для Portainer (registry-only):
```yaml
version: "3.9"
services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: trading_app
      POSTGRES_USER: trading
      POSTGRES_PASSWORD: trading
    volumes:
      - postgres_data:/var/lib/postgresql/data

  backend:
    image: registry.example.com/my-team/trading-backend:latest
    depends_on:
      - postgres
    environment:
      DB_HOST: postgres
      DB_PORT: "5432"
      PORT: "5000"
      DATABASE_URL: postgresql+psycopg://trading:trading@postgres:5432/trading_app
      ALEMBIC_DATABASE_URL: postgresql+psycopg://trading:trading@postgres:5432/trading_app
    ports:
      - "5001:5000"

  frontend:
    image: registry.example.com/my-team/trading-frontend:latest
    depends_on:
      - backend
    ports:
      - "8050:80"

volumes:
  postgres_data:
```

## Загрузка ENV в shell
```bash
set -a; source dev.env; set +a;
```
