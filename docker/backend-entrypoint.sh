#!/usr/bin/env sh
set -eu

echo "Waiting for database ${DB_HOST:-postgres}:${DB_PORT:-5432} ..."
python - <<'PY'
import os
import socket
import sys
import time

host = os.getenv("DB_HOST", "postgres")
port = int(os.getenv("DB_PORT", "5432"))
timeout = int(os.getenv("DB_WAIT_TIMEOUT", "90"))
deadline = time.time() + timeout

while time.time() < deadline:
    try:
        with socket.create_connection((host, port), timeout=2):
            print(f"Database is reachable: {host}:{port}")
            sys.exit(0)
    except OSError:
        time.sleep(1)

print(f"Timeout waiting for database: {host}:{port}", file=sys.stderr)
sys.exit(1)
PY

echo "Applying Alembic migrations ..."
alembic -c alembic.ini upgrade head

echo "Starting backend on port ${PORT:-5000} ..."
exec uvicorn ports.api.app:app --host 0.0.0.0 --port "${PORT:-5000}" --log-level info
