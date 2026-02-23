#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${COMPOSE_FILE:-$ROOT_DIR/docker-compose.yml}"
BACKEND_DIR="${BACKEND_DIR:-$ROOT_DIR/src/backend}"
FRONTEND_DIR="${FRONTEND_DIR:-$ROOT_DIR/src/frontend}"

REGISTRY2_URL="${REGISTRY2_URL:-localhost:5000}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
BACKEND_IMAGE_NAME="${BACKEND_IMAGE_NAME:-trading-backend}"
FRONTEND_IMAGE_NAME="${FRONTEND_IMAGE_NAME:-trading-frontend}"

BACKEND_IMAGE_REF="${REGISTRY2_URL}/${BACKEND_IMAGE_NAME}:${IMAGE_TAG}"
FRONTEND_IMAGE_REF="${REGISTRY2_URL}/${FRONTEND_IMAGE_NAME}:${IMAGE_TAG}"

SKIP_GIT_PULL="${SKIP_GIT_PULL:-0}"

update_repo() {
  local dir="$1"

  if [[ "$SKIP_GIT_PULL" == "1" ]]; then
    echo "[skip] git pull disabled for: $dir"
    return 0
  fi

  if [[ ! -d "$dir/.git" ]]; then
    echo "[skip] not a git repository: $dir"
    return 0
  fi

  echo "[git] updating: $dir"
  git -C "$dir" fetch --all --prune
  git -C "$dir" pull --ff-only
}

echo "[1/5] Updating repositories"
update_repo "$BACKEND_DIR"
update_repo "$FRONTEND_DIR"

echo "[2/5] Building backend image: $BACKEND_IMAGE_REF"
docker build -t "$BACKEND_IMAGE_REF" "$BACKEND_DIR"

echo "[3/5] Building frontend image: $FRONTEND_IMAGE_REF"
docker build -t "$FRONTEND_IMAGE_REF" "$FRONTEND_DIR"

echo "[4/5] Pushing images to registry2: $REGISTRY2_URL"
docker push "$BACKEND_IMAGE_REF"
docker push "$FRONTEND_IMAGE_REF"

echo "[5/5] Restarting stack from registry images"
export BACKEND_IMAGE="$BACKEND_IMAGE_REF"
export FRONTEND_IMAGE="$FRONTEND_IMAGE_REF"
docker compose -f "$COMPOSE_FILE" pull backend frontend
docker compose -f "$COMPOSE_FILE" up -d --no-build --force-recreate postgres backend frontend

echo "Done."
echo "Backend image:  $BACKEND_IMAGE_REF"
echo "Frontend image: $FRONTEND_IMAGE_REF"
