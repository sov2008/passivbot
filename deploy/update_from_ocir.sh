#!/usr/bin/env bash
set -euo pipefail

# Требуется заранее выполненный docker login в OCIR на инстансе.
IMAGE="${OCIR_URL}/${OCIR_NAMESPACE}/${OCIR_REPO}"
TAG="${TAG:-latest}"

echo "[updater] pulling ${IMAGE}:${TAG}..."
if docker pull "${IMAGE}:${TAG}"; then
  echo "[updater] restarting via docker-compose..."
  cd /opt/passivbot && TAG="${TAG}" docker compose pull && docker compose up -d
else
  echo "[updater] pull failed, will retry later"
fi
