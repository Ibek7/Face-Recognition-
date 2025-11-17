#!/usr/bin/env bash
# Simple container healthcheck script for Docker/Kubernetes
# Exits 0 when healthy, non-zero otherwise

set -euo pipefail

# Check that the API server responds on localhost:8000/health or /healthz
HOST=${1:-"http://127.0.0.1:8000"}
TIMEOUT=${2:-5}

if command -v curl >/dev/null 2>&1; then
  STATUS=$(curl -s -o /dev/null -w "%{http_code}" --max-time "$TIMEOUT" "$HOST/health" || true)
  if [ "$STATUS" = "200" ]; then
    echo "OK: $HOST/health returned 200"
    exit 0
  fi

  STATUS2=$(curl -s -o /dev/null -w "%{http_code}" --max-time "$TIMEOUT" "$HOST/healthz" || true)
  if [ "$STATUS2" = "200" ]; then
    echo "OK: $HOST/healthz returned 200"
    exit 0
  fi

  echo "UNHEALTHY: no 200 response (health checks returned $STATUS and $STATUS2)"
  exit 2
else
  # Fallback: try nc if available
  if command -v nc >/dev/null 2>&1; then
    nc -z -w "$TIMEOUT" 127.0.0.1 8000 && exit 0 || exit 2
  fi
  echo "No network tools available to perform healthcheck"
  exit 1
fi
