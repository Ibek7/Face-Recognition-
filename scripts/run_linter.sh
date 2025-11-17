#!/usr/bin/env bash
# Run linters and static analysis tools
set -euo pipefail

echo "Running flake8..."
if command -v flake8 >/dev/null 2>&1; then
  flake8 --max-line-length=120 || true
else
  echo "flake8 not installed; install with: pip install flake8"
fi

echo "Running bandit (security checks)..."
if command -v bandit >/dev/null 2>&1; then
  bandit -r src -lll || true
else
  echo "bandit not installed; install with: pip install bandit"
fi

# Optionally run mypy if available
if command -v mypy >/dev/null 2>&1; then
  echo "Running mypy..."
  mypy src || true
fi
