#!/usr/bin/env bash
# Install pre-commit hooks and optionally dependencies
set -euo pipefail

python -m pip install --upgrade pip
if [ -f requirements-dev.local.txt ]; then
  pip install -r requirements-dev.local.txt
fi

# Install pre-commit
pip install pre-commit || true

# Install hooks
pre-commit install || true

# Optionally run once on all files
pre-commit run --all-files || true

echo "Pre-commit installed and hooks run (where available)."
