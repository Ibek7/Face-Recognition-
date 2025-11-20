#!/usr/bin/env bash
# Cleanup script to remove cache, temporary files, and build artifacts

set -euo pipefail

echo "🧹 Cleaning up Face Recognition project..."

# Python cache files
echo "Removing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
find . -type f -name "*.pyd" -delete 2>/dev/null || true

# Python egg and dist
echo "Removing build artifacts..."
rm -rf build/ dist/ *.egg-info/ .eggs/ 2>/dev/null || true

# Test and coverage files
echo "Removing test artifacts..."
rm -rf .pytest_cache/ .tox/ .coverage htmlcov/ coverage.xml .coverage.* 2>/dev/null || true

# Jupyter notebook checkpoints
echo "Removing Jupyter checkpoints..."
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true

# MyPy cache
echo "Removing mypy cache..."
rm -rf .mypy_cache/ 2>/dev/null || true

# Log files
echo "Removing log files..."
find . -type f -name "*.log" -delete 2>/dev/null || true
rm -rf logs/*.log 2>/dev/null || true

# Temporary files
echo "Removing temporary files..."
find . -type f -name ".DS_Store" -delete 2>/dev/null || true
find . -type f -name "Thumbs.db" -delete 2>/dev/null || true
find . -type f -name "*~" -delete 2>/dev/null || true
find . -type f -name "*.swp" -delete 2>/dev/null || true
find . -type f -name "*.swo" -delete 2>/dev/null || true

# Optional: Docker cleanup
if command -v docker >/dev/null 2>&1; then
    read -p "🐳 Clean Docker images and containers? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing stopped containers..."
        docker container prune -f 2>/dev/null || true
        
        echo "Removing dangling images..."
        docker image prune -f 2>/dev/null || true
        
        echo "Removing build cache..."
        docker builder prune -f 2>/dev/null || true
    fi
fi

# Optional: Remove virtual environment
read -p "🔥 Remove virtual environment (venv/)? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Removing venv..."
    rm -rf venv/ 2>/dev/null || true
fi

echo "✅ Cleanup complete!"

# Show disk space saved
if command -v du >/dev/null 2>&1; then
    echo ""
    echo "💾 Current project size:"
    du -sh . 2>/dev/null || true
fi
