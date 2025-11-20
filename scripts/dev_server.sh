#!/usr/bin/env bash
# Quick local development startup script
# Starts the API server with hot-reload and useful dev settings

set -euo pipefail

echo "🚀 Starting Face Recognition API in development mode..."

# Check if virtual environment is activated
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "⚠️  Warning: Virtual environment not activated"
    echo "   Run: source venv/bin/activate"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if dependencies are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "❌ FastAPI not found. Installing dependencies..."
    pip install -r requirements.txt
fi

# Load environment variables if .env exists
if [ -f .env ]; then
    echo "📋 Loading environment variables from .env"
    set -a
    source .env
    set +a
else
    echo "⚠️  No .env file found. Using defaults."
    echo "   Copy .env.template to .env and customize."
fi

# Set development defaults
export LOG_LEVEL=${LOG_LEVEL:-DEBUG}
export DEBUG=${DEBUG:-true}
export RELOAD=${RELOAD:-true}

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}

# Create required directories
mkdir -p data/uploads models logs

echo ""
echo "📝 Configuration:"
echo "   Host: $HOST"
echo "   Port: $PORT"
echo "   Log Level: $LOG_LEVEL"
echo "   Hot Reload: $RELOAD"
echo ""
echo "📚 API Docs: http://localhost:$PORT/docs"
echo "🔍 Health Check: http://localhost:$PORT/health"
echo ""

# Start the server
uvicorn src.api_server:app \
    --host "$HOST" \
    --port "$PORT" \
    --reload \
    --log-level "${LOG_LEVEL,,}" \
    --access-log

# Note: Press Ctrl+C to stop
