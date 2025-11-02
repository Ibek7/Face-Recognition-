# Multi-stage build for Face Recognition API
# Stage 1: Builder - Install dependencies and compile
FROM python:3.10-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    cmake \
    libopencv-dev \
    python3-opencv \
    libboost-all-dev \
    libgtk-3-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    gfortran \
    openexr \
    libatlas-base-dev \
    python3-dev \
    libtbb2 \
    libtbb-dev \
    libdc1394-22-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Stage 2: Runtime - Create minimal production image
FROM python:3.10-slim

# Security labels
LABEL maintainer="Face Recognition Team" \
      description="Face Recognition API with real-time capabilities" \
      version="1.0.0" \
      security.scan="enabled"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/opt/venv/bin:$PATH" \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    WORKERS=2

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-core4.5 \
    libopencv-imgproc4.5 \
    libopencv-imgcodecs4.5 \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    libice6 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set work directory
WORKDIR /app

# Create non-root user first for security
RUN groupadd -r -g 1000 appuser && \
    useradd -r -u 1000 -g appuser -m -s /sbin/nologin appuser

# Create directories with proper permissions
RUN mkdir -p /app/data/models \
             /app/data/uploads \
             /app/data/exports \
             /app/logs \
             /app/cache && \
    chown -R appuser:appuser /app

# Copy application code with proper ownership
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check with improved reliability
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${API_PORT}/health/live || exit 1

# Add startup script for graceful initialization
COPY --chown=appuser:appuser <<EOF /app/entrypoint.sh
#!/bin/sh
set -e

echo "Starting Face Recognition API..."
echo "Python version: \$(python --version)"
echo "Working directory: \$(pwd)"

# Initialize database if needed
if [ ! -f "/app/data/face_recognition.db" ]; then
    echo "Initializing database..."
    python -c "from src.database import DatabaseManager; DatabaseManager()"
fi

# Start the application
exec uvicorn src.api_server:app \\
    --host \${API_HOST} \\
    --port \${API_PORT} \\
    --workers \${WORKERS} \\
    --log-level info \\
    --access-log \\
    --use-colors
EOF

RUN chmod +x /app/entrypoint.sh

# Run the application via entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
