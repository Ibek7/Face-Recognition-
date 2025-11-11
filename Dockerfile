# Stage 1: Builder
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake libopencv-dev libdlib-dev

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Production
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends libopencv-core4.5 libopencv-imgproc4.5 libgomp1 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY src ./src

# Expose port and define entrypoint
EXPOSE 8000
CMD ["uvicorn", "src.api_server:app", "--host", "0.0.0.0", "--port", "8000"]

