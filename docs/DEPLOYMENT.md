# Production Deployment Guide

This guide covers deploying the Face Recognition System in production environments with high availability, scalability, and security.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Infrastructure Requirements](#infrastructure-requirements)
3. [Security Configuration](#security-configuration)
4. [Database Setup](#database-setup)
5. [Application Deployment](#application-deployment)
6. [Load Balancing](#load-balancing)
7. [Monitoring & Logging](#monitoring--logging)
8. [Backup & Recovery](#backup--recovery)
9. [Performance Optimization](#performance-optimization)
10. [Troubleshooting](#troubleshooting)

## Architecture Overview

### Production Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Internet / CDN                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                Load Balancer (Nginx/HAProxy)                │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              Application Cluster                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  API Server │  │  API Server │  │  API Server │         │
│  │   (Docker)  │  │   (Docker)  │  │   (Docker)  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                  Data Layer                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ PostgreSQL  │  │    Redis    │  │   MinIO     │         │
│  │  (Primary)  │  │   Cache     │  │ (File Store)│         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ PostgreSQL  │  │ Prometheus  │  │   Grafana   │         │
│  │ (Replica)   │  │ Monitoring  │  │ Dashboard   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

- **Load Balancer**: SSL termination, request routing, health checks
- **API Servers**: Face recognition processing, API endpoints
- **PostgreSQL**: User data, embeddings, recognition results
- **Redis**: Session cache, temporary data, task queues
- **MinIO**: Image storage, model artifacts
- **Monitoring**: System metrics, alerts, logging

## Infrastructure Requirements

### Minimum Production Requirements

**Application Servers** (3+ instances):
- CPU: 4 cores (8 threads)
- RAM: 8 GB
- Storage: 50 GB SSD
- GPU: Optional (NVIDIA GTX 1060 or better for ML inference)

**Database Server**:
- CPU: 4 cores
- RAM: 16 GB
- Storage: 100 GB SSD (with replication)
- IOPS: 3000+ for optimal performance

**Load Balancer**:
- CPU: 2 cores
- RAM: 4 GB
- Storage: 20 GB
- Network: High bandwidth interface

### Recommended Production Setup

**Application Servers** (5+ instances):
- CPU: 8 cores (16 threads)
- RAM: 16 GB
- Storage: 100 GB NVMe SSD
- GPU: NVIDIA RTX 3070 or better

**Database Cluster**:
- Primary: 8 cores, 32 GB RAM, 500 GB SSD
- Replica: 4 cores, 16 GB RAM, 500 GB SSD
- Backup storage: 1 TB with daily snapshots

## Security Configuration

### SSL/TLS Setup

**1. SSL Certificate Installation**

```bash
# Using Let's Encrypt
sudo apt install certbot nginx
sudo certbot --nginx -d your-domain.com

# Or using custom certificates
sudo cp your-cert.pem /etc/ssl/certs/
sudo cp your-key.pem /etc/ssl/private/
sudo chmod 600 /etc/ssl/private/your-key.pem
```

**2. Nginx SSL Configuration**

```nginx
# /etc/nginx/sites-available/face-recognition
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name your-domain.com;

    # SSL Configuration
    ssl_certificate /etc/ssl/certs/your-cert.pem;
    ssl_certificate_key /etc/ssl/private/your-key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-Frame-Options DENY always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # Rate Limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;

    # Proxy Configuration
    location / {
        proxy_pass http://backend_servers;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
        
        # File upload limits
        client_max_body_size 10M;
    }

    # WebSocket Support
    location /ws/ {
        proxy_pass http://backend_servers;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

# Backend server pool
upstream backend_servers {
    least_conn;
    server api-server-1:8000 max_fails=3 fail_timeout=30s;
    server api-server-2:8000 max_fails=3 fail_timeout=30s;
    server api-server-3:8000 max_fails=3 fail_timeout=30s;
    
    # Health checks
    keepalive 32;
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    listen [::]:80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}
```

### Authentication & Authorization

**1. API Key Authentication**

```python
# src/auth.py
import hashlib
import secrets
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

class APIKeyManager:
    def __init__(self):
        self.api_keys = self._load_api_keys()
    
    def _load_api_keys(self):
        # Load from environment or secure storage
        return {
            "api_key_hash_1": {"user": "admin", "permissions": ["read", "write"]},
            "api_key_hash_2": {"user": "readonly", "permissions": ["read"]}
        }
    
    def verify_api_key(self, credentials: HTTPAuthorizationCredentials = Depends(security)):
        api_key_hash = hashlib.sha256(credentials.credentials.encode()).hexdigest()
        
        if api_key_hash not in self.api_keys:
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        return self.api_keys[api_key_hash]

api_key_manager = APIKeyManager()

# Usage in API endpoints
@app.post("/secure-endpoint")
async def secure_endpoint(user_info = Depends(api_key_manager.verify_api_key)):
    return {"message": "Access granted", "user": user_info["user"]}
```

**2. JWT Authentication**

```python
# src/jwt_auth.py
import jwt
from datetime import datetime, timedelta
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer

SECRET_KEY = "your-secret-key"  # Use environment variable
ALGORITHM = "HS256"

def create_access_token(data: dict, expires_delta: timedelta = timedelta(hours=24)):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

### Network Security

**1. Firewall Configuration**

```bash
# Ubuntu/Debian with ufw
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable

# Database access (internal network only)
sudo ufw allow from 10.0.0.0/8 to any port 5432
sudo ufw allow from 10.0.0.0/8 to any port 6379
```

**2. Docker Security**

```dockerfile
# Use non-root user
FROM python:3.9-slim

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Database Setup

### PostgreSQL Production Configuration

**1. Installation and Basic Setup**

```bash
# Install PostgreSQL 14
sudo apt update
sudo apt install postgresql-14 postgresql-client-14 postgresql-contrib-14

# Configure PostgreSQL
sudo -u postgres psql
```

```sql
-- Create database and user
CREATE DATABASE face_recognition;
CREATE USER face_app WITH ENCRYPTED PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE face_recognition TO face_app;

-- Performance tuning (adjust based on your hardware)
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;

-- Reload configuration
SELECT pg_reload_conf();
```

**2. Database Optimization**

```sql
-- Create indexes for face recognition queries
CREATE INDEX idx_face_embeddings_person_id ON face_embeddings(person_id);
CREATE INDEX idx_face_embeddings_quality ON face_embeddings(quality_score DESC);
CREATE INDEX idx_recognition_results_person_id ON recognition_results(person_id);
CREATE INDEX idx_recognition_results_timestamp ON recognition_results(created_at DESC);

-- Partitioning for large datasets
CREATE TABLE recognition_results_y2024m01 PARTITION OF recognition_results
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- Vacuum and analyze
VACUUM ANALYZE;
```

**3. Backup Configuration**

```bash
# Automated backup script
#!/bin/bash
# /opt/backups/postgres_backup.sh

BACKUP_DIR="/opt/backups/postgres"
DATE=$(date +%Y%m%d_%H%M%S)
DB_NAME="face_recognition"

mkdir -p $BACKUP_DIR

# Create backup
pg_dump -h localhost -U face_app $DB_NAME | gzip > $BACKUP_DIR/backup_$DATE.sql.gz

# Keep only last 7 days
find $BACKUP_DIR -name "backup_*.sql.gz" -mtime +7 -delete

# Upload to S3 (optional)
# aws s3 cp $BACKUP_DIR/backup_$DATE.sql.gz s3://your-backup-bucket/
```

```bash
# Crontab entry for daily backups
0 2 * * * /opt/backups/postgres_backup.sh
```

### Redis Configuration

**1. Production Redis Setup**

```bash
# Install Redis
sudo apt install redis-server

# Configure Redis
sudo nano /etc/redis/redis.conf
```

```conf
# /etc/redis/redis.conf

# Network
bind 127.0.0.1
port 6379
protected-mode yes

# Security
requirepass your_redis_password

# Memory management
maxmemory 1gb
maxmemory-policy allkeys-lru

# Persistence
save 900 1
save 300 10
save 60 10000

# AOF
appendonly yes
appendfsync everysec

# Performance
tcp-keepalive 300
timeout 0
```

## Application Deployment

### Docker Production Setup

**1. Multi-stage Dockerfile**

```dockerfile
# Multi-stage build for production
FROM python:3.9-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    cmake \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.9-slim as production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy application
COPY . .
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set PATH for user packages
ENV PATH=/root/.local/bin:$PATH

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=10)"

CMD ["uvicorn", "src.api_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

**2. Production Docker Compose**

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  app:
    build: 
      context: .
      target: production
    environment:
      - DATABASE_URL=postgresql://face_app:${DB_PASSWORD}@postgres:5432/face_recognition
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
      - SECRET_KEY=${SECRET_KEY}
      - DEBUG=false
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '1'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G
    networks:
      - app-network
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:14-alpine
    environment:
      - POSTGRES_DB=face_recognition
      - POSTGRES_USER=face_app
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./docker/postgres-init:/docker-entrypoint-initdb.d
    deploy:
      resources:
        limits:
          memory: 2G
    networks:
      - app-network

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD} --maxmemory 512mb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    networks:
      - app-network

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx/prod.conf:/etc/nginx/nginx.conf:ro
      - ./docker/ssl:/etc/ssl/certs:ro
    depends_on:
      - app
    networks:
      - app-network

volumes:
  postgres_data:
    driver: local
  redis_data:
    driver: local

networks:
  app-network:
    driver: bridge
```

### Kubernetes Deployment

**1. Deployment Configuration**

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: face-recognition-api
  labels:
    app: face-recognition
spec:
  replicas: 5
  selector:
    matchLabels:
      app: face-recognition
  template:
    metadata:
      labels:
        app: face-recognition
    spec:
      containers:
      - name: api
        image: face-recognition:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: redis-url
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: face-recognition-service
spec:
  selector:
    app: face-recognition
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: face-recognition-ingress
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  tls:
  - hosts:
    - your-domain.com
    secretName: tls-secret
  rules:
  - host: your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: face-recognition-service
            port:
              number: 80
```

## Load Balancing

### HAProxy Configuration

```haproxy
# /etc/haproxy/haproxy.cfg
global
    maxconn 4096
    log 127.0.0.1:514 local0
    
defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms
    option httplog

# Frontend
frontend face_recognition_frontend
    bind *:80
    bind *:443 ssl crt /etc/ssl/certs/combined.pem
    redirect scheme https if !{ ssl_fc }
    
    # Rate limiting
    stick-table type ip size 100k expire 30s store http_req_rate(10s)
    http-request track-sc0 src
    http-request reject if { sc_http_req_rate(0) gt 20 }
    
    default_backend face_recognition_backend

# Backend
backend face_recognition_backend
    balance roundrobin
    option httpchk GET /health
    
    server api1 10.0.1.10:8000 check inter 5s fall 3 rise 2
    server api2 10.0.1.11:8000 check inter 5s fall 3 rise 2
    server api3 10.0.1.12:8000 check inter 5s fall 3 rise 2
    
# Stats page
listen stats
    bind *:8404
    stats enable
    stats uri /stats
    stats refresh 30s
    stats admin if TRUE
```

### Application-Level Load Balancing

```python
# src/load_balancer.py
import random
import time
from typing import List, Dict
from dataclasses import dataclass
import httpx

@dataclass
class ServerInstance:
    url: str
    weight: int = 1
    healthy: bool = True
    last_check: float = 0
    response_time: float = 0

class LoadBalancer:
    def __init__(self, servers: List[str]):
        self.servers = [ServerInstance(url=url) for url in servers]
        self.health_check_interval = 30  # seconds
    
    async def health_check(self):
        """Check health of all servers."""
        async with httpx.AsyncClient() as client:
            for server in self.servers:
                try:
                    start_time = time.time()
                    response = await client.get(f"{server.url}/health", timeout=5)
                    server.response_time = time.time() - start_time
                    server.healthy = response.status_code == 200
                    server.last_check = time.time()
                except Exception:
                    server.healthy = False
                    server.last_check = time.time()
    
    def get_server(self) -> ServerInstance:
        """Get next server using weighted round-robin."""
        healthy_servers = [s for s in self.servers if s.healthy]
        
        if not healthy_servers:
            raise Exception("No healthy servers available")
        
        # Weighted selection based on response time (lower is better)
        weights = [1 / (server.response_time + 0.1) for server in healthy_servers]
        return random.choices(healthy_servers, weights=weights)[0]

# Usage
load_balancer = LoadBalancer([
    "http://api-server-1:8000",
    "http://api-server-2:8000",
    "http://api-server-3:8000"
])

@app.middleware("http")
async def load_balance_middleware(request, call_next):
    # Periodic health check
    if time.time() - getattr(load_balancer, 'last_health_check', 0) > 30:
        await load_balancer.health_check()
        load_balancer.last_health_check = time.time()
    
    response = await call_next(request)
    return response
```

## Monitoring & Logging

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'face-recognition-api'
    static_configs:
      - targets: ['api-server-1:8000', 'api-server-2:8000', 'api-server-3:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
    
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:9187']
    
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:9121']
    
  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx:9113']

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Custom Metrics

```python
# src/metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
REQUEST_COUNT = Counter('face_recognition_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('face_recognition_request_duration_seconds', 'Request duration')
ACTIVE_CONNECTIONS = Gauge('face_recognition_active_connections', 'Active connections')
FACE_RECOGNITION_TIME = Histogram('face_recognition_processing_seconds', 'Face recognition processing time')

class MetricsMiddleware:
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            start_time = time.time()
            
            # Track active connections
            ACTIVE_CONNECTIONS.inc()
            
            try:
                await self.app(scope, receive, send)
                status = "success"
            except Exception as e:
                status = "error"
                raise
            finally:
                # Record metrics
                duration = time.time() - start_time
                REQUEST_DURATION.observe(duration)
                REQUEST_COUNT.labels(
                    method=scope["method"],
                    endpoint=scope["path"],
                    status=status
                ).inc()
                ACTIVE_CONNECTIONS.dec()
        else:
            await self.app(scope, receive, send)

# Add metrics endpoint
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

### Logging Configuration

```python
# src/logging_config.py
import logging
import json
from datetime import datetime
from pythonjsonlogger import jsonlogger

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record['timestamp'] = datetime.utcnow().isoformat()
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        
        # Add request ID if available
        if hasattr(record, 'request_id'):
            log_record['request_id'] = record.request_id

def setup_logging():
    formatter = CustomJsonFormatter('%(timestamp)s %(level)s %(logger)s %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # File handler
    file_handler = logging.FileHandler('/var/log/face-recognition.log')
    file_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Specific loggers
    logging.getLogger('uvicorn.error').setLevel(logging.INFO)
    logging.getLogger('uvicorn.access').setLevel(logging.INFO)
```

### Alert Rules

```yaml
# alert_rules.yml
groups:
- name: face_recognition_alerts
  rules:
  - alert: HighErrorRate
    expr: rate(face_recognition_requests_total{status="error"}[5m]) > 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} errors per second"
  
  - alert: HighResponseTime
    expr: histogram_quantile(0.95, rate(face_recognition_request_duration_seconds_bucket[5m])) > 2.0
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High response time detected"
      description: "95th percentile response time is {{ $value }} seconds"
  
  - alert: DatabaseConnectionFailure
    expr: up{job="postgres"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Database connection failure"
      description: "PostgreSQL database is not responding"
```

## Backup & Recovery

### Automated Backup Strategy

```bash
#!/bin/bash
# /opt/backups/full_backup.sh

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_ROOT="/opt/backups"
S3_BUCKET="your-backup-bucket"

# Create backup directories
mkdir -p $BACKUP_ROOT/database
mkdir -p $BACKUP_ROOT/files
mkdir -p $BACKUP_ROOT/models

# Database backup
echo "Backing up database..."
pg_dump -h postgres -U face_app face_recognition | gzip > $BACKUP_ROOT/database/db_$BACKUP_DATE.sql.gz

# File storage backup (if using local storage)
echo "Backing up uploaded files..."
tar -czf $BACKUP_ROOT/files/files_$BACKUP_DATE.tar.gz /app/data/uploads/

# Model files backup
echo "Backing up model files..."
tar -czf $BACKUP_ROOT/models/models_$BACKUP_DATE.tar.gz /app/models/

# Upload to S3
if command -v aws &> /dev/null; then
    echo "Uploading to S3..."
    aws s3 sync $BACKUP_ROOT/ s3://$S3_BUCKET/backups/$(date +%Y%m%d)/
fi

# Cleanup old backups (keep 30 days)
find $BACKUP_ROOT -name "*.gz" -mtime +30 -delete
find $BACKUP_ROOT -name "*.tar.gz" -mtime +30 -delete

echo "Backup completed: $BACKUP_DATE"
```

### Disaster Recovery Plan

```bash
#!/bin/bash
# /opt/recovery/restore.sh

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <backup_date>"
    echo "Example: $0 20240115_140000"
    exit 1
fi

BACKUP_DATE=$1
BACKUP_ROOT="/opt/backups"

echo "Starting disaster recovery for backup: $BACKUP_DATE"

# Stop services
echo "Stopping services..."
docker-compose down

# Restore database
echo "Restoring database..."
gunzip -c $BACKUP_ROOT/database/db_$BACKUP_DATE.sql.gz | psql -h localhost -U face_app face_recognition

# Restore files
echo "Restoring files..."
tar -xzf $BACKUP_ROOT/files/files_$BACKUP_DATE.tar.gz -C /

# Restore models
echo "Restoring models..."
tar -xzf $BACKUP_ROOT/models/models_$BACKUP_DATE.tar.gz -C /

# Start services
echo "Starting services..."
docker-compose up -d

echo "Recovery completed successfully"
```

## Performance Optimization

### Database Performance Tuning

```sql
-- Optimize for face recognition workloads

-- Increase work_mem for sorting operations
SET work_mem = '256MB';

-- Optimize for read-heavy workloads
SET effective_cache_size = '2GB';
SET shared_buffers = '512MB';

-- Connection pooling settings
SET max_connections = 200;

-- Vacuum and analyze schedule
-- Add to crontab: 0 2 * * * psql -c "VACUUM ANALYZE;"

-- Partitioning for large tables
CREATE TABLE recognition_results_partitioned (
    LIKE recognition_results INCLUDING ALL
) PARTITION BY RANGE (created_at);

-- Create monthly partitions
CREATE TABLE recognition_results_2024_01 PARTITION OF recognition_results_partitioned
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
```

### Application Performance

```python
# src/performance_optimizations.py
import asyncio
import uvloop
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Use uvloop for better async performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Thread pool for CPU-bound tasks
cpu_executor = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())

# Async face recognition
async def async_face_recognition(image_data):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(cpu_executor, process_face_recognition, image_data)

# Connection pooling
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600
)

# Redis connection pooling
import aioredis

redis_pool = aioredis.ConnectionPool.from_url(
    REDIS_URL,
    max_connections=20
)

# Caching decorator
from functools import wraps
import pickle

def cache_result(expiration=300):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            redis = aioredis.Redis(connection_pool=redis_pool)
            cached = await redis.get(cache_key)
            
            if cached:
                return pickle.loads(cached)
            
            # Compute result
            result = await func(*args, **kwargs)
            
            # Cache result
            await redis.setex(cache_key, expiration, pickle.dumps(result))
            
            return result
        return wrapper
    return decorator

# Usage
@cache_result(expiration=600)
async def get_person_embeddings(person_id):
    # Expensive database operation
    pass
```

### GPU Optimization

```python
# src/gpu_optimization.py
import torch
from torch.utils.data import DataLoader
import numpy as np

class GPUOptimizedModel:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path)
        
        # Optimize for inference
        self.model.eval()
        if hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
    
    def batch_predict(self, images, batch_size=32):
        """Process multiple images efficiently."""
        dataset = torch.utils.data.TensorDataset(torch.stack(images))
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
        
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                batch = batch[0].to(self.device)
                pred = self.model(batch)
                predictions.append(pred.cpu())
        
        return torch.cat(predictions)
    
    @torch.inference_mode()
    def predict_single(self, image):
        """Optimized single image prediction."""
        image = image.unsqueeze(0).to(self.device)
        return self.model(image)
```

This deployment guide provides comprehensive coverage of production deployment considerations. Let me know if you need me to expand on any particular section or add additional topics!