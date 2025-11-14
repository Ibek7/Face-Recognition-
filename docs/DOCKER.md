# Docker Best Practices

## Multi-Stage Builds

This project uses multi-stage Docker builds to optimize image size and security.

### Production Build

Build the optimized production image:

```bash
docker build -f Dockerfile.prod -t face-recognition:prod .
```

**Benefits:**
- Smaller final image (~500MB vs 2GB+)
- No build tools in production
- Faster deployment
- Better security

### Development Build

For local development with hot reload:

```bash
docker build -t face-recognition:dev .
docker run -v $(pwd)/src:/app/src -p 8000:8000 face-recognition:dev
```

## Image Optimization

### Current Optimizations
1. **Multi-stage builds** - Separate builder and runtime stages
2. **Layer caching** - Dependencies installed before code copy
3. **Minimal base image** - Using `python:3.10-slim`
4. **Cleanup** - Removing apt cache and unnecessary files

### Size Comparison
- Development image: ~1.5GB
- Production image: ~500MB
- Alpine-based (future): ~300MB

## Security Best Practices

### 1. Non-Root User
Future improvement:
```dockerfile
RUN useradd -m -u 1000 appuser
USER appuser
```

### 2. Vulnerability Scanning
```bash
docker scan face-recognition:prod
```

### 3. Image Signing
```bash
docker trust sign face-recognition:prod
```

## Docker Compose

### Development
```bash
docker-compose up
```

### Production
```bash
docker-compose -f docker-compose.prod.yml up -d
```

## Performance Tips

### 1. BuildKit
Enable BuildKit for faster builds:
```bash
DOCKER_BUILDKIT=1 docker build -f Dockerfile.prod -t face-recognition:prod .
```

### 2. Cache Mounts
Already implemented in Dockerfile.prod for pip cache

### 3. Parallel Builds
```bash
docker-compose build --parallel
```

## Deployment

### Kubernetes
```bash
kubectl apply -f k8s/deployment.yaml
```

### Cloud Run
```bash
gcloud run deploy face-recognition --image gcr.io/project/face-recognition:prod
```

### Docker Swarm
```bash
docker stack deploy -c docker-compose.prod.yml face-recognition
```

## Monitoring

### Container Health
```bash
docker ps --format "table {{.Names}}\t{{.Status}}"
```

### Resource Usage
```bash
docker stats face-recognition
```

### Logs
```bash
docker logs -f face-recognition --tail 100
```

## Troubleshooting

### Build Failures
```bash
docker build --no-cache -f Dockerfile.prod -t face-recognition:prod .
```

### Permission Issues
```bash
docker run --user $(id -u):$(id -g) ...
```

### Network Issues
```bash
docker run --network host ...
```
