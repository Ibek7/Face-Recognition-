# Production Deployment Checklist

Use this checklist before deploying to production.

## Pre-Deployment

### Security
- [ ] Change all default passwords and secrets
- [ ] Generate strong `SECRET_KEY` in `.env`
- [ ] Set `API_TOKEN` for authentication
- [ ] Enable HTTPS/TLS
- [ ] Review and restrict CORS origins
- [ ] Scan Docker image for vulnerabilities: `docker scan`
- [ ] Update all dependencies to latest secure versions
- [ ] Enable rate limiting
- [ ] Configure firewall rules

### Configuration
- [ ] Set `DEBUG=false`
- [ ] Configure production database (PostgreSQL recommended)
- [ ] Set appropriate `LOG_LEVEL` (INFO or WARNING)
- [ ] Configure log rotation
- [ ] Set resource limits (memory, CPU)
- [ ] Configure backup strategy
- [ ] Set up monitoring and alerting
- [ ] Configure auto-scaling rules (if applicable)

### Performance
- [ ] Enable caching (`ENABLE_CACHING=true`)
- [ ] Configure Redis for distributed caching
- [ ] Optimize database indexes
- [ ] Set appropriate worker count (`API_WORKERS`)
- [ ] Enable GPU if available (`ENABLE_GPU=true`)
- [ ] Test with production-like load
- [ ] Set up CDN for static assets (if applicable)

### Database
- [ ] Create database backups
- [ ] Test database restore procedure
- [ ] Set up automated backup schedule
- [ ] Configure connection pooling
- [ ] Review and optimize queries
- [ ] Set up database monitoring
- [ ] Plan for database migrations

### Infrastructure
- [ ] Choose deployment platform (Docker, Kubernetes, Cloud Run, etc.)
- [ ] Set up load balancer
- [ ] Configure health checks
- [ ] Set up CI/CD pipeline
- [ ] Configure auto-restart on failure
- [ ] Set up log aggregation
- [ ] Configure metrics collection

## Deployment Steps

### 1. Build Production Image
```bash
docker build -f Dockerfile.prod -t face-recognition:v1.0.0 .
```

### 2. Tag Image
```bash
docker tag face-recognition:v1.0.0 gcr.io/your-project/face-recognition:v1.0.0
docker tag face-recognition:v1.0.0 gcr.io/your-project/face-recognition:latest
```

### 3. Push to Registry
```bash
docker push gcr.io/your-project/face-recognition:v1.0.0
docker push gcr.io/your-project/face-recognition:latest
```

### 4. Deploy
```bash
# Kubernetes
kubectl apply -f k8s/

# Docker Compose
docker-compose -f docker-compose.prod.yml up -d

# Cloud Run
gcloud run deploy face-recognition \
  --image gcr.io/your-project/face-recognition:v1.0.0 \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### 5. Verify Deployment
```bash
# Health check
curl https://your-domain.com/health

# API documentation
curl https://your-domain.com/docs
```

## Post-Deployment

### Monitoring
- [ ] Verify health check endpoint
- [ ] Check application logs
- [ ] Monitor CPU and memory usage
- [ ] Monitor API response times
- [ ] Set up error tracking (Sentry, etc.)
- [ ] Configure uptime monitoring
- [ ] Set up alerts for critical issues

### Testing
- [ ] Run smoke tests in production
- [ ] Test critical API endpoints
- [ ] Verify database connectivity
- [ ] Test file uploads
- [ ] Verify face recognition accuracy
- [ ] Test rate limiting
- [ ] Verify CORS configuration

### Documentation
- [ ] Update deployment documentation
- [ ] Document production environment variables
- [ ] Create runbook for common issues
- [ ] Document rollback procedure
- [ ] Update API documentation
- [ ] Share access credentials securely

### Backup & Recovery
- [ ] Verify backup automation is working
- [ ] Test database restore
- [ ] Document recovery procedures
- [ ] Set up disaster recovery plan
- [ ] Test rollback procedure

## Rollback Plan

If issues occur:

### 1. Quick Rollback
```bash
# Kubernetes
kubectl rollout undo deployment/face-recognition

# Cloud Run
gcloud run services update-traffic face-recognition \
  --to-revisions=PREVIOUS_REVISION=100
```

### 2. Full Rollback
```bash
# Redeploy previous version
docker pull gcr.io/your-project/face-recognition:v0.9.0
kubectl set image deployment/face-recognition \
  app=gcr.io/your-project/face-recognition:v0.9.0
```

### 3. Database Rollback
```bash
# Restore from backup
psql -U user -d face_recognition < backup_pre_deployment.sql
```

## Environment Variables for Production

```bash
# Database
DATABASE_URL=postgresql://user:secure_password@db-host:5432/face_recognition

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=8

# Security
SECRET_KEY=<generated-secret-key>
API_TOKEN=<generated-api-token>
CORS_ORIGINS=https://yourdomain.com

# Performance
ENABLE_CACHING=true
REDIS_URL=redis://redis-host:6379/0
ENABLE_GPU=true

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/face_recognition/app.log

# Monitoring
ENABLE_MONITORING=true
SENTRY_DSN=https://your-sentry-dsn
```

## Maintenance Window

Schedule regular maintenance:
- [ ] Weekly: Review logs and metrics
- [ ] Monthly: Update dependencies
- [ ] Quarterly: Performance optimization review
- [ ] Annually: Security audit

## Emergency Contacts

- DevOps Lead: [contact info]
- Database Admin: [contact info]
- Security Team: [contact info]
- On-call Engineer: [rotation schedule]

## Sign-off

- [ ] Reviewed by: ___________________ Date: ___________
- [ ] Approved by: ___________________ Date: ___________
- [ ] Deployed by: ___________________ Date: ___________

---

**Note:** Keep this checklist updated with your specific deployment requirements.
