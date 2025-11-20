# Security Checks and Hardening

This document outlines security scanning and hardening procedures for the Face Recognition application.

## Table of Contents

- [Dependency Scanning](#dependency-scanning)
- [Container Security](#container-security)
- [Code Security](#code-security)
- [Runtime Security](#runtime-security)
- [Compliance](#compliance)

## Dependency Scanning

### Python Dependencies

Check for known vulnerabilities in Python packages:

```bash
# Install safety
pip install safety

# Scan requirements
safety check --file requirements.txt

# Scan installed packages
safety check
```

### Alternative: pip-audit

```bash
pip install pip-audit
pip-audit -r requirements.txt
```

## Container Security

### Image Scanning with Trivy

```bash
# Install Trivy
brew install aquasecurity/trivy/trivy  # macOS
# or download from https://github.com/aquasecurity/trivy

# Scan Docker image
trivy image face-recognition:latest

# Scan with severity filters
trivy image --severity HIGH,CRITICAL face-recognition:latest
```

### Docker Best Practices

- ✅ Use multi-stage builds (see `Dockerfile`)
- ✅ Run as non-root user
- ✅ Minimize image layers
- ✅ Use specific base image versions
- ✅ Scan images before deployment

### Example: Add non-root user to Dockerfile

```dockerfile
RUN addgroup --system appuser && adduser --system --ingroup appuser appuser
USER appuser
```

## Code Security

### Static Analysis with Bandit

```bash
# Run bandit security linter
bandit -r src -lll

# Generate report
bandit -r src -f json -o bandit-report.json
```

### Secrets Detection

```bash
# Install gitleaks
brew install gitleaks

# Scan repository for secrets
gitleaks detect --source . --verbose

# Scan specific commits
gitleaks protect --staged
```

## Runtime Security

### API Security Headers

Ensure the following headers are set:

- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `Strict-Transport-Security: max-age=31536000`
- `Content-Security-Policy`

### Rate Limiting

Implement rate limiting (see `src/adaptive_rate_limiter.py`):

```python
from src.adaptive_rate_limiter import TokenBucketLimiter

limiter = TokenBucketLimiter(rate=100, capacity=200)
```

### Input Validation

- Validate all user inputs
- Sanitize file uploads
- Use Pydantic models for API validation
- Implement size limits on requests

## Compliance

### GDPR/Privacy Considerations

When dealing with facial recognition data:

1. **Consent**: Obtain explicit consent before collecting biometric data
2. **Storage**: Encrypt facial embeddings at rest
3. **Retention**: Implement data retention policies
4. **Access**: Log all access to biometric data
5. **Deletion**: Provide mechanisms for data deletion

### Audit Logging

Enable comprehensive audit logging:

```python
from src.audit_compliance import AuditLogger

audit = AuditLogger()
audit.log_access(user_id, resource, action)
```

## Security Checklist

- [ ] Run dependency scans before deployment
- [ ] Scan Docker images for vulnerabilities
- [ ] Enable HTTPS/TLS in production
- [ ] Use environment variables for secrets (never commit)
- [ ] Implement authentication and authorization
- [ ] Enable audit logging
- [ ] Set up monitoring and alerting
- [ ] Regular security updates
- [ ] Backup encryption keys securely
- [ ] Document incident response procedures

## Security Contacts

For security issues, please contact:
- Security Team: security@example.com
- Report vulnerabilities privately via GitHub Security Advisories

## Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CIS Docker Benchmark](https://www.cisecurity.org/benchmark/docker)
- [Python Security Best Practices](https://python.readthedocs.io/en/latest/library/security_warnings.html)
