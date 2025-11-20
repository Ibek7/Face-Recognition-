# Developer Guide

Quick setup guide for local development on the Face Recognition project.

## Prerequisites

- Python 3.10+
- Docker (optional, for containerized development)
- Git

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/Ibek7/Face-Recognition-.git
cd Face-Recognition-

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.local.txt
```

### 2. Install Development Tools

```bash
# Run setup script
./scripts/setup_precommit.sh

# Or manually
pip install pre-commit
pre-commit install
```

### 3. Run the Application

```bash
# Start API server
uvicorn src.api_server:app --reload --host 0.0.0.0 --port 8000

# Visit http://localhost:8000/docs for API documentation
```

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_api_integration.py
```

### Code Quality

```bash
# Run linters
./scripts/run_linter.sh

# Format code
black src tests
isort src tests

# Type checking
mypy src
```

### Docker Development

```bash
# Build image
./scripts/build_image.sh

# Run container
docker run -p 8000:8000 face-recognition:local

# Or use docker-compose
docker-compose up
```

## Project Structure

```
.
├── src/                    # Application source code
│   ├── api_server.py      # FastAPI application
│   ├── detection.py       # Face detection
│   ├── embeddings.py      # Face embeddings
│   └── ...
├── scripts/               # Utility scripts
├── tests/                 # Test files
├── docs/                  # Documentation
├── data/                  # Data files (not in git)
├── models/                # Model files (not in git)
└── notebooks/             # Jupyter notebooks
```

## Common Tasks

### Add New Endpoint

1. Edit `src/api_server.py`
2. Add route handler
3. Add tests in `tests/`
4. Update API docs

### Update Dependencies

```bash
# Add to requirements.txt
pip install <package>
pip freeze | grep <package> >> requirements.txt

# Check for vulnerabilities
safety check
```

### Database Migrations

```bash
# Create backup
python scripts/backup_restore.py create

# Run migration
# (add your migration commands here)
```

### Generate Documentation

```bash
# Generate API docs
python scripts/generate_docs.py --app src.api_server:app --output docs/api

# Serve docs locally
python scripts/generate_docs.py --app src.api_server:app --serve --port 8080
```

## Environment Variables

Create a `.env` file (see `.env.template`):

```env
LOG_LEVEL=DEBUG
DATABASE_URL=postgresql://user:pass@localhost/dbname
REDIS_URL=redis://localhost:6379
SECRET_KEY=your-secret-key-here
```

## Debugging

### Enable Debug Logging

```python
from src.app_logging import configure_app_logging

configure_app_logging(log_level="DEBUG", log_file="debug.log")
```

### Profile Performance

```bash
python scripts/advanced_profiler.py --target src.api_server:app
```

### Load Testing

```bash
python scripts/performance_load_test.py --url http://localhost:8000 --requests 1000
```

## Git Workflow

```bash
# Create feature branch
git checkout -b feature/my-feature

# Make changes and commit
git add .
git commit -m "feat: add new feature"

# Push and create PR
git push origin feature/my-feature
```

## Code Style

- Follow PEP 8
- Use type hints
- Write docstrings (Google style)
- Keep functions small and focused
- Add tests for new features

## Troubleshooting

### Import Errors

```bash
# Ensure you're in the virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Port Already in Use

```bash
# Find process using port 8000
lsof -ti:8000

# Kill process
kill -9 $(lsof -ti:8000)
```

### Docker Issues

```bash
# Clean up Docker
docker system prune -a

# Rebuild image
docker-compose build --no-cache
```

## Resources

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [API Documentation](docs/API.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Security Checks](docs/SECURITY_CHECKS.md)

## Getting Help

- Open an issue on GitHub
- Check existing documentation in `docs/`
- Review code examples in `notebooks/`
