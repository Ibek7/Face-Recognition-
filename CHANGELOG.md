# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- API rate limiting middleware with configurable token-bucket algorithm
- Docker Compose override file for local Postgres and Adminer development
- GitHub issue templates for bug reports and feature requests
- Pull request template with comprehensive checklist
- Centralized logging configuration with JSON formatter and performance tracking
- Multi-stage Dockerfile for optimized production builds
- Pre-commit hooks for code quality (black, isort, flake8, mypy)
- Comprehensive developer tooling (Makefile, requirements-dev.txt)
- API usage examples (Python client and curl scripts)
- Contributing guidelines and development workflow documentation

### Changed
- Enhanced README with detailed usage examples and configuration guide
- Improved API health endpoints with readiness and liveness checks
- Strengthened input validation in face detection module
- Extended CI/CD pipeline with security scanning and code quality checks

### Fixed
- Face detection parameter validation edge cases
- API error handling and response consistency

## Release Process

### Creating a Release

1. **Update Version Numbers**
   ```bash
   # Update version in relevant files
   vim setup.py  # or pyproject.toml
   ```

2. **Update CHANGELOG.md**
   ```bash
   # Move unreleased changes to new version section
   # Add release date
   ```

3. **Commit and Tag**
   ```bash
   git add CHANGELOG.md setup.py
   git commit -m "chore: bump version to X.Y.Z"
   git tag -a vX.Y.Z -m "Release version X.Y.Z"
   git push origin main --tags
   ```

4. **Create GitHub Release**
   - Go to GitHub releases
   - Create new release from tag
   - Copy changelog section for release notes
   - Attach any relevant artifacts

### Version Numbering

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes and minor improvements
