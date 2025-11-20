# Contributing Guidelines

Thank you for contributing to Face Recognition! This guide outlines our development process.

## Quick Start

1. Fork and clone the repository
2. Create a virtual environment: `python3 -m venv venv && source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt -r requirements-dev.local.txt`
4. Install pre-commit: `./scripts/setup_precommit.sh`

## Development Workflow

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation
- `refactor/description` - Code refactoring

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `test:` Tests
- `chore:` Maintenance

Examples:
```
feat: add batch face processing endpoint
fix: resolve embedding cache memory leak
docs: update installation instructions
```

## Code Standards

### Python Style

- Follow PEP 8
- Use type hints
- Max line length: 120
- Google-style docstrings

```python
def process_image(image: np.ndarray, threshold: float = 0.8) -> List[Face]:
    """
    Process image and detect faces.
    
    Args:
        image: Input image array
        threshold: Detection confidence threshold
        
    Returns:
        List of detected Face objects
    """
    pass
```

### Testing

- Write tests for new features
- Aim for >80% coverage
- Run tests: `pytest`
- Run with coverage: `pytest --cov=src`

### Code Quality

Before committing:
```bash
# Format code
black src tests
isort src tests

# Run linters
./scripts/run_linter.sh

# Run tests
pytest
```

## Pull Requests

### Before Submitting

- [ ] Tests pass
- [ ] Linters pass
- [ ] Documentation updated
- [ ] Commit messages follow conventions
- [ ] Rebased on latest main

### PR Process

1. Push branch to your fork
2. Create PR with descriptive title
3. Fill out PR template
4. Address review comments
5. Wait for approval from maintainer

### PR Template

```markdown
## Description
What does this PR do?

## Type
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Refactoring

## Testing
- [ ] Tests added/updated
- [ ] All tests passing

## Checklist
- [ ] Code formatted
- [ ] Docs updated
- [ ] No new warnings
```

## Testing Guidelines

### Unit Tests

```python
def test_face_detector_finds_faces():
    # Arrange
    detector = FaceDetector()
    image = load_sample_image()
    
    # Act
    faces = detector.detect(image)
    
    # Assert
    assert len(faces) > 0
```

### Integration Tests

Test complete workflows in `tests/test_api_integration.py`

## Documentation

- Update API docs for new endpoints
- Add docstrings to all public functions
- Update README.md for major features
- Include usage examples

## Reporting Issues

### Bug Reports

Include:
- Python version
- OS
- Steps to reproduce
- Expected vs actual behavior
- Error messages

### Feature Requests

Include:
- Use case
- Proposed solution
- Alternative approaches

## Code Review

- Be constructive and respectful
- Focus on code, not the person
- Explain reasoning
- Suggest improvements

## Getting Help

- Check documentation in `docs/`
- Review existing issues/PRs
- Open a discussion on GitHub

## License

By contributing, you agree your contributions will be licensed under the project's LICENSE.

Thank you! 🚀
