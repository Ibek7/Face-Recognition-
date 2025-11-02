# Contributing to Face Recognition System

Thank you for your interest in contributing to the Face Recognition System! This document provides guidelines and instructions for contributing to the project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)

## 🤝 Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. We expect all contributors to:

- Be respectful and considerate
- Accept constructive criticism gracefully
- Focus on what's best for the community
- Show empathy towards others

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of face recognition concepts
- Familiarity with OpenCV and FastAPI

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Face-Recognition-.git
   cd Face-Recognition-
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/Ibek7/Face-Recognition-.git
   ```

## 🛠️ Development Setup

### 1. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

### 3. Install Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

This will automatically run linting and formatting checks before each commit.

### 4. Set Up Environment Variables

```bash
cp .env.example .env
# Edit .env with your local configuration
```

### 5. Initialize the Database

```bash
python -c "from src.database import DatabaseManager; db = DatabaseManager(); print('Database initialized')"
```

## 💡 How to Contribute

### Types of Contributions

We welcome various types of contributions:

- **Bug Fixes**: Fix issues and bugs
- **Features**: Implement new features
- **Documentation**: Improve or add documentation
- **Tests**: Add or improve test coverage
- **Performance**: Optimize existing code
- **Refactoring**: Improve code quality

### Before Starting Work

1. **Check existing issues**: Look for related issues or discussions
2. **Create an issue**: If none exists, create one describing your proposed changes
3. **Wait for approval**: For major changes, wait for maintainer feedback
4. **Claim the issue**: Comment on the issue to let others know you're working on it

## 📝 Coding Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line length**: Maximum 100 characters
- **Quotes**: Use double quotes for strings
- **Imports**: Use absolute imports, grouped and sorted with `isort`
- **Docstrings**: Use Google-style docstrings

### Code Formatting

We use the following tools (automatically run via pre-commit):

- **Black**: Code formatting
  ```bash
  black src tests scripts
  ```

- **isort**: Import sorting
  ```bash
  isort src tests scripts
  ```

- **Flake8**: Linting
  ```bash
  flake8 src tests scripts
  ```

### Example Code Style

```python
"""
Module docstring describing the module's purpose.
"""

import os
from typing import List, Optional

import numpy as np
import cv2

from src.detection import FaceDetector


class ExampleClass:
    """
    Class docstring describing the class.
    
    Attributes:
        attribute_name: Description of the attribute
    """
    
    def __init__(self, param: str):
        """
        Initialize the class.
        
        Args:
            param: Description of the parameter
        """
        self.attribute_name = param
    
    def example_method(self, arg1: int, arg2: Optional[str] = None) -> List[str]:
        """
        Method docstring with Google-style format.
        
        Args:
            arg1: Description of arg1
            arg2: Optional description of arg2
            
        Returns:
            List of results
            
        Raises:
            ValueError: If arg1 is negative
        """
        if arg1 < 0:
            raise ValueError("arg1 must be non-negative")
        
        # Implementation here
        return ["result"]
```

### Type Hints

- Use type hints for all function parameters and return values
- Use `typing` module for complex types
- Use `Optional` for nullable parameters

### Error Handling

- Always validate input parameters
- Provide meaningful error messages
- Use appropriate exception types
- Log errors with proper context

## 🧪 Testing Guidelines

### Writing Tests

- Write tests for all new features
- Maintain or improve code coverage (minimum 70%)
- Use pytest for all tests
- Follow the Arrange-Act-Assert pattern

### Test Structure

```python
import pytest
from src.module import function_to_test


class TestFunctionName:
    """Test suite for function_to_test."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        return {"key": "value"}
    
    def test_valid_input(self, sample_data):
        """Test function with valid input."""
        result = function_to_test(sample_data)
        assert result == expected_value
    
    def test_invalid_input(self):
        """Test function with invalid input."""
        with pytest.raises(ValueError):
            function_to_test(None)
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_detection.py

# Run specific test
pytest tests/test_detection.py::TestFaceDetector::test_valid_input

# Run tests with markers
pytest -m "not slow"
```

## 🔄 Pull Request Process

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `test/` - Test additions or improvements
- `refactor/` - Code refactoring

### 2. Make Your Changes

- Write clear, concise commit messages
- Follow conventional commits format:
  - `feat:` - New feature
  - `fix:` - Bug fix
  - `docs:` - Documentation
  - `test:` - Tests
  - `refactor:` - Code refactoring
  - `ci:` - CI/CD changes
  - `chore:` - Maintenance tasks

Example:
```bash
git commit -m "feat: add face quality assessment metrics"
git commit -m "fix: resolve memory leak in embedding generation"
```

### 3. Keep Your Branch Updated

```bash
git fetch upstream
git rebase upstream/main
```

### 4. Run Quality Checks

Before submitting, ensure all checks pass:

```bash
# Format code
black src tests scripts
isort src tests scripts

# Run linters
flake8 src tests scripts

# Run tests
pytest --cov=src

# Type checking
mypy src
```

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear title describing the change
- Detailed description of what and why
- Reference to related issue(s)
- Screenshots (if applicable)

### PR Checklist

Before submitting your PR, ensure:

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] Branch is up to date with main
- [ ] No merge conflicts
- [ ] PR description is complete

## 🐛 Issue Reporting

### Bug Reports

When reporting bugs, include:

- **Description**: Clear description of the issue
- **Steps to Reproduce**: Exact steps to reproduce the bug
- **Expected Behavior**: What you expected to happen
- **Actual Behavior**: What actually happened
- **Environment**: OS, Python version, package versions
- **Screenshots**: If applicable
- **Logs**: Relevant error messages or logs

### Feature Requests

When requesting features, include:

- **Problem**: What problem does this solve?
- **Solution**: Proposed solution or feature
- **Alternatives**: Alternative solutions considered
- **Additional Context**: Any other relevant information

## 📚 Additional Resources

- [Project Documentation](docs/)
- [API Documentation](docs/API.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Security Guidelines](docs/SECURITY.md)

## 🎓 Learning Resources

- [OpenCV Documentation](https://docs.opencv.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [NumPy Documentation](https://numpy.org/doc/)
- [pytest Documentation](https://docs.pytest.org/)

## ❓ Questions?

If you have questions:

1. Check existing documentation
2. Search existing issues
3. Ask in issue discussions
4. Create a new issue with the `question` label

## 🙏 Thank You!

Thank you for contributing to the Face Recognition System! Your efforts help make this project better for everyone.

---

**Happy Coding! 🚀**
