#!/usr/bin/env bash
# Code quality check script
# Runs linting, formatting, type checking, and tests

set -e  # Exit on error

echo "=========================================="
echo "Code Quality Check"
echo "=========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if virtual environment is activated
if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo -e "${YELLOW}Warning: Virtual environment not activated${NC}"
    echo "Activate with: source .venv/bin/activate"
    echo ""
fi

# 1. Code Formatting Check
echo "1. Checking code formatting with Black..."
if black --check src/ tests/ scripts/; then
    echo -e "${GREEN}✓ Code formatting passed${NC}"
else
    echo -e "${RED}✗ Code formatting failed${NC}"
    echo "Run: black src/ tests/ scripts/"
    exit 1
fi
echo ""

# 2. Import Sorting Check
echo "2. Checking import sorting with isort..."
if isort --check-only src/ tests/ scripts/; then
    echo -e "${GREEN}✓ Import sorting passed${NC}"
else
    echo -e "${RED}✗ Import sorting failed${NC}"
    echo "Run: isort src/ tests/ scripts/"
    exit 1
fi
echo ""

# 3. Linting with flake8
echo "3. Running flake8 linter..."
if flake8 src/ tests/ scripts/ --count --statistics; then
    echo -e "${GREEN}✓ Linting passed${NC}"
else
    echo -e "${RED}✗ Linting failed${NC}"
    exit 1
fi
echo ""

# 4. Type Checking with mypy
echo "4. Running type checker (mypy)..."
if mypy src/ --ignore-missing-imports; then
    echo -e "${GREEN}✓ Type checking passed${NC}"
else
    echo -e "${YELLOW}⚠ Type checking had issues (non-blocking)${NC}"
fi
echo ""

# 5. Security Check with bandit
echo "5. Running security checks (bandit)..."
if bandit -r src/ -ll -q; then
    echo -e "${GREEN}✓ Security check passed${NC}"
else
    echo -e "${YELLOW}⚠ Security issues found (review recommended)${NC}"
fi
echo ""

# 6. Running Tests
echo "6. Running tests with pytest..."
if pytest tests/ -v --cov=src --cov-report=term-missing; then
    echo -e "${GREEN}✓ All tests passed${NC}"
else
    echo -e "${RED}✗ Tests failed${NC}"
    exit 1
fi
echo ""

# Summary
echo "=========================================="
echo -e "${GREEN}✓ All quality checks passed!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  - Review coverage report above"
echo "  - Commit your changes"
echo "  - Push to remote repository"
echo ""
