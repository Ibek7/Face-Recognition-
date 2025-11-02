# Face Recognition System - Development Makefile
# Comprehensive task automation for development workflow

PYTHON := python3
PIP := pip
VENV := .venv
VENV_BIN := $(VENV)/bin
PYTHON_VENV := $(VENV_BIN)/python
PIP_VENV := $(VENV_BIN)/pip

# Docker configuration
DOCKER_IMAGE := face-recognition
DOCKER_TAG := latest
DOCKER_COMPOSE := docker-compose

# Source and test directories
SRC_DIR := src
TEST_DIR := tests
SCRIPT_DIR := scripts

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help message
	@echo "$(BLUE)Face Recognition System - Development Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# ========================================
# Environment Setup
# ========================================

.PHONY: venv
venv: ## Create virtual environment
	@echo "$(BLUE)Creating virtual environment...$(NC)"
	$(PYTHON) -m venv $(VENV)
	@echo "$(GREEN)✓ Virtual environment created$(NC)"
	@echo "Activate it with: source $(VENV_BIN)/activate"

.PHONY: install
install: ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	$(PIP_VENV) install --upgrade pip setuptools wheel
	$(PIP_VENV) install -r requirements.txt
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

.PHONY: install-dev
install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(PIP_VENV) install --upgrade pip setuptools wheel
	$(PIP_VENV) install -r requirements.txt
	$(PIP_VENV) install -r requirements-dev.txt
	@echo "$(GREEN)✓ Development dependencies installed$(NC)"

.PHONY: setup
setup: venv install-dev ## Complete development setup
	@echo "$(BLUE)Setting up pre-commit hooks...$(NC)"
	$(VENV_BIN)/pre-commit install
	@echo "$(GREEN)✓ Development environment ready!$(NC)"

# ========================================
# Code Quality
# ========================================

.PHONY: format
format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	$(VENV_BIN)/black $(SRC_DIR) $(TEST_DIR) $(SCRIPT_DIR)
	$(VENV_BIN)/isort $(SRC_DIR) $(TEST_DIR) $(SCRIPT_DIR)
	@echo "$(GREEN)✓ Code formatted$(NC)"

.PHONY: lint
lint: ## Run all linters (flake8, pylint, mypy)
	@echo "$(BLUE)Running linters...$(NC)"
	$(VENV_BIN)/flake8 $(SRC_DIR) $(TEST_DIR) $(SCRIPT_DIR) || true
	$(VENV_BIN)/pylint $(SRC_DIR) --exit-zero
	$(VENV_BIN)/mypy $(SRC_DIR) --ignore-missing-imports || true
	@echo "$(GREEN)✓ Linting complete$(NC)"

.PHONY: check
check: format lint ## Format and lint code
	@echo "$(GREEN)✓ Code quality checks complete$(NC)"

.PHONY: type-check
type-check: ## Run type checking with mypy
	@echo "$(BLUE)Running type checker...$(NC)"
	$(VENV_BIN)/mypy $(SRC_DIR) --ignore-missing-imports
	@echo "$(GREEN)✓ Type checking complete$(NC)"

# ========================================
# Testing
# ========================================

.PHONY: test
test: ## Run unit tests
	@echo "$(BLUE)Running tests...$(NC)"
	$(VENV_BIN)/pytest $(TEST_DIR) -v
	@echo "$(GREEN)✓ Tests passed$(NC)"

.PHONY: test-coverage
test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	$(VENV_BIN)/pytest $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)✓ Coverage report generated: htmlcov/index.html$(NC)"

.PHONY: test-fast
test-fast: ## Run tests without slow tests
	@echo "$(BLUE)Running fast tests...$(NC)"
	$(VENV_BIN)/pytest $(TEST_DIR) -v -m "not slow"
	@echo "$(GREEN)✓ Fast tests passed$(NC)"

.PHONY: test-watch
test-watch: ## Run tests in watch mode
	@echo "$(BLUE)Running tests in watch mode...$(NC)"
	$(VENV_BIN)/pytest-watch $(TEST_DIR) -v

# ========================================
# Application
# ========================================

.PHONY: run-api
run-api: ## Start the API server
	@echo "$(BLUE)Starting API server...$(NC)"
	$(VENV_BIN)/uvicorn src.api_server:app --reload --host 0.0.0.0 --port 8000

.PHONY: run-batch
run-batch: ## Run batch face detection
	@echo "$(BLUE)Running batch detection...$(NC)"
	$(PYTHON_VENV) $(SCRIPT_DIR)/batch_detect.py --input-dir data/images --output-dir results

.PHONY: run-benchmark
run-benchmark: ## Run performance benchmark
	@echo "$(BLUE)Running benchmark...$(NC)"
	$(PYTHON_VENV) $(SCRIPT_DIR)/benchmark.py --test-images data/images --output-dir benchmark_results

# ========================================
# Docker
# ========================================

.PHONY: docker-build
docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	@echo "$(GREEN)✓ Docker image built$(NC)"

.PHONY: docker-run
docker-run: ## Run Docker container
	@echo "$(BLUE)Running Docker container...$(NC)"
	docker run -p 8000:8000 -v $$(pwd)/data:/app/data $(DOCKER_IMAGE):$(DOCKER_TAG)

.PHONY: docker-compose-up
docker-compose-up: ## Start services with docker-compose
	@echo "$(BLUE)Starting services...$(NC)"
	$(DOCKER_COMPOSE) up --build

.PHONY: docker-compose-down
docker-compose-down: ## Stop docker-compose services
	@echo "$(BLUE)Stopping services...$(NC)"
	$(DOCKER_COMPOSE) down

.PHONY: docker-clean
docker-clean: ## Clean Docker images and containers
	@echo "$(BLUE)Cleaning Docker resources...$(NC)"
	docker system prune -f
	@echo "$(GREEN)✓ Docker cleaned$(NC)"

# ========================================
# Database
# ========================================

.PHONY: db-init
db-init: ## Initialize database
	@echo "$(BLUE)Initializing database...$(NC)"
	$(PYTHON_VENV) -c "from src.database import DatabaseManager; db = DatabaseManager(); print('Database initialized')"
	@echo "$(GREEN)✓ Database initialized$(NC)"

.PHONY: db-reset
db-reset: ## Reset database (WARNING: deletes all data)
	@echo "$(RED)WARNING: This will delete all data!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -f *.db face_recognition*.db; \
		$(MAKE) db-init; \
		echo "$(GREEN)✓ Database reset$(NC)"; \
	fi

# ========================================
# Documentation
# ========================================

.PHONY: docs
docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	@echo "Documentation available in docs/"
	@echo "$(GREEN)✓ Documentation ready$(NC)"

.PHONY: docs-serve
docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8080$(NC)"
	$(PYTHON_VENV) -m http.server 8080 --directory docs/

# ========================================
# Cleaning
# ========================================

.PHONY: clean
clean: ## Clean build and cache files
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	rm -rf runs __pycache__ .pytest_cache .coverage .mypy_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	@echo "$(GREEN)✓ Cleaned$(NC)"

.PHONY: clean-all
clean-all: clean docker-clean ## Clean everything including Docker
	@echo "$(BLUE)Deep cleaning...$(NC)"
	rm -rf htmlcov coverage.xml *.log logs/ results/ benchmark_results/
	rm -rf $(VENV)
	@echo "$(GREEN)✓ Deep clean complete$(NC)"

# ========================================
# Security
# ========================================

.PHONY: security-scan
security-scan: ## Run security vulnerability scan
	@echo "$(BLUE)Running security scan...$(NC)"
	$(VENV_BIN)/bandit -r $(SRC_DIR) -f json -o bandit-report.json
	$(VENV_BIN)/safety check
	@echo "$(GREEN)✓ Security scan complete$(NC)"

# ========================================
# Utilities
# ========================================

.PHONY: deps-update
deps-update: ## Update dependencies
	@echo "$(BLUE)Updating dependencies...$(NC)"
	$(PIP_VENV) install --upgrade pip setuptools wheel
	$(PIP_VENV) install --upgrade -r requirements.txt
	@echo "$(GREEN)✓ Dependencies updated$(NC)"

.PHONY: deps-list
deps-list: ## List installed packages
	@echo "$(BLUE)Installed packages:$(NC)"
	$(PIP_VENV) list

.PHONY: info
info: ## Show project information
	@echo "$(BLUE)Project Information$(NC)"
	@echo "Python: $$($(PYTHON_VENV) --version)"
	@echo "Virtual Environment: $(VENV)"
	@echo "Docker Image: $(DOCKER_IMAGE):$(DOCKER_TAG)"
