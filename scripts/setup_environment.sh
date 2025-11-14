#!/bin/bash

###############################################################################
# Environment Setup Script
# 
# Automated environment setup for the Face Recognition project.
# Handles Python environment, dependencies, database, and configuration.
###############################################################################

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
VENV_DIR="${PROJECT_ROOT}/.venv"
ENV_FILE="${PROJECT_ROOT}/.env"

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Print header
print_header() {
    echo ""
    echo "=========================================="
    echo "  Face Recognition Environment Setup"
    echo "=========================================="
    echo ""
}

# Check if command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Check system requirements
check_requirements() {
    log_step "Checking system requirements..."
    
    local missing_deps=()
    
    # Check Python
    if ! command_exists python3; then
        missing_deps+=("python3")
    else
        local python_ver=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
        log_info "Found Python $python_ver"
    fi
    
    # Check pip
    if ! command_exists pip3; then
        missing_deps+=("pip3")
    fi
    
    # Check git
    if ! command_exists git; then
        missing_deps+=("git")
    fi
    
    # Check Docker (optional)
    if command_exists docker; then
        log_info "Docker installed: $(docker --version | head -1)"
    else
        log_warn "Docker not found (optional for containerized deployment)"
    fi
    
    # Check Docker Compose (optional)
    if command_exists docker-compose; then
        log_info "Docker Compose installed: $(docker-compose --version)"
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Missing required dependencies: ${missing_deps[*]}"
        log_error "Please install them and run this script again"
        exit 1
    fi
    
    log_info "✓ All required dependencies found"
}

# Create Python virtual environment
setup_venv() {
    log_step "Setting up Python virtual environment..."
    
    if [ -d "$VENV_DIR" ]; then
        log_warn "Virtual environment already exists at $VENV_DIR"
        read -p "Do you want to recreate it? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_DIR"
        else
            log_info "Using existing virtual environment"
            return 0
        fi
    fi
    
    python3 -m venv "$VENV_DIR"
    log_info "✓ Virtual environment created at $VENV_DIR"
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    
    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --quiet --upgrade pip setuptools wheel
    
    log_info "✓ Virtual environment ready"
}

# Install Python dependencies
install_dependencies() {
    log_step "Installing Python dependencies..."
    
    source "$VENV_DIR/bin/activate"
    
    # Install production dependencies
    if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
        log_info "Installing production dependencies..."
        pip install --quiet -r "$PROJECT_ROOT/requirements.txt"
        log_info "✓ Production dependencies installed"
    else
        log_warn "requirements.txt not found"
    fi
    
    # Install development dependencies
    if [ -f "$PROJECT_ROOT/requirements-dev.txt" ]; then
        log_info "Installing development dependencies..."
        pip install --quiet -r "$PROJECT_ROOT/requirements-dev.txt"
        log_info "✓ Development dependencies installed"
    else
        log_warn "requirements-dev.txt not found"
    fi
}

# Setup environment file
setup_env_file() {
    log_step "Setting up environment configuration..."
    
    if [ -f "$ENV_FILE" ]; then
        log_warn "Environment file already exists at $ENV_FILE"
        read -p "Do you want to overwrite it? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Keeping existing .env file"
            return 0
        fi
    fi
    
    # Copy from example if exists
    if [ -f "$PROJECT_ROOT/.env.example" ]; then
        cp "$PROJECT_ROOT/.env.example" "$ENV_FILE"
        log_info "✓ Created .env from .env.example"
    else
        # Create basic .env file
        cat > "$ENV_FILE" << EOF
# Environment
ENVIRONMENT=development
DEBUG=true

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
WORKERS=4

# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/face_recognition

# Redis
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=$(openssl rand -hex 32)

# Models
MODEL_PATH=models
DETECTION_MODEL=yolov8
RECOGNITION_MODEL=facenet

# Logging
LOG_LEVEL=DEBUG
EOF
        log_info "✓ Created default .env file"
    fi
    
    log_warn "Please review and update $ENV_FILE with your settings"
}

# Create necessary directories
create_directories() {
    log_step "Creating project directories..."
    
    local dirs=(
        "$PROJECT_ROOT/data"
        "$PROJECT_ROOT/data/images"
        "$PROJECT_ROOT/data/videos"
        "$PROJECT_ROOT/logs"
        "$PROJECT_ROOT/models"
        "$PROJECT_ROOT/backups"
        "$PROJECT_ROOT/backups/db"
    )
    
    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log_info "Created directory: $dir"
        fi
    done
    
    log_info "✓ All directories ready"
}

# Setup database
setup_database() {
    log_step "Setting up database..."
    
    read -p "Do you want to setup the database now? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Skipping database setup"
        return 0
    fi
    
    # Check if PostgreSQL is running
    if command_exists psql; then
        log_info "PostgreSQL client found"
        
        # Try to connect to database
        if psql -lqt | cut -d \| -f 1 | grep -qw face_recognition 2>/dev/null; then
            log_info "Database 'face_recognition' already exists"
        else
            log_warn "Database 'face_recognition' not found"
            log_info "Please create it manually or use Docker Compose"
        fi
    else
        log_warn "PostgreSQL client not found"
        log_info "You can use Docker Compose to run PostgreSQL:"
        log_info "  docker-compose -f docker-compose.dev.yml up -d postgres"
    fi
}

# Run database migrations
run_migrations() {
    log_step "Running database migrations..."
    
    read -p "Do you want to run database migrations now? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Skipping migrations"
        return 0
    fi
    
    source "$VENV_DIR/bin/activate"
    
    if [ -f "$PROJECT_ROOT/scripts/migrate_database.py" ]; then
        python "$PROJECT_ROOT/scripts/migrate_database.py" migrate
        log_info "✓ Migrations completed"
    else
        log_warn "Migration script not found"
    fi
}

# Download ML models
download_models() {
    log_step "Checking ML models..."
    
    if [ -d "$PROJECT_ROOT/models" ] && [ "$(ls -A $PROJECT_ROOT/models)" ]; then
        log_info "Model directory is not empty"
        return 0
    fi
    
    log_warn "Model directory is empty"
    log_info "Please download the required models:"
    log_info "  - Detection model (YOLOv8 or MTCNN)"
    log_info "  - Recognition model (FaceNet or ArcFace)"
    log_info ""
    log_info "Refer to docs/MODEL_VERSIONING.md for details"
}

# Run tests
run_tests() {
    log_step "Running tests..."
    
    read -p "Do you want to run tests now? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Skipping tests"
        return 0
    fi
    
    source "$VENV_DIR/bin/activate"
    
    if command_exists pytest; then
        pytest tests/ -v --tb=short || log_warn "Some tests failed"
        log_info "✓ Tests completed"
    else
        log_warn "pytest not found, skipping tests"
    fi
}

# Setup pre-commit hooks
setup_git_hooks() {
    log_step "Setting up git hooks..."
    
    if [ ! -d "$PROJECT_ROOT/.git" ]; then
        log_warn "Not a git repository, skipping hooks"
        return 0
    fi
    
    read -p "Do you want to setup git hooks? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Skipping git hooks"
        return 0
    fi
    
    source "$VENV_DIR/bin/activate"
    
    if command_exists pre-commit; then
        pre-commit install
        log_info "✓ Git hooks installed"
    else
        log_warn "pre-commit not found, skipping hooks"
    fi
}

# Print completion message
print_completion() {
    echo ""
    echo "=========================================="
    echo "  Setup Complete! 🎉"
    echo "=========================================="
    echo ""
    log_info "To activate the virtual environment, run:"
    echo "  source .venv/bin/activate"
    echo ""
    log_info "To start the development server:"
    echo "  python src/api_server.py"
    echo ""
    log_info "Or use Docker Compose:"
    echo "  docker-compose -f docker-compose.dev.yml up"
    echo ""
    log_info "Next steps:"
    echo "  1. Review and update .env file"
    echo "  2. Download ML models"
    echo "  3. Run database migrations"
    echo "  4. Start developing!"
    echo ""
    log_info "For more information, see:"
    echo "  - README.md"
    echo "  - docs/DEPLOYMENT.md"
    echo "  - docs/DATABASE_MIGRATION.md"
    echo ""
}

# Main execution
main() {
    print_header
    
    cd "$PROJECT_ROOT"
    
    check_requirements
    setup_venv
    install_dependencies
    setup_env_file
    create_directories
    setup_database
    run_migrations
    download_models
    run_tests
    setup_git_hooks
    
    print_completion
}

# Run main function
main "$@"
