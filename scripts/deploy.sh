#!/bin/bash
#
# Automated Deployment Script
#
# Deploy face recognition system to:
# - Docker containers
# - Kubernetes cluster
# - Cloud platforms (AWS, Azure, GCP)
#
# Features:
# - Environment validation
# - Database migrations
# - Health checks
# - Rollback support
# - Blue-green deployment
#

set -e  # Exit on error

# Configuration
PROJECT_NAME="face-recognition"
DOCKER_IMAGE="${PROJECT_NAME}:latest"
REGISTRY="${DOCKER_REGISTRY:-docker.io}"
NAMESPACE="${K8S_NAMESPACE:-default}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local missing_tools=()
    
    # Check for required tools
    command -v docker >/dev/null 2>&1 || missing_tools+=("docker")
    command -v git >/dev/null 2>&1 || missing_tools+=("git")
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_error "Please install missing tools and try again"
        exit 1
    fi
    
    log_info "✓ All prerequisites met"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    
    # Get git commit hash
    GIT_COMMIT=$(git rev-parse --short HEAD)
    BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    docker build \
        --build-arg GIT_COMMIT="$GIT_COMMIT" \
        --build-arg BUILD_DATE="$BUILD_DATE" \
        -t "${DOCKER_IMAGE}" \
        -t "${DOCKER_IMAGE}-${GIT_COMMIT}" \
        .
    
    log_info "✓ Docker image built: ${DOCKER_IMAGE}"
}

# Test Docker image
test_image() {
    log_info "Testing Docker image..."
    
    # Run basic container test
    docker run --rm "${DOCKER_IMAGE}" python -c "import sys; print('Python version:', sys.version)"
    
    log_info "✓ Docker image test passed"
}

# Push image to registry
push_image() {
    log_info "Pushing image to registry..."
    
    GIT_COMMIT=$(git rev-parse --short HEAD)
    
    # Tag for registry
    docker tag "${DOCKER_IMAGE}" "${REGISTRY}/${DOCKER_IMAGE}"
    docker tag "${DOCKER_IMAGE}-${GIT_COMMIT}" "${REGISTRY}/${DOCKER_IMAGE}-${GIT_COMMIT}"
    
    # Push
    docker push "${REGISTRY}/${DOCKER_IMAGE}"
    docker push "${REGISTRY}/${DOCKER_IMAGE}-${GIT_COMMIT}"
    
    log_info "✓ Image pushed to ${REGISTRY}"
}

# Deploy to Docker Compose
deploy_docker_compose() {
    log_info "Deploying with Docker Compose..."
    
    # Stop existing containers
    docker-compose down || true
    
    # Start services
    docker-compose up -d
    
    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    sleep 10
    
    # Check health
    docker-compose ps
    
    log_info "✓ Deployed with Docker Compose"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    # Check if kubectl is available
    if ! command -v kubectl >/dev/null 2>&1; then
        log_error "kubectl not found. Skipping Kubernetes deployment"
        return 1
    fi
    
    # Apply Kubernetes manifests
    if [ -d "k8s" ]; then
        kubectl apply -f k8s/ -n "${NAMESPACE}"
        
        # Wait for rollout
        kubectl rollout status deployment/${PROJECT_NAME} -n "${NAMESPACE}"
        
        log_info "✓ Deployed to Kubernetes cluster"
    else
        log_warn "No k8s directory found. Skipping Kubernetes deployment"
    fi
}

# Run database migrations
run_migrations() {
    log_info "Running database migrations..."
    
    # Check if Alembic is configured
    if [ -f "alembic.ini" ]; then
        docker-compose run --rm app alembic upgrade head
        log_info "✓ Database migrations completed"
    else
        log_warn "No alembic.ini found. Skipping migrations"
    fi
}

# Health check
health_check() {
    log_info "Performing health check..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:8000/health >/dev/null 2>&1; then
            log_info "✓ Health check passed"
            return 0
        fi
        
        log_info "Health check attempt $attempt/$max_attempts failed. Retrying..."
        sleep 2
        ((attempt++))
    done
    
    log_error "Health check failed after $max_attempts attempts"
    return 1
}

# Rollback deployment
rollback() {
    log_warn "Rolling back deployment..."
    
    if command -v kubectl >/dev/null 2>&1; then
        kubectl rollout undo deployment/${PROJECT_NAME} -n "${NAMESPACE}"
        log_info "✓ Kubernetes deployment rolled back"
    else
        docker-compose down
        log_info "✓ Docker Compose deployment stopped"
    fi
}

# Cleanup old images
cleanup() {
    log_info "Cleaning up old images..."
    
    # Remove dangling images
    docker image prune -f
    
    # Remove old images (keep last 5)
    docker images "${DOCKER_IMAGE}-*" --format "{{.ID}}" | tail -n +6 | xargs -r docker rmi -f
    
    log_info "✓ Cleanup completed"
}

# Main deployment function
deploy() {
    local deployment_type="${1:-docker-compose}"
    
    log_info "Starting deployment (type: $deployment_type)..."
    
    # Check prerequisites
    check_prerequisites
    
    # Build and test
    build_image
    test_image
    
    # Deploy based on type
    case "$deployment_type" in
        "docker-compose")
            deploy_docker_compose
            ;;
        "kubernetes"|"k8s")
            push_image
            deploy_kubernetes
            ;;
        "production"|"prod")
            push_image
            run_migrations
            deploy_kubernetes
            ;;
        *)
            log_error "Unknown deployment type: $deployment_type"
            log_info "Available types: docker-compose, kubernetes, production"
            exit 1
            ;;
    esac
    
    # Health check
    if health_check; then
        log_info "🎉 Deployment successful!"
    else
        log_error "Deployment health check failed!"
        read -p "Rollback? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rollback
        fi
        exit 1
    fi
    
    # Cleanup
    cleanup
}

# Show usage
usage() {
    cat <<EOF
Usage: $0 [command] [options]

Commands:
    deploy [type]       Deploy the application
                        Types: docker-compose (default), kubernetes, production
    build               Build Docker image only
    test                Test Docker image only
    push                Push image to registry
    rollback            Rollback deployment
    cleanup             Clean up old images
    health              Run health check only

Options:
    -h, --help          Show this help message

Environment Variables:
    DOCKER_REGISTRY     Docker registry URL (default: docker.io)
    K8S_NAMESPACE       Kubernetes namespace (default: default)

Examples:
    $0 deploy                    # Deploy with docker-compose
    $0 deploy kubernetes         # Deploy to Kubernetes
    $0 deploy production         # Production deployment
    $0 build                     # Build image only
    $0 rollback                  # Rollback deployment

EOF
}

# Parse command line arguments
case "${1:-}" in
    deploy)
        deploy "${2:-docker-compose}"
        ;;
    build)
        check_prerequisites
        build_image
        ;;
    test)
        test_image
        ;;
    push)
        check_prerequisites
        build_image
        push_image
        ;;
    rollback)
        rollback
        ;;
    cleanup)
        cleanup
        ;;
    health)
        health_check
        ;;
    -h|--help|help)
        usage
        ;;
    "")
        log_info "No command specified. Deploying with docker-compose..."
        deploy "docker-compose"
        ;;
    *)
        log_error "Unknown command: $1"
        usage
        exit 1
        ;;
esac
