"""
Health check endpoints for FastAPI application.

Provides /health and /healthz endpoints for container orchestration
and load balancer health checks.
"""

from datetime import datetime
from typing import Dict, Any
import psutil
from fastapi import APIRouter, status
from pydantic import BaseModel


router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str
    timestamp: str
    uptime_seconds: float
    version: str = "1.0.0"


class DetailedHealthResponse(HealthResponse):
    """Detailed health check with system metrics."""
    
    cpu_percent: float
    memory_percent: float
    disk_percent: float


# Track application start time
_start_time = datetime.utcnow()


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint.
    
    Returns:
        Health status information
    """
    uptime = (datetime.utcnow() - _start_time).total_seconds()
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": uptime,
        "version": "1.0.0",
    }


@router.get("/healthz", status_code=status.HTTP_200_OK)
async def healthz() -> Dict[str, str]:
    """
    Kubernetes-style health check endpoint.
    
    Returns:
        Simple OK response
    """
    return {"status": "ok"}


@router.get("/health/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check() -> Dict[str, Any]:
    """
    Detailed health check with system metrics.
    
    Returns:
        Detailed health status with resource usage
    """
    uptime = (datetime.utcnow() - _start_time).total_seconds()
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": uptime,
        "version": "1.0.0",
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent,
    }


@router.get("/readiness")
async def readiness_check() -> Dict[str, Any]:
    """
    Kubernetes readiness probe endpoint.
    
    Returns:
        Readiness status
    """
    # Add checks for dependencies (database, cache, etc.)
    # For now, always return ready
    return {
        "ready": True,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/liveness")
async def liveness_check() -> Dict[str, str]:
    """
    Kubernetes liveness probe endpoint.
    
    Returns:
        Liveness status
    """
    return {"alive": "true"}


def include_health_routes(app):
    """
    Include health check routes in FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    app.include_router(router, tags=["health"])
