"""
System information endpoint for debugging and monitoring.

Provides detailed system metrics and application info.
"""

import platform
import sys
import psutil
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, Depends
from pydantic import BaseModel


router = APIRouter()


class SystemInfo(BaseModel):
    """System information model."""
    
    hostname: str
    platform: str
    python_version: str
    cpu_count: int
    cpu_percent: float
    memory_total_gb: float
    memory_available_gb: float
    memory_percent: float
    disk_total_gb: float
    disk_used_gb: float
    disk_percent: float
    uptime_seconds: float


class ProcessInfo(BaseModel):
    """Process information model."""
    
    pid: int
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    num_threads: int
    num_fds: int
    create_time: str


class ApplicationInfo(BaseModel):
    """Application information model."""
    
    name: str = "Face Recognition API"
    version: str = "1.0.0"
    python_version: str
    environment: str = "development"


# Track application start time
_start_time = datetime.utcnow()


def get_system_metrics() -> Dict[str, Any]:
    """
    Get current system metrics.
    
    Returns:
        Dictionary of system metrics
    """
    # CPU info
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    
    # Memory info
    memory = psutil.virtual_memory()
    memory_total = memory.total / (1024 ** 3)  # Convert to GB
    memory_available = memory.available / (1024 ** 3)
    memory_percent = memory.percent
    
    # Disk info
    disk = psutil.disk_usage('/')
    disk_total = disk.total / (1024 ** 3)
    disk_used = disk.used / (1024 ** 3)
    disk_percent = disk.percent
    
    # Uptime
    uptime = (datetime.utcnow() - _start_time).total_seconds()
    
    return {
        "hostname": platform.node(),
        "platform": f"{platform.system()} {platform.release()}",
        "python_version": sys.version,
        "cpu_count": cpu_count,
        "cpu_percent": cpu_percent,
        "memory_total_gb": round(memory_total, 2),
        "memory_available_gb": round(memory_available, 2),
        "memory_percent": memory_percent,
        "disk_total_gb": round(disk_total, 2),
        "disk_used_gb": round(disk_used, 2),
        "disk_percent": disk_percent,
        "uptime_seconds": uptime,
    }


def get_process_metrics() -> Dict[str, Any]:
    """
    Get current process metrics.
    
    Returns:
        Dictionary of process metrics
    """
    process = psutil.Process()
    
    # Memory info
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / (1024 ** 2)
    
    # File descriptors (Unix only)
    try:
        num_fds = process.num_fds()
    except AttributeError:
        num_fds = 0
    
    return {
        "pid": process.pid,
        "cpu_percent": process.cpu_percent(interval=0.1),
        "memory_percent": process.memory_percent(),
        "memory_mb": round(memory_mb, 2),
        "num_threads": process.num_threads(),
        "num_fds": num_fds,
        "create_time": datetime.fromtimestamp(process.create_time()).isoformat(),
    }


@router.get("/system/info", response_model=SystemInfo)
async def system_info() -> Dict[str, Any]:
    """
    Get system information.
    
    Returns:
        System metrics
    """
    return get_system_metrics()


@router.get("/system/process", response_model=ProcessInfo)
async def process_info() -> Dict[str, Any]:
    """
    Get process information.
    
    Returns:
        Process metrics
    """
    return get_process_metrics()


@router.get("/system/application")
async def application_info() -> Dict[str, Any]:
    """
    Get application information.
    
    Returns:
        Application details
    """
    return {
        "name": "Face Recognition API",
        "version": "1.0.0",
        "python_version": sys.version,
        "environment": "development",
        "start_time": _start_time.isoformat(),
        "current_time": datetime.utcnow().isoformat(),
    }


@router.get("/system/debug")
async def debug_info() -> Dict[str, Any]:
    """
    Get comprehensive debug information.
    
    Returns:
        Combined system, process, and application info
    """
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "system": get_system_metrics(),
        "process": get_process_metrics(),
        "application": {
            "name": "Face Recognition API",
            "version": "1.0.0",
            "python_version": sys.version,
            "start_time": _start_time.isoformat(),
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "path": sys.path[:5],  # First 5 paths
        },
        "environment": {
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
        }
    }


def include_system_routes(app):
    """
    Include system info routes in FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    app.include_router(router, tags=["system"])


# Example usage in api_server.py:
"""
from src.system_info import include_system_routes

app = FastAPI()
include_system_routes(app)

# Access at:
# GET /system/info
# GET /system/process
# GET /system/application
# GET /system/debug
"""
