#!/usr/bin/env python3
"""
Comprehensive Health Check System

Provides detailed health checks for all system components:
- API server health
- Database connectivity
- Redis cache
- ML model availability
- External dependencies
- System resources (CPU, memory, disk)
"""

import asyncio
import time
import psutil
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum

from fastapi import FastAPI, Response
from pydantic import BaseModel
import redis.asyncio as aioredis
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import create_async_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health check status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseModel):
    """Individual component health status"""
    name: str
    status: HealthStatus
    message: Optional[str] = None
    latency_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.utcnow()
        super().__init__(**data)


class HealthCheckResponse(BaseModel):
    """Overall health check response"""
    status: HealthStatus
    version: str
    uptime_seconds: float
    timestamp: datetime
    components: List[ComponentHealth]
    
    @property
    def is_healthy(self) -> bool:
        return self.status == HealthStatus.HEALTHY


class HealthChecker:
    """System health checker"""
    
    def __init__(
        self,
        database_url: Optional[str] = None,
        redis_url: Optional[str] = None,
        version: str = "1.0.0"
    ):
        self.database_url = database_url
        self.redis_url = redis_url
        self.version = version
        self.start_time = time.time()
        
        # Component checkers
        self.checkers = [
            self.check_api,
            self.check_database,
            self.check_redis,
            self.check_cpu,
            self.check_memory,
            self.check_disk,
            self.check_models
        ]
    
    async def check_health(self, detailed: bool = True) -> HealthCheckResponse:
        """
        Perform comprehensive health check
        
        Args:
            detailed: Include detailed component checks
        
        Returns:
            HealthCheckResponse with overall status
        """
        components = []
        
        if detailed:
            # Run all component checks in parallel
            results = await asyncio.gather(
                *[checker() for checker in self.checkers],
                return_exceptions=True
            )
            
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Health check error: {result}")
                    components.append(
                        ComponentHealth(
                            name="unknown",
                            status=HealthStatus.UNHEALTHY,
                            message=str(result)
                        )
                    )
                else:
                    components.append(result)
        else:
            # Quick check - just API
            api_check = await self.check_api()
            components.append(api_check)
        
        # Determine overall status
        overall_status = self._determine_overall_status(components)
        
        return HealthCheckResponse(
            status=overall_status,
            version=self.version,
            uptime_seconds=time.time() - self.start_time,
            timestamp=datetime.utcnow(),
            components=components
        )
    
    def _determine_overall_status(
        self,
        components: List[ComponentHealth]
    ) -> HealthStatus:
        """Determine overall system health"""
        if not components:
            return HealthStatus.UNHEALTHY
        
        # Count component statuses
        unhealthy_count = sum(
            1 for c in components if c.status == HealthStatus.UNHEALTHY
        )
        degraded_count = sum(
            1 for c in components if c.status == HealthStatus.DEGRADED
        )
        
        # If any critical component is unhealthy
        if unhealthy_count > 0:
            return HealthStatus.UNHEALTHY
        
        # If any component is degraded
        if degraded_count > 0:
            return HealthStatus.DEGRADED
        
        # All healthy
        return HealthStatus.HEALTHY
    
    async def check_api(self) -> ComponentHealth:
        """Check API server health"""
        try:
            start_time = time.perf_counter()
            
            # Simple check - API is running if we got here
            await asyncio.sleep(0.001)  # Simulate minimal work
            
            latency = (time.perf_counter() - start_time) * 1000
            
            return ComponentHealth(
                name="api",
                status=HealthStatus.HEALTHY,
                message="API server running",
                latency_ms=latency
            )
        except Exception as e:
            return ComponentHealth(
                name="api",
                status=HealthStatus.UNHEALTHY,
                message=f"API check failed: {e}"
            )
    
    async def check_database(self) -> ComponentHealth:
        """Check database connectivity"""
        if not self.database_url:
            return ComponentHealth(
                name="database",
                status=HealthStatus.HEALTHY,
                message="Database not configured"
            )
        
        try:
            start_time = time.perf_counter()
            
            # Create engine and test connection
            engine = create_async_engine(
                self.database_url,
                pool_pre_ping=True,
                pool_size=1
            )
            
            async with engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            
            await engine.dispose()
            
            latency = (time.perf_counter() - start_time) * 1000
            
            return ComponentHealth(
                name="database",
                status=HealthStatus.HEALTHY,
                message="Database connected",
                latency_ms=latency
            )
        except Exception as e:
            return ComponentHealth(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {e}"
            )
    
    async def check_redis(self) -> ComponentHealth:
        """Check Redis connectivity"""
        if not self.redis_url:
            return ComponentHealth(
                name="redis",
                status=HealthStatus.HEALTHY,
                message="Redis not configured"
            )
        
        try:
            start_time = time.perf_counter()
            
            # Test Redis connection
            redis = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            
            await redis.ping()
            
            # Get some stats
            info = await redis.info()
            
            await redis.close()
            
            latency = (time.perf_counter() - start_time) * 1000
            
            return ComponentHealth(
                name="redis",
                status=HealthStatus.HEALTHY,
                message="Redis connected",
                latency_ms=latency,
                details={
                    "connected_clients": info.get("connected_clients"),
                    "used_memory_human": info.get("used_memory_human")
                }
            )
        except Exception as e:
            return ComponentHealth(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                message=f"Redis connection failed: {e}"
            )
    
    async def check_cpu(self) -> ComponentHealth:
        """Check CPU usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            
            # Determine status based on CPU usage
            if cpu_percent < 70:
                status = HealthStatus.HEALTHY
                message = "CPU usage normal"
            elif cpu_percent < 90:
                status = HealthStatus.DEGRADED
                message = "CPU usage high"
            else:
                status = HealthStatus.UNHEALTHY
                message = "CPU usage critical"
            
            return ComponentHealth(
                name="cpu",
                status=status,
                message=message,
                details={
                    "usage_percent": cpu_percent,
                    "cpu_count": cpu_count
                }
            )
        except Exception as e:
            return ComponentHealth(
                name="cpu",
                status=HealthStatus.UNHEALTHY,
                message=f"CPU check failed: {e}"
            )
    
    async def check_memory(self) -> ComponentHealth:
        """Check memory usage"""
        try:
            memory = psutil.virtual_memory()
            
            # Determine status based on memory usage
            if memory.percent < 70:
                status = HealthStatus.HEALTHY
                message = "Memory usage normal"
            elif memory.percent < 90:
                status = HealthStatus.DEGRADED
                message = "Memory usage high"
            else:
                status = HealthStatus.UNHEALTHY
                message = "Memory usage critical"
            
            return ComponentHealth(
                name="memory",
                status=status,
                message=message,
                details={
                    "usage_percent": memory.percent,
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "used_gb": round(memory.used / (1024**3), 2)
                }
            )
        except Exception as e:
            return ComponentHealth(
                name="memory",
                status=HealthStatus.UNHEALTHY,
                message=f"Memory check failed: {e}"
            )
    
    async def check_disk(self) -> ComponentHealth:
        """Check disk usage"""
        try:
            disk = psutil.disk_usage('/')
            
            # Determine status based on disk usage
            if disk.percent < 70:
                status = HealthStatus.HEALTHY
                message = "Disk usage normal"
            elif disk.percent < 90:
                status = HealthStatus.DEGRADED
                message = "Disk usage high"
            else:
                status = HealthStatus.UNHEALTHY
                message = "Disk usage critical"
            
            return ComponentHealth(
                name="disk",
                status=status,
                message=message,
                details={
                    "usage_percent": disk.percent,
                    "total_gb": round(disk.total / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "used_gb": round(disk.used / (1024**3), 2)
                }
            )
        except Exception as e:
            return ComponentHealth(
                name="disk",
                status=HealthStatus.UNHEALTHY,
                message=f"Disk check failed: {e}"
            )
    
    async def check_models(self) -> ComponentHealth:
        """Check ML models availability"""
        try:
            from pathlib import Path
            
            model_path = Path("models")
            
            if not model_path.exists():
                return ComponentHealth(
                    name="models",
                    status=HealthStatus.DEGRADED,
                    message="Model directory not found"
                )
            
            # Check for expected models
            expected_models = ["yolov8n-face.pt", "facenet512.pth"]
            found_models = [
                m.name for m in model_path.glob("*.pt*") 
                if m.is_file()
            ]
            
            if len(found_models) >= len(expected_models):
                status = HealthStatus.HEALTHY
                message = "All models available"
            elif found_models:
                status = HealthStatus.DEGRADED
                message = "Some models missing"
            else:
                status = HealthStatus.UNHEALTHY
                message = "No models found"
            
            return ComponentHealth(
                name="models",
                status=status,
                message=message,
                details={
                    "found_models": found_models,
                    "model_count": len(found_models)
                }
            )
        except Exception as e:
            return ComponentHealth(
                name="models",
                status=HealthStatus.UNHEALTHY,
                message=f"Model check failed: {e}"
            )


def setup_health_endpoints(app: FastAPI, checker: HealthChecker):
    """Setup health check endpoints"""
    
    @app.get("/health", response_model=HealthCheckResponse)
    async def health_check():
        """Basic health check"""
        return await checker.check_health(detailed=False)
    
    @app.get("/health/detailed", response_model=HealthCheckResponse)
    async def detailed_health_check():
        """Detailed health check with all components"""
        return await checker.check_health(detailed=True)
    
    @app.get("/healthz")
    async def kubernetes_health():
        """Kubernetes liveness probe"""
        result = await checker.check_health(detailed=False)
        
        if result.is_healthy:
            return Response(content="OK", status_code=200)
        else:
            return Response(content="UNHEALTHY", status_code=503)
    
    @app.get("/ready")
    async def kubernetes_readiness():
        """Kubernetes readiness probe"""
        result = await checker.check_health(detailed=True)
        
        # Check critical components
        critical_components = ["api", "database"]
        critical_healthy = all(
            c.status == HealthStatus.HEALTHY
            for c in result.components
            if c.name in critical_components
        )
        
        if critical_healthy:
            return Response(content="READY", status_code=200)
        else:
            return Response(content="NOT READY", status_code=503)


# Example usage
if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI
    
    app = FastAPI(title="Face Recognition API with Health Checks")
    
    # Initialize health checker
    checker = HealthChecker(
        database_url="postgresql+asyncpg://postgres:postgres@localhost:5432/face_recognition",
        redis_url="redis://localhost:6379/0",
        version="1.0.0"
    )
    
    # Setup health endpoints
    setup_health_endpoints(app, checker)
    
    @app.get("/")
    async def root():
        return {
            "message": "Face Recognition API",
            "health_endpoints": ["/health", "/health/detailed", "/healthz", "/ready"]
        }
    
    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8000)
