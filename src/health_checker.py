"""
Health check system for monitoring service dependencies.

Provides health checks for databases, APIs, and other services.
"""

from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import httpx
import logging

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health check status."""
    
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CheckResult:
    """Health check result."""
    
    def __init__(
        self,
        name: str,
        status: HealthStatus,
        message: Optional[str] = None,
        response_time: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize check result.
        
        Args:
            name: Check name
            status: Health status
            message: Status message
            response_time: Response time in seconds
            metadata: Additional metadata
        """
        self.name = name
        self.status = status
        self.message = message
        self.response_time = response_time
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "response_time": self.response_time,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class HealthCheck:
    """Health check definition."""
    
    def __init__(
        self,
        name: str,
        check_func: Callable,
        critical: bool = True,
        timeout: float = 5.0,
        interval: int = 60
    ):
        """
        Initialize health check.
        
        Args:
            name: Check name
            check_func: Async check function
            critical: Is check critical
            timeout: Timeout in seconds
            interval: Check interval in seconds
        """
        self.name = name
        self.check_func = check_func
        self.critical = critical
        self.timeout = timeout
        self.interval = interval
        self.last_result: Optional[CheckResult] = None
        self.last_check: Optional[datetime] = None
    
    async def run(self) -> CheckResult:
        """Run health check."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Run check with timeout
            await asyncio.wait_for(
                self.check_func(),
                timeout=self.timeout
            )
            
            response_time = asyncio.get_event_loop().time() - start_time
            
            result = CheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="Check passed",
                response_time=response_time
            )
        
        except asyncio.TimeoutError:
            result = CheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check timed out after {self.timeout}s"
            )
        
        except Exception as e:
            result = CheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}"
            )
        
        self.last_result = result
        self.last_check = datetime.utcnow()
        
        return result


class DatabaseHealthCheck:
    """Database health check."""
    
    @staticmethod
    async def check_postgres(connection_url: str) -> None:
        """Check PostgreSQL connection."""
        import asyncpg
        
        conn = await asyncpg.connect(connection_url)
        try:
            await conn.execute("SELECT 1")
        finally:
            await conn.close()
    
    @staticmethod
    async def check_redis(host: str, port: int = 6379) -> None:
        """Check Redis connection."""
        import redis.asyncio as aioredis
        
        client = aioredis.Redis(host=host, port=port)
        try:
            await client.ping()
        finally:
            await client.close()
    
    @staticmethod
    async def check_mongodb(connection_url: str) -> None:
        """Check MongoDB connection."""
        from motor.motor_asyncio import AsyncIOMotorClient
        
        client = AsyncIOMotorClient(connection_url)
        try:
            await client.admin.command("ping")
        finally:
            client.close()


class ApiHealthCheck:
    """API health check."""
    
    @staticmethod
    async def check_http_endpoint(
        url: str,
        method: str = "GET",
        expected_status: int = 200,
        timeout: float = 5.0
    ) -> None:
        """
        Check HTTP endpoint.
        
        Args:
            url: Endpoint URL
            method: HTTP method
            expected_status: Expected status code
            timeout: Request timeout
        
        Raises:
            Exception: If check fails
        """
        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=method,
                url=url,
                timeout=timeout
            )
            
            if response.status_code != expected_status:
                raise Exception(
                    f"Expected status {expected_status}, got {response.status_code}"
                )


class SystemHealthCheck:
    """System resource health check."""
    
    @staticmethod
    async def check_disk_space(
        path: str = "/",
        min_free_gb: float = 1.0
    ) -> None:
        """
        Check disk space.
        
        Args:
            path: Path to check
            min_free_gb: Minimum free GB required
        
        Raises:
            Exception: If insufficient space
        """
        import shutil
        
        stat = shutil.disk_usage(path)
        free_gb = stat.free / (1024 ** 3)
        
        if free_gb < min_free_gb:
            raise Exception(
                f"Low disk space: {free_gb:.2f}GB free, need {min_free_gb}GB"
            )
    
    @staticmethod
    async def check_memory(min_free_mb: float = 100.0) -> None:
        """
        Check available memory.
        
        Args:
            min_free_mb: Minimum free MB required
        
        Raises:
            Exception: If insufficient memory
        """
        import psutil
        
        memory = psutil.virtual_memory()
        free_mb = memory.available / (1024 ** 2)
        
        if free_mb < min_free_mb:
            raise Exception(
                f"Low memory: {free_mb:.2f}MB free, need {min_free_mb}MB"
            )


class HealthChecker:
    """Manage health checks."""
    
    def __init__(self):
        """Initialize health checker."""
        self.checks: Dict[str, HealthCheck] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    def register_check(
        self,
        name: str,
        check_func: Callable,
        critical: bool = True,
        timeout: float = 5.0,
        interval: int = 60
    ):
        """
        Register health check.
        
        Args:
            name: Check name
            check_func: Async check function
            critical: Is check critical
            timeout: Timeout in seconds
            interval: Check interval in seconds
        """
        check = HealthCheck(
            name=name,
            check_func=check_func,
            critical=critical,
            timeout=timeout,
            interval=interval
        )
        
        self.checks[name] = check
        logger.info(f"Registered health check: {name}")
    
    async def run_check(self, name: str) -> CheckResult:
        """Run specific health check."""
        if name not in self.checks:
            raise ValueError(f"Health check not found: {name}")
        
        check = self.checks[name]
        return await check.run()
    
    async def run_all_checks(self) -> Dict[str, CheckResult]:
        """Run all health checks."""
        results = {}
        
        tasks = [check.run() for check in self.checks.values()]
        check_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for check_name, result in zip(self.checks.keys(), check_results):
            if isinstance(result, Exception):
                results[check_name] = CheckResult(
                    name=check_name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check error: {str(result)}"
                )
            else:
                results[check_name] = result
        
        return results
    
    async def get_overall_health(self) -> Dict[str, Any]:
        """Get overall health status."""
        results = await self.run_all_checks()
        
        # Determine overall status
        critical_unhealthy = any(
            r.status == HealthStatus.UNHEALTHY and self.checks[r.name].critical
            for r in results.values()
        )
        
        any_unhealthy = any(
            r.status == HealthStatus.UNHEALTHY
            for r in results.values()
        )
        
        if critical_unhealthy:
            overall_status = HealthStatus.UNHEALTHY
        elif any_unhealthy:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        return {
            "status": overall_status.value,
            "checks": {name: result.to_dict() for name, result in results.items()},
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def start_monitoring(self):
        """Start continuous monitoring."""
        if self._running:
            logger.warning("Health monitoring already running")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("Started health monitoring")
    
    async def stop_monitoring(self):
        """Stop continuous monitoring."""
        if not self._running:
            return
        
        self._running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped health monitoring")
    
    async def _monitor_loop(self):
        """Continuous monitoring loop."""
        while self._running:
            try:
                await self.run_all_checks()
                
                # Wait for next check interval
                min_interval = min(
                    check.interval for check in self.checks.values()
                )
                await asyncio.sleep(min_interval)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(60)


# Example usage:
"""
from fastapi import FastAPI
from src.health_checker import HealthChecker, DatabaseHealthCheck, ApiHealthCheck

app = FastAPI()
health_checker = HealthChecker()

@app.on_event("startup")
async def startup():
    # Register checks
    health_checker.register_check(
        name="postgres",
        check_func=lambda: DatabaseHealthCheck.check_postgres("postgresql://..."),
        critical=True
    )
    
    health_checker.register_check(
        name="redis",
        check_func=lambda: DatabaseHealthCheck.check_redis("localhost"),
        critical=True
    )
    
    health_checker.register_check(
        name="external_api",
        check_func=lambda: ApiHealthCheck.check_http_endpoint("https://api.example.com/health"),
        critical=False
    )
    
    # Start monitoring
    health_checker.start_monitoring()

@app.get("/health")
async def health():
    return await health_checker.get_overall_health()

@app.on_event("shutdown")
async def shutdown():
    await health_checker.stop_monitoring()
"""
