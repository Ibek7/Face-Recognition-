"""
Database connection health check for monitoring.

Monitors database connection pool and query health.
"""

import asyncio
from typing import Dict, Optional
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy import text
import logging

logger = logging.getLogger(__name__)


class DatabaseHealthCheck:
    """Monitor database connection health."""
    
    def __init__(
        self,
        engine: AsyncEngine,
        timeout: int = 5,
        check_interval: int = 60
    ):
        """
        Initialize health checker.
        
        Args:
            engine: SQLAlchemy async engine
            timeout: Query timeout in seconds
            check_interval: Health check interval in seconds
        """
        self.engine = engine
        self.timeout = timeout
        self.check_interval = check_interval
        self.last_check: Optional[datetime] = None
        self.last_status: Optional[bool] = None
        self.consecutive_failures = 0
        self._running = False
    
    async def check_connection(self) -> Dict[str, any]:
        """
        Check database connection.
        
        Returns:
            Health check results
        """
        start_time = datetime.utcnow()
        
        try:
            # Execute simple query with timeout
            async with self.engine.begin() as conn:
                result = await asyncio.wait_for(
                    conn.execute(text("SELECT 1")),
                    timeout=self.timeout
                )
                await result.close()
            
            response_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Get pool statistics
            pool_status = self._get_pool_stats()
            
            # Update status
            self.last_check = datetime.utcnow()
            self.last_status = True
            self.consecutive_failures = 0
            
            return {
                "status": "healthy",
                "timestamp": self.last_check.isoformat(),
                "response_time_ms": round(response_time * 1000, 2),
                "pool": pool_status,
                "consecutive_failures": 0
            }
        
        except asyncio.TimeoutError:
            self.consecutive_failures += 1
            logger.error(f"Database health check timeout after {self.timeout}s")
            
            return {
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": "Query timeout",
                "consecutive_failures": self.consecutive_failures
            }
        
        except Exception as e:
            self.consecutive_failures += 1
            logger.error(f"Database health check failed: {e}")
            
            return {
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "consecutive_failures": self.consecutive_failures
            }
    
    def _get_pool_stats(self) -> Dict[str, int]:
        """
        Get connection pool statistics.
        
        Returns:
            Pool statistics
        """
        pool = self.engine.pool
        
        return {
            "size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "total": pool.size() + pool.overflow()
        }
    
    async def start_monitoring(self):
        """Start periodic health monitoring."""
        if self._running:
            logger.warning("Health monitoring already running")
            return
        
        self._running = True
        logger.info(
            f"Starting database health monitoring "
            f"(interval: {self.check_interval}s)"
        )
        
        while self._running:
            try:
                result = await self.check_connection()
                
                if result["status"] == "unhealthy":
                    logger.warning(
                        f"Database unhealthy: {result.get('error', 'Unknown')}"
                    )
                
                # Alert on threshold
                if self.consecutive_failures >= 3:
                    logger.critical(
                        f"Database critical: {self.consecutive_failures} "
                        "consecutive failures"
                    )
                
                await asyncio.sleep(self.check_interval)
            
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def stop_monitoring(self):
        """Stop health monitoring."""
        if not self._running:
            return
        
        self._running = False
        logger.info("Stopped database health monitoring")
    
    def is_healthy(self, max_age_seconds: int = 120) -> bool:
        """
        Check if database is healthy.
        
        Args:
            max_age_seconds: Max age of last check
        
        Returns:
            True if healthy
        """
        if not self.last_check or not self.last_status:
            return False
        
        # Check if last check is recent
        age = (datetime.utcnow() - self.last_check).total_seconds()
        
        if age > max_age_seconds:
            return False
        
        return self.consecutive_failures == 0
    
    async def wait_for_healthy(
        self,
        timeout: int = 30,
        retry_interval: int = 2
    ) -> bool:
        """
        Wait for database to become healthy.
        
        Args:
            timeout: Maximum wait time
            retry_interval: Retry interval
        
        Returns:
            True if healthy
        """
        deadline = datetime.utcnow() + timedelta(seconds=timeout)
        
        while datetime.utcnow() < deadline:
            result = await self.check_connection()
            
            if result["status"] == "healthy":
                return True
            
            await asyncio.sleep(retry_interval)
        
        return False


# Example usage in api_server.py:
"""
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import create_async_engine
from src.db_health_check import DatabaseHealthCheck

app = FastAPI()

# Create engine
engine = create_async_engine(
    "postgresql+asyncpg://user:pass@localhost/db"
)

# Create health checker
db_health = DatabaseHealthCheck(
    engine=engine,
    timeout=5,
    check_interval=60
)

@app.on_event("startup")
async def startup():
    # Start monitoring
    asyncio.create_task(db_health.start_monitoring())
    
    # Wait for healthy connection
    is_healthy = await db_health.wait_for_healthy(timeout=30)
    
    if not is_healthy:
        raise RuntimeError("Database not healthy at startup")

@app.on_event("shutdown")
async def shutdown():
    # Stop monitoring
    await db_health.stop_monitoring()

@app.get("/health/db")
async def health_check():
    result = await db_health.check_connection()
    
    if result["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=result)
    
    return result

# Check if healthy
if db_health.is_healthy():
    print("Database is healthy")
"""
