"""
Connection pool manager for external services.

Provides connection pooling, health checks, and failover for HTTP clients.
"""

import asyncio
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta
import httpx
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ConnectionStatus(str, Enum):
    """Connection status."""
    
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class PooledConnection:
    """Pooled HTTP connection with health tracking."""
    
    def __init__(
        self,
        name: str,
        base_url: str,
        timeout: int = 30,
        max_retries: int = 3,
        pool_size: int = 10
    ):
        """
        Initialize pooled connection.
        
        Args:
            name: Connection name
            base_url: Base URL for requests
            timeout: Request timeout
            max_retries: Maximum retry attempts
            pool_size: Connection pool size
        """
        self.name = name
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.pool_size = pool_size
        
        # Create HTTP client with connection pooling
        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=pool_size,
                max_keepalive_connections=pool_size // 2
            )
        )
        
        # Health tracking
        self.status = ConnectionStatus.UNKNOWN
        self.last_check: Optional[datetime] = None
        self.consecutive_failures = 0
        self.total_requests = 0
        self.failed_requests = 0
        self.avg_response_time = 0.0
    
    async def get(self, path: str, **kwargs) -> httpx.Response:
        """Make GET request."""
        return await self._request("GET", path, **kwargs)
    
    async def post(self, path: str, **kwargs) -> httpx.Response:
        """Make POST request."""
        return await self._request("POST", path, **kwargs)
    
    async def put(self, path: str, **kwargs) -> httpx.Response:
        """Make PUT request."""
        return await self._request("PUT", path, **kwargs)
    
    async def delete(self, path: str, **kwargs) -> httpx.Response:
        """Make DELETE request."""
        return await self._request("DELETE", path, **kwargs)
    
    async def _request(
        self,
        method: str,
        path: str,
        **kwargs
    ) -> httpx.Response:
        """Make HTTP request with retries."""
        self.total_requests += 1
        start_time = datetime.utcnow()
        
        for attempt in range(self.max_retries):
            try:
                response = await self.client.request(method, path, **kwargs)
                
                # Update metrics
                duration = (datetime.utcnow() - start_time).total_seconds()
                self._update_metrics(duration, success=True)
                
                response.raise_for_status()
                return response
            
            except httpx.HTTPStatusError as e:
                # Don't retry client errors (4xx)
                if 400 <= e.response.status_code < 500:
                    self.failed_requests += 1
                    raise
                
                # Retry server errors (5xx)
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.failed_requests += 1
                    self._update_metrics(0, success=False)
                    raise
            
            except Exception as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    self.failed_requests += 1
                    self._update_metrics(0, success=False)
                    raise
        
        raise RuntimeError("Max retries exceeded")
    
    def _update_metrics(self, duration: float, success: bool):
        """Update connection metrics."""
        if success:
            # Update average response time (exponential moving average)
            alpha = 0.3
            self.avg_response_time = (
                alpha * duration +
                (1 - alpha) * self.avg_response_time
            )
            self.consecutive_failures = 0
            self.status = ConnectionStatus.HEALTHY
        else:
            self.consecutive_failures += 1
            
            if self.consecutive_failures >= 3:
                self.status = ConnectionStatus.UNHEALTHY
            elif self.consecutive_failures >= 1:
                self.status = ConnectionStatus.DEGRADED
    
    async def health_check(self, path: str = "/health") -> bool:
        """
        Perform health check.
        
        Args:
            path: Health check endpoint
        
        Returns:
            True if healthy
        """
        try:
            response = await self.client.get(path, timeout=5.0)
            self.last_check = datetime.utcnow()
            
            if response.status_code == 200:
                self.status = ConnectionStatus.HEALTHY
                self.consecutive_failures = 0
                return True
            else:
                self.status = ConnectionStatus.DEGRADED
                return False
        
        except Exception as e:
            logger.error(f"Health check failed for {self.name}: {e}")
            self.status = ConnectionStatus.UNHEALTHY
            self.consecutive_failures += 1
            return False
    
    async def close(self):
        """Close connection pool."""
        await self.client.aclose()
    
    def get_stats(self) -> dict:
        """Get connection statistics."""
        success_rate = (
            (self.total_requests - self.failed_requests) / self.total_requests * 100
            if self.total_requests > 0
            else 0
        )
        
        return {
            "name": self.name,
            "base_url": self.base_url,
            "status": self.status.value,
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "success_rate": round(success_rate, 2),
            "avg_response_time": round(self.avg_response_time, 3),
            "consecutive_failures": self.consecutive_failures,
            "last_check": self.last_check.isoformat() if self.last_check else None
        }


class ConnectionPool:
    """Manage multiple pooled connections."""
    
    def __init__(self, health_check_interval: int = 60):
        """
        Initialize connection pool manager.
        
        Args:
            health_check_interval: Health check interval (seconds)
        """
        self.connections: Dict[str, PooledConnection] = {}
        self.health_check_interval = health_check_interval
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False
    
    def add_connection(
        self,
        name: str,
        base_url: str,
        timeout: int = 30,
        max_retries: int = 3,
        pool_size: int = 10
    ):
        """
        Add connection to pool.
        
        Args:
            name: Unique connection name
            base_url: Base URL
            timeout: Request timeout
            max_retries: Max retry attempts
            pool_size: Pool size
        """
        connection = PooledConnection(
            name=name,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            pool_size=pool_size
        )
        
        self.connections[name] = connection
        logger.info(f"Added connection pool: {name} ({base_url})")
    
    def get_connection(self, name: str) -> PooledConnection:
        """
        Get connection by name.
        
        Args:
            name: Connection name
        
        Returns:
            Pooled connection
        
        Raises:
            ValueError: If connection not found
        """
        if name not in self.connections:
            raise ValueError(f"Connection not found: {name}")
        
        return self.connections[name]
    
    async def remove_connection(self, name: str):
        """Remove connection from pool."""
        if name in self.connections:
            await self.connections[name].close()
            del self.connections[name]
            logger.info(f"Removed connection pool: {name}")
    
    async def _health_check_loop(self):
        """Periodic health check loop."""
        while self._running:
            try:
                # Check all connections
                for connection in self.connections.values():
                    await connection.health_check()
                
                await asyncio.sleep(self.health_check_interval)
            
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def start_health_checks(self):
        """Start periodic health checks."""
        if self._running:
            return
        
        self._running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info("Started connection pool health checks")
    
    async def stop_health_checks(self):
        """Stop health checks."""
        if not self._running:
            return
        
        self._running = False
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped connection pool health checks")
    
    async def close_all(self):
        """Close all connections."""
        for connection in self.connections.values():
            await connection.close()
        
        self.connections.clear()
        logger.info("Closed all connection pools")
    
    def get_all_stats(self) -> List[dict]:
        """Get statistics for all connections."""
        return [
            conn.get_stats()
            for conn in self.connections.values()
        ]
    
    def get_healthy_connections(self) -> List[str]:
        """Get list of healthy connection names."""
        return [
            name
            for name, conn in self.connections.items()
            if conn.status == ConnectionStatus.HEALTHY
        ]


# Global connection pool
connection_pool = ConnectionPool()


# Example usage:
"""
from fastapi import FastAPI
from src.connection_pool import connection_pool

app = FastAPI()

@app.on_event("startup")
async def startup():
    # Add connections
    connection_pool.add_connection(
        name="api_service",
        base_url="https://api.example.com",
        timeout=30,
        pool_size=20
    )
    
    connection_pool.add_connection(
        name="auth_service",
        base_url="https://auth.example.com",
        timeout=10,
        pool_size=10
    )
    
    # Start health checks
    await connection_pool.start_health_checks()

@app.on_event("shutdown")
async def shutdown():
    await connection_pool.stop_health_checks()
    await connection_pool.close_all()

@app.get("/api/external-data")
async def get_external_data():
    # Get connection
    api_conn = connection_pool.get_connection("api_service")
    
    # Make request
    response = await api_conn.get("/data")
    
    return response.json()

@app.get("/pool/stats")
async def pool_stats():
    return connection_pool.get_all_stats()

@app.get("/pool/health")
async def pool_health():
    healthy = connection_pool.get_healthy_connections()
    return {
        "healthy_connections": healthy,
        "total_connections": len(connection_pool.connections)
    }
"""
