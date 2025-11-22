"""
Request timeout handler middleware for FastAPI.

Prevents long-running requests from consuming resources indefinitely.
"""

import asyncio
from typing import Callable, Optional
from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
import logging
import time

logger = logging.getLogger(__name__)


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Enforce request timeout limits."""
    
    def __init__(
        self,
        app,
        timeout: float = 30.0,
        exclude_paths: Optional[list] = None
    ):
        """
        Initialize timeout middleware.
        
        Args:
            app: FastAPI application
            timeout: Default timeout in seconds
            exclude_paths: Paths to exclude from timeout
        """
        super().__init__(app)
        self.timeout = timeout
        self.exclude_paths = exclude_paths or []
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """
        Process request with timeout.
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
        
        Returns:
            Response
        
        Raises:
            HTTPException: On timeout
        """
        # Check if path is excluded
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
        
        # Get custom timeout from header
        custom_timeout = request.headers.get("X-Request-Timeout")
        timeout = float(custom_timeout) if custom_timeout else self.timeout
        
        start_time = time.time()
        
        try:
            # Execute request with timeout
            response = await asyncio.wait_for(
                call_next(request),
                timeout=timeout
            )
            
            # Add processing time header
            processing_time = time.time() - start_time
            response.headers["X-Processing-Time"] = f"{processing_time:.3f}"
            
            return response
        
        except asyncio.TimeoutError:
            processing_time = time.time() - start_time
            
            logger.warning(
                f"Request timeout: {request.method} {request.url.path} "
                f"({processing_time:.1f}s)"
            )
            
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"Request timeout after {timeout}s"
            )


class TimeoutConfig:
    """Configure timeouts for different endpoints."""
    
    def __init__(self):
        """Initialize timeout configuration."""
        self.timeouts = {}
        self.default_timeout = 30.0
    
    def set_timeout(self, path: str, timeout: float):
        """
        Set timeout for specific path.
        
        Args:
            path: URL path or pattern
            timeout: Timeout in seconds
        """
        self.timeouts[path] = timeout
        logger.info(f"Set timeout for {path}: {timeout}s")
    
    def get_timeout(self, path: str) -> float:
        """
        Get timeout for path.
        
        Args:
            path: URL path
        
        Returns:
            Timeout in seconds
        """
        # Exact match
        if path in self.timeouts:
            return self.timeouts[path]
        
        # Pattern match
        for pattern, timeout in self.timeouts.items():
            if path.startswith(pattern):
                return timeout
        
        return self.default_timeout


async def with_timeout(coro, timeout: float = 30.0):
    """
    Execute coroutine with timeout.
    
    Args:
        coro: Coroutine to execute
        timeout: Timeout in seconds
    
    Returns:
        Coroutine result
    
    Raises:
        asyncio.TimeoutError: On timeout
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.error(f"Operation timeout after {timeout}s")
        raise


# Global timeout config
timeout_config = TimeoutConfig()


# Example usage in api_server.py:
"""
from fastapi import FastAPI
from src.timeout_handler import TimeoutMiddleware, timeout_config, with_timeout

app = FastAPI()

# Add timeout middleware
app.add_middleware(
    TimeoutMiddleware,
    timeout=30.0,  # 30 second default
    exclude_paths=["/health", "/metrics"]
)

# Configure custom timeouts
timeout_config.set_timeout("/api/batch", 120.0)  # 2 minutes
timeout_config.set_timeout("/api/upload", 60.0)  # 1 minute

@app.get("/api/data")
async def get_data():
    # Use timeout wrapper for specific operations
    async def fetch_data():
        await asyncio.sleep(5)
        return {"data": "result"}
    
    result = await with_timeout(fetch_data(), timeout=10.0)
    return result

# Client can override timeout via header
# curl -H "X-Request-Timeout: 60" http://localhost:8000/api/data
"""
