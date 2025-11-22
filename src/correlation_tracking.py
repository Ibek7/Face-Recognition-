"""
Correlation ID tracking across services.

Provides request correlation for distributed tracing and debugging.
"""

import uuid
from typing import Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger(__name__)


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """Middleware to track correlation IDs across requests."""
    
    def __init__(
        self,
        app,
        header_name: str = "X-Correlation-ID",
        generate_if_missing: bool = True,
    ):
        """
        Initialize correlation ID middleware.
        
        Args:
            app: FastAPI application
            header_name: Header name for correlation ID
            generate_if_missing: Generate ID if not provided
        """
        super().__init__(app)
        self.header_name = header_name
        self.generate_if_missing = generate_if_missing
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process request with correlation ID tracking.
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
            
        Returns:
            Response with correlation ID header
        """
        # Extract or generate correlation ID
        correlation_id = request.headers.get(self.header_name)
        
        if not correlation_id and self.generate_if_missing:
            correlation_id = generate_correlation_id()
        
        # Store in request state
        request.state.correlation_id = correlation_id
        
        # Add to logger context
        logger_adapter = CorrelationLogger(logger, correlation_id)
        request.state.logger = logger_adapter
        
        # Process request
        response = await call_next(request)
        
        # Add correlation ID to response headers
        if correlation_id:
            response.headers[self.header_name] = correlation_id
        
        return response


class CorrelationLogger(logging.LoggerAdapter):
    """Logger adapter that includes correlation ID."""
    
    def __init__(self, logger: logging.Logger, correlation_id: Optional[str]):
        """
        Initialize correlation logger.
        
        Args:
            logger: Base logger
            correlation_id: Correlation ID to include
        """
        super().__init__(logger, {})
        self.correlation_id = correlation_id
    
    def process(self, msg, kwargs):
        """
        Add correlation ID to log message.
        
        Args:
            msg: Log message
            kwargs: Additional kwargs
            
        Returns:
            Modified message and kwargs
        """
        if self.correlation_id:
            msg = f"[{self.correlation_id}] {msg}"
        return msg, kwargs


def generate_correlation_id() -> str:
    """
    Generate a new correlation ID.
    
    Returns:
        Correlation ID string
    """
    return str(uuid.uuid4())


def get_correlation_id(request: Request) -> Optional[str]:
    """
    Get correlation ID from request.
    
    Args:
        request: FastAPI request
        
    Returns:
        Correlation ID or None
    """
    return getattr(request.state, "correlation_id", None)


def get_correlation_logger(request: Request) -> logging.Logger:
    """
    Get correlation-aware logger from request.
    
    Args:
        request: FastAPI request
        
    Returns:
        Logger with correlation ID
    """
    return getattr(request.state, "logger", logger)


class CorrelationContext:
    """Context manager for correlation ID."""
    
    def __init__(self, correlation_id: Optional[str] = None):
        """
        Initialize context.
        
        Args:
            correlation_id: Correlation ID to use
        """
        self.correlation_id = correlation_id or generate_correlation_id()
        self._token = None
    
    def __enter__(self):
        """Enter context."""
        # Store in context var if using contextvars
        return self.correlation_id
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        pass


def propagate_correlation_id(headers: dict, request: Request) -> dict:
    """
    Propagate correlation ID to outgoing requests.
    
    Args:
        headers: Request headers dict
        request: Original FastAPI request
        
    Returns:
        Headers with correlation ID
    """
    correlation_id = get_correlation_id(request)
    if correlation_id:
        headers["X-Correlation-ID"] = correlation_id
    return headers


# Example usage:
"""
from fastapi import FastAPI, Request
from src.correlation_tracking import (
    CorrelationIDMiddleware,
    get_correlation_id,
    get_correlation_logger
)

app = FastAPI()

# Add middleware
app.add_middleware(CorrelationIDMiddleware)

@app.get("/api/data")
async def get_data(request: Request):
    # Get correlation ID
    correlation_id = get_correlation_id(request)
    
    # Use correlation-aware logger
    logger = get_correlation_logger(request)
    logger.info("Processing request")
    
    # When making external requests, propagate the ID
    import httpx
    headers = propagate_correlation_id({}, request)
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.example.com/data",
            headers=headers
        )
    
    return {"correlation_id": correlation_id}
"""
