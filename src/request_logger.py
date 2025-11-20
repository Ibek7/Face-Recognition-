"""
Request and response logging middleware for FastAPI.

Logs all HTTP requests and responses with detailed information.
"""

import time
import json
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all requests and responses."""
    
    def __init__(
        self,
        app,
        log_request_body: bool = False,
        log_response_body: bool = False,
        exclude_paths: list = None,
    ):
        """
        Initialize logging middleware.
        
        Args:
            app: FastAPI application
            log_request_body: Whether to log request body
            log_response_body: Whether to log response body
            exclude_paths: Paths to exclude from logging
        """
        super().__init__(app)
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.exclude_paths = exclude_paths or ["/health", "/healthz", "/metrics"]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Log request and response.
        
        Args:
            request: Incoming request
            call_next: Next handler
            
        Returns:
            Response
        """
        # Skip excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        # Generate request ID
        request_id = self._generate_request_id()
        
        # Log request
        await self._log_request(request, request_id)
        
        # Process request and time it
        start_time = time.time()
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Log response
            self._log_response(request, response, duration, request_id)
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
        
        except Exception as e:
            duration = time.time() - start_time
            self._log_error(request, e, duration, request_id)
            raise
    
    async def _log_request(self, request: Request, request_id: str):
        """
        Log incoming request.
        
        Args:
            request: Request object
            request_id: Unique request identifier
        """
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "headers": dict(request.headers),
            "client": request.client.host if request.client else None,
        }
        
        # Log request body if enabled
        if self.log_request_body and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    log_data["body"] = body.decode("utf-8")[:1000]  # Limit size
            except Exception:
                log_data["body"] = "<unable to decode>"
        
        logger.info(f"Request: {json.dumps(log_data)}")
    
    def _log_response(
        self,
        request: Request,
        response: Response,
        duration: float,
        request_id: str
    ):
        """
        Log response.
        
        Args:
            request: Request object
            response: Response object
            duration: Request duration in seconds
            request_id: Request identifier
        """
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration * 1000, 2),
        }
        
        # Determine log level based on status code
        if response.status_code >= 500:
            logger.error(f"Response: {json.dumps(log_data)}")
        elif response.status_code >= 400:
            logger.warning(f"Response: {json.dumps(log_data)}")
        else:
            logger.info(f"Response: {json.dumps(log_data)}")
    
    def _log_error(
        self,
        request: Request,
        error: Exception,
        duration: float,
        request_id: str
    ):
        """
        Log error.
        
        Args:
            request: Request object
            error: Exception that occurred
            duration: Request duration in seconds
            request_id: Request identifier
        """
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "error": str(error),
            "error_type": type(error).__name__,
            "duration_ms": round(duration * 1000, 2),
        }
        
        logger.error(f"Error: {json.dumps(log_data)}", exc_info=True)
    
    def _generate_request_id(self) -> str:
        """
        Generate unique request ID.
        
        Returns:
            Request ID string
        """
        import uuid
        return str(uuid.uuid4())


class StructuredLogger:
    """Helper for structured logging."""
    
    def __init__(self, name: str):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
        """
        self.logger = logging.getLogger(name)
    
    def log(self, level: str, message: str, **kwargs):
        """
        Log structured message.
        
        Args:
            level: Log level (info, warning, error, etc.)
            message: Log message
            **kwargs: Additional structured data
        """
        log_data = {"message": message, **kwargs}
        log_func = getattr(self.logger, level.lower())
        log_func(json.dumps(log_data))
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.log("info", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.log("warning", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.log("error", message, **kwargs)


def get_request_id(request: Request) -> str:
    """
    Get request ID from request headers.
    
    Args:
        request: FastAPI request
        
    Returns:
        Request ID
    """
    return request.headers.get("X-Request-ID", "unknown")


# Example usage:
"""
from src.request_logger import RequestLoggingMiddleware

app = FastAPI()

app.add_middleware(
    RequestLoggingMiddleware,
    log_request_body=True,
    log_response_body=False,
    exclude_paths=["/health", "/metrics"]
)
"""
