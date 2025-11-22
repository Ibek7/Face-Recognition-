"""
HTTP request/response interceptors for FastAPI.

Provides middleware for logging, modifying, and analyzing HTTP traffic.
"""

import time
import json
from typing import Callable, Optional, List
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.datastructures import Headers
import logging

logger = logging.getLogger(__name__)


class RequestInterceptor:
    """Intercept and modify incoming requests."""
    
    def __init__(self):
        """Initialize request interceptor."""
        self.handlers: List[Callable] = []
    
    def add_handler(self, handler: Callable):
        """
        Add request handler.
        
        Args:
            handler: Async function(request) -> request
        """
        self.handlers.append(handler)
    
    async def process(self, request: Request) -> Request:
        """
        Process request through all handlers.
        
        Args:
            request: Incoming request
        
        Returns:
            Modified request
        """
        for handler in self.handlers:
            request = await handler(request)
        
        return request


class ResponseInterceptor:
    """Intercept and modify outgoing responses."""
    
    def __init__(self):
        """Initialize response interceptor."""
        self.handlers: List[Callable] = []
    
    def add_handler(self, handler: Callable):
        """
        Add response handler.
        
        Args:
            handler: Async function(response) -> response
        """
        self.handlers.append(handler)
    
    async def process(self, response: Response) -> Response:
        """
        Process response through all handlers.
        
        Args:
            response: Outgoing response
        
        Returns:
            Modified response
        """
        for handler in self.handlers:
            response = await handler(response)
        
        return response


class InterceptorMiddleware(BaseHTTPMiddleware):
    """HTTP interceptor middleware."""
    
    def __init__(
        self,
        app,
        request_interceptor: RequestInterceptor,
        response_interceptor: ResponseInterceptor,
        log_requests: bool = True,
        log_responses: bool = True
    ):
        """
        Initialize interceptor middleware.
        
        Args:
            app: FastAPI app
            request_interceptor: Request interceptor
            response_interceptor: Response interceptor
            log_requests: Log requests
            log_responses: Log responses
        """
        super().__init__(app)
        self.request_interceptor = request_interceptor
        self.response_interceptor = response_interceptor
        self.log_requests = log_requests
        self.log_responses = log_responses
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and response."""
        start_time = time.time()
        
        # Log request
        if self.log_requests:
            await self._log_request(request)
        
        # Process request interceptors
        request = await self.request_interceptor.process(request)
        
        # Call next middleware/handler
        response = await call_next(request)
        
        # Process response interceptors
        response = await self.response_interceptor.process(response)
        
        # Log response
        if self.log_responses:
            duration = time.time() - start_time
            await self._log_response(request, response, duration)
        
        return response
    
    async def _log_request(self, request: Request):
        """Log incoming request."""
        # Get request body if available
        body_preview = ""
        
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                
                # Limit body size in log
                if len(body) > 500:
                    body_preview = body[:500].decode('utf-8', errors='ignore') + "..."
                else:
                    body_preview = body.decode('utf-8', errors='ignore')
            except Exception:
                body_preview = "<binary data>"
        
        logger.info(
            f"→ {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        if body_preview:
            logger.debug(f"  Body: {body_preview}")
    
    async def _log_response(
        self,
        request: Request,
        response: Response,
        duration: float
    ):
        """Log outgoing response."""
        logger.info(
            f"← {request.method} {request.url.path} "
            f"→ {response.status_code} ({duration*1000:.2f}ms)"
        )


class RequestLogger:
    """Advanced request/response logger."""
    
    def __init__(self, include_headers: bool = False, include_body: bool = False):
        """
        Initialize request logger.
        
        Args:
            include_headers: Log headers
            include_body: Log request/response bodies
        """
        self.include_headers = include_headers
        self.include_body = include_body
    
    async def log_request(self, request: Request):
        """Log detailed request information."""
        log_data = {
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client": request.client.host if request.client else None
        }
        
        if self.include_headers:
            log_data["headers"] = dict(request.headers)
        
        if self.include_body and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                log_data["body"] = body.decode('utf-8', errors='ignore')
            except Exception:
                log_data["body"] = "<binary>"
        
        logger.info(f"Request: {json.dumps(log_data, indent=2)}")
    
    async def log_response(
        self,
        request: Request,
        response: Response,
        duration: float
    ):
        """Log detailed response information."""
        log_data = {
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration * 1000, 2)
        }
        
        if self.include_headers:
            log_data["headers"] = dict(response.headers)
        
        logger.info(f"Response: {json.dumps(log_data, indent=2)}")


# Common interceptor handlers

async def add_security_headers(response: Response) -> Response:
    """Add security headers to response."""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


async def add_cors_headers(response: Response) -> Response:
    """Add CORS headers to response."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


async def add_request_id(request: Request) -> Request:
    """Add unique request ID to request."""
    import uuid
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    return request


async def add_request_id_to_response(response: Response) -> Response:
    """Add request ID to response headers."""
    # Note: This requires request context to be available
    # In practice, you'd pass request_id through the response pipeline
    return response


async def log_slow_requests(request: Request) -> Request:
    """Track request start time for slow request detection."""
    request.state.start_time = time.time()
    return request


async def detect_slow_response(response: Response) -> Response:
    """Detect and log slow responses."""
    # Note: This requires access to request.state.start_time
    # Implementation would check elapsed time
    return response


# Global interceptors
request_interceptor = RequestInterceptor()
response_interceptor = ResponseInterceptor()


# Example usage:
"""
from fastapi import FastAPI
from src.http_interceptors import (
    InterceptorMiddleware,
    request_interceptor,
    response_interceptor,
    add_security_headers,
    add_request_id
)

app = FastAPI()

# Add interceptor handlers
request_interceptor.add_handler(add_request_id)
response_interceptor.add_handler(add_security_headers)

# Add middleware
app.add_middleware(
    InterceptorMiddleware,
    request_interceptor=request_interceptor,
    response_interceptor=response_interceptor,
    log_requests=True,
    log_responses=True
)

# Custom interceptor
async def track_api_version(request: Request) -> Request:
    version = request.headers.get("X-API-Version", "v1")
    request.state.api_version = version
    return request

request_interceptor.add_handler(track_api_version)

# Custom response handler
async def add_processing_time(response: Response) -> Response:
    # Calculate processing time
    response.headers["X-Processing-Time"] = "0.123"
    return response

response_interceptor.add_handler(add_processing_time)

@app.get("/api/data")
async def get_data(request: Request):
    # Access intercepted data
    request_id = request.state.request_id
    api_version = request.state.api_version
    
    return {
        "request_id": request_id,
        "api_version": api_version,
        "data": "example"
    }
"""
