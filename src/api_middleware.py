"""
Middleware components for the Face Recognition API.
Includes rate limiting, error handling, logging, and security features.
"""

import time
import json
import logging
from typing import Callable, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import asyncio
import hashlib

# Configure logging
logger = logging.getLogger(__name__)

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware with per-IP and per-endpoint limits.
    """
    
    def __init__(
        self,
        app: ASGIApp,
        requests_per_minute: int = 100,
        burst_requests: int = 10,
        cleanup_interval: int = 300  # 5 minutes
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_requests = burst_requests
        self.cleanup_interval = cleanup_interval
        
        # Storage for request tracking
        self.request_history: Dict[str, deque] = defaultdict(lambda: deque())
        self.blocked_ips: Dict[str, datetime] = {}
        
        # Last cleanup time
        self.last_cleanup = time.time()
        
        # Endpoint-specific limits
        self.endpoint_limits = {
            "/upload": 20,
            "/recognize": 50,
            "/batch/jobs": 10,
        }
    
    def get_client_ip(self, request: Request) -> str:
        """Extract client IP address."""
        # Check for forwarded headers first
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to direct connection
        return request.client.host if request.client else "unknown"
    
    def get_endpoint_limit(self, path: str) -> int:
        """Get rate limit for specific endpoint."""
        for endpoint, limit in self.endpoint_limits.items():
            if path.startswith(endpoint):
                return limit
        return self.requests_per_minute
    
    def cleanup_old_requests(self):
        """Remove old request records to prevent memory bloat."""
        current_time = time.time()
        
        # Only cleanup periodically
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        cutoff_time = current_time - 3600  # 1 hour ago
        
        # Clean request history
        for ip, requests in list(self.request_history.items()):
            # Remove old requests
            while requests and requests[0] < cutoff_time:
                requests.popleft()
            
            # Remove empty entries
            if not requests:
                del self.request_history[ip]
        
        # Clean blocked IPs
        unblock_time = datetime.now() - timedelta(minutes=10)
        self.blocked_ips = {
            ip: block_time 
            for ip, block_time in self.blocked_ips.items()
            if block_time > unblock_time
        }
        
        self.last_cleanup = current_time
    
    def is_rate_limited(self, ip: str, path: str) -> tuple[bool, Dict[str, Any]]:
        """Check if IP is rate limited."""
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Check if IP is temporarily blocked
        if ip in self.blocked_ips:
            block_time = self.blocked_ips[ip]
            if datetime.now() - block_time < timedelta(minutes=10):
                return True, {
                    "error": "IP temporarily blocked",
                    "retry_after": 600,
                    "blocked_until": (block_time + timedelta(minutes=10)).isoformat()
                }
            else:
                # Unblock IP
                del self.blocked_ips[ip]
        
        # Get request history for this IP
        requests = self.request_history[ip]
        
        # Remove requests older than 1 minute
        while requests and requests[0] < minute_ago:
            requests.popleft()
        
        # Get limit for this endpoint
        limit = self.get_endpoint_limit(path)
        
        # Check rate limit
        if len(requests) >= limit:
            # Block IP if excessive requests
            if len(requests) > limit * 2:
                self.blocked_ips[ip] = datetime.now()
                logger.warning(f"Blocked IP {ip} for excessive requests")
            
            return True, {
                "error": "Rate limit exceeded",
                "limit": limit,
                "window": "1 minute",
                "retry_after": 60,
                "requests_made": len(requests)
            }
        
        # Record this request
        requests.append(current_time)
        
        return False, {
            "requests_remaining": limit - len(requests),
            "reset_time": minute_ago + 60
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting."""
        # Cleanup old records periodically
        self.cleanup_old_requests()
        
        # Get client IP
        client_ip = self.get_client_ip(request)
        
        # Check rate limit
        is_limited, limit_info = self.is_rate_limited(client_ip, request.url.path)
        
        if is_limited:
            return JSONResponse(
                status_code=429,
                content={
                    "success": False,
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "message": "Rate limit exceeded",
                    "details": limit_info,
                    "timestamp": datetime.now().isoformat()
                },
                headers={
                    "Retry-After": str(limit_info.get("retry_after", 60)),
                    "X-RateLimit-Limit": str(self.get_endpoint_limit(request.url.path)),
                    "X-RateLimit-Remaining": "0"
                }
            )
        
        # Process request
        start_time = time.time()
        response = await call_next(request)
        processing_time = time.time() - start_time
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.get_endpoint_limit(request.url.path))
        response.headers["X-RateLimit-Remaining"] = str(limit_info.get("requests_remaining", 0))
        response.headers["X-Processing-Time"] = f"{processing_time:.3f}"
        
        return response


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Security middleware for headers, request validation, and basic protection.
    """
    
    def __init__(self, app: ASGIApp, max_request_size: int = 10 * 1024 * 1024):
        super().__init__(app)
        self.max_request_size = max_request_size
        
        # Security headers
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin"
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with security checks."""
        # Check request size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_request_size:
            return JSONResponse(
                status_code=413,
                content={
                    "success": False,
                    "error_code": "REQUEST_TOO_LARGE",
                    "message": f"Request size exceeds maximum of {self.max_request_size} bytes",
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        for header, value in self.security_headers.items():
            response.headers[header] = value
        
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive logging middleware for request/response tracking.
    """
    
    def __init__(self, app: ASGIApp, log_level: int = logging.INFO):
        super().__init__(app)
        self.logger = logging.getLogger("api.requests")
        self.logger.setLevel(log_level)
        
        # Don't log health checks by default
        self.skip_paths = {"/health", "/docs", "/openapi.json", "/redoc"}
    
    def should_log(self, path: str) -> bool:
        """Determine if request should be logged."""
        return path not in self.skip_paths
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request and response details."""
        start_time = time.time()
        
        # Get request details
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        request_id = hashlib.md5(f"{client_ip}{start_time}".encode()).hexdigest()[:8]
        
        # Log request
        if self.should_log(request.url.path):
            self.logger.info(
                f"[{request_id}] {request.method} {request.url.path} - "
                f"IP: {client_ip} - UA: {user_agent[:100]}"
            )
        
        # Process request
        try:
            response = await call_next(request)
            processing_time = time.time() - start_time
            
            # Log response
            if self.should_log(request.url.path):
                self.logger.info(
                    f"[{request_id}] Response: {response.status_code} - "
                    f"Time: {processing_time:.3f}s"
                )
            
            # Add request ID to response
            response.headers["X-Request-ID"] = request_id
            
            return response
        
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Log error
            self.logger.error(
                f"[{request_id}] ERROR: {str(e)} - Time: {processing_time:.3f}s"
            )
            
            # Re-raise exception
            raise


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Global error handling middleware with structured error responses.
    """
    
    def __init__(self, app: ASGIApp, debug: bool = False):
        super().__init__(app)
        self.debug = debug
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle errors and return structured responses."""
        try:
            return await call_next(request)
        
        except HTTPException as e:
            # Handle FastAPI HTTP exceptions
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "success": False,
                    "error_code": "HTTP_ERROR",
                    "message": e.detail,
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        except ValueError as e:
            # Handle validation errors
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error_code": "VALIDATION_ERROR",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        except FileNotFoundError as e:
            # Handle file not found errors
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error_code": "FILE_NOT_FOUND",
                    "message": "Requested file not found",
                    "details": str(e) if self.debug else None,
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        except PermissionError as e:
            # Handle permission errors
            return JSONResponse(
                status_code=403,
                content={
                    "success": False,
                    "error_code": "PERMISSION_DENIED",
                    "message": "Access denied",
                    "details": str(e) if self.debug else None,
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        except Exception as e:
            # Handle all other errors
            logger.exception(f"Unhandled error: {str(e)}")
            
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error_code": "INTERNAL_ERROR",
                    "message": "Internal server error",
                    "details": str(e) if self.debug else "An unexpected error occurred",
                    "timestamp": datetime.now().isoformat()
                }
            )


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware for collecting API metrics and performance data.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.metrics = {
            "requests_total": 0,
            "requests_by_method": defaultdict(int),
            "requests_by_status": defaultdict(int),
            "response_times": deque(maxlen=1000),
            "errors_total": 0,
            "start_time": time.time()
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        response_times = list(self.metrics["response_times"])
        
        return {
            "requests_total": self.metrics["requests_total"],
            "requests_by_method": dict(self.metrics["requests_by_method"]),
            "requests_by_status": dict(self.metrics["requests_by_status"]),
            "errors_total": self.metrics["errors_total"],
            "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
            "uptime_seconds": time.time() - self.metrics["start_time"]
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Collect metrics for each request."""
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Update metrics
        self.metrics["requests_total"] += 1
        self.metrics["requests_by_method"][request.method] += 1
        self.metrics["requests_by_status"][response.status_code] += 1
        self.metrics["response_times"].append(response_time)
        
        if response.status_code >= 400:
            self.metrics["errors_total"] += 1
        
        return response


# Global metrics instance
metrics_middleware = None

def get_metrics() -> Dict[str, Any]:
    """Get current API metrics."""
    global metrics_middleware
    if metrics_middleware:
        return metrics_middleware.get_metrics()
    return {}


# Middleware configuration function
def setup_middleware(app, config: Dict[str, Any] = None):
    """
    Setup all middleware for the FastAPI application.
    
    Args:
        app: FastAPI application instance
        config: Configuration dictionary
    """
    global metrics_middleware
    
    if config is None:
        config = {}
    
    # Add middleware in reverse order (last added = first executed)
    
    # Error handling (should be first to catch all errors)
    app.add_middleware(
        ErrorHandlingMiddleware,
        debug=config.get("debug", False)
    )
    
    # Security middleware
    app.add_middleware(
        SecurityMiddleware,
        max_request_size=config.get("max_request_size", 10 * 1024 * 1024)
    )
    
    # Rate limiting
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=config.get("rate_limit_per_minute", 100),
        burst_requests=config.get("burst_requests", 10)
    )
    
    # Logging
    app.add_middleware(
        LoggingMiddleware,
        log_level=logging.DEBUG if config.get("debug", False) else logging.INFO
    )
    
    # Metrics (should be last to measure everything)
    metrics_middleware = MetricsMiddleware(app)
    app.add_middleware(MetricsMiddleware)
    
    logger.info("All middleware configured successfully")