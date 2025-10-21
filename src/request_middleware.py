# Advanced Request Middleware System

import logging
import time
import uuid
import hashlib
from typing import Callable, Any, Optional, Dict, List
from functools import wraps
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import json

@dataclass
class RequestMetadata:
    """Metadata for a request."""
    request_id: str
    timestamp: float
    method: str
    path: str
    client_ip: str
    user_agent: str
    start_time: float
    end_time: Optional[float] = None
    status_code: Optional[int] = None
    response_size: int = 0
    processing_time_ms: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'request_id': self.request_id,
            'timestamp': datetime.fromtimestamp(self.timestamp).isoformat(),
            'method': self.method,
            'path': self.path,
            'client_ip': self.client_ip,
            'status_code': self.status_code,
            'processing_time_ms': self.processing_time_ms,
            'response_size': self.response_size
        }

class RequestIDMiddleware:
    """Middleware for request ID generation and tracking."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.context_var = threading.local()
    
    def generate_request_id(self) -> str:
        """Generate unique request ID."""
        return str(uuid.uuid4())
    
    def set_request_id(self, request_id: str):
        """Set request ID in context."""
        self.context_var.request_id = request_id
    
    def get_request_id(self) -> str:
        """Get current request ID."""
        return getattr(self.context_var, 'request_id', None)
    
    def process_request(self) -> str:
        """Process incoming request."""
        request_id = self.generate_request_id()
        self.set_request_id(request_id)
        self.logger.debug(f"New request: {request_id}")
        return request_id

class RateLimitMiddleware:
    """Middleware for rate limiting."""
    
    def __init__(self, requests_per_second: float = 100, 
                 window_size_seconds: int = 60):
        self.requests_per_second = requests_per_second
        self.window_size_seconds = window_size_seconds
        self.max_requests_per_window = int(requests_per_second * window_size_seconds)
        
        self.client_requests: Dict[str, deque] = defaultdict(deque)
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def is_rate_limited(self, client_id: str) -> bool:
        """Check if client is rate limited."""
        with self.lock:
            now = time.time()
            window_start = now - self.window_size_seconds
            
            # Remove old requests
            self.client_requests[client_id] = deque(
                req_time for req_time in self.client_requests[client_id]
                if req_time > window_start
            )
            
            if len(self.client_requests[client_id]) >= self.max_requests_per_window:
                return True
            
            # Record new request
            self.client_requests[client_id].append(now)
            return False
    
    def get_client_stats(self, client_id: str) -> Dict[str, Any]:
        """Get rate limit stats for client."""
        with self.lock:
            requests = self.client_requests.get(client_id, deque())
            return {
                'requests_in_window': len(requests),
                'max_requests': self.max_requests_per_window,
                'remaining': max(0, self.max_requests_per_window - len(requests))
            }

class RequestLoggingMiddleware:
    """Middleware for comprehensive request logging."""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.request_history: deque = deque(maxlen=max_history)
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def log_request(self, metadata: RequestMetadata):
        """Log request metadata."""
        with self.lock:
            self.request_history.append(metadata)
            
            log_msg = (f"[{metadata.request_id}] {metadata.method} {metadata.path} "
                      f"- Status: {metadata.status_code}, "
                      f"Time: {metadata.processing_time_ms:.1f}ms")
            
            if metadata.status_code and metadata.status_code >= 400:
                self.logger.warning(log_msg)
            else:
                self.logger.info(log_msg)
    
    def get_request_history(self, limit: int = 100) -> List[Dict]:
        """Get recent request history."""
        with self.lock:
            recent = list(self.request_history)[-limit:]
            return [r.to_dict() for r in recent]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get request statistics."""
        with self.lock:
            if not self.request_history:
                return {}
            
            requests = list(self.request_history)
            successful = [r for r in requests if r.status_code and r.status_code < 400]
            failed = [r for r in requests if r.status_code and r.status_code >= 400]
            
            processing_times = [r.processing_time_ms for r in requests]
            
            return {
                'total_requests': len(requests),
                'successful': len(successful),
                'failed': len(failed),
                'success_rate': (len(successful) / len(requests) * 100) if requests else 0,
                'avg_processing_time_ms': sum(processing_times) / len(processing_times) if processing_times else 0,
                'min_processing_time_ms': min(processing_times) if processing_times else 0,
                'max_processing_time_ms': max(processing_times) if processing_times else 0
            }

class RequestValidationMiddleware:
    """Middleware for request validation."""
    
    def __init__(self):
        self.validators: Dict[str, Callable] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_validator(self, path_pattern: str, validator: Callable):
        """Register validator for path."""
        self.validators[path_pattern] = validator
    
    def validate_request(self, path: str, data: Any) -> tuple[bool, Optional[str]]:
        """Validate request."""
        for pattern, validator in self.validators.items():
            if pattern in path:
                try:
                    is_valid = validator(data)
                    if not is_valid:
                        return False, f"Validation failed for {pattern}"
                except Exception as e:
                    self.logger.error(f"Validator error: {e}")
                    return False, str(e)
        
        return True, None

class RequestCacheMiddleware:
    """Middleware for request caching."""
    
    def __init__(self, ttl_seconds: int = 300):
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, tuple] = {}  # (response, expiry_time)
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def get_cache_key(self, method: str, path: str, params: Dict) -> str:
        """Generate cache key."""
        cache_data = f"{method}:{path}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(cache_data.encode()).hexdigest()
    
    def get_cached_response(self, cache_key: str) -> Optional[Any]:
        """Get cached response if valid."""
        with self.lock:
            if cache_key in self.cache:
                response, expiry_time = self.cache[cache_key]
                if time.time() < expiry_time:
                    return response
                else:
                    del self.cache[cache_key]
            
            return None
    
    def cache_response(self, cache_key: str, response: Any):
        """Cache response."""
        with self.lock:
            expiry_time = time.time() + self.ttl_seconds
            self.cache[cache_key] = (response, expiry_time)
    
    def clear_cache(self):
        """Clear all cache."""
        with self.lock:
            self.cache.clear()

class CompressionMiddleware:
    """Middleware for response compression."""
    
    def __init__(self, min_size_bytes: int = 1024):
        self.min_size_bytes = min_size_bytes
        self.logger = logging.getLogger(__name__)
    
    def should_compress(self, response_size: int, content_type: str) -> bool:
        """Check if response should be compressed."""
        compressible_types = [
            'application/json',
            'text/html',
            'text/css',
            'application/javascript'
        ]
        
        return (response_size >= self.min_size_bytes and
                any(ct in content_type for ct in compressible_types))
    
    def compress_response(self, response: Any) -> bytes:
        """Compress response."""
        import gzip
        
        if isinstance(response, str):
            data = response.encode('utf-8')
        else:
            data = json.dumps(response).encode('utf-8')
        
        return gzip.compress(data)

class SecurityHeadersMiddleware:
    """Middleware for security headers."""
    
    def __init__(self):
        self.security_headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'",
            'Referrer-Policy': 'strict-origin-when-cross-origin'
        }
    
    def get_security_headers(self) -> Dict[str, str]:
        """Get security headers."""
        return self.security_headers.copy()

class MiddlewareChain:
    """Chain multiple middleware."""
    
    def __init__(self):
        self.middleware: List[Callable] = []
        self.logger = logging.getLogger(__name__)
    
    def add_middleware(self, middleware: Callable) -> 'MiddlewareChain':
        """Add middleware to chain."""
        self.middleware.append(middleware)
        return self
    
    def process_request(self, request_context: Dict) -> Dict:
        """Process request through middleware chain."""
        for mw in self.middleware:
            request_context = mw(request_context) or request_context
        
        return request_context
    
    def process_response(self, request_context: Dict, 
                        response: Any) -> Any:
        """Process response through middleware chain."""
        for mw in reversed(self.middleware):
            if hasattr(mw, 'process_response'):
                response = mw.process_response(request_context, response)
        
        return response

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create middleware instances
    request_id_mw = RequestIDMiddleware()
    rate_limit_mw = RateLimitMiddleware(requests_per_second=10)
    logging_mw = RequestLoggingMiddleware()
    validation_mw = RequestValidationMiddleware()
    cache_mw = RequestCacheMiddleware(ttl_seconds=60)
    compression_mw = CompressionMiddleware()
    security_mw = SecurityHeadersMiddleware()
    
    # Simulate requests
    for i in range(5):
        request_id = request_id_mw.process_request()
        
        # Check rate limit
        is_limited = rate_limit_mw.is_rate_limited(f"client_{i % 2}")
        
        # Create metadata
        metadata = RequestMetadata(
            request_id=request_id,
            timestamp=time.time(),
            method="GET",
            path=f"/api/detect/{i}",
            client_ip="192.168.1.1",
            user_agent="Mozilla/5.0",
            start_time=time.time(),
            end_time=time.time() + 0.1,
            status_code=200 if not is_limited else 429,
            response_size=1024
        )
        
        metadata.processing_time_ms = (metadata.end_time - metadata.start_time) * 1000
        
        # Log request
        logging_mw.log_request(metadata)
    
    # Print statistics
    stats = logging_mw.get_statistics()
    print(f"\nRequest Statistics:")
    print(json.dumps(stats, indent=2))
    
    # Print security headers
    headers = security_mw.get_security_headers()
    print(f"\nSecurity Headers:")
    for key, value in headers.items():
        print(f"  {key}: {value}")