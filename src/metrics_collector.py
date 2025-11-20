"""
Metrics collector for API performance monitoring.

Collects and exports metrics in Prometheus format.
"""

import time
from typing import Dict, Optional
from collections import defaultdict
from datetime import datetime
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST


# Define Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

REQUEST_SIZE = Histogram(
    'http_request_size_bytes',
    'HTTP request size in bytes',
    ['method', 'endpoint']
)

RESPONSE_SIZE = Histogram(
    'http_response_size_bytes',
    'HTTP response size in bytes',
    ['method', 'endpoint']
)

ACTIVE_REQUESTS = Gauge(
    'http_requests_active',
    'Number of active HTTP requests'
)

FACE_DETECTIONS = Counter(
    'face_detections_total',
    'Total face detections performed',
    ['status']
)

FACE_RECOGNITIONS = Counter(
    'face_recognitions_total',
    'Total face recognitions performed',
    ['status']
)

EMBEDDING_GENERATION = Histogram(
    'embedding_generation_seconds',
    'Time to generate face embeddings'
)


class MetricsCollector(BaseHTTPMiddleware):
    """Middleware to collect API metrics."""
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Collect metrics for each request.
        
        Args:
            request: Incoming request
            call_next: Next handler
            
        Returns:
            Response with metrics collected
        """
        # Skip metrics endpoint itself
        if request.url.path == "/metrics":
            return await call_next(request)
        
        # Track active requests
        ACTIVE_REQUESTS.inc()
        
        # Extract endpoint (remove path parameters)
        endpoint = self._get_endpoint(request)
        method = request.method
        
        # Track request size
        request_size = int(request.headers.get("content-length", 0))
        REQUEST_SIZE.labels(method=method, endpoint=endpoint).observe(request_size)
        
        # Time the request
        start_time = time.time()
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Record metrics
            REQUEST_COUNT.labels(
                method=method,
                endpoint=endpoint,
                status=response.status_code
            ).inc()
            
            REQUEST_DURATION.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
            # Track response size
            response_size = int(response.headers.get("content-length", 0))
            RESPONSE_SIZE.labels(
                method=method,
                endpoint=endpoint
            ).observe(response_size)
            
            return response
        
        finally:
            # Decrement active requests
            ACTIVE_REQUESTS.dec()
    
    def _get_endpoint(self, request: Request) -> str:
        """
        Extract endpoint pattern from request.
        
        Args:
            request: Request object
            
        Returns:
            Endpoint pattern (e.g., /api/v1/users/{id})
        """
        # Try to get route pattern
        if hasattr(request, "scope") and "route" in request.scope:
            route = request.scope["route"]
            if hasattr(route, "path"):
                return route.path
        
        # Fallback to raw path
        return request.url.path


class InMemoryMetrics:
    """In-memory metrics storage (alternative to Prometheus)."""
    
    def __init__(self):
        """Initialize metrics storage."""
        self.requests = defaultdict(int)
        self.durations = defaultdict(list)
        self.errors = defaultdict(int)
        self.start_time = datetime.utcnow()
    
    def record_request(
        self,
        method: str,
        endpoint: str,
        status: int,
        duration: float
    ):
        """
        Record a request metric.
        
        Args:
            method: HTTP method
            endpoint: Endpoint path
            status: Response status code
            duration: Request duration in seconds
        """
        key = f"{method}:{endpoint}"
        self.requests[key] += 1
        self.durations[key].append(duration)
        
        if status >= 400:
            self.errors[key] += 1
    
    def get_metrics(self) -> Dict:
        """
        Get current metrics.
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds(),
            "endpoints": {}
        }
        
        for key in self.requests:
            durations = self.durations[key]
            metrics["endpoints"][key] = {
                "requests": self.requests[key],
                "errors": self.errors[key],
                "avg_duration": sum(durations) / len(durations) if durations else 0,
                "min_duration": min(durations) if durations else 0,
                "max_duration": max(durations) if durations else 0,
            }
        
        return metrics


# Global in-memory metrics instance
in_memory_metrics = InMemoryMetrics()


def track_face_detection(success: bool):
    """
    Track face detection metric.
    
    Args:
        success: Whether detection was successful
    """
    status = "success" if success else "failure"
    FACE_DETECTIONS.labels(status=status).inc()


def track_face_recognition(success: bool):
    """
    Track face recognition metric.
    
    Args:
        success: Whether recognition was successful
    """
    status = "success" if success else "failure"
    FACE_RECOGNITIONS.labels(status=status).inc()


def track_embedding_time(duration: float):
    """
    Track embedding generation time.
    
    Args:
        duration: Time in seconds
    """
    EMBEDDING_GENERATION.observe(duration)


async def metrics_endpoint():
    """
    Prometheus metrics endpoint.
    
    Returns:
        Metrics in Prometheus format
    """
    from fastapi import Response
    
    metrics_output = generate_latest()
    return Response(
        content=metrics_output,
        media_type=CONTENT_TYPE_LATEST
    )


# Example usage in api_server.py:
"""
from src.metrics_collector import MetricsCollector, metrics_endpoint

app = FastAPI()

# Add metrics middleware
app.add_middleware(MetricsCollector)

# Add metrics endpoint
@app.get("/metrics")
async def metrics():
    return await metrics_endpoint()

# Use in endpoints
from src.metrics_collector import track_face_detection

@app.post("/detect")
async def detect_faces(image: UploadFile):
    try:
        result = detect(image)
        track_face_detection(success=True)
        return result
    except Exception as e:
        track_face_detection(success=False)
        raise
"""
