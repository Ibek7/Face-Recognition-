#!/usr/bin/env python3
"""
Prometheus Metrics Exporter

Exports custom metrics for the Face Recognition system:
- Request metrics (rate, latency, errors)
- Model performance metrics
- System resource metrics
- Business metrics (detections, recognitions, enrollments)
"""

import time
import psutil
import logging
from typing import Dict, Any, Optional, Callable
from functools import wraps
from contextlib import contextmanager

from prometheus_client import (
    Counter, Histogram, Gauge, Info, Summary,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)
from prometheus_client.multiprocess import MultiProcessCollector
from fastapi import FastAPI, Response, Request
from starlette.middleware.base import BaseHTTPMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Metrics Registry
# ============================================================================

# Create custom registry
registry = CollectorRegistry()

# ============================================================================
# Request Metrics
# ============================================================================

# HTTP request counter
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status'],
    registry=registry
)

# HTTP request duration
http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency in seconds',
    ['method', 'endpoint'],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=registry
)

# HTTP request size
http_request_size_bytes = Summary(
    'http_request_size_bytes',
    'HTTP request size in bytes',
    ['method', 'endpoint'],
    registry=registry
)

# HTTP response size
http_response_size_bytes = Summary(
    'http_response_size_bytes',
    'HTTP response size in bytes',
    ['method', 'endpoint'],
    registry=registry
)

# Active requests
http_requests_active = Gauge(
    'http_requests_active',
    'Number of active HTTP requests',
    ['method', 'endpoint'],
    registry=registry
)

# ============================================================================
# Face Detection Metrics
# ============================================================================

# Detection counter
face_detections_total = Counter(
    'face_detections_total',
    'Total face detections',
    ['model', 'status'],
    registry=registry
)

# Faces detected per request
faces_detected_per_request = Histogram(
    'faces_detected_per_request',
    'Number of faces detected per request',
    buckets=(0, 1, 2, 3, 5, 10, 20, 50),
    registry=registry
)

# Detection duration
face_detection_duration_seconds = Histogram(
    'face_detection_duration_seconds',
    'Face detection processing time',
    ['model'],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
    registry=registry
)

# Detection confidence
face_detection_confidence = Histogram(
    'face_detection_confidence',
    'Face detection confidence scores',
    ['model'],
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
    registry=registry
)

# ============================================================================
# Face Recognition Metrics
# ============================================================================

# Recognition counter
face_recognitions_total = Counter(
    'face_recognitions_total',
    'Total face recognitions',
    ['model', 'status'],
    registry=registry
)

# Recognition matches
face_recognition_matches = Counter(
    'face_recognition_matches',
    'Face recognition matches found',
    ['model'],
    registry=registry
)

# Recognition duration
face_recognition_duration_seconds = Histogram(
    'face_recognition_duration_seconds',
    'Face recognition processing time',
    ['model'],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
    registry=registry
)

# Recognition confidence
face_recognition_confidence = Histogram(
    'face_recognition_confidence',
    'Face recognition confidence scores',
    ['model'],
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
    registry=registry
)

# Embedding distance
embedding_distance = Histogram(
    'embedding_distance',
    'Face embedding distances',
    ['model'],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    registry=registry
)

# ============================================================================
# Person Management Metrics
# ============================================================================

# Total persons
total_persons = Gauge(
    'total_persons',
    'Total number of enrolled persons',
    registry=registry
)

# Person operations
person_operations_total = Counter(
    'person_operations_total',
    'Person management operations',
    ['operation'],  # create, update, delete
    registry=registry
)

# Total embeddings
total_embeddings = Gauge(
    'total_embeddings',
    'Total number of face embeddings',
    registry=registry
)

# Embeddings per person
embeddings_per_person = Histogram(
    'embeddings_per_person',
    'Number of embeddings per person',
    buckets=(1, 2, 3, 5, 10, 20, 50),
    registry=registry
)

# ============================================================================
# Model Performance Metrics
# ============================================================================

# Model inference time
model_inference_duration_seconds = Histogram(
    'model_inference_duration_seconds',
    'Model inference time',
    ['model_name', 'model_type'],  # model_type: detection, recognition
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
    registry=registry
)

# Model memory usage
model_memory_usage_bytes = Gauge(
    'model_memory_usage_bytes',
    'Model memory usage in bytes',
    ['model_name'],
    registry=registry
)

# Model load time
model_load_duration_seconds = Gauge(
    'model_load_duration_seconds',
    'Model loading time',
    ['model_name'],
    registry=registry
)

# ============================================================================
# Cache Metrics
# ============================================================================

# Cache hits
cache_hits_total = Counter(
    'cache_hits_total',
    'Total cache hits',
    ['cache_name'],
    registry=registry
)

# Cache misses
cache_misses_total = Counter(
    'cache_misses_total',
    'Total cache misses',
    ['cache_name'],
    registry=registry
)

# Cache size
cache_size = Gauge(
    'cache_size',
    'Current cache size',
    ['cache_name'],
    registry=registry
)

# Cache evictions
cache_evictions_total = Counter(
    'cache_evictions_total',
    'Total cache evictions',
    ['cache_name'],
    registry=registry
)

# ============================================================================
# Database Metrics
# ============================================================================

# Database queries
database_queries_total = Counter(
    'database_queries_total',
    'Total database queries',
    ['operation'],  # select, insert, update, delete
    registry=registry
)

# Database query duration
database_query_duration_seconds = Histogram(
    'database_query_duration_seconds',
    'Database query duration',
    ['operation'],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
    registry=registry
)

# Database connections
database_connections_active = Gauge(
    'database_connections_active',
    'Active database connections',
    registry=registry
)

# Database connection pool
database_connection_pool_size = Gauge(
    'database_connection_pool_size',
    'Database connection pool size',
    registry=registry
)

# ============================================================================
# System Metrics
# ============================================================================

# CPU usage
cpu_usage_percent = Gauge(
    'cpu_usage_percent',
    'CPU usage percentage',
    registry=registry
)

# Memory usage
memory_usage_bytes = Gauge(
    'memory_usage_bytes',
    'Memory usage in bytes',
    registry=registry
)

# Disk usage
disk_usage_bytes = Gauge(
    'disk_usage_bytes',
    'Disk usage in bytes',
    ['mount_point'],
    registry=registry
)

# GPU metrics (if available)
gpu_utilization_percent = Gauge(
    'gpu_utilization_percent',
    'GPU utilization percentage',
    ['gpu_id'],
    registry=registry
)

gpu_memory_usage_bytes = Gauge(
    'gpu_memory_usage_bytes',
    'GPU memory usage in bytes',
    ['gpu_id'],
    registry=registry
)

# ============================================================================
# Application Metrics
# ============================================================================

# Application info
app_info = Info(
    'app',
    'Application information',
    registry=registry
)

# Uptime
app_uptime_seconds = Gauge(
    'app_uptime_seconds',
    'Application uptime in seconds',
    registry=registry
)

# Start time
app_start_time = Gauge(
    'app_start_time',
    'Application start time (Unix timestamp)',
    registry=registry
)


# ============================================================================
# Metrics Collector
# ============================================================================

class MetricsCollector:
    """Collects and updates system metrics"""
    
    def __init__(self):
        self.start_time = time.time()
        app_start_time.set(self.start_time)
    
    def update_system_metrics(self):
        """Update system resource metrics"""
        # CPU usage
        cpu_usage_percent.set(psutil.cpu_percent(interval=1))
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_usage_bytes.set(memory.used)
        
        # Disk usage
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_usage_bytes.labels(mount_point=partition.mountpoint).set(usage.used)
            except Exception:
                pass
        
        # Uptime
        app_uptime_seconds.set(time.time() - self.start_time)
    
    def update_gpu_metrics(self):
        """Update GPU metrics (requires pynvml)"""
        try:
            import pynvml
            
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # GPU utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_utilization_percent.labels(gpu_id=str(i)).set(util.gpu)
                
                # GPU memory
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory_usage_bytes.labels(gpu_id=str(i)).set(mem_info.used)
            
            pynvml.nvmlShutdown()
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Failed to update GPU metrics: {e}")


# ============================================================================
# Decorators for Automatic Instrumentation
# ============================================================================

def track_detection(model: str):
    """Decorator to track face detection metrics"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # Track metrics
                duration = time.time() - start_time
                face_detection_duration_seconds.labels(model=model).observe(duration)
                face_detections_total.labels(model=model, status='success').inc()
                
                # Track faces detected
                if hasattr(result, 'faces'):
                    num_faces = len(result.faces)
                    faces_detected_per_request.observe(num_faces)
                    
                    # Track confidence scores
                    for face in result.faces:
                        face_detection_confidence.labels(model=model).observe(face.confidence)
                
                return result
                
            except Exception as e:
                face_detections_total.labels(model=model, status='error').inc()
                raise
        
        return wrapper
    return decorator


def track_recognition(model: str):
    """Decorator to track face recognition metrics"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # Track metrics
                duration = time.time() - start_time
                face_recognition_duration_seconds.labels(model=model).observe(duration)
                face_recognitions_total.labels(model=model, status='success').inc()
                
                # Track matches
                if hasattr(result, 'matches'):
                    if result.matches:
                        face_recognition_matches.labels(model=model).inc()
                        
                        # Track confidence and distance
                        for match in result.matches:
                            face_recognition_confidence.labels(model=model).observe(match.confidence)
                            embedding_distance.labels(model=model).observe(match.distance)
                
                return result
                
            except Exception as e:
                face_recognitions_total.labels(model=model, status='error').inc()
                raise
        
        return wrapper
    return decorator


@contextmanager
def track_database_query(operation: str):
    """Context manager to track database queries"""
    start_time = time.time()
    
    try:
        yield
        
        # Track successful query
        duration = time.time() - start_time
        database_query_duration_seconds.labels(operation=operation).observe(duration)
        database_queries_total.labels(operation=operation).inc()
        
    except Exception:
        # Still track failed queries
        duration = time.time() - start_time
        database_query_duration_seconds.labels(operation=operation).observe(duration)
        raise


# ============================================================================
# Middleware
# ============================================================================

class PrometheusMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for Prometheus metrics"""
    
    async def dispatch(self, request: Request, call_next):
        # Skip metrics endpoint
        if request.url.path == "/metrics":
            return await call_next(request)
        
        method = request.method
        endpoint = request.url.path
        
        # Track active requests
        http_requests_active.labels(method=method, endpoint=endpoint).inc()
        
        # Track request size
        if request.headers.get("content-length"):
            size = int(request.headers.get("content-length"))
            http_request_size_bytes.labels(method=method, endpoint=endpoint).observe(size)
        
        # Time the request
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Track response
            duration = time.time() - start_time
            status = response.status_code
            
            http_request_duration_seconds.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
            http_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status=status
            ).inc()
            
            # Track response size
            if hasattr(response, "headers"):
                if "content-length" in response.headers:
                    size = int(response.headers["content-length"])
                    http_response_size_bytes.labels(
                        method=method,
                        endpoint=endpoint
                    ).observe(size)
            
            return response
            
        finally:
            http_requests_active.labels(method=method, endpoint=endpoint).dec()


# ============================================================================
# Metrics Endpoint
# ============================================================================

def setup_metrics_endpoint(app: FastAPI, collector: MetricsCollector):
    """Setup metrics endpoint"""
    
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint"""
        # Update system metrics before exposing
        collector.update_system_metrics()
        collector.update_gpu_metrics()
        
        # Generate metrics
        data = generate_latest(registry)
        
        return Response(content=data, media_type=CONTENT_TYPE_LATEST)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from fastapi import FastAPI
    import uvicorn
    
    app = FastAPI()
    
    # Initialize metrics collector
    collector = MetricsCollector()
    
    # Add middleware
    app.add_middleware(PrometheusMiddleware)
    
    # Setup metrics endpoint
    setup_metrics_endpoint(app, collector)
    
    # Set app info
    app_info.info({
        'version': '1.0.0',
        'name': 'Face Recognition API',
        'environment': 'production'
    })
    
    @app.get("/")
    async def root():
        return {"message": "Metrics exporter running"}
    
    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8000)
