"""
Application metrics aggregator for monitoring and observability.

Provides custom metric types and aggregation for application monitoring.
"""

import time
from typing import Dict, List, Optional, Callable
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of metrics."""
    
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class Metric:
    """Base metric class."""
    
    def __init__(self, name: str, description: str, labels: Optional[dict] = None):
        """
        Initialize metric.
        
        Args:
            name: Metric name
            description: Metric description
            labels: Metric labels
        """
        self.name = name
        self.description = description
        self.labels = labels or {}
        self.created_at = datetime.utcnow()


class Counter(Metric):
    """Counter metric - monotonically increasing."""
    
    def __init__(self, name: str, description: str, labels: Optional[dict] = None):
        """Initialize counter."""
        super().__init__(name, description, labels)
        self.value = 0
    
    def inc(self, amount: float = 1.0):
        """Increment counter."""
        if amount < 0:
            raise ValueError("Counter can only increase")
        self.value += amount
    
    def get(self) -> float:
        """Get current value."""
        return self.value


class Gauge(Metric):
    """Gauge metric - can go up and down."""
    
    def __init__(self, name: str, description: str, labels: Optional[dict] = None):
        """Initialize gauge."""
        super().__init__(name, description, labels)
        self.value = 0
    
    def set(self, value: float):
        """Set gauge value."""
        self.value = value
    
    def inc(self, amount: float = 1.0):
        """Increment gauge."""
        self.value += amount
    
    def dec(self, amount: float = 1.0):
        """Decrement gauge."""
        self.value -= amount
    
    def get(self) -> float:
        """Get current value."""
        return self.value


class Histogram(Metric):
    """Histogram metric - tracks distribution of values."""
    
    def __init__(
        self,
        name: str,
        description: str,
        buckets: Optional[List[float]] = None,
        labels: Optional[dict] = None
    ):
        """
        Initialize histogram.
        
        Args:
            name: Metric name
            description: Metric description
            buckets: Histogram buckets
            labels: Metric labels
        """
        super().__init__(name, description, labels)
        self.buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        self.observations = []
        self.sum = 0
        self.count = 0
    
    def observe(self, value: float):
        """Record observation."""
        self.observations.append(value)
        self.sum += value
        self.count += 1
    
    def get_distribution(self) -> Dict[float, int]:
        """Get distribution across buckets."""
        distribution = {bucket: 0 for bucket in self.buckets}
        
        for obs in self.observations:
            for bucket in self.buckets:
                if obs <= bucket:
                    distribution[bucket] += 1
        
        return distribution
    
    def get_stats(self) -> dict:
        """Get histogram statistics."""
        if not self.observations:
            return {
                "count": 0,
                "sum": 0,
                "avg": 0,
                "min": 0,
                "max": 0
            }
        
        return {
            "count": self.count,
            "sum": self.sum,
            "avg": self.sum / self.count,
            "min": min(self.observations),
            "max": max(self.observations)
        }


class Summary(Metric):
    """Summary metric - calculates quantiles."""
    
    def __init__(
        self,
        name: str,
        description: str,
        quantiles: Optional[List[float]] = None,
        labels: Optional[dict] = None
    ):
        """
        Initialize summary.
        
        Args:
            name: Metric name
            description: Metric description
            quantiles: Quantiles to track
            labels: Metric labels
        """
        super().__init__(name, description, labels)
        self.quantiles = quantiles or [0.5, 0.9, 0.95, 0.99]
        self.observations = []
        self.sum = 0
        self.count = 0
    
    def observe(self, value: float):
        """Record observation."""
        self.observations.append(value)
        self.sum += value
        self.count += 1
    
    def get_quantiles(self) -> Dict[float, float]:
        """Calculate quantile values."""
        if not self.observations:
            return {q: 0 for q in self.quantiles}
        
        sorted_obs = sorted(self.observations)
        results = {}
        
        for q in self.quantiles:
            index = int(len(sorted_obs) * q)
            results[q] = sorted_obs[min(index, len(sorted_obs) - 1)]
        
        return results


class MetricsAggregator:
    """Centralized metrics aggregator."""
    
    def __init__(self):
        """Initialize metrics aggregator."""
        self.metrics: Dict[str, Metric] = {}
        self.metric_history = defaultdict(list)
        self.max_history_size = 1000
    
    def counter(
        self,
        name: str,
        description: str,
        labels: Optional[dict] = None
    ) -> Counter:
        """
        Get or create counter metric.
        
        Args:
            name: Metric name
            description: Metric description
            labels: Metric labels
        
        Returns:
            Counter instance
        """
        key = self._get_key(name, labels)
        
        if key not in self.metrics:
            self.metrics[key] = Counter(name, description, labels)
        
        return self.metrics[key]
    
    def gauge(
        self,
        name: str,
        description: str,
        labels: Optional[dict] = None
    ) -> Gauge:
        """Get or create gauge metric."""
        key = self._get_key(name, labels)
        
        if key not in self.metrics:
            self.metrics[key] = Gauge(name, description, labels)
        
        return self.metrics[key]
    
    def histogram(
        self,
        name: str,
        description: str,
        buckets: Optional[List[float]] = None,
        labels: Optional[dict] = None
    ) -> Histogram:
        """Get or create histogram metric."""
        key = self._get_key(name, labels)
        
        if key not in self.metrics:
            self.metrics[key] = Histogram(name, description, buckets, labels)
        
        return self.metrics[key]
    
    def summary(
        self,
        name: str,
        description: str,
        quantiles: Optional[List[float]] = None,
        labels: Optional[dict] = None
    ) -> Summary:
        """Get or create summary metric."""
        key = self._get_key(name, labels)
        
        if key not in self.metrics:
            self.metrics[key] = Summary(name, description, quantiles, labels)
        
        return self.metrics[key]
    
    def _get_key(self, name: str, labels: Optional[dict]) -> str:
        """Generate metric key."""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def record_timing(self, name: str, duration: float, labels: Optional[dict] = None):
        """Record timing metric."""
        hist = self.histogram(
            name=f"{name}_duration_seconds",
            description=f"Duration of {name}",
            labels=labels
        )
        hist.observe(duration)
    
    def get_all_metrics(self) -> Dict[str, dict]:
        """Get all metrics."""
        result = {}
        
        for key, metric in self.metrics.items():
            if isinstance(metric, Counter):
                result[key] = {
                    "type": "counter",
                    "value": metric.get()
                }
            elif isinstance(metric, Gauge):
                result[key] = {
                    "type": "gauge",
                    "value": metric.get()
                }
            elif isinstance(metric, Histogram):
                result[key] = {
                    "type": "histogram",
                    "stats": metric.get_stats(),
                    "distribution": metric.get_distribution()
                }
            elif isinstance(metric, Summary):
                result[key] = {
                    "type": "summary",
                    "quantiles": metric.get_quantiles(),
                    "count": metric.count,
                    "sum": metric.sum
                }
        
        return result


# Global metrics aggregator
metrics_aggregator = MetricsAggregator()


# Decorator for timing functions
def timed(metric_name: str):
    """Decorator to time function execution."""
    def decorator(func: Callable):
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start
                metrics_aggregator.record_timing(metric_name, duration)
        
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start
                metrics_aggregator.record_timing(metric_name, duration)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Example usage:
"""
from fastapi import FastAPI
from src.metrics_aggregator import metrics_aggregator, timed

app = FastAPI()

# Counter example
requests_counter = metrics_aggregator.counter(
    "http_requests_total",
    "Total HTTP requests",
    labels={"method": "GET", "endpoint": "/api/data"}
)

@app.get("/api/data")
@timed("api_data_request")
async def get_data():
    requests_counter.inc()
    return {"data": "result"}

# Gauge example
active_connections = metrics_aggregator.gauge(
    "active_connections",
    "Number of active connections"
)

@app.on_event("startup")
async def startup():
    active_connections.inc()

# Get all metrics
@app.get("/metrics")
async def get_metrics():
    return metrics_aggregator.get_all_metrics()
"""
