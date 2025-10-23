# Observability & Metrics Collection System

import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import statistics
import json

class MetricUnit(Enum):
    """Metric measurement units."""
    SECONDS = "s"
    MILLISECONDS = "ms"
    MICROSECONDS = "us"
    BYTES = "B"
    KILOBYTES = "KB"
    MEGABYTES = "MB"
    PERCENTAGE = "%"
    COUNT = "count"
    OPS_PER_SEC = "ops/sec"

class AggregationType(Enum):
    """Metric aggregation types."""
    SUM = "sum"
    AVERAGE = "average"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    COUNTER = "counter"

@dataclass
class Metric:
    """Single metric measurement."""
    name: str
    value: float
    unit: MetricUnit
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    aggregation_type: AggregationType = AggregationType.GAUGE
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'value': self.value,
            'unit': self.unit.value,
            'timestamp': self.timestamp,
            'tags': self.tags,
            'aggregation_type': self.aggregation_type.value
        }

@dataclass
class MetricSnapshot:
    """Snapshot of metric statistics."""
    name: str
    count: int
    sum: float
    min: float
    max: float
    mean: float
    median: float
    stddev: float
    p95: float
    p99: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'count': self.count,
            'sum': self.sum,
            'min': self.min,
            'max': self.max,
            'mean': self.mean,
            'median': self.median,
            'stddev': self.stddev,
            'p95': self.p95,
            'p99': self.p99
        }

class MetricCollector:
    """Collect and aggregate metrics."""
    
    def __init__(self, name: str, retention_size: int = 10000):
        self.name = name
        self.metrics: List[Metric] = []
        self.retention_size = retention_size
        self.lock = threading.RLock()
    
    def record(self, metric: Metric):
        """Record metric."""
        with self.lock:
            self.metrics.append(metric)
            
            # Trim to retention size
            if len(self.metrics) > self.retention_size:
                self.metrics = self.metrics[-self.retention_size:]
    
    def get_snapshot(self) -> Optional[MetricSnapshot]:
        """Get metric statistics."""
        with self.lock:
            if not self.metrics:
                return None
            
            values = [m.value for m in self.metrics]
            
            sorted_values = sorted(values)
            p95_index = int(len(sorted_values) * 0.95)
            p99_index = int(len(sorted_values) * 0.99)
            
            return MetricSnapshot(
                name=self.name,
                count=len(values),
                sum=sum(values),
                min=min(values),
                max=max(values),
                mean=statistics.mean(values),
                median=statistics.median(values),
                stddev=statistics.stdev(values) if len(values) > 1 else 0,
                p95=sorted_values[p95_index] if p95_index < len(sorted_values) else 0,
                p99=sorted_values[p99_index] if p99_index < len(sorted_values) else 0
            )
    
    def reset(self):
        """Reset metrics."""
        with self.lock:
            self.metrics.clear()

class MetricsRegistry:
    """Registry for all metrics."""
    
    def __init__(self):
        self.collectors: Dict[str, MetricCollector] = {}
        self.lock = threading.RLock()
    
    def register_collector(self, name: str, 
                          retention_size: int = 10000) -> MetricCollector:
        """Register metric collector."""
        with self.lock:
            if name not in self.collectors:
                self.collectors[name] = MetricCollector(name, retention_size)
            
            return self.collectors[name]
    
    def record_metric(self, collector_name: str, value: float,
                     unit: MetricUnit, tags: Dict = None,
                     aggregation: AggregationType = AggregationType.GAUGE):
        """Record metric."""
        collector = self.register_collector(collector_name)
        
        metric = Metric(
            name=collector_name,
            value=value,
            unit=unit,
            tags=tags or {},
            aggregation_type=aggregation
        )
        
        collector.record(metric)
    
    def get_all_snapshots(self) -> Dict[str, MetricSnapshot]:
        """Get snapshots for all collectors."""
        with self.lock:
            snapshots = {}
            
            for name, collector in self.collectors.items():
                snapshot = collector.get_snapshot()
                if snapshot:
                    snapshots[name] = snapshot
            
            return snapshots
    
    def export_metrics(self) -> Dict:
        """Export all metrics."""
        snapshots = self.get_all_snapshots()
        
        return {
            'timestamp': time.time(),
            'metrics': {name: snap.to_dict() 
                       for name, snap in snapshots.items()}
        }

class ObservabilityContext:
    """Context for observability operations."""
    
    def __init__(self):
        self.registry = MetricsRegistry()
        self.events: List[Dict] = []
        self.lock = threading.RLock()
    
    def record_latency(self, operation_name: str, duration_ms: float,
                      tags: Dict = None):
        """Record operation latency."""
        self.registry.record_metric(
            f"{operation_name}.latency",
            duration_ms,
            MetricUnit.MILLISECONDS,
            tags,
            AggregationType.HISTOGRAM
        )
    
    def record_throughput(self, operation_name: str, count: float,
                         tags: Dict = None):
        """Record operation throughput."""
        self.registry.record_metric(
            f"{operation_name}.throughput",
            count,
            MetricUnit.OPS_PER_SEC,
            tags,
            AggregationType.COUNTER
        )
    
    def record_error(self, operation_name: str, error_type: str,
                    tags: Dict = None):
        """Record error."""
        tags = tags or {}
        tags['error_type'] = error_type
        
        self.registry.record_metric(
            f"{operation_name}.errors",
            1.0,
            MetricUnit.COUNT,
            tags,
            AggregationType.COUNTER
        )
    
    def record_event(self, event_type: str, message: str, 
                    severity: str = "info"):
        """Record event."""
        with self.lock:
            self.events.append({
                'timestamp': time.time(),
                'event_type': event_type,
                'message': message,
                'severity': severity
            })
    
    def get_health_status(self) -> Dict:
        """Get system health status."""
        snapshots = self.registry.get_all_snapshots()
        
        health = {
            'healthy': True,
            'metrics': {}
        }
        
        for name, snapshot in snapshots.items():
            # Simple health check: if error rate > 5%
            if 'error' in name and snapshot.mean > 5:
                health['healthy'] = False
            
            health['metrics'][name] = {
                'mean': snapshot.mean,
                'p95': snapshot.p95,
                'p99': snapshot.p99
            }
        
        return health

class MetricsExporter(ABC):
    """Base class for metrics exporters."""
    
    @abstractmethod
    def export(self, metrics: Dict) -> bool:
        """Export metrics. Return True if successful."""
        pass

class PrometheusExporter(MetricsExporter):
    """Export metrics in Prometheus format."""
    
    def export(self, metrics: Dict) -> bool:
        """Export to Prometheus format."""
        lines = []
        
        for metric_name, data in metrics.get('metrics', {}).items():
            lines.append(f"# TYPE {metric_name} gauge")
            lines.append(f"{metric_name}{{}} {data['mean']}")
        
        return '\n'.join(lines)

class JSONExporter(MetricsExporter):
    """Export metrics as JSON."""
    
    def export(self, metrics: Dict) -> str:
        """Export as JSON."""
        return json.dumps(metrics, indent=2)

class AlertRule:
    """Alert rule for metrics."""
    
    def __init__(self, name: str, condition: Callable, 
                severity: str = "warning"):
        self.name = name
        self.condition = condition
        self.severity = severity
    
    def evaluate(self, metrics: Dict) -> Optional[str]:
        """Evaluate rule. Return alert message if triggered."""
        try:
            if self.condition(metrics):
                return f"Alert [{self.severity}]: {self.name}"
            return None
        except Exception:
            return None

class AlertManager:
    """Manage alerts based on metrics."""
    
    def __init__(self):
        self.rules: List[AlertRule] = []
        self.alerts: List[str] = []
        self.lock = threading.RLock()
    
    def add_rule(self, rule: AlertRule):
        """Add alert rule."""
        self.rules.append(rule)
    
    def evaluate_alerts(self, metrics: Dict) -> List[str]:
        """Evaluate all rules."""
        alerts = []
        
        for rule in self.rules:
            alert = rule.evaluate(metrics)
            if alert:
                alerts.append(alert)
        
        with self.lock:
            self.alerts.extend(alerts)
        
        return alerts
    
    def get_recent_alerts(self, limit: int = 100) -> List[str]:
        """Get recent alerts."""
        with self.lock:
            return self.alerts[-limit:]

# Example usage
if __name__ == "__main__":
    from abc import ABC, abstractmethod
    
    # Create observability context
    obs = ObservabilityContext()
    
    # Record metrics
    for i in range(100):
        latency = 50 + (i % 20)
        obs.record_latency("face_detection", latency)
    
    # Get snapshots
    snapshots = obs.registry.get_all_snapshots()
    
    for name, snapshot in snapshots.items():
        print(f"Metric: {name}")
        print(f"  Mean: {snapshot.mean:.2f}ms")
        print(f"  P95: {snapshot.p95:.2f}ms")
        print(f"  P99: {snapshot.p99:.2f}ms")
    
    # Export metrics
    metrics = obs.registry.export_metrics()
    exporter = JSONExporter()
    print(f"\nJSON Export:\n{exporter.export(metrics)}")
