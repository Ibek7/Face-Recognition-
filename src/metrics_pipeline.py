# Metrics & Monitoring Pipeline System

import threading
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import json
from datetime import datetime, timedelta

class MetricType(Enum):
    """Type of metric."""
    COUNTER = "counter"        # Monotonically increasing
    GAUGE = "gauge"            # Point-in-time value
    HISTOGRAM = "histogram"    # Distribution of values
    SUMMARY = "summary"        # Aggregated statistics

@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: float
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'value': self.value,
            'tags': self.tags
        }

class Counter:
    """Counter metric - monotonically increasing."""
    
    def __init__(self, name: str, help_text: str = ""):
        self.name = name
        self.help_text = help_text
        self.value = 0
        self.lock = threading.RLock()
    
    def increment(self, amount: float = 1) -> None:
        """Increment counter."""
        with self.lock:
            self.value += amount
    
    def get_value(self) -> float:
        """Get counter value."""
        with self.lock:
            return self.value
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'type': 'counter',
            'value': self.get_value()
        }

class Gauge:
    """Gauge metric - point-in-time value."""
    
    def __init__(self, name: str, help_text: str = ""):
        self.name = name
        self.help_text = help_text
        self.value = 0
        self.history: deque = deque(maxlen=1000)
        self.lock = threading.RLock()
    
    def set_value(self, value: float) -> None:
        """Set gauge value."""
        with self.lock:
            self.value = value
            self.history.append(MetricPoint(time.time(), value))
    
    def increment(self, amount: float = 1) -> None:
        """Increment gauge."""
        with self.lock:
            self.value += amount
            self.history.append(MetricPoint(time.time(), self.value))
    
    def decrement(self, amount: float = 1) -> None:
        """Decrement gauge."""
        with self.lock:
            self.value -= amount
            self.history.append(MetricPoint(time.time(), self.value))
    
    def get_value(self) -> float:
        """Get gauge value."""
        with self.lock:
            return self.value
    
    def get_average(self) -> float:
        """Get average value."""
        with self.lock:
            if not self.history:
                return 0
            return sum(p.value for p in self.history) / len(self.history)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'type': 'gauge',
            'value': self.get_value(),
            'average': self.get_average()
        }

class Histogram:
    """Histogram metric - distribution of values."""
    
    def __init__(self, name: str, buckets: List[float] = None, help_text: str = ""):
        self.name = name
        self.help_text = help_text
        self.buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
        self.bucket_counts: Dict[float, int] = {b: 0 for b in self.buckets}
        self.sum = 0
        self.count = 0
        self.lock = threading.RLock()
    
    def observe(self, value: float) -> None:
        """Record observation."""
        with self.lock:
            self.sum += value
            self.count += 1
            
            # Update bucket counts
            for bucket in self.buckets:
                if value <= bucket:
                    self.bucket_counts[bucket] += 1
    
    def get_statistics(self) -> Dict:
        """Get histogram statistics."""
        with self.lock:
            return {
                'count': self.count,
                'sum': self.sum,
                'average': self.sum / self.count if self.count > 0 else 0,
                'buckets': self.bucket_counts
            }
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        stats = self.get_statistics()
        return {
            'name': self.name,
            'type': 'histogram',
            'count': stats['count'],
            'sum': stats['sum'],
            'average': stats['average'],
            'buckets': stats['buckets']
        }

class Summary:
    """Summary metric - aggregated statistics."""
    
    def __init__(self, name: str, help_text: str = ""):
        self.name = name
        self.help_text = help_text
        self.values: deque = deque(maxlen=10000)
        self.lock = threading.RLock()
    
    def observe(self, value: float) -> None:
        """Record observation."""
        with self.lock:
            self.values.append(value)
    
    def get_percentile(self, percentile: float) -> float:
        """Get percentile."""
        with self.lock:
            if not self.values:
                return 0
            
            sorted_values = sorted(self.values)
            index = int(len(sorted_values) * percentile / 100)
            return sorted_values[index]
    
    def get_statistics(self) -> Dict:
        """Get summary statistics."""
        with self.lock:
            if not self.values:
                return {
                    'count': 0,
                    'sum': 0,
                    'average': 0,
                    'p50': 0,
                    'p95': 0,
                    'p99': 0
                }
            
            sorted_values = sorted(self.values)
            return {
                'count': len(self.values),
                'sum': sum(self.values),
                'average': sum(self.values) / len(self.values),
                'p50': sorted_values[int(len(sorted_values) * 0.5)],
                'p95': sorted_values[int(len(sorted_values) * 0.95)],
                'p99': sorted_values[int(len(sorted_values) * 0.99)]
            }
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        stats = self.get_statistics()
        return {
            'name': self.name,
            'type': 'summary',
            'count': stats['count'],
            'sum': stats['sum'],
            'average': stats['average'],
            'percentiles': {
                'p50': stats['p50'],
                'p95': stats['p95'],
                'p99': stats['p99']
            }
        }

class MetricsRegistry:
    """Central metrics registry."""
    
    def __init__(self):
        self.counters: Dict[str, Counter] = {}
        self.gauges: Dict[str, Gauge] = {}
        self.histograms: Dict[str, Histogram] = {}
        self.summaries: Dict[str, Summary] = {}
        self.lock = threading.RLock()
    
    def register_counter(self, name: str, help_text: str = "") -> Counter:
        """Register counter."""
        with self.lock:
            if name not in self.counters:
                self.counters[name] = Counter(name, help_text)
            return self.counters[name]
    
    def register_gauge(self, name: str, help_text: str = "") -> Gauge:
        """Register gauge."""
        with self.lock:
            if name not in self.gauges:
                self.gauges[name] = Gauge(name, help_text)
            return self.gauges[name]
    
    def register_histogram(self, name: str, buckets: List[float] = None,
                          help_text: str = "") -> Histogram:
        """Register histogram."""
        with self.lock:
            if name not in self.histograms:
                self.histograms[name] = Histogram(name, buckets, help_text)
            return self.histograms[name]
    
    def register_summary(self, name: str, help_text: str = "") -> Summary:
        """Register summary."""
        with self.lock:
            if name not in self.summaries:
                self.summaries[name] = Summary(name, help_text)
            return self.summaries[name]
    
    def get_all_metrics(self) -> Dict:
        """Get all metrics."""
        with self.lock:
            return {
                'counters': {name: m.to_dict() for name, m in self.counters.items()},
                'gauges': {name: m.to_dict() for name, m in self.gauges.items()},
                'histograms': {name: m.to_dict() for name, m in self.histograms.items()},
                'summaries': {name: m.to_dict() for name, m in self.summaries.items()}
            }

class AlertingRule:
    """Alerting rule for metrics."""
    
    def __init__(self, name: str, metric_name: str, condition: Callable, 
                 message: str, severity: str = "warning"):
        self.name = name
        self.metric_name = metric_name
        self.condition = condition
        self.message = message
        self.severity = severity
        self.triggered = False
        self.last_triggered = None
    
    def check(self, metric_value: float) -> Optional[str]:
        """Check if alert should trigger."""
        if self.condition(metric_value):
            if not self.triggered:
                self.triggered = True
                self.last_triggered = datetime.now()
                return f"[{self.severity.upper()}] {self.message}"
            return None
        else:
            self.triggered = False
            return None

class MonitoringPipeline:
    """Complete monitoring pipeline."""
    
    def __init__(self):
        self.registry = MetricsRegistry()
        self.rules: Dict[str, AlertingRule] = {}
        self.alert_handlers: List[Callable] = []
        self.lock = threading.RLock()
    
    def register_alert_rule(self, rule: AlertingRule) -> None:
        """Register alerting rule."""
        with self.lock:
            self.rules[rule.name] = rule
    
    def add_alert_handler(self, handler: Callable) -> None:
        """Add alert handler."""
        with self.lock:
            self.alert_handlers.append(handler)
    
    def check_alerts(self) -> List[str]:
        """Check all alerts."""
        alerts = []
        metrics = self.registry.get_all_metrics()
        
        with self.lock:
            for rule in self.rules.values():
                # Get metric value
                metric_value = self._get_metric_value(metrics, rule.metric_name)
                
                if metric_value is not None:
                    alert = rule.check(metric_value)
                    if alert:
                        alerts.append(alert)
                        
                        # Call handlers
                        for handler in self.alert_handlers:
                            try:
                                handler(alert)
                            except Exception as e:
                                print(f"Error in alert handler: {e}")
        
        return alerts
    
    def _get_metric_value(self, metrics: Dict, metric_name: str) -> Optional[float]:
        """Get metric value from registry."""
        for metric_type in metrics.values():
            if metric_name in metric_type:
                return metric_type[metric_name].get('value')
        return None
    
    def get_report(self) -> Dict:
        """Get monitoring report."""
        with self.lock:
            return {
                'timestamp': datetime.now().isoformat(),
                'metrics': self.registry.get_all_metrics(),
                'alerts': self.rules
            }

# Example usage
if __name__ == "__main__":
    pipeline = MonitoringPipeline()
    
    # Create metrics
    requests_counter = pipeline.registry.register_counter("requests_total")
    response_time = pipeline.registry.register_histogram("response_time_ms")
    active_connections = pipeline.registry.register_gauge("active_connections")
    
    # Record some data
    requests_counter.increment(5)
    for ms in [10, 25, 50, 100, 150]:
        response_time.observe(ms)
    
    active_connections.set_value(42)
    
    # Create alert rule
    def high_response_time(value):
        return value > 100
    
    alert_rule = AlertingRule(
        "high_response_time",
        "response_time_ms",
        high_response_time,
        "Response time exceeding 100ms"
    )
    pipeline.register_alert_rule(alert_rule)
    
    # Check alerts
    alerts = pipeline.check_alerts()
    print(f"Alerts: {alerts}")
    
    # Get report
    report = pipeline.get_report()
    print(f"\nMetrics Report:")
    print(json.dumps(report, indent=2, default=str))
