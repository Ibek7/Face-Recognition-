# Advanced Monitoring and Alerting System

import logging
import time
import threading
import json
from typing import Callable, List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import numpy as np

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = 0
    WARNING = 1
    ERROR = 2
    CRITICAL = 3

class AlertType(Enum):
    """Types of alerts."""
    PERFORMANCE = "performance"
    ERROR_RATE = "error_rate"
    RESOURCE = "resource"
    ANOMALY = "anomaly"
    THRESHOLD = "threshold"
    SLA_VIOLATION = "sla_violation"

@dataclass
class Alert:
    """Alert notification."""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    timestamp: float = field(default_factory=time.time)
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    acknowledged: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'alert_id': self.alert_id,
            'type': self.alert_type.value,
            'severity': self.severity.name,
            'message': self.message,
            'timestamp': datetime.fromtimestamp(self.timestamp).isoformat(),
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'threshold': self.threshold,
            'acknowledged': self.acknowledged
        }

@dataclass
class MetricThreshold:
    """Metric threshold configuration."""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    comparison_type: str  # 'gt', 'lt', 'eq'
    window_size_seconds: int = 60
    samples_required: int = 5

class MetricAnomalyDetector:
    """Detect anomalies in metrics using statistical methods."""
    
    def __init__(self, window_size: int = 100, z_score_threshold: float = 3.0):
        self.window_size = window_size
        self.z_score_threshold = z_score_threshold
        self.metric_history: Dict[str, deque] = {}
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def record_metric(self, metric_name: str, value: float):
        """Record metric value."""
        with self.lock:
            if metric_name not in self.metric_history:
                self.metric_history[metric_name] = deque(maxlen=self.window_size)
            
            self.metric_history[metric_name].append(value)
    
    def detect_anomaly(self, metric_name: str, value: float) -> tuple[bool, float]:
        """Detect if value is anomalous using z-score."""
        
        with self.lock:
            if metric_name not in self.metric_history:
                return False, 0.0
            
            history = list(self.metric_history[metric_name])
            
            if len(history) < 5:
                return False, 0.0
            
            mean = np.mean(history)
            std = np.std(history)
            
            if std == 0:
                return False, 0.0
            
            z_score = abs((value - mean) / std)
            is_anomalous = z_score > self.z_score_threshold
            
            return is_anomalous, z_score
    
    def get_metric_stats(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for metric."""
        with self.lock:
            if metric_name not in self.metric_history:
                return {}
            
            history = list(self.metric_history[metric_name])
            
            if not history:
                return {}
            
            return {
                'mean': float(np.mean(history)),
                'std': float(np.std(history)),
                'min': float(np.min(history)),
                'max': float(np.max(history)),
                'median': float(np.median(history))
            }

class ThresholdMonitor:
    """Monitor metrics against thresholds."""
    
    def __init__(self):
        self.thresholds: Dict[str, MetricThreshold] = {}
        self.metric_values: Dict[str, deque] = {}
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def register_threshold(self, threshold: MetricThreshold):
        """Register metric threshold."""
        with self.lock:
            self.thresholds[threshold.metric_name] = threshold
            self.metric_values[threshold.metric_name] = deque(
                maxlen=threshold.samples_required * 2
            )
    
    def record_value(self, metric_name: str, value: float) -> Optional[AlertSeverity]:
        """Record metric value and check against thresholds."""
        
        with self.lock:
            if metric_name not in self.thresholds:
                return None
            
            threshold = self.thresholds[metric_name]
            self.metric_values[metric_name].append(value)
            
            # Check if we have enough samples
            values = list(self.metric_values[metric_name])
            if len(values) < threshold.samples_required:
                return None
            
            # Check recent samples
            recent_values = values[-threshold.samples_required:]
            
            # Determine severity based on comparison
            critical_violations = 0
            warning_violations = 0
            
            for val in recent_values:
                if self._compare_value(val, threshold.critical_threshold, threshold.comparison_type):
                    critical_violations += 1
                elif self._compare_value(val, threshold.warning_threshold, threshold.comparison_type):
                    warning_violations += 1
            
            # Require majority of samples to trigger alert
            if critical_violations > threshold.samples_required / 2:
                return AlertSeverity.CRITICAL
            elif warning_violations > threshold.samples_required / 2:
                return AlertSeverity.WARNING
            
            return None
    
    def _compare_value(self, value: float, threshold: float, comparison_type: str) -> bool:
        """Compare value against threshold."""
        if comparison_type == 'gt':
            return value > threshold
        elif comparison_type == 'lt':
            return value < threshold
        elif comparison_type == 'eq':
            return abs(value - threshold) < 0.01
        return False

class AlertManager:
    """Manage alerts and notifications."""
    
    def __init__(self, max_alerts: int = 10000):
        self.alerts: deque = deque(maxlen=max_alerts)
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_handlers: List[Callable] = []
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def register_handler(self, handler: Callable):
        """Register alert handler."""
        self.alert_handlers.append(handler)
    
    def create_alert(self, alert_type: AlertType, severity: AlertSeverity,
                    message: str, metric_name: Optional[str] = None,
                    metric_value: Optional[float] = None,
                    threshold: Optional[float] = None) -> Alert:
        """Create and dispatch alert."""
        
        alert_id = f"alert_{int(time.time() * 1000)}"
        
        alert = Alert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            message=message,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold
        )
        
        with self.lock:
            self.alerts.append(alert)
            self.active_alerts[alert_id] = alert
        
        # Dispatch to handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Alert handler error: {e}")
        
        return alert
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge alert."""
        with self.lock:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].acknowledged = True
    
    def get_active_alerts(self) -> List[Alert]:
        """Get active unacknowledged alerts."""
        with self.lock:
            return [a for a in self.active_alerts.values() 
                   if not a.acknowledged]
    
    def get_alert_history(self, limit: int = 100,
                         alert_type: Optional[AlertType] = None) -> List[Dict]:
        """Get alert history."""
        with self.lock:
            alerts = list(self.alerts)
            
            if alert_type:
                alerts = [a for a in alerts if a.alert_type == alert_type]
            
            return [a.to_dict() for a in alerts[-limit:]]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        with self.lock:
            if not self.alerts:
                return {}
            
            alerts = list(self.alerts)
            
            critical_count = sum(1 for a in alerts if a.severity == AlertSeverity.CRITICAL)
            error_count = sum(1 for a in alerts if a.severity == AlertSeverity.ERROR)
            warning_count = sum(1 for a in alerts if a.severity == AlertSeverity.WARNING)
            
            alert_types = {}
            for alert in alerts:
                alert_types[alert.alert_type.value] = alert_types.get(alert.alert_type.value, 0) + 1
            
            return {
                'total_alerts': len(alerts),
                'active_alerts': len(self.active_alerts),
                'critical_count': critical_count,
                'error_count': error_count,
                'warning_count': warning_count,
                'alert_breakdown': alert_types
            }

class HealthMonitor:
    """Monitor system health."""
    
    def __init__(self, alert_manager: AlertManager):
        self.alert_manager = alert_manager
        self.anomaly_detector = MetricAnomalyDetector()
        self.threshold_monitor = ThresholdMonitor()
        self.logger = logging.getLogger(__name__)
        
        self.is_monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self, check_interval: float = 10.0):
        """Start health monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(check_interval,),
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("Health monitoring stopped")
    
    def _monitor_loop(self, check_interval: float):
        """Monitor loop."""
        while self.is_monitoring:
            try:
                # Simulate metric collection
                import psutil
                process = psutil.Process()
                
                # Memory usage
                memory_mb = process.memory_info().rss / 1024 / 1024
                self.anomaly_detector.record_metric('memory_mb', memory_mb)
                
                # CPU usage
                cpu_percent = process.cpu_percent(interval=0.1)
                self.anomaly_detector.record_metric('cpu_percent', cpu_percent)
                
                # Check for anomalies
                is_memory_anomaly, z_score = self.anomaly_detector.detect_anomaly(
                    'memory_mb', memory_mb
                )
                
                if is_memory_anomaly:
                    self.alert_manager.create_alert(
                        alert_type=AlertType.ANOMALY,
                        severity=AlertSeverity.WARNING,
                        message=f"Memory anomaly detected: {memory_mb:.1f}MB (z-score: {z_score:.2f})",
                        metric_name='memory_mb',
                        metric_value=memory_mb
                    )
                
                time.sleep(check_interval)
            
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
    
    def register_metric_threshold(self, threshold: MetricThreshold):
        """Register metric threshold."""
        self.threshold_monitor.register_threshold(threshold)
    
    def record_metric(self, metric_name: str, value: float):
        """Record metric value."""
        severity = self.threshold_monitor.record_value(metric_name, value)
        
        if severity:
            threshold = self.threshold_monitor.thresholds[metric_name]
            self.alert_manager.create_alert(
                alert_type=AlertType.THRESHOLD,
                severity=severity,
                message=f"{metric_name} exceeded threshold: {value:.2f}",
                metric_name=metric_name,
                metric_value=value,
                threshold=getattr(threshold, f'{severity.name.lower()}_threshold', None)
            )

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create alert manager
    alert_manager = AlertManager()
    
    # Register alert handler
    def log_alert(alert):
        print(f"🚨 Alert: [{alert.severity.name}] {alert.message}")
    
    alert_manager.register_handler(log_alert)
    
    # Create health monitor
    monitor = HealthMonitor(alert_manager)
    
    # Register thresholds
    monitor.register_metric_threshold(MetricThreshold(
        metric_name='response_time_ms',
        warning_threshold=100,
        critical_threshold=500,
        comparison_type='gt',
        samples_required=3
    ))
    
    # Simulate metric recording
    monitor.record_metric('response_time_ms', 50)
    monitor.record_metric('response_time_ms', 55)
    monitor.record_metric('response_time_ms', 60)
    monitor.record_metric('response_time_ms', 150)  # Warning
    monitor.record_metric('response_time_ms', 160)
    monitor.record_metric('response_time_ms', 170)  # Should trigger alert
    
    # Print statistics
    time.sleep(1)
    stats = alert_manager.get_alert_statistics()
    print(f"\nAlert Statistics:")
    print(json.dumps(stats, indent=2))
    
    # Print alert history
    print(f"\nAlert History:")
    history = alert_manager.get_alert_history(limit=5)
    for alert in history:
        print(f"  - {alert['severity']}: {alert['message']}")