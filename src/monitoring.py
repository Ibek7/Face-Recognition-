"""
Performance monitoring and metrics collection for face recognition system.
Tracks timing, accuracy, and system resource usage.
"""

import time
import psutil
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import threading
import logging

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Single performance measurement."""
    name: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemMetrics:
    """System resource metrics."""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    gpu_percent: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)

class PerformanceMonitor:
    """Performance monitoring system for face recognition operations."""
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize performance monitor.
        
        Args:
            max_history: Maximum number of metrics to keep in memory
        """
        self.max_history = max_history
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.system_metrics = deque(maxlen=max_history)
        
        # Timing contexts
        self.active_timers: Dict[str, float] = {}
        
        # System monitoring
        self.monitoring = False
        self.monitor_thread = None
        self.monitor_interval = 1.0  # seconds
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
    def start_system_monitoring(self, interval: float = 1.0) -> None:
        """
        Start background system resource monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self.monitoring:
            return
            
        self.monitor_interval = interval
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitor_thread.start()
        logger.info("Started system monitoring")
    
    def stop_system_monitoring(self) -> None:
        """Stop background system monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Stopped system monitoring")
    
    def _monitor_system(self) -> None:
        """Background thread for system monitoring."""
        while self.monitoring:
            try:
                # Get system metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                
                # Try to get GPU metrics (requires additional packages)
                gpu_percent = None
                gpu_memory = None
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # Use first GPU
                        gpu_percent = gpu.load * 100
                        gpu_memory = gpu.memoryUsed
                except ImportError:
                    pass  # GPU monitoring not available
                
                metrics = SystemMetrics(
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_used_mb=memory.used / (1024 * 1024),
                    gpu_percent=gpu_percent,
                    gpu_memory_mb=gpu_memory
                )
                
                with self.lock:
                    self.system_metrics.append(metrics)
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                time.sleep(self.monitor_interval)
    
    def record_metric(self, name: str, value: float, unit: str = "", **metadata) -> None:
        """
        Record a performance metric.
        
        Args:
            name: Metric name
            value: Metric value
            unit: Unit of measurement
            **metadata: Additional metadata
        """
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            metadata=metadata
        )
        
        with self.lock:
            self.metrics[name].append(metric)
    
    def start_timer(self, name: str) -> None:
        """Start a timing measurement."""
        self.active_timers[name] = time.time()
    
    def end_timer(self, name: str, **metadata) -> float:
        """
        End a timing measurement and record the duration.
        
        Args:
            name: Timer name
            **metadata: Additional metadata
            
        Returns:
            Elapsed time in seconds
        """
        if name not in self.active_timers:
            logger.warning(f"Timer '{name}' was not started")
            return 0.0
        
        elapsed = time.time() - self.active_timers[name]
        del self.active_timers[name]
        
        self.record_metric(name, elapsed, "seconds", **metadata)
        return elapsed
    
    def time_function(self, name: str, **metadata):
        """
        Decorator for timing function execution.
        
        Args:
            name: Metric name
            **metadata: Additional metadata
        """
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                self.start_timer(name)
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    self.end_timer(name, **metadata)
            return wrapper
        return decorator
    
    def get_metric_statistics(self, name: str) -> Dict[str, float]:
        """
        Get statistical summary of a metric.
        
        Args:
            name: Metric name
            
        Returns:
            Dictionary with statistical measures
        """
        with self.lock:
            if name not in self.metrics or not self.metrics[name]:
                return {}
            
            values = [m.value for m in self.metrics[name]]
            
            return {
                'count': len(values),
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'p95': np.percentile(values, 95),
                'p99': np.percentile(values, 99)
            }
    
    def get_recent_metrics(self, name: str, duration_minutes: int = 10) -> List[PerformanceMetric]:
        """
        Get recent metrics within specified duration.
        
        Args:
            name: Metric name
            duration_minutes: Duration in minutes
            
        Returns:
            List of recent metrics
        """
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        
        with self.lock:
            if name not in self.metrics:
                return []
            
            return [m for m in self.metrics[name] if m.timestamp >= cutoff_time]
    
    def get_system_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of system metrics."""
        with self.lock:
            if not self.system_metrics:
                return {}
            
            cpu_values = [m.cpu_percent for m in self.system_metrics]
            memory_values = [m.memory_percent for m in self.system_metrics]
            
            summary = {
                'cpu': {
                    'current': cpu_values[-1] if cpu_values else 0,
                    'average': np.mean(cpu_values),
                    'max': np.max(cpu_values)
                },
                'memory': {
                    'current': memory_values[-1] if memory_values else 0,
                    'average': np.mean(memory_values),
                    'max': np.max(memory_values)
                }
            }
            
            # Add GPU metrics if available
            gpu_values = [m.gpu_percent for m in self.system_metrics if m.gpu_percent is not None]
            if gpu_values:
                summary['gpu'] = {
                    'current': gpu_values[-1],
                    'average': np.mean(gpu_values),
                    'max': np.max(gpu_values)
                }
            
            return summary
    
    def export_metrics(self, filepath: str) -> None:
        """
        Export metrics to JSON file.
        
        Args:
            filepath: Output file path
        """
        export_data = {
            'exported_at': datetime.now().isoformat(),
            'metrics': {},
            'system_metrics': []
        }
        
        with self.lock:
            # Export performance metrics
            for name, metric_list in self.metrics.items():
                export_data['metrics'][name] = []
                for metric in metric_list:
                    export_data['metrics'][name].append({
                        'value': metric.value,
                        'unit': metric.unit,
                        'timestamp': metric.timestamp.isoformat(),
                        'metadata': metric.metadata
                    })
            
            # Export system metrics
            for sys_metric in self.system_metrics:
                export_data['system_metrics'].append({
                    'cpu_percent': sys_metric.cpu_percent,
                    'memory_percent': sys_metric.memory_percent,
                    'memory_used_mb': sys_metric.memory_used_mb,
                    'gpu_percent': sys_metric.gpu_percent,
                    'gpu_memory_mb': sys_metric.gpu_memory_mb,
                    'timestamp': sys_metric.timestamp.isoformat()
                })
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported metrics to {filepath}")
    
    def generate_report(self) -> str:
        """Generate a text report of performance metrics."""
        report_lines = [
            "Face Recognition Performance Report",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        # System metrics summary
        sys_summary = self.get_system_metrics_summary()
        if sys_summary:
            report_lines.extend([
                "SYSTEM METRICS",
                "-" * 20,
                f"CPU: {sys_summary['cpu']['current']:.1f}% (avg: {sys_summary['cpu']['average']:.1f}%, max: {sys_summary['cpu']['max']:.1f}%)",
                f"Memory: {sys_summary['memory']['current']:.1f}% (avg: {sys_summary['memory']['average']:.1f}%, max: {sys_summary['memory']['max']:.1f}%)"
            ])
            
            if 'gpu' in sys_summary:
                report_lines.append(f"GPU: {sys_summary['gpu']['current']:.1f}% (avg: {sys_summary['gpu']['average']:.1f}%, max: {sys_summary['gpu']['max']:.1f}%)")
            
            report_lines.append("")
        
        # Performance metrics
        with self.lock:
            if self.metrics:
                report_lines.extend([
                    "PERFORMANCE METRICS",
                    "-" * 25
                ])
                
                for name in sorted(self.metrics.keys()):
                    stats = self.get_metric_statistics(name)
                    if stats:
                        unit = self.metrics[name][-1].unit if self.metrics[name] else ""
                        report_lines.extend([
                            f"{name}:",
                            f"  Count: {stats['count']}",
                            f"  Mean: {stats['mean']:.4f} {unit}",
                            f"  Median: {stats['median']:.4f} {unit}",
                            f"  Std: {stats['std']:.4f} {unit}",
                            f"  Min: {stats['min']:.4f} {unit}",
                            f"  Max: {stats['max']:.4f} {unit}",
                            f"  P95: {stats['p95']:.4f} {unit}",
                            f"  P99: {stats['p99']:.4f} {unit}",
                            ""
                        ])
        
        return "\\n".join(report_lines)
    
    def clear_metrics(self) -> None:
        """Clear all stored metrics."""
        with self.lock:
            self.metrics.clear()
            self.system_metrics.clear()
        logger.info("Cleared all metrics")

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

def monitor_performance(name: str, **metadata):
    """Decorator for monitoring function performance."""
    return performance_monitor.time_function(name, **metadata)