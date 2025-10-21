# Performance Profiling and Diagnostics System

import logging
import time
import tracemalloc
import psutil
import threading
from typing import Callable, Any, Optional, Dict, List, Tuple
from functools import wraps
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, deque
import json
import io
import sys
from contextlib import redirect_stdout, redirect_stderr

@dataclass
class PerformanceMetric:
    """Single performance metric."""
    name: str
    value: float
    unit: str
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'value': self.value,
            'unit': self.unit,
            'timestamp': datetime.fromtimestamp(self.timestamp).isoformat()
        }

@dataclass
class FunctionProfile:
    """Profile data for a function."""
    function_name: str
    call_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    memory_peak: float = 0.0
    memory_avg: float = 0.0
    errors: int = 0
    
    def update_stats(self):
        """Update average and stats."""
        if self.call_count > 0:
            self.avg_time = self.total_time / self.call_count

class PerformanceProfiler:
    """Comprehensive performance profiler."""
    
    def __init__(self, enable_memory_tracking: bool = True):
        self.enable_memory_tracking = enable_memory_tracking
        self.profiles: Dict[str, FunctionProfile] = {}
        self.metrics: deque = deque(maxlen=10000)
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile function execution."""
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = 0
            
            if self.enable_memory_tracking:
                tracemalloc.start()
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            try:
                result = func(*args, **kwargs)
                return result
            
            except Exception as e:
                with self.lock:
                    if func.__name__ not in self.profiles:
                        self.profiles[func.__name__] = FunctionProfile(func.__name__)
                    self.profiles[func.__name__].errors += 1
                raise
            
            finally:
                elapsed_time = time.time() - start_time
                peak_memory = 0
                
                if self.enable_memory_tracking:
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    peak_memory = peak / 1024 / 1024  # MB
                
                with self.lock:
                    if func.__name__ not in self.profiles:
                        self.profiles[func.__name__] = FunctionProfile(func.__name__)
                    
                    profile = self.profiles[func.__name__]
                    profile.call_count += 1
                    profile.total_time += elapsed_time
                    profile.min_time = min(profile.min_time, elapsed_time)
                    profile.max_time = max(profile.max_time, elapsed_time)
                    profile.memory_peak = max(profile.memory_peak, peak_memory)
                    profile.update_stats()
                    
                    # Record metric
                    self.metrics.append(PerformanceMetric(
                        name=func.__name__,
                        value=elapsed_time,
                        unit='seconds'
                    ))
        
        return wrapper
    
    def get_profile(self, func_name: str) -> Optional[FunctionProfile]:
        """Get profile for function."""
        with self.lock:
            return self.profiles.get(func_name)
    
    def get_all_profiles(self) -> Dict[str, FunctionProfile]:
        """Get all profiles."""
        with self.lock:
            return dict(self.profiles)
    
    def reset(self):
        """Reset all profiles."""
        with self.lock:
            self.profiles.clear()
            self.metrics.clear()

class ResourceMonitor:
    """Monitor system resources."""
    
    def __init__(self, check_interval: float = 1.0):
        self.check_interval = check_interval
        self.process = psutil.Process()
        self.history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=3600))
        self.is_monitoring = False
        self.monitor_thread = None
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def start(self):
        """Start resource monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info("Resource monitor started")
    
    def stop(self):
        """Stop resource monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("Resource monitor stopped")
    
    def _monitor_loop(self):
        """Monitor loop."""
        while self.is_monitoring:
            try:
                timestamp = time.time()
                
                # CPU usage
                cpu_percent = self.process.cpu_percent(interval=0.1)
                
                # Memory usage
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                # Disk I/O
                io_counters = self.process.io_counters()
                
                with self.lock:
                    self.history['cpu_percent'].append((timestamp, cpu_percent))
                    self.history['memory_mb'].append((timestamp, memory_mb))
                    self.history['disk_read_mb'].append((timestamp, io_counters.read_bytes / 1024 / 1024))
                    self.history['disk_write_mb'].append((timestamp, io_counters.write_bytes / 1024 / 1024))
                
                time.sleep(self.check_interval)
            
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current resource statistics."""
        try:
            memory_info = self.process.memory_info()
            cpu_percent = self.process.cpu_percent(interval=0.1)
            
            return {
                'cpu_percent': cpu_percent,
                'memory_mb': memory_info.rss / 1024 / 1024,
                'memory_percent': self.process.memory_percent(),
                'num_threads': self.process.num_threads(),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting stats: {e}")
            return {}
    
    def get_history(self, metric: str, seconds: int = 300) -> List[Tuple[float, float]]:
        """Get metric history."""
        with self.lock:
            if metric not in self.history:
                return []
            
            cutoff_time = time.time() - seconds
            return [(t, v) for t, v in self.history[metric] if t >= cutoff_time]

class BottleneckDetector:
    """Detect performance bottlenecks."""
    
    def __init__(self, profiler: PerformanceProfiler, 
                 latency_threshold_ms: float = 100):
        self.profiler = profiler
        self.latency_threshold_ms = latency_threshold_ms
        self.logger = logging.getLogger(__name__)
    
    def detect_bottlenecks(self) -> Dict[str, Any]:
        """Detect performance bottlenecks."""
        profiles = self.profiler.get_all_profiles()
        bottlenecks = []
        
        for func_name, profile in profiles.items():
            if profile.avg_time * 1000 > self.latency_threshold_ms:
                bottlenecks.append({
                    'function': func_name,
                    'avg_time_ms': profile.avg_time * 1000,
                    'max_time_ms': profile.max_time * 1000,
                    'call_count': profile.call_count,
                    'total_time_ms': profile.total_time * 1000,
                    'memory_peak_mb': profile.memory_peak
                })
        
        # Sort by total time
        bottlenecks.sort(key=lambda x: x['total_time_ms'], reverse=True)
        
        return {
            'bottlenecks': bottlenecks,
            'bottleneck_count': len(bottlenecks),
            'most_expensive': bottlenecks[0] if bottlenecks else None
        }
    
    def get_recommendations(self) -> List[str]:
        """Get optimization recommendations."""
        recommendations = []
        profiles = self.profiler.get_all_profiles()
        bottlenecks = self.detect_bottlenecks()
        
        if bottlenecks['bottleneck_count'] > 0:
            recommendations.append(
                f"Optimize {bottlenecks['bottleneck_count']} bottleneck functions"
            )
            
            # Check for high memory usage
            high_memory_funcs = [p for p in profiles.values() 
                               if p.memory_peak > 500]  # 500 MB
            if high_memory_funcs:
                recommendations.append(
                    f"Consider memory optimization for {len(high_memory_funcs)} "
                    f"functions with high memory usage"
                )
            
            # Check for high error rates
            error_funcs = [p for p in profiles.values() if p.errors > 0]
            if error_funcs:
                recommendations.append(
                    f"Investigate error handling in {len(error_funcs)} functions"
                )
        
        return recommendations

class DiagnosticsCollector:
    """Collect comprehensive diagnostics."""
    
    def __init__(self, profiler: PerformanceProfiler,
                 resource_monitor: ResourceMonitor):
        self.profiler = profiler
        self.resource_monitor = resource_monitor
        self.logger = logging.getLogger(__name__)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive diagnostic report."""
        
        profiles = self.profiler.get_all_profiles()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'function_profiles': {
                name: {
                    'call_count': p.call_count,
                    'total_time_sec': p.total_time,
                    'avg_time_ms': p.avg_time * 1000,
                    'min_time_ms': p.min_time * 1000,
                    'max_time_ms': p.max_time * 1000,
                    'memory_peak_mb': p.memory_peak,
                    'errors': p.errors
                }
                for name, p in profiles.items()
            },
            'system_resources': self.resource_monitor.get_current_stats(),
            'bottleneck_analysis': BottleneckDetector(self.profiler).detect_bottlenecks()
        }
        
        return report
    
    def save_report(self, filepath: str):
        """Save diagnostics report to file."""
        report = self.generate_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Diagnostic report saved to {filepath}")

# Global instances
_profiler = PerformanceProfiler()
_resource_monitor = ResourceMonitor()

def profile(func: Callable) -> Callable:
    """Global profiling decorator."""
    return _profiler.profile_function(func)

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    @profile
    def slow_operation():
        """Simulate slow operation."""
        time.sleep(0.5)
        return "done"
    
    @profile
    def memory_intensive_operation():
        """Simulate memory-intensive operation."""
        data = [i for i in range(1000000)]
        return len(data)
    
    # Run profiled functions
    slow_operation()
    memory_intensive_operation()
    
    # Start monitoring
    _resource_monitor.start()
    
    # Run again
    for _ in range(3):
        slow_operation()
    
    time.sleep(2)
    
    # Generate diagnostics
    collector = DiagnosticsCollector(_profiler, _resource_monitor)
    report = collector.generate_report()
    
    print("Diagnostic Report:")
    print(json.dumps(report, indent=2, default=str))
    
    # Check bottlenecks
    detector = BottleneckDetector(_profiler, latency_threshold_ms=100)
    print("\nRecommendations:")
    for rec in detector.get_recommendations():
        print(f"- {rec}")
    
    _resource_monitor.stop()