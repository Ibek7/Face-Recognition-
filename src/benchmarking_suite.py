# Performance Optimization & Benchmarking Suite

import time
import threading
import statistics
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

class MetricType(Enum):
    """Types of metrics."""
    THROUGHPUT = "throughput"  # ops/sec
    LATENCY = "latency"  # ms
    MEMORY = "memory"  # MB
    CPU = "cpu"  # percentage
    ERROR_RATE = "error_rate"  # percentage

@dataclass
class MetricDataPoint:
    """Single metric data point."""
    timestamp: float
    value: float
    metric_type: MetricType
    label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'value': self.value,
            'metric_type': self.metric_type.value,
            'label': self.label,
            'metadata': self.metadata
        }

@dataclass
class BenchmarkResult:
    """Benchmark execution result."""
    name: str
    duration_ms: float
    iterations: int
    min_ms: float
    max_ms: float
    mean_ms: float
    median_ms: float
    stddev_ms: float
    throughput: float  # ops/sec
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'duration_ms': self.duration_ms,
            'iterations': self.iterations,
            'min_ms': self.min_ms,
            'max_ms': self.max_ms,
            'mean_ms': self.mean_ms,
            'median_ms': self.median_ms,
            'stddev_ms': self.stddev_ms,
            'throughput': self.throughput,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }

class Benchmark:
    """Single benchmark."""
    
    def __init__(self, name: str, func: Callable, iterations: int = 1000):
        self.name = name
        self.func = func
        self.iterations = iterations
        self.timings: List[float] = []
    
    def run(self) -> BenchmarkResult:
        """Run benchmark."""
        self.timings = []
        start_time = time.time()
        
        for _ in range(self.iterations):
            iter_start = time.time()
            self.func()
            iter_duration = (time.time() - iter_start) * 1000  # ms
            self.timings.append(iter_duration)
        
        total_duration = (time.time() - start_time) * 1000  # ms
        
        # Calculate statistics
        min_time = min(self.timings)
        max_time = max(self.timings)
        mean_time = statistics.mean(self.timings)
        median_time = statistics.median(self.timings)
        stddev_time = statistics.stdev(self.timings) if len(self.timings) > 1 else 0
        throughput = (self.iterations / (total_duration / 1000))  # ops/sec
        
        return BenchmarkResult(
            name=self.name,
            duration_ms=total_duration,
            iterations=self.iterations,
            min_ms=min_time,
            max_ms=max_time,
            mean_ms=mean_time,
            median_ms=median_time,
            stddev_ms=stddev_time,
            throughput=throughput
        )

class BenchmarkSuite:
    """Collection of benchmarks."""
    
    def __init__(self, name: str):
        self.name = name
        self.benchmarks: List[Benchmark] = []
        self.results: List[BenchmarkResult] = []
        self.lock = threading.RLock()
    
    def add_benchmark(self, name: str, func: Callable, iterations: int = 1000):
        """Add benchmark."""
        benchmark = Benchmark(name, func, iterations)
        self.benchmarks.append(benchmark)
    
    def run_all(self) -> List[BenchmarkResult]:
        """Run all benchmarks."""
        self.results = []
        
        for benchmark in self.benchmarks:
            result = benchmark.run()
            with self.lock:
                self.results.append(result)
        
        return self.results
    
    def get_summary(self) -> Dict:
        """Get benchmark summary."""
        with self.lock:
            if not self.results:
                return {'benchmarks': 0}
            
            total_throughput = sum(r.throughput for r in self.results)
            avg_latency = statistics.mean(r.mean_ms for r in self.results)
            
            return {
                'suite_name': self.name,
                'benchmarks_count': len(self.results),
                'total_throughput': total_throughput,
                'avg_latency_ms': avg_latency,
                'results': [r.to_dict() for r in self.results]
            }

class MetricsCollector:
    """Collect performance metrics."""
    
    def __init__(self, window_size: int = 1000):
        self.metrics: Dict[str, List[MetricDataPoint]] = {}
        self.window_size = window_size
        self.lock = threading.RLock()
    
    def record(self, metric_type: MetricType, value: float, 
              label: str = "", metadata: Dict = None):
        """Record metric."""
        data_point = MetricDataPoint(
            timestamp=time.time(),
            value=value,
            metric_type=metric_type,
            label=label,
            metadata=metadata or {}
        )
        
        with self.lock:
            key = f"{metric_type.value}:{label}"
            if key not in self.metrics:
                self.metrics[key] = []
            
            self.metrics[key].append(data_point)
            
            # Trim to window size
            if len(self.metrics[key]) > self.window_size:
                self.metrics[key] = self.metrics[key][-self.window_size:]
    
    def get_statistics(self, metric_type: MetricType, 
                      label: str = "") -> Optional[Dict]:
        """Get statistics for metric."""
        with self.lock:
            key = f"{metric_type.value}:{label}"
            data_points = self.metrics.get(key, [])
            
            if not data_points:
                return None
            
            values = [dp.value for dp in data_points]
            
            return {
                'metric': key,
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'stddev': statistics.stdev(values) if len(values) > 1 else 0
            }

class PerformanceProfile:
    """Profile code performance."""
    
    def __init__(self, name: str):
        self.name = name
        self.sections: Dict[str, List[float]] = {}
        self.lock = threading.RLock()
    
    def start_section(self, section_name: str) -> float:
        """Start timing section."""
        return time.time()
    
    def end_section(self, section_name: str, start_time: float):
        """End timing section."""
        duration = (time.time() - start_time) * 1000  # ms
        
        with self.lock:
            if section_name not in self.sections:
                self.sections[section_name] = []
            self.sections[section_name].append(duration)
    
    def get_profile(self) -> Dict:
        """Get profiling results."""
        with self.lock:
            profile = {}
            
            for section, times in self.sections.items():
                profile[section] = {
                    'calls': len(times),
                    'total_ms': sum(times),
                    'min_ms': min(times),
                    'max_ms': max(times),
                    'mean_ms': statistics.mean(times),
                    'median_ms': statistics.median(times)
                }
            
            return profile

class OptimizationRule(ABC):
    """Base class for optimization rules."""
    
    @abstractmethod
    def evaluate(self, metrics: Dict) -> Tuple[bool, str]:
        """Evaluate if optimization needed. Returns (needs_optimization, reason)."""
        pass

class LatencyOptimizationRule(OptimizationRule):
    """Rule for latency optimization."""
    
    def __init__(self, threshold_ms: float = 100):
        self.threshold_ms = threshold_ms
    
    def evaluate(self, metrics: Dict) -> Tuple[bool, str]:
        """Check if latency exceeds threshold."""
        latency = metrics.get('latency_ms', 0)
        
        if latency > self.threshold_ms:
            return True, f"Latency {latency}ms exceeds threshold {self.threshold_ms}ms"
        
        return False, "Latency within acceptable range"

class ThroughputOptimizationRule(OptimizationRule):
    """Rule for throughput optimization."""
    
    def __init__(self, min_throughput: float = 1000):  # ops/sec
        self.min_throughput = min_throughput
    
    def evaluate(self, metrics: Dict) -> Tuple[bool, str]:
        """Check if throughput below minimum."""
        throughput = metrics.get('throughput', 0)
        
        if throughput < self.min_throughput:
            return True, f"Throughput {throughput} ops/sec below minimum {self.min_throughput}"
        
        return False, "Throughput acceptable"

class PerformanceOptimizer:
    """Optimize performance."""
    
    def __init__(self):
        self.rules: List[OptimizationRule] = []
        self.optimization_history: List[Dict] = []
        self.lock = threading.RLock()
    
    def add_rule(self, rule: OptimizationRule):
        """Add optimization rule."""
        self.rules.append(rule)
    
    def analyze(self, metrics: Dict) -> Dict:
        """Analyze metrics and suggest optimizations."""
        recommendations = []
        
        for rule in self.rules:
            needs_opt, reason = rule.evaluate(metrics)
            
            if needs_opt:
                recommendations.append({
                    'rule': rule.__class__.__name__,
                    'reason': reason,
                    'suggested_action': self._get_action(rule)
                })
        
        result = {
            'timestamp': time.time(),
            'optimization_needed': len(recommendations) > 0,
            'recommendations': recommendations
        }
        
        with self.lock:
            self.optimization_history.append(result)
        
        return result
    
    def _get_action(self, rule) -> str:
        """Get suggested action for rule."""
        if isinstance(rule, LatencyOptimizationRule):
            return "Optimize query execution, add caching, or use parallel processing"
        elif isinstance(rule, ThroughputOptimizationRule):
            return "Increase worker threads, optimize algorithms, or add load balancing"
        else:
            return "Review performance bottlenecks"

# Need to import ABC
from abc import ABC, abstractmethod

# Example usage
if __name__ == "__main__":
    # Create benchmark suite
    suite = BenchmarkSuite("Face Recognition Benchmarks")
    
    # Add benchmarks
    def face_detection():
        # Simulate face detection
        total = sum(i for i in range(100))
        return total
    
    def face_encoding():
        # Simulate face encoding
        total = sum(i * 2 for i in range(150))
        return total
    
    suite.add_benchmark("face_detection", face_detection, iterations=100)
    suite.add_benchmark("face_encoding", face_encoding, iterations=100)
    
    # Run benchmarks
    print("Running benchmark suite...\n")
    results = suite.run_all()
    
    # Print results
    for result in results:
        print(f"Benchmark: {result.name}")
        print(f"  Mean: {result.mean_ms:.3f}ms")
        print(f"  Median: {result.median_ms:.3f}ms")
        print(f"  Throughput: {result.throughput:.0f} ops/sec")
        print()
    
    # Get summary
    summary = suite.get_summary()
    print(f"Summary:")
    print(f"  Total Throughput: {summary['total_throughput']:.0f} ops/sec")
    print(f"  Avg Latency: {summary['avg_latency_ms']:.3f}ms")
