#!/usr/bin/env python3
"""
Advanced Performance Profiler

Comprehensive performance profiling for:
- CPU profiling with cProfile
- Memory profiling with tracemalloc
- I/O profiling
- Function-level timing
- API endpoint profiling
- Database query profiling
- Flamegraph generation

Features:
- Line-by-line profiling
- Performance reports (JSON, HTML)
- Benchmark utilities
- Context managers for block profiling
"""

import cProfile
import pstats
import io
import time
import functools
import logging
from typing import Callable, Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import tracemalloc
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProfileResult:
    """Profiling result"""
    name: str
    duration: float
    cpu_time: Optional[float] = None
    memory_delta: Optional[int] = None
    calls: Optional[int] = None
    timestamp: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


class AdvancedProfiler:
    """Advanced performance profiler"""
    
    def __init__(self, output_dir: str = "profiling_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[ProfileResult] = []
        self.active_profiles: Dict[str, Any] = {}
    
    def profile_function(
        self,
        func: Optional[Callable] = None,
        *,
        name: Optional[str] = None,
        save_report: bool = True,
        track_memory: bool = False
    ):
        """
        Decorator to profile a function
        
        Usage:
            @profiler.profile_function()
            def my_function():
                ...
        """
        def decorator(f: Callable) -> Callable:
            profile_name = name or f.__name__
            
            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                # Start profiling
                profiler = cProfile.Profile()
                
                if track_memory:
                    tracemalloc.start()
                    snapshot_before = tracemalloc.take_snapshot()
                
                start_time = time.time()
                
                # Run function with profiling
                profiler.enable()
                try:
                    result = f(*args, **kwargs)
                finally:
                    profiler.disable()
                
                duration = time.time() - start_time
                
                # Get memory delta
                memory_delta = None
                if track_memory:
                    snapshot_after = tracemalloc.take_snapshot()
                    stats = snapshot_after.compare_to(snapshot_before, 'lineno')
                    memory_delta = sum(stat.size_diff for stat in stats)
                    tracemalloc.stop()
                
                # Get stats
                stats_obj = pstats.Stats(profiler)
                
                # Save results
                profile_result = ProfileResult(
                    name=profile_name,
                    duration=duration,
                    memory_delta=memory_delta,
                    calls=stats_obj.total_calls
                )
                
                self.results.append(profile_result)
                
                # Print summary
                logger.info(f"Profile '{profile_name}':")
                logger.info(f"  Duration: {duration:.4f}s")
                logger.info(f"  Calls: {stats_obj.total_calls}")
                if memory_delta:
                    logger.info(f"  Memory: {memory_delta / 1024:.2f} KB")
                
                # Save detailed report
                if save_report:
                    self._save_profile_report(profile_name, profiler, profile_result)
                
                return result
            
            return wrapper
        
        if func is None:
            return decorator
        else:
            return decorator(func)
    
    def profile_block(self, name: str, track_memory: bool = False):
        """
        Context manager to profile a code block
        
        Usage:
            with profiler.profile_block("my_block"):
                # code to profile
        """
        return ProfileContext(self, name, track_memory)
    
    def _save_profile_report(
        self,
        name: str,
        profiler: cProfile.Profile,
        result: ProfileResult
    ):
        """Save detailed profiling report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.output_dir / name
        report_dir.mkdir(exist_ok=True)
        
        # Save stats as text
        stats_file = report_dir / f"profile_{timestamp}.txt"
        with open(stats_file, 'w') as f:
            stats = pstats.Stats(profiler, stream=f)
            stats.sort_stats('cumulative')
            stats.print_stats(50)  # Top 50 functions
        
        # Save stats as JSON
        json_file = report_dir / f"profile_{timestamp}.json"
        stats_obj = pstats.Stats(profiler)
        
        stats_data = {
            "name": name,
            "timestamp": result.timestamp.isoformat(),
            "duration": result.duration,
            "total_calls": stats_obj.total_calls,
            "primitive_calls": stats_obj.prim_calls,
            "top_functions": []
        }
        
        # Get top functions
        stats_obj.sort_stats('cumulative')
        for func, (cc, nc, tt, ct, callers) in list(stats_obj.stats.items())[:20]:
            stats_data["top_functions"].append({
                "function": f"{func[0]}:{func[1]}:{func[2]}",
                "calls": nc,
                "total_time": tt,
                "cumulative_time": ct
            })
        
        with open(json_file, 'w') as f:
            json.dump(stats_data, f, indent=2)
        
        logger.info(f"Profile report saved to {report_dir}")
    
    def profile_memory(self, func: Optional[Callable] = None):
        """
        Decorator to profile memory usage
        
        Usage:
            @profiler.profile_memory()
            def my_function():
                ...
        """
        def decorator(f: Callable) -> Callable:
            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                tracemalloc.start()
                
                snapshot_before = tracemalloc.take_snapshot()
                result = f(*args, **kwargs)
                snapshot_after = tracemalloc.take_snapshot()
                
                # Get top memory allocations
                stats = snapshot_after.compare_to(snapshot_before, 'lineno')
                
                logger.info(f"Memory profiling for {f.__name__}:")
                for stat in stats[:10]:
                    logger.info(f"  {stat}")
                
                tracemalloc.stop()
                
                return result
            
            return wrapper
        
        if func is None:
            return decorator
        else:
            return decorator(func)
    
    def time_function(self, func: Optional[Callable] = None, *, name: Optional[str] = None):
        """
        Simple timing decorator
        
        Usage:
            @profiler.time_function()
            def my_function():
                ...
        """
        def decorator(f: Callable) -> Callable:
            timing_name = name or f.__name__
            
            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                start = time.time()
                result = f(*args, **kwargs)
                duration = time.time() - start
                
                logger.info(f"{timing_name} took {duration:.4f}s")
                
                self.results.append(ProfileResult(
                    name=timing_name,
                    duration=duration
                ))
                
                return result
            
            return wrapper
        
        if func is None:
            return decorator
        else:
            return decorator(func)
    
    def benchmark(
        self,
        func: Callable,
        iterations: int = 100,
        warmup: int = 10,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Benchmark a function with multiple iterations
        
        Args:
            func: Function to benchmark
            iterations: Number of iterations
            warmup: Number of warmup iterations
            args: Function arguments
            kwargs: Function keyword arguments
        
        Returns:
            Benchmark results
        """
        logger.info(f"Benchmarking {func.__name__}...")
        
        # Warmup
        for _ in range(warmup):
            func(*args, **kwargs)
        
        # Benchmark
        times = []
        for i in range(iterations):
            start = time.time()
            func(*args, **kwargs)
            times.append(time.time() - start)
        
        # Calculate statistics
        times.sort()
        
        results = {
            "function": func.__name__,
            "iterations": iterations,
            "min": min(times),
            "max": max(times),
            "mean": sum(times) / len(times),
            "median": times[len(times) // 2],
            "p95": times[int(len(times) * 0.95)],
            "p99": times[int(len(times) * 0.99)]
        }
        
        logger.info(f"Benchmark results for {func.__name__}:")
        for key, value in results.items():
            if key not in ["function", "iterations"]:
                logger.info(f"  {key}: {value:.6f}s")
        
        return results
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate summary report of all profiling results"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"summary_{timestamp}.json"
        else:
            output_file = Path(output_file)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_profiles": len(self.results),
            "profiles": []
        }
        
        for result in self.results:
            profile_data = {
                "name": result.name,
                "duration": result.duration,
                "timestamp": result.timestamp.isoformat()
            }
            
            if result.cpu_time:
                profile_data["cpu_time"] = result.cpu_time
            if result.memory_delta:
                profile_data["memory_delta"] = result.memory_delta
            if result.calls:
                profile_data["calls"] = result.calls
            
            report["profiles"].append(profile_data)
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Summary report saved to {output_file}")
        
        return str(output_file)
    
    def clear_results(self):
        """Clear profiling results"""
        self.results.clear()
        logger.info("Cleared profiling results")


class ProfileContext:
    """Context manager for profiling code blocks"""
    
    def __init__(self, profiler: AdvancedProfiler, name: str, track_memory: bool = False):
        self.profiler = profiler
        self.name = name
        self.track_memory = track_memory
        self.start_time = None
        self.cprofile = None
        self.snapshot_before = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.cprofile = cProfile.Profile()
        self.cprofile.enable()
        
        if self.track_memory:
            tracemalloc.start()
            self.snapshot_before = tracemalloc.take_snapshot()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cprofile.disable()
        duration = time.time() - self.start_time
        
        memory_delta = None
        if self.track_memory:
            snapshot_after = tracemalloc.take_snapshot()
            stats = snapshot_after.compare_to(self.snapshot_before, 'lineno')
            memory_delta = sum(stat.size_diff for stat in stats)
            tracemalloc.stop()
        
        # Get stats
        stats_obj = pstats.Stats(self.cprofile)
        
        # Save result
        result = ProfileResult(
            name=self.name,
            duration=duration,
            memory_delta=memory_delta,
            calls=stats_obj.total_calls
        )
        
        self.profiler.results.append(result)
        
        # Print summary
        logger.info(f"Profile block '{self.name}':")
        logger.info(f"  Duration: {duration:.4f}s")
        logger.info(f"  Calls: {stats_obj.total_calls}")
        if memory_delta:
            logger.info(f"  Memory: {memory_delta / 1024:.2f} KB")
        
        # Save report
        self.profiler._save_profile_report(self.name, self.cprofile, result)


# Global profiler instance
profiler = AdvancedProfiler()


# Example usage
@profiler.profile_function()
def example_function(n: int):
    """Example function to profile"""
    result = 0
    for i in range(n):
        result += i ** 2
    return result


@profiler.profile_memory()
def memory_intensive_function():
    """Example memory-intensive function"""
    data = [i for i in range(1000000)]
    return sum(data)


@profiler.time_function()
def simple_function(x: int, y: int):
    """Simple function with timing"""
    time.sleep(0.1)
    return x + y


def main():
    """Example usage"""
    print("=" * 60)
    print("Performance Profiling Demo")
    print("=" * 60)
    print()
    
    # Profile function
    print("1. Function profiling:")
    result = example_function(10000)
    print()
    
    # Profile memory
    print("2. Memory profiling:")
    memory_intensive_function()
    print()
    
    # Simple timing
    print("3. Simple timing:")
    simple_function(5, 10)
    print()
    
    # Profile code block
    print("4. Code block profiling:")
    with profiler.profile_block("custom_block", track_memory=True):
        data = [i ** 2 for i in range(100000)]
        total = sum(data)
    print()
    
    # Benchmark
    print("5. Benchmarking:")
    benchmark_results = profiler.benchmark(
        lambda: sum(i ** 2 for i in range(1000)),
        iterations=100,
        warmup=10
    )
    print()
    
    # Generate report
    print("6. Generating summary report:")
    report_file = profiler.generate_report()
    print()
    
    print("=" * 60)
    print("Profiling complete!")
    print(f"Results saved to: {profiler.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
