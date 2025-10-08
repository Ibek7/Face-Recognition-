# Advanced Performance Optimization System

import asyncio
import time
import psutil
import numpy as np
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from dataclasses import dataclass
from collections import deque
import threading
import queue
import logging
from contextlib import contextmanager

# Performance monitoring and optimization
@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    cpu_usage: float
    memory_usage: float
    processing_time: float
    throughput: float
    queue_length: int
    active_workers: int
    error_rate: float
    cache_hit_rate: float

class PerformanceOptimizer:
    """Advanced performance optimization system."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or mp.cpu_count()
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
        
        # Performance tracking
        self.metrics_history = deque(maxlen=1000)
        self.processing_times = deque(maxlen=100)
        self.error_count = 0
        self.total_requests = 0
        
        # Adaptive configuration
        self.current_batch_size = 1
        self.optimal_batch_size = 1
        self.optimization_interval = 100  # requests
        
        # Cache system
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.max_cache_size = 1000
        
        # Load balancing
        self.worker_loads = {}
        self.worker_queue = queue.Queue()
        
        # Start monitoring
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_performance)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def _monitor_performance(self):
        """Continuous performance monitoring."""
        while self.monitoring_active:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Optimize based on metrics
                if len(self.metrics_history) >= 10:
                    self._adaptive_optimization()
                
                time.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                logging.error(f"Performance monitoring error: {e}")
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        avg_processing_time = (
            np.mean(list(self.processing_times)) 
            if self.processing_times else 0
        )
        
        throughput = (
            len(self.processing_times) / sum(self.processing_times)
            if self.processing_times and sum(self.processing_times) > 0 else 0
        )
        
        error_rate = (
            self.error_count / max(self.total_requests, 1)
        )
        
        cache_hit_rate = (
            self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
        )
        
        return PerformanceMetrics(
            cpu_usage=cpu_percent,
            memory_usage=memory.percent,
            processing_time=avg_processing_time,
            throughput=throughput,
            queue_length=self.worker_queue.qsize(),
            active_workers=len(self.worker_loads),
            error_rate=error_rate,
            cache_hit_rate=cache_hit_rate
        )
    
    def _adaptive_optimization(self):
        """Adaptive optimization based on performance metrics."""
        recent_metrics = list(self.metrics_history)[-10:]
        
        avg_cpu = np.mean([m.cpu_usage for m in recent_metrics])
        avg_memory = np.mean([m.memory_usage for m in recent_metrics])
        avg_processing_time = np.mean([m.processing_time for m in recent_metrics])
        
        # Optimize batch size
        if avg_cpu < 70 and avg_memory < 80:
            # System has capacity, increase batch size
            self.optimal_batch_size = min(self.optimal_batch_size + 1, 32)
        elif avg_cpu > 90 or avg_memory > 90:
            # System overloaded, decrease batch size
            self.optimal_batch_size = max(self.optimal_batch_size - 1, 1)
        
        # Optimize worker count
        if avg_processing_time > 2.0 and len(self.worker_loads) < self.max_workers:
            self._add_worker()
        elif avg_processing_time < 0.5 and len(self.worker_loads) > 1:
            self._remove_worker()
        
        # Cache optimization
        if self.cache_hit_rate < 0.3:
            self._optimize_cache()
    
    def _add_worker(self):
        """Add a new worker to the pool."""
        worker_id = f"worker_{len(self.worker_loads)}"
        self.worker_loads[worker_id] = 0
        logging.info(f"Added worker: {worker_id}")
    
    def _remove_worker(self):
        """Remove a worker from the pool."""
        if self.worker_loads:
            worker_id = min(self.worker_loads.keys(), 
                          key=lambda k: self.worker_loads[k])
            del self.worker_loads[worker_id]
            logging.info(f"Removed worker: {worker_id}")
    
    def _optimize_cache(self):
        """Optimize cache performance."""
        if len(self.cache) > self.max_cache_size * 0.8:
            # Remove least recently used items
            items_to_remove = len(self.cache) - int(self.max_cache_size * 0.7)
            for _ in range(items_to_remove):
                if self.cache:
                    self.cache.pop(next(iter(self.cache)))
    
    @contextmanager
    def performance_tracking(self, operation_name: str):
        """Context manager for tracking operation performance."""
        start_time = time.time()
        self.total_requests += 1
        
        try:
            yield
        except Exception as e:
            self.error_count += 1
            raise
        finally:
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            logging.debug(f"{operation_name} completed in {processing_time:.3f}s")
    
    def get_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        import hashlib
        key_str = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def cached_operation(self, cache_key: str, operation_func, *args, **kwargs):
        """Execute operation with caching."""
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        
        self.cache_misses += 1
        result = operation_func(*args, **kwargs)
        
        if len(self.cache) < self.max_cache_size:
            self.cache[cache_key] = result
        
        return result
    
    async def async_batch_process(self, items: List, process_func, batch_size: int = None):
        """Process items in optimized batches."""
        if batch_size is None:
            batch_size = self.optimal_batch_size
        
        results = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            with self.performance_tracking(f"batch_process_{len(batch)}"):
                # Process batch asynchronously
                loop = asyncio.get_event_loop()
                batch_results = await loop.run_in_executor(
                    self.thread_pool, 
                    self._process_batch,
                    batch, 
                    process_func
                )
                results.extend(batch_results)
        
        return results
    
    def _process_batch(self, batch: List, process_func) -> List:
        """Process a batch of items."""
        return [process_func(item) for item in batch]
    
    def get_performance_report(self) -> Dict:
        """Generate comprehensive performance report."""
        if not self.metrics_history:
            return {"status": "No metrics available"}
        
        recent_metrics = list(self.metrics_history)[-50:]  # Last 50 measurements
        
        report = {
            "system_performance": {
                "avg_cpu_usage": np.mean([m.cpu_usage for m in recent_metrics]),
                "avg_memory_usage": np.mean([m.memory_usage for m in recent_metrics]),
                "avg_processing_time": np.mean([m.processing_time for m in recent_metrics]),
                "current_throughput": recent_metrics[-1].throughput if recent_metrics else 0
            },
            "optimization_status": {
                "current_batch_size": self.current_batch_size,
                "optimal_batch_size": self.optimal_batch_size,
                "active_workers": len(self.worker_loads),
                "queue_length": self.worker_queue.qsize()
            },
            "cache_performance": {
                "hit_rate": self.cache_hits / max(self.cache_hits + self.cache_misses, 1),
                "cache_size": len(self.cache),
                "max_cache_size": self.max_cache_size
            },
            "error_metrics": {
                "error_rate": self.error_count / max(self.total_requests, 1),
                "total_errors": self.error_count,
                "total_requests": self.total_requests
            },
            "recommendations": self._generate_recommendations(recent_metrics)
        }
        
        return report
    
    def _generate_recommendations(self, metrics: List[PerformanceMetrics]) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        avg_cpu = np.mean([m.cpu_usage for m in metrics])
        avg_memory = np.mean([m.memory_usage for m in metrics])
        avg_processing_time = np.mean([m.processing_time for m in metrics])
        avg_error_rate = np.mean([m.error_rate for m in metrics])
        
        if avg_cpu > 90:
            recommendations.append("High CPU usage detected. Consider reducing batch size or adding more workers.")
        
        if avg_memory > 90:
            recommendations.append("High memory usage detected. Consider optimizing data structures or implementing memory pooling.")
        
        if avg_processing_time > 5.0:
            recommendations.append("High processing times detected. Consider algorithm optimization or hardware upgrade.")
        
        if avg_error_rate > 0.05:
            recommendations.append("High error rate detected. Review error logs and implement additional error handling.")
        
        cache_hit_rate = self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
        if cache_hit_rate < 0.3:
            recommendations.append("Low cache hit rate. Consider optimizing cache strategy or increasing cache size.")
        
        if not recommendations:
            recommendations.append("System performance is optimal.")
        
        return recommendations
    
    def shutdown(self):
        """Shutdown performance optimizer."""
        self.monitoring_active = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)


class GPUOptimizer:
    """GPU-specific performance optimization."""
    
    def __init__(self):
        self.gpu_available = self._check_gpu_availability()
        self.device = "cuda" if self.gpu_available else "cpu"
        self.batch_sizes = {"small": 1, "medium": 4, "large": 8, "xlarge": 16}
        self.optimal_batch_size = "small"
        
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def optimize_model_for_inference(self, model):
        """Optimize model for inference performance."""
        if not self.gpu_available:
            return model
        
        try:
            import torch
            
            # Move to GPU
            model = model.to(self.device)
            
            # Set to evaluation mode
            model.eval()
            
            # Enable mixed precision if available
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                model = model.half()  # Use FP16
            
            # Compile model if available (PyTorch 2.0+)
            if hasattr(torch, 'compile'):
                model = torch.compile(model)
            
            return model
            
        except Exception as e:
            logging.warning(f"GPU optimization failed: {e}")
            return model
    
    def benchmark_batch_sizes(self, model, sample_input):
        """Benchmark different batch sizes to find optimal."""
        if not self.gpu_available:
            return self.optimal_batch_size
        
        results = {}
        
        for size_name, batch_size in self.batch_sizes.items():
            try:
                # Create batch
                batch_input = sample_input.repeat(batch_size, 1, 1, 1)
                
                # Benchmark
                start_time = time.time()
                with torch.no_grad():
                    _ = model(batch_input)
                
                processing_time = time.time() - start_time
                throughput = batch_size / processing_time
                
                results[size_name] = {
                    "batch_size": batch_size,
                    "processing_time": processing_time,
                    "throughput": throughput
                }
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    break
                else:
                    logging.warning(f"Benchmark error for batch size {batch_size}: {e}")
        
        # Find optimal batch size
        if results:
            optimal = max(results.keys(), key=lambda k: results[k]["throughput"])
            self.optimal_batch_size = optimal
            logging.info(f"Optimal batch size: {optimal} ({self.batch_sizes[optimal]})")
        
        return self.optimal_batch_size


class MemoryOptimizer:
    """Memory optimization utilities."""
    
    def __init__(self):
        self.memory_threshold = 0.85  # 85% memory usage threshold
        self.cleanup_interval = 100   # Cleanup every 100 operations
        self.operation_count = 0
        
    def memory_efficient_operation(self, operation_func):
        """Decorator for memory-efficient operations."""
        def wrapper(*args, **kwargs):
            self.operation_count += 1
            
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > self.memory_threshold * 100:
                self._emergency_cleanup()
            
            # Execute operation
            try:
                result = operation_func(*args, **kwargs)
                
                # Periodic cleanup
                if self.operation_count % self.cleanup_interval == 0:
                    self._periodic_cleanup()
                
                return result
                
            except MemoryError:
                self._emergency_cleanup()
                # Retry once after cleanup
                return operation_func(*args, **kwargs)
        
        return wrapper
    
    def _emergency_cleanup(self):
        """Emergency memory cleanup."""
        import gc
        
        # Force garbage collection
        collected = gc.collect()
        
        # Clear caches if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        logging.warning(f"Emergency memory cleanup: collected {collected} objects")
    
    def _periodic_cleanup(self):
        """Periodic memory maintenance."""
        import gc
        gc.collect()
    
    def get_memory_usage(self) -> Dict:
        """Get detailed memory usage information."""
        memory = psutil.virtual_memory()
        
        usage_info = {
            "total": memory.total,
            "available": memory.available,
            "percent": memory.percent,
            "used": memory.used,
            "free": memory.free,
            "status": "normal"
        }
        
        if memory.percent > 90:
            usage_info["status"] = "critical"
        elif memory.percent > 80:
            usage_info["status"] = "warning"
        
        return usage_info


# Integration with face recognition system
class OptimizedFaceRecognitionSystem:
    """Face recognition system with advanced performance optimization."""
    
    def __init__(self):
        self.performance_optimizer = PerformanceOptimizer()
        self.gpu_optimizer = GPUOptimizer()
        self.memory_optimizer = MemoryOptimizer()
        
        self.face_detector = None
        self.face_encoder = None
        self.similarity_threshold = 0.7
        
    def initialize_models(self):
        """Initialize and optimize models."""
        # Load models (implementation depends on your model architecture)
        # self.face_detector = load_face_detector()
        # self.face_encoder = load_face_encoder()
        
        # Optimize models for inference
        if self.face_encoder:
            self.face_encoder = self.gpu_optimizer.optimize_model_for_inference(self.face_encoder)
    
    @memory_optimizer.memory_efficient_operation
    async def recognize_faces_batch(self, images: List[np.ndarray]) -> List[Dict]:
        """Optimized batch face recognition."""
        with self.performance_optimizer.performance_tracking("batch_face_recognition"):
            
            # Process in optimized batches
            results = await self.performance_optimizer.async_batch_process(
                images, 
                self._recognize_single_face,
                batch_size=self.gpu_optimizer.batch_sizes[self.gpu_optimizer.optimal_batch_size]
            )
            
            return results
    
    def _recognize_single_face(self, image: np.ndarray) -> Dict:
        """Recognize faces in a single image."""
        # Generate cache key
        cache_key = self.performance_optimizer.get_cache_key(
            image.shape, 
            image.dtype,
            hash(image.tobytes())
        )
        
        # Try cached result first
        return self.performance_optimizer.cached_operation(
            cache_key,
            self._perform_recognition,
            image
        )
    
    def _perform_recognition(self, image: np.ndarray) -> Dict:
        """Perform actual face recognition."""
        try:
            # Detect faces
            faces = self._detect_faces(image)
            
            if not faces:
                return {"faces": [], "processing_time": 0}
            
            # Extract embeddings
            embeddings = self._extract_embeddings(faces)
            
            # Match against database
            matches = self._match_embeddings(embeddings)
            
            return {
                "faces": matches,
                "face_count": len(faces),
                "processing_time": time.time()
            }
            
        except Exception as e:
            logging.error(f"Face recognition error: {e}")
            return {"error": str(e), "faces": []}
    
    def _detect_faces(self, image: np.ndarray) -> List[np.ndarray]:
        """Detect faces in image."""
        # Implementation would use your face detection model
        # This is a placeholder
        return []
    
    def _extract_embeddings(self, faces: List[np.ndarray]) -> List[np.ndarray]:
        """Extract face embeddings."""
        # Implementation would use your face encoding model
        # This is a placeholder
        return []
    
    def _match_embeddings(self, embeddings: List[np.ndarray]) -> List[Dict]:
        """Match embeddings against database."""
        # Implementation would query your database
        # This is a placeholder
        return []
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status."""
        performance_report = self.performance_optimizer.get_performance_report()
        memory_usage = self.memory_optimizer.get_memory_usage()
        
        return {
            "performance": performance_report,
            "memory": memory_usage,
            "gpu_status": {
                "available": self.gpu_optimizer.gpu_available,
                "device": self.gpu_optimizer.device,
                "optimal_batch_size": self.gpu_optimizer.optimal_batch_size
            },
            "system_health": self._assess_system_health(performance_report, memory_usage)
        }
    
    def _assess_system_health(self, performance: Dict, memory: Dict) -> str:
        """Assess overall system health."""
        issues = []
        
        if performance.get("system_performance", {}).get("avg_cpu_usage", 0) > 90:
            issues.append("high_cpu")
        
        if memory.get("percent", 0) > 90:
            issues.append("high_memory")
        
        if performance.get("error_metrics", {}).get("error_rate", 0) > 0.05:
            issues.append("high_errors")
        
        if not issues:
            return "healthy"
        elif len(issues) == 1:
            return "warning"
        else:
            return "critical"
    
    def shutdown(self):
        """Shutdown optimization system."""
        self.performance_optimizer.shutdown()


# Example usage and testing
if __name__ == "__main__":
    async def test_optimization_system():
        """Test the optimization system."""
        
        # Initialize optimized face recognition system
        system = OptimizedFaceRecognitionSystem()
        system.initialize_models()
        
        # Generate test data
        test_images = [np.random.rand(224, 224, 3) for _ in range(10)]
        
        # Test batch processing
        print("Testing batch face recognition...")
        start_time = time.time()
        
        results = await system.recognize_faces_batch(test_images)
        
        processing_time = time.time() - start_time
        print(f"Processed {len(test_images)} images in {processing_time:.2f} seconds")
        
        # Get system status
        status = system.get_system_status()
        print("\nSystem Status:")
        print(f"Health: {status['system_health']}")
        print(f"Memory Usage: {status['memory']['percent']:.1f}%")
        print(f"GPU Available: {status['gpu_status']['available']}")
        
        # Performance report
        perf_report = status['performance']
        print(f"Average CPU: {perf_report['system_performance']['avg_cpu_usage']:.1f}%")
        print(f"Cache Hit Rate: {perf_report['cache_performance']['hit_rate']:.2f}")
        print(f"Error Rate: {perf_report['error_metrics']['error_rate']:.3f}")
        
        # Recommendations
        print("\nRecommendations:")
        for rec in perf_report['recommendations']:
            print(f"- {rec}")
        
        # Cleanup
        system.shutdown()
    
    # Run test
    asyncio.run(test_optimization_system())