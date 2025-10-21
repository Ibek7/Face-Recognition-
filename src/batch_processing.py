# Advanced Batch Processing Optimization

import logging
import time
import threading
from typing import List, Callable, Any, Optional, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from queue import PriorityQueue, Queue
import numpy as np
from datetime import datetime

class BatchPriority(Enum):
    """Batch processing priority levels."""
    LOW = 3
    NORMAL = 2
    HIGH = 1
    CRITICAL = 0

@dataclass
class BatchItem:
    """Individual item in a batch."""
    item_id: str
    data: Any
    priority: BatchPriority = BatchPriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    
    def __lt__(self, other):
        """For priority queue ordering."""
        if self.priority.value == other.priority.value:
            return self.timestamp < other.timestamp
        return self.priority.value < other.priority.value

@dataclass
class BatchMetrics:
    """Metrics for batch processing."""
    batch_id: str
    size: int
    start_time: float
    end_time: Optional[float] = None
    processing_time: float = 0.0
    throughput: float = 0.0
    wait_time: float = 0.0
    items_processed: int = 0
    success_rate: float = 0.0
    errors: int = 0

class AdaptiveBatcher:
    """Adaptive batch size optimization."""
    
    def __init__(self, min_batch_size: int = 1, max_batch_size: int = 128,
                 target_latency_ms: float = 100):
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_latency_ms = target_latency_ms
        self.current_batch_size = min(32, max_batch_size)
        
        self.recent_latencies: deque = deque(maxlen=100)
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def record_latency(self, latency_ms: float, batch_size: int):
        """Record processing latency for a batch."""
        with self.lock:
            self.recent_latencies.append((latency_ms, batch_size))
            self._adapt_batch_size()
    
    def _adapt_batch_size(self):
        """Adapt batch size based on latency."""
        if len(self.recent_latencies) < 10:
            return
        
        avg_latency = np.mean([l[0] for l in list(self.recent_latencies)[-10:]])
        avg_batch_size = np.mean([b[1] for b in list(self.recent_latencies)[-10:]])
        
        if avg_latency > self.target_latency_ms * 1.1:
            # Reduce batch size if latency too high
            new_size = max(self.min_batch_size, int(self.current_batch_size * 0.9))
        elif avg_latency < self.target_latency_ms * 0.8:
            # Increase batch size if latency acceptable
            new_size = min(self.max_batch_size, int(self.current_batch_size * 1.1))
        else:
            return
        
        if new_size != self.current_batch_size:
            self.logger.info(f"Adapting batch size: {self.current_batch_size} -> {new_size} "
                           f"(latency: {avg_latency:.1f}ms)")
            self.current_batch_size = new_size
    
    def get_recommended_batch_size(self) -> int:
        """Get recommended batch size."""
        with self.lock:
            return self.current_batch_size

class DynamicBatchProcessor:
    """Process items in dynamic batches."""
    
    def __init__(self, batch_process_func: Callable, 
                 max_batch_size: int = 128,
                 max_wait_ms: float = 100,
                 enable_adaptive: bool = True):
        self.batch_process_func = batch_process_func
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.enable_adaptive = enable_adaptive
        
        self.priority_queue: PriorityQueue = PriorityQueue()
        self.result_queue: Dict[str, Any] = {}
        self.metrics: List[BatchMetrics] = []
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        self.adaptive_batcher = AdaptiveBatcher(
            min_batch_size=1,
            max_batch_size=max_batch_size
        )
        
        self.is_running = False
        self.processor_thread = None
    
    def start(self):
        """Start batch processor."""
        if self.is_running:
            return
        
        self.is_running = True
        self.processor_thread = threading.Thread(
            target=self._process_batches,
            daemon=True
        )
        self.processor_thread.start()
        self.logger.info("Batch processor started")
    
    def stop(self):
        """Stop batch processor."""
        self.is_running = False
        if self.processor_thread:
            self.processor_thread.join(timeout=5)
        self.logger.info("Batch processor stopped")
    
    def add_item(self, item: BatchItem) -> str:
        """Add item to batch queue."""
        self.priority_queue.put(item)
        return item.item_id
    
    def _process_batches(self):
        """Process batches in background thread."""
        batch_id = 0
        
        while self.is_running:
            batch_start = time.time()
            batch_size = (self.adaptive_batcher.get_recommended_batch_size()
                         if self.enable_adaptive else self.max_batch_size)
            
            batch_items = self._collect_batch(batch_size)
            
            if not batch_items:
                time.sleep(0.01)
                continue
            
            # Create batch metrics
            metrics = BatchMetrics(
                batch_id=f"batch_{batch_id}",
                size=len(batch_items),
                start_time=batch_start,
                wait_time=time.time() - min(item.timestamp for item in batch_items)
            )
            
            try:
                # Process batch
                process_start = time.time()
                results = self._execute_batch(batch_items)
                process_time = (time.time() - process_start) * 1000  # ms
                
                # Store results
                with self.lock:
                    for item, result in zip(batch_items, results):
                        self.result_queue[item.item_id] = result
                
                # Update metrics
                metrics.end_time = time.time()
                metrics.processing_time = process_time
                metrics.throughput = len(batch_items) / (metrics.processing_time / 1000) if metrics.processing_time > 0 else 0
                metrics.items_processed = len(batch_items)
                metrics.success_rate = 1.0
                
                # Record latency for adaptation
                if self.enable_adaptive:
                    self.adaptive_batcher.record_latency(process_time, len(batch_items))
                
                self.logger.debug(f"Processed batch {batch_id}: {len(batch_items)} items in {process_time:.1f}ms")
            
            except Exception as e:
                metrics.end_time = time.time()
                metrics.processing_time = (time.time() - batch_start) * 1000
                metrics.errors = len(batch_items)
                metrics.success_rate = 0.0
                self.logger.error(f"Batch processing failed: {e}")
            
            with self.lock:
                self.metrics.append(metrics)
            
            batch_id += 1
    
    def _collect_batch(self, batch_size: int) -> List[BatchItem]:
        """Collect items for next batch."""
        batch = []
        deadline = time.time() + (self.max_wait_ms / 1000)
        
        while len(batch) < batch_size:
            timeout = max(0.001, deadline - time.time())
            try:
                item = self.priority_queue.get(timeout=timeout)
                batch.append(item)
            except:
                break
        
        return batch
    
    def _execute_batch(self, batch_items: List[BatchItem]) -> List[Any]:
        """Execute batch processing."""
        batch_data = [item.data for item in batch_items]
        return self.batch_process_func(batch_data)
    
    def get_result(self, item_id: str, timeout: float = 5.0) -> Any:
        """Get result for an item."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self.lock:
                if item_id in self.result_queue:
                    return self.result_queue.pop(item_id)
            
            time.sleep(0.01)
        
        raise TimeoutError(f"Result not available for item {item_id}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics."""
        with self.lock:
            if not self.metrics:
                return {}
            
            recent_metrics = self.metrics[-100:]
            
            return {
                'total_batches': len(self.metrics),
                'avg_batch_size': np.mean([m.size for m in recent_metrics]),
                'avg_latency_ms': np.mean([m.processing_time for m in recent_metrics]),
                'total_throughput': sum(m.throughput for m in recent_metrics) / len(recent_metrics),
                'avg_success_rate': np.mean([m.success_rate for m in recent_metrics]),
                'current_batch_size': self.adaptive_batcher.get_recommended_batch_size()
            }

class PriorityBatchQueue:
    """Priority-aware batch processing queue."""
    
    def __init__(self, process_func: Callable, max_batch_size: int = 128):
        self.process_func = process_func
        self.max_batch_size = max_batch_size
        
        self.queues = {priority: deque() for priority in BatchPriority}
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def enqueue(self, item: BatchItem):
        """Enqueue item with priority."""
        with self.lock:
            self.queues[item.priority].append(item)
    
    def get_next_batch(self) -> List[BatchItem]:
        """Get next batch respecting priorities."""
        with self.lock:
            batch = []
            
            # Fill batch from highest to lowest priority
            for priority in sorted([p for p in BatchPriority], 
                                  key=lambda x: x.value):
                while len(batch) < self.max_batch_size and self.queues[priority]:
                    batch.append(self.queues[priority].popleft())
            
            return batch
    
    def process_batches(self) -> List[Dict[str, Any]]:
        """Process all pending batches."""
        results = []
        
        while True:
            batch = self.get_next_batch()
            if not batch:
                break
            
            try:
                batch_data = [item.data for item in batch]
                batch_results = self.process_func(batch_data)
                
                results.append({
                    'batch_size': len(batch),
                    'status': 'success',
                    'results': batch_results
                })
            except Exception as e:
                self.logger.error(f"Batch processing error: {e}")
                results.append({
                    'batch_size': len(batch),
                    'status': 'failed',
                    'error': str(e)
                })
        
        return results

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example batch processing function
    def batch_face_detection(images):
        """Simulated batch face detection."""
        time.sleep(0.1)  # Simulate processing
        return [{"faces": len(img) % 3} for img in images]
    
    # Create dynamic batch processor
    processor = DynamicBatchProcessor(
        batch_process_func=batch_face_detection,
        max_batch_size=32,
        max_wait_ms=100,
        enable_adaptive=True
    )
    
    processor.start()
    
    # Submit items
    item_ids = []
    for i in range(100):
        item = BatchItem(
            item_id=f"img_{i}",
            data=f"image_{i}",
            priority=BatchPriority.NORMAL
        )
        item_ids.append(processor.add_item(item))
    
    # Get results
    time.sleep(2)
    for item_id in item_ids[:5]:
        try:
            result = processor.get_result(item_id)
            print(f"Result for {item_id}: {result}")
        except TimeoutError:
            print(f"Timeout for {item_id}")
    
    metrics = processor.get_metrics()
    print(f"Metrics: {metrics}")
    
    processor.stop()