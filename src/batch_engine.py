# Batch Processing Engine

import threading
import time
from typing import List, Callable, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
from queue import Queue

class JobStatus(Enum):
    """Job execution status."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

@dataclass
class BatchJob:
    """Batch processing job."""
    job_id: str
    batch_id: str
    items: List[Any]
    processor: Callable
    status: JobStatus = JobStatus.PENDING
    results: List[Any] = None
    error: str = None
    start_time: float = None
    end_time: float = None
    retry_count: int = 0
    max_retries: int = 3
    
    def execute(self) -> bool:
        """Execute batch job."""
        import traceback
        
        try:
            self.status = JobStatus.RUNNING
            self.start_time = time.time()
            
            results = []
            for item in self.items:
                result = self.processor(item)
                results.append(result)
            
            self.results = results
            self.status = JobStatus.COMPLETED
            self.end_time = time.time()
            return True
        
        except Exception as e:
            self.error = str(e)
            self.retry_count += 1
            
            if self.retry_count < self.max_retries:
                self.status = JobStatus.PENDING
            else:
                self.status = JobStatus.FAILED
                self.end_time = time.time()
            
            return False
    
    def duration(self) -> Optional[float]:
        """Get execution duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

class BatchProcessor:
    """Process items in batches."""
    
    def __init__(self, batch_size: int = 100, max_workers: int = 4):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.jobs: Dict[str, BatchJob] = {}
        self.job_queue: Queue = Queue()
        self.workers = []
        self.lock = threading.RLock()
        self._start_workers()
    
    def submit_batch(self, batch_id: str, items: List[Any],
                    processor: Callable) -> List[str]:
        """Submit batch for processing."""
        import uuid
        
        job_ids = []
        
        with self.lock:
            # Create jobs for batches
            for i in range(0, len(items), self.batch_size):
                batch_items = items[i:i + self.batch_size]
                job_id = str(uuid.uuid4())
                
                job = BatchJob(
                    job_id=job_id,
                    batch_id=batch_id,
                    items=batch_items,
                    processor=processor
                )
                
                self.jobs[job_id] = job
                self.job_queue.put(job_id)
                job_ids.append(job_id)
        
        return job_ids
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get job status."""
        with self.lock:
            if job_id not in self.jobs:
                return None
            
            job = self.jobs[job_id]
            return {
                'job_id': job.job_id,
                'status': job.status.value,
                'retry_count': job.retry_count,
                'error': job.error,
                'duration': job.duration(),
                'item_count': len(job.items),
                'result_count': len(job.results) if job.results else 0
            }
    
    def get_batch_results(self, batch_id: str) -> List[Any]:
        """Get all results for batch."""
        results = []
        
        with self.lock:
            for job in self.jobs.values():
                if job.batch_id == batch_id and job.results:
                    results.extend(job.results)
        
        return results
    
    def _start_workers(self) -> None:
        """Start worker threads."""
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker, daemon=True)
            worker.start()
            self.workers.append(worker)
    
    def _worker(self) -> None:
        """Worker thread."""
        while True:
            try:
                job_id = self.job_queue.get(timeout=1)
                
                with self.lock:
                    if job_id in self.jobs:
                        job = self.jobs[job_id]
                        job.execute()
                
                self.job_queue.task_done()
            
            except Exception:
                pass

class ScheduledBatchJob:
    """Schedule batch jobs."""
    
    def __init__(self):
        self.jobs: Dict[str, Dict] = {}
        self.lock = threading.RLock()
    
    def schedule(self, job_name: str, schedule: str,
                task: Callable, batch_processor: BatchProcessor) -> None:
        """Schedule job (cron-like)."""
        self.jobs[job_name] = {
            'schedule': schedule,
            'task': task,
            'processor': batch_processor,
            'last_run': None,
            'next_run': None
        }
    
    def get_schedule_status(self) -> Dict:
        """Get schedule status."""
        with self.lock:
            return {
                name: {
                    'last_run': job['last_run'],
                    'next_run': job['next_run']
                }
                for name, job in self.jobs.items()
            }

class BatchResult:
    """Aggregated batch results."""
    
    def __init__(self):
        self.results: List[Any] = []
        self.errors: List[str] = []
        self.success_count = 0
        self.failure_count = 0
        self.lock = threading.RLock()
    
    def add_result(self, result: Any) -> None:
        """Add result."""
        with self.lock:
            self.results.append(result)
            self.success_count += 1
    
    def add_error(self, error: str) -> None:
        """Add error."""
        with self.lock:
            self.errors.append(error)
            self.failure_count += 1
    
    def get_summary(self) -> Dict:
        """Get summary."""
        with self.lock:
            return {
                'total': self.success_count + self.failure_count,
                'success': self.success_count,
                'failure': self.failure_count,
                'success_rate': self.success_count / (self.success_count + self.failure_count)
                               if (self.success_count + self.failure_count) > 0 else 0
            }

# Example usage
if __name__ == "__main__":
    processor = BatchProcessor(batch_size=10, max_workers=2)
    
    # Define processor function
    def square(x):
        return x * x
    
    # Submit batch
    items = list(range(100))
    job_ids = processor.submit_batch("batch_1", items, square)
    
    print(f"Submitted {len(job_ids)} jobs")
    
    # Wait for completion
    time.sleep(2)
    
    # Get results
    results = processor.get_batch_results("batch_1")
    print(f"Results: {len(results)} items processed")
    print(f"Sample results: {results[:5]}")
