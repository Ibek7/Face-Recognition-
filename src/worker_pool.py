"""
Async Worker Pool

Manage concurrent task execution with worker pools:
- Async task queue processing
- Worker lifecycle management
- Priority queues
- Task retry and timeout handling
- Resource pooling and load balancing
"""

import asyncio
import logging
from typing import Callable, Any, Optional, Dict, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import time
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskPriority(int, Enum):
    """Task priority levels"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class TaskStatus(str, Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class Task:
    """Task definition"""
    id: str
    func: Callable
    args: tuple = ()
    kwargs: dict = None
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: Optional[float] = None
    max_retries: int = 0
    created_at: datetime = None
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def __lt__(self, other):
        """For priority queue comparison"""
        return self.priority.value > other.priority.value  # Higher priority first


@dataclass
class TaskResult:
    """Task execution result"""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[Exception] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    
    @property
    def execution_time(self) -> Optional[float]:
        """Calculate execution time in seconds"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class Worker:
    """Individual worker for processing tasks"""
    
    def __init__(self, worker_id: int, pool: 'WorkerPool'):
        self.worker_id = worker_id
        self.pool = pool
        self.current_task: Optional[Task] = None
        self.tasks_processed = 0
        self.is_running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start worker processing loop"""
        self.is_running = True
        self._task = asyncio.create_task(self._process_loop())
        logger.info(f"Worker {self.worker_id} started")
    
    async def stop(self):
        """Stop worker gracefully"""
        self.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info(f"Worker {self.worker_id} stopped")
    
    async def _process_loop(self):
        """Main worker processing loop"""
        while self.is_running:
            try:
                # Get task from pool queue
                task = await self.pool.get_task()
                
                if task is None:
                    await asyncio.sleep(0.1)
                    continue
                
                self.current_task = task
                
                # Execute task
                result = await self._execute_task(task)
                
                # Store result
                self.pool.results[task.id] = result
                
                self.tasks_processed += 1
                self.current_task = None
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {e}")
    
    async def _execute_task(self, task: Task) -> TaskResult:
        """Execute a single task with retry and timeout"""
        result = TaskResult(
            task_id=task.id,
            status=TaskStatus.RUNNING,
            started_at=datetime.now()
        )
        
        retry_count = 0
        
        while retry_count <= task.max_retries:
            try:
                # Execute with timeout if specified
                if task.timeout:
                    task_result = await asyncio.wait_for(
                        self._run_task_func(task),
                        timeout=task.timeout
                    )
                else:
                    task_result = await self._run_task_func(task)
                
                result.status = TaskStatus.COMPLETED
                result.result = task_result
                result.retry_count = retry_count
                break
                
            except asyncio.TimeoutError:
                result.status = TaskStatus.TIMEOUT
                result.error = TimeoutError(f"Task {task.id} timed out after {task.timeout}s")
                logger.warning(f"Task {task.id} timeout (attempt {retry_count + 1}/{task.max_retries + 1})")
                retry_count += 1
                
            except Exception as e:
                result.status = TaskStatus.FAILED
                result.error = e
                logger.warning(f"Task {task.id} failed: {e} (attempt {retry_count + 1}/{task.max_retries + 1})")
                retry_count += 1
                
                if retry_count <= task.max_retries:
                    await asyncio.sleep(min(2 ** retry_count, 10))  # Exponential backoff
        
        result.completed_at = datetime.now()
        return result
    
    async def _run_task_func(self, task: Task) -> Any:
        """Run task function (async or sync)"""
        if asyncio.iscoroutinefunction(task.func):
            return await task.func(*task.args, **task.kwargs)
        else:
            # Run sync function in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.pool.thread_pool,
                lambda: task.func(*task.args, **task.kwargs)
            )


class WorkerPool:
    """Async worker pool for concurrent task processing"""
    
    def __init__(
        self,
        num_workers: int = 4,
        max_queue_size: int = 1000,
        thread_pool_size: int = 10
    ):
        self.num_workers = num_workers
        self.max_queue_size = max_queue_size
        
        # Task queue (priority queue)
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_queue_size)
        
        # Workers
        self.workers: List[Worker] = []
        
        # Results storage
        self.results: Dict[str, TaskResult] = {}
        
        # Thread pool for sync functions
        self.thread_pool = ThreadPoolExecutor(max_workers=thread_pool_size)
        
        # Statistics
        self.stats = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'tasks_timeout': 0,
            'tasks_cancelled': 0
        }
        
        self.is_running = False
    
    async def start(self):
        """Start all workers"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Create and start workers
        for i in range(self.num_workers):
            worker = Worker(i, self)
            self.workers.append(worker)
            await worker.start()
        
        logger.info(f"Worker pool started with {self.num_workers} workers")
    
    async def stop(self, timeout: Optional[float] = 10.0):
        """Stop all workers gracefully"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop all workers
        stop_tasks = [worker.stop() for worker in self.workers]
        
        if timeout:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*stop_tasks),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning("Worker shutdown timed out")
        else:
            await asyncio.gather(*stop_tasks)
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        logger.info("Worker pool stopped")
    
    async def submit_task(
        self,
        task_id: str,
        func: Callable,
        *args,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: Optional[float] = None,
        max_retries: int = 0,
        **kwargs
    ) -> str:
        """
        Submit a task to the worker pool
        
        Args:
            task_id: Unique task identifier
            func: Function to execute (async or sync)
            args: Positional arguments for function
            priority: Task priority
            timeout: Task timeout in seconds
            max_retries: Maximum retry attempts
            kwargs: Keyword arguments for function
        
        Returns:
            Task ID
        """
        task = Task(
            id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout,
            max_retries=max_retries
        )
        
        await self.task_queue.put(task)
        self.stats['tasks_submitted'] += 1
        
        logger.debug(f"Task {task_id} submitted with priority {priority.name}")
        
        return task_id
    
    async def get_task(self) -> Optional[Task]:
        """Get next task from queue"""
        try:
            task = await asyncio.wait_for(self.task_queue.get(), timeout=0.1)
            return task
        except asyncio.TimeoutError:
            return None
    
    async def get_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[TaskResult]:
        """
        Wait for and get task result
        
        Args:
            task_id: Task identifier
            timeout: Maximum wait time in seconds
        
        Returns:
            TaskResult or None if timeout
        """
        start_time = time.time()
        
        while True:
            if task_id in self.results:
                result = self.results[task_id]
                
                # Update statistics
                if result.status == TaskStatus.COMPLETED:
                    self.stats['tasks_completed'] += 1
                elif result.status == TaskStatus.FAILED:
                    self.stats['tasks_failed'] += 1
                elif result.status == TaskStatus.TIMEOUT:
                    self.stats['tasks_timeout'] += 1
                elif result.status == TaskStatus.CANCELLED:
                    self.stats['tasks_cancelled'] += 1
                
                return result
            
            if timeout and (time.time() - start_time) > timeout:
                return None
            
            await asyncio.sleep(0.1)
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task"""
        # Note: Can only cancel pending tasks, not running ones
        # This is a simplified implementation
        if task_id in self.results:
            return False
        
        self.results[task_id] = TaskResult(
            task_id=task_id,
            status=TaskStatus.CANCELLED
        )
        
        self.stats['tasks_cancelled'] += 1
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        active_workers = sum(1 for w in self.workers if w.current_task is not None)
        
        return {
            **self.stats,
            'num_workers': self.num_workers,
            'active_workers': active_workers,
            'queue_size': self.task_queue.qsize(),
            'total_processed': sum(w.tasks_processed for w in self.workers)
        }
    
    async def wait_all(self, timeout: Optional[float] = None):
        """Wait for all queued tasks to complete"""
        start_time = time.time()
        
        while self.task_queue.qsize() > 0 or any(w.current_task for w in self.workers):
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError("Wait all tasks timeout")
            
            await asyncio.sleep(0.1)


# Example usage
async def example_async_task(name: str, duration: float):
    """Example async task"""
    await asyncio.sleep(duration)
    return f"Task {name} completed after {duration}s"


def example_sync_task(name: str, value: int):
    """Example sync task"""
    time.sleep(0.5)
    return f"Task {name} result: {value * 2}"


async def main():
    """Example usage of worker pool"""
    # Create worker pool
    pool = WorkerPool(num_workers=4)
    await pool.start()
    
    print("Submitting tasks...\n")
    
    # Submit async tasks
    for i in range(5):
        await pool.submit_task(
            task_id=f"async_task_{i}",
            func=example_async_task,
            name=f"async_{i}",
            duration=1.0,
            priority=TaskPriority.NORMAL
        )
    
    # Submit sync tasks
    for i in range(5):
        await pool.submit_task(
            task_id=f"sync_task_{i}",
            func=example_sync_task,
            name=f"sync_{i}",
            value=i * 10,
            priority=TaskPriority.HIGH
        )
    
    # Submit task with timeout and retry
    await pool.submit_task(
        task_id="timeout_task",
        func=example_async_task,
        name="timeout_test",
        duration=5.0,
        timeout=2.0,
        max_retries=2,
        priority=TaskPriority.CRITICAL
    )
    
    # Wait for all tasks
    print("Waiting for tasks to complete...\n")
    await pool.wait_all(timeout=10.0)
    
    # Get results
    print("Results:")
    print("=" * 60)
    
    for task_id in [f"async_task_{i}" for i in range(5)] + [f"sync_task_{i}" for i in range(5)] + ["timeout_task"]:
        result = await pool.get_result(task_id)
        if result:
            print(f"\n{task_id}:")
            print(f"  Status: {result.status.value}")
            print(f"  Result: {result.result}")
            print(f"  Execution time: {result.execution_time:.2f}s" if result.execution_time else "  Execution time: N/A")
            if result.error:
                print(f"  Error: {result.error}")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Pool Statistics:")
    print("=" * 60)
    
    stats = pool.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Stop pool
    await pool.stop()


if __name__ == "__main__":
    asyncio.run(main())
