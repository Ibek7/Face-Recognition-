"""
Background task queue manager for async job processing.

Handles long-running tasks asynchronously with priority support.
"""

import asyncio
import logging
from typing import Callable, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 3
    NORMAL = 2
    HIGH = 1
    CRITICAL = 0


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Represents a background task."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    func: Callable = None
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    
    def __lt__(self, other):
        """Compare tasks by priority for queue ordering."""
        return self.priority.value < other.priority.value


class TaskQueue:
    """Manage background task execution."""
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize task queue.
        
        Args:
            max_workers: Maximum concurrent workers
        """
        self.max_workers = max_workers
        self.queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.tasks: Dict[str, Task] = {}
        self.workers: list = []
        self.running = False
    
    async def start(self):
        """Start task queue workers."""
        if self.running:
            return
        
        self.running = True
        self.workers = [
            asyncio.create_task(self._worker(i))
            for i in range(self.max_workers)
        ]
        logger.info(f"Task queue started with {self.max_workers} workers")
    
    async def stop(self):
        """Stop task queue workers."""
        self.running = False
        
        # Cancel workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        logger.info("Task queue stopped")
    
    async def submit(
        self,
        func: Callable,
        *args,
        priority: TaskPriority = TaskPriority.NORMAL,
        **kwargs
    ) -> str:
        """
        Submit task to queue.
        
        Args:
            func: Function to execute
            *args: Function arguments
            priority: Task priority
            **kwargs: Function keyword arguments
            
        Returns:
            Task ID
        """
        task = Task(
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority
        )
        
        self.tasks[task.id] = task
        await self.queue.put((priority.value, task))
        
        logger.info(f"Task submitted: {task.id} (priority: {priority.name})")
        return task.id
    
    async def _worker(self, worker_id: int):
        """
        Worker coroutine to process tasks.
        
        Args:
            worker_id: Worker identifier
        """
        logger.info(f"Worker {worker_id} started")
        
        while self.running:
            try:
                # Get task from queue
                _, task = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=1.0
                )
                
                # Execute task
                await self._execute_task(task, worker_id)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
        
        logger.info(f"Worker {worker_id} stopped")
    
    async def _execute_task(self, task: Task, worker_id: int):
        """
        Execute a task.
        
        Args:
            task: Task to execute
            worker_id: Worker executing the task
        """
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.utcnow()
        
        logger.info(f"Worker {worker_id} executing task {task.id}")
        
        try:
            # Execute function
            if asyncio.iscoroutinefunction(task.func):
                result = await task.func(*task.args, **task.kwargs)
            else:
                # Run sync function in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    task.func,
                    *task.args
                )
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            logger.info(f"Task {task.id} completed")
        
        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
            logger.error(f"Task {task.id} failed: {e}")
        
        finally:
            task.completed_at = datetime.utcnow()
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get task by ID.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task object or None
        """
        return self.tasks.get(task_id)
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """
        Get task status.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task status or None
        """
        task = self.tasks.get(task_id)
        return task.status if task else None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get queue statistics.
        
        Returns:
            Statistics dictionary
        """
        statuses = {}
        for task in self.tasks.values():
            status = task.status.value
            statuses[status] = statuses.get(status, 0) + 1
        
        return {
            "total_tasks": len(self.tasks),
            "queue_size": self.queue.qsize(),
            "workers": self.max_workers,
            "statuses": statuses,
        }


# Global task queue instance
task_queue = TaskQueue()


# Example usage:
"""
from src.task_queue import task_queue, TaskPriority

# Start queue on app startup
@app.on_event("startup")
async def startup():
    await task_queue.start()

# Stop queue on shutdown
@app.on_event("shutdown")
async def shutdown():
    await task_queue.stop()

# Submit tasks
async def process_image(image_path: str):
    # Long-running operation
    await asyncio.sleep(5)
    return {"processed": image_path}

task_id = await task_queue.submit(
    process_image,
    "/path/to/image.jpg",
    priority=TaskPriority.HIGH
)

# Check task status
status = task_queue.get_task_status(task_id)
task = task_queue.get_task(task_id)
"""
