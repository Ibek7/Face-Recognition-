"""
Background job scheduler with cron-like scheduling.

Provides scheduled task execution with cron expressions and intervals.
"""

import asyncio
from typing import Callable, Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import croniter
import logging

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Job execution status."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobExecution:
    """Record of job execution."""
    
    def __init__(self, job_name: str):
        """Initialize job execution."""
        self.job_name = job_name
        self.started_at = datetime.utcnow()
        self.completed_at: Optional[datetime] = None
        self.status = JobStatus.RUNNING
        self.error: Optional[str] = None
        self.result: Any = None
    
    def complete(self, result: Any = None):
        """Mark as completed."""
        self.completed_at = datetime.utcnow()
        self.status = JobStatus.COMPLETED
        self.result = result
    
    def fail(self, error: str):
        """Mark as failed."""
        self.completed_at = datetime.utcnow()
        self.status = JobStatus.FAILED
        self.error = error
    
    def duration(self) -> float:
        """Get execution duration in seconds."""
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return (datetime.utcnow() - self.started_at).total_seconds()


class ScheduledJob:
    """Scheduled job definition."""
    
    def __init__(
        self,
        name: str,
        func: Callable,
        schedule: Optional[str] = None,
        interval: Optional[int] = None,
        args: tuple = (),
        kwargs: dict = None,
        enabled: bool = True
    ):
        """
        Initialize scheduled job.
        
        Args:
            name: Job name
            func: Function to execute
            schedule: Cron expression
            interval: Interval in seconds
            args: Function arguments
            kwargs: Function keyword arguments
            enabled: Whether job is enabled
        """
        self.name = name
        self.func = func
        self.schedule = schedule
        self.interval = interval
        self.args = args
        self.kwargs = kwargs or {}
        self.enabled = enabled
        self.last_run: Optional[datetime] = None
        self.next_run: Optional[datetime] = None
        self.executions: List[JobExecution] = []
        self.max_history = 100
        
        # Initialize next run time
        if schedule:
            self._init_cron()
        elif interval:
            self.next_run = datetime.utcnow()
    
    def _init_cron(self):
        """Initialize cron schedule."""
        try:
            cron = croniter.croniter(self.schedule, datetime.utcnow())
            self.next_run = cron.get_next(datetime)
        except Exception as e:
            logger.error(f"Invalid cron expression for {self.name}: {e}")
            self.enabled = False
    
    def should_run(self) -> bool:
        """Check if job should run now."""
        if not self.enabled or not self.next_run:
            return False
        
        return datetime.utcnow() >= self.next_run
    
    def update_next_run(self):
        """Update next run time."""
        if self.schedule:
            # Cron schedule
            try:
                cron = croniter.croniter(self.schedule, datetime.utcnow())
                self.next_run = cron.get_next(datetime)
            except Exception as e:
                logger.error(f"Error updating cron for {self.name}: {e}")
                self.enabled = False
        
        elif self.interval:
            # Interval schedule
            self.next_run = datetime.utcnow() + timedelta(seconds=self.interval)
    
    def add_execution(self, execution: JobExecution):
        """Add execution record."""
        self.executions.append(execution)
        
        # Limit history
        if len(self.executions) > self.max_history:
            self.executions = self.executions[-self.max_history:]


class JobScheduler:
    """Background job scheduler."""
    
    def __init__(self):
        """Initialize job scheduler."""
        self.jobs: Dict[str, ScheduledJob] = {}
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
    
    def schedule(
        self,
        name: str,
        func: Callable,
        schedule: Optional[str] = None,
        interval: Optional[int] = None,
        args: tuple = (),
        kwargs: dict = None,
        enabled: bool = True
    ):
        """
        Schedule a job.
        
        Args:
            name: Job name (unique)
            func: Function to execute
            schedule: Cron expression (e.g., "0 * * * *")
            interval: Interval in seconds
            args: Function arguments
            kwargs: Function keyword arguments
            enabled: Whether job is enabled
        """
        if not schedule and not interval:
            raise ValueError("Either schedule or interval must be provided")
        
        job = ScheduledJob(
            name=name,
            func=func,
            schedule=schedule,
            interval=interval,
            args=args,
            kwargs=kwargs,
            enabled=enabled
        )
        
        self.jobs[name] = job
        
        logger.info(
            f"Scheduled job: {name} "
            f"({'cron: ' + schedule if schedule else 'interval: ' + str(interval) + 's'})"
        )
    
    def unschedule(self, name: str):
        """
        Unschedule a job.
        
        Args:
            name: Job name
        """
        if name in self.jobs:
            del self.jobs[name]
            logger.info(f"Unscheduled job: {name}")
    
    def enable_job(self, name: str):
        """Enable a job."""
        if name in self.jobs:
            self.jobs[name].enabled = True
            logger.info(f"Enabled job: {name}")
    
    def disable_job(self, name: str):
        """Disable a job."""
        if name in self.jobs:
            self.jobs[name].enabled = False
            logger.info(f"Disabled job: {name}")
    
    async def run_job_now(self, name: str) -> JobExecution:
        """
        Run job immediately.
        
        Args:
            name: Job name
        
        Returns:
            Job execution record
        """
        if name not in self.jobs:
            raise ValueError(f"Job not found: {name}")
        
        job = self.jobs[name]
        return await self._execute_job(job)
    
    async def _execute_job(self, job: ScheduledJob) -> JobExecution:
        """Execute a job."""
        execution = JobExecution(job.name)
        job.add_execution(execution)
        
        logger.info(f"Executing job: {job.name}")
        
        try:
            # Execute function
            if asyncio.iscoroutinefunction(job.func):
                result = await job.func(*job.args, **job.kwargs)
            else:
                # Run sync function in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: job.func(*job.args, **job.kwargs)
                )
            
            execution.complete(result)
            job.last_run = datetime.utcnow()
            
            logger.info(
                f"Job completed: {job.name} "
                f"(duration: {execution.duration():.2f}s)"
            )
        
        except Exception as e:
            execution.fail(str(e))
            logger.error(f"Job failed: {job.name}: {e}")
        
        finally:
            # Update next run time
            job.update_next_run()
        
        return execution
    
    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self._running:
            try:
                # Check all jobs
                for job in list(self.jobs.values()):
                    if job.should_run():
                        # Execute job in background
                        asyncio.create_task(self._execute_job(job))
                
                # Sleep for 1 second
                await asyncio.sleep(1)
            
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(1)
    
    async def start(self):
        """Start the scheduler."""
        if self._running:
            logger.warning("Scheduler already running")
            return
        
        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        
        logger.info("Job scheduler started")
    
    async def stop(self):
        """Stop the scheduler."""
        if not self._running:
            return
        
        self._running = False
        
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Job scheduler stopped")
    
    def get_job_status(self, name: str) -> Optional[dict]:
        """Get job status."""
        if name not in self.jobs:
            return None
        
        job = self.jobs[name]
        
        # Get recent executions
        recent = job.executions[-10:] if job.executions else []
        
        return {
            "name": job.name,
            "enabled": job.enabled,
            "schedule": job.schedule,
            "interval": job.interval,
            "last_run": job.last_run.isoformat() if job.last_run else None,
            "next_run": job.next_run.isoformat() if job.next_run else None,
            "total_executions": len(job.executions),
            "recent_executions": [
                {
                    "started_at": e.started_at.isoformat(),
                    "status": e.status.value,
                    "duration": e.duration(),
                    "error": e.error
                }
                for e in recent
            ]
        }
    
    def get_all_jobs(self) -> List[dict]:
        """Get all job statuses."""
        return [
            self.get_job_status(name)
            for name in self.jobs.keys()
        ]


# Global scheduler
scheduler = JobScheduler()


# Decorator for scheduling jobs
def scheduled_job(schedule: Optional[str] = None, interval: Optional[int] = None):
    """
    Decorator to schedule a job.
    
    Args:
        schedule: Cron expression
        interval: Interval in seconds
    """
    def decorator(func: Callable):
        job_name = func.__name__
        scheduler.schedule(
            name=job_name,
            func=func,
            schedule=schedule,
            interval=interval
        )
        return func
    
    return decorator


# Example usage:
"""
from fastapi import FastAPI
from src.job_scheduler import scheduler, scheduled_job

app = FastAPI()

# Schedule with decorator
@scheduled_job(schedule="0 * * * *")  # Every hour
async def hourly_cleanup():
    print("Running hourly cleanup...")
    await asyncio.sleep(1)
    print("Cleanup complete")

# Schedule manually
async def backup_database():
    print("Backing up database...")
    await asyncio.sleep(2)
    print("Backup complete")

scheduler.schedule(
    name="database_backup",
    func=backup_database,
    schedule="0 0 * * *"  # Daily at midnight
)

# Interval-based job
scheduler.schedule(
    name="health_check",
    func=lambda: print("Health check"),
    interval=60  # Every 60 seconds
)

@app.on_event("startup")
async def startup():
    await scheduler.start()

@app.on_event("shutdown")
async def shutdown():
    await scheduler.stop()

@app.get("/jobs")
async def list_jobs():
    return scheduler.get_all_jobs()

@app.post("/jobs/{name}/run")
async def run_job(name: str):
    execution = await scheduler.run_job_now(name)
    return {
        "status": execution.status.value,
        "duration": execution.duration()
    }
"""
