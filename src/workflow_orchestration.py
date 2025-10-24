# Workflow Orchestration Engine

import threading
import time
import json
import uuid
from typing import Dict, Any, List, Optional, Callable, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"

class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"

@dataclass
class Task:
    """Workflow task."""
    task_id: str
    name: str
    handler: Callable
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    timeout_sec: float = 300.0
    params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'task_id': self.task_id,
            'name': self.name,
            'dependencies': self.dependencies,
            'retry_count': self.retry_count,
            'timeout_sec': self.timeout_sec
        }

@dataclass
class TaskExecution:
    """Task execution record."""
    execution_id: str
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        duration = None
        if self.start_time and self.end_time:
            duration = self.end_time - self.start_time
        
        return {
            'execution_id': self.execution_id,
            'task_id': self.task_id,
            'status': self.status.value,
            'duration_sec': duration,
            'error': self.error
        }

class Workflow:
    """Workflow definition."""
    
    def __init__(self, workflow_id: str, name: str):
        self.workflow_id = workflow_id
        self.name = name
        self.tasks: Dict[str, Task] = {}
        self.edges: Dict[str, List[str]] = defaultdict(list)
        self.created_at = time.time()
    
    def add_task(self, task: Task) -> None:
        """Add task to workflow."""
        self.tasks[task.task_id] = task
        
        # Build dependency graph
        for dep in task.dependencies:
            self.edges[dep].append(task.task_id)
    
    def get_executable_tasks(self, completed_tasks: Set[str]) -> List[str]:
        """Get tasks ready to execute."""
        executable = []
        
        for task_id, task in self.tasks.items():
            if task_id in completed_tasks:
                continue
            
            # Check if all dependencies completed
            deps_completed = all(
                dep in completed_tasks for dep in task.dependencies
            )
            
            if deps_completed:
                executable.append(task_id)
        
        return executable
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'workflow_id': self.workflow_id,
            'name': self.name,
            'task_count': len(self.tasks),
            'created_at': self.created_at
        }

class WorkflowExecutor:
    """Execute workflows."""
    
    def __init__(self, workflow: Workflow):
        self.workflow = workflow
        self.execution_id = str(uuid.uuid4())
        self.status = WorkflowStatus.PENDING
        self.task_executions: Dict[str, TaskExecution] = {}
        self.completed_tasks: Set[str] = set()
        self.failed_tasks: Set[str] = set()
        
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
        self.lock = threading.RLock()
    
    def execute(self) -> Tuple[bool, Dict]:
        """Execute workflow."""
        with self.lock:
            self.status = WorkflowStatus.RUNNING
            self.start_time = time.time()
        
        try:
            while True:
                # Get executable tasks
                executable = self.workflow.get_executable_tasks(self.completed_tasks)
                
                if not executable:
                    # Check if workflow complete
                    if len(self.completed_tasks) == len(self.workflow.tasks):
                        with self.lock:
                            self.status = WorkflowStatus.COMPLETED
                        break
                    
                    # Check for failed tasks
                    if self.failed_tasks:
                        with self.lock:
                            self.status = WorkflowStatus.FAILED
                        break
                    
                    break
                
                # Execute tasks
                for task_id in executable:
                    self._execute_task(task_id)
                
                time.sleep(0.1)  # Small delay to allow task execution
        
        except Exception as e:
            with self.lock:
                self.status = WorkflowStatus.FAILED
            
            return False, {'error': str(e)}
        
        finally:
            with self.lock:
                self.end_time = time.time()
        
        return self.status == WorkflowStatus.COMPLETED, self.get_status()
    
    def _execute_task(self, task_id: str) -> None:
        """Execute single task."""
        task = self.workflow.tasks[task_id]
        
        # Create execution record
        exec_id = str(uuid.uuid4())
        execution = TaskExecution(
            execution_id=exec_id,
            task_id=task_id,
            status=TaskStatus.RUNNING
        )
        
        with self.lock:
            self.task_executions[exec_id] = execution
        
        execution.start_time = time.time()
        
        try:
            # Execute task handler
            result = task.handler(**task.params)
            
            execution.result = result
            execution.status = TaskStatus.COMPLETED
            
            with self.lock:
                self.completed_tasks.add(task_id)
        
        except Exception as e:
            execution.error = str(e)
            
            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                execution.status = TaskStatus.RETRYING
            else:
                execution.status = TaskStatus.FAILED
                
                with self.lock:
                    self.failed_tasks.add(task_id)
        
        finally:
            execution.end_time = time.time()
    
    def get_status(self) -> Dict:
        """Get workflow execution status."""
        with self.lock:
            duration = None
            if self.start_time and self.end_time:
                duration = self.end_time - self.start_time
            
            return {
                'execution_id': self.execution_id,
                'workflow': self.workflow.to_dict(),
                'status': self.status.value,
                'duration_sec': duration,
                'completed_tasks': len(self.completed_tasks),
                'failed_tasks': len(self.failed_tasks),
                'total_tasks': len(self.workflow.tasks),
                'task_executions': [
                    exec.to_dict() for exec in self.task_executions.values()
                ]
            }

class WorkflowBuilder:
    """Build workflows fluently."""
    
    def __init__(self, workflow_id: str, name: str):
        self.workflow = Workflow(workflow_id, name)
    
    def add_task(self, name: str, handler: Callable, 
                dependencies: List[str] = None, 
                params: Dict = None) -> 'WorkflowBuilder':
        """Add task to workflow."""
        task_id = f"task_{len(self.workflow.tasks)}"
        
        task = Task(
            task_id=task_id,
            name=name,
            handler=handler,
            dependencies=dependencies or [],
            params=params or {}
        )
        
        self.workflow.add_task(task)
        return self
    
    def build(self) -> Workflow:
        """Build workflow."""
        return self.workflow

class WorkflowScheduler:
    """Schedule and manage workflows."""
    
    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self.active_executions: Dict[str, WorkflowExecutor] = {}
        self.completed_executions: List[Dict] = []
        self.lock = threading.RLock()
    
    def submit_workflow(self, workflow: Workflow) -> str:
        """Submit workflow for execution."""
        with self.lock:
            if len(self.active_executions) >= self.max_concurrent:
                raise Exception("Max concurrent workflows reached")
            
            executor = WorkflowExecutor(workflow)
            self.active_executions[executor.execution_id] = executor
            
            # Execute in thread
            thread = threading.Thread(target=self._run_workflow, args=(executor,))
            thread.daemon = True
            thread.start()
            
            return executor.execution_id
    
    def _run_workflow(self, executor: WorkflowExecutor) -> None:
        """Run workflow in thread."""
        try:
            success, result = executor.execute()
            
            with self.lock:
                self.completed_executions.append(result)
                del self.active_executions[executor.execution_id]
        
        except Exception as e:
            with self.lock:
                del self.active_executions[executor.execution_id]
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict]:
        """Get execution status."""
        with self.lock:
            executor = self.active_executions.get(execution_id)
            
            if executor:
                return executor.get_status()
            
            return None
    
    def get_scheduler_status(self) -> Dict:
        """Get scheduler status."""
        with self.lock:
            return {
                'active_workflows': len(self.active_executions),
                'completed_workflows': len(self.completed_executions),
                'max_concurrent': self.max_concurrent,
                'utilization': len(self.active_executions) / self.max_concurrent
            }

# Example usage
if __name__ == "__main__":
    # Define workflow
    def task1():
        time.sleep(0.2)
        return "Task 1 complete"
    
    def task2():
        time.sleep(0.2)
        return "Task 2 complete"
    
    def task3_handler():
        time.sleep(0.2)
        return "Task 3 complete"
    
    # Build workflow
    builder = WorkflowBuilder("wf1", "Sample Workflow")
    workflow = (builder
                .add_task("Initialize", task1)
                .add_task("Process", task2, dependencies=["task_0"])
                .add_task("Finalize", task3_handler, dependencies=["task_1"])
                .build())
    
    # Execute workflow
    executor = WorkflowExecutor(workflow)
    success, result = executor.execute()
    
    print(f"Workflow Success: {success}")
    print(f"Status:")
    print(json.dumps(result, indent=2, default=str))
