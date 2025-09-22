"""
Background Task Manager for Project QuickNav

Provides comprehensive background task management including:
- Async task queue with priority support
- Task scheduling and cron-like functionality
- Resource throttling and backpressure handling
- Task retry and error handling
- Progress tracking and cancellation
- Worker pool management
- Real-time task monitoring
"""

import asyncio
import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union, AsyncIterator
from dataclasses import dataclass, asdict, field
from enum import Enum
import json
import uuid
from collections import defaultdict, deque
import functools
import heapq
import signal
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class TaskProgress:
    """Task progress information."""
    current: int = 0
    total: int = 100
    message: str = ""
    percentage: float = 0.0

    def update(self, current: int, total: int = None, message: str = None):
        """Update progress."""
        self.current = current
        if total is not None:
            self.total = total
        if message is not None:
            self.message = message
        self.percentage = (self.current / self.total) * 100 if self.total > 0 else 0


@dataclass
class TaskResult:
    """Task execution result."""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time_ms: float = 0
    retry_count: int = 0
    progress: TaskProgress = field(default_factory=TaskProgress)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskDefinition:
    """Task definition with metadata."""
    task_id: str
    name: str
    function: Callable
    args: tuple
    kwargs: dict
    priority: TaskPriority
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    timeout_seconds: Optional[float] = None
    scheduled_at: Optional[datetime] = None
    depends_on: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


class ResourceManager:
    """Manages resource allocation for tasks."""

    def __init__(self, max_cpu_tasks: int = 4, max_io_tasks: int = 10, max_memory_mb: int = 1024):
        self.max_cpu_tasks = max_cpu_tasks
        self.max_io_tasks = max_io_tasks
        self.max_memory_mb = max_memory_mb

        self.active_cpu_tasks = 0
        self.active_io_tasks = 0
        self.allocated_memory_mb = 0

        self.lock = threading.Lock()

    def can_allocate(self, requirements: Dict[str, Any]) -> bool:
        """Check if resources can be allocated for task."""
        with self.lock:
            cpu_needed = requirements.get('cpu_intensive', False)
            io_needed = requirements.get('io_intensive', False)
            memory_needed = requirements.get('memory_mb', 0)

            if cpu_needed and self.active_cpu_tasks >= self.max_cpu_tasks:
                return False

            if io_needed and self.active_io_tasks >= self.max_io_tasks:
                return False

            if self.allocated_memory_mb + memory_needed > self.max_memory_mb:
                return False

            return True

    def allocate(self, requirements: Dict[str, Any]) -> bool:
        """Allocate resources for task."""
        with self.lock:
            if not self.can_allocate(requirements):
                return False

            if requirements.get('cpu_intensive', False):
                self.active_cpu_tasks += 1

            if requirements.get('io_intensive', False):
                self.active_io_tasks += 1

            memory_needed = requirements.get('memory_mb', 0)
            self.allocated_memory_mb += memory_needed

            return True

    def release(self, requirements: Dict[str, Any]):
        """Release resources after task completion."""
        with self.lock:
            if requirements.get('cpu_intensive', False):
                self.active_cpu_tasks = max(0, self.active_cpu_tasks - 1)

            if requirements.get('io_intensive', False):
                self.active_io_tasks = max(0, self.active_io_tasks - 1)

            memory_used = requirements.get('memory_mb', 0)
            self.allocated_memory_mb = max(0, self.allocated_memory_mb - memory_used)

    def get_stats(self) -> Dict[str, Any]:
        """Get resource utilization statistics."""
        with self.lock:
            return {
                'cpu_utilization': self.active_cpu_tasks / self.max_cpu_tasks,
                'io_utilization': self.active_io_tasks / self.max_io_tasks,
                'memory_utilization': self.allocated_memory_mb / self.max_memory_mb,
                'active_cpu_tasks': self.active_cpu_tasks,
                'active_io_tasks': self.active_io_tasks,
                'allocated_memory_mb': self.allocated_memory_mb
            }


class TaskQueue:
    """Priority-based task queue with backpressure handling."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.queue = []  # Priority heap
        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)
        self.not_full = threading.Condition(self.lock)
        self.counter = 0  # For stable sorting

    def put(self, task: TaskDefinition, block: bool = True, timeout: Optional[float] = None) -> bool:
        """Add task to queue."""
        with self.not_full:
            if not block and len(self.queue) >= self.max_size:
                return False

            if block:
                end_time = time.time() + timeout if timeout else None
                while len(self.queue) >= self.max_size:
                    if timeout:
                        remaining = end_time - time.time()
                        if remaining <= 0:
                            return False
                        self.not_full.wait(remaining)
                    else:
                        self.not_full.wait()

            # Add to priority heap
            priority = task.priority.value
            self.counter += 1
            heapq.heappush(self.queue, (priority, self.counter, task))

            self.not_empty.notify()
            return True

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[TaskDefinition]:
        """Get highest priority task from queue."""
        with self.not_empty:
            if not block and not self.queue:
                return None

            if block:
                end_time = time.time() + timeout if timeout else None
                while not self.queue:
                    if timeout:
                        remaining = end_time - time.time()
                        if remaining <= 0:
                            return None
                        self.not_empty.wait(remaining)
                    else:
                        self.not_empty.wait()

            priority, counter, task = heapq.heappop(self.queue)
            self.not_full.notify()
            return task

    def size(self) -> int:
        """Get queue size."""
        with self.lock:
            return len(self.queue)

    def is_full(self) -> bool:
        """Check if queue is full."""
        with self.lock:
            return len(self.queue) >= self.max_size

    def clear(self):
        """Clear all tasks from queue."""
        with self.lock:
            self.queue.clear()
            self.not_full.notify_all()


class TaskScheduler:
    """Cron-like task scheduler."""

    def __init__(self, task_manager):
        self.task_manager = task_manager
        self.scheduled_tasks = {}
        self.scheduler_active = False
        self.scheduler_thread = None

    def start(self):
        """Start task scheduler."""
        if self.scheduler_active:
            return

        self.scheduler_active = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        logger.info("Task scheduler started")

    def stop(self):
        """Stop task scheduler."""
        self.scheduler_active = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5.0)
        logger.info("Task scheduler stopped")

    def schedule_recurring(self, name: str, function: Callable, interval_seconds: float,
                          args: tuple = (), kwargs: dict = None, priority: TaskPriority = TaskPriority.NORMAL):
        """Schedule recurring task."""
        kwargs = kwargs or {}
        self.scheduled_tasks[name] = {
            'function': function,
            'args': args,
            'kwargs': kwargs,
            'interval_seconds': interval_seconds,
            'priority': priority,
            'next_run': datetime.utcnow() + timedelta(seconds=interval_seconds),
            'last_run': None
        }
        logger.info(f"Scheduled recurring task: {name} (every {interval_seconds}s)")

    def schedule_at(self, name: str, function: Callable, run_at: datetime,
                   args: tuple = (), kwargs: dict = None, priority: TaskPriority = TaskPriority.NORMAL):
        """Schedule task to run at specific time."""
        kwargs = kwargs or {}
        self.scheduled_tasks[name] = {
            'function': function,
            'args': args,
            'kwargs': kwargs,
            'interval_seconds': None,  # One-time task
            'priority': priority,
            'next_run': run_at,
            'last_run': None
        }
        logger.info(f"Scheduled task: {name} at {run_at}")

    def unschedule(self, name: str) -> bool:
        """Remove scheduled task."""
        if name in self.scheduled_tasks:
            del self.scheduled_tasks[name]
            logger.info(f"Unscheduled task: {name}")
            return True
        return False

    def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.scheduler_active:
            try:
                now = datetime.utcnow()
                tasks_to_run = []

                # Check for tasks ready to run
                for name, task_info in list(self.scheduled_tasks.items()):
                    if now >= task_info['next_run']:
                        tasks_to_run.append((name, task_info))

                # Submit ready tasks
                for name, task_info in tasks_to_run:
                    try:
                        # Submit task
                        self.task_manager.submit_task(
                            name=f"scheduled_{name}",
                            function=task_info['function'],
                            args=task_info['args'],
                            kwargs=task_info['kwargs'],
                            priority=task_info['priority']
                        )

                        task_info['last_run'] = now

                        # Update next run time for recurring tasks
                        if task_info['interval_seconds']:
                            task_info['next_run'] = now + timedelta(seconds=task_info['interval_seconds'])
                        else:
                            # Remove one-time tasks
                            del self.scheduled_tasks[name]

                    except Exception as e:
                        logger.error(f"Failed to submit scheduled task {name}: {e}")

                time.sleep(1)  # Check every second

            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                time.sleep(5)


class TaskWorker:
    """Individual task worker."""

    def __init__(self, worker_id: str, task_manager):
        self.worker_id = worker_id
        self.task_manager = task_manager
        self.current_task = None
        self.active = False
        self.worker_thread = None
        self.stats = {
            'tasks_completed': 0,
            'total_execution_time': 0,
            'errors': 0,
            'started_at': datetime.utcnow()
        }

    def start(self):
        """Start worker."""
        if self.active:
            return

        self.active = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.debug(f"Worker {self.worker_id} started")

    def stop(self):
        """Stop worker."""
        self.active = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        logger.debug(f"Worker {self.worker_id} stopped")

    def _worker_loop(self):
        """Main worker loop."""
        while self.active:
            try:
                # Get task from queue
                task = self.task_manager.task_queue.get(timeout=1.0)
                if not task:
                    continue

                self.current_task = task
                result = self._execute_task(task)
                self.task_manager._handle_task_completion(task, result)
                self.current_task = None

            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {e}")
                if self.current_task:
                    error_result = TaskResult(
                        task_id=self.current_task.task_id,
                        status=TaskStatus.FAILED,
                        error=str(e)
                    )
                    self.task_manager._handle_task_completion(self.current_task, error_result)
                    self.current_task = None

    def _execute_task(self, task: TaskDefinition) -> TaskResult:
        """Execute a single task."""
        start_time = time.perf_counter()
        result = TaskResult(
            task_id=task.task_id,
            status=TaskStatus.RUNNING,
            started_at=datetime.utcnow()
        )

        try:
            # Check dependencies
            if not self.task_manager._check_dependencies(task):
                result.status = TaskStatus.FAILED
                result.error = "Dependencies not satisfied"
                return result

            # Allocate resources
            if not self.task_manager.resource_manager.allocate(task.resource_requirements):
                result.status = TaskStatus.FAILED
                result.error = "Insufficient resources"
                return result

            try:
                # Create progress callback
                def update_progress(current: int, total: int = None, message: str = None):
                    result.progress.update(current, total, message)
                    self.task_manager._notify_progress(task.task_id, result.progress)

                # Execute task function
                if asyncio.iscoroutinefunction(task.function):
                    # Run async function
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        # Add progress callback to kwargs if function supports it
                        if 'progress_callback' in task.function.__code__.co_varnames:
                            task.kwargs['progress_callback'] = update_progress

                        task_result = loop.run_until_complete(
                            asyncio.wait_for(
                                task.function(*task.args, **task.kwargs),
                                timeout=task.timeout_seconds
                            )
                        )
                    finally:
                        loop.close()
                else:
                    # Run sync function
                    if 'progress_callback' in task.function.__code__.co_varnames:
                        task.kwargs['progress_callback'] = update_progress

                    task_result = task.function(*task.args, **task.kwargs)

                result.result = task_result
                result.status = TaskStatus.COMPLETED
                result.progress.update(result.progress.total, result.progress.total, "Completed")

            finally:
                # Release resources
                self.task_manager.resource_manager.release(task.resource_requirements)

            self.stats['tasks_completed'] += 1

        except asyncio.TimeoutError:
            result.status = TaskStatus.FAILED
            result.error = "Task timed out"
            self.stats['errors'] += 1

        except Exception as e:
            result.status = TaskStatus.FAILED
            result.error = str(e)
            self.stats['errors'] += 1

        finally:
            execution_time = (time.perf_counter() - start_time) * 1000
            result.execution_time_ms = execution_time
            result.completed_at = datetime.utcnow()
            self.stats['total_execution_time'] += execution_time

        return result


class BackgroundTaskManager:
    """Main background task manager."""

    def __init__(self, num_workers: int = 4, queue_size: int = 1000):
        self.num_workers = num_workers
        self.task_queue = TaskQueue(queue_size)
        self.resource_manager = ResourceManager()
        self.scheduler = TaskScheduler(self)

        # Workers
        self.workers = []
        self.worker_stats = {}

        # Task tracking
        self.active_tasks = {}
        self.completed_tasks = {}
        self.task_results = {}
        self.task_dependencies = defaultdict(set)
        self.task_dependents = defaultdict(set)

        # Callbacks
        self.progress_callbacks = {}
        self.completion_callbacks = {}

        # Monitoring
        self.stats = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_execution_time': 0,
            'started_at': datetime.utcnow()
        }
        self.stats_lock = threading.Lock()

        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()

    def start(self):
        """Start task manager."""
        # Start workers
        for i in range(self.num_workers):
            worker = TaskWorker(f"worker_{i}", self)
            worker.start()
            self.workers.append(worker)

        # Start scheduler
        self.scheduler.start()

        logger.info(f"Background task manager started with {self.num_workers} workers")

    def stop(self):
        """Stop task manager."""
        # Stop scheduler
        self.scheduler.stop()

        # Stop workers
        for worker in self.workers:
            worker.stop()

        # Clear queues
        self.task_queue.clear()

        logger.info("Background task manager stopped")

    def submit_task(self,
                   name: str,
                   function: Callable,
                   args: tuple = (),
                   kwargs: dict = None,
                   priority: TaskPriority = TaskPriority.NORMAL,
                   max_retries: int = 3,
                   retry_delay_seconds: float = 1.0,
                   timeout_seconds: Optional[float] = None,
                   depends_on: List[str] = None,
                   tags: List[str] = None,
                   resource_requirements: Dict[str, Any] = None,
                   progress_callback: Optional[Callable] = None,
                   completion_callback: Optional[Callable] = None) -> str:
        """Submit task for background execution."""

        task_id = str(uuid.uuid4())
        kwargs = kwargs or {}
        depends_on = depends_on or []
        tags = tags or []
        resource_requirements = resource_requirements or {}

        # Create task definition
        task = TaskDefinition(
            task_id=task_id,
            name=name,
            function=function,
            args=args,
            kwargs=kwargs,
            priority=priority,
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds,
            timeout_seconds=timeout_seconds,
            depends_on=depends_on,
            tags=tags,
            resource_requirements=resource_requirements
        )

        # Register callbacks
        if progress_callback:
            self.progress_callbacks[task_id] = progress_callback

        if completion_callback:
            self.completion_callbacks[task_id] = completion_callback

        # Setup dependencies
        for dep_task_id in depends_on:
            self.task_dependencies[task_id].add(dep_task_id)
            self.task_dependents[dep_task_id].add(task_id)

        # Add to queue
        if self.task_queue.put(task, block=False):
            self.active_tasks[task_id] = task

            with self.stats_lock:
                self.stats['tasks_submitted'] += 1

            logger.debug(f"Submitted task: {name} ({task_id})")
            return task_id
        else:
            raise Exception("Task queue is full")

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task."""
        if task_id in self.active_tasks:
            # Mark as cancelled
            result = TaskResult(
                task_id=task_id,
                status=TaskStatus.CANCELLED,
                completed_at=datetime.utcnow()
            )
            self._handle_task_completion(self.active_tasks[task_id], result)
            return True
        return False

    def get_task_status(self, task_id: str) -> Optional[TaskResult]:
        """Get current status of a task."""
        if task_id in self.task_results:
            return self.task_results[task_id]

        if task_id in self.active_tasks:
            # Task is pending or running
            for worker in self.workers:
                if worker.current_task and worker.current_task.task_id == task_id:
                    return TaskResult(
                        task_id=task_id,
                        status=TaskStatus.RUNNING,
                        started_at=datetime.utcnow()
                    )

            return TaskResult(
                task_id=task_id,
                status=TaskStatus.PENDING
            )

        return None

    def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Optional[TaskResult]:
        """Wait for task completion."""
        start_time = time.time()

        while True:
            result = self.get_task_status(task_id)
            if result and result.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                return result

            if timeout and (time.time() - start_time) > timeout:
                return None

            time.sleep(0.1)

    def get_tasks_by_tag(self, tag: str) -> List[TaskDefinition]:
        """Get all tasks with specific tag."""
        tasks = []
        for task in self.active_tasks.values():
            if tag in task.tags:
                tasks.append(task)
        return tasks

    def _check_dependencies(self, task: TaskDefinition) -> bool:
        """Check if task dependencies are satisfied."""
        for dep_task_id in task.depends_on:
            dep_result = self.get_task_status(dep_task_id)
            if not dep_result or dep_result.status != TaskStatus.COMPLETED:
                return False
        return True

    def _handle_task_completion(self, task: TaskDefinition, result: TaskResult):
        """Handle task completion."""
        # Store result
        self.task_results[task.task_id] = result

        # Remove from active tasks
        if task.task_id in self.active_tasks:
            del self.active_tasks[task.task_id]

        # Update statistics
        with self.stats_lock:
            if result.status == TaskStatus.COMPLETED:
                self.stats['tasks_completed'] += 1
            elif result.status == TaskStatus.FAILED:
                self.stats['tasks_failed'] += 1

            if result.execution_time_ms:
                self.stats['total_execution_time'] += result.execution_time_ms

        # Handle retries for failed tasks
        if result.status == TaskStatus.FAILED and result.retry_count < task.max_retries:
            self._retry_task(task, result.retry_count + 1)
            return

        # Call completion callback
        if task.task_id in self.completion_callbacks:
            try:
                self.completion_callbacks[task.task_id](result)
            except Exception as e:
                logger.error(f"Completion callback error: {e}")
            finally:
                del self.completion_callbacks[task.task_id]

        # Submit dependent tasks
        self._submit_dependent_tasks(task.task_id)

        logger.debug(f"Task completed: {task.name} ({task.task_id}) - {result.status.value}")

    def _retry_task(self, task: TaskDefinition, retry_count: int):
        """Retry a failed task."""
        # Create new task with updated retry count
        retry_task = TaskDefinition(
            task_id=task.task_id,
            name=f"{task.name} (retry {retry_count})",
            function=task.function,
            args=task.args,
            kwargs=task.kwargs,
            priority=task.priority,
            max_retries=task.max_retries,
            retry_delay_seconds=task.retry_delay_seconds,
            timeout_seconds=task.timeout_seconds,
            depends_on=task.depends_on,
            tags=task.tags + ['retry'],
            resource_requirements=task.resource_requirements
        )

        # Schedule retry after delay
        def delayed_retry():
            time.sleep(task.retry_delay_seconds)
            self.task_queue.put(retry_task)
            self.active_tasks[task.task_id] = retry_task

        retry_thread = threading.Thread(target=delayed_retry, daemon=True)
        retry_thread.start()

        logger.info(f"Retrying task: {task.name} (attempt {retry_count + 1}/{task.max_retries + 1})")

    def _submit_dependent_tasks(self, completed_task_id: str):
        """Submit tasks that depend on the completed task."""
        if completed_task_id in self.task_dependents:
            for dependent_task_id in self.task_dependents[completed_task_id]:
                if dependent_task_id in self.active_tasks:
                    task = self.active_tasks[dependent_task_id]
                    if self._check_dependencies(task):
                        # Dependencies satisfied, task will be picked up by workers
                        pass

    def _notify_progress(self, task_id: str, progress: TaskProgress):
        """Notify progress callback."""
        if task_id in self.progress_callbacks:
            try:
                self.progress_callbacks[task_id](progress)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive task manager statistics."""
        with self.stats_lock:
            stats = self.stats.copy()

        # Add queue and resource stats
        stats.update({
            'queue_size': self.task_queue.size(),
            'active_tasks': len(self.active_tasks),
            'worker_count': len(self.workers),
            'resource_stats': self.resource_manager.get_stats()
        })

        # Worker statistics
        worker_stats = {}
        for worker in self.workers:
            worker_stats[worker.worker_id] = {
                'current_task': worker.current_task.name if worker.current_task else None,
                'tasks_completed': worker.stats['tasks_completed'],
                'total_execution_time': worker.stats['total_execution_time'],
                'errors': worker.stats['errors'],
                'uptime_hours': (datetime.utcnow() - worker.stats['started_at']).total_seconds() / 3600
            }

        stats['workers'] = worker_stats

        return stats

    def get_task_summary(self) -> Dict[str, Any]:
        """Get summary of all tasks."""
        summary = {
            'by_status': defaultdict(int),
            'by_priority': defaultdict(int),
            'by_tag': defaultdict(int),
            'recent_completions': []
        }

        # Count active tasks
        for task in self.active_tasks.values():
            summary['by_status']['pending'] += 1
            summary['by_priority'][task.priority.name] += 1
            for tag in task.tags:
                summary['by_tag'][tag] += 1

        # Count completed tasks
        for result in self.task_results.values():
            summary['by_status'][result.status.value] += 1

        # Recent completions
        recent_results = sorted(
            [r for r in self.task_results.values() if r.completed_at],
            key=lambda x: x.completed_at,
            reverse=True
        )[:10]

        summary['recent_completions'] = [
            {
                'task_id': r.task_id,
                'status': r.status.value,
                'completed_at': r.completed_at.isoformat() if r.completed_at else None,
                'execution_time_ms': r.execution_time_ms
            }
            for r in recent_results
        ]

        return summary

    def cleanup_completed_tasks(self, older_than_hours: int = 24):
        """Clean up old completed task results."""
        cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)

        tasks_to_remove = []
        for task_id, result in self.task_results.items():
            if result.completed_at and result.completed_at < cutoff_time:
                tasks_to_remove.append(task_id)

        for task_id in tasks_to_remove:
            del self.task_results[task_id]
            # Clean up callbacks if any
            self.progress_callbacks.pop(task_id, None)
            self.completion_callbacks.pop(task_id, None)

        logger.info(f"Cleaned up {len(tasks_to_remove)} old task results")

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


# Decorator for background task execution
def background_task(task_manager: BackgroundTaskManager = None,
                   priority: TaskPriority = TaskPriority.NORMAL,
                   max_retries: int = 3,
                   timeout_seconds: Optional[float] = None,
                   tags: List[str] = None):
    """Decorator for executing functions as background tasks."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            manager = task_manager or get_task_manager()
            return manager.submit_task(
                name=f"{func.__module__}.{func.__name__}",
                function=func,
                args=args,
                kwargs=kwargs,
                priority=priority,
                max_retries=max_retries,
                timeout_seconds=timeout_seconds,
                tags=tags or []
            )
        return wrapper
    return decorator


# Global task manager instance
_global_task_manager: Optional[BackgroundTaskManager] = None


def get_task_manager() -> BackgroundTaskManager:
    """Get or create global task manager."""
    global _global_task_manager
    if _global_task_manager is None:
        _global_task_manager = BackgroundTaskManager()
        _global_task_manager.start()
    return _global_task_manager