"""
Async Scheduler - High-performance task scheduling for pipeline operations

Features:
- Cron-like scheduling with async/await support
- Priority-based task execution
- Resource-aware scheduling
- Backpressure handling
- Task dependencies and chaining
- Persistent state management
- Performance monitoring
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Awaitable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import croniter
import heapq
from collections import defaultdict

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


@dataclass
class ScheduledTask:
    """Represents a scheduled task with all metadata."""
    task_id: str
    coro: Callable[..., Awaitable[Any]]
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    next_run: float = 0.0
    interval: Optional[float] = None
    cron_expression: Optional[str] = None
    max_retries: int = 3
    retry_delay: float = 60.0
    timeout: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)

    # Runtime state
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = 0
    last_run: Optional[float] = None
    last_error: Optional[str] = None
    total_runtime: float = 0.0
    avg_runtime: float = 0.0

    def __lt__(self, other):
        """Priority queue comparison - lower priority value = higher priority."""
        if self.priority != other.priority:
            return self.priority.value < other.priority.value
        return self.next_run < other.next_run


class ResourceManager:
    """Manages resource allocation for scheduled tasks."""

    def __init__(self, config: Dict[str, Any]):
        self.max_cpu_tasks = config.get('max_cpu_tasks', 4)
        self.max_memory_mb = config.get('max_memory_mb', 1024)
        self.max_io_tasks = config.get('max_io_tasks', 8)

        self.current_cpu_tasks = 0
        self.current_memory_mb = 0
        self.current_io_tasks = 0

        self._lock = asyncio.Lock()

    async def can_allocate(self, requirements: Dict[str, Any]) -> bool:
        """Check if resources can be allocated for a task."""
        async with self._lock:
            cpu_needed = requirements.get('cpu_intensive', False)
            memory_needed = requirements.get('memory_mb', 0)
            io_needed = requirements.get('io_intensive', False)

            if cpu_needed and self.current_cpu_tasks >= self.max_cpu_tasks:
                return False

            if memory_needed > 0 and (self.current_memory_mb + memory_needed) > self.max_memory_mb:
                return False

            if io_needed and self.current_io_tasks >= self.max_io_tasks:
                return False

            return True

    async def allocate(self, requirements: Dict[str, Any]) -> bool:
        """Allocate resources for a task."""
        async with self._lock:
            if not await self.can_allocate(requirements):
                return False

            if requirements.get('cpu_intensive', False):
                self.current_cpu_tasks += 1

            memory_needed = requirements.get('memory_mb', 0)
            if memory_needed > 0:
                self.current_memory_mb += memory_needed

            if requirements.get('io_intensive', False):
                self.current_io_tasks += 1

            return True

    async def release(self, requirements: Dict[str, Any]):
        """Release resources after task completion."""
        async with self._lock:
            if requirements.get('cpu_intensive', False):
                self.current_cpu_tasks = max(0, self.current_cpu_tasks - 1)

            memory_needed = requirements.get('memory_mb', 0)
            if memory_needed > 0:
                self.current_memory_mb = max(0, self.current_memory_mb - memory_needed)

            if requirements.get('io_intensive', False):
                self.current_io_tasks = max(0, self.current_io_tasks - 1)


class AsyncScheduler:
    """
    High-performance async task scheduler with resource management.

    Features:
    - Priority-based scheduling
    - Cron expressions and interval scheduling
    - Resource-aware execution
    - Task dependencies
    - Retry mechanisms
    - Performance tracking
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_concurrent_tasks = config.get('max_concurrent_tasks', 10)
        self.health_check_interval = config.get('health_check_interval', 30)

        # Task management
        self._tasks: Dict[str, ScheduledTask] = {}
        self._task_queue: List[ScheduledTask] = []
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._completed_tasks: Dict[str, ScheduledTask] = {}

        # Resource management
        self._resource_manager = ResourceManager(config.get('resources', {}))

        # State management
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self._semaphore = asyncio.Semaphore(self.max_concurrent_tasks)

        # Metrics and monitoring
        self._metrics = {
            'tasks_scheduled': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'tasks_retried': 0,
            'total_runtime': 0.0,
            'avg_task_runtime': 0.0
        }

        # Dependencies tracking
        self._dependency_graph: Dict[str, List[str]] = defaultdict(list)
        self._dependents: Dict[str, List[str]] = defaultdict(list)

    async def start(self):
        """Start the scheduler."""
        if self._running:
            logger.warning("Scheduler already running")
            return

        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Async scheduler started")

    async def shutdown(self, timeout: float = 30.0):
        """Shutdown the scheduler gracefully."""
        if not self._running:
            return

        logger.info("Shutting down scheduler...")
        self._running = False

        # Cancel scheduler loop
        if self._scheduler_task:
            self._scheduler_task.cancel()

        # Cancel running tasks
        if self._running_tasks:
            logger.info(f"Cancelling {len(self._running_tasks)} running tasks...")
            for task in self._running_tasks.values():
                task.cancel()

            # Wait for tasks to complete or timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._running_tasks.values(), return_exceptions=True),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning("Some tasks did not complete within timeout")

        logger.info("Scheduler shutdown complete")

    async def schedule_periodic(self, task_id: str, coro: Callable[..., Awaitable[Any]],
                              interval: float, start_delay: float = 0,
                              priority: TaskPriority = TaskPriority.NORMAL,
                              max_retries: int = 3, timeout: Optional[float] = None,
                              resource_requirements: Dict[str, Any] = None,
                              **kwargs) -> bool:
        """
        Schedule a periodic task.

        Args:
            task_id: Unique identifier for the task
            coro: Coroutine function to execute
            interval: Interval between executions in seconds
            start_delay: Delay before first execution
            priority: Task priority
            max_retries: Maximum retry attempts
            timeout: Task timeout in seconds
            resource_requirements: Resource requirements dict
            **kwargs: Arguments to pass to the coroutine

        Returns:
            True if scheduled successfully
        """
        try:
            if task_id in self._tasks:
                logger.warning(f"Task {task_id} already scheduled, updating...")
                await self.cancel_task(task_id)

            next_run = time.time() + start_delay

            task = ScheduledTask(
                task_id=task_id,
                coro=coro,
                kwargs=kwargs,
                priority=priority,
                next_run=next_run,
                interval=interval,
                max_retries=max_retries,
                timeout=timeout,
                resource_requirements=resource_requirements or {}
            )

            self._tasks[task_id] = task
            heapq.heappush(self._task_queue, task)
            self._metrics['tasks_scheduled'] += 1

            logger.info(f"Scheduled periodic task {task_id} with {interval}s interval")
            return True

        except Exception as e:
            logger.error(f"Failed to schedule task {task_id}: {e}")
            return False

    async def schedule_cron(self, task_id: str, coro: Callable[..., Awaitable[Any]],
                           cron_expression: str, priority: TaskPriority = TaskPriority.NORMAL,
                           max_retries: int = 3, timeout: Optional[float] = None,
                           resource_requirements: Dict[str, Any] = None,
                           **kwargs) -> bool:
        """
        Schedule a task using cron expression.

        Args:
            task_id: Unique identifier for the task
            coro: Coroutine function to execute
            cron_expression: Cron expression (e.g., "0 */6 * * *")
            priority: Task priority
            max_retries: Maximum retry attempts
            timeout: Task timeout in seconds
            resource_requirements: Resource requirements dict
            **kwargs: Arguments to pass to the coroutine

        Returns:
            True if scheduled successfully
        """
        try:
            if task_id in self._tasks:
                logger.warning(f"Task {task_id} already scheduled, updating...")
                await self.cancel_task(task_id)

            # Calculate next run time from cron expression
            cron = croniter.croniter(cron_expression, datetime.now())
            next_run = cron.get_next()

            task = ScheduledTask(
                task_id=task_id,
                coro=coro,
                kwargs=kwargs,
                priority=priority,
                next_run=next_run,
                cron_expression=cron_expression,
                max_retries=max_retries,
                timeout=timeout,
                resource_requirements=resource_requirements or {}
            )

            self._tasks[task_id] = task
            heapq.heappush(self._task_queue, task)
            self._metrics['tasks_scheduled'] += 1

            logger.info(f"Scheduled cron task {task_id} with expression '{cron_expression}'")
            return True

        except Exception as e:
            logger.error(f"Failed to schedule cron task {task_id}: {e}")
            return False

    async def schedule_once(self, task_id: str, coro: Callable[..., Awaitable[Any]],
                           delay: float = 0, priority: TaskPriority = TaskPriority.NORMAL,
                           max_retries: int = 3, timeout: Optional[float] = None,
                           dependencies: List[str] = None,
                           resource_requirements: Dict[str, Any] = None,
                           **kwargs) -> bool:
        """
        Schedule a one-time task.

        Args:
            task_id: Unique identifier for the task
            coro: Coroutine function to execute
            delay: Delay before execution in seconds
            priority: Task priority
            max_retries: Maximum retry attempts
            timeout: Task timeout in seconds
            dependencies: List of task IDs this task depends on
            resource_requirements: Resource requirements dict
            **kwargs: Arguments to pass to the coroutine

        Returns:
            True if scheduled successfully
        """
        try:
            if task_id in self._tasks:
                logger.warning(f"Task {task_id} already scheduled, updating...")
                await self.cancel_task(task_id)

            next_run = time.time() + delay

            task = ScheduledTask(
                task_id=task_id,
                coro=coro,
                kwargs=kwargs,
                priority=priority,
                next_run=next_run,
                max_retries=max_retries,
                timeout=timeout,
                dependencies=dependencies or [],
                resource_requirements=resource_requirements or {}
            )

            self._tasks[task_id] = task

            # Handle dependencies
            if dependencies:
                for dep_id in dependencies:
                    self._dependency_graph[dep_id].append(task_id)
                    self._dependents[task_id].append(dep_id)

                # Only add to queue if dependencies are satisfied
                if await self._are_dependencies_satisfied(task_id):
                    heapq.heappush(self._task_queue, task)
            else:
                heapq.heappush(self._task_queue, task)

            self._metrics['tasks_scheduled'] += 1

            logger.info(f"Scheduled one-time task {task_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to schedule task {task_id}: {e}")
            return False

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a scheduled task."""
        try:
            # Cancel if currently running
            if task_id in self._running_tasks:
                self._running_tasks[task_id].cancel()
                del self._running_tasks[task_id]

            # Remove from scheduled tasks
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task.status = TaskStatus.CANCELLED
                del self._tasks[task_id]

                # Remove from queue (expensive but necessary)
                self._task_queue = [t for t in self._task_queue if t.task_id != task_id]
                heapq.heapify(self._task_queue)

            logger.info(f"Cancelled task {task_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {e}")
            return False

    async def _scheduler_loop(self):
        """Main scheduler loop."""
        logger.info("Scheduler loop started")

        while self._running:
            try:
                current_time = time.time()

                # Process ready tasks
                while (self._task_queue and
                       self._task_queue[0].next_run <= current_time and
                       len(self._running_tasks) < self.max_concurrent_tasks):

                    task = heapq.heappop(self._task_queue)

                    # Check if task still exists (could have been cancelled)
                    if task.task_id not in self._tasks:
                        continue

                    # Check dependencies
                    if not await self._are_dependencies_satisfied(task.task_id):
                        # Re-queue for later
                        task.next_run = current_time + 60  # Check again in 1 minute
                        heapq.heappush(self._task_queue, task)
                        continue

                    # Check resource availability
                    if not await self._resource_manager.can_allocate(task.resource_requirements):
                        # Re-queue for later
                        task.next_run = current_time + 30  # Check again in 30 seconds
                        heapq.heappush(self._task_queue, task)
                        continue

                    # Execute task
                    await self._execute_task(task)

                # Clean up completed tasks
                await self._cleanup_completed_tasks()

                # Sleep until next task or health check
                sleep_time = min(
                    self.health_check_interval,
                    (self._task_queue[0].next_run - current_time) if self._task_queue else self.health_check_interval
                )
                sleep_time = max(0.1, sleep_time)  # Minimum 100ms sleep

                await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(5)  # Back off on error

    async def _execute_task(self, task: ScheduledTask):
        """Execute a scheduled task."""
        task.status = TaskStatus.RUNNING
        task.attempts += 1

        # Allocate resources
        await self._resource_manager.allocate(task.resource_requirements)

        # Create and start task
        execution_task = asyncio.create_task(self._run_task_with_timeout(task))
        self._running_tasks[task.task_id] = execution_task

        logger.debug(f"Started execution of task {task.task_id}")

    async def _run_task_with_timeout(self, task: ScheduledTask):
        """Run a task with timeout and error handling."""
        start_time = time.time()

        try:
            async with self._semaphore:
                if task.timeout:
                    await asyncio.wait_for(
                        task.coro(**task.kwargs),
                        timeout=task.timeout
                    )
                else:
                    await task.coro(**task.kwargs)

            # Task completed successfully
            runtime = time.time() - start_time
            task.status = TaskStatus.COMPLETED
            task.last_run = start_time
            task.total_runtime += runtime
            task.avg_runtime = task.total_runtime / task.attempts
            task.last_error = None

            self._metrics['tasks_completed'] += 1
            self._metrics['total_runtime'] += runtime
            self._metrics['avg_task_runtime'] = (
                self._metrics['total_runtime'] / max(1, self._metrics['tasks_completed'])
            )

            logger.debug(f"Task {task.task_id} completed in {runtime:.2f}s")

            # Schedule next run if periodic
            if task.interval or task.cron_expression:
                await self._schedule_next_run(task)
            else:
                # Move to completed tasks
                self._completed_tasks[task.task_id] = task
                if task.task_id in self._tasks:
                    del self._tasks[task.task_id]

            # Trigger dependent tasks
            await self._trigger_dependent_tasks(task.task_id)

        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED
            logger.info(f"Task {task.task_id} was cancelled")

        except asyncio.TimeoutError:
            task.status = TaskStatus.FAILED
            task.last_error = "Task timed out"
            logger.warning(f"Task {task.task_id} timed out")
            await self._handle_task_failure(task)

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.last_error = str(e)
            logger.error(f"Task {task.task_id} failed: {e}")
            await self._handle_task_failure(task)

        finally:
            # Release resources
            await self._resource_manager.release(task.resource_requirements)

            # Remove from running tasks
            if task.task_id in self._running_tasks:
                del self._running_tasks[task.task_id]

    async def _handle_task_failure(self, task: ScheduledTask):
        """Handle task failure and retries."""
        self._metrics['tasks_failed'] += 1

        if task.attempts < task.max_retries:
            # Schedule retry
            task.status = TaskStatus.RETRYING
            task.next_run = time.time() + (task.retry_delay * task.attempts)  # Exponential backoff
            heapq.heappush(self._task_queue, task)
            self._metrics['tasks_retried'] += 1

            logger.info(f"Scheduling retry {task.attempts}/{task.max_retries} for task {task.task_id}")
        else:
            # Max retries exceeded
            logger.error(f"Task {task.task_id} failed permanently after {task.attempts} attempts")

            if not (task.interval or task.cron_expression):
                # Move to completed tasks for one-time tasks
                self._completed_tasks[task.task_id] = task
                if task.task_id in self._tasks:
                    del self._tasks[task.task_id]

    async def _schedule_next_run(self, task: ScheduledTask):
        """Schedule the next run for a periodic task."""
        current_time = time.time()

        if task.interval:
            # Interval-based scheduling
            task.next_run = current_time + task.interval
        elif task.cron_expression:
            # Cron-based scheduling
            cron = croniter.croniter(task.cron_expression, datetime.fromtimestamp(current_time))
            task.next_run = cron.get_next()

        task.status = TaskStatus.PENDING
        heapq.heappush(self._task_queue, task)

    async def _are_dependencies_satisfied(self, task_id: str) -> bool:
        """Check if all dependencies for a task are satisfied."""
        if task_id not in self._tasks:
            return False

        dependencies = self._tasks[task_id].dependencies
        if not dependencies:
            return True

        for dep_id in dependencies:
            if dep_id in self._completed_tasks:
                dep_task = self._completed_tasks[dep_id]
                if dep_task.status != TaskStatus.COMPLETED:
                    return False
            elif dep_id in self._tasks:
                # Dependency is still scheduled/running
                return False
            else:
                # Dependency not found
                logger.warning(f"Dependency {dep_id} not found for task {task_id}")
                return False

        return True

    async def _trigger_dependent_tasks(self, completed_task_id: str):
        """Trigger tasks that depend on the completed task."""
        if completed_task_id not in self._dependency_graph:
            return

        dependent_task_ids = self._dependency_graph[completed_task_id]
        current_time = time.time()

        for task_id in dependent_task_ids:
            if task_id in self._tasks and await self._are_dependencies_satisfied(task_id):
                task = self._tasks[task_id]
                if task.status == TaskStatus.PENDING and task not in self._task_queue:
                    task.next_run = current_time  # Execute immediately
                    heapq.heappush(self._task_queue, task)
                    logger.info(f"Triggered dependent task {task_id}")

    async def _cleanup_completed_tasks(self):
        """Clean up old completed tasks to prevent memory leaks."""
        if len(self._completed_tasks) > 1000:  # Keep only last 1000 completed tasks
            # Sort by completion time and keep most recent
            sorted_tasks = sorted(
                self._completed_tasks.items(),
                key=lambda x: x[1].last_run or 0,
                reverse=True
            )

            # Keep only the 1000 most recent
            keep_tasks = dict(sorted_tasks[:1000])
            self._completed_tasks.clear()
            self._completed_tasks.update(keep_tasks)

    # Public API methods

    async def get_status(self) -> Dict[str, Any]:
        """Get scheduler status and metrics."""
        return {
            "running": self._running,
            "metrics": self._metrics.copy(),
            "scheduled_tasks": len(self._tasks),
            "running_tasks": len(self._running_tasks),
            "completed_tasks": len(self._completed_tasks),
            "queue_length": len(self._task_queue),
            "resource_usage": {
                "cpu_tasks": self._resource_manager.current_cpu_tasks,
                "memory_mb": self._resource_manager.current_memory_mb,
                "io_tasks": self._resource_manager.current_io_tasks
            }
        }

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task."""
        task = None

        if task_id in self._tasks:
            task = self._tasks[task_id]
        elif task_id in self._completed_tasks:
            task = self._completed_tasks[task_id]

        if not task:
            return None

        return {
            "task_id": task.task_id,
            "status": task.status.value,
            "priority": task.priority.name,
            "attempts": task.attempts,
            "last_run": task.last_run,
            "next_run": task.next_run if task.status == TaskStatus.PENDING else None,
            "avg_runtime": task.avg_runtime,
            "last_error": task.last_error,
            "is_running": task_id in self._running_tasks
        }

    async def list_tasks(self, status_filter: Optional[TaskStatus] = None) -> List[Dict[str, Any]]:
        """List all tasks with optional status filtering."""
        tasks = []

        # Add scheduled tasks
        for task in self._tasks.values():
            if status_filter is None or task.status == status_filter:
                tasks.append(await self.get_task_status(task.task_id))

        # Add completed tasks
        for task in self._completed_tasks.values():
            if status_filter is None or task.status == status_filter:
                tasks.append(await self.get_task_status(task.task_id))

        return sorted(tasks, key=lambda x: x.get('next_run', 0) or 0)