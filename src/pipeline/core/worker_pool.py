"""
Worker Pool - Distributed task execution with load balancing

Features:
- Dynamic worker scaling based on load
- Task queuing with priority handling
- Load balancing across workers
- Health monitoring and auto-recovery
- Memory and CPU usage tracking
- Graceful degradation
- Cross-platform process management
"""

import asyncio
import logging
import multiprocessing
import psutil
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Awaitable, Union
import pickle
import queue
import threading

logger = logging.getLogger(__name__)


class WorkerType(Enum):
    """Types of workers available."""
    ASYNC = "async"          # Asyncio tasks
    THREAD = "thread"        # Thread pool
    PROCESS = "process"      # Process pool


class TaskType(Enum):
    """Task execution types."""
    CPU_BOUND = "cpu_bound"
    IO_BOUND = "io_bound"
    MIXED = "mixed"


@dataclass
class WorkerTask:
    """Represents a task submitted to the worker pool."""
    task_id: str
    coro: Callable[..., Awaitable[Any]]
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    task_type: TaskType = TaskType.IO_BOUND
    priority: int = 5  # 1 = highest, 10 = lowest
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    result: Any = None

    def __post_init__(self):
        if not self.task_id:
            self.task_id = str(uuid.uuid4())

    def __lt__(self, other):
        """Priority queue comparison."""
        return self.priority < other.priority


class WorkerStats:
    """Statistics for worker performance monitoring."""

    def __init__(self):
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.total_execution_time = 0.0
        self.avg_execution_time = 0.0
        self.memory_usage_mb = 0.0
        self.cpu_usage_percent = 0.0
        self.last_activity = time.time()

    def record_task_completion(self, execution_time: float, memory_mb: float = 0):
        """Record successful task completion."""
        self.tasks_completed += 1
        self.total_execution_time += execution_time
        self.avg_execution_time = self.total_execution_time / max(1, self.tasks_completed)
        self.memory_usage_mb = max(self.memory_usage_mb, memory_mb)
        self.last_activity = time.time()

    def record_task_failure(self):
        """Record task failure."""
        self.tasks_failed += 1
        self.last_activity = time.time()

    def get_success_rate(self) -> float:
        """Get task success rate."""
        total = self.tasks_completed + self.tasks_failed
        return self.tasks_completed / max(1, total)

    def get_throughput(self, window_seconds: float = 300) -> float:
        """Get tasks per second in the given window."""
        if time.time() - self.last_activity > window_seconds:
            return 0.0
        return self.tasks_completed / max(1, window_seconds)


class AsyncWorker:
    """Individual async worker for handling coroutines."""

    def __init__(self, worker_id: str, stats: WorkerStats):
        self.worker_id = worker_id
        self.stats = stats
        self.current_task: Optional[WorkerTask] = None
        self.is_busy = False
        self._task_handle: Optional[asyncio.Task] = None

    async def execute_task(self, task: WorkerTask) -> Any:
        """Execute a task and return result."""
        self.current_task = task
        self.is_busy = True
        task.started_at = time.time()

        try:
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024

            if task.timeout:
                result = await asyncio.wait_for(
                    task.coro(*task.args, **task.kwargs),
                    timeout=task.timeout
                )
            else:
                result = await task.coro(*task.args, **task.kwargs)

            task.completed_at = time.time()
            task.result = result

            # Record performance metrics
            execution_time = task.completed_at - task.started_at
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_used = end_memory - start_memory

            self.stats.record_task_completion(execution_time, memory_used)

            logger.debug(f"Worker {self.worker_id} completed task {task.task_id} in {execution_time:.2f}s")
            return result

        except Exception as e:
            task.error = str(e)
            self.stats.record_task_failure()
            logger.error(f"Worker {self.worker_id} failed task {task.task_id}: {e}")
            raise

        finally:
            self.current_task = None
            self.is_busy = False

    def get_status(self) -> Dict[str, Any]:
        """Get current worker status."""
        return {
            "worker_id": self.worker_id,
            "is_busy": self.is_busy,
            "current_task": self.current_task.task_id if self.current_task else None,
            "stats": {
                "tasks_completed": self.stats.tasks_completed,
                "tasks_failed": self.stats.tasks_failed,
                "avg_execution_time": self.stats.avg_execution_time,
                "success_rate": self.stats.get_success_rate(),
                "throughput": self.stats.get_throughput()
            }
        }


class WorkerPool:
    """
    High-performance worker pool with adaptive scaling and load balancing.

    Features:
    - Multiple execution strategies (async, thread, process)
    - Dynamic scaling based on queue length and system load
    - Priority-based task scheduling
    - Health monitoring and recovery
    - Resource usage tracking
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_workers = config.get('min_workers', 2)
        self.max_workers = config.get('max_workers', multiprocessing.cpu_count() * 2)
        self.scale_threshold = config.get('scale_threshold', 5)
        self.health_check_interval = config.get('health_check_interval', 30)

        # Worker management
        self.async_workers: Dict[str, AsyncWorker] = {}
        self.thread_executor: Optional[ThreadPoolExecutor] = None
        self.process_executor: Optional[ProcessPoolExecutor] = None

        # Task queues
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.pending_tasks: Dict[str, WorkerTask] = {}
        self.completed_tasks: Dict[str, WorkerTask] = {}

        # State management
        self.running = False
        self.dispatcher_task: Optional[asyncio.Task] = None
        self.health_monitor_task: Optional[asyncio.Task] = None
        self.scaler_task: Optional[asyncio.Task] = None

        # Metrics
        self.pool_stats = WorkerStats()
        self.system_monitor = psutil.Process()

        # Load balancing
        self._round_robin_index = 0

    async def start(self):
        """Start the worker pool."""
        if self.running:
            logger.warning("Worker pool already running")
            return

        logger.info("Starting worker pool...")

        # Initialize executors
        thread_workers = self.config.get('thread_workers', 4)
        process_workers = self.config.get('process_workers', multiprocessing.cpu_count())

        self.thread_executor = ThreadPoolExecutor(
            max_workers=thread_workers,
            thread_name_prefix="quicknav-worker"
        )

        self.process_executor = ProcessPoolExecutor(
            max_workers=process_workers
        )

        # Create initial async workers
        for i in range(self.min_workers):
            await self._create_async_worker(f"async-worker-{i}")

        # Start background tasks
        self.dispatcher_task = asyncio.create_task(self._dispatcher_loop())
        self.health_monitor_task = asyncio.create_task(self._health_monitor_loop())
        self.scaler_task = asyncio.create_task(self._scaler_loop())

        self.running = True
        logger.info(f"Worker pool started with {len(self.async_workers)} async workers")

    async def shutdown(self, timeout: float = 30.0):
        """Shutdown the worker pool gracefully."""
        if not self.running:
            return

        logger.info("Shutting down worker pool...")
        self.running = False

        # Cancel background tasks
        for task in [self.dispatcher_task, self.health_monitor_task, self.scaler_task]:
            if task:
                task.cancel()

        # Wait for running tasks to complete
        busy_workers = [w for w in self.async_workers.values() if w.is_busy]
        if busy_workers:
            logger.info(f"Waiting for {len(busy_workers)} busy workers to complete...")
            try:
                await asyncio.wait_for(
                    asyncio.gather(*[w._task_handle for w in busy_workers if w._task_handle], return_exceptions=True),
                    timeout=timeout * 0.7
                )
            except asyncio.TimeoutError:
                logger.warning("Some async workers did not complete within timeout")

        # Shutdown executors
        if self.thread_executor:
            self.thread_executor.shutdown(wait=True, timeout=timeout * 0.15)

        if self.process_executor:
            self.process_executor.shutdown(wait=True, timeout=timeout * 0.15)

        logger.info("Worker pool shutdown complete")

    async def submit(self, task_id: str, coro: Callable[..., Awaitable[Any]],
                    args: tuple = (), kwargs: Dict[str, Any] = None,
                    task_type: TaskType = TaskType.IO_BOUND,
                    priority: int = 5, timeout: Optional[float] = None,
                    max_retries: int = 3) -> str:
        """
        Submit a task for execution.

        Args:
            task_id: Unique task identifier
            coro: Coroutine to execute
            args: Positional arguments
            kwargs: Keyword arguments
            task_type: Type of task (CPU/IO bound)
            priority: Task priority (1=highest, 10=lowest)
            timeout: Task timeout in seconds
            max_retries: Maximum retry attempts

        Returns:
            Task ID for tracking
        """
        if not self.running:
            raise RuntimeError("Worker pool is not running")

        task = WorkerTask(
            task_id=task_id,
            coro=coro,
            args=args,
            kwargs=kwargs or {},
            task_type=task_type,
            priority=priority,
            timeout=timeout,
            max_retries=max_retries
        )

        self.pending_tasks[task_id] = task
        await self.task_queue.put((priority, time.time(), task))

        logger.debug(f"Submitted task {task_id} with priority {priority}")
        return task_id

    async def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """
        Wait for and return task result.

        Args:
            task_id: Task identifier
            timeout: Maximum time to wait

        Returns:
            Task result

        Raises:
            asyncio.TimeoutError: If timeout exceeded
            RuntimeError: If task failed
        """
        start_time = time.time()

        while task_id not in self.completed_tasks:
            if timeout and (time.time() - start_time) > timeout:
                raise asyncio.TimeoutError(f"Task {task_id} did not complete within {timeout}s")

            await asyncio.sleep(0.1)

        task = self.completed_tasks[task_id]

        if task.error:
            raise RuntimeError(f"Task {task_id} failed: {task.error}")

        return task.result

    async def _dispatcher_loop(self):
        """Main task dispatcher loop."""
        logger.info("Task dispatcher started")

        while self.running:
            try:
                # Get next task with timeout
                try:
                    priority, queued_time, task = await asyncio.wait_for(
                        self.task_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Find available worker
                worker = await self._get_available_worker(task.task_type)

                if worker:
                    # Execute task
                    worker._task_handle = asyncio.create_task(
                        self._execute_task_with_retry(worker, task)
                    )
                else:
                    # No workers available, re-queue task
                    await self.task_queue.put((priority, queued_time, task))
                    await asyncio.sleep(0.1)  # Brief delay to prevent tight loop

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Dispatcher loop error: {e}")
                await asyncio.sleep(1)

    async def _execute_task_with_retry(self, worker: AsyncWorker, task: WorkerTask):
        """Execute task with retry logic."""
        while task.retry_count <= task.max_retries:
            try:
                result = await worker.execute_task(task)
                task.result = result

                # Move to completed tasks
                if task.task_id in self.pending_tasks:
                    del self.pending_tasks[task.task_id]
                self.completed_tasks[task.task_id] = task

                self.pool_stats.record_task_completion(
                    task.completed_at - task.started_at
                )

                logger.debug(f"Task {task.task_id} completed successfully")
                return result

            except Exception as e:
                task.retry_count += 1
                task.error = str(e)

                if task.retry_count <= task.max_retries:
                    logger.warning(f"Task {task.task_id} failed (attempt {task.retry_count}), retrying...")
                    await asyncio.sleep(min(2 ** task.retry_count, 60))  # Exponential backoff
                else:
                    logger.error(f"Task {task.task_id} failed permanently after {task.retry_count} attempts")

                    # Move to completed tasks (with error)
                    if task.task_id in self.pending_tasks:
                        del self.pending_tasks[task.task_id]
                    self.completed_tasks[task.task_id] = task

                    self.pool_stats.record_task_failure()
                    break

    async def _get_available_worker(self, task_type: TaskType) -> Optional[AsyncWorker]:
        """Get an available worker for the given task type."""
        # For now, only async workers are implemented
        # TODO: Add thread/process worker selection based on task_type

        available_workers = [w for w in self.async_workers.values() if not w.is_busy]

        if not available_workers:
            return None

        # Round-robin load balancing
        worker = available_workers[self._round_robin_index % len(available_workers)]
        self._round_robin_index += 1

        return worker

    async def _create_async_worker(self, worker_id: str) -> AsyncWorker:
        """Create a new async worker."""
        stats = WorkerStats()
        worker = AsyncWorker(worker_id, stats)
        self.async_workers[worker_id] = worker

        logger.debug(f"Created async worker {worker_id}")
        return worker

    async def _remove_async_worker(self, worker_id: str):
        """Remove an async worker."""
        if worker_id in self.async_workers:
            worker = self.async_workers[worker_id]

            # Wait for current task to complete if busy
            if worker.is_busy and worker._task_handle:
                try:
                    await asyncio.wait_for(worker._task_handle, timeout=30)
                except asyncio.TimeoutError:
                    logger.warning(f"Worker {worker_id} did not complete current task within timeout")

            del self.async_workers[worker_id]
            logger.debug(f"Removed async worker {worker_id}")

    async def _health_monitor_loop(self):
        """Monitor worker health and restart failed workers."""
        logger.info("Health monitor started")

        while self.running:
            try:
                # Check worker health
                unhealthy_workers = []

                for worker_id, worker in self.async_workers.items():
                    # Check if worker is stuck
                    if (worker.is_busy and worker.current_task and
                        time.time() - worker.current_task.started_at > 300):  # 5 minutes
                        logger.warning(f"Worker {worker_id} appears to be stuck")
                        unhealthy_workers.append(worker_id)

                    # Check worker stats
                    success_rate = worker.stats.get_success_rate()
                    if success_rate < 0.8 and worker.stats.tasks_completed > 10:
                        logger.warning(f"Worker {worker_id} has low success rate: {success_rate:.2%}")

                # Restart unhealthy workers
                for worker_id in unhealthy_workers:
                    await self._restart_worker(worker_id)

                # Update system metrics
                self.pool_stats.memory_usage_mb = self.system_monitor.memory_info().rss / 1024 / 1024
                self.pool_stats.cpu_usage_percent = self.system_monitor.cpu_percent()

                await asyncio.sleep(self.health_check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(30)

    async def _restart_worker(self, worker_id: str):
        """Restart a worker."""
        try:
            logger.info(f"Restarting worker {worker_id}")

            # Remove old worker
            await self._remove_async_worker(worker_id)

            # Create new worker
            await self._create_async_worker(worker_id)

        except Exception as e:
            logger.error(f"Failed to restart worker {worker_id}: {e}")

    async def _scaler_loop(self):
        """Automatically scale workers based on load."""
        logger.info("Auto-scaler started")

        while self.running:
            try:
                queue_size = self.task_queue.qsize()
                worker_count = len(self.async_workers)
                busy_workers = len([w for w in self.async_workers.values() if w.is_busy])

                # Scale up if queue is backing up
                if (queue_size > self.scale_threshold and
                    worker_count < self.max_workers and
                    busy_workers / max(1, worker_count) > 0.8):

                    new_worker_id = f"async-worker-scaled-{int(time.time())}"
                    await self._create_async_worker(new_worker_id)
                    logger.info(f"Scaled up to {len(self.async_workers)} workers")

                # Scale down if workers are idle
                elif (queue_size == 0 and
                      worker_count > self.min_workers and
                      busy_workers / max(1, worker_count) < 0.2):

                    # Find an idle worker to remove
                    idle_workers = [wid for wid, w in self.async_workers.items() if not w.is_busy]
                    if idle_workers:
                        worker_to_remove = idle_workers[0]
                        await self._remove_async_worker(worker_to_remove)
                        logger.info(f"Scaled down to {len(self.async_workers)} workers")

                await asyncio.sleep(30)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-scaler error: {e}")
                await asyncio.sleep(60)

    # Public API methods

    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive worker pool status."""
        worker_statuses = [w.get_status() for w in self.async_workers.values()]

        return {
            "running": self.running,
            "workers": {
                "async_workers": len(self.async_workers),
                "min_workers": self.min_workers,
                "max_workers": self.max_workers,
                "busy_workers": len([w for w in self.async_workers.values() if w.is_busy])
            },
            "tasks": {
                "queued": self.task_queue.qsize(),
                "pending": len(self.pending_tasks),
                "completed": len(self.completed_tasks)
            },
            "pool_stats": {
                "tasks_completed": self.pool_stats.tasks_completed,
                "tasks_failed": self.pool_stats.tasks_failed,
                "success_rate": self.pool_stats.get_success_rate(),
                "avg_execution_time": self.pool_stats.avg_execution_time,
                "throughput": self.pool_stats.get_throughput()
            },
            "system": {
                "memory_mb": self.pool_stats.memory_usage_mb,
                "cpu_percent": self.pool_stats.cpu_usage_percent
            },
            "worker_details": worker_statuses
        }

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task."""
        if task_id in self.pending_tasks:
            task = self.pending_tasks[task_id]
            status = "pending"
        elif task_id in self.completed_tasks:
            task = self.completed_tasks[task_id]
            status = "completed" if not task.error else "failed"
        else:
            return None

        return {
            "task_id": task.task_id,
            "status": status,
            "task_type": task.task_type.value,
            "priority": task.priority,
            "retry_count": task.retry_count,
            "max_retries": task.max_retries,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "execution_time": (task.completed_at - task.started_at) if task.completed_at and task.started_at else None,
            "error": task.error
        }

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        if task_id in self.pending_tasks:
            task = self.pending_tasks[task_id]
            task.error = "Task cancelled"

            del self.pending_tasks[task_id]
            self.completed_tasks[task_id] = task

            logger.info(f"Cancelled task {task_id}")
            return True

        return False

    async def clear_completed_tasks(self, older_than_hours: int = 24):
        """Clear old completed tasks to prevent memory leaks."""
        cutoff_time = time.time() - (older_than_hours * 3600)
        tasks_to_remove = []

        for task_id, task in self.completed_tasks.items():
            if task.completed_at and task.completed_at < cutoff_time:
                tasks_to_remove.append(task_id)

        for task_id in tasks_to_remove:
            del self.completed_tasks[task_id]

        if tasks_to_remove:
            logger.info(f"Cleared {len(tasks_to_remove)} old completed tasks")

        return len(tasks_to_remove)