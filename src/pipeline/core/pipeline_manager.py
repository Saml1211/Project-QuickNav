"""
Pipeline Manager - Central orchestration for all data pipelines

Manages the lifecycle of all pipeline components including:
- Initialization and configuration
- Startup and shutdown coordination
- Health monitoring and recovery
- Resource management and scaling
- Integration with GUI and MCP server
"""

import asyncio
import logging
import signal
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import json

from .scheduler import AsyncScheduler
from .worker_pool import WorkerPool
from ..config.pipeline_config import PipelineConfig
from ..monitoring.metrics_collector import MetricsCollector
from ..monitoring.alerting import AlertManager
from ...database.database_manager import get_database_manager

logger = logging.getLogger(__name__)


class PipelineManager:
    """
    Central manager for all data pipeline operations.

    Provides unified interface for:
    - Pipeline lifecycle management
    - Component coordination
    - Health monitoring
    - Performance metrics
    - Error handling and recovery
    """

    def __init__(self, config_path: Optional[str] = None, db_manager=None):
        """
        Initialize pipeline manager.

        Args:
            config_path: Path to configuration file
            db_manager: Optional database manager instance
        """
        self.config = PipelineConfig(config_path)
        self.db_manager = db_manager or get_database_manager()

        # Core components
        self.scheduler = AsyncScheduler(self.config.scheduler)
        self.worker_pool = WorkerPool(self.config.workers)
        self.metrics = MetricsCollector(self.config.monitoring)
        self.alerts = AlertManager(self.config.alerting)

        # Pipeline components (initialized on demand)
        self._pipelines: Dict[str, Any] = {}
        self._health_checks: Dict[str, Callable] = {}
        self._startup_hooks: List[Callable] = []
        self._shutdown_hooks: List[Callable] = []

        # State management
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._health_check_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None

        # GUI integration
        self._gui_callbacks: Dict[str, Callable] = {}

        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            asyncio.create_task(self.shutdown())

        try:
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
        except ValueError:
            # Running in thread, signals won't work
            pass

    async def start(self, components: Optional[List[str]] = None) -> bool:
        """
        Start the pipeline manager and specified components.

        Args:
            components: List of component names to start, or None for all

        Returns:
            True if startup successful
        """
        if self._running:
            logger.warning("Pipeline manager already running")
            return True

        logger.info("Starting Pipeline Manager...")

        try:
            # Start core components
            await self.scheduler.start()
            await self.worker_pool.start()
            await self.metrics.start()
            await self.alerts.start()

            # Initialize and start pipeline components
            if components is None:
                components = self.config.enabled_components

            for component_name in components:
                success = await self._start_component(component_name)
                if not success and self.config.strict_startup:
                    logger.error(f"Failed to start required component: {component_name}")
                    await self.shutdown()
                    return False

            # Run startup hooks
            for hook in self._startup_hooks:
                try:
                    if asyncio.iscoroutinefunction(hook):
                        await hook()
                    else:
                        hook()
                except Exception as e:
                    logger.error(f"Startup hook failed: {e}")

            # Start background tasks
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            self._metrics_task = asyncio.create_task(self._metrics_collection_loop())

            self._running = True
            logger.info("Pipeline Manager started successfully")

            # Record startup in database
            await self._record_lifecycle_event("startup", {"components": components})

            return True

        except Exception as e:
            logger.error(f"Failed to start Pipeline Manager: {e}")
            await self.shutdown()
            return False

    async def _start_component(self, component_name: str) -> bool:
        """Start a specific pipeline component."""
        try:
            component_config = self.config.get_component_config(component_name)
            if not component_config.get('enabled', True):
                logger.info(f"Component {component_name} is disabled, skipping")
                return True

            logger.info(f"Starting component: {component_name}")

            # Import and initialize component based on type
            if component_name == "document_monitor":
                from ..ingestion.document_monitor import DocumentMonitor
                component = DocumentMonitor(component_config, self.db_manager)

            elif component_name == "realtime_processor":
                from ..ingestion.realtime_processor import RealtimeProcessor
                component = RealtimeProcessor(component_config, self.db_manager)

            elif component_name == "analytics_processor":
                from ..batch.analytics_processor import AnalyticsProcessor
                component = AnalyticsProcessor(component_config, self.db_manager)

            elif component_name == "ml_feature_processor":
                from ..batch.ml_feature_processor import MLFeatureProcessor
                component = MLFeatureProcessor(component_config, self.db_manager)

            elif component_name == "activity_streamer":
                from ..streaming.activity_streamer import ActivityStreamer
                component = ActivityStreamer(component_config, self.db_manager)

            elif component_name == "data_validator":
                from ..quality.data_validator import DataValidator
                component = DataValidator(component_config, self.db_manager)

            elif component_name == "anomaly_detector":
                from ..quality.anomaly_detector import AnomalyDetector
                component = AnomalyDetector(component_config, self.db_manager)

            else:
                logger.error(f"Unknown component: {component_name}")
                return False

            # Start the component
            await component.start()

            # Register component
            self._pipelines[component_name] = component
            if hasattr(component, 'health_check'):
                self._health_checks[component_name] = component.health_check

            # Schedule periodic tasks if configured
            if component_config.get('schedule'):
                schedule_config = component_config['schedule']
                if hasattr(component, 'run_batch'):
                    await self.scheduler.schedule_periodic(
                        task_id=f"{component_name}_batch",
                        coro=component.run_batch,
                        interval=schedule_config.get('interval', 3600),
                        start_delay=schedule_config.get('start_delay', 0)
                    )

            logger.info(f"Component {component_name} started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start component {component_name}: {e}")
            return False

    async def shutdown(self, timeout: float = 30.0):
        """
        Gracefully shutdown all pipeline components.

        Args:
            timeout: Maximum time to wait for shutdown
        """
        if not self._running:
            return

        logger.info("Shutting down Pipeline Manager...")
        self._running = False
        self._shutdown_event.set()

        try:
            # Run shutdown hooks
            for hook in self._shutdown_hooks:
                try:
                    if asyncio.iscoroutinefunction(hook):
                        await hook()
                    else:
                        hook()
                except Exception as e:
                    logger.error(f"Shutdown hook failed: {e}")

            # Stop background tasks
            if self._health_check_task:
                self._health_check_task.cancel()
            if self._metrics_task:
                self._metrics_task.cancel()

            # Stop pipeline components
            shutdown_tasks = []
            for name, component in self._pipelines.items():
                if hasattr(component, 'shutdown'):
                    shutdown_tasks.append(component.shutdown())

            if shutdown_tasks:
                await asyncio.wait_for(
                    asyncio.gather(*shutdown_tasks, return_exceptions=True),
                    timeout=timeout * 0.6
                )

            # Stop core components
            await asyncio.wait_for(
                asyncio.gather(
                    self.scheduler.shutdown(),
                    self.worker_pool.shutdown(),
                    self.metrics.shutdown(),
                    self.alerts.shutdown(),
                    return_exceptions=True
                ),
                timeout=timeout * 0.4
            )

            # Record shutdown in database
            await self._record_lifecycle_event("shutdown")

            logger.info("Pipeline Manager shutdown complete")

        except asyncio.TimeoutError:
            logger.warning(f"Shutdown timed out after {timeout}s")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    async def _health_check_loop(self):
        """Background task to monitor component health."""
        while self._running:
            try:
                health_status = {}

                for name, check_func in self._health_checks.items():
                    try:
                        if asyncio.iscoroutinefunction(check_func):
                            status = await check_func()
                        else:
                            status = check_func()
                        health_status[name] = status
                    except Exception as e:
                        health_status[name] = {"healthy": False, "error": str(e)}
                        logger.warning(f"Health check failed for {name}: {e}")

                # Check for unhealthy components
                unhealthy = [name for name, status in health_status.items()
                           if not status.get("healthy", True)]

                if unhealthy:
                    await self.alerts.send_alert(
                        "component_unhealthy",
                        f"Unhealthy components: {', '.join(unhealthy)}",
                        {"components": unhealthy, "status": health_status}
                    )

                # Update metrics
                await self.metrics.record_health_status(health_status)

                # Notify GUI if callback registered
                if "health_update" in self._gui_callbacks:
                    try:
                        self._gui_callbacks["health_update"](health_status)
                    except Exception as e:
                        logger.debug(f"GUI health callback failed: {e}")

                await asyncio.sleep(self.config.health_check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(60)  # Back off on error

    async def _metrics_collection_loop(self):
        """Background task to collect and aggregate metrics."""
        while self._running:
            try:
                # Collect metrics from all components
                all_metrics = {}

                for name, component in self._pipelines.items():
                    if hasattr(component, 'get_metrics'):
                        try:
                            metrics = await component.get_metrics()
                            all_metrics[name] = metrics
                        except Exception as e:
                            logger.warning(f"Failed to get metrics from {name}: {e}")

                # Store metrics in database
                await self.metrics.store_metrics(all_metrics)

                # Check for performance alerts
                await self._check_performance_alerts(all_metrics)

                # Notify GUI if callback registered
                if "metrics_update" in self._gui_callbacks:
                    try:
                        self._gui_callbacks["metrics_update"](all_metrics)
                    except Exception as e:
                        logger.debug(f"GUI metrics callback failed: {e}")

                await asyncio.sleep(self.config.metrics_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection loop error: {e}")
                await asyncio.sleep(300)  # Back off on error

    async def _check_performance_alerts(self, metrics: Dict[str, Any]):
        """Check metrics for performance issues and send alerts."""
        try:
            # Check memory usage
            for component, component_metrics in metrics.items():
                if "memory_mb" in component_metrics:
                    memory_mb = component_metrics["memory_mb"]
                    if memory_mb > self.config.memory_alert_threshold:
                        await self.alerts.send_alert(
                            "high_memory_usage",
                            f"Component {component} using {memory_mb}MB",
                            {"component": component, "memory_mb": memory_mb}
                        )

                # Check processing times
                if "avg_processing_time_ms" in component_metrics:
                    processing_time = component_metrics["avg_processing_time_ms"]
                    if processing_time > self.config.processing_time_alert_threshold:
                        await self.alerts.send_alert(
                            "slow_processing",
                            f"Component {component} slow processing: {processing_time}ms",
                            {"component": component, "processing_time_ms": processing_time}
                        )

                # Check error rates
                if "error_rate" in component_metrics:
                    error_rate = component_metrics["error_rate"]
                    if error_rate > self.config.error_rate_alert_threshold:
                        await self.alerts.send_alert(
                            "high_error_rate",
                            f"Component {component} error rate: {error_rate:.2%}",
                            {"component": component, "error_rate": error_rate}
                        )

        except Exception as e:
            logger.error(f"Performance alert check failed: {e}")

    async def _record_lifecycle_event(self, event_type: str, details: Dict[str, Any] = None):
        """Record pipeline lifecycle events in database."""
        try:
            await self.db_manager.execute_write(
                """
                INSERT INTO pipeline_lifecycle_events
                (event_id, event_type, timestamp, details)
                VALUES (?, ?, ?, ?)
                """,
                [
                    f"pipeline_{event_type}_{int(time.time())}",
                    event_type,
                    datetime.now(),
                    json.dumps(details) if details else None
                ]
            )
        except Exception as e:
            logger.warning(f"Failed to record lifecycle event: {e}")

    # Public API methods

    async def get_status(self) -> Dict[str, Any]:
        """Get current status of all pipeline components."""
        status = {
            "running": self._running,
            "components": {},
            "scheduler": await self.scheduler.get_status() if hasattr(self.scheduler, 'get_status') else {},
            "worker_pool": await self.worker_pool.get_status() if hasattr(self.worker_pool, 'get_status') else {},
            "uptime_seconds": time.time() - getattr(self, '_start_time', time.time())
        }

        for name, component in self._pipelines.items():
            if hasattr(component, 'get_status'):
                try:
                    status["components"][name] = await component.get_status()
                except Exception as e:
                    status["components"][name] = {"error": str(e)}

        return status

    async def restart_component(self, component_name: str) -> bool:
        """Restart a specific component."""
        try:
            if component_name in self._pipelines:
                # Stop component
                component = self._pipelines[component_name]
                if hasattr(component, 'shutdown'):
                    await component.shutdown()

                # Remove from tracking
                del self._pipelines[component_name]
                if component_name in self._health_checks:
                    del self._health_checks[component_name]

            # Restart component
            success = await self._start_component(component_name)

            if success:
                logger.info(f"Component {component_name} restarted successfully")
                await self.alerts.send_alert(
                    "component_restarted",
                    f"Component {component_name} restarted",
                    {"component": component_name}
                )
            else:
                logger.error(f"Failed to restart component {component_name}")

            return success

        except Exception as e:
            logger.error(f"Error restarting component {component_name}: {e}")
            return False

    def register_gui_callback(self, event_type: str, callback: Callable):
        """Register GUI callback for pipeline events."""
        self._gui_callbacks[event_type] = callback

    def register_startup_hook(self, hook: Callable):
        """Register hook to run during startup."""
        self._startup_hooks.append(hook)

    def register_shutdown_hook(self, hook: Callable):
        """Register hook to run during shutdown."""
        self._shutdown_hooks.append(hook)

    async def trigger_batch_job(self, component_name: str, job_params: Dict[str, Any] = None) -> bool:
        """Manually trigger a batch job for a component."""
        try:
            if component_name not in self._pipelines:
                logger.error(f"Component {component_name} not found")
                return False

            component = self._pipelines[component_name]
            if not hasattr(component, 'run_batch'):
                logger.error(f"Component {component_name} does not support batch jobs")
                return False

            logger.info(f"Triggering batch job for {component_name}")

            # Submit to worker pool for async execution
            await self.worker_pool.submit(
                task_id=f"manual_batch_{component_name}_{int(time.time())}",
                coro=component.run_batch,
                kwargs=job_params or {}
            )

            return True

        except Exception as e:
            logger.error(f"Failed to trigger batch job for {component_name}: {e}")
            return False

    @asynccontextmanager
    async def managed_lifecycle(self, components: Optional[List[str]] = None):
        """Context manager for automatic startup/shutdown."""
        try:
            success = await self.start(components)
            if not success:
                raise RuntimeError("Failed to start pipeline manager")
            yield self
        finally:
            await self.shutdown()

    # Integration with existing QuickNav components

    async def process_document_event(self, event_type: str, file_path: str, project_id: str = None):
        """Process document-related events from the GUI or file system."""
        try:
            if "realtime_processor" in self._pipelines:
                processor = self._pipelines["realtime_processor"]
                await processor.process_event(event_type, file_path, project_id)

            if "activity_streamer" in self._pipelines:
                streamer = self._pipelines["activity_streamer"]
                await streamer.record_document_activity(event_type, file_path, project_id)

        except Exception as e:
            logger.error(f"Failed to process document event: {e}")

    async def record_user_activity(self, session_id: str, activity_type: str,
                                 project_id: str = None, **kwargs):
        """Record user activity for analytics."""
        try:
            if "activity_streamer" in self._pipelines:
                streamer = self._pipelines["activity_streamer"]
                await streamer.record_activity(session_id, activity_type, project_id, **kwargs)

        except Exception as e:
            logger.error(f"Failed to record user activity: {e}")

    async def get_recommendations(self, user_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get ML-based recommendations for the user."""
        try:
            if "ml_feature_processor" in self._pipelines:
                processor = self._pipelines["ml_feature_processor"]
                if hasattr(processor, 'get_recommendations'):
                    return await processor.get_recommendations(user_context)
            return []

        except Exception as e:
            logger.error(f"Failed to get recommendations: {e}")
            return []