"""
Performance Monitoring Dashboard and Alerting for Project QuickNav

Provides comprehensive monitoring including:
- Real-time performance dashboards
- Alerting and notification system
- Health checks and SLA monitoring
- Performance trend analysis
- Resource utilization tracking
- Custom metric collection
- Performance regression detection
"""

import asyncio
import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import json
import statistics
from collections import defaultdict, deque
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import sqlite3
import os

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Alert:
    """Performance alert."""
    alert_id: str
    title: str
    description: str
    severity: AlertSeverity
    metric_name: str
    threshold: float
    current_value: float
    timestamp: datetime
    acknowledged: bool = False
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class Metric:
    """Performance metric."""
    name: str
    type: MetricType
    value: Union[int, float]
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    description: str = ""


@dataclass
class HealthCheck:
    """Health check result."""
    name: str
    status: str  # healthy, warning, critical
    message: str
    response_time_ms: float
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SLATarget:
    """Service Level Agreement target."""
    name: str
    metric_name: str
    target_value: float
    comparison: str  # 'less_than', 'greater_than', 'equals'
    time_window_minutes: int
    description: str


class MetricCollector:
    """Collects and stores performance metrics."""

    def __init__(self, max_metrics_per_series: int = 10000):
        self.max_metrics_per_series = max_metrics_per_series
        self.metrics = defaultdict(lambda: deque(maxlen=max_metrics_per_series))
        self.metric_metadata = {}
        self.lock = threading.RLock()

    def record_metric(self, name: str, value: Union[int, float],
                     metric_type: MetricType = MetricType.GAUGE,
                     tags: Dict[str, str] = None, unit: str = "",
                     description: str = ""):
        """Record a metric value."""
        metric = Metric(
            name=name,
            type=metric_type,
            value=value,
            timestamp=datetime.utcnow(),
            tags=tags or {},
            unit=unit,
            description=description
        )

        with self.lock:
            series_key = self._get_series_key(name, tags or {})
            self.metrics[series_key].append(metric)

            # Store metadata
            if name not in self.metric_metadata:
                self.metric_metadata[name] = {
                    'type': metric_type,
                    'unit': unit,
                    'description': description,
                    'tags': set()
                }

            # Update tags
            for tag_key in (tags or {}):
                self.metric_metadata[name]['tags'].add(tag_key)

    def increment_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None):
        """Increment a counter metric."""
        self.record_metric(name, value, MetricType.COUNTER, tags)

    def set_gauge(self, name: str, value: Union[int, float], tags: Dict[str, str] = None):
        """Set a gauge metric value."""
        self.record_metric(name, value, MetricType.GAUGE, tags)

    def record_timer(self, name: str, duration_ms: float, tags: Dict[str, str] = None):
        """Record a timer metric."""
        self.record_metric(name, duration_ms, MetricType.TIMER, tags, "ms")

    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a histogram metric."""
        self.record_metric(name, value, MetricType.HISTOGRAM, tags)

    def get_metric_series(self, name: str, tags: Dict[str, str] = None,
                         since: Optional[datetime] = None) -> List[Metric]:
        """Get metric series data."""
        with self.lock:
            series_key = self._get_series_key(name, tags or {})
            if series_key not in self.metrics:
                return []

            metrics = list(self.metrics[series_key])

            if since:
                metrics = [m for m in metrics if m.timestamp >= since]

            return metrics

    def get_metric_summary(self, name: str, tags: Dict[str, str] = None,
                          since: Optional[datetime] = None) -> Dict[str, Any]:
        """Get statistical summary of a metric."""
        metrics = self.get_metric_series(name, tags, since)

        if not metrics:
            return {}

        values = [m.value for m in metrics]

        summary = {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'latest': values[-1],
            'first_timestamp': metrics[0].timestamp,
            'last_timestamp': metrics[-1].timestamp
        }

        if len(values) > 1:
            summary['median'] = statistics.median(values)
            summary['stddev'] = statistics.stdev(values)

            # Percentiles
            if len(values) >= 10:
                sorted_values = sorted(values)
                summary['p50'] = statistics.median(sorted_values)
                summary['p95'] = sorted_values[int(len(sorted_values) * 0.95)]
                summary['p99'] = sorted_values[int(len(sorted_values) * 0.99)]

        return summary

    def _get_series_key(self, name: str, tags: Dict[str, str]) -> str:
        """Generate series key from name and tags."""
        if not tags:
            return name

        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}#{tag_str}"

    def get_all_metric_names(self) -> List[str]:
        """Get all metric names."""
        with self.lock:
            return list(self.metric_metadata.keys())

    def cleanup_old_metrics(self, older_than_hours: int = 24):
        """Remove old metrics to save memory."""
        cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)

        with self.lock:
            for series_key in list(self.metrics.keys()):
                series = self.metrics[series_key]
                # Filter out old metrics
                filtered_metrics = deque(
                    (m for m in series if m.timestamp >= cutoff_time),
                    maxlen=self.max_metrics_per_series
                )
                self.metrics[series_key] = filtered_metrics

                # Remove empty series
                if not filtered_metrics:
                    del self.metrics[series_key]


class AlertManager:
    """Manages performance alerts and notifications."""

    def __init__(self, metric_collector: MetricCollector):
        self.metric_collector = metric_collector
        self.alert_rules = {}
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        self.notification_channels = {}
        self.lock = threading.RLock()

    def add_alert_rule(self, name: str, metric_name: str, threshold: float,
                      comparison: str, severity: AlertSeverity,
                      description: str = "", tags: Dict[str, str] = None,
                      time_window_minutes: int = 5, min_occurrences: int = 1):
        """Add an alert rule."""
        rule = {
            'name': name,
            'metric_name': metric_name,
            'threshold': threshold,
            'comparison': comparison,  # 'greater_than', 'less_than', 'equals'
            'severity': severity,
            'description': description,
            'tags': tags or {},
            'time_window_minutes': time_window_minutes,
            'min_occurrences': min_occurrences,
            'enabled': True,
            'created_at': datetime.utcnow()
        }

        with self.lock:
            self.alert_rules[name] = rule

        logger.info(f"Added alert rule: {name}")

    def remove_alert_rule(self, name: str) -> bool:
        """Remove an alert rule."""
        with self.lock:
            if name in self.alert_rules:
                del self.alert_rules[name]
                logger.info(f"Removed alert rule: {name}")
                return True
        return False

    def add_notification_channel(self, name: str, channel_type: str, config: Dict[str, Any]):
        """Add notification channel (email, webhook, etc.)."""
        self.notification_channels[name] = {
            'type': channel_type,
            'config': config,
            'enabled': True
        }
        logger.info(f"Added notification channel: {name} ({channel_type})")

    def check_alerts(self):
        """Check all alert rules against current metrics."""
        current_time = datetime.utcnow()

        with self.lock:
            for rule_name, rule in self.alert_rules.items():
                if not rule['enabled']:
                    continue

                try:
                    self._check_single_alert_rule(rule_name, rule, current_time)
                except Exception as e:
                    logger.error(f"Error checking alert rule {rule_name}: {e}")

    def _check_single_alert_rule(self, rule_name: str, rule: Dict[str, Any], current_time: datetime):
        """Check a single alert rule."""
        metric_name = rule['metric_name']
        threshold = rule['threshold']
        comparison = rule['comparison']
        time_window = timedelta(minutes=rule['time_window_minutes'])
        min_occurrences = rule['min_occurrences']

        # Get recent metrics
        since = current_time - time_window
        metrics = self.metric_collector.get_metric_series(
            metric_name, rule['tags'], since
        )

        if not metrics:
            return

        # Check violations
        violations = []
        for metric in metrics:
            if self._check_threshold(metric.value, threshold, comparison):
                violations.append(metric)

        # Check if alert should be triggered
        should_alert = len(violations) >= min_occurrences
        alert_key = f"{rule_name}#{self.metric_collector._get_series_key(metric_name, rule['tags'])}"

        if should_alert and alert_key not in self.active_alerts:
            # Trigger new alert
            self._trigger_alert(rule_name, rule, violations[-1], current_time)
        elif not should_alert and alert_key in self.active_alerts:
            # Resolve existing alert
            self._resolve_alert(alert_key, current_time)

    def _check_threshold(self, value: float, threshold: float, comparison: str) -> bool:
        """Check if value violates threshold."""
        if comparison == 'greater_than':
            return value > threshold
        elif comparison == 'less_than':
            return value < threshold
        elif comparison == 'equals':
            return abs(value - threshold) < 0.001  # Float comparison
        elif comparison == 'not_equals':
            return abs(value - threshold) >= 0.001
        else:
            return False

    def _trigger_alert(self, rule_name: str, rule: Dict[str, Any],
                      triggering_metric: Metric, current_time: datetime):
        """Trigger a new alert."""
        alert_id = f"{rule_name}_{int(current_time.timestamp())}"
        metric_name = rule['metric_name']
        alert_key = f"{rule_name}#{self.metric_collector._get_series_key(metric_name, rule['tags'])}"

        alert = Alert(
            alert_id=alert_id,
            title=f"Alert: {rule_name}",
            description=rule['description'] or f"{metric_name} {rule['comparison']} {rule['threshold']}",
            severity=rule['severity'],
            metric_name=metric_name,
            threshold=rule['threshold'],
            current_value=triggering_metric.value,
            timestamp=current_time,
            tags=list(rule['tags'].keys())
        )

        self.active_alerts[alert_key] = alert
        self.alert_history.append(alert)

        # Send notifications
        self._send_notifications(alert)

        logger.warning(f"Alert triggered: {alert.title} (value: {alert.current_value}, threshold: {alert.threshold})")

    def _resolve_alert(self, alert_key: str, current_time: datetime):
        """Resolve an active alert."""
        if alert_key in self.active_alerts:
            alert = self.active_alerts[alert_key]
            alert.resolved = True
            alert.resolved_at = current_time

            del self.active_alerts[alert_key]

            # Send resolution notification
            self._send_resolution_notification(alert)

            logger.info(f"Alert resolved: {alert.title}")

    def _send_notifications(self, alert: Alert):
        """Send alert notifications to all channels."""
        for channel_name, channel in self.notification_channels.items():
            if not channel['enabled']:
                continue

            try:
                if channel['type'] == 'email':
                    self._send_email_notification(alert, channel['config'])
                elif channel['type'] == 'webhook':
                    self._send_webhook_notification(alert, channel['config'])
            except Exception as e:
                logger.error(f"Failed to send notification via {channel_name}: {e}")

    def _send_email_notification(self, alert: Alert, config: Dict[str, Any]):
        """Send email notification."""
        smtp_server = config.get('smtp_server')
        smtp_port = config.get('smtp_port', 587)
        username = config.get('username')
        password = config.get('password')
        from_email = config.get('from_email')
        to_emails = config.get('to_emails', [])

        if not all([smtp_server, username, password, from_email, to_emails]):
            logger.error("Incomplete email configuration")
            return

        subject = f"[{alert.severity.value.upper()}] {alert.title}"

        body = f"""
Performance Alert Triggered

Alert: {alert.title}
Severity: {alert.severity.value.upper()}
Metric: {alert.metric_name}
Current Value: {alert.current_value}
Threshold: {alert.threshold}
Time: {alert.timestamp}

Description: {alert.description}

Alert ID: {alert.alert_id}
"""

        msg = MimeMultipart()
        msg['From'] = from_email
        msg['To'] = ', '.join(to_emails)
        msg['Subject'] = subject
        msg.attach(MimeText(body, 'plain'))

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(username, password)
            server.send_message(msg)

    def _send_webhook_notification(self, alert: Alert, config: Dict[str, Any]):
        """Send webhook notification."""
        import requests

        url = config.get('url')
        headers = config.get('headers', {})

        payload = {
            'alert_id': alert.alert_id,
            'title': alert.title,
            'description': alert.description,
            'severity': alert.severity.value,
            'metric_name': alert.metric_name,
            'current_value': alert.current_value,
            'threshold': alert.threshold,
            'timestamp': alert.timestamp.isoformat(),
            'tags': alert.tags
        }

        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()

    def _send_resolution_notification(self, alert: Alert):
        """Send alert resolution notification."""
        # Similar to _send_notifications but for resolution
        # Implementation would be similar to above methods
        pass

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an active alert."""
        with self.lock:
            for alert in self.active_alerts.values():
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    logger.info(f"Alert acknowledged: {alert.title}")
                    return True
        return False

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        with self.lock:
            return list(self.active_alerts.values())

    def get_alert_history(self, limit: int = 50) -> List[Alert]:
        """Get recent alert history."""
        return list(self.alert_history)[-limit:]


class HealthCheckManager:
    """Manages system health checks."""

    def __init__(self):
        self.health_checks = {}
        self.health_results = deque(maxlen=1000)
        self.lock = threading.RLock()

    def register_health_check(self, name: str, check_function: Callable,
                            interval_seconds: int = 60, timeout_seconds: int = 5):
        """Register a health check function."""
        self.health_checks[name] = {
            'function': check_function,
            'interval_seconds': interval_seconds,
            'timeout_seconds': timeout_seconds,
            'last_run': None,
            'enabled': True
        }
        logger.info(f"Registered health check: {name}")

    def run_health_checks(self):
        """Run all enabled health checks."""
        current_time = datetime.utcnow()

        for name, check_config in self.health_checks.items():
            if not check_config['enabled']:
                continue

            last_run = check_config['last_run']
            interval = timedelta(seconds=check_config['interval_seconds'])

            if last_run is None or current_time - last_run >= interval:
                try:
                    self._run_single_health_check(name, check_config, current_time)
                except Exception as e:
                    logger.error(f"Health check {name} failed: {e}")
                    self._record_health_check_result(
                        name, "critical", f"Health check failed: {e}", 0, current_time
                    )

    def _run_single_health_check(self, name: str, config: Dict[str, Any], current_time: datetime):
        """Run a single health check."""
        start_time = time.perf_counter()

        try:
            # Run health check with timeout
            result = asyncio.wait_for(
                self._run_check_function(config['function']),
                timeout=config['timeout_seconds']
            )

            # Handle result
            if isinstance(result, dict):
                status = result.get('status', 'healthy')
                message = result.get('message', 'OK')
                details = result.get('details', {})
            else:
                status = 'healthy' if result else 'critical'
                message = 'OK' if result else 'Check failed'
                details = {}

            response_time = (time.perf_counter() - start_time) * 1000
            self._record_health_check_result(name, status, message, response_time, current_time, details)

            config['last_run'] = current_time

        except asyncio.TimeoutError:
            response_time = (time.perf_counter() - start_time) * 1000
            self._record_health_check_result(
                name, "critical", "Health check timed out", response_time, current_time
            )

    async def _run_check_function(self, check_function: Callable):
        """Run health check function (async or sync)."""
        if asyncio.iscoroutinefunction(check_function):
            return await check_function()
        else:
            return check_function()

    def _record_health_check_result(self, name: str, status: str, message: str,
                                   response_time_ms: float, timestamp: datetime,
                                   details: Dict[str, Any] = None):
        """Record health check result."""
        result = HealthCheck(
            name=name,
            status=status,
            message=message,
            response_time_ms=response_time_ms,
            timestamp=timestamp,
            details=details or {}
        )

        with self.lock:
            self.health_results.append(result)

    def get_latest_health_status(self) -> Dict[str, HealthCheck]:
        """Get latest health status for all checks."""
        latest_results = {}

        with self.lock:
            for result in reversed(self.health_results):
                if result.name not in latest_results:
                    latest_results[result.name] = result

        return latest_results

    def get_overall_health_status(self) -> Dict[str, Any]:
        """Get overall system health status."""
        latest_results = self.get_latest_health_status()

        if not latest_results:
            return {'status': 'unknown', 'message': 'No health checks configured'}

        status_counts = defaultdict(int)
        total_response_time = 0

        for result in latest_results.values():
            status_counts[result.status] += 1
            total_response_time += result.response_time_ms

        # Determine overall status
        if status_counts['critical'] > 0:
            overall_status = 'critical'
        elif status_counts['warning'] > 0:
            overall_status = 'warning'
        else:
            overall_status = 'healthy'

        avg_response_time = total_response_time / len(latest_results)

        return {
            'status': overall_status,
            'total_checks': len(latest_results),
            'healthy_checks': status_counts['healthy'],
            'warning_checks': status_counts['warning'],
            'critical_checks': status_counts['critical'],
            'avg_response_time_ms': avg_response_time,
            'last_updated': datetime.utcnow().isoformat()
        }


class PerformanceMonitor:
    """Main performance monitoring system."""

    def __init__(self, db_path: str = "data/monitoring.db"):
        self.db_path = db_path
        self.metric_collector = MetricCollector()
        self.alert_manager = AlertManager(self.metric_collector)
        self.health_check_manager = HealthCheckManager()

        # SLA tracking
        self.sla_targets = {}

        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread = None

        # Dashboard data
        self.dashboard_data = {}

        # Setup database
        self._setup_database()

        # Setup default health checks
        self._setup_default_health_checks()

        # Setup default alerts
        self._setup_default_alerts()

    def _setup_database(self):
        """Setup monitoring database."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metric_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp DATETIME NOT NULL,
                    tags TEXT DEFAULT '{}',
                    unit TEXT DEFAULT ''
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS alert_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT NOT NULL,
                    rule_name TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    threshold_value REAL NOT NULL,
                    actual_value REAL NOT NULL,
                    triggered_at DATETIME NOT NULL,
                    resolved_at DATETIME,
                    acknowledged BOOLEAN DEFAULT FALSE
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS health_check_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    check_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT NOT NULL,
                    response_time_ms REAL NOT NULL,
                    timestamp DATETIME NOT NULL,
                    details TEXT DEFAULT '{}'
                )
            """)

            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metric_name_time ON metric_snapshots(metric_name, timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alert_time ON alert_events(triggered_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_health_check_time ON health_check_results(check_name, timestamp)")

    def _setup_default_health_checks(self):
        """Setup default health checks."""
        def check_database_connection():
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("SELECT 1").fetchone()
                return {'status': 'healthy', 'message': 'Database connection OK'}
            except Exception as e:
                return {'status': 'critical', 'message': f'Database connection failed: {e}'}

        def check_memory_usage():
            try:
                import psutil
                memory = psutil.virtual_memory()
                if memory.percent > 90:
                    return {'status': 'critical', 'message': f'High memory usage: {memory.percent}%'}
                elif memory.percent > 80:
                    return {'status': 'warning', 'message': f'Memory usage: {memory.percent}%'}
                else:
                    return {'status': 'healthy', 'message': f'Memory usage: {memory.percent}%'}
            except ImportError:
                return {'status': 'warning', 'message': 'psutil not available for memory check'}

        def check_disk_space():
            try:
                import psutil
                disk = psutil.disk_usage('/')
                usage_percent = (disk.used / disk.total) * 100
                if usage_percent > 90:
                    return {'status': 'critical', 'message': f'High disk usage: {usage_percent:.1f}%'}
                elif usage_percent > 80:
                    return {'status': 'warning', 'message': f'Disk usage: {usage_percent:.1f}%'}
                else:
                    return {'status': 'healthy', 'message': f'Disk usage: {usage_percent:.1f}%'}
            except ImportError:
                return {'status': 'warning', 'message': 'psutil not available for disk check'}

        self.health_check_manager.register_health_check('database', check_database_connection, 30)
        self.health_check_manager.register_health_check('memory', check_memory_usage, 60)
        self.health_check_manager.register_health_check('disk', check_disk_space, 300)

    def _setup_default_alerts(self):
        """Setup default alert rules."""
        # Response time alerts
        self.alert_manager.add_alert_rule(
            'high_response_time',
            'response_time_ms',
            1000,  # 1 second
            'greater_than',
            AlertSeverity.WARNING,
            'Response time exceeded 1 second'
        )

        # Error rate alerts
        self.alert_manager.add_alert_rule(
            'high_error_rate',
            'error_rate',
            0.05,  # 5%
            'greater_than',
            AlertSeverity.ERROR,
            'Error rate exceeded 5%'
        )

        # Cache hit rate alerts
        self.alert_manager.add_alert_rule(
            'low_cache_hit_rate',
            'cache_hit_rate',
            0.5,  # 50%
            'less_than',
            AlertSeverity.WARNING,
            'Cache hit rate below 50%'
        )

    def start_monitoring(self):
        """Start performance monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Performance monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Run health checks
                self.health_check_manager.run_health_checks()

                # Check alerts
                self.alert_manager.check_alerts()

                # Update dashboard data
                self._update_dashboard_data()

                # Store snapshots in database
                self._store_metric_snapshots()

                # Cleanup old data
                self._cleanup_old_data()

                time.sleep(10)  # Run every 10 seconds

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(10)

    def _update_dashboard_data(self):
        """Update dashboard data."""
        current_time = datetime.utcnow()

        # Get recent performance data
        since_1h = current_time - timedelta(hours=1)
        since_24h = current_time - timedelta(hours=24)

        dashboard_data = {
            'timestamp': current_time.isoformat(),
            'health_status': self.health_check_manager.get_overall_health_status(),
            'active_alerts': len(self.alert_manager.get_active_alerts()),
            'metrics': {}
        }

        # Get key metrics
        for metric_name in self.metric_collector.get_all_metric_names():
            summary_1h = self.metric_collector.get_metric_summary(metric_name, since=since_1h)
            summary_24h = self.metric_collector.get_metric_summary(metric_name, since=since_24h)

            if summary_1h:
                dashboard_data['metrics'][metric_name] = {
                    '1h': summary_1h,
                    '24h': summary_24h
                }

        self.dashboard_data = dashboard_data

    def _store_metric_snapshots(self):
        """Store current metric values in database."""
        try:
            current_time = datetime.utcnow()

            with sqlite3.connect(self.db_path) as conn:
                for metric_name in self.metric_collector.get_all_metric_names():
                    # Get latest value
                    recent_metrics = self.metric_collector.get_metric_series(
                        metric_name, since=current_time - timedelta(minutes=1)
                    )

                    if recent_metrics:
                        latest_metric = recent_metrics[-1]
                        conn.execute("""
                            INSERT INTO metric_snapshots
                            (metric_name, value, timestamp, tags, unit)
                            VALUES (?, ?, ?, ?, ?)
                        """, [
                            metric_name,
                            latest_metric.value,
                            current_time,
                            json.dumps(latest_metric.tags),
                            latest_metric.unit
                        ])

        except Exception as e:
            logger.error(f"Failed to store metric snapshots: {e}")

    def _cleanup_old_data(self):
        """Clean up old monitoring data."""
        try:
            # Clean up in-memory metrics
            self.metric_collector.cleanup_old_metrics(24)

            # Clean up database records older than 30 days
            cutoff_time = datetime.utcnow() - timedelta(days=30)

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM metric_snapshots WHERE timestamp < ?", [cutoff_time])
                conn.execute("DELETE FROM health_check_results WHERE timestamp < ?", [cutoff_time])

                # Keep alert events for 90 days
                alert_cutoff = datetime.utcnow() - timedelta(days=90)
                conn.execute("DELETE FROM alert_events WHERE triggered_at < ?", [alert_cutoff])

        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")

    def record_operation_metric(self, operation: str, duration_ms: float, success: bool = True):
        """Record metrics for an operation."""
        tags = {'operation': operation, 'status': 'success' if success else 'error'}

        self.metric_collector.record_timer(f'operation_duration', duration_ms, tags)
        self.metric_collector.increment_counter(f'operation_count', 1, tags)

        if not success:
            self.metric_collector.increment_counter('error_count', 1, {'operation': operation})

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data."""
        return self.dashboard_data

    def get_metrics_api(self, metric_name: str = None, since_hours: int = 1) -> Dict[str, Any]:
        """Get metrics data for API consumption."""
        since = datetime.utcnow() - timedelta(hours=since_hours)

        if metric_name:
            # Get specific metric
            metrics = self.metric_collector.get_metric_series(metric_name, since=since)
            summary = self.metric_collector.get_metric_summary(metric_name, since=since)

            return {
                'metric_name': metric_name,
                'data_points': [asdict(m) for m in metrics],
                'summary': summary
            }
        else:
            # Get all metrics
            result = {}
            for name in self.metric_collector.get_all_metric_names():
                summary = self.metric_collector.get_metric_summary(name, since=since)
                if summary:
                    result[name] = summary

            return result

    def add_sla_target(self, name: str, metric_name: str, target_value: float,
                      comparison: str, time_window_minutes: int = 60,
                      description: str = ""):
        """Add SLA target for monitoring."""
        self.sla_targets[name] = SLATarget(
            name=name,
            metric_name=metric_name,
            target_value=target_value,
            comparison=comparison,
            time_window_minutes=time_window_minutes,
            description=description
        )

    def get_sla_compliance(self) -> Dict[str, Any]:
        """Get SLA compliance status."""
        compliance = {}
        current_time = datetime.utcnow()

        for sla_name, sla in self.sla_targets.items():
            since = current_time - timedelta(minutes=sla.time_window_minutes)
            summary = self.metric_collector.get_metric_summary(sla.metric_name, since=since)

            if summary:
                # Calculate compliance based on comparison type
                if sla.comparison == 'less_than':
                    compliant = summary.get('mean', 0) < sla.target_value
                    compliance_percentage = min(100, (sla.target_value / summary.get('mean', 1)) * 100)
                elif sla.comparison == 'greater_than':
                    compliant = summary.get('mean', 0) > sla.target_value
                    compliance_percentage = min(100, (summary.get('mean', 0) / sla.target_value) * 100)
                else:
                    compliant = True
                    compliance_percentage = 100

                compliance[sla_name] = {
                    'compliant': compliant,
                    'compliance_percentage': compliance_percentage,
                    'current_value': summary.get('mean', 0),
                    'target_value': sla.target_value,
                    'time_window_minutes': sla.time_window_minutes,
                    'description': sla.description
                }
            else:
                compliance[sla_name] = {
                    'compliant': None,
                    'compliance_percentage': 0,
                    'current_value': None,
                    'target_value': sla.target_value,
                    'error': 'No data available'
                }

        return compliance


# Context manager for operation timing
class timed_operation:
    """Context manager for timing operations and recording metrics."""

    def __init__(self, operation_name: str, monitor: PerformanceMonitor = None,
                 tags: Dict[str, str] = None):
        self.operation_name = operation_name
        self.monitor = monitor
        self.tags = tags or {}
        self.start_time = None
        self.success = True

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self.start_time) * 1000
        self.success = exc_type is None

        if self.monitor:
            self.monitor.record_operation_metric(
                self.operation_name, duration_ms, self.success
            )


# Global monitor instance
_global_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get or create global performance monitor."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
        _global_monitor.start_monitoring()
    return _global_monitor