"""
Analytics Processor - Batch processing for data analytics and reporting

Features:
- Scheduled analytics computation and aggregation
- User behavior analysis and pattern detection
- Project usage statistics and trends
- Document lifecycle analytics
- Performance metrics and KPI calculation
- Data warehouse optimization
- Historical trend analysis
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
import statistics
from dataclasses import dataclass
from enum import Enum

from ...database.database_manager import DatabaseManager

logger = logging.getLogger(__name__)


class AnalyticsJobType(Enum):
    """Types of analytics jobs."""
    DAILY_AGGREGATION = "daily_aggregation"
    WEEKLY_SUMMARY = "weekly_summary"
    MONTHLY_REPORT = "monthly_report"
    USER_BEHAVIOR = "user_behavior"
    PROJECT_ANALYTICS = "project_analytics"
    DOCUMENT_LIFECYCLE = "document_lifecycle"
    PERFORMANCE_METRICS = "performance_metrics"
    TREND_ANALYSIS = "trend_analysis"


@dataclass
class AnalyticsJob:
    """Represents an analytics processing job."""
    job_id: str
    job_type: AnalyticsJobType
    start_date: datetime
    end_date: datetime
    parameters: Dict[str, Any]
    created_at: datetime
    status: str = "pending"
    progress: float = 0.0
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class UserBehaviorAnalyzer:
    """Analyzes user behavior patterns and usage statistics."""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    async def analyze_user_patterns(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Analyze user behavior patterns for the given time period."""
        try:
            # User activity patterns
            activity_patterns = await self._analyze_activity_patterns(start_date, end_date)

            # Search behavior analysis
            search_patterns = await self._analyze_search_patterns(start_date, end_date)

            # Navigation patterns
            navigation_patterns = await self._analyze_navigation_patterns(start_date, end_date)

            # Session analysis
            session_analysis = await self._analyze_sessions(start_date, end_date)

            return {
                'analysis_period': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat()
                },
                'activity_patterns': activity_patterns,
                'search_patterns': search_patterns,
                'navigation_patterns': navigation_patterns,
                'session_analysis': session_analysis,
                'generated_at': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"User behavior analysis failed: {e}")
            raise

    async def _analyze_activity_patterns(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Analyze activity patterns by time and type."""
        query = """
        SELECT
            activity_type,
            strftime('%H', timestamp) as hour,
            strftime('%w', timestamp) as day_of_week,
            COUNT(*) as activity_count,
            AVG(response_time_ms) as avg_response_time
        FROM user_activities
        WHERE timestamp BETWEEN ? AND ?
        GROUP BY activity_type, hour, day_of_week
        ORDER BY activity_count DESC
        """

        results = await self.db_manager.execute_query(
            query, [start_date.isoformat(), end_date.isoformat()], prefer_analytics=True
        )

        # Process results into structured patterns
        patterns = {
            'hourly_distribution': {},
            'daily_distribution': {},
            'activity_type_distribution': {},
            'peak_hours': [],
            'peak_days': []
        }

        hourly_totals = {}
        daily_totals = {}
        activity_totals = {}

        for row in results:
            hour = int(row['hour'])
            day = int(row['day_of_week'])
            activity_type = row['activity_type']
            count = row['activity_count']

            # Aggregate by hour
            hourly_totals[hour] = hourly_totals.get(hour, 0) + count

            # Aggregate by day
            daily_totals[day] = daily_totals.get(day, 0) + count

            # Aggregate by activity type
            activity_totals[activity_type] = activity_totals.get(activity_type, 0) + count

        patterns['hourly_distribution'] = hourly_totals
        patterns['daily_distribution'] = daily_totals
        patterns['activity_type_distribution'] = activity_totals

        # Find peak times
        if hourly_totals:
            peak_hour = max(hourly_totals.items(), key=lambda x: x[1])
            patterns['peak_hours'] = [peak_hour[0]]

        if daily_totals:
            peak_day = max(daily_totals.items(), key=lambda x: x[1])
            patterns['peak_days'] = [peak_day[0]]

        return patterns

    async def _analyze_search_patterns(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Analyze search behavior and query patterns."""
        query = """
        SELECT
            query_text,
            query_type,
            results_count,
            selected_result_index,
            search_time_ms,
            cache_hit,
            COUNT(*) as frequency
        FROM search_analytics
        WHERE timestamp BETWEEN ? AND ?
        GROUP BY query_text, query_type
        ORDER BY frequency DESC
        LIMIT 100
        """

        results = await self.db_manager.execute_query(
            query, [start_date.isoformat(), end_date.isoformat()], prefer_analytics=True
        )

        patterns = {
            'most_frequent_queries': [],
            'query_type_distribution': {},
            'avg_results_per_query': 0.0,
            'avg_search_time_ms': 0.0,
            'cache_hit_rate': 0.0,
            'selection_patterns': {}
        }

        total_queries = len(results)
        total_results = 0
        total_search_time = 0
        cache_hits = 0
        query_types = {}
        selection_indices = {}

        for row in results:
            patterns['most_frequent_queries'].append({
                'query': row['query_text'],
                'type': row['query_type'],
                'frequency': row['frequency'],
                'avg_results': row['results_count'],
                'avg_search_time': row['search_time_ms']
            })

            # Aggregate statistics
            total_results += row['results_count'] or 0
            total_search_time += row['search_time_ms'] or 0

            if row['cache_hit']:
                cache_hits += 1

            # Query type distribution
            query_type = row['query_type']
            query_types[query_type] = query_types.get(query_type, 0) + row['frequency']

            # Selection patterns
            if row['selected_result_index'] is not None:
                idx = row['selected_result_index']
                selection_indices[idx] = selection_indices.get(idx, 0) + 1

        if total_queries > 0:
            patterns['avg_results_per_query'] = total_results / total_queries
            patterns['avg_search_time_ms'] = total_search_time / total_queries
            patterns['cache_hit_rate'] = cache_hits / total_queries

        patterns['query_type_distribution'] = query_types
        patterns['selection_patterns'] = selection_indices

        return patterns

    async def _analyze_navigation_patterns(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Analyze navigation patterns and project access."""
        query = """
        SELECT
            ua.project_id,
            p.project_code,
            p.project_name,
            COUNT(*) as access_count,
            COUNT(DISTINCT ua.session_id) as unique_sessions,
            MIN(ua.timestamp) as first_access,
            MAX(ua.timestamp) as last_access,
            AVG(ua.response_time_ms) as avg_response_time
        FROM user_activities ua
        LEFT JOIN projects p ON ua.project_id = p.project_id
        WHERE ua.timestamp BETWEEN ? AND ?
            AND ua.activity_type IN ('navigate', 'document_open')
            AND ua.project_id IS NOT NULL
        GROUP BY ua.project_id
        ORDER BY access_count DESC
        LIMIT 50
        """

        results = await self.db_manager.execute_query(
            query, [start_date.isoformat(), end_date.isoformat()], prefer_analytics=True
        )

        patterns = {
            'most_accessed_projects': [],
            'access_distribution': {},
            'session_patterns': {},
            'temporal_patterns': {}
        }

        for row in results:
            project_info = {
                'project_id': row['project_id'],
                'project_code': row['project_code'],
                'project_name': row['project_name'],
                'access_count': row['access_count'],
                'unique_sessions': row['unique_sessions'],
                'avg_response_time': row['avg_response_time'],
                'first_access': row['first_access'],
                'last_access': row['last_access']
            }
            patterns['most_accessed_projects'].append(project_info)

        return patterns

    async def _analyze_sessions(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Analyze user session characteristics."""
        query = """
        SELECT
            session_duration_seconds,
            total_navigations,
            total_searches,
            ai_interactions,
            app_version,
            os_platform
        FROM user_sessions
        WHERE session_start BETWEEN ? AND ?
            AND session_duration_seconds IS NOT NULL
        """

        results = await self.db_manager.execute_query(
            query, [start_date.isoformat(), end_date.isoformat()], prefer_analytics=True
        )

        if not results:
            return {'error': 'No session data available'}

        durations = [row['session_duration_seconds'] for row in results if row['session_duration_seconds']]
        navigations = [row['total_navigations'] for row in results if row['total_navigations']]
        searches = [row['total_searches'] for row in results if row['total_searches']]
        ai_interactions = [row['ai_interactions'] for row in results if row['ai_interactions']]

        analysis = {
            'total_sessions': len(results),
            'duration_stats': self._calculate_stats(durations),
            'navigation_stats': self._calculate_stats(navigations),
            'search_stats': self._calculate_stats(searches),
            'ai_interaction_stats': self._calculate_stats(ai_interactions),
            'platform_distribution': self._calculate_distribution(results, 'os_platform'),
            'version_distribution': self._calculate_distribution(results, 'app_version')
        }

        return analysis

    def _calculate_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistical metrics for a list of values."""
        if not values:
            return {'count': 0}

        return {
            'count': len(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0,
            'min': min(values),
            'max': max(values),
            'percentile_25': statistics.quantiles(values, n=4)[0] if len(values) > 3 else min(values),
            'percentile_75': statistics.quantiles(values, n=4)[2] if len(values) > 3 else max(values)
        }

    def _calculate_distribution(self, results: List[Dict], field: str) -> Dict[str, int]:
        """Calculate distribution of values for a field."""
        distribution = {}
        for row in results:
            value = row.get(field, 'unknown')
            distribution[value] = distribution.get(value, 0) + 1
        return distribution


class ProjectAnalyzer:
    """Analyzes project-level statistics and trends."""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    async def analyze_project_metrics(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Analyze comprehensive project metrics."""
        try:
            # Project access metrics
            access_metrics = await self._analyze_project_access(start_date, end_date)

            # Document metrics
            document_metrics = await self._analyze_project_documents(start_date, end_date)

            # Growth metrics
            growth_metrics = await self._analyze_project_growth(start_date, end_date)

            return {
                'analysis_period': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat()
                },
                'access_metrics': access_metrics,
                'document_metrics': document_metrics,
                'growth_metrics': growth_metrics,
                'generated_at': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Project metrics analysis failed: {e}")
            raise

    async def _analyze_project_access(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Analyze project access patterns."""
        query = """
        SELECT
            p.project_code,
            p.project_name,
            p.range_folder,
            COUNT(ua.activity_id) as total_activities,
            COUNT(DISTINCT ua.session_id) as unique_users,
            COUNT(DISTINCT DATE(ua.timestamp)) as active_days,
            AVG(ua.response_time_ms) as avg_response_time,
            SUM(CASE WHEN ua.activity_type = 'document_open' THEN 1 ELSE 0 END) as document_opens
        FROM projects p
        LEFT JOIN user_activities ua ON p.project_id = ua.project_id
            AND ua.timestamp BETWEEN ? AND ?
        WHERE p.is_active = 1
        GROUP BY p.project_id
        HAVING total_activities > 0
        ORDER BY total_activities DESC
        """

        results = await self.db_manager.execute_query(
            query, [start_date.isoformat(), end_date.isoformat()], prefer_analytics=True
        )

        metrics = {
            'total_active_projects': len(results),
            'top_projects': results[:20] if results else [],
            'range_distribution': {},
            'activity_distribution': self._calculate_activity_distribution(results)
        }

        # Calculate range distribution
        for row in results:
            range_folder = row['range_folder'] or 'Unknown'
            if range_folder not in metrics['range_distribution']:
                metrics['range_distribution'][range_folder] = {
                    'project_count': 0,
                    'total_activities': 0,
                    'avg_response_time': 0.0
                }

            metrics['range_distribution'][range_folder]['project_count'] += 1
            metrics['range_distribution'][range_folder]['total_activities'] += row['total_activities']
            metrics['range_distribution'][range_folder]['avg_response_time'] += row['avg_response_time'] or 0

        # Average the response times
        for range_data in metrics['range_distribution'].values():
            if range_data['project_count'] > 0:
                range_data['avg_response_time'] /= range_data['project_count']

        return metrics

    async def _analyze_project_documents(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Analyze document metrics by project."""
        query = """
        SELECT
            p.project_code,
            p.project_name,
            COUNT(d.document_id) as total_documents,
            COUNT(DISTINCT d.document_type) as document_types,
            COUNT(CASE WHEN d.status_weight > 1.0 THEN 1 END) as high_status_docs,
            AVG(d.file_size_bytes) as avg_file_size,
            MAX(d.modified_at) as latest_modification
        FROM projects p
        LEFT JOIN documents d ON p.project_id = d.project_id
            AND d.is_active = 1
        WHERE p.is_active = 1
        GROUP BY p.project_id
        HAVING total_documents > 0
        ORDER BY total_documents DESC
        """

        results = await self.db_manager.execute_query(query, prefer_analytics=True)

        if not results:
            return {'error': 'No document data available'}

        doc_counts = [row['total_documents'] for row in results]
        file_sizes = [row['avg_file_size'] for row in results if row['avg_file_size']]

        metrics = {
            'projects_with_documents': len(results),
            'document_count_stats': self._calculate_stats(doc_counts),
            'file_size_stats': self._calculate_stats(file_sizes),
            'top_documented_projects': results[:10],
            'document_type_distribution': await self._get_document_type_distribution()
        }

        return metrics

    async def _analyze_project_growth(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Analyze project creation and growth trends."""
        query = """
        SELECT
            DATE(created_at) as creation_date,
            COUNT(*) as projects_created,
            range_folder
        FROM projects
        WHERE created_at BETWEEN ? AND ?
        GROUP BY DATE(created_at), range_folder
        ORDER BY creation_date
        """

        results = await self.db_manager.execute_query(
            query, [start_date.isoformat(), end_date.isoformat()], prefer_analytics=True
        )

        growth_metrics = {
            'new_projects_count': sum(row['projects_created'] for row in results),
            'daily_creation_trend': results,
            'range_growth': {}
        }

        # Calculate growth by range
        for row in results:
            range_folder = row['range_folder'] or 'Unknown'
            if range_folder not in growth_metrics['range_growth']:
                growth_metrics['range_growth'][range_folder] = 0
            growth_metrics['range_growth'][range_folder] += row['projects_created']

        return growth_metrics

    async def _get_document_type_distribution(self) -> Dict[str, int]:
        """Get distribution of document types across all projects."""
        query = """
        SELECT document_type, COUNT(*) as count
        FROM documents
        WHERE is_active = 1 AND document_type IS NOT NULL
        GROUP BY document_type
        ORDER BY count DESC
        """

        results = await self.db_manager.execute_query(query, prefer_analytics=True)
        return {row['document_type']: row['count'] for row in results}

    def _calculate_activity_distribution(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate activity distribution statistics."""
        if not results:
            return {}

        activities = [row['total_activities'] for row in results]
        return {
            'quartiles': {
                'low_activity': len([a for a in activities if a <= 5]),
                'medium_activity': len([a for a in activities if 5 < a <= 20]),
                'high_activity': len([a for a in activities if a > 20])
            },
            'stats': self._calculate_stats(activities)
        }


class AnalyticsProcessor:
    """
    Main analytics processor for batch analytics jobs.

    Handles scheduled analytics processing including:
    - User behavior analysis
    - Project metrics computation
    - Trend analysis and forecasting
    - Performance metric calculation
    - Data aggregation and summarization
    """

    def __init__(self, config: Dict[str, Any], db_manager: DatabaseManager):
        self.config = config
        self.db_manager = db_manager

        # Configuration
        self.batch_size = config.get('batch_size', 1000)
        self.max_processing_time = config.get('max_processing_time', 3600)  # 1 hour
        self.retention_days = config.get('retention_days', 90)

        # Components
        self.user_analyzer = UserBehaviorAnalyzer(db_manager)
        self.project_analyzer = ProjectAnalyzer(db_manager)

        # State management
        self.running = False
        self.current_jobs: Dict[str, AnalyticsJob] = {}

        # Performance metrics
        self.metrics = {
            'jobs_completed': 0,
            'jobs_failed': 0,
            'total_processing_time': 0.0,
            'avg_job_time': 0.0,
            'last_run': None
        }

    async def start(self):
        """Start the analytics processor."""
        if self.running:
            logger.warning("Analytics processor already running")
            return

        logger.info("Starting analytics processor...")
        self.running = True

        logger.info("Analytics processor started")

    async def shutdown(self):
        """Shutdown the analytics processor."""
        if not self.running:
            return

        logger.info("Shutting down analytics processor...")
        self.running = False

        # Cancel running jobs
        for job in self.current_jobs.values():
            job.status = "cancelled"

        logger.info("Analytics processor shutdown complete")

    async def run_batch(self, job_type: str = None, parameters: Dict[str, Any] = None):
        """Run batch analytics processing."""
        if not self.running:
            logger.warning("Analytics processor not running")
            return

        # Default to daily aggregation if no job type specified
        if job_type is None:
            job_type = AnalyticsJobType.DAILY_AGGREGATION.value

        try:
            job_type_enum = AnalyticsJobType(job_type)
        except ValueError:
            logger.error(f"Unknown job type: {job_type}")
            return

        # Set default time range
        end_date = datetime.now()
        if job_type_enum == AnalyticsJobType.DAILY_AGGREGATION:
            start_date = end_date - timedelta(days=1)
        elif job_type_enum == AnalyticsJobType.WEEKLY_SUMMARY:
            start_date = end_date - timedelta(days=7)
        elif job_type_enum == AnalyticsJobType.MONTHLY_REPORT:
            start_date = end_date - timedelta(days=30)
        else:
            start_date = end_date - timedelta(days=7)  # Default to week

        # Override with parameters if provided
        if parameters:
            if 'start_date' in parameters:
                start_date = datetime.fromisoformat(parameters['start_date'])
            if 'end_date' in parameters:
                end_date = datetime.fromisoformat(parameters['end_date'])

        job_id = f"{job_type}_{int(time.time())}"
        job = AnalyticsJob(
            job_id=job_id,
            job_type=job_type_enum,
            start_date=start_date,
            end_date=end_date,
            parameters=parameters or {},
            created_at=datetime.now()
        )

        await self._process_job(job)

    async def _process_job(self, job: AnalyticsJob):
        """Process a single analytics job."""
        start_time = time.time()
        job.status = "running"
        self.current_jobs[job.job_id] = job

        try:
            logger.info(f"Processing analytics job {job.job_id}: {job.job_type.value}")

            if job.job_type == AnalyticsJobType.DAILY_AGGREGATION:
                results = await self._run_daily_aggregation(job)
            elif job.job_type == AnalyticsJobType.WEEKLY_SUMMARY:
                results = await self._run_weekly_summary(job)
            elif job.job_type == AnalyticsJobType.MONTHLY_REPORT:
                results = await self._run_monthly_report(job)
            elif job.job_type == AnalyticsJobType.USER_BEHAVIOR:
                results = await self._run_user_behavior_analysis(job)
            elif job.job_type == AnalyticsJobType.PROJECT_ANALYTICS:
                results = await self._run_project_analytics(job)
            else:
                raise ValueError(f"Unsupported job type: {job.job_type}")

            job.results = results
            job.status = "completed"
            job.progress = 100.0

            # Store results in database
            await self._store_analytics_results(job)

            processing_time = time.time() - start_time
            self.metrics['jobs_completed'] += 1
            self.metrics['total_processing_time'] += processing_time
            self.metrics['avg_job_time'] = self.metrics['total_processing_time'] / max(1, self.metrics['jobs_completed'])
            self.metrics['last_run'] = datetime.now()

            logger.info(f"Completed analytics job {job.job_id} in {processing_time:.1f}s")

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            self.metrics['jobs_failed'] += 1
            logger.error(f"Analytics job {job.job_id} failed: {e}")

        finally:
            if job.job_id in self.current_jobs:
                del self.current_jobs[job.job_id]

    async def _run_daily_aggregation(self, job: AnalyticsJob) -> Dict[str, Any]:
        """Run daily data aggregation."""
        results = {
            'job_type': 'daily_aggregation',
            'period': {
                'start_date': job.start_date.isoformat(),
                'end_date': job.end_date.isoformat()
            }
        }

        # Basic activity aggregation
        activity_query = """
        SELECT
            activity_type,
            COUNT(*) as count,
            AVG(response_time_ms) as avg_response_time,
            COUNT(DISTINCT session_id) as unique_sessions
        FROM user_activities
        WHERE timestamp BETWEEN ? AND ?
        GROUP BY activity_type
        """

        activity_results = await self.db_manager.execute_query(
            activity_query, [job.start_date.isoformat(), job.end_date.isoformat()], prefer_analytics=True
        )

        results['activity_summary'] = activity_results

        # Project access aggregation
        project_query = """
        SELECT
            project_id,
            COUNT(*) as access_count,
            COUNT(DISTINCT session_id) as unique_users
        FROM user_activities
        WHERE timestamp BETWEEN ? AND ?
            AND project_id IS NOT NULL
        GROUP BY project_id
        ORDER BY access_count DESC
        LIMIT 10
        """

        project_results = await self.db_manager.execute_query(
            project_query, [job.start_date.isoformat(), job.end_date.isoformat()], prefer_analytics=True
        )

        results['top_projects'] = project_results

        return results

    async def _run_weekly_summary(self, job: AnalyticsJob) -> Dict[str, Any]:
        """Run weekly summary analysis."""
        results = {
            'job_type': 'weekly_summary',
            'period': {
                'start_date': job.start_date.isoformat(),
                'end_date': job.end_date.isoformat()
            }
        }

        # Include user behavior analysis
        user_behavior = await self.user_analyzer.analyze_user_patterns(job.start_date, job.end_date)
        results['user_behavior'] = user_behavior

        # Include project metrics
        project_metrics = await self.project_analyzer.analyze_project_metrics(job.start_date, job.end_date)
        results['project_metrics'] = project_metrics

        return results

    async def _run_monthly_report(self, job: AnalyticsJob) -> Dict[str, Any]:
        """Run comprehensive monthly report."""
        results = {
            'job_type': 'monthly_report',
            'period': {
                'start_date': job.start_date.isoformat(),
                'end_date': job.end_date.isoformat()
            }
        }

        # Include all available analyses
        user_behavior = await self.user_analyzer.analyze_user_patterns(job.start_date, job.end_date)
        project_metrics = await self.project_analyzer.analyze_project_metrics(job.start_date, job.end_date)

        results['user_behavior'] = user_behavior
        results['project_metrics'] = project_metrics

        # Add trend analysis
        results['trends'] = await self._analyze_trends(job.start_date, job.end_date)

        return results

    async def _run_user_behavior_analysis(self, job: AnalyticsJob) -> Dict[str, Any]:
        """Run focused user behavior analysis."""
        return await self.user_analyzer.analyze_user_patterns(job.start_date, job.end_date)

    async def _run_project_analytics(self, job: AnalyticsJob) -> Dict[str, Any]:
        """Run focused project analytics."""
        return await self.project_analyzer.analyze_project_metrics(job.start_date, job.end_date)

    async def _analyze_trends(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Analyze trends over the given period."""
        # Compare with previous period
        period_length = end_date - start_date
        prev_start = start_date - period_length
        prev_end = start_date

        # Current period metrics
        current_activity = await self._get_activity_count(start_date, end_date)
        current_users = await self._get_unique_users(start_date, end_date)
        current_projects = await self._get_active_projects(start_date, end_date)

        # Previous period metrics
        prev_activity = await self._get_activity_count(prev_start, prev_end)
        prev_users = await self._get_unique_users(prev_start, prev_end)
        prev_projects = await self._get_active_projects(prev_start, prev_end)

        trends = {
            'activity_trend': self._calculate_trend(prev_activity, current_activity),
            'user_trend': self._calculate_trend(prev_users, current_users),
            'project_trend': self._calculate_trend(prev_projects, current_projects)
        }

        return trends

    async def _get_activity_count(self, start_date: datetime, end_date: datetime) -> int:
        """Get total activity count for period."""
        query = """
        SELECT COUNT(*) as count
        FROM user_activities
        WHERE timestamp BETWEEN ? AND ?
        """
        result = await self.db_manager.execute_query(
            query, [start_date.isoformat(), end_date.isoformat()], prefer_analytics=True
        )
        return result[0]['count'] if result else 0

    async def _get_unique_users(self, start_date: datetime, end_date: datetime) -> int:
        """Get unique user count for period."""
        query = """
        SELECT COUNT(DISTINCT session_id) as count
        FROM user_activities
        WHERE timestamp BETWEEN ? AND ?
        """
        result = await self.db_manager.execute_query(
            query, [start_date.isoformat(), end_date.isoformat()], prefer_analytics=True
        )
        return result[0]['count'] if result else 0

    async def _get_active_projects(self, start_date: datetime, end_date: datetime) -> int:
        """Get active project count for period."""
        query = """
        SELECT COUNT(DISTINCT project_id) as count
        FROM user_activities
        WHERE timestamp BETWEEN ? AND ?
            AND project_id IS NOT NULL
        """
        result = await self.db_manager.execute_query(
            query, [start_date.isoformat(), end_date.isoformat()], prefer_analytics=True
        )
        return result[0]['count'] if result else 0

    def _calculate_trend(self, previous: int, current: int) -> Dict[str, Any]:
        """Calculate trend between two values."""
        if previous == 0:
            percentage_change = 100.0 if current > 0 else 0.0
        else:
            percentage_change = ((current - previous) / previous) * 100

        return {
            'previous_value': previous,
            'current_value': current,
            'absolute_change': current - previous,
            'percentage_change': round(percentage_change, 2),
            'direction': 'up' if current > previous else 'down' if current < previous else 'stable'
        }

    async def _store_analytics_results(self, job: AnalyticsJob):
        """Store analytics results in database."""
        try:
            date_bucket = job.start_date.date()
            time_period = 'daily' if job.job_type == AnalyticsJobType.DAILY_AGGREGATION else 'weekly'

            await self.db_manager.execute_write(
                """
                INSERT OR REPLACE INTO analytics_aggregates
                (aggregate_id, metric_name, time_period, date_bucket, metric_details, created_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                [
                    job.job_id,
                    job.job_type.value,
                    time_period,
                    date_bucket,
                    json.dumps(job.results)
                ]
            )

        except Exception as e:
            logger.error(f"Failed to store analytics results for {job.job_id}: {e}")

    # Public API methods

    async def get_status(self) -> Dict[str, Any]:
        """Get processor status."""
        return {
            "running": self.running,
            "current_jobs": len(self.current_jobs),
            "metrics": self.metrics.copy(),
            "job_details": [
                {
                    "job_id": job.job_id,
                    "job_type": job.job_type.value,
                    "status": job.status,
                    "progress": job.progress
                }
                for job in self.current_jobs.values()
            ]
        }

    async def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self.metrics.copy()

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        healthy = True
        issues = []

        if not self.running:
            healthy = False
            issues.append("Processor not running")

        # Check for stuck jobs
        current_time = time.time()
        for job in self.current_jobs.values():
            job_age = current_time - job.created_at.timestamp()
            if job_age > self.max_processing_time:
                issues.append(f"Long-running job detected: {job.job_id}")

        # Check recent activity
        if self.metrics['last_run']:
            last_run_age = datetime.now() - self.metrics['last_run']
            if last_run_age > timedelta(days=2):
                issues.append(f"No recent analytics runs: {last_run_age.days} days ago")

        return {
            "healthy": healthy,
            "issues": issues,
            "metrics": self.metrics.copy()
        }