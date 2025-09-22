"""
Pre-built Analytics Queries for Project QuickNav

This module provides optimized, ready-to-use analytics queries for common
business intelligence and machine learning use cases.

Query Categories:
- User Behavior Analytics
- Project Popularity and Trends
- Document Lifecycle Analysis
- Performance Metrics
- ML Feature Extraction
- Search Analytics
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)


class AnalyticsQueries:
    """Pre-built analytics queries optimized for both DuckDB and SQLite."""

    def __init__(self, db_manager):
        self.db = db_manager

    # =====================================================
    # USER BEHAVIOR ANALYTICS
    # =====================================================

    def get_user_engagement_metrics(self, days: int = 30) -> Dict[str, Any]:
        """
        Get comprehensive user engagement metrics.

        Returns:
            Dictionary with engagement statistics
        """
        query = """
        WITH daily_engagement AS (
            SELECT
                DATE(timestamp) as date,
                COUNT(DISTINCT session_id) as unique_users,
                COUNT(*) as total_activities,
                AVG(response_time_ms) as avg_response_time,
                COUNT(CASE WHEN activity_type = 'navigate' THEN 1 END) as navigations,
                COUNT(CASE WHEN activity_type = 'search' THEN 1 END) as searches,
                COUNT(CASE WHEN activity_type = 'ai_query' THEN 1 END) as ai_queries,
                COUNT(CASE WHEN activity_type = 'document_open' THEN 1 END) as doc_opens
            FROM user_activities
            WHERE timestamp > datetime('now', '-{} days')
            GROUP BY DATE(timestamp)
        ),
        session_stats AS (
            SELECT
                AVG(session_duration_seconds) as avg_session_duration,
                AVG(total_navigations) as avg_navigations_per_session,
                AVG(ai_interactions) as avg_ai_interactions_per_session
            FROM user_sessions
            WHERE session_start > datetime('now', '-{} days')
        )
        SELECT
            COUNT(*) as total_days,
            AVG(unique_users) as avg_daily_users,
            SUM(total_activities) as total_activities,
            AVG(avg_response_time) as overall_avg_response_time,
            SUM(navigations) as total_navigations,
            SUM(searches) as total_searches,
            SUM(ai_queries) as total_ai_queries,
            SUM(doc_opens) as total_doc_opens,
            (SELECT avg_session_duration FROM session_stats) as avg_session_duration,
            (SELECT avg_navigations_per_session FROM session_stats) as avg_navigations_per_session,
            (SELECT avg_ai_interactions_per_session FROM session_stats) as avg_ai_interactions_per_session
        FROM daily_engagement
        """.format(days, days)

        result = self.db.execute_query(query, prefer_analytics=True)
        return result[0] if result else {}

    def get_user_activity_heatmap(self, days: int = 30) -> List[Dict]:
        """
        Get activity heatmap data by hour and day of week.

        Returns:
            List of activity counts by time periods
        """
        query = """
        SELECT
            CAST(strftime('%w', timestamp) AS INTEGER) as day_of_week,  -- 0=Sunday
            CAST(strftime('%H', timestamp) AS INTEGER) as hour_of_day,
            COUNT(*) as activity_count,
            COUNT(DISTINCT session_id) as unique_users
        FROM user_activities
        WHERE timestamp > datetime('now', '-{} days')
        GROUP BY day_of_week, hour_of_day
        ORDER BY day_of_week, hour_of_day
        """.format(days)

        return self.db.execute_query(query, prefer_analytics=True)

    def get_user_journey_analysis(self, session_id: str = None, limit: int = 100) -> List[Dict]:
        """
        Analyze user journey patterns.

        Args:
            session_id: Specific session to analyze, or None for all sessions
            limit: Maximum number of sessions to analyze

        Returns:
            List of journey steps with timing and context
        """
        base_query = """
        SELECT
            ua.session_id,
            ua.timestamp,
            ua.activity_type,
            ua.project_id,
            ua.search_query,
            ua.input_method,
            ua.ui_component,
            ua.response_time_ms,
            p.project_name,
            ROW_NUMBER() OVER (PARTITION BY ua.session_id ORDER BY ua.timestamp) as step_number,
            LAG(ua.activity_type) OVER (PARTITION BY ua.session_id ORDER BY ua.timestamp) as prev_activity,
            CAST((julianday(ua.timestamp) - julianday(LAG(ua.timestamp) OVER (PARTITION BY ua.session_id ORDER BY ua.timestamp))) * 86400 AS INTEGER) as seconds_since_prev
        FROM user_activities ua
        LEFT JOIN projects p ON ua.project_id = p.project_id
        WHERE ua.timestamp > datetime('now', '-30 days')
        """

        if session_id:
            query = base_query + f" AND ua.session_id = '{session_id}'"
        else:
            query = base_query + f"""
            AND ua.session_id IN (
                SELECT session_id FROM user_sessions
                WHERE session_start > datetime('now', '-30 days')
                ORDER BY session_start DESC
                LIMIT {limit}
            )
            """

        query += " ORDER BY ua.session_id, ua.timestamp"

        return self.db.execute_query(query, prefer_analytics=True)

    # =====================================================
    # PROJECT POPULARITY AND TRENDS
    # =====================================================

    def get_project_popularity_trends(self, weeks: int = 12) -> List[Dict]:
        """
        Get project popularity trends over time.

        Args:
            weeks: Number of weeks to analyze

        Returns:
            Weekly access counts by project
        """
        query = """
        SELECT
            p.project_code,
            p.project_name,
            strftime('%Y-%W', ua.timestamp) as year_week,
            COUNT(*) as weekly_access_count,
            COUNT(DISTINCT ua.session_id) as unique_users,
            AVG(ua.response_time_ms) as avg_response_time
        FROM user_activities ua
        JOIN projects p ON ua.project_id = p.project_id
        WHERE ua.timestamp > datetime('now', '-{} days')
            AND ua.activity_type IN ('navigate', 'document_open')
        GROUP BY p.project_id, p.project_code, p.project_name, strftime('%Y-%W', ua.timestamp)
        HAVING weekly_access_count > 1  -- Filter noise
        ORDER BY p.project_code, year_week DESC
        """.format(weeks * 7)

        return self.db.execute_query(query, prefer_analytics=True)

    def get_project_abandonment_analysis(self, days: int = 30) -> List[Dict]:
        """
        Analyze projects that users navigate to but don't engage with deeply.

        Returns:
            Projects with high navigation but low document access
        """
        query = """
        WITH project_metrics AS (
            SELECT
                p.project_id,
                p.project_code,
                p.project_name,
                COUNT(CASE WHEN ua.activity_type = 'navigate' THEN 1 END) as navigation_count,
                COUNT(CASE WHEN ua.activity_type = 'document_open' THEN 1 END) as document_opens,
                COUNT(DISTINCT ua.session_id) as unique_visitors,
                AVG(ua.response_time_ms) as avg_response_time
            FROM user_activities ua
            JOIN projects p ON ua.project_id = p.project_id
            WHERE ua.timestamp > datetime('now', '-{} days')
            GROUP BY p.project_id, p.project_code, p.project_name
            HAVING navigation_count > 0
        )
        SELECT
            *,
            CASE
                WHEN navigation_count > 0 THEN CAST(document_opens AS FLOAT) / navigation_count
                ELSE 0
            END as engagement_ratio,
            CASE
                WHEN navigation_count > 5 AND (CAST(document_opens AS FLOAT) / navigation_count) < 0.3 THEN 'High Abandonment'
                WHEN navigation_count > 2 AND (CAST(document_opens AS FLOAT) / navigation_count) < 0.5 THEN 'Medium Abandonment'
                ELSE 'Good Engagement'
            END as engagement_category
        FROM project_metrics
        ORDER BY navigation_count DESC, engagement_ratio ASC
        """.format(days)

        return self.db.execute_query(query, prefer_analytics=True)

    # =====================================================
    # DOCUMENT LIFECYCLE ANALYSIS
    # =====================================================

    def get_document_version_analysis(self, project_id: str = None) -> List[Dict]:
        """
        Analyze document version patterns and evolution.

        Args:
            project_id: Specific project to analyze, or None for all projects

        Returns:
            Document version statistics and patterns
        """
        base_query = """
        WITH version_stats AS (
            SELECT
                project_id,
                document_type,
                COUNT(*) as total_versions,
                MAX(version_numeric) as latest_version,
                MIN(version_numeric) as first_version,
                COUNT(CASE WHEN status_tags LIKE '%AS-BUILT%' THEN 1 END) as as_built_count,
                COUNT(CASE WHEN status_tags LIKE '%FINAL%' THEN 1 END) as final_count,
                COUNT(CASE WHEN status_tags LIKE '%DRAFT%' THEN 1 END) as draft_count,
                AVG(CASE WHEN word_count > 0 THEN word_count END) as avg_word_count,
                MAX(document_date) as latest_document_date,
                MIN(document_date) as first_document_date
            FROM documents
            WHERE document_type IS NOT NULL
        """

        if project_id:
            query = base_query + f" AND project_id = '{project_id}'"

        query += """
            GROUP BY project_id, document_type
        ),
        project_names AS (
            SELECT project_id, project_name FROM projects
        )
        SELECT
            vs.*,
            pn.project_name,
            CASE
                WHEN latest_version > first_version THEN latest_version - first_version
                ELSE 0
            END as version_iterations,
            CASE
                WHEN latest_document_date IS NOT NULL AND first_document_date IS NOT NULL
                THEN CAST((julianday(latest_document_date) - julianday(first_document_date)) AS INTEGER)
                ELSE 0
            END as development_days
        FROM version_stats vs
        LEFT JOIN project_names pn ON vs.project_id = pn.project_id
        ORDER BY vs.project_id, vs.document_type
        """

        return self.db.execute_query(query, prefer_analytics=True)

    def get_document_access_patterns(self, days: int = 30) -> List[Dict]:
        """
        Analyze which document types are accessed most frequently.

        Returns:
            Document type access statistics
        """
        query = """
        SELECT
            d.document_type,
            d.folder_category,
            COUNT(*) as access_count,
            COUNT(DISTINCT ua.session_id) as unique_users,
            COUNT(DISTINCT d.project_id) as projects_with_type,
            AVG(ua.response_time_ms) as avg_access_time,
            MAX(ua.timestamp) as last_accessed,
            ROUND(AVG(d.status_weight), 2) as avg_status_weight
        FROM user_activities ua
        JOIN documents d ON ua.document_id = d.document_id
        WHERE ua.activity_type = 'document_open'
            AND ua.timestamp > datetime('now', '-{} days')
            AND d.document_type IS NOT NULL
        GROUP BY d.document_type, d.folder_category
        ORDER BY access_count DESC
        """.format(days)

        return self.db.execute_query(query, prefer_analytics=True)

    # =====================================================
    # PERFORMANCE METRICS
    # =====================================================

    def get_performance_bottlenecks(self, days: int = 7) -> List[Dict]:
        """
        Identify performance bottlenecks in the system.

        Returns:
            Operations with high response times
        """
        query = """
        WITH percentiles AS (
            SELECT
                activity_type,
                ui_component,
                COUNT(*) as operation_count,
                AVG(response_time_ms) as avg_response_time,
                MIN(response_time_ms) as min_response_time,
                MAX(response_time_ms) as max_response_time,
                -- Approximate percentiles using ORDER BY and LIMIT
                (SELECT response_time_ms FROM user_activities sub
                 WHERE sub.activity_type = ua.activity_type
                   AND sub.ui_component = ua.ui_component
                   AND sub.response_time_ms IS NOT NULL
                   AND sub.timestamp > datetime('now', '-{} days')
                 ORDER BY response_time_ms
                 LIMIT 1 OFFSET (COUNT(*) * 95 / 100)) as p95_response_time,
                COUNT(CASE WHEN success = 0 THEN 1 END) as error_count,
                ROUND(COUNT(CASE WHEN success = 0 THEN 1 END) * 100.0 / COUNT(*), 2) as error_rate
            FROM user_activities ua
            WHERE timestamp > datetime('now', '-{} days')
                AND response_time_ms IS NOT NULL
            GROUP BY activity_type, ui_component
            HAVING operation_count > 5  -- Filter low-volume operations
        )
        SELECT
            *,
            CASE
                WHEN avg_response_time > 2000 THEN 'Critical'
                WHEN avg_response_time > 1000 THEN 'Warning'
                WHEN avg_response_time > 500 THEN 'Monitor'
                ELSE 'Good'
            END as performance_status
        FROM percentiles
        ORDER BY avg_response_time DESC, operation_count DESC
        """.format(days, days)

        return self.db.execute_query(query, prefer_analytics=True)

    def get_cache_effectiveness(self, days: int = 7) -> List[Dict]:
        """
        Analyze cache hit rates and effectiveness.

        Returns:
            Cache performance metrics
        """
        query = """
        SELECT
            query_type,
            COUNT(*) as total_searches,
            SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END) as cache_hits,
            ROUND(SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as cache_hit_rate,
            AVG(CASE WHEN cache_hit THEN search_time_ms END) as avg_cached_time,
            AVG(CASE WHEN NOT cache_hit THEN search_time_ms END) as avg_uncached_time,
            AVG(results_count) as avg_results_count
        FROM search_analytics
        WHERE timestamp > datetime('now', '-{} days')
        GROUP BY query_type
        ORDER BY total_searches DESC
        """.format(days)

        return self.db.execute_query(query, prefer_analytics=True)

    # =====================================================
    # ML FEATURE EXTRACTION
    # =====================================================

    def get_user_similarity_features(self, days: int = 90) -> List[Dict]:
        """
        Extract features for user similarity analysis.

        Returns:
            User behavior feature vectors
        """
        query = """
        WITH user_features AS (
            SELECT
                session_id as user_proxy,
                -- Activity type distribution
                COUNT(CASE WHEN activity_type = 'navigate' THEN 1 END) as navigate_count,
                COUNT(CASE WHEN activity_type = 'search' THEN 1 END) as search_count,
                COUNT(CASE WHEN activity_type = 'ai_query' THEN 1 END) as ai_query_count,
                COUNT(CASE WHEN activity_type = 'document_open' THEN 1 END) as document_open_count,

                -- Timing patterns
                AVG(response_time_ms) as avg_response_time,
                COUNT(*) as total_activities,

                -- Input method preferences
                COUNT(CASE WHEN input_method = 'keyboard' THEN 1 END) as keyboard_usage,
                COUNT(CASE WHEN input_method = 'hotkey' THEN 1 END) as hotkey_usage,
                COUNT(CASE WHEN input_method = 'ai_chat' THEN 1 END) as ai_chat_usage,

                -- UI component usage
                COUNT(CASE WHEN ui_component = 'main_search' THEN 1 END) as main_search_usage,
                COUNT(CASE WHEN ui_component = 'ai_panel' THEN 1 END) as ai_panel_usage,

                -- Temporal patterns
                COUNT(CASE WHEN CAST(strftime('%H', timestamp) AS INTEGER) BETWEEN 6 AND 12 THEN 1 END) as morning_usage,
                COUNT(CASE WHEN CAST(strftime('%H', timestamp) AS INTEGER) BETWEEN 12 AND 18 THEN 1 END) as afternoon_usage,
                COUNT(CASE WHEN CAST(strftime('%H', timestamp) AS INTEGER) BETWEEN 18 AND 24 THEN 1 END) as evening_usage,

                -- Project diversity
                COUNT(DISTINCT project_id) as unique_projects,

                -- Error patterns
                COUNT(CASE WHEN success = 0 THEN 1 END) as error_count

            FROM user_activities
            WHERE timestamp > datetime('now', '-{} days')
                AND session_id IS NOT NULL
            GROUP BY session_id
            HAVING total_activities > 10  -- Filter low-activity users
        )
        SELECT
            *,
            -- Normalized features (0-1 scale)
            CASE WHEN total_activities > 0 THEN CAST(navigate_count AS FLOAT) / total_activities ELSE 0 END as navigate_ratio,
            CASE WHEN total_activities > 0 THEN CAST(search_count AS FLOAT) / total_activities ELSE 0 END as search_ratio,
            CASE WHEN total_activities > 0 THEN CAST(ai_query_count AS FLOAT) / total_activities ELSE 0 END as ai_ratio,
            CASE WHEN total_activities > 0 THEN CAST(document_open_count AS FLOAT) / total_activities ELSE 0 END as document_ratio,
            CASE WHEN total_activities > 0 THEN CAST(error_count AS FLOAT) / total_activities ELSE 0 END as error_ratio
        FROM user_features
        ORDER BY total_activities DESC
        """.format(days)

        return self.db.execute_query(query, prefer_analytics=True)

    def get_project_similarity_features(self) -> List[Dict]:
        """
        Extract features for project similarity analysis.

        Returns:
            Project feature vectors based on document types and user interactions
        """
        query = """
        WITH project_features AS (
            SELECT
                p.project_id,
                p.project_code,
                p.project_name,

                -- Document type distribution
                COUNT(CASE WHEN d.document_type = 'lld' THEN 1 END) as lld_count,
                COUNT(CASE WHEN d.document_type = 'hld' THEN 1 END) as hld_count,
                COUNT(CASE WHEN d.document_type = 'co' THEN 1 END) as co_count,
                COUNT(CASE WHEN d.document_type = 'floor_plan' THEN 1 END) as floor_plan_count,
                COUNT(CASE WHEN d.document_type = 'scope' THEN 1 END) as scope_count,

                -- Folder category distribution
                COUNT(CASE WHEN d.folder_category = 'System Designs' THEN 1 END) as system_designs_count,
                COUNT(CASE WHEN d.folder_category = 'Sales Handover' THEN 1 END) as sales_handover_count,
                COUNT(CASE WHEN d.folder_category = 'BOM & Orders' THEN 1 END) as bom_orders_count,

                -- Document maturity indicators
                AVG(CASE WHEN d.status_weight IS NOT NULL THEN d.status_weight ELSE 0 END) as avg_status_weight,
                COUNT(CASE WHEN d.status_tags LIKE '%AS-BUILT%' THEN 1 END) as as_built_count,
                MAX(COALESCE(d.version_numeric, 0)) as max_version,

                -- User interaction metrics (last 90 days)
                COUNT(CASE WHEN ua.timestamp > datetime('now', '-90 days') THEN 1 END) as recent_interactions,
                COUNT(DISTINCT CASE WHEN ua.timestamp > datetime('now', '-90 days') THEN ua.session_id END) as unique_recent_users,
                AVG(CASE WHEN ua.timestamp > datetime('now', '-90 days') THEN ua.response_time_ms END) as avg_interaction_time,

                -- Content metrics
                AVG(CASE WHEN d.word_count > 0 THEN d.word_count END) as avg_word_count,
                COUNT(d.document_id) as total_documents

            FROM projects p
            LEFT JOIN documents d ON p.project_id = d.project_id
            LEFT JOIN user_activities ua ON p.project_id = ua.project_id
            GROUP BY p.project_id, p.project_code, p.project_name
        )
        SELECT
            *,
            -- Normalized features
            CASE WHEN total_documents > 0 THEN CAST(lld_count AS FLOAT) / total_documents ELSE 0 END as lld_ratio,
            CASE WHEN total_documents > 0 THEN CAST(hld_count AS FLOAT) / total_documents ELSE 0 END as hld_ratio,
            CASE WHEN total_documents > 0 THEN CAST(co_count AS FLOAT) / total_documents ELSE 0 END as co_ratio,
            CASE WHEN total_documents > 0 THEN CAST(as_built_count AS FLOAT) / total_documents ELSE 0 END as completion_ratio
        FROM project_features
        WHERE total_documents > 0  -- Only projects with documents
        ORDER BY recent_interactions DESC, total_documents DESC
        """

        return self.db.execute_query(query, prefer_analytics=True)

    # =====================================================
    # SEARCH ANALYTICS
    # =====================================================

    def get_search_query_analysis(self, days: int = 30) -> List[Dict]:
        """
        Analyze search query patterns and effectiveness.

        Returns:
            Search query statistics and patterns
        """
        query = """
        WITH query_analysis AS (
            SELECT
                query_text,
                query_type,
                COUNT(*) as search_frequency,
                AVG(results_count) as avg_results,
                AVG(search_time_ms) as avg_search_time,
                COUNT(CASE WHEN selected_result_index IS NOT NULL THEN 1 END) as selections_made,
                COUNT(CASE WHEN results_count = 0 THEN 1 END) as zero_result_searches,
                MAX(timestamp) as last_search_time,
                COUNT(DISTINCT session_id) as unique_users
            FROM search_analytics
            WHERE timestamp > datetime('now', '-{} days')
            GROUP BY query_text, query_type
        )
        SELECT
            *,
            CASE
                WHEN search_frequency > 0 THEN CAST(selections_made AS FLOAT) / search_frequency
                ELSE 0
            END as selection_rate,
            CASE
                WHEN search_frequency > 0 THEN CAST(zero_result_searches AS FLOAT) / search_frequency
                ELSE 0
            END as zero_result_rate,
            CASE
                WHEN zero_result_rate > 0.5 THEN 'Poor'
                WHEN selection_rate < 0.3 THEN 'Needs Improvement'
                WHEN selection_rate > 0.7 THEN 'Excellent'
                ELSE 'Good'
            END as search_quality
        FROM query_analysis
        ORDER BY search_frequency DESC, avg_search_time ASC
        """.format(days)

        return self.db.execute_query(query, prefer_analytics=True)

    def get_failed_searches(self, days: int = 7) -> List[Dict]:
        """
        Identify searches that returned no results or weren't acted upon.

        Returns:
            Failed or unsuccessful search attempts
        """
        query = """
        SELECT
            query_text,
            query_type,
            COUNT(*) as failure_count,
            AVG(search_time_ms) as avg_search_time,
            MAX(timestamp) as last_failure,
            COUNT(DISTINCT session_id) as affected_users,
            -- Attempt to categorize failure types
            CASE
                WHEN AVG(results_count) = 0 THEN 'No Results'
                WHEN AVG(CASE WHEN selected_result_index IS NULL THEN 1.0 ELSE 0.0 END) > 0.8 THEN 'No Selection Made'
                WHEN AVG(search_time_ms) > 5000 THEN 'Slow Search'
                ELSE 'Other'
            END as failure_type
        FROM search_analytics
        WHERE timestamp > datetime('now', '-{} days')
            AND (results_count = 0 OR selected_result_index IS NULL OR search_time_ms > 5000)
        GROUP BY query_text, query_type
        HAVING failure_count > 1  -- Only repeated failures
        ORDER BY failure_count DESC, affected_users DESC
        """.format(days)

        return self.db.execute_query(query, prefer_analytics=True)

    # =====================================================
    # REPORTING UTILITIES
    # =====================================================

    def generate_daily_summary(self, date: str = None) -> Dict[str, Any]:
        """
        Generate a comprehensive daily summary report.

        Args:
            date: Date in YYYY-MM-DD format, defaults to yesterday

        Returns:
            Dictionary with daily metrics
        """
        if not date:
            # Default to yesterday
            yesterday = datetime.now() - timedelta(days=1)
            date = yesterday.strftime('%Y-%m-%d')

        # User Activity Summary
        activity_query = """
        SELECT
            COUNT(DISTINCT session_id) as unique_users,
            COUNT(*) as total_activities,
            AVG(response_time_ms) as avg_response_time,
            COUNT(CASE WHEN activity_type = 'navigate' THEN 1 END) as navigations,
            COUNT(CASE WHEN activity_type = 'search' THEN 1 END) as searches,
            COUNT(CASE WHEN activity_type = 'ai_query' THEN 1 END) as ai_queries,
            COUNT(CASE WHEN activity_type = 'document_open' THEN 1 END) as document_opens,
            COUNT(CASE WHEN success = 0 THEN 1 END) as errors
        FROM user_activities
        WHERE DATE(timestamp) = ?
        """

        # Project Access Summary
        project_query = """
        SELECT
            COUNT(DISTINCT project_id) as unique_projects_accessed,
            project_id as most_accessed_project,
            COUNT(*) as access_count
        FROM user_activities
        WHERE DATE(timestamp) = ? AND project_id IS NOT NULL
        GROUP BY project_id
        ORDER BY COUNT(*) DESC
        LIMIT 1
        """

        # AI Usage Summary
        ai_query = """
        SELECT
            COUNT(DISTINCT conversation_id) as conversations,
            COUNT(*) as total_messages,
            SUM(token_count) as total_tokens,
            AVG(response_time_ms) as avg_ai_response_time
        FROM ai_messages
        WHERE DATE(timestamp) = ?
        """

        activity_result = self.db.execute_query(activity_query, [date])
        project_result = self.db.execute_query(project_query, [date])
        ai_result = self.db.execute_query(ai_query, [date])

        return {
            'date': date,
            'user_activity': activity_result[0] if activity_result else {},
            'top_project': project_result[0] if project_result else {},
            'ai_usage': ai_result[0] if ai_result else {},
            'generated_at': datetime.now().isoformat()
        }

    def export_analytics_data(self, start_date: str, end_date: str,
                             tables: List[str] = None) -> Dict[str, List]:
        """
        Export analytics data for external analysis.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            tables: List of table names to export, defaults to main analytics tables

        Returns:
            Dictionary with table names as keys and data as values
        """
        if not tables:
            tables = ['user_activities', 'search_analytics', 'ai_messages']

        export_data = {}

        for table in tables:
            query = f"""
            SELECT * FROM {table}
            WHERE DATE(timestamp) BETWEEN ? AND ?
            ORDER BY timestamp
            """

            try:
                data = self.db.execute_query(query, [start_date, end_date], prefer_analytics=True)
                export_data[table] = data
                logger.info(f"Exported {len(data)} records from {table}")
            except Exception as e:
                logger.error(f"Failed to export data from {table}: {e}")
                export_data[table] = []

        return export_data