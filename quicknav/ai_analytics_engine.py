"""
AI-Powered Analytics Engine for Project QuickNav

Provides intelligent analytics, insights, and predictive capabilities for
project navigation patterns, document usage, and user behavior analysis.

Features:
- Usage pattern analysis and trend detection
- Predictive project and document recommendations
- Natural language insight generation
- Performance metrics and optimization suggestions
- User behavior modeling and personalization
- Real-time analytics dashboard data
"""

import json
import logging
import asyncio
import sqlite3
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import statistics

logger = logging.getLogger(__name__)

try:
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn/pandas not available. Advanced analytics features will be disabled.")
    SKLEARN_AVAILABLE = False


@dataclass
class UsageEvent:
    """Represents a user interaction event."""
    event_id: str
    user_id: str
    event_type: str  # 'project_access', 'document_view', 'search_query', etc.
    timestamp: datetime
    project_id: Optional[str]
    document_path: Optional[str]
    query_text: Optional[str]
    success: bool
    duration_seconds: float
    metadata: Dict[str, Any]


@dataclass
class ProjectInsight:
    """Represents an insight about a project."""
    project_id: str
    insight_type: str  # 'usage_pattern', 'document_gap', 'anomaly', etc.
    title: str
    description: str
    confidence: float
    priority: str  # 'high', 'medium', 'low'
    actionable: bool
    recommendations: List[str]
    metadata: Dict[str, Any]


@dataclass
class UserBehaviorProfile:
    """Represents a user's behavior profile."""
    user_id: str
    primary_activities: List[str]
    preferred_project_types: List[str]
    active_hours: List[int]  # Hours of day when most active
    avg_session_duration: float
    query_complexity: str  # 'simple', 'moderate', 'complex'
    expertise_level: str  # 'beginner', 'intermediate', 'expert'
    personalization_factors: Dict[str, float]


class AnalyticsDatabase:
    """Database for storing and retrieving analytics data."""

    def __init__(self, db_path: str = "data/analytics.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._initialize_tables()

    def _initialize_tables(self):
        """Initialize database tables."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS usage_events (
                event_id TEXT PRIMARY KEY,
                user_id TEXT,
                event_type TEXT,
                timestamp TEXT,
                project_id TEXT,
                document_path TEXT,
                query_text TEXT,
                success BOOLEAN,
                duration_seconds REAL,
                metadata TEXT
            );

            CREATE TABLE IF NOT EXISTS project_insights (
                insight_id TEXT PRIMARY KEY,
                project_id TEXT,
                insight_type TEXT,
                title TEXT,
                description TEXT,
                confidence REAL,
                priority TEXT,
                actionable BOOLEAN,
                recommendations TEXT,
                metadata TEXT,
                created_at TEXT
            );

            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                profile_data TEXT,
                created_at TEXT,
                updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS analytics_cache (
                cache_key TEXT PRIMARY KEY,
                cache_data TEXT,
                expires_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_events_user_time ON usage_events(user_id, timestamp);
            CREATE INDEX IF NOT EXISTS idx_events_project ON usage_events(project_id);
            CREATE INDEX IF NOT EXISTS idx_insights_project ON project_insights(project_id);
        """)
        self.conn.commit()

    def log_event(self, event: UsageEvent):
        """Log a usage event."""
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO usage_events
                (event_id, user_id, event_type, timestamp, project_id, document_path,
                 query_text, success, duration_seconds, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_id, event.user_id, event.event_type,
                event.timestamp.isoformat(), event.project_id, event.document_path,
                event.query_text, event.success, event.duration_seconds,
                json.dumps(event.metadata)
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Failed to log event: {e}")

    def get_events(self, user_id: Optional[str] = None, event_type: Optional[str] = None,
                   start_date: Optional[datetime] = None, end_date: Optional[datetime] = None,
                   limit: int = 1000) -> List[UsageEvent]:
        """Retrieve usage events with filters."""
        query = "SELECT * FROM usage_events WHERE 1=1"
        params = []

        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        try:
            cursor = self.conn.execute(query, params)
            events = []

            for row in cursor.fetchall():
                event = UsageEvent(
                    event_id=row[0],
                    user_id=row[1],
                    event_type=row[2],
                    timestamp=datetime.fromisoformat(row[3]),
                    project_id=row[4],
                    document_path=row[5],
                    query_text=row[6],
                    success=bool(row[7]),
                    duration_seconds=row[8],
                    metadata=json.loads(row[9]) if row[9] else {}
                )
                events.append(event)

            return events

        except Exception as e:
            logger.error(f"Failed to get events: {e}")
            return []

    def store_insight(self, insight: ProjectInsight):
        """Store a project insight."""
        try:
            insight_id = f"{insight.project_id}_{insight.insight_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            self.conn.execute("""
                INSERT OR REPLACE INTO project_insights
                (insight_id, project_id, insight_type, title, description,
                 confidence, priority, actionable, recommendations, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                insight_id, insight.project_id, insight.insight_type,
                insight.title, insight.description, insight.confidence,
                insight.priority, insight.actionable,
                json.dumps(insight.recommendations), json.dumps(insight.metadata),
                datetime.now().isoformat()
            ))
            self.conn.commit()

        except Exception as e:
            logger.error(f"Failed to store insight: {e}")

    def get_insights(self, project_id: Optional[str] = None,
                    insight_type: Optional[str] = None) -> List[ProjectInsight]:
        """Get project insights."""
        query = "SELECT * FROM project_insights WHERE 1=1"
        params = []

        if project_id:
            query += " AND project_id = ?"
            params.append(project_id)

        if insight_type:
            query += " AND insight_type = ?"
            params.append(insight_type)

        query += " ORDER BY confidence DESC, created_at DESC"

        try:
            cursor = self.conn.execute(query, params)
            insights = []

            for row in cursor.fetchall():
                insight = ProjectInsight(
                    project_id=row[1],
                    insight_type=row[2],
                    title=row[3],
                    description=row[4],
                    confidence=row[5],
                    priority=row[6],
                    actionable=bool(row[7]),
                    recommendations=json.loads(row[8]) if row[8] else [],
                    metadata=json.loads(row[9]) if row[9] else {}
                )
                insights.append(insight)

            return insights

        except Exception as e:
            logger.error(f"Failed to get insights: {e}")
            return []


class PatternAnalyzer:
    """Analyzes usage patterns and trends."""

    def __init__(self, db: AnalyticsDatabase):
        self.db = db

    def analyze_project_usage_patterns(self, days: int = 30) -> Dict[str, Any]:
        """Analyze project usage patterns over time."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        events = self.db.get_events(
            event_type='project_access',
            start_date=start_date,
            end_date=end_date
        )

        if not events:
            return {"error": "No project access data available"}

        # Analyze patterns
        patterns = {
            "total_accesses": len(events),
            "unique_projects": len(set(e.project_id for e in events if e.project_id)),
            "daily_activity": self._analyze_daily_activity(events),
            "project_popularity": self._analyze_project_popularity(events),
            "user_activity": self._analyze_user_activity(events),
            "success_rate": sum(1 for e in events if e.success) / len(events) if events else 0,
            "avg_session_duration": statistics.mean(e.duration_seconds for e in events) if events else 0
        }

        # Detect trends
        patterns["trends"] = self._detect_trends(events, days)

        return patterns

    def analyze_search_patterns(self, days: int = 30) -> Dict[str, Any]:
        """Analyze search query patterns."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        events = self.db.get_events(
            event_type='search_query',
            start_date=start_date,
            end_date=end_date
        )

        if not events:
            return {"error": "No search data available"}

        patterns = {
            "total_searches": len(events),
            "unique_queries": len(set(e.query_text for e in events if e.query_text)),
            "success_rate": sum(1 for e in events if e.success) / len(events),
            "popular_queries": self._analyze_popular_queries(events),
            "query_complexity": self._analyze_query_complexity(events),
            "search_trends": self._analyze_search_trends(events)
        }

        return patterns

    def _analyze_daily_activity(self, events: List[UsageEvent]) -> Dict[str, Any]:
        """Analyze daily activity patterns."""
        daily_counts = defaultdict(int)
        hourly_counts = defaultdict(int)

        for event in events:
            day = event.timestamp.strftime('%Y-%m-%d')
            hour = event.timestamp.hour

            daily_counts[day] += 1
            hourly_counts[hour] += 1

        return {
            "daily_counts": dict(daily_counts),
            "hourly_distribution": dict(hourly_counts),
            "peak_hour": max(hourly_counts, key=hourly_counts.get) if hourly_counts else None,
            "busiest_day": max(daily_counts, key=daily_counts.get) if daily_counts else None
        }

    def _analyze_project_popularity(self, events: List[UsageEvent]) -> Dict[str, Any]:
        """Analyze project popularity."""
        project_counts = Counter(e.project_id for e in events if e.project_id)

        return {
            "top_projects": dict(project_counts.most_common(10)),
            "total_unique_projects": len(project_counts),
            "access_distribution": {
                "high_activity": len([p for p, c in project_counts.items() if c > 10]),
                "medium_activity": len([p for p, c in project_counts.items() if 3 <= c <= 10]),
                "low_activity": len([p for p, c in project_counts.items() if c < 3])
            }
        }

    def _analyze_user_activity(self, events: List[UsageEvent]) -> Dict[str, Any]:
        """Analyze user activity patterns."""
        user_counts = Counter(e.user_id for e in events)
        user_success_rates = defaultdict(list)

        for event in events:
            user_success_rates[event.user_id].append(event.success)

        success_rates = {
            user: sum(successes) / len(successes)
            for user, successes in user_success_rates.items()
        }

        return {
            "active_users": len(user_counts),
            "user_activity_distribution": dict(user_counts.most_common(10)),
            "user_success_rates": success_rates,
            "avg_success_rate": statistics.mean(success_rates.values()) if success_rates else 0
        }

    def _analyze_popular_queries(self, events: List[UsageEvent]) -> Dict[str, Any]:
        """Analyze popular search queries."""
        query_counts = Counter(e.query_text for e in events if e.query_text)
        query_success = defaultdict(list)

        for event in events:
            if event.query_text:
                query_success[event.query_text].append(event.success)

        successful_queries = {
            query: sum(successes) / len(successes)
            for query, successes in query_success.items()
        }

        return {
            "most_common_queries": dict(query_counts.most_common(10)),
            "highest_success_rate_queries": dict(
                sorted(successful_queries.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
            "lowest_success_rate_queries": dict(
                sorted(successful_queries.items(), key=lambda x: x[1])[:10]
            )
        }

    def _analyze_query_complexity(self, events: List[UsageEvent]) -> Dict[str, Any]:
        """Analyze query complexity patterns."""
        complexities = []

        for event in events:
            if event.query_text:
                # Simple complexity measure: word count + presence of numbers/quotes
                word_count = len(event.query_text.split())
                has_numbers = any(c.isdigit() for c in event.query_text)
                has_quotes = '"' in event.query_text or "'" in event.query_text

                complexity = word_count + (2 if has_numbers else 0) + (2 if has_quotes else 0)
                complexities.append(complexity)

        if not complexities:
            return {}

        return {
            "avg_complexity": statistics.mean(complexities),
            "complexity_distribution": {
                "simple": len([c for c in complexities if c <= 3]),
                "moderate": len([c for c in complexities if 4 <= c <= 7]),
                "complex": len([c for c in complexities if c > 7])
            }
        }

    def _detect_trends(self, events: List[UsageEvent], days: int) -> Dict[str, Any]:
        """Detect usage trends."""
        # Group events by day
        daily_events = defaultdict(list)
        for event in events:
            day = event.timestamp.strftime('%Y-%m-%d')
            daily_events[day].append(event)

        # Calculate daily metrics
        daily_metrics = []
        for day in sorted(daily_events.keys()):
            day_events = daily_events[day]
            metrics = {
                'date': day,
                'count': len(day_events),
                'success_rate': sum(1 for e in day_events if e.success) / len(day_events),
                'avg_duration': statistics.mean(e.duration_seconds for e in day_events)
            }
            daily_metrics.append(metrics)

        # Detect trends
        trends = {}

        if len(daily_metrics) >= 7:  # Need at least a week of data
            counts = [m['count'] for m in daily_metrics]
            success_rates = [m['success_rate'] for m in daily_metrics]

            # Simple trend detection using correlation with time
            time_points = list(range(len(counts)))

            # Calculate correlation between time and metrics
            if SKLEARN_AVAILABLE:
                count_trend = np.corrcoef(time_points, counts)[0, 1]
                success_trend = np.corrcoef(time_points, success_rates)[0, 1]

                trends = {
                    'usage_trend': 'increasing' if count_trend > 0.3 else 'decreasing' if count_trend < -0.3 else 'stable',
                    'success_trend': 'improving' if success_trend > 0.3 else 'declining' if success_trend < -0.3 else 'stable',
                    'trend_strength': abs(count_trend)
                }

        return trends

    def _analyze_search_trends(self, events: List[UsageEvent]) -> Dict[str, Any]:
        """Analyze search-specific trends."""
        # Analyze query evolution over time
        query_terms = defaultdict(int)

        for event in events:
            if event.query_text:
                terms = event.query_text.lower().split()
                for term in terms:
                    if len(term) > 2:  # Ignore very short terms
                        query_terms[term] += 1

        return {
            "trending_terms": dict(Counter(query_terms).most_common(20)),
            "search_evolution": "More analysis needed with larger dataset"
        }


class InsightGenerator:
    """Generates actionable insights from analytics data."""

    def __init__(self, db: AnalyticsDatabase, pattern_analyzer: PatternAnalyzer):
        self.db = db
        self.pattern_analyzer = pattern_analyzer

    async def generate_project_insights(self, project_id: str) -> List[ProjectInsight]:
        """Generate insights for a specific project."""
        insights = []

        # Get project events
        events = self.db.get_events(project_id=project_id, limit=500)

        if not events:
            return insights

        # Usage pattern insights
        usage_insight = self._analyze_project_usage(project_id, events)
        if usage_insight:
            insights.append(usage_insight)

        # Document gap analysis
        doc_gap_insight = self._analyze_document_gaps(project_id, events)
        if doc_gap_insight:
            insights.append(doc_gap_insight)

        # Anomaly detection
        anomaly_insight = self._detect_project_anomalies(project_id, events)
        if anomaly_insight:
            insights.append(anomaly_insight)

        # Store insights in database
        for insight in insights:
            self.db.store_insight(insight)

        return insights

    def _analyze_project_usage(self, project_id: str, events: List[UsageEvent]) -> Optional[ProjectInsight]:
        """Analyze project usage patterns."""
        if len(events) < 5:  # Need minimum data
            return None

        # Calculate metrics
        access_events = [e for e in events if e.event_type == 'project_access']
        search_events = [e for e in events if e.event_type == 'search_query']

        if not access_events:
            return None

        avg_duration = statistics.mean(e.duration_seconds for e in access_events)
        success_rate = sum(1 for e in access_events if e.success) / len(access_events)

        # Generate insight
        if avg_duration > 300:  # More than 5 minutes
            title = "Extended Project Sessions Detected"
            description = f"Users spend an average of {avg_duration/60:.1f} minutes in this project, indicating complex navigation or thorough document review."
            priority = "medium"
            recommendations = [
                "Consider adding project bookmarks for frequently accessed areas",
                "Review project structure for navigation optimization"
            ]
        elif success_rate < 0.7:
            title = "Low Success Rate in Project Navigation"
            description = f"Only {success_rate*100:.1f}% of project navigation attempts are successful."
            priority = "high"
            recommendations = [
                "Review project folder organization",
                "Check for missing or moved documents",
                "Consider adding project navigation guides"
            ]
        else:
            title = "Healthy Project Usage Pattern"
            description = f"Project shows good navigation success rate ({success_rate*100:.1f}%) with reasonable session durations."
            priority = "low"
            recommendations = [
                "Maintain current project organization",
                "Consider this project as a template for similar projects"
            ]

        return ProjectInsight(
            project_id=project_id,
            insight_type="usage_pattern",
            title=title,
            description=description,
            confidence=0.8,
            priority=priority,
            actionable=True,
            recommendations=recommendations,
            metadata={
                "avg_duration": avg_duration,
                "success_rate": success_rate,
                "total_events": len(events)
            }
        )

    def _analyze_document_gaps(self, project_id: str, events: List[UsageEvent]) -> Optional[ProjectInsight]:
        """Analyze potential document gaps."""
        search_events = [e for e in events if e.event_type == 'search_query' and not e.success]

        if len(search_events) < 3:
            return None

        # Analyze failed searches
        failed_queries = [e.query_text for e in search_events if e.query_text]

        # Common document types that users search for
        doc_types = ['lld', 'hld', 'floor plan', 'change order', 'scope', 'photos']
        missing_docs = []

        for doc_type in doc_types:
            if any(doc_type in query.lower() for query in failed_queries):
                missing_docs.append(doc_type)

        if missing_docs:
            return ProjectInsight(
                project_id=project_id,
                insight_type="document_gap",
                title="Potential Missing Documents Detected",
                description=f"Users frequently search for {', '.join(missing_docs)} documents without success.",
                confidence=0.7,
                priority="medium",
                actionable=True,
                recommendations=[
                    f"Check if {doc_type} documents exist and are properly named" for doc_type in missing_docs
                ] + [
                    "Review document naming conventions",
                    "Consider adding document type folders if missing"
                ],
                metadata={
                    "failed_searches": len(search_events),
                    "missing_doc_types": missing_docs
                }
            )

        return None

    def _detect_project_anomalies(self, project_id: str, events: List[UsageEvent]) -> Optional[ProjectInsight]:
        """Detect anomalies in project usage."""
        if len(events) < 10:
            return None

        # Check for unusual patterns
        recent_events = [e for e in events if e.timestamp > datetime.now() - timedelta(days=7)]
        historical_events = [e for e in events if e.timestamp <= datetime.now() - timedelta(days=7)]

        if not historical_events:
            return None

        # Compare recent vs historical activity
        recent_daily_avg = len(recent_events) / 7
        historical_daily_avg = len(historical_events) / max((datetime.now() - min(e.timestamp for e in historical_events)).days, 1)

        activity_ratio = recent_daily_avg / historical_daily_avg if historical_daily_avg > 0 else 0

        if activity_ratio > 3:  # Significant increase
            return ProjectInsight(
                project_id=project_id,
                insight_type="anomaly",
                title="Unusual Spike in Project Activity",
                description=f"Project activity has increased by {activity_ratio:.1f}x in the past week.",
                confidence=0.9,
                priority="medium",
                actionable=True,
                recommendations=[
                    "Monitor project for potential issues or deadlines",
                    "Ensure adequate resources are available",
                    "Check if increased activity indicates project urgency"
                ],
                metadata={
                    "activity_ratio": activity_ratio,
                    "recent_events": len(recent_events),
                    "historical_avg": historical_daily_avg
                }
            )

        return None

    async def generate_user_insights(self, user_id: str) -> Dict[str, Any]:
        """Generate insights about user behavior."""
        events = self.db.get_events(user_id=user_id, limit=1000)

        if not events:
            return {"error": "No user data available"}

        insights = {
            "activity_summary": self._analyze_user_activity_summary(events),
            "preferences": self._detect_user_preferences(events),
            "efficiency_metrics": self._calculate_user_efficiency(events),
            "recommendations": self._generate_user_recommendations(events)
        }

        return insights

    def _analyze_user_activity_summary(self, events: List[UsageEvent]) -> Dict[str, Any]:
        """Analyze user activity summary."""
        project_accesses = [e for e in events if e.event_type == 'project_access']
        searches = [e for e in events if e.event_type == 'search_query']

        return {
            "total_interactions": len(events),
            "project_accesses": len(project_accesses),
            "searches_performed": len(searches),
            "unique_projects": len(set(e.project_id for e in events if e.project_id)),
            "avg_session_duration": statistics.mean(e.duration_seconds for e in events) if events else 0,
            "overall_success_rate": sum(1 for e in events if e.success) / len(events) if events else 0
        }

    def _detect_user_preferences(self, events: List[UsageEvent]) -> Dict[str, Any]:
        """Detect user preferences from behavior."""
        # Analyze project types
        project_ids = [e.project_id for e in events if e.project_id]
        project_patterns = Counter(project_ids)

        # Analyze activity times
        activity_hours = [e.timestamp.hour for e in events]
        peak_hours = Counter(activity_hours).most_common(3)

        # Analyze query patterns
        query_events = [e for e in events if e.event_type == 'search_query' and e.query_text]
        common_terms = []

        if query_events:
            all_terms = []
            for event in query_events:
                terms = event.query_text.lower().split()
                all_terms.extend(term for term in terms if len(term) > 2)
            common_terms = [term for term, count in Counter(all_terms).most_common(10)]

        return {
            "most_accessed_projects": dict(project_patterns.most_common(5)),
            "preferred_activity_hours": [hour for hour, count in peak_hours],
            "common_search_terms": common_terms,
            "activity_pattern": "regular" if len(set(e.timestamp.date() for e in events)) > 5 else "occasional"
        }

    def _calculate_user_efficiency(self, events: List[UsageEvent]) -> Dict[str, Any]:
        """Calculate user efficiency metrics."""
        search_events = [e for e in events if e.event_type == 'search_query']

        if not search_events:
            return {}

        # Search efficiency
        successful_searches = [e for e in search_events if e.success]
        search_efficiency = len(successful_searches) / len(search_events)

        # Average time to success
        successful_durations = [e.duration_seconds for e in successful_searches]
        avg_time_to_success = statistics.mean(successful_durations) if successful_durations else 0

        # Query refinement rate (measure of learning)
        unique_queries = len(set(e.query_text for e in search_events if e.query_text))
        refinement_rate = unique_queries / len(search_events) if search_events else 0

        return {
            "search_efficiency": search_efficiency,
            "avg_time_to_success": avg_time_to_success,
            "query_refinement_rate": refinement_rate,
            "efficiency_trend": "improving" if search_efficiency > 0.7 else "needs_improvement"
        }

    def _generate_user_recommendations(self, events: List[UsageEvent]) -> List[str]:
        """Generate personalized recommendations for the user."""
        recommendations = []

        # Analyze success patterns
        search_events = [e for e in events if e.event_type == 'search_query']
        if search_events:
            success_rate = sum(1 for e in search_events if e.success) / len(search_events)

            if success_rate < 0.5:
                recommendations.append("Consider using more specific search terms or project numbers")
                recommendations.append("Try using the project navigation tools for better results")

        # Analyze time patterns
        activity_hours = [e.timestamp.hour for e in events]
        if activity_hours:
            peak_hour = Counter(activity_hours).most_common(1)[0][0]
            recommendations.append(f"Your peak productivity appears to be around {peak_hour}:00")

        # Analyze project patterns
        project_accesses = [e for e in events if e.event_type == 'project_access']
        if len(set(e.project_id for e in project_accesses if e.project_id)) > 10:
            recommendations.append("Consider using the recent projects feature to quickly access frequently used projects")

        return recommendations


class AnalyticsEngine:
    """Main analytics engine coordinating all analytics components."""

    def __init__(self, db_path: str = "data/analytics.db"):
        self.db = AnalyticsDatabase(db_path)
        self.pattern_analyzer = PatternAnalyzer(self.db)
        self.insight_generator = InsightGenerator(self.db, self.pattern_analyzer)

    async def log_user_interaction(self, user_id: str, event_type: str,
                                  project_id: Optional[str] = None,
                                  document_path: Optional[str] = None,
                                  query_text: Optional[str] = None,
                                  success: bool = True,
                                  duration_seconds: float = 0.0,
                                  metadata: Optional[Dict] = None):
        """Log a user interaction event."""
        event = UsageEvent(
            event_id=f"{user_id}_{event_type}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            user_id=user_id,
            event_type=event_type,
            timestamp=datetime.now(),
            project_id=project_id,
            document_path=document_path,
            query_text=query_text,
            success=success,
            duration_seconds=duration_seconds,
            metadata=metadata or {}
        )

        self.db.log_event(event)

    async def get_dashboard_data(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        try:
            dashboard = {
                "overview": await self._get_overview_metrics(days),
                "project_patterns": self.pattern_analyzer.analyze_project_usage_patterns(days),
                "search_patterns": self.pattern_analyzer.analyze_search_patterns(days),
                "recent_insights": self.db.get_insights()[:10],
                "user_activity": await self._get_user_activity_summary(days),
                "performance_metrics": await self._get_performance_metrics(days)
            }

            return dashboard

        except Exception as e:
            logger.error(f"Failed to get dashboard data: {e}")
            return {"error": str(e)}

    async def _get_overview_metrics(self, days: int) -> Dict[str, Any]:
        """Get overview metrics."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        events = self.db.get_events(start_date=start_date, end_date=end_date)

        return {
            "total_interactions": len(events),
            "unique_users": len(set(e.user_id for e in events)),
            "unique_projects": len(set(e.project_id for e in events if e.project_id)),
            "success_rate": sum(1 for e in events if e.success) / len(events) if events else 0,
            "avg_session_duration": statistics.mean(e.duration_seconds for e in events) if events else 0
        }

    async def _get_user_activity_summary(self, days: int) -> Dict[str, Any]:
        """Get user activity summary."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        events = self.db.get_events(start_date=start_date, end_date=end_date)
        user_events = defaultdict(list)

        for event in events:
            user_events[event.user_id].append(event)

        return {
            "active_users": len(user_events),
            "user_activity_levels": {
                "high": len([u for u, e in user_events.items() if len(e) > 50]),
                "medium": len([u for u, e in user_events.items() if 10 <= len(e) <= 50]),
                "low": len([u for u, e in user_events.items() if len(e) < 10])
            }
        }

    async def _get_performance_metrics(self, days: int) -> Dict[str, Any]:
        """Get system performance metrics."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        events = self.db.get_events(start_date=start_date, end_date=end_date)

        if not events:
            return {}

        # Response time analysis
        response_times = [e.duration_seconds for e in events if e.duration_seconds > 0]

        return {
            "avg_response_time": statistics.mean(response_times) if response_times else 0,
            "median_response_time": statistics.median(response_times) if response_times else 0,
            "slow_queries": len([t for t in response_times if t > 5.0]),
            "performance_trend": "stable"  # Could be enhanced with trend analysis
        }

    async def generate_natural_language_insights(self, context: str = "general") -> List[str]:
        """Generate natural language insights about the system usage."""
        insights = []

        try:
            # Get recent patterns
            project_patterns = self.pattern_analyzer.analyze_project_usage_patterns(30)
            search_patterns = self.pattern_analyzer.analyze_search_patterns(30)

            # Generate insights based on patterns
            if project_patterns.get("total_accesses", 0) > 100:
                insights.append(
                    f"Your team has been very active with {project_patterns['total_accesses']} project accesses in the last 30 days."
                )

            success_rate = project_patterns.get("success_rate", 0)
            if success_rate > 0.9:
                insights.append("Navigation success rate is excellent - the system is working well for your team.")
            elif success_rate < 0.7:
                insights.append("Navigation success rate could be improved - consider reviewing project organization.")

            # Search insights
            if search_patterns.get("total_searches", 0) > 0:
                search_success = search_patterns.get("success_rate", 0)
                if search_success < 0.6:
                    insights.append("Search success rate is low - users might benefit from search training or better document organization.")

            # Popular projects insight
            popular_projects = project_patterns.get("project_popularity", {}).get("top_projects", {})
            if popular_projects:
                top_project = max(popular_projects, key=popular_projects.get)
                insights.append(f"Project {top_project} is your most accessed project with {popular_projects[top_project]} accesses.")

            # Trend insights
            trends = project_patterns.get("trends", {})
            if trends.get("usage_trend") == "increasing":
                insights.append("Project usage is trending upward - your team's activity is increasing.")
            elif trends.get("usage_trend") == "decreasing":
                insights.append("Project usage is declining - consider checking if users are facing barriers.")

        except Exception as e:
            logger.error(f"Failed to generate natural language insights: {e}")
            insights.append("Unable to generate insights at this time.")

        return insights[:5]  # Return top 5 insights


# Export main classes
__all__ = [
    'AnalyticsEngine',
    'AnalyticsDatabase',
    'PatternAnalyzer',
    'InsightGenerator',
    'UsageEvent',
    'ProjectInsight',
    'UserBehaviorProfile'
]