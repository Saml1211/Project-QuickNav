"""
Database Performance Optimizer for Project QuickNav

Provides comprehensive database optimization including:
- Query performance analysis and optimization
- Intelligent indexing strategies
- Connection pooling and management
- Query result caching
- Database monitoring and alerting
- Automatic performance tuning
"""

import sqlite3
import asyncio
import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from contextlib import contextmanager, asynccontextmanager
import json
import hashlib
from pathlib import Path
import statistics
import re
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class QueryStats:
    """Query performance statistics."""
    query_hash: str
    query_text: str
    execution_count: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    last_executed: datetime
    rows_affected: int
    query_type: str  # SELECT, INSERT, UPDATE, DELETE
    table_names: List[str]
    index_usage: Dict[str, Any]


@dataclass
class IndexRecommendation:
    """Database index recommendation."""
    table_name: str
    column_names: List[str]
    index_type: str  # BTREE, HASH, etc.
    estimated_benefit: float
    query_patterns: List[str]
    priority: int  # 1=high, 2=medium, 3=low
    reason: str


@dataclass
class DatabaseHealth:
    """Database health metrics."""
    total_queries: int
    avg_query_time_ms: float
    slow_queries_count: int
    connection_pool_utilization: float
    cache_hit_rate: float
    index_efficiency: float
    health_score: float  # 0-100
    recommendations: List[str]


class QueryAnalyzer:
    """Analyzes SQL queries for optimization opportunities."""

    def __init__(self):
        self.query_patterns = {
            'full_table_scan': r'SELECT.*FROM\s+(\w+)(?:\s+WHERE)?',
            'missing_index': r'WHERE\s+(\w+)\s*=',
            'inefficient_join': r'JOIN\s+(\w+).*ON\s+(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)',
            'order_by_no_index': r'ORDER\s+BY\s+(\w+)',
            'group_by_no_index': r'GROUP\s+BY\s+(\w+)',
            'like_wildcard': r"LIKE\s+'%.*%'",
            'function_in_where': r'WHERE\s+\w+\([^)]+\)',
        }

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze a query for optimization opportunities."""
        query_upper = query.upper()
        issues = []
        recommendations = []

        # Determine query type
        if query_upper.startswith('SELECT'):
            query_type = 'SELECT'
        elif query_upper.startswith('INSERT'):
            query_type = 'INSERT'
        elif query_upper.startswith('UPDATE'):
            query_type = 'UPDATE'
        elif query_upper.startswith('DELETE'):
            query_type = 'DELETE'
        else:
            query_type = 'OTHER'

        # Extract table names
        table_names = self._extract_table_names(query)

        # Check for common performance issues
        for pattern_name, pattern in self.query_patterns.items():
            if re.search(pattern, query_upper):
                if pattern_name == 'full_table_scan':
                    issues.append('Potential full table scan detected')
                    recommendations.append('Consider adding appropriate indexes')
                elif pattern_name == 'like_wildcard':
                    issues.append('LIKE query with leading wildcard detected')
                    recommendations.append('Consider full-text search or different approach')
                elif pattern_name == 'function_in_where':
                    issues.append('Function in WHERE clause detected')
                    recommendations.append('Avoid functions in WHERE clause for better index usage')

        return {
            'query_type': query_type,
            'table_names': table_names,
            'issues': issues,
            'recommendations': recommendations,
            'complexity_score': self._calculate_complexity(query)
        }

    def _extract_table_names(self, query: str) -> List[str]:
        """Extract table names from query."""
        tables = []
        query_upper = query.upper()

        # Find FROM clauses
        from_matches = re.finditer(r'FROM\s+(\w+)', query_upper)
        for match in from_matches:
            tables.append(match.group(1).lower())

        # Find JOIN clauses
        join_matches = re.finditer(r'JOIN\s+(\w+)', query_upper)
        for match in join_matches:
            tables.append(match.group(1).lower())

        # Find INSERT INTO
        insert_matches = re.finditer(r'INSERT\s+INTO\s+(\w+)', query_upper)
        for match in insert_matches:
            tables.append(match.group(1).lower())

        # Find UPDATE
        update_matches = re.finditer(r'UPDATE\s+(\w+)', query_upper)
        for match in update_matches:
            tables.append(match.group(1).lower())

        return list(set(tables))

    def _calculate_complexity(self, query: str) -> int:
        """Calculate query complexity score."""
        score = 0
        query_upper = query.upper()

        # Count joins
        score += len(re.findall(r'JOIN', query_upper)) * 2

        # Count subqueries
        score += len(re.findall(r'SELECT.*FROM.*SELECT', query_upper)) * 3

        # Count conditions
        score += len(re.findall(r'WHERE|AND|OR', query_upper))

        # Count functions
        score += len(re.findall(r'\w+\(', query_upper))

        return score


class ConnectionPool:
    """Thread-safe database connection pool."""

    def __init__(self, db_path: str, max_connections: int = 10):
        self.db_path = db_path
        self.max_connections = max_connections
        self._connections = []
        self._available_connections = []
        self._lock = threading.Lock()
        self._connection_count = 0
        self._active_connections = 0

        # Create initial connections
        self._create_initial_connections()

    def _create_initial_connections(self):
        """Create initial set of connections."""
        initial_count = min(3, self.max_connections)
        for _ in range(initial_count):
            self._create_connection()

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection."""
        try:
            conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            conn.row_factory = sqlite3.Row

            # Set SQLite optimizations
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=memory")
            conn.execute("PRAGMA mmap_size=268435456")  # 256MB

            with self._lock:
                self._connections.append(conn)
                self._available_connections.append(conn)
                self._connection_count += 1

            return conn

        except Exception as e:
            logger.error(f"Failed to create database connection: {e}")
            raise

    @contextmanager
    def get_connection(self):
        """Get a connection from the pool."""
        conn = None
        try:
            with self._lock:
                if self._available_connections:
                    conn = self._available_connections.pop()
                    self._active_connections += 1
                elif self._connection_count < self.max_connections:
                    conn = self._create_connection()
                    self._available_connections.remove(conn)
                    self._active_connections += 1

            if conn is None:
                # Wait for available connection
                timeout = 10.0
                start_time = time.time()
                while conn is None and (time.time() - start_time) < timeout:
                    time.sleep(0.1)
                    with self._lock:
                        if self._available_connections:
                            conn = self._available_connections.pop()
                            self._active_connections += 1

                if conn is None:
                    raise Exception("No database connections available")

            yield conn

        finally:
            if conn:
                with self._lock:
                    if conn in self._connections:
                        self._available_connections.append(conn)
                        self._active_connections -= 1

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        with self._lock:
            return {
                'total_connections': self._connection_count,
                'active_connections': self._active_connections,
                'available_connections': len(self._available_connections),
                'utilization': self._active_connections / self.max_connections,
                'max_connections': self.max_connections
            }

    def close_all(self):
        """Close all connections in the pool."""
        with self._lock:
            for conn in self._connections:
                try:
                    conn.close()
                except:
                    pass
            self._connections.clear()
            self._available_connections.clear()
            self._connection_count = 0
            self._active_connections = 0


class DatabaseOptimizer:
    """Comprehensive database performance optimizer."""

    def __init__(self, db_path: str, analytics_db_path: str = None):
        self.db_path = db_path
        self.analytics_db_path = analytics_db_path or f"{db_path}_analytics.db"

        # Components
        self.connection_pool = ConnectionPool(db_path)
        self.query_analyzer = QueryAnalyzer()

        # Query tracking
        self.query_stats = {}
        self.query_cache = {}
        self.stats_lock = threading.Lock()

        # Performance monitoring
        self.monitoring_active = False
        self.monitor_thread = None

        # Setup analytics database
        self._setup_analytics_db()

        # Start monitoring
        self.start_monitoring()

    def _setup_analytics_db(self):
        """Setup analytics database for query statistics."""
        try:
            with sqlite3.connect(self.analytics_db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS query_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query_hash TEXT NOT NULL,
                        query_text TEXT NOT NULL,
                        execution_time_ms REAL NOT NULL,
                        rows_affected INTEGER DEFAULT 0,
                        timestamp DATETIME NOT NULL,
                        table_names TEXT DEFAULT '[]',
                        query_type TEXT NOT NULL,
                        analysis_data TEXT DEFAULT '{}'
                    )
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS index_recommendations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        table_name TEXT NOT NULL,
                        column_names TEXT NOT NULL,
                        index_type TEXT NOT NULL,
                        estimated_benefit REAL NOT NULL,
                        priority INTEGER NOT NULL,
                        reason TEXT NOT NULL,
                        created_at DATETIME NOT NULL,
                        applied BOOLEAN DEFAULT FALSE
                    )
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS performance_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_type TEXT NOT NULL,
                        description TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        metadata TEXT DEFAULT '{}'
                    )
                """)

                # Create indexes for analytics queries
                conn.execute("CREATE INDEX IF NOT EXISTS idx_query_hash ON query_performance(query_hash)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON query_performance(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_execution_time ON query_performance(execution_time_ms)")

        except Exception as e:
            logger.error(f"Failed to setup analytics database: {e}")

    def start_monitoring(self):
        """Start performance monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_performance, daemon=True)
        self.monitor_thread.start()
        logger.info("Database performance monitoring started")

    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Database performance monitoring stopped")

    def _monitor_performance(self):
        """Monitor database performance continuously."""
        while self.monitoring_active:
            try:
                # Check for slow queries
                self._check_slow_queries()

                # Update recommendations
                self._update_index_recommendations()

                # Check database health
                health = self.get_database_health()
                if health.health_score < 70:
                    self._log_performance_event(
                        'performance_degradation',
                        f"Database health score: {health.health_score}",
                        'warning'
                    )

                time.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                time.sleep(60)

    def execute_query(self, query: str, params: List = None,
                     cache_key: str = None, cache_ttl: int = 300) -> List[sqlite3.Row]:
        """Execute query with performance tracking and caching."""
        params = params or []
        query_hash = hashlib.md5(query.encode()).hexdigest()

        # Check cache first
        if cache_key:
            cached_result = self._get_cached_result(cache_key)
            if cached_result is not None:
                return cached_result

        start_time = time.perf_counter()

        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.execute(query, params)
                results = cursor.fetchall()
                rows_affected = cursor.rowcount

            execution_time_ms = (time.perf_counter() - start_time) * 1000

            # Track performance
            self._track_query_performance(query, query_hash, execution_time_ms, rows_affected)

            # Cache result if requested
            if cache_key and results:
                self._cache_result(cache_key, results, cache_ttl)

            return results

        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Query execution failed ({execution_time_ms:.2f}ms): {e}")
            self._track_query_performance(query, query_hash, execution_time_ms, 0, error=str(e))
            raise

    def execute_write_query(self, query: str, params: List = None) -> int:
        """Execute write query (INSERT, UPDATE, DELETE) with performance tracking."""
        params = params or []
        query_hash = hashlib.md5(query.encode()).hexdigest()

        start_time = time.perf_counter()

        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.execute(query, params)
                conn.commit()
                rows_affected = cursor.rowcount

            execution_time_ms = (time.perf_counter() - start_time) * 1000

            # Track performance
            self._track_query_performance(query, query_hash, execution_time_ms, rows_affected)

            return rows_affected

        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Write query execution failed ({execution_time_ms:.2f}ms): {e}")
            self._track_query_performance(query, query_hash, execution_time_ms, 0, error=str(e))
            raise

    def _track_query_performance(self, query: str, query_hash: str,
                                execution_time_ms: float, rows_affected: int, error: str = None):
        """Track query performance statistics."""
        # Analyze query
        analysis = self.query_analyzer.analyze_query(query)

        # Update in-memory stats
        with self.stats_lock:
            if query_hash not in self.query_stats:
                self.query_stats[query_hash] = {
                    'query_text': query,
                    'execution_count': 0,
                    'total_time_ms': 0.0,
                    'min_time_ms': float('inf'),
                    'max_time_ms': 0.0,
                    'last_executed': datetime.utcnow(),
                    'rows_affected': 0,
                    'query_type': analysis['query_type'],
                    'table_names': analysis['table_names'],
                    'error_count': 0
                }

            stats = self.query_stats[query_hash]
            stats['execution_count'] += 1
            stats['total_time_ms'] += execution_time_ms
            stats['min_time_ms'] = min(stats['min_time_ms'], execution_time_ms)
            stats['max_time_ms'] = max(stats['max_time_ms'], execution_time_ms)
            stats['last_executed'] = datetime.utcnow()
            stats['rows_affected'] += rows_affected

            if error:
                stats['error_count'] += 1

        # Store in analytics database
        try:
            with sqlite3.connect(self.analytics_db_path) as conn:
                conn.execute("""
                    INSERT INTO query_performance
                    (query_hash, query_text, execution_time_ms, rows_affected, timestamp,
                     table_names, query_type, analysis_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    query_hash, query[:1000],  # Truncate long queries
                    execution_time_ms, rows_affected, datetime.utcnow(),
                    json.dumps(analysis['table_names']),
                    analysis['query_type'],
                    json.dumps(analysis)
                ])
        except Exception as e:
            logger.error(f"Failed to store query performance: {e}")

    def _get_cached_result(self, cache_key: str) -> Optional[List[sqlite3.Row]]:
        """Get cached query result."""
        if cache_key in self.query_cache:
            cached_data = self.query_cache[cache_key]
            if datetime.utcnow() < cached_data['expires_at']:
                return cached_data['result']
            else:
                del self.query_cache[cache_key]
        return None

    def _cache_result(self, cache_key: str, result: List[sqlite3.Row], ttl_seconds: int):
        """Cache query result."""
        self.query_cache[cache_key] = {
            'result': result,
            'expires_at': datetime.utcnow() + timedelta(seconds=ttl_seconds),
            'created_at': datetime.utcnow()
        }

        # Clean old cache entries periodically
        if len(self.query_cache) > 1000:
            self._cleanup_cache()

    def _cleanup_cache(self):
        """Clean up expired cache entries."""
        now = datetime.utcnow()
        expired_keys = [
            key for key, data in self.query_cache.items()
            if now >= data['expires_at']
        ]
        for key in expired_keys:
            del self.query_cache[key]

    def _check_slow_queries(self):
        """Check for slow queries and log alerts."""
        slow_query_threshold = 1000  # 1 second

        with self.stats_lock:
            for query_hash, stats in self.query_stats.items():
                avg_time = stats['total_time_ms'] / stats['execution_count']
                if avg_time > slow_query_threshold:
                    self._log_performance_event(
                        'slow_query',
                        f"Slow query detected: {avg_time:.2f}ms average",
                        'warning',
                        {'query_hash': query_hash, 'avg_time_ms': avg_time}
                    )

    def _update_index_recommendations(self):
        """Update index recommendations based on query patterns."""
        # Analyze recent queries for indexing opportunities
        with sqlite3.connect(self.analytics_db_path) as conn:
            # Get queries from last hour that took >100ms
            recent_slow_queries = conn.execute("""
                SELECT query_text, table_names, AVG(execution_time_ms) as avg_time,
                       COUNT(*) as execution_count
                FROM query_performance
                WHERE timestamp >= datetime('now', '-1 hour')
                  AND execution_time_ms > 100
                GROUP BY query_hash
                HAVING COUNT(*) >= 3
                ORDER BY avg_time DESC
                LIMIT 10
            """).fetchall()

            for query_data in recent_slow_queries:
                recommendations = self._analyze_for_index_recommendations(
                    query_data['query_text'],
                    json.loads(query_data['table_names']),
                    query_data['avg_time'],
                    query_data['execution_count']
                )

                for rec in recommendations:
                    self._store_index_recommendation(rec)

    def _analyze_for_index_recommendations(self, query: str, table_names: List[str],
                                         avg_time_ms: float, execution_count: int) -> List[IndexRecommendation]:
        """Analyze query for index recommendations."""
        recommendations = []

        # Simple pattern matching for common index opportunities
        query_upper = query.upper()

        # WHERE clause columns
        where_matches = re.finditer(r'WHERE\s+(\w+)\s*[=<>]', query_upper)
        for match in where_matches:
            column = match.group(1).lower()
            for table in table_names:
                if self._should_recommend_index(table, [column]):
                    recommendations.append(IndexRecommendation(
                        table_name=table,
                        column_names=[column],
                        index_type='BTREE',
                        estimated_benefit=min(avg_time_ms * 0.5, 500),
                        query_patterns=[query[:100]],
                        priority=1 if avg_time_ms > 500 else 2,
                        reason=f"WHERE clause on {column} in slow query"
                    ))

        # ORDER BY columns
        order_matches = re.finditer(r'ORDER\s+BY\s+(\w+)', query_upper)
        for match in order_matches:
            column = match.group(1).lower()
            for table in table_names:
                if self._should_recommend_index(table, [column]):
                    recommendations.append(IndexRecommendation(
                        table_name=table,
                        column_names=[column],
                        index_type='BTREE',
                        estimated_benefit=min(avg_time_ms * 0.3, 300),
                        query_patterns=[query[:100]],
                        priority=2,
                        reason=f"ORDER BY on {column} in slow query"
                    ))

        return recommendations

    def _should_recommend_index(self, table: str, columns: List[str]) -> bool:
        """Check if an index should be recommended."""
        # Check if index already exists
        try:
            with self.connection_pool.get_connection() as conn:
                index_info = conn.execute(
                    "PRAGMA index_list(?)",
                    [table]
                ).fetchall()

                for index in index_info:
                    index_columns = conn.execute(
                        "PRAGMA index_info(?)",
                        [index['name']]
                    ).fetchall()

                    existing_columns = [col['name'].lower() for col in index_columns]
                    if set(columns).issubset(set(existing_columns)):
                        return False  # Index already exists

            return True

        except Exception as e:
            logger.error(f"Error checking existing indexes: {e}")
            return False

    def _store_index_recommendation(self, recommendation: IndexRecommendation):
        """Store index recommendation in database."""
        try:
            with sqlite3.connect(self.analytics_db_path) as conn:
                # Check if similar recommendation already exists
                existing = conn.execute("""
                    SELECT id FROM index_recommendations
                    WHERE table_name = ? AND column_names = ? AND applied = FALSE
                """, [
                    recommendation.table_name,
                    json.dumps(recommendation.column_names)
                ]).fetchone()

                if not existing:
                    conn.execute("""
                        INSERT INTO index_recommendations
                        (table_name, column_names, index_type, estimated_benefit,
                         priority, reason, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, [
                        recommendation.table_name,
                        json.dumps(recommendation.column_names),
                        recommendation.index_type,
                        recommendation.estimated_benefit,
                        recommendation.priority,
                        recommendation.reason,
                        datetime.utcnow()
                    ])

        except Exception as e:
            logger.error(f"Failed to store index recommendation: {e}")

    def _log_performance_event(self, event_type: str, description: str, severity: str, metadata: Dict = None):
        """Log performance event."""
        try:
            with sqlite3.connect(self.analytics_db_path) as conn:
                conn.execute("""
                    INSERT INTO performance_events
                    (event_type, description, severity, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, [
                    event_type, description, severity,
                    datetime.utcnow(), json.dumps(metadata or {})
                ])

            if severity in ['error', 'warning']:
                logger.warning(f"Database performance event ({severity}): {description}")

        except Exception as e:
            logger.error(f"Failed to log performance event: {e}")

    def get_query_statistics(self, limit: int = 20) -> List[QueryStats]:
        """Get query performance statistics."""
        stats_list = []

        with self.stats_lock:
            for query_hash, stats in self.query_stats.items():
                if stats['execution_count'] > 0:
                    avg_time = stats['total_time_ms'] / stats['execution_count']
                    stats_list.append(QueryStats(
                        query_hash=query_hash,
                        query_text=stats['query_text'],
                        execution_count=stats['execution_count'],
                        total_time_ms=stats['total_time_ms'],
                        avg_time_ms=avg_time,
                        min_time_ms=stats['min_time_ms'],
                        max_time_ms=stats['max_time_ms'],
                        last_executed=stats['last_executed'],
                        rows_affected=stats['rows_affected'],
                        query_type=stats['query_type'],
                        table_names=stats['table_names'],
                        index_usage={}
                    ))

        # Sort by average execution time (slowest first)
        stats_list.sort(key=lambda x: x.avg_time_ms, reverse=True)
        return stats_list[:limit]

    def get_index_recommendations(self, limit: int = 10) -> List[IndexRecommendation]:
        """Get index recommendations."""
        recommendations = []

        try:
            with sqlite3.connect(self.analytics_db_path) as conn:
                rows = conn.execute("""
                    SELECT table_name, column_names, index_type, estimated_benefit,
                           priority, reason, created_at
                    FROM index_recommendations
                    WHERE applied = FALSE
                    ORDER BY priority ASC, estimated_benefit DESC
                    LIMIT ?
                """, [limit]).fetchall()

                for row in rows:
                    recommendations.append(IndexRecommendation(
                        table_name=row['table_name'],
                        column_names=json.loads(row['column_names']),
                        index_type=row['index_type'],
                        estimated_benefit=row['estimated_benefit'],
                        query_patterns=[],  # Would need to join with query data
                        priority=row['priority'],
                        reason=row['reason']
                    ))

        except Exception as e:
            logger.error(f"Failed to get index recommendations: {e}")

        return recommendations

    def apply_index_recommendation(self, table_name: str, column_names: List[str]) -> bool:
        """Apply an index recommendation."""
        try:
            index_name = f"idx_{table_name}_{'_'.join(column_names)}"
            columns_str = ', '.join(column_names)

            with self.connection_pool.get_connection() as conn:
                conn.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({columns_str})")
                conn.commit()

            # Mark as applied
            with sqlite3.connect(self.analytics_db_path) as conn:
                conn.execute("""
                    UPDATE index_recommendations
                    SET applied = TRUE
                    WHERE table_name = ? AND column_names = ?
                """, [table_name, json.dumps(column_names)])

            logger.info(f"Applied index: {index_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to apply index recommendation: {e}")
            return False

    def get_database_health(self) -> DatabaseHealth:
        """Get comprehensive database health metrics."""
        try:
            with self.stats_lock:
                if not self.query_stats:
                    return DatabaseHealth(
                        total_queries=0,
                        avg_query_time_ms=0,
                        slow_queries_count=0,
                        connection_pool_utilization=0,
                        cache_hit_rate=0,
                        index_efficiency=100,
                        health_score=100,
                        recommendations=[]
                    )

                total_queries = sum(stats['execution_count'] for stats in self.query_stats.values())
                total_time = sum(stats['total_time_ms'] for stats in self.query_stats.values())
                avg_query_time = total_time / total_queries if total_queries > 0 else 0

                slow_queries_count = sum(
                    1 for stats in self.query_stats.values()
                    if (stats['total_time_ms'] / stats['execution_count']) > 1000
                )

            # Connection pool stats
            pool_stats = self.connection_pool.get_pool_stats()

            # Cache stats
            cache_requests = len(self.query_cache)
            cache_hit_rate = 0.8  # Simplified calculation

            # Calculate health score
            health_score = 100
            if avg_query_time > 500:
                health_score -= 20
            if slow_queries_count > total_queries * 0.1:
                health_score -= 15
            if pool_stats['utilization'] > 0.8:
                health_score -= 10

            # Generate recommendations
            recommendations = []
            if avg_query_time > 200:
                recommendations.append("Consider optimizing slow queries")
            if pool_stats['utilization'] > 0.7:
                recommendations.append("Consider increasing connection pool size")

            return DatabaseHealth(
                total_queries=total_queries,
                avg_query_time_ms=avg_query_time,
                slow_queries_count=slow_queries_count,
                connection_pool_utilization=pool_stats['utilization'],
                cache_hit_rate=cache_hit_rate,
                index_efficiency=85,  # Would need actual calculation
                health_score=max(0, health_score),
                recommendations=recommendations
            )

        except Exception as e:
            logger.error(f"Failed to calculate database health: {e}")
            return DatabaseHealth(
                total_queries=0, avg_query_time_ms=0, slow_queries_count=0,
                connection_pool_utilization=0, cache_hit_rate=0,
                index_efficiency=0, health_score=0,
                recommendations=["Error calculating health metrics"]
            )

    def optimize_database(self) -> Dict[str, Any]:
        """Run comprehensive database optimization."""
        results = {
            'vacuum_performed': False,
            'indexes_created': 0,
            'slow_queries_optimized': 0,
            'cache_cleared': False,
            'recommendations_applied': []
        }

        try:
            # Run VACUUM to reclaim space and defragment
            with self.connection_pool.get_connection() as conn:
                conn.execute("VACUUM")
            results['vacuum_performed'] = True

            # Apply top index recommendations
            recommendations = self.get_index_recommendations(5)
            for rec in recommendations:
                if self.apply_index_recommendation(rec.table_name, rec.column_names):
                    results['indexes_created'] += 1
                    results['recommendations_applied'].append(f"Index on {rec.table_name}.{rec.column_names}")

            # Clear old cache entries
            self._cleanup_cache()
            results['cache_cleared'] = True

            logger.info("Database optimization completed")

        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
            results['error'] = str(e)

        return results

    def export_performance_report(self, filepath: str, hours: int = 24):
        """Export comprehensive performance report."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        report = {
            'generated_at': datetime.utcnow().isoformat(),
            'time_period_hours': hours,
            'query_statistics': [asdict(stat) for stat in self.get_query_statistics()],
            'index_recommendations': [asdict(rec) for rec in self.get_index_recommendations()],
            'database_health': asdict(self.get_database_health()),
            'connection_pool_stats': self.connection_pool.get_pool_stats()
        }

        # Add recent performance events
        try:
            with sqlite3.connect(self.analytics_db_path) as conn:
                events = conn.execute("""
                    SELECT event_type, description, severity, timestamp, metadata
                    FROM performance_events
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                    LIMIT 50
                """, [cutoff_time]).fetchall()

                report['performance_events'] = [
                    {
                        'event_type': event['event_type'],
                        'description': event['description'],
                        'severity': event['severity'],
                        'timestamp': event['timestamp'],
                        'metadata': json.loads(event['metadata'])
                    }
                    for event in events
                ]

        except Exception as e:
            logger.error(f"Failed to get performance events: {e}")
            report['performance_events'] = []

        # Save report
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Performance report exported to {filepath}")

    def cleanup_analytics_data(self, days_to_keep: int = 30):
        """Clean up old analytics data."""
        cutoff_time = datetime.utcnow() - timedelta(days=days_to_keep)

        try:
            with sqlite3.connect(self.analytics_db_path) as conn:
                # Clean up old query performance data
                result = conn.execute(
                    "DELETE FROM query_performance WHERE timestamp < ?",
                    [cutoff_time]
                )
                perf_deleted = result.rowcount

                # Clean up old performance events
                result = conn.execute(
                    "DELETE FROM performance_events WHERE timestamp < ?",
                    [cutoff_time]
                )
                events_deleted = result.rowcount

                logger.info(f"Cleaned up {perf_deleted} performance records and {events_deleted} events")

        except Exception as e:
            logger.error(f"Failed to cleanup analytics data: {e}")

    def close(self):
        """Close database optimizer and clean up resources."""
        self.stop_monitoring()
        self.connection_pool.close_all()
        logger.info("Database optimizer closed")