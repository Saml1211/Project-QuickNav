"""
Database Manager for Project QuickNav Analytics & ML

This module provides a unified interface for both transactional and analytical
database operations with automatic fallback from DuckDB to SQLite.

Features:
- Dual database support (DuckDB primary, SQLite fallback)
- Automatic schema initialization and migration
- Performance-optimized queries for analytics
- ML feature storage and retrieval
- Time-series data management with archival
- User activity tracking and analytics
"""

import os
import json
import sqlite3
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import hashlib

logger = logging.getLogger(__name__)

# Try to import DuckDB with fallback to SQLite only
try:
    import duckdb
    DUCKDB_AVAILABLE = True
    logger.info("DuckDB available - using for analytical workloads")
except ImportError:
    DUCKDB_AVAILABLE = False
    logger.warning("DuckDB not available - falling back to SQLite only")


class DatabaseManager:
    """
    Unified database manager supporting both DuckDB (analytical) and SQLite (transactional).
    Automatically handles fallback and provides optimized query patterns.
    """

    def __init__(self, db_path: Optional[str] = None, use_duckdb: bool = True):
        """
        Initialize database manager.

        Args:
            db_path: Custom database path, if None uses platform default
            use_duckdb: Whether to attempt using DuckDB (falls back to SQLite if unavailable)
        """
        self.db_path = self._get_database_path(db_path)
        self.use_duckdb = use_duckdb and DUCKDB_AVAILABLE

        # Initialize connections
        self.conn = None
        self.sqlite_conn = None
        self._initialize_databases()

    def _get_database_path(self, custom_path: Optional[str]) -> Path:
        """Get the appropriate database path for the platform."""
        if custom_path:
            return Path(custom_path)

        # Use platform-appropriate data directory
        if os.name == 'nt':  # Windows
            data_dir = Path(os.environ.get('APPDATA', '')) / 'QuickNav' / 'data'
        else:  # Unix-like
            data_dir = Path.home() / '.local' / 'share' / 'quicknav'

        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    def _initialize_databases(self):
        """Initialize database connections and schema."""
        try:
            if self.use_duckdb:
                # Primary DuckDB connection for analytics
                duckdb_path = self.db_path / 'quicknav_analytics.duckdb'
                self.conn = duckdb.connect(str(duckdb_path))
                logger.info(f"Connected to DuckDB at {duckdb_path}")

            # SQLite connection for transactional data (always available)
            sqlite_path = self.db_path / 'quicknav_transactional.db'
            self.sqlite_conn = sqlite3.connect(str(sqlite_path), check_same_thread=False)
            self.sqlite_conn.row_factory = sqlite3.Row  # Enable column access by name
            logger.info(f"Connected to SQLite at {sqlite_path}")

            # Initialize schema
            self._initialize_schema()

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            # Fallback to SQLite only
            if self.conn:
                self.conn.close()
                self.conn = None
            self.use_duckdb = False

    def _initialize_schema(self):
        """Initialize database schema on both connections."""
        schema_path = Path(__file__).parent / 'schema.sql'

        if not schema_path.exists():
            logger.warning(f"Schema file not found at {schema_path}")
            return

        schema_sql = schema_path.read_text(encoding='utf-8')

        # Apply schema to DuckDB if available
        if self.conn:
            try:
                self.conn.execute(schema_sql)
                logger.info("DuckDB schema initialized successfully")
            except Exception as e:
                logger.error(f"DuckDB schema initialization failed: {e}")

        # Apply schema to SQLite
        try:
            self.sqlite_conn.executescript(schema_sql)
            self.sqlite_conn.commit()
            logger.info("SQLite schema initialized successfully")
        except Exception as e:
            logger.error(f"SQLite schema initialization failed: {e}")

    def _get_connection(self, prefer_analytics: bool = False) -> Union[sqlite3.Connection, Any]:
        """
        Get appropriate database connection.

        Args:
            prefer_analytics: Whether to prefer DuckDB for analytical queries

        Returns:
            Database connection (DuckDB or SQLite)
        """
        if prefer_analytics and self.conn:
            return self.conn
        return self.sqlite_conn

    def execute_query(self, query: str, params: Optional[List] = None,
                     prefer_analytics: bool = False) -> List[Dict]:
        """
        Execute a query and return results as list of dictionaries.

        Args:
            query: SQL query to execute
            params: Query parameters
            prefer_analytics: Whether to prefer DuckDB for this query

        Returns:
            List of result dictionaries
        """
        conn = self._get_connection(prefer_analytics)
        params = params or []

        try:
            if self.use_duckdb and prefer_analytics and conn == self.conn:
                # DuckDB query
                result = conn.execute(query, params).fetchall()
                if result:
                    columns = [desc[0] for desc in conn.description]
                    return [dict(zip(columns, row)) for row in result]
                return []
            else:
                # SQLite query
                cursor = conn.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            return []

    def execute_write(self, query: str, params: Optional[List] = None,
                     use_transaction: bool = True) -> bool:
        """
        Execute a write query (INSERT, UPDATE, DELETE).

        Args:
            query: SQL query to execute
            params: Query parameters
            use_transaction: Whether to use transaction (SQLite only)

        Returns:
            Success status
        """
        params = params or []

        try:
            # Write to both databases if available
            success = True

            # DuckDB write
            if self.conn:
                try:
                    self.conn.execute(query, params)
                except Exception as e:
                    logger.error(f"DuckDB write failed: {e}")
                    success = False

            # SQLite write
            try:
                if use_transaction:
                    self.sqlite_conn.execute("BEGIN")
                self.sqlite_conn.execute(query, params)
                if use_transaction:
                    self.sqlite_conn.commit()
            except Exception as e:
                logger.error(f"SQLite write failed: {e}")
                if use_transaction:
                    self.sqlite_conn.rollback()
                success = False

            return success

        except Exception as e:
            logger.error(f"Write operation failed: {e}")
            return False

    # =====================================================
    # PROJECT MANAGEMENT
    # =====================================================

    def upsert_project(self, project_id: str, project_code: str, project_name: str,
                      full_path: str, range_folder: str = None, metadata: Dict = None) -> bool:
        """Insert or update a project record."""
        query = """
        INSERT OR REPLACE INTO projects
        (project_id, project_code, project_name, full_path, range_folder, metadata, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """
        params = [
            project_id, project_code, project_name, full_path, range_folder,
            json.dumps(metadata) if metadata else None
        ]
        return self.execute_write(query, params)

    def get_project(self, project_id: str) -> Optional[Dict]:
        """Get project by ID."""
        query = "SELECT * FROM projects WHERE project_id = ?"
        results = self.execute_query(query, [project_id])
        return results[0] if results else None

    def search_projects(self, search_term: str, limit: int = 10) -> List[Dict]:
        """Search projects by code or name."""
        query = """
        SELECT * FROM projects
        WHERE project_code LIKE ? OR project_name LIKE ?
        ORDER BY
            CASE
                WHEN project_code = ? THEN 1
                WHEN project_code LIKE ? THEN 2
                WHEN project_name LIKE ? THEN 3
                ELSE 4
            END,
            project_code
        LIMIT ?
        """
        search_pattern = f"%{search_term}%"
        exact_match = search_term
        prefix_match = f"{search_term}%"

        params = [search_pattern, search_pattern, exact_match, prefix_match, prefix_match, limit]
        return self.execute_query(query, params)

    # =====================================================
    # DOCUMENT MANAGEMENT
    # =====================================================

    def upsert_document(self, document_data: Dict) -> bool:
        """Insert or update a document record."""
        # Generate document ID if not provided
        if 'document_id' not in document_data:
            document_data['document_id'] = str(uuid.uuid4())

        # Convert arrays to JSON strings for SQLite compatibility
        if 'status_tags' in document_data and isinstance(document_data['status_tags'], list):
            document_data['status_tags'] = json.dumps(document_data['status_tags'])

        if 'embedding_vector' in document_data and isinstance(document_data['embedding_vector'], list):
            document_data['embedding_vector'] = json.dumps(document_data['embedding_vector'])

        query = """
        INSERT OR REPLACE INTO documents
        (document_id, project_id, file_path, filename, file_extension, file_size_bytes,
         document_type, folder_category, version_string, version_numeric, version_type,
         status_tags, status_weight, document_date, date_format, modified_at,
         content_hash, word_count, page_count, content_preview, embedding_vector,
         classification_confidence, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """

        params = [
            document_data.get('document_id'),
            document_data.get('project_id'),
            document_data.get('file_path'),
            document_data.get('filename'),
            document_data.get('file_extension'),
            document_data.get('file_size_bytes'),
            document_data.get('document_type'),
            document_data.get('folder_category'),
            document_data.get('version_string'),
            document_data.get('version_numeric'),
            document_data.get('version_type'),
            document_data.get('status_tags'),
            document_data.get('status_weight'),
            document_data.get('document_date'),
            document_data.get('date_format'),
            document_data.get('modified_at'),
            document_data.get('content_hash'),
            document_data.get('word_count'),
            document_data.get('page_count'),
            document_data.get('content_preview'),
            document_data.get('embedding_vector'),
            document_data.get('classification_confidence')
        ]

        return self.execute_write(query, params)

    def get_latest_documents(self, project_id: str, document_type: str = None) -> List[Dict]:
        """Get latest version of documents for a project."""
        query = """
        SELECT * FROM v_latest_documents
        WHERE project_id = ?
        """
        params = [project_id]

        if document_type:
            query += " AND document_type = ?"
            params.append(document_type)

        query += " ORDER BY status_weight DESC, version_numeric DESC"

        return self.execute_query(query, params, prefer_analytics=True)

    def get_document_versions(self, project_id: str, document_type: str) -> List[Dict]:
        """Get all versions of a specific document type for a project."""
        query = """
        SELECT filename, version_string, version_numeric, document_date, status_tags, file_path
        FROM documents
        WHERE project_id = ? AND document_type = ?
        ORDER BY version_numeric DESC, document_date DESC
        """
        return self.execute_query(query, [project_id, document_type])

    # =====================================================
    # USER ACTIVITY TRACKING
    # =====================================================

    def start_session(self, user_id: str = 'default_user', app_version: str = None,
                     os_platform: str = None) -> str:
        """Start a new user session."""
        session_id = str(uuid.uuid4())
        query = """
        INSERT INTO user_sessions
        (session_id, user_id, app_version, os_platform)
        VALUES (?, ?, ?, ?)
        """
        params = [session_id, user_id, app_version, os_platform]

        if self.execute_write(query, params):
            return session_id
        return None

    def end_session(self, session_id: str) -> bool:
        """End a user session and calculate duration."""
        query = """
        UPDATE user_sessions
        SET session_end = CURRENT_TIMESTAMP,
            session_duration_seconds = CAST((julianday(CURRENT_TIMESTAMP) - julianday(session_start)) * 86400 AS INTEGER)
        WHERE session_id = ?
        """
        return self.execute_write(query, [session_id])

    def record_activity(self, session_id: str, activity_type: str,
                       project_id: str = None, document_id: str = None,
                       search_query: str = None, search_results_count: int = None,
                       action_details: Dict = None, response_time_ms: int = None,
                       input_method: str = None, ui_component: str = None,
                       success: bool = True, error_message: str = None) -> str:
        """Record a user activity."""
        activity_id = str(uuid.uuid4())

        query = """
        INSERT INTO user_activities
        (activity_id, session_id, activity_type, project_id, document_id,
         search_query, search_results_count, action_details, response_time_ms,
         input_method, ui_component, success, error_message)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        params = [
            activity_id, session_id, activity_type, project_id, document_id,
            search_query, search_results_count,
            json.dumps(action_details) if action_details else None,
            response_time_ms, input_method, ui_component, success, error_message
        ]

        if self.execute_write(query, params):
            return activity_id
        return None

    # =====================================================
    # ANALYTICS QUERIES
    # =====================================================

    def get_recent_activity(self, days: int = 7) -> List[Dict]:
        """Get recent activity summary."""
        query = """
        SELECT * FROM v_recent_activity
        WHERE activity_date > DATE('now', '-{} days')
        ORDER BY activity_date DESC, count DESC
        """.format(days)
        return self.execute_query(query, prefer_analytics=True)

    def get_popular_projects(self, days: int = 30, limit: int = 10) -> List[Dict]:
        """Get most popular projects."""
        query = """
        SELECT * FROM v_popular_projects
        LIMIT ?
        """
        return self.execute_query(query, [limit], prefer_analytics=True)

    def get_user_productivity(self, days: int = 30) -> List[Dict]:
        """Get user productivity metrics."""
        query = """
        SELECT * FROM v_user_productivity
        WHERE date > DATE('now', '-{} days')
        ORDER BY date DESC
        """.format(days)
        return self.execute_query(query, prefer_analytics=True)

    def get_search_analytics(self, days: int = 7) -> List[Dict]:
        """Get search analytics."""
        query = """
        SELECT
            query_type,
            COUNT(*) as search_count,
            AVG(search_time_ms) as avg_search_time,
            AVG(results_count) as avg_results,
            SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END) as cache_hits
        FROM search_analytics
        WHERE timestamp > datetime('now', '-{} days')
        GROUP BY query_type
        ORDER BY search_count DESC
        """.format(days)
        return self.execute_query(query, prefer_analytics=True)

    # =====================================================
    # ML FEATURES
    # =====================================================

    def store_ml_features(self, feature_type: str, entity_id: str,
                         feature_vector: List[float], metadata: Dict = None,
                         model_version: str = None) -> str:
        """Store ML features for an entity."""
        feature_id = str(uuid.uuid4())

        query = """
        INSERT OR REPLACE INTO ml_features
        (feature_id, feature_type, entity_id, feature_vector, feature_metadata,
         model_version, training_timestamp)
        VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """

        params = [
            feature_id, feature_type, entity_id,
            json.dumps(feature_vector),
            json.dumps(metadata) if metadata else None,
            model_version
        ]

        if self.execute_write(query, params):
            return feature_id
        return None

    def get_ml_features(self, feature_type: str, entity_id: str = None) -> List[Dict]:
        """Retrieve ML features."""
        query = "SELECT * FROM ml_features WHERE feature_type = ?"
        params = [feature_type]

        if entity_id:
            query += " AND entity_id = ?"
            params.append(entity_id)

        query += " ORDER BY updated_at DESC"

        return self.execute_query(query, params)

    def get_user_item_matrix(self, days: int = 90) -> List[Dict]:
        """Get user-item interaction matrix for recommendations."""
        query = """
        SELECT
            session_id as user_proxy,
            project_id,
            COUNT(*) as interaction_count,
            SUM(CASE WHEN activity_type = 'document_open' THEN 2 ELSE 1 END) as weighted_score,
            MAX(timestamp) as last_interaction
        FROM user_activities
        WHERE timestamp > datetime('now', '-{} days')
            AND project_id IS NOT NULL
        GROUP BY session_id, project_id
        ORDER BY weighted_score DESC
        """.format(days)
        return self.execute_query(query, prefer_analytics=True)

    # =====================================================
    # AI CONVERSATION TRACKING
    # =====================================================

    def start_ai_conversation(self, session_id: str, model_used: str = None) -> str:
        """Start a new AI conversation."""
        conversation_id = str(uuid.uuid4())
        query = """
        INSERT INTO ai_conversations
        (conversation_id, session_id, model_used)
        VALUES (?, ?, ?)
        """
        params = [conversation_id, session_id, model_used]

        if self.execute_write(query, params):
            return conversation_id
        return None

    def add_ai_message(self, conversation_id: str, role: str, content: str,
                      tool_calls: List = None, tool_results: Dict = None,
                      token_count: int = None, response_time_ms: int = None) -> str:
        """Add a message to an AI conversation."""
        message_id = str(uuid.uuid4())

        query = """
        INSERT INTO ai_messages
        (message_id, conversation_id, role, content, tool_calls, tool_results,
         token_count, response_time_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """

        params = [
            message_id, conversation_id, role, content,
            json.dumps(tool_calls) if tool_calls else None,
            json.dumps(tool_results) if tool_results else None,
            token_count, response_time_ms
        ]

        if self.execute_write(query, params):
            # Update conversation totals
            self._update_conversation_stats(conversation_id)
            return message_id
        return None

    def _update_conversation_stats(self, conversation_id: str):
        """Update conversation statistics."""
        query = """
        UPDATE ai_conversations
        SET total_messages = (
                SELECT COUNT(*) FROM ai_messages
                WHERE conversation_id = ?
            ),
            total_tokens = (
                SELECT COALESCE(SUM(token_count), 0) FROM ai_messages
                WHERE conversation_id = ?
            )
        WHERE conversation_id = ?
        """
        self.execute_write(query, [conversation_id, conversation_id, conversation_id])

    # =====================================================
    # DATA ARCHIVAL AND MAINTENANCE
    # =====================================================

    def archive_old_activities(self, days: int = 30) -> int:
        """Archive activities older than specified days."""
        # First, create archive table if it doesn't exist
        create_archive_query = """
        CREATE TABLE IF NOT EXISTS user_activities_archive AS
        SELECT * FROM user_activities WHERE 0 = 1
        """
        self.execute_write(create_archive_query)

        # Move old data to archive
        archive_query = """
        INSERT INTO user_activities_archive
        SELECT * FROM user_activities
        WHERE timestamp < datetime('now', '-{} days')
        """.format(days)

        # Count rows to be archived
        count_query = """
        SELECT COUNT(*) as count FROM user_activities
        WHERE timestamp < datetime('now', '-{} days')
        """.format(days)

        count_result = self.execute_query(count_query)
        rows_to_archive = count_result[0]['count'] if count_result else 0

        if rows_to_archive > 0:
            # Archive the data
            self.execute_write(archive_query)

            # Delete from main table
            delete_query = """
            DELETE FROM user_activities
            WHERE timestamp < datetime('now', '-{} days')
            """.format(days)
            self.execute_write(delete_query)

            logger.info(f"Archived {rows_to_archive} activity records older than {days} days")

        return rows_to_archive

    def cleanup_old_ai_messages(self, days: int = 60) -> int:
        """Clean up old AI messages."""
        count_query = """
        SELECT COUNT(*) as count FROM ai_messages
        WHERE timestamp < datetime('now', '-{} days')
        """.format(days)

        count_result = self.execute_query(count_query)
        rows_to_delete = count_result[0]['count'] if count_result else 0

        if rows_to_delete > 0:
            delete_query = """
            DELETE FROM ai_messages
            WHERE timestamp < datetime('now', '-{} days')
            """.format(days)
            self.execute_write(delete_query)

            logger.info(f"Cleaned up {rows_to_delete} AI messages older than {days} days")

        return rows_to_delete

    def vacuum_database(self):
        """Optimize database by reclaiming space."""
        try:
            if self.conn:
                self.conn.execute("VACUUM")
            self.sqlite_conn.execute("VACUUM")
            logger.info("Database vacuum completed successfully")
        except Exception as e:
            logger.error(f"Database vacuum failed: {e}")

    # =====================================================
    # UTILITIES
    # =====================================================

    def get_database_stats(self) -> Dict:
        """Get database statistics."""
        stats = {}

        # Count records in main tables
        tables = [
            'projects', 'documents', 'user_sessions', 'user_activities',
            'ai_conversations', 'ai_messages', 'search_analytics', 'ml_features'
        ]

        for table in tables:
            try:
                result = self.execute_query(f"SELECT COUNT(*) as count FROM {table}")
                stats[table] = result[0]['count'] if result else 0
            except:
                stats[table] = 0

        # Database file sizes
        if self.db_path.exists():
            stats['database_size_mb'] = sum(
                f.stat().st_size for f in self.db_path.glob('*.db*')
            ) / (1024 * 1024)

        return stats

    def close(self):
        """Close database connections."""
        try:
            if self.conn:
                self.conn.close()
            if self.sqlite_conn:
                self.sqlite_conn.close()
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Singleton instance for global use
_db_manager_instance = None


def get_database_manager(db_path: Optional[str] = None) -> DatabaseManager:
    """Get singleton database manager instance."""
    global _db_manager_instance
    if _db_manager_instance is None:
        _db_manager_instance = DatabaseManager(db_path)
    return _db_manager_instance