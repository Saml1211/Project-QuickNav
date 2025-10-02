"""
Data Management System for Project QuickNav

This module provides comprehensive data persistence, management, and synchronization
capabilities including:
- Project metadata storage and retrieval
- Activity tracking and access logs
- Notes/comments system
- Backup and restore functionality
- Export/import capabilities
- Data synchronization options
- Data integrity validation
"""

import os
import json
import sqlite3
import threading
import hashlib
import shutil
import gzip
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProjectMetadata:
    """Project metadata structure."""
    project_number: str
    project_name: str
    project_path: str
    created_date: datetime
    last_accessed: datetime
    access_count: int = 0
    favorite: bool = False
    tags: List[str] = None
    custom_fields: Dict[str, Any] = None
    notes: str = ""

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.custom_fields is None:
            self.custom_fields = {}


@dataclass
class ActivityLog:
    """Activity log entry structure."""
    id: str
    timestamp: datetime
    project_number: str
    project_name: str
    action: str  # 'access', 'search', 'document_open', 'note_add', etc.
    details: Dict[str, Any]
    duration_ms: Optional[int] = None

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class ProjectNote:
    """Project note structure."""
    id: str
    project_number: str
    title: str
    content: str
    created_date: datetime
    modified_date: datetime
    tags: List[str] = None
    priority: str = "normal"  # low, normal, high, urgent

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if self.tags is None:
            self.tags = []


class DataManager:
    """Comprehensive data management system."""

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize data manager.

        Args:
            data_dir: Custom data directory path
        """
        self.data_dir = Path(data_dir) if data_dir else self._get_default_data_dir()
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Database and file paths
        self.db_path = self.data_dir / "quicknav.db"
        self.metadata_file = self.data_dir / "project_metadata.json"
        self.config_file = self.data_dir / "data_config.json"
        self.backup_dir = self.data_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)

        # Thread safety
        self._lock = threading.RLock()
        self._db_lock = threading.RLock()

        # Configuration
        self.config = self._load_config()

        # Initialize database
        self._init_database()

        # In-memory caches
        self._metadata_cache = {}
        self._activity_cache = []
        self._cache_dirty = False

        logger.info(f"DataManager initialized with data directory: {self.data_dir}")

    def _get_default_data_dir(self) -> Path:
        """Get default data directory path."""
        if os.name == 'nt':  # Windows
            base_dir = Path(os.environ.get('APPDATA', '')) / 'QuickNav'
        else:  # Unix-like
            base_dir = Path.home() / '.local' / 'share' / 'quicknav'

        return base_dir / 'data'

    def _load_config(self) -> Dict[str, Any]:
        """Load data manager configuration."""
        default_config = {
            "version": "1.0.0",
            "auto_backup": True,
            "backup_interval_hours": 24,
            "max_backups": 30,
            "activity_log_retention_days": 365,
            "cache_enabled": True,
            "compression_enabled": True,
            "sync_enabled": False,
            "sync_provider": "none",  # none, onedrive, dropbox, gdrive
            "sync_config": {},
            "export_formats": ["json", "csv", "xlsx"],
            "integrity_check_enabled": True,
            "auto_repair_enabled": True
        }

        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in loaded_config:
                        loaded_config[key] = value
                return loaded_config
        except Exception as e:
            logger.warning(f"Failed to load data config: {e}")

        return default_config

    def _save_config(self):
        """Save data manager configuration."""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save data config: {e}")

    def _init_database(self):
        """Initialize SQLite database."""
        try:
            with self._db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                # Create tables
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS project_metadata (
                        project_number TEXT PRIMARY KEY,
                        project_name TEXT NOT NULL,
                        project_path TEXT NOT NULL,
                        created_date TEXT NOT NULL,
                        last_accessed TEXT NOT NULL,
                        access_count INTEGER DEFAULT 0,
                        favorite INTEGER DEFAULT 0,
                        tags TEXT,
                        custom_fields TEXT,
                        notes TEXT,
                        data_hash TEXT
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS activity_log (
                        id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        project_number TEXT,
                        project_name TEXT,
                        action TEXT NOT NULL,
                        details TEXT,
                        duration_ms INTEGER
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS project_notes (
                        id TEXT PRIMARY KEY,
                        project_number TEXT NOT NULL,
                        title TEXT NOT NULL,
                        content TEXT,
                        created_date TEXT NOT NULL,
                        modified_date TEXT NOT NULL,
                        tags TEXT,
                        priority TEXT DEFAULT 'normal'
                    )
                """)

                # Create indexes
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_project_last_accessed ON project_metadata(last_accessed)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_activity_timestamp ON activity_log(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_activity_project ON activity_log(project_number)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_notes_project ON project_notes(project_number)")

                conn.commit()
                conn.close()

                logger.info("Database initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    # Project Metadata Management
    def get_project_metadata(self, project_number: str) -> Optional[ProjectMetadata]:
        """Get project metadata by project number."""
        with self._lock:
            # Check cache first
            if project_number in self._metadata_cache:
                return self._metadata_cache[project_number]

            try:
                with self._db_lock:
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()

                    cursor.execute("""
                        SELECT * FROM project_metadata WHERE project_number = ?
                    """, (project_number,))

                    row = cursor.fetchone()
                    conn.close()

                    if row:
                        metadata = ProjectMetadata(
                            project_number=row[0],
                            project_name=row[1],
                            project_path=row[2],
                            created_date=datetime.fromisoformat(row[3]),
                            last_accessed=datetime.fromisoformat(row[4]),
                            access_count=row[5],
                            favorite=bool(row[6]),
                            tags=json.loads(row[7]) if row[7] else [],
                            custom_fields=json.loads(row[8]) if row[8] else {},
                            notes=row[9] or ""
                        )

                        # Cache the result
                        self._metadata_cache[project_number] = metadata
                        return metadata

            except Exception as e:
                logger.error(f"Failed to get project metadata: {e}")

        return None

    def save_project_metadata(self, metadata: ProjectMetadata) -> bool:
        """Save or update project metadata."""
        with self._lock:
            try:
                # Calculate data hash for integrity
                data_for_hash = json.dumps(asdict(metadata), sort_keys=True, default=str)
                data_hash = hashlib.sha256(data_for_hash.encode()).hexdigest()

                with self._db_lock:
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()

                    cursor.execute("""
                        INSERT OR REPLACE INTO project_metadata
                        (project_number, project_name, project_path, created_date,
                         last_accessed, access_count, favorite, tags, custom_fields,
                         notes, data_hash)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        metadata.project_number,
                        metadata.project_name,
                        metadata.project_path,
                        metadata.created_date.isoformat(),
                        metadata.last_accessed.isoformat(),
                        metadata.access_count,
                        int(metadata.favorite),
                        json.dumps(metadata.tags),
                        json.dumps(metadata.custom_fields),
                        metadata.notes,
                        data_hash
                    ))

                    conn.commit()
                    conn.close()

                # Update cache
                self._metadata_cache[metadata.project_number] = metadata
                self._cache_dirty = True

                logger.debug(f"Saved metadata for project {metadata.project_number}")
                return True

            except Exception as e:
                logger.error(f"Failed to save project metadata: {e}")
                return False

    def update_project_access(self, project_number: str, project_name: str, project_path: str):
        """Update project access time and count."""
        metadata = self.get_project_metadata(project_number)

        if metadata is None:
            # Create new metadata
            metadata = ProjectMetadata(
                project_number=project_number,
                project_name=project_name,
                project_path=project_path,
                created_date=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1
            )
        else:
            # Update existing
            metadata.last_accessed = datetime.now()
            metadata.access_count += 1
            metadata.project_name = project_name  # Update in case it changed
            metadata.project_path = project_path

        self.save_project_metadata(metadata)

        # Log activity
        self.log_activity(
            project_number=project_number,
            project_name=project_name,
            action="access",
            details={"path": project_path}
        )

    def get_recent_projects(self, limit: int = 20) -> List[ProjectMetadata]:
        """Get recently accessed projects."""
        with self._lock:
            try:
                with self._db_lock:
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()

                    cursor.execute("""
                        SELECT * FROM project_metadata
                        ORDER BY last_accessed DESC
                        LIMIT ?
                    """, (limit,))

                    rows = cursor.fetchall()
                    conn.close()

                    projects = []
                    for row in rows:
                        metadata = ProjectMetadata(
                            project_number=row[0],
                            project_name=row[1],
                            project_path=row[2],
                            created_date=datetime.fromisoformat(row[3]),
                            last_accessed=datetime.fromisoformat(row[4]),
                            access_count=row[5],
                            favorite=bool(row[6]),
                            tags=json.loads(row[7]) if row[7] else [],
                            custom_fields=json.loads(row[8]) if row[8] else {},
                            notes=row[9] or ""
                        )
                        projects.append(metadata)

                    return projects

            except Exception as e:
                logger.error(f"Failed to get recent projects: {e}")
                return []

    def get_favorite_projects(self) -> List[ProjectMetadata]:
        """Get favorite projects."""
        with self._lock:
            try:
                with self._db_lock:
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()

                    cursor.execute("""
                        SELECT * FROM project_metadata
                        WHERE favorite = 1
                        ORDER BY last_accessed DESC
                    """)

                    rows = cursor.fetchall()
                    conn.close()

                    projects = []
                    for row in rows:
                        metadata = ProjectMetadata(
                            project_number=row[0],
                            project_name=row[1],
                            project_path=row[2],
                            created_date=datetime.fromisoformat(row[3]),
                            last_accessed=datetime.fromisoformat(row[4]),
                            access_count=row[5],
                            favorite=bool(row[6]),
                            tags=json.loads(row[7]) if row[7] else [],
                            custom_fields=json.loads(row[8]) if row[8] else {},
                            notes=row[9] or ""
                        )
                        projects.append(metadata)

                    return projects

            except Exception as e:
                logger.error(f"Failed to get favorite projects: {e}")
                return []

    def search_projects(self, query: str, tags: List[str] = None) -> List[ProjectMetadata]:
        """Search projects by name, number, or tags."""
        with self._lock:
            try:
                with self._db_lock:
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()

                    # Build search query
                    where_conditions = []
                    params = []

                    if query:
                        where_conditions.append("""
                            (project_number LIKE ? OR project_name LIKE ? OR notes LIKE ?)
                        """)
                        query_param = f"%{query}%"
                        params.extend([query_param, query_param, query_param])

                    if tags:
                        for tag in tags:
                            where_conditions.append("tags LIKE ?")
                            params.append(f"%{tag}%")

                    where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"

                    cursor.execute(f"""
                        SELECT * FROM project_metadata
                        WHERE {where_clause}
                        ORDER BY last_accessed DESC
                    """, params)

                    rows = cursor.fetchall()
                    conn.close()

                    projects = []
                    for row in rows:
                        metadata = ProjectMetadata(
                            project_number=row[0],
                            project_name=row[1],
                            project_path=row[2],
                            created_date=datetime.fromisoformat(row[3]),
                            last_accessed=datetime.fromisoformat(row[4]),
                            access_count=row[5],
                            favorite=bool(row[6]),
                            tags=json.loads(row[7]) if row[7] else [],
                            custom_fields=json.loads(row[8]) if row[8] else {},
                            notes=row[9] or ""
                        )
                        projects.append(metadata)

                    return projects

            except Exception as e:
                logger.error(f"Failed to search projects: {e}")
                return []

    # Activity Logging
    def log_activity(self, project_number: str, project_name: str, action: str,
                    details: Dict[str, Any] = None, duration_ms: Optional[int] = None):
        """Log user activity."""
        if not self.config.get("activity_log_enabled", True):
            return

        activity = ActivityLog(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            project_number=project_number,
            project_name=project_name,
            action=action,
            details=details or {},
            duration_ms=duration_ms
        )

        # Add to memory cache for performance
        self._activity_cache.append(activity)

        # Batch write to database
        if len(self._activity_cache) >= 10:
            self._flush_activity_cache()

    def _flush_activity_cache(self):
        """Flush activity cache to database."""
        if not self._activity_cache:
            return

        try:
            with self._db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                for activity in self._activity_cache:
                    cursor.execute("""
                        INSERT INTO activity_log
                        (id, timestamp, project_number, project_name, action, details, duration_ms)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        activity.id,
                        activity.timestamp.isoformat(),
                        activity.project_number,
                        activity.project_name,
                        activity.action,
                        json.dumps(activity.details),
                        activity.duration_ms
                    ))

                conn.commit()
                conn.close()

                logger.debug(f"Flushed {len(self._activity_cache)} activities to database")
                self._activity_cache.clear()

        except Exception as e:
            logger.error(f"Failed to flush activity cache: {e}")

    def get_activity_logs(self, project_number: str = None, action: str = None,
                         start_date: datetime = None, end_date: datetime = None,
                         limit: int = 100) -> List[ActivityLog]:
        """Get activity logs with optional filtering."""
        try:
            with self._db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                # Build query
                where_conditions = []
                params = []

                if project_number:
                    where_conditions.append("project_number = ?")
                    params.append(project_number)

                if action:
                    where_conditions.append("action = ?")
                    params.append(action)

                if start_date:
                    where_conditions.append("timestamp >= ?")
                    params.append(start_date.isoformat())

                if end_date:
                    where_conditions.append("timestamp <= ?")
                    params.append(end_date.isoformat())

                where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"

                cursor.execute(f"""
                    SELECT * FROM activity_log
                    WHERE {where_clause}
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, params + [limit])

                rows = cursor.fetchall()
                conn.close()

                activities = []
                for row in rows:
                    activity = ActivityLog(
                        id=row[0],
                        timestamp=datetime.fromisoformat(row[1]),
                        project_number=row[2],
                        project_name=row[3],
                        action=row[4],
                        details=json.loads(row[5]) if row[5] else {},
                        duration_ms=row[6]
                    )
                    activities.append(activity)

                return activities

        except Exception as e:
            logger.error(f"Failed to get activity logs: {e}")
            return []

    def get_activity_stats(self, days: int = 30) -> Dict[str, Any]:
        """Get activity statistics."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        try:
            with self._db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                # Total activities
                cursor.execute("""
                    SELECT COUNT(*) FROM activity_log
                    WHERE timestamp >= ?
                """, (start_date.isoformat(),))
                total_activities = cursor.fetchone()[0]

                # Activities by action
                cursor.execute("""
                    SELECT action, COUNT(*) FROM activity_log
                    WHERE timestamp >= ?
                    GROUP BY action
                    ORDER BY COUNT(*) DESC
                """, (start_date.isoformat(),))
                by_action = dict(cursor.fetchall())

                # Most accessed projects
                cursor.execute("""
                    SELECT project_number, project_name, COUNT(*) as access_count
                    FROM activity_log
                    WHERE timestamp >= ? AND action = 'access'
                    GROUP BY project_number, project_name
                    ORDER BY access_count DESC
                    LIMIT 10
                """, (start_date.isoformat(),))
                top_projects = [
                    {"project_number": row[0], "project_name": row[1], "access_count": row[2]}
                    for row in cursor.fetchall()
                ]

                # Daily activity counts
                cursor.execute("""
                    SELECT DATE(timestamp) as date, COUNT(*) as count
                    FROM activity_log
                    WHERE timestamp >= ?
                    GROUP BY DATE(timestamp)
                    ORDER BY date
                """, (start_date.isoformat(),))
                daily_activity = dict(cursor.fetchall())

                conn.close()

                return {
                    "period_days": days,
                    "total_activities": total_activities,
                    "by_action": by_action,
                    "top_projects": top_projects,
                    "daily_activity": daily_activity
                }

        except Exception as e:
            logger.error(f"Failed to get activity stats: {e}")
            return {}

    # Notes System
    def add_project_note(self, project_number: str, title: str, content: str,
                        tags: List[str] = None, priority: str = "normal") -> str:
        """Add a note to a project."""
        note = ProjectNote(
            id=str(uuid.uuid4()),
            project_number=project_number,
            title=title,
            content=content,
            created_date=datetime.now(),
            modified_date=datetime.now(),
            tags=tags or [],
            priority=priority
        )

        try:
            with self._db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO project_notes
                    (id, project_number, title, content, created_date, modified_date, tags, priority)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    note.id,
                    note.project_number,
                    note.title,
                    note.content,
                    note.created_date.isoformat(),
                    note.modified_date.isoformat(),
                    json.dumps(note.tags),
                    note.priority
                ))

                conn.commit()
                conn.close()

                logger.debug(f"Added note to project {project_number}")
                return note.id

        except Exception as e:
            logger.error(f"Failed to add project note: {e}")
            return ""

    def get_project_notes(self, project_number: str) -> List[ProjectNote]:
        """Get all notes for a project."""
        try:
            with self._db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT * FROM project_notes
                    WHERE project_number = ?
                    ORDER BY created_date DESC
                """, (project_number,))

                rows = cursor.fetchall()
                conn.close()

                notes = []
                for row in rows:
                    note = ProjectNote(
                        id=row[0],
                        project_number=row[1],
                        title=row[2],
                        content=row[3],
                        created_date=datetime.fromisoformat(row[4]),
                        modified_date=datetime.fromisoformat(row[5]),
                        tags=json.loads(row[6]) if row[6] else [],
                        priority=row[7]
                    )
                    notes.append(note)

                return notes

        except Exception as e:
            logger.error(f"Failed to get project notes: {e}")
            return []

    def update_project_note(self, note_id: str, title: str = None, content: str = None,
                           tags: List[str] = None, priority: str = None) -> bool:
        """Update a project note."""
        try:
            with self._db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                # Get current note
                cursor.execute("SELECT * FROM project_notes WHERE id = ?", (note_id,))
                row = cursor.fetchone()

                if not row:
                    return False

                # Update fields
                new_title = title if title is not None else row[2]
                new_content = content if content is not None else row[3]
                new_tags = json.dumps(tags) if tags is not None else row[6]
                new_priority = priority if priority is not None else row[7]

                cursor.execute("""
                    UPDATE project_notes
                    SET title = ?, content = ?, modified_date = ?, tags = ?, priority = ?
                    WHERE id = ?
                """, (
                    new_title,
                    new_content,
                    datetime.now().isoformat(),
                    new_tags,
                    new_priority,
                    note_id
                ))

                conn.commit()
                conn.close()

                return True

        except Exception as e:
            logger.error(f"Failed to update project note: {e}")
            return False

    def delete_project_note(self, note_id: str) -> bool:
        """Delete a project note."""
        try:
            with self._db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cursor.execute("DELETE FROM project_notes WHERE id = ?", (note_id,))

                conn.commit()
                conn.close()

                return True

        except Exception as e:
            logger.error(f"Failed to delete project note: {e}")
            return False

    # Backup and Restore
    def create_backup(self, description: str = "") -> str:
        """Create a backup of all data."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"quicknav_backup_{timestamp}"
        backup_path = self.backup_dir / f"{backup_name}.json"

        try:
            # Flush any pending activities
            self._flush_activity_cache()

            # Create backup data
            backup_data = {
                "version": "1.0.0",
                "timestamp": timestamp,
                "description": description,
                "metadata": {},
                "activity_logs": [],
                "notes": [],
                "config": self.config
            }

            # Export project metadata
            with self._db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                # Get all metadata
                cursor.execute("SELECT * FROM project_metadata")
                for row in cursor.fetchall():
                    backup_data["metadata"][row[0]] = {
                        "project_number": row[0],
                        "project_name": row[1],
                        "project_path": row[2],
                        "created_date": row[3],
                        "last_accessed": row[4],
                        "access_count": row[5],
                        "favorite": bool(row[6]),
                        "tags": json.loads(row[7]) if row[7] else [],
                        "custom_fields": json.loads(row[8]) if row[8] else {},
                        "notes": row[9] or ""
                    }

                # Get activity logs (last 30 days to keep size reasonable)
                thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()
                cursor.execute("""
                    SELECT * FROM activity_log
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                """, (thirty_days_ago,))

                for row in cursor.fetchall():
                    backup_data["activity_logs"].append({
                        "id": row[0],
                        "timestamp": row[1],
                        "project_number": row[2],
                        "project_name": row[3],
                        "action": row[4],
                        "details": json.loads(row[5]) if row[5] else {},
                        "duration_ms": row[6]
                    })

                # Get all notes
                cursor.execute("SELECT * FROM project_notes")
                for row in cursor.fetchall():
                    backup_data["notes"].append({
                        "id": row[0],
                        "project_number": row[1],
                        "title": row[2],
                        "content": row[3],
                        "created_date": row[4],
                        "modified_date": row[5],
                        "tags": json.loads(row[6]) if row[6] else [],
                        "priority": row[7]
                    })

                conn.close()

            # Save backup
            if self.config.get("compression_enabled", True):
                with gzip.open(f"{backup_path}.gz", 'wt', encoding='utf-8') as f:
                    json.dump(backup_data, f, indent=2, ensure_ascii=False)
                backup_path = f"{backup_path}.gz"
            else:
                with open(backup_path, 'w', encoding='utf-8') as f:
                    json.dump(backup_data, f, indent=2, ensure_ascii=False)

            # Cleanup old backups
            self._cleanup_old_backups()

            logger.info(f"Created backup: {backup_path}")
            return str(backup_path)

        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise

    def restore_backup(self, backup_path: str, merge: bool = True) -> bool:
        """Restore data from backup."""
        try:
            backup_path = Path(backup_path)

            if not backup_path.exists():
                raise FileNotFoundError(f"Backup file not found: {backup_path}")

            # Load backup data
            if backup_path.suffix == '.gz':
                with gzip.open(backup_path, 'rt', encoding='utf-8') as f:
                    backup_data = json.load(f)
            else:
                with open(backup_path, 'r', encoding='utf-8') as f:
                    backup_data = json.load(f)

            # Create current backup before restore
            if not merge:
                self.create_backup("Pre-restore backup")

            with self._db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                if not merge:
                    # Clear existing data
                    cursor.execute("DELETE FROM project_metadata")
                    cursor.execute("DELETE FROM activity_log")
                    cursor.execute("DELETE FROM project_notes")

                # Restore metadata
                for project_data in backup_data.get("metadata", {}).values():
                    cursor.execute("""
                        INSERT OR REPLACE INTO project_metadata
                        (project_number, project_name, project_path, created_date,
                         last_accessed, access_count, favorite, tags, custom_fields, notes)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        project_data["project_number"],
                        project_data["project_name"],
                        project_data["project_path"],
                        project_data["created_date"],
                        project_data["last_accessed"],
                        project_data["access_count"],
                        int(project_data["favorite"]),
                        json.dumps(project_data["tags"]),
                        json.dumps(project_data["custom_fields"]),
                        project_data["notes"]
                    ))

                # Restore activity logs (if not merging or activity doesn't exist)
                for activity in backup_data.get("activity_logs", []):
                    if merge:
                        cursor.execute("SELECT id FROM activity_log WHERE id = ?", (activity["id"],))
                        if cursor.fetchone():
                            continue  # Skip existing

                    cursor.execute("""
                        INSERT OR REPLACE INTO activity_log
                        (id, timestamp, project_number, project_name, action, details, duration_ms)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        activity["id"],
                        activity["timestamp"],
                        activity["project_number"],
                        activity["project_name"],
                        activity["action"],
                        json.dumps(activity["details"]),
                        activity["duration_ms"]
                    ))

                # Restore notes
                for note in backup_data.get("notes", []):
                    if merge:
                        cursor.execute("SELECT id FROM project_notes WHERE id = ?", (note["id"],))
                        if cursor.fetchone():
                            continue  # Skip existing

                    cursor.execute("""
                        INSERT OR REPLACE INTO project_notes
                        (id, project_number, title, content, created_date, modified_date, tags, priority)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        note["id"],
                        note["project_number"],
                        note["title"],
                        note["content"],
                        note["created_date"],
                        note["modified_date"],
                        json.dumps(note["tags"]),
                        note["priority"]
                    ))

                conn.commit()
                conn.close()

            # Clear caches
            self._metadata_cache.clear()
            self._activity_cache.clear()

            logger.info(f"Restored backup from: {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            raise

    def _cleanup_old_backups(self):
        """Clean up old backup files."""
        try:
            max_backups = self.config.get("max_backups", 30)
            backup_files = []

            for file_path in self.backup_dir.glob("quicknav_backup_*.json*"):
                backup_files.append((file_path, file_path.stat().st_mtime))

            # Sort by modification time (newest first)
            backup_files.sort(key=lambda x: x[1], reverse=True)

            # Remove excess backups
            for file_path, _ in backup_files[max_backups:]:
                try:
                    file_path.unlink()
                    logger.debug(f"Deleted old backup: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete backup {file_path}: {e}")

        except Exception as e:
            logger.warning(f"Failed to cleanup old backups: {e}")

    def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups."""
        backups = []

        try:
            for file_path in self.backup_dir.glob("quicknav_backup_*.json*"):
                stat = file_path.stat()

                # Try to extract timestamp from filename
                filename = file_path.stem.replace('.json', '')
                if filename.startswith('quicknav_backup_'):
                    timestamp_str = filename[16:]  # Remove 'quicknav_backup_'
                    try:
                        timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                    except:
                        timestamp = datetime.fromtimestamp(stat.st_mtime)
                else:
                    timestamp = datetime.fromtimestamp(stat.st_mtime)

                backups.append({
                    "path": str(file_path),
                    "filename": file_path.name,
                    "timestamp": timestamp,
                    "size_bytes": stat.st_size,
                    "compressed": file_path.suffix == '.gz'
                })

            # Sort by timestamp (newest first)
            backups.sort(key=lambda x: x["timestamp"], reverse=True)

        except Exception as e:
            logger.error(f"Failed to list backups: {e}")

        return backups

    # Export/Import
    def export_data(self, export_path: str, format: str = "json",
                   include_activities: bool = True, include_notes: bool = True,
                   date_range: tuple = None) -> bool:
        """Export data in various formats."""
        try:
            export_path = Path(export_path)

            # Prepare data
            export_data = {
                "export_info": {
                    "timestamp": datetime.now().isoformat(),
                    "format": format,
                    "version": "1.0.0"
                },
                "projects": [],
                "activity_logs": [],
                "notes": []
            }

            with self._db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                # Export project metadata
                cursor.execute("SELECT * FROM project_metadata ORDER BY last_accessed DESC")
                for row in cursor.fetchall():
                    project_data = {
                        "project_number": row[0],
                        "project_name": row[1],
                        "project_path": row[2],
                        "created_date": row[3],
                        "last_accessed": row[4],
                        "access_count": row[5],
                        "favorite": bool(row[6]),
                        "tags": json.loads(row[7]) if row[7] else [],
                        "custom_fields": json.loads(row[8]) if row[8] else {},
                        "notes": row[9] or ""
                    }
                    export_data["projects"].append(project_data)

                # Export activity logs
                if include_activities:
                    activity_query = "SELECT * FROM activity_log"
                    activity_params = []

                    if date_range:
                        start_date, end_date = date_range
                        activity_query += " WHERE timestamp BETWEEN ? AND ?"
                        activity_params = [start_date.isoformat(), end_date.isoformat()]

                    activity_query += " ORDER BY timestamp DESC"

                    cursor.execute(activity_query, activity_params)
                    for row in cursor.fetchall():
                        activity_data = {
                            "id": row[0],
                            "timestamp": row[1],
                            "project_number": row[2],
                            "project_name": row[3],
                            "action": row[4],
                            "details": json.loads(row[5]) if row[5] else {},
                            "duration_ms": row[6]
                        }
                        export_data["activity_logs"].append(activity_data)

                # Export notes
                if include_notes:
                    cursor.execute("SELECT * FROM project_notes ORDER BY created_date DESC")
                    for row in cursor.fetchall():
                        note_data = {
                            "id": row[0],
                            "project_number": row[1],
                            "title": row[2],
                            "content": row[3],
                            "created_date": row[4],
                            "modified_date": row[5],
                            "tags": json.loads(row[6]) if row[6] else [],
                            "priority": row[7]
                        }
                        export_data["notes"].append(note_data)

                conn.close()

            # Save in requested format
            if format.lower() == "json":
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)

            elif format.lower() == "csv":
                import csv

                # Export projects to CSV
                projects_path = export_path.parent / f"{export_path.stem}_projects.csv"
                with open(projects_path, 'w', newline='', encoding='utf-8') as f:
                    if export_data["projects"]:
                        writer = csv.DictWriter(f, fieldnames=export_data["projects"][0].keys())
                        writer.writeheader()
                        for project in export_data["projects"]:
                            # Convert complex fields to strings
                            project_copy = project.copy()
                            project_copy["tags"] = ",".join(project["tags"])
                            project_copy["custom_fields"] = json.dumps(project["custom_fields"])
                            writer.writerow(project_copy)

                # Export activities to CSV if included
                if include_activities and export_data["activity_logs"]:
                    activities_path = export_path.parent / f"{export_path.stem}_activities.csv"
                    with open(activities_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=export_data["activity_logs"][0].keys())
                        writer.writeheader()
                        for activity in export_data["activity_logs"]:
                            activity_copy = activity.copy()
                            activity_copy["details"] = json.dumps(activity["details"])
                            writer.writerow(activity_copy)

                # Export notes to CSV if included
                if include_notes and export_data["notes"]:
                    notes_path = export_path.parent / f"{export_path.stem}_notes.csv"
                    with open(notes_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=export_data["notes"][0].keys())
                        writer.writeheader()
                        for note in export_data["notes"]:
                            note_copy = note.copy()
                            note_copy["tags"] = ",".join(note["tags"])
                            writer.writerow(note_copy)

            elif format.lower() == "xlsx":
                try:
                    import pandas as pd

                    # Create Excel writer
                    with pd.ExcelWriter(export_path, engine='openpyxl') as writer:
                        # Projects sheet
                        if export_data["projects"]:
                            projects_df = pd.DataFrame(export_data["projects"])
                            projects_df.to_excel(writer, sheet_name='Projects', index=False)

                        # Activities sheet
                        if include_activities and export_data["activity_logs"]:
                            activities_df = pd.DataFrame(export_data["activity_logs"])
                            activities_df.to_excel(writer, sheet_name='Activities', index=False)

                        # Notes sheet
                        if include_notes and export_data["notes"]:
                            notes_df = pd.DataFrame(export_data["notes"])
                            notes_df.to_excel(writer, sheet_name='Notes', index=False)

                except ImportError:
                    raise Exception("pandas and openpyxl required for Excel export")

            else:
                raise ValueError(f"Unsupported export format: {format}")

            logger.info(f"Exported data to: {export_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            raise

    def import_data(self, import_path: str, merge: bool = True) -> bool:
        """Import data from exported file."""
        try:
            import_path = Path(import_path)

            if not import_path.exists():
                raise FileNotFoundError(f"Import file not found: {import_path}")

            # Determine format and load data
            if import_path.suffix.lower() == '.json':
                with open(import_path, 'r', encoding='utf-8') as f:
                    import_data = json.load(f)

            elif import_path.suffix.lower() == '.csv':
                raise NotImplementedError("CSV import not yet implemented")

            elif import_path.suffix.lower() == '.xlsx':
                raise NotImplementedError("Excel import not yet implemented")

            else:
                raise ValueError(f"Unsupported import format: {import_path.suffix}")

            # Create backup before import
            if not merge:
                self.create_backup("Pre-import backup")

            with self._db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                # Import projects
                for project in import_data.get("projects", []):
                    if merge:
                        # Check if project exists
                        cursor.execute(
                            "SELECT project_number FROM project_metadata WHERE project_number = ?",
                            (project["project_number"],)
                        )
                        if cursor.fetchone():
                            continue  # Skip existing

                    cursor.execute("""
                        INSERT OR REPLACE INTO project_metadata
                        (project_number, project_name, project_path, created_date,
                         last_accessed, access_count, favorite, tags, custom_fields, notes)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        project["project_number"],
                        project["project_name"],
                        project["project_path"],
                        project["created_date"],
                        project["last_accessed"],
                        project["access_count"],
                        int(project["favorite"]),
                        json.dumps(project["tags"]),
                        json.dumps(project["custom_fields"]),
                        project["notes"]
                    ))

                # Import activities
                for activity in import_data.get("activity_logs", []):
                    if merge:
                        cursor.execute("SELECT id FROM activity_log WHERE id = ?", (activity["id"],))
                        if cursor.fetchone():
                            continue  # Skip existing

                    cursor.execute("""
                        INSERT OR REPLACE INTO activity_log
                        (id, timestamp, project_number, project_name, action, details, duration_ms)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        activity["id"],
                        activity["timestamp"],
                        activity["project_number"],
                        activity["project_name"],
                        activity["action"],
                        json.dumps(activity["details"]),
                        activity["duration_ms"]
                    ))

                # Import notes
                for note in import_data.get("notes", []):
                    if merge:
                        cursor.execute("SELECT id FROM project_notes WHERE id = ?", (note["id"],))
                        if cursor.fetchone():
                            continue  # Skip existing

                    cursor.execute("""
                        INSERT OR REPLACE INTO project_notes
                        (id, project_number, title, content, created_date, modified_date, tags, priority)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        note["id"],
                        note["project_number"],
                        note["title"],
                        note["content"],
                        note["created_date"],
                        note["modified_date"],
                        json.dumps(note["tags"]),
                        note["priority"]
                    ))

                conn.commit()
                conn.close()

            # Clear caches
            self._metadata_cache.clear()
            self._activity_cache.clear()

            logger.info(f"Imported data from: {import_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to import data: {e}")
            raise

    # Data Integrity and Validation
    def validate_data_integrity(self) -> Dict[str, Any]:
        """Validate data integrity and report issues."""
        issues = []
        stats = {}

        try:
            with self._db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                # Check for orphaned activities
                cursor.execute("""
                    SELECT COUNT(*) FROM activity_log a
                    LEFT JOIN project_metadata p ON a.project_number = p.project_number
                    WHERE p.project_number IS NULL AND a.project_number IS NOT NULL
                """)
                orphaned_activities = cursor.fetchone()[0]
                if orphaned_activities > 0:
                    issues.append(f"{orphaned_activities} orphaned activity log entries")

                # Check for orphaned notes
                cursor.execute("""
                    SELECT COUNT(*) FROM project_notes n
                    LEFT JOIN project_metadata p ON n.project_number = p.project_number
                    WHERE p.project_number IS NULL
                """)
                orphaned_notes = cursor.fetchone()[0]
                if orphaned_notes > 0:
                    issues.append(f"{orphaned_notes} orphaned project notes")

                # Check for invalid JSON in metadata
                cursor.execute("SELECT project_number, tags, custom_fields FROM project_metadata")
                invalid_json_count = 0
                for row in cursor.fetchall():
                    try:
                        if row[1]:
                            json.loads(row[1])
                        if row[2]:
                            json.loads(row[2])
                    except json.JSONDecodeError:
                        invalid_json_count += 1

                if invalid_json_count > 0:
                    issues.append(f"{invalid_json_count} projects with invalid JSON data")

                # Get database statistics
                cursor.execute("SELECT COUNT(*) FROM project_metadata")
                stats['total_projects'] = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM activity_log")
                stats['total_activities'] = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM project_notes")
                stats['total_notes'] = cursor.fetchone()[0]

                # Check database file size
                stats['database_size_bytes'] = self.db_path.stat().st_size if self.db_path.exists() else 0

                conn.close()

        except Exception as e:
            issues.append(f"Database validation error: {e}")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }

    def repair_data_integrity(self) -> Dict[str, Any]:
        """Attempt to repair data integrity issues."""
        repairs_made = []

        try:
            # Create backup before repairs
            backup_path = self.create_backup("Pre-repair backup")

            with self._db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                # Remove orphaned activities
                cursor.execute("""
                    DELETE FROM activity_log
                    WHERE id IN (
                        SELECT a.id FROM activity_log a
                        LEFT JOIN project_metadata p ON a.project_number = p.project_number
                        WHERE p.project_number IS NULL AND a.project_number IS NOT NULL
                    )
                """)
                if cursor.rowcount > 0:
                    repairs_made.append(f"Removed {cursor.rowcount} orphaned activity log entries")

                # Remove orphaned notes
                cursor.execute("""
                    DELETE FROM project_notes
                    WHERE id IN (
                        SELECT n.id FROM project_notes n
                        LEFT JOIN project_metadata p ON n.project_number = p.project_number
                        WHERE p.project_number IS NULL
                    )
                """)
                if cursor.rowcount > 0:
                    repairs_made.append(f"Removed {cursor.rowcount} orphaned project notes")

                # Fix invalid JSON data
                cursor.execute("SELECT project_number, tags, custom_fields FROM project_metadata")
                json_fixes = 0
                for row in cursor.fetchall():
                    project_number = row[0]
                    fixed_tags = "[]"
                    fixed_custom_fields = "{}"

                    try:
                        if row[1]:
                            json.loads(row[1])
                            fixed_tags = row[1]
                    except json.JSONDecodeError:
                        json_fixes += 1

                    try:
                        if row[2]:
                            json.loads(row[2])
                            fixed_custom_fields = row[2]
                    except json.JSONDecodeError:
                        json_fixes += 1

                    if json_fixes > 0:
                        cursor.execute("""
                            UPDATE project_metadata
                            SET tags = ?, custom_fields = ?
                            WHERE project_number = ?
                        """, (fixed_tags, fixed_custom_fields, project_number))

                if json_fixes > 0:
                    repairs_made.append(f"Fixed {json_fixes} invalid JSON fields")

                conn.commit()
                conn.close()

        except Exception as e:
            repairs_made.append(f"Repair error: {e}")

        return {
            "success": len(repairs_made) > 0,
            "repairs_made": repairs_made,
            "backup_created": backup_path if 'backup_path' in locals() else None,
            "timestamp": datetime.now().isoformat()
        }

    # Cleanup and Maintenance
    def cleanup_old_data(self, retention_days: int = None) -> Dict[str, Any]:
        """Clean up old data based on retention policy."""
        if retention_days is None:
            retention_days = self.config.get("activity_log_retention_days", 365)

        cutoff_date = datetime.now() - timedelta(days=retention_days)
        cleanup_stats = {}

        try:
            with self._db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                # Clean up old activity logs
                cursor.execute("""
                    DELETE FROM activity_log
                    WHERE timestamp < ?
                """, (cutoff_date.isoformat(),))
                cleanup_stats['activities_removed'] = cursor.rowcount

                # Vacuum database to reclaim space
                cursor.execute("VACUUM")

                conn.commit()
                conn.close()

            # Clear cache
            self._activity_cache.clear()

            # Cleanup old backups
            self._cleanup_old_backups()

            cleanup_stats['success'] = True
            cleanup_stats['retention_days'] = retention_days
            cleanup_stats['cutoff_date'] = cutoff_date.isoformat()

        except Exception as e:
            cleanup_stats['success'] = False
            cleanup_stats['error'] = str(e)

        return cleanup_stats

    def get_data_statistics(self) -> Dict[str, Any]:
        """Get comprehensive data statistics."""
        stats = {}

        try:
            with self._db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                # Project statistics
                cursor.execute("SELECT COUNT(*) FROM project_metadata")
                stats['total_projects'] = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM project_metadata WHERE favorite = 1")
                stats['favorite_projects'] = cursor.fetchone()[0]

                cursor.execute("SELECT AVG(access_count) FROM project_metadata")
                stats['avg_access_count'] = cursor.fetchone()[0] or 0

                # Activity statistics
                cursor.execute("SELECT COUNT(*) FROM activity_log")
                stats['total_activities'] = cursor.fetchone()[0]

                cursor.execute("""
                    SELECT action, COUNT(*) FROM activity_log
                    GROUP BY action ORDER BY COUNT(*) DESC
                """)
                stats['activities_by_type'] = dict(cursor.fetchall())

                # Notes statistics
                cursor.execute("SELECT COUNT(*) FROM project_notes")
                stats['total_notes'] = cursor.fetchone()[0]

                cursor.execute("""
                    SELECT priority, COUNT(*) FROM project_notes
                    GROUP BY priority ORDER BY COUNT(*) DESC
                """)
                stats['notes_by_priority'] = dict(cursor.fetchall())

                # Database size
                stats['database_size_bytes'] = self.db_path.stat().st_size if self.db_path.exists() else 0
                stats['database_size_mb'] = round(stats['database_size_bytes'] / (1024 * 1024), 2)

                # Data directory size
                total_size = sum(f.stat().st_size for f in self.data_dir.rglob('*') if f.is_file())
                stats['total_data_size_bytes'] = total_size
                stats['total_data_size_mb'] = round(total_size / (1024 * 1024), 2)

                # Backup statistics
                backups = self.list_backups()
                stats['total_backups'] = len(backups)
                if backups:
                    stats['latest_backup'] = backups[0]['timestamp'].isoformat()
                    stats['total_backup_size_mb'] = round(
                        sum(b['size_bytes'] for b in backups) / (1024 * 1024), 2
                    )

                conn.close()

        except Exception as e:
            stats['error'] = str(e)

        return stats

    # Synchronization (placeholder for future implementation)
    def setup_sync(self, provider: str, config: Dict[str, Any]) -> bool:
        """Setup data synchronization with cloud provider."""
        # Placeholder for future cloud sync implementation
        supported_providers = ["onedrive", "dropbox", "gdrive"]

        if provider not in supported_providers:
            raise ValueError(f"Unsupported sync provider: {provider}")

        self.config["sync_enabled"] = True
        self.config["sync_provider"] = provider
        self.config["sync_config"] = config
        self._save_config()

        logger.info(f"Sync configured for provider: {provider}")
        return True

    def sync_data(self) -> Dict[str, Any]:
        """Synchronize data with configured cloud provider."""
        # Placeholder for future implementation
        return {
            "success": False,
            "message": "Cloud synchronization not yet implemented"
        }

    # Cleanup
    def close(self):
        """Close data manager and clean up resources."""
        # Flush any pending activities
        self._flush_activity_cache()

        # Save configuration
        self._save_config()

        # Clear caches
        self._metadata_cache.clear()
        self._activity_cache.clear()

        logger.info("DataManager closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Utility functions for easy access
def get_default_data_manager() -> DataManager:
    """Get a default data manager instance."""
    return DataManager()


def create_backup_now(description: str = "") -> str:
    """Create a backup immediately."""
    with get_default_data_manager() as dm:
        return dm.create_backup(description)


def validate_data_now() -> Dict[str, Any]:
    """Validate data integrity immediately."""
    with get_default_data_manager() as dm:
        return dm.validate_data_integrity()