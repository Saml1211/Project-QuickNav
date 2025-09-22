"""
Multi-Level Caching System for Project QuickNav

Implements L1 (memory), L2 (SQLite), and L3 (file system) caching
with TTL support, compression, and intelligent cache warming.
"""

import os
import json
import pickle
import sqlite3
import gzip
import hashlib
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from collections import OrderedDict
import logging
from dataclasses import dataclass
from pathlib import Path

from .config import get_config

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    ttl_seconds: Optional[int] = None
    access_count: int = 0
    last_accessed: datetime = None
    size_bytes: int = 0
    compressed: bool = False

    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = self.created_at

    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl_seconds is None:
            return False
        expiry_time = self.created_at + timedelta(seconds=self.ttl_seconds)
        return datetime.utcnow() > expiry_time

    def update_access(self):
        """Update access statistics"""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()


class LRUCache:
    """Thread-safe LRU cache with size limits"""

    def __init__(self, max_size_mb: int = 64, max_items: int = 10000):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_items = max_items
        self.cache = OrderedDict()
        self.current_size = 0
        self.lock = threading.RLock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size_evictions': 0
        }

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if not entry.is_expired():
                    # Move to end (most recently used)
                    self.cache.move_to_end(key)
                    entry.update_access()
                    self.stats['hits'] += 1
                    return entry.value
                else:
                    # Remove expired entry
                    self._remove_entry(key)

            self.stats['misses'] += 1
            return None

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in cache"""
        try:
            # Estimate size
            size_bytes = self._estimate_size(value)

            with self.lock:
                # Remove existing entry if present
                if key in self.cache:
                    self._remove_entry(key)

                # Check if we need to make space
                while (len(self.cache) >= self.max_items or
                       self.current_size + size_bytes > self.max_size_bytes):
                    if not self._evict_oldest():
                        # Can't evict anything, item too large
                        return False

                # Create and store entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=datetime.utcnow(),
                    ttl_seconds=ttl_seconds,
                    size_bytes=size_bytes
                )

                self.cache[key] = entry
                self.current_size += size_bytes
                return True

        except Exception as e:
            logger.error(f"Error setting cache entry {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete entry from cache"""
        with self.lock:
            if key in self.cache:
                self._remove_entry(key)
                return True
            return False

    def clear(self):
        """Clear all entries"""
        with self.lock:
            self.cache.clear()
            self.current_size = 0

    def cleanup_expired(self) -> int:
        """Remove expired entries"""
        removed_count = 0
        with self.lock:
            expired_keys = [key for key, entry in self.cache.items() if entry.is_expired()]
            for key in expired_keys:
                self._remove_entry(key)
                removed_count += 1

        return removed_count

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0

            return {
                'hit_rate': hit_rate,
                'total_entries': len(self.cache),
                'current_size_mb': self.current_size / (1024 * 1024),
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'utilization': self.current_size / self.max_size_bytes if self.max_size_bytes > 0 else 0,
                **self.stats
            }

    def _remove_entry(self, key: str):
        """Remove entry and update size"""
        if key in self.cache:
            entry = self.cache.pop(key)
            self.current_size -= entry.size_bytes

    def _evict_oldest(self) -> bool:
        """Evict least recently used entry"""
        if not self.cache:
            return False

        # Find entry with oldest last_accessed time
        oldest_key = min(self.cache.keys(),
                        key=lambda k: self.cache[k].last_accessed)

        self._remove_entry(oldest_key)
        self.stats['evictions'] += 1
        return True

    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value"""
        try:
            if isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, (int, float, bool)):
                return 32  # Rough estimate
            elif isinstance(value, (list, tuple)):
                return sum(self._estimate_size(item) for item in value) + 64
            elif isinstance(value, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v)
                          for k, v in value.items()) + 64
            else:
                # Use pickle size as estimate
                return len(pickle.dumps(value))
        except Exception:
            return 1024  # Default estimate


class SQLiteCache:
    """SQLite-based persistent cache (L2)"""

    def __init__(self, db_path: str = "data/cache.db"):
        self.db_path = db_path
        self.connection_pool = {}
        self.pool_lock = threading.Lock()
        self._setup_database()

    def _setup_database(self):
        """Setup cache database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        conn = self._get_connection()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    value BLOB NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    ttl_seconds INTEGER,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP,
                    size_bytes INTEGER DEFAULT 0,
                    compressed BOOLEAN DEFAULT FALSE
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_created ON cache_entries(created_at)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_accessed ON cache_entries(last_accessed)
            """)

            conn.commit()

        except Exception as e:
            logger.error(f"Error setting up cache database: {e}")
        finally:
            self._return_connection(conn)

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection from pool"""
        thread_id = threading.get_ident()

        with self.pool_lock:
            if thread_id not in self.connection_pool:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                self.connection_pool[thread_id] = conn
            return self.connection_pool[thread_id]

    def _return_connection(self, conn: sqlite3.Connection):
        """Return connection to pool (no-op for thread-local connections)"""
        pass

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        conn = self._get_connection()
        try:
            cursor = conn.execute("""
                SELECT value, created_at, ttl_seconds, compressed, access_count
                FROM cache_entries
                WHERE key = ?
            """, [key])

            row = cursor.fetchone()
            if not row:
                return None

            # Check expiration
            created_at = datetime.fromisoformat(row['created_at'])
            ttl_seconds = row['ttl_seconds']

            if ttl_seconds and (datetime.utcnow() - created_at).total_seconds() > ttl_seconds:
                # Entry expired, remove it
                self.delete(key)
                return None

            # Update access statistics
            conn.execute("""
                UPDATE cache_entries
                SET access_count = access_count + 1, last_accessed = ?
                WHERE key = ?
            """, [datetime.utcnow().isoformat(), key])
            conn.commit()

            # Deserialize value
            value_data = row['value']
            compressed = row['compressed']

            if compressed:
                value_data = gzip.decompress(value_data)

            return pickle.loads(value_data)

        except Exception as e:
            logger.error(f"Error getting cache entry {key}: {e}")
            return None
        finally:
            self._return_connection(conn)

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None,
           compress: bool = True) -> bool:
        """Set value in cache"""
        try:
            # Serialize value
            value_data = pickle.dumps(value)
            original_size = len(value_data)

            # Compress if enabled and beneficial
            compressed = False
            if compress and original_size > 1024:  # Only compress larger values
                compressed_data = gzip.compress(value_data)
                if len(compressed_data) < original_size * 0.8:  # 20% compression threshold
                    value_data = compressed_data
                    compressed = True

            size_bytes = len(value_data)

            conn = self._get_connection()
            try:
                now = datetime.utcnow().isoformat()
                conn.execute("""
                    INSERT OR REPLACE INTO cache_entries
                    (key, value, created_at, ttl_seconds, access_count, last_accessed, size_bytes, compressed)
                    VALUES (?, ?, ?, ?, 0, ?, ?, ?)
                """, [key, value_data, now, ttl_seconds, now, size_bytes, compressed])

                conn.commit()
                return True

            except Exception as e:
                logger.error(f"Error setting cache entry {key}: {e}")
                return False
            finally:
                self._return_connection(conn)

        except Exception as e:
            logger.error(f"Error serializing cache entry {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete entry from cache"""
        conn = self._get_connection()
        try:
            cursor = conn.execute("DELETE FROM cache_entries WHERE key = ?", [key])
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error deleting cache entry {key}: {e}")
            return False
        finally:
            self._return_connection(conn)

    def clear(self):
        """Clear all entries"""
        conn = self._get_connection()
        try:
            conn.execute("DELETE FROM cache_entries")
            conn.commit()
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
        finally:
            self._return_connection(conn)

    def cleanup_expired(self) -> int:
        """Remove expired entries"""
        conn = self._get_connection()
        try:
            now = datetime.utcnow()
            cursor = conn.execute("""
                DELETE FROM cache_entries
                WHERE ttl_seconds IS NOT NULL
                AND datetime(created_at, '+' || ttl_seconds || ' seconds') < ?
            """, [now.isoformat()])
            conn.commit()
            return cursor.rowcount
        except Exception as e:
            logger.error(f"Error cleaning up expired cache entries: {e}")
            return 0
        finally:
            self._return_connection(conn)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        conn = self._get_connection()
        try:
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total_entries,
                    SUM(size_bytes) as total_size_bytes,
                    AVG(access_count) as avg_access_count,
                    COUNT(CASE WHEN compressed THEN 1 END) as compressed_entries
                FROM cache_entries
            """)

            row = cursor.fetchone()
            if row:
                return {
                    'total_entries': row['total_entries'],
                    'total_size_mb': (row['total_size_bytes'] or 0) / (1024 * 1024),
                    'avg_access_count': row['avg_access_count'] or 0,
                    'compressed_entries': row['compressed_entries'],
                    'compression_ratio': (row['compressed_entries'] / row['total_entries']
                                        if row['total_entries'] > 0 else 0)
                }
            return {}

        except Exception as e:
            logger.error(f"Error getting cache statistics: {e}")
            return {}
        finally:
            self._return_connection(conn)


class FileSystemCache:
    """File system-based cache (L3) for large objects"""

    def __init__(self, cache_dir: str = "data/fs_cache", max_size_mb: int = 1024):
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Metadata tracking
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()
        self.metadata_lock = threading.Lock()

    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load cache metadata"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading cache metadata: {e}")
        return {}

    def _save_metadata(self):
        """Save cache metadata"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving cache metadata: {e}")

    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key"""
        # Use hash to avoid filesystem issues with key characters
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            with self.metadata_lock:
                if key not in self.metadata:
                    return None

                entry_meta = self.metadata[key]

                # Check expiration
                created_at = datetime.fromisoformat(entry_meta['created_at'])
                ttl_seconds = entry_meta.get('ttl_seconds')

                if ttl_seconds and (datetime.utcnow() - created_at).total_seconds() > ttl_seconds:
                    self.delete(key)
                    return None

                # Update access statistics
                entry_meta['access_count'] = entry_meta.get('access_count', 0) + 1
                entry_meta['last_accessed'] = datetime.utcnow().isoformat()

            # Load value from file
            file_path = self._get_file_path(key)
            if not file_path.exists():
                with self.metadata_lock:
                    self.metadata.pop(key, None)
                return None

            with open(file_path, 'rb') as f:
                data = f.read()

                # Decompress if needed
                if entry_meta.get('compressed', False):
                    data = gzip.decompress(data)

                return pickle.loads(data)

        except Exception as e:
            logger.error(f"Error getting cache entry {key}: {e}")
            return None

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None,
           compress: bool = True) -> bool:
        """Set value in cache"""
        try:
            # Serialize value
            data = pickle.dumps(value)
            original_size = len(data)

            # Compress if beneficial
            compressed = False
            if compress and original_size > 4096:  # Only compress larger values
                compressed_data = gzip.compress(data)
                if len(compressed_data) < original_size * 0.8:
                    data = compressed_data
                    compressed = True

            # Check size limits
            if len(data) > self.max_size_bytes:
                logger.warning(f"Cache entry {key} too large: {len(data)} bytes")
                return False

            # Make space if needed
            self._make_space(len(data))

            # Write to file
            file_path = self._get_file_path(key)
            with open(file_path, 'wb') as f:
                f.write(data)

            # Update metadata
            with self.metadata_lock:
                now = datetime.utcnow().isoformat()
                self.metadata[key] = {
                    'created_at': now,
                    'last_accessed': now,
                    'ttl_seconds': ttl_seconds,
                    'size_bytes': len(data),
                    'compressed': compressed,
                    'access_count': 0
                }
                self._save_metadata()

            return True

        except Exception as e:
            logger.error(f"Error setting cache entry {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete entry from cache"""
        try:
            with self.metadata_lock:
                if key not in self.metadata:
                    return False

                # Remove file
                file_path = self._get_file_path(key)
                if file_path.exists():
                    file_path.unlink()

                # Remove metadata
                del self.metadata[key]
                self._save_metadata()

            return True

        except Exception as e:
            logger.error(f"Error deleting cache entry {key}: {e}")
            return False

    def clear(self):
        """Clear all entries"""
        try:
            with self.metadata_lock:
                # Remove all cache files
                for file_path in self.cache_dir.glob("*.cache"):
                    try:
                        file_path.unlink()
                    except Exception:
                        pass

                # Clear metadata
                self.metadata.clear()
                self._save_metadata()

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

    def cleanup_expired(self) -> int:
        """Remove expired entries"""
        removed_count = 0
        now = datetime.utcnow()

        with self.metadata_lock:
            expired_keys = []
            for key, entry_meta in self.metadata.items():
                created_at = datetime.fromisoformat(entry_meta['created_at'])
                ttl_seconds = entry_meta.get('ttl_seconds')

                if ttl_seconds and (now - created_at).total_seconds() > ttl_seconds:
                    expired_keys.append(key)

            for key in expired_keys:
                if self.delete(key):
                    removed_count += 1

        return removed_count

    def _make_space(self, required_bytes: int):
        """Make space for new entry by removing least recently used entries"""
        with self.metadata_lock:
            current_size = sum(entry['size_bytes'] for entry in self.metadata.values())

            if current_size + required_bytes <= self.max_size_bytes:
                return

            # Sort by last accessed time (oldest first)
            sorted_entries = sorted(
                self.metadata.items(),
                key=lambda x: x[1].get('last_accessed', '1970-01-01T00:00:00')
            )

            for key, entry_meta in sorted_entries:
                if current_size + required_bytes <= self.max_size_bytes:
                    break

                # Remove entry
                file_path = self._get_file_path(key)
                if file_path.exists():
                    file_path.unlink()

                current_size -= entry_meta['size_bytes']
                del self.metadata[key]

            self._save_metadata()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.metadata_lock:
            if not self.metadata:
                return {
                    'total_entries': 0,
                    'total_size_mb': 0,
                    'avg_access_count': 0,
                    'compressed_entries': 0
                }

            total_size = sum(entry['size_bytes'] for entry in self.metadata.values())
            compressed_count = sum(1 for entry in self.metadata.values()
                                 if entry.get('compressed', False))
            avg_access = sum(entry.get('access_count', 0) for entry in self.metadata.values()) / len(self.metadata)

            return {
                'total_entries': len(self.metadata),
                'total_size_mb': total_size / (1024 * 1024),
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'utilization': total_size / self.max_size_bytes if self.max_size_bytes > 0 else 0,
                'avg_access_count': avg_access,
                'compressed_entries': compressed_count,
                'compression_ratio': compressed_count / len(self.metadata)
            }


class MultiLevelCache:
    """Multi-level cache with L1 (memory), L2 (SQLite), L3 (filesystem)"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config().cache

        # Initialize cache levels
        self.l1_cache = LRUCache(
            max_size_mb=self.config.l1_cache_size_mb,
            max_items=10000
        )

        self.l2_cache = SQLiteCache("data/l2_cache.db")

        self.l3_cache = FileSystemCache(
            cache_dir="data/l3_cache",
            max_size_mb=self.config.l3_cache_size_mb
        )

        self.default_ttl = self.config.default_ttl_seconds
        self.enable_compression = self.config.enable_compression

        # Statistics
        self.stats = {
            'total_requests': 0,
            'l1_hits': 0,
            'l2_hits': 0,
            'l3_hits': 0,
            'misses': 0
        }

        # Start cleanup thread
        self._start_cleanup_thread()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with fallback through levels"""
        self.stats['total_requests'] += 1

        # L1: Memory cache
        value = self.l1_cache.get(key)
        if value is not None:
            self.stats['l1_hits'] += 1
            return value

        # L2: SQLite cache
        value = self.l2_cache.get(key)
        if value is not None:
            self.stats['l2_hits'] += 1
            # Promote to L1
            self.l1_cache.set(key, value, self.default_ttl)
            return value

        # L3: File system cache
        value = self.l3_cache.get(key)
        if value is not None:
            self.stats['l3_hits'] += 1
            # Promote to L2 and L1
            self.l2_cache.set(key, value, self.default_ttl, self.enable_compression)
            self.l1_cache.set(key, value, self.default_ttl)
            return value

        self.stats['misses'] += 1
        return None

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in all cache levels"""
        if ttl_seconds is None:
            ttl_seconds = self.default_ttl

        success = True

        # Set in all levels
        success &= self.l1_cache.set(key, value, ttl_seconds)
        success &= self.l2_cache.set(key, value, ttl_seconds, self.enable_compression)

        # Only store in L3 if value is large enough to benefit
        try:
            value_size = len(pickle.dumps(value))
            if value_size > 4096:  # 4KB threshold
                success &= self.l3_cache.set(key, value, ttl_seconds, self.enable_compression)
        except Exception:
            pass

        return success

    def delete(self, key: str) -> bool:
        """Delete from all cache levels"""
        success = True
        success &= self.l1_cache.delete(key)
        success &= self.l2_cache.delete(key)
        success &= self.l3_cache.delete(key)
        return success

    def clear(self):
        """Clear all cache levels"""
        self.l1_cache.clear()
        self.l2_cache.clear()
        self.l3_cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        total_requests = self.stats['total_requests']

        if total_requests > 0:
            hit_rate = (self.stats['l1_hits'] + self.stats['l2_hits'] + self.stats['l3_hits']) / total_requests
            l1_hit_rate = self.stats['l1_hits'] / total_requests
            l2_hit_rate = self.stats['l2_hits'] / total_requests
            l3_hit_rate = self.stats['l3_hits'] / total_requests
        else:
            hit_rate = l1_hit_rate = l2_hit_rate = l3_hit_rate = 0

        return {
            'overall': {
                'total_requests': total_requests,
                'hit_rate': hit_rate,
                'miss_rate': self.stats['misses'] / total_requests if total_requests > 0 else 0,
                'l1_hit_rate': l1_hit_rate,
                'l2_hit_rate': l2_hit_rate,
                'l3_hit_rate': l3_hit_rate
            },
            'l1_memory': self.l1_cache.get_stats(),
            'l2_sqlite': self.l2_cache.get_stats(),
            'l3_filesystem': self.l3_cache.get_stats()
        }

    def _start_cleanup_thread(self):
        """Start background cleanup thread"""
        def cleanup_loop():
            while True:
                try:
                    time.sleep(3600)  # Run every hour
                    self.cleanup_expired()
                except Exception as e:
                    logger.error(f"Error in cache cleanup: {e}")

        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()

    def cleanup_expired(self) -> Dict[str, int]:
        """Clean up expired entries from all levels"""
        return {
            'l1_cleaned': self.l1_cache.cleanup_expired(),
            'l2_cleaned': self.l2_cache.cleanup_expired(),
            'l3_cleaned': self.l3_cache.cleanup_expired()
        }


class CacheManager:
    """High-level cache management with semantic caching"""

    def __init__(self):
        self.config = get_config()
        self.cache = MultiLevelCache()

        # Semantic cache keys for different data types
        self.cache_prefixes = {
            'project_features': 'pf:',
            'document_features': 'df:',
            'user_features': 'uf:',
            'search_results': 'sr:',
            'document_list': 'dl:',
            'project_metadata': 'pm:',
            'user_session': 'us:',
            'analytics': 'an:'
        }

    def cache_project_features(self, project_id: str, features: Dict[str, Any],
                             ttl_hours: int = 24) -> bool:
        """Cache project features"""
        key = f"{self.cache_prefixes['project_features']}{project_id}"
        return self.cache.set(key, features, ttl_hours * 3600)

    def get_project_features(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get cached project features"""
        key = f"{self.cache_prefixes['project_features']}{project_id}"
        return self.cache.get(key)

    def cache_search_results(self, query_hash: str, results: List[Dict[str, Any]],
                           ttl_minutes: int = 30) -> bool:
        """Cache search results"""
        key = f"{self.cache_prefixes['search_results']}{query_hash}"
        return self.cache.set(key, results, ttl_minutes * 60)

    def get_search_results(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached search results"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        key = f"{self.cache_prefixes['search_results']}{query_hash}"
        return self.cache.get(key)

    def cache_document_list(self, project_id: str, doc_type: str,
                          documents: List[Dict[str, Any]], ttl_hours: int = 6) -> bool:
        """Cache document list for project"""
        key = f"{self.cache_prefixes['document_list']}{project_id}:{doc_type}"
        return self.cache.set(key, documents, ttl_hours * 3600)

    def get_document_list(self, project_id: str, doc_type: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached document list"""
        key = f"{self.cache_prefixes['document_list']}{project_id}:{doc_type}"
        return self.cache.get(key)

    def cache_user_session(self, session_id: str, session_data: Dict[str, Any],
                         ttl_minutes: int = 60) -> bool:
        """Cache user session data"""
        key = f"{self.cache_prefixes['user_session']}{session_id}"
        return self.cache.set(key, session_data, ttl_minutes * 60)

    def get_user_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get cached user session data"""
        key = f"{self.cache_prefixes['user_session']}{session_id}"
        return self.cache.get(key)

    def cache_analytics_data(self, metric_name: str, data: Any,
                           ttl_hours: int = 12) -> bool:
        """Cache analytics data"""
        key = f"{self.cache_prefixes['analytics']}{metric_name}"
        return self.cache.set(key, data, ttl_hours * 3600)

    def get_analytics_data(self, metric_name: str) -> Optional[Any]:
        """Get cached analytics data"""
        key = f"{self.cache_prefixes['analytics']}{metric_name}"
        return self.cache.get(key)

    def invalidate_project_cache(self, project_id: str):
        """Invalidate all cache entries for a project"""
        patterns_to_clear = [
            f"{self.cache_prefixes['project_features']}{project_id}",
            f"{self.cache_prefixes['document_list']}{project_id}:",
            f"{self.cache_prefixes['project_metadata']}{project_id}"
        ]

        for pattern in patterns_to_clear:
            # For now, we'll delete exact matches
            # In a more sophisticated implementation, we could scan for pattern matches
            self.cache.delete(pattern)

    def warm_cache(self, project_ids: List[str], feature_store = None):
        """Warm cache with frequently accessed data"""
        if not feature_store:
            return

        logger.info(f"Warming cache for {len(project_ids)} projects")

        for project_id in project_ids:
            try:
                # Pre-load project features
                features = feature_store.get_features(project_id, 'project')
                if features:
                    self.cache_project_features(project_id, features)

                # Pre-load common document lists
                for doc_type in ['lld', 'hld', 'floor_plans', 'change_order']:
                    # This would need integration with document scanner
                    pass

            except Exception as e:
                logger.error(f"Error warming cache for project {project_id}: {e}")

    def get_cache_health(self) -> Dict[str, Any]:
        """Get cache health metrics"""
        stats = self.cache.get_stats()

        # Determine health status
        overall_hit_rate = stats['overall']['hit_rate']
        l1_utilization = stats['l1_memory']['utilization']

        if overall_hit_rate > 0.8 and l1_utilization < 0.9:
            health = 'excellent'
        elif overall_hit_rate > 0.6 and l1_utilization < 0.95:
            health = 'good'
        elif overall_hit_rate > 0.4:
            health = 'fair'
        else:
            health = 'poor'

        return {
            'health_status': health,
            'hit_rate': overall_hit_rate,
            'memory_utilization': l1_utilization,
            'recommendations': self._get_cache_recommendations(stats)
        }

    def _get_cache_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Get cache optimization recommendations"""
        recommendations = []

        hit_rate = stats['overall']['hit_rate']
        l1_util = stats['l1_memory']['utilization']

        if hit_rate < 0.5:
            recommendations.append("Consider increasing cache TTL values")

        if l1_util > 0.9:
            recommendations.append("Consider increasing L1 cache size")

        if stats['overall']['l2_hit_rate'] > stats['overall']['l1_hit_rate']:
            recommendations.append("Consider optimizing L1 cache policies")

        return recommendations


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager