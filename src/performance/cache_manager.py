"""
Advanced Caching Manager for Project QuickNav

Provides intelligent multi-level caching with:
- Redis integration for distributed caching
- Intelligent cache warming and prefetching
- Cache invalidation strategies
- Performance-aware cache policies
- Memory-efficient data structures
- Real-time cache analytics
"""

import json
import hashlib
import time
import threading
import logging
import pickle
import gzip
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, TypeVar, Generic
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from collections import OrderedDict
import redis
import sqlite3
import os

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CacheStrategy(Enum):
    """Cache strategy options."""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    ADAPTIVE = "adaptive"


class CacheLevel(Enum):
    """Cache level hierarchy."""
    L1_MEMORY = "l1_memory"
    L2_REDIS = "l2_redis"
    L3_DISK = "l3_disk"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl_seconds: Optional[int]
    size_bytes: int
    compressed: bool
    cache_level: CacheLevel
    tags: List[str]
    metadata: Dict[str, Any]

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl_seconds is None:
            return False
        return (datetime.utcnow() - self.created_at).total_seconds() > self.ttl_seconds

    def update_access(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int
    misses: int
    evictions: int
    size_bytes: int
    entry_count: int
    hit_rate: float
    avg_access_time_ms: float
    memory_pressure: float


class IntelligentCacheEviction:
    """Intelligent cache eviction using multiple strategies."""

    def __init__(self, strategy: CacheStrategy = CacheStrategy.ADAPTIVE):
        self.strategy = strategy
        self.access_patterns = {}
        self.frequency_counter = {}

    def should_evict(self, entry: CacheEntry, memory_pressure: float) -> float:
        """Calculate eviction score (higher = more likely to evict)."""
        now = datetime.utcnow()
        age_hours = (now - entry.last_accessed).total_seconds() / 3600

        if self.strategy == CacheStrategy.LRU:
            return age_hours

        elif self.strategy == CacheStrategy.LFU:
            return 1.0 / (entry.access_count + 1)

        elif self.strategy == CacheStrategy.TTL:
            if entry.ttl_seconds:
                remaining_ratio = max(0, (entry.ttl_seconds - (now - entry.created_at).total_seconds()) / entry.ttl_seconds)
                return 1.0 - remaining_ratio
            return age_hours

        else:  # ADAPTIVE
            # Combine multiple factors
            age_score = min(age_hours / 24, 1.0)  # Normalize to 0-1
            frequency_score = 1.0 / (entry.access_count + 1)
            size_score = min(entry.size_bytes / (1024 * 1024), 1.0)  # Normalize by MB
            memory_score = memory_pressure

            # Weighted combination
            return (age_score * 0.3 + frequency_score * 0.3 + size_score * 0.2 + memory_score * 0.2)


class CacheWarmupManager:
    """Manages cache warming and prefetching."""

    def __init__(self, cache_manager):
        self.cache_manager = cache_manager
        self.warmup_patterns = {}
        self.prefetch_queue = asyncio.Queue()
        self.warmup_active = False

    async def start_warmup_worker(self):
        """Start background warmup worker."""
        self.warmup_active = True
        while self.warmup_active:
            try:
                warmup_task = await asyncio.wait_for(
                    self.prefetch_queue.get(),
                    timeout=5.0
                )
                await self._execute_warmup_task(warmup_task)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Cache warmup error: {e}")

    def stop_warmup_worker(self):
        """Stop warmup worker."""
        self.warmup_active = False

    async def schedule_warmup(self, pattern: str, data_source: Callable, priority: int = 1):
        """Schedule cache warmup for a pattern."""
        warmup_task = {
            'pattern': pattern,
            'data_source': data_source,
            'priority': priority,
            'scheduled_at': datetime.utcnow()
        }
        await self.prefetch_queue.put(warmup_task)

    async def _execute_warmup_task(self, task: Dict[str, Any]):
        """Execute a warmup task."""
        try:
            pattern = task['pattern']
            data_source = task['data_source']

            # Get data from source
            if asyncio.iscoroutinefunction(data_source):
                data = await data_source()
            else:
                data = data_source()

            # Store in cache with long TTL for warmed data
            await self.cache_manager.set(
                pattern,
                data,
                ttl_seconds=3600,  # 1 hour TTL for warmed data
                tags=['warmed'],
                force_level=CacheLevel.L2_REDIS
            )

            logger.debug(f"Cache warmed: {pattern}")

        except Exception as e:
            logger.error(f"Failed to warm cache for {task['pattern']}: {e}")


class RedisManager:
    """Redis connection and operation manager."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client = None
        self.connection_pool = None
        self._connect()

    def _connect(self):
        """Connect to Redis."""
        try:
            self.connection_pool = redis.ConnectionPool.from_url(self.redis_url)
            self.redis_client = redis.Redis(connection_pool=self.connection_pool)

            # Test connection
            self.redis_client.ping()
            logger.info("Connected to Redis")

        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            self.redis_client = None

    def is_available(self) -> bool:
        """Check if Redis is available."""
        if not self.redis_client:
            return False
        try:
            self.redis_client.ping()
            return True
        except:
            return False

    async def get(self, key: str) -> Optional[bytes]:
        """Get value from Redis."""
        if not self.is_available():
            return None
        try:
            return self.redis_client.get(key)
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None

    async def set(self, key: str, value: bytes, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in Redis."""
        if not self.is_available():
            return False
        try:
            if ttl_seconds:
                return self.redis_client.setex(key, ttl_seconds, value)
            else:
                return self.redis_client.set(key, value)
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from Redis."""
        if not self.is_available():
            return False
        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False

    async def delete_pattern(self, pattern: str) -> int:
        """Delete keys matching pattern."""
        if not self.is_available():
            return 0
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Redis delete pattern error: {e}")
            return 0


class CacheManager:
    """Advanced multi-level cache manager."""

    def __init__(self,
                 redis_url: str = "redis://localhost:6379",
                 l1_max_size_mb: int = 128,
                 l3_cache_dir: str = "data/cache",
                 strategy: CacheStrategy = CacheStrategy.ADAPTIVE):

        self.strategy = strategy
        self.redis_manager = RedisManager(redis_url)
        self.eviction_manager = IntelligentCacheEviction(strategy)
        self.warmup_manager = CacheWarmupManager(self)

        # L1 Cache (Memory)
        self.l1_cache = OrderedDict()
        self.l1_max_size_bytes = l1_max_size_mb * 1024 * 1024
        self.l1_current_size = 0
        self.l1_lock = threading.RLock()

        # L3 Cache (Disk)
        self.l3_cache_dir = l3_cache_dir
        os.makedirs(l3_cache_dir, exist_ok=True)

        # Statistics
        self.stats = {
            'l1_hits': 0,
            'l1_misses': 0,
            'l2_hits': 0,
            'l2_misses': 0,
            'l3_hits': 0,
            'l3_misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
        self.stats_lock = threading.Lock()

        # Performance tracking
        self.access_times = []
        self.access_times_lock = threading.Lock()

        # Cache invalidation patterns
        self.invalidation_patterns = {}

        # Start background tasks
        asyncio.create_task(self.warmup_manager.start_warmup_worker())

    async def get(self, key: str, tags: Optional[List[str]] = None) -> Optional[Any]:
        """Get value from cache with fallback through levels."""
        start_time = time.perf_counter()

        with self.stats_lock:
            self.stats['total_requests'] += 1

        try:
            # L1: Memory cache
            value = await self._get_l1(key)
            if value is not None:
                with self.stats_lock:
                    self.stats['l1_hits'] += 1
                return value

            with self.stats_lock:
                self.stats['l1_misses'] += 1

            # L2: Redis cache
            value = await self._get_l2(key)
            if value is not None:
                with self.stats_lock:
                    self.stats['l2_hits'] += 1
                # Promote to L1
                await self._set_l1(key, value)
                return value

            with self.stats_lock:
                self.stats['l2_misses'] += 1

            # L3: Disk cache
            value = await self._get_l3(key)
            if value is not None:
                with self.stats_lock:
                    self.stats['l3_hits'] += 1
                # Promote to L2 and L1
                await self._set_l2(key, value)
                await self._set_l1(key, value)
                return value

            with self.stats_lock:
                self.stats['l3_misses'] += 1

            return None

        finally:
            # Record access time
            access_time_ms = (time.perf_counter() - start_time) * 1000
            with self.access_times_lock:
                self.access_times.append(access_time_ms)
                # Keep only recent access times
                if len(self.access_times) > 1000:
                    self.access_times = self.access_times[-500:]

    async def set(self,
                  key: str,
                  value: Any,
                  ttl_seconds: Optional[int] = None,
                  tags: Optional[List[str]] = None,
                  force_level: Optional[CacheLevel] = None,
                  compress: bool = True) -> bool:
        """Set value in cache with intelligent placement."""

        tags = tags or []

        # Serialize and optionally compress
        serialized_value = await self._serialize_value(value, compress)
        size_bytes = len(serialized_value) if isinstance(serialized_value, bytes) else len(str(serialized_value).encode())

        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
            access_count=1,
            ttl_seconds=ttl_seconds,
            size_bytes=size_bytes,
            compressed=compress and size_bytes > 1024,
            cache_level=force_level or self._determine_cache_level(size_bytes, ttl_seconds),
            tags=tags,
            metadata={}
        )

        success = True

        # Store in appropriate levels
        if force_level == CacheLevel.L1_MEMORY or entry.cache_level == CacheLevel.L1_MEMORY:
            success &= await self._set_l1(key, entry)

        if force_level == CacheLevel.L2_REDIS or entry.cache_level == CacheLevel.L2_REDIS or not force_level:
            success &= await self._set_l2(key, entry, ttl_seconds)

        if force_level == CacheLevel.L3_DISK or entry.cache_level == CacheLevel.L3_DISK or size_bytes > 1024 * 1024:
            success &= await self._set_l3(key, entry, ttl_seconds)

        return success

    async def delete(self, key: str) -> bool:
        """Delete from all cache levels."""
        success = True
        success &= await self._delete_l1(key)
        success &= await self._delete_l2(key)
        success &= await self._delete_l3(key)
        return success

    async def delete_by_tags(self, tags: List[str]) -> int:
        """Delete entries by tags."""
        deleted_count = 0

        # For L1, we need to scan all entries
        with self.l1_lock:
            keys_to_delete = []
            for cache_key, entry in self.l1_cache.items():
                if isinstance(entry, CacheEntry) and any(tag in entry.tags for tag in tags):
                    keys_to_delete.append(cache_key)

            for cache_key in keys_to_delete:
                if await self._delete_l1(cache_key):
                    deleted_count += 1

        # For Redis, use pattern matching if possible
        for tag in tags:
            pattern = f"tag:{tag}:*"
            deleted_count += await self.redis_manager.delete_pattern(pattern)

        return deleted_count

    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern."""
        deleted_count = 0

        # L1 Cache
        with self.l1_lock:
            keys_to_delete = [key for key in self.l1_cache.keys() if self._matches_pattern(key, pattern)]
            for key in keys_to_delete:
                if await self._delete_l1(key):
                    deleted_count += 1

        # L2 Cache (Redis)
        deleted_count += await self.redis_manager.delete_pattern(pattern)

        return deleted_count

    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """Check if key matches pattern."""
        # Simple wildcard matching
        import fnmatch
        return fnmatch.fnmatch(key, pattern)

    def _determine_cache_level(self, size_bytes: int, ttl_seconds: Optional[int]) -> CacheLevel:
        """Determine optimal cache level for data."""
        # Small, frequently accessed data -> L1
        if size_bytes < 64 * 1024:  # < 64KB
            return CacheLevel.L1_MEMORY

        # Medium data with short TTL -> L2
        if size_bytes < 1024 * 1024 and (ttl_seconds is None or ttl_seconds < 3600):  # < 1MB, < 1 hour
            return CacheLevel.L2_REDIS

        # Large data or long-term storage -> L3
        return CacheLevel.L3_DISK

    async def _serialize_value(self, value: Any, compress: bool) -> bytes:
        """Serialize and optionally compress value."""
        try:
            # Serialize
            if isinstance(value, (str, int, float, bool)):
                serialized = json.dumps(value).encode()
            else:
                serialized = pickle.dumps(value)

            # Compress if enabled and beneficial
            if compress and len(serialized) > 1024:
                compressed = gzip.compress(serialized)
                if len(compressed) < len(serialized) * 0.8:  # 20% compression threshold
                    return compressed

            return serialized

        except Exception as e:
            logger.error(f"Serialization error: {e}")
            raise

    async def _deserialize_value(self, data: bytes, compressed: bool) -> Any:
        """Deserialize and optionally decompress value."""
        try:
            # Decompress if needed
            if compressed:
                data = gzip.decompress(data)

            # Try JSON first, then pickle
            try:
                return json.loads(data.decode())
            except (json.JSONDecodeError, UnicodeDecodeError):
                return pickle.loads(data)

        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            raise

    # L1 Cache Methods
    async def _get_l1(self, key: str) -> Optional[Any]:
        """Get from L1 memory cache."""
        with self.l1_lock:
            if key in self.l1_cache:
                entry = self.l1_cache[key]
                if isinstance(entry, CacheEntry):
                    if not entry.is_expired():
                        entry.update_access()
                        # Move to end (LRU)
                        self.l1_cache.move_to_end(key)
                        return entry.value
                    else:
                        # Remove expired entry
                        del self.l1_cache[key]
                        self.l1_current_size -= entry.size_bytes
                else:
                    # Legacy entry, just return value
                    self.l1_cache.move_to_end(key)
                    return entry
        return None

    async def _set_l1(self, key: str, value: Union[Any, CacheEntry]) -> bool:
        """Set in L1 memory cache."""
        try:
            with self.l1_lock:
                # Handle both CacheEntry and raw values
                if isinstance(value, CacheEntry):
                    entry = value
                    size_bytes = entry.size_bytes
                else:
                    # Create entry for raw value
                    estimated_size = len(str(value).encode())
                    entry = CacheEntry(
                        key=key,
                        value=value,
                        created_at=datetime.utcnow(),
                        last_accessed=datetime.utcnow(),
                        access_count=1,
                        ttl_seconds=None,
                        size_bytes=estimated_size,
                        compressed=False,
                        cache_level=CacheLevel.L1_MEMORY,
                        tags=[],
                        metadata={}
                    )
                    size_bytes = estimated_size

                # Remove existing entry if present
                if key in self.l1_cache:
                    old_entry = self.l1_cache[key]
                    if isinstance(old_entry, CacheEntry):
                        self.l1_current_size -= old_entry.size_bytes
                    del self.l1_cache[key]

                # Evict entries if necessary
                while (self.l1_current_size + size_bytes > self.l1_max_size_bytes and
                       len(self.l1_cache) > 0):
                    await self._evict_l1_entry()

                # Add new entry
                self.l1_cache[key] = entry
                self.l1_current_size += size_bytes

                return True

        except Exception as e:
            logger.error(f"L1 cache set error: {e}")
            return False

    async def _delete_l1(self, key: str) -> bool:
        """Delete from L1 cache."""
        try:
            with self.l1_lock:
                if key in self.l1_cache:
                    entry = self.l1_cache[key]
                    if isinstance(entry, CacheEntry):
                        self.l1_current_size -= entry.size_bytes
                    del self.l1_cache[key]
                    return True
            return False
        except Exception as e:
            logger.error(f"L1 cache delete error: {e}")
            return False

    async def _evict_l1_entry(self):
        """Evict entry from L1 cache using intelligent strategy."""
        if not self.l1_cache:
            return

        memory_pressure = self.l1_current_size / self.l1_max_size_bytes

        # Calculate eviction scores for all entries
        scores = {}
        for cache_key, entry in self.l1_cache.items():
            if isinstance(entry, CacheEntry):
                scores[cache_key] = self.eviction_manager.should_evict(entry, memory_pressure)
            else:
                # Legacy entry, give it high eviction score
                scores[cache_key] = 1.0

        # Evict entry with highest score
        if scores:
            key_to_evict = max(scores.keys(), key=lambda k: scores[k])
            entry = self.l1_cache[key_to_evict]
            if isinstance(entry, CacheEntry):
                self.l1_current_size -= entry.size_bytes
            del self.l1_cache[key_to_evict]

            with self.stats_lock:
                self.stats['evictions'] += 1

    # L2 Cache Methods (Redis)
    async def _get_l2(self, key: str) -> Optional[Any]:
        """Get from L2 Redis cache."""
        try:
            data = await self.redis_manager.get(f"quicknav:{key}")
            if data:
                # Try to deserialize as CacheEntry first
                try:
                    entry_data = pickle.loads(data)
                    if isinstance(entry_data, dict) and 'value' in entry_data:
                        entry = CacheEntry(**entry_data)
                        if not entry.is_expired():
                            entry.update_access()
                            return entry.value
                    else:
                        # Legacy data
                        return entry_data
                except:
                    # Fallback to raw data
                    return data.decode() if isinstance(data, bytes) else data
        except Exception as e:
            logger.error(f"L2 cache get error: {e}")
        return None

    async def _set_l2(self, key: str, value: Union[Any, CacheEntry], ttl_seconds: Optional[int] = None) -> bool:
        """Set in L2 Redis cache."""
        try:
            if isinstance(value, CacheEntry):
                # Serialize CacheEntry
                entry_dict = asdict(value)
                data = pickle.dumps(entry_dict)
                ttl = ttl_seconds or value.ttl_seconds
            else:
                # Serialize raw value
                data = pickle.dumps(value)
                ttl = ttl_seconds

            return await self.redis_manager.set(f"quicknav:{key}", data, ttl)

        except Exception as e:
            logger.error(f"L2 cache set error: {e}")
            return False

    async def _delete_l2(self, key: str) -> bool:
        """Delete from L2 Redis cache."""
        try:
            return await self.redis_manager.delete(f"quicknav:{key}")
        except Exception as e:
            logger.error(f"L2 cache delete error: {e}")
            return False

    # L3 Cache Methods (Disk)
    async def _get_l3(self, key: str) -> Optional[Any]:
        """Get from L3 disk cache."""
        try:
            cache_file = os.path.join(self.l3_cache_dir, f"{self._hash_key(key)}.cache")
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    data = f.read()

                # Try to deserialize as CacheEntry
                try:
                    entry_data = pickle.loads(data)
                    if isinstance(entry_data, dict) and 'value' in entry_data:
                        entry = CacheEntry(**entry_data)
                        if not entry.is_expired():
                            return entry.value
                    else:
                        return entry_data
                except:
                    # Fallback
                    return pickle.loads(data)

        except Exception as e:
            logger.error(f"L3 cache get error: {e}")
        return None

    async def _set_l3(self, key: str, value: Union[Any, CacheEntry], ttl_seconds: Optional[int] = None) -> bool:
        """Set in L3 disk cache."""
        try:
            cache_file = os.path.join(self.l3_cache_dir, f"{self._hash_key(key)}.cache")

            if isinstance(value, CacheEntry):
                data = pickle.dumps(asdict(value))
            else:
                data = pickle.dumps(value)

            with open(cache_file, 'wb') as f:
                f.write(data)

            return True

        except Exception as e:
            logger.error(f"L3 cache set error: {e}")
            return False

    async def _delete_l3(self, key: str) -> bool:
        """Delete from L3 disk cache."""
        try:
            cache_file = os.path.join(self.l3_cache_dir, f"{self._hash_key(key)}.cache")
            if os.path.exists(cache_file):
                os.remove(cache_file)
                return True
        except Exception as e:
            logger.error(f"L3 cache delete error: {e}")
        return False

    def _hash_key(self, key: str) -> str:
        """Generate hash for cache key."""
        return hashlib.md5(key.encode()).hexdigest()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self.stats_lock:
            total_requests = self.stats['total_requests']
            if total_requests > 0:
                overall_hit_rate = (
                    (self.stats['l1_hits'] + self.stats['l2_hits'] + self.stats['l3_hits'])
                    / total_requests
                )
                l1_hit_rate = self.stats['l1_hits'] / total_requests
                l2_hit_rate = self.stats['l2_hits'] / total_requests
                l3_hit_rate = self.stats['l3_hits'] / total_requests
            else:
                overall_hit_rate = l1_hit_rate = l2_hit_rate = l3_hit_rate = 0

            stats = self.stats.copy()

        with self.access_times_lock:
            avg_access_time = sum(self.access_times) / len(self.access_times) if self.access_times else 0

        with self.l1_lock:
            l1_utilization = self.l1_current_size / self.l1_max_size_bytes if self.l1_max_size_bytes > 0 else 0
            l1_entry_count = len(self.l1_cache)

        return {
            'overall_hit_rate': overall_hit_rate,
            'l1_hit_rate': l1_hit_rate,
            'l2_hit_rate': l2_hit_rate,
            'l3_hit_rate': l3_hit_rate,
            'avg_access_time_ms': avg_access_time,
            'l1_utilization': l1_utilization,
            'l1_entry_count': l1_entry_count,
            'l1_size_mb': self.l1_current_size / (1024 * 1024),
            'total_requests': total_requests,
            'evictions': stats['evictions'],
            'redis_available': self.redis_manager.is_available()
        }

    async def warm_cache(self, patterns: Dict[str, Callable]):
        """Warm cache with predefined patterns."""
        for pattern, data_source in patterns.items():
            await self.warmup_manager.schedule_warmup(pattern, data_source)

    def clear_all_caches(self):
        """Clear all cache levels."""
        # Clear L1
        with self.l1_lock:
            self.l1_cache.clear()
            self.l1_current_size = 0

        # Clear L2 (Redis) - async operation
        asyncio.create_task(self.redis_manager.delete_pattern("quicknav:*"))

        # Clear L3 (Disk)
        try:
            for file in os.listdir(self.l3_cache_dir):
                if file.endswith('.cache'):
                    os.remove(os.path.join(self.l3_cache_dir, file))
        except Exception as e:
            logger.error(f"Failed to clear L3 cache: {e}")


# Global cache manager instance
_global_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get or create global cache manager."""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager()
    return _global_cache_manager


# Decorator for caching function results
def cached(key_template: str = None,
           ttl_seconds: int = 3600,
           tags: List[str] = None,
           cache_manager: CacheManager = None):
    """Decorator for caching function results."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            if key_template:
                cache_key = key_template.format(*args, **kwargs)
            else:
                cache_key = f"{func.__module__}.{func.__name__}:{hash((args, tuple(kwargs.items())))}"

            cm = cache_manager or get_cache_manager()

            # Try to get from cache
            cached_result = await cm.get(cache_key, tags)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            await cm.set(cache_key, result, ttl_seconds, tags)
            return result

        def sync_wrapper(*args, **kwargs):
            return asyncio.run(async_wrapper(*args, **kwargs))

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator