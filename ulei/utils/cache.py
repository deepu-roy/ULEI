"""
Response caching utilities for cost control and performance optimization.
"""

import hashlib
import json
import logging
import pickle
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ulei.utils.errors import CacheError

logger = logging.getLogger(__name__)


class CacheBackend:
    """Abstract base class for cache backends."""

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        raise NotImplementedError

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL."""
        raise NotImplementedError

    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        raise NotImplementedError

    def clear(self) -> None:
        """Clear all cache entries."""
        raise NotImplementedError

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        raise NotImplementedError


class MemoryCache(CacheBackend):
    """In-memory cache backend with TTL support."""

    def __init__(self, max_size: int = 1000):
        """Initialize memory cache.

        Args:
            max_size: Maximum number of entries to keep in memory
        """
        self.max_size = max_size
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        if key not in self._cache:
            self._misses += 1
            return None

        entry = self._cache[key]

        # Check TTL
        if entry["expires_at"] and time.time() > entry["expires_at"]:
            del self._cache[key]
            if key in self._access_times:
                del self._access_times[key]
            self._misses += 1
            return None

        # Update access time and return value
        self._access_times[key] = time.time()
        self._hits += 1
        return entry["value"]

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in memory cache."""
        # Calculate expiration time
        expires_at = None
        if ttl is not None:
            expires_at = time.time() + ttl

        # Evict old entries if at capacity
        if len(self._cache) >= self.max_size and key not in self._cache:
            self._evict_oldest()

        # Store entry
        self._cache[key] = {"value": value, "expires_at": expires_at, "created_at": time.time()}
        self._access_times[key] = time.time()

    def delete(self, key: str) -> bool:
        """Delete value from memory cache."""
        if key in self._cache:
            del self._cache[key]
            if key in self._access_times:
                del self._access_times[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all memory cache entries."""
        self._cache.clear()
        self._access_times.clear()
        self._hits = 0
        self._misses = 0

    def _evict_oldest(self) -> None:
        """Evict oldest accessed entry."""
        if not self._access_times:
            return

        oldest_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        self.delete(oldest_key)

    def stats(self) -> Dict[str, Any]:
        """Get memory cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        return {
            "backend": "memory",
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
        }


class SQLiteCache(CacheBackend):
    """SQLite-based persistent cache backend."""

    def __init__(self, cache_file: Union[str, Path]):
        """Initialize SQLite cache.

        Args:
            cache_file: Path to SQLite cache file
        """
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)

        self._hits = 0
        self._misses = 0

        self._init_database()

    def _init_database(self) -> None:
        """Initialize SQLite database schema."""
        try:
            with sqlite3.connect(str(self.cache_file)) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cache_entries (
                        key TEXT PRIMARY KEY,
                        value BLOB NOT NULL,
                        created_at REAL NOT NULL,
                        expires_at REAL,
                        access_count INTEGER DEFAULT 0,
                        last_accessed REAL NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_expires_at ON cache_entries(expires_at)
                """)
                conn.commit()
        except Exception as e:
            raise CacheError(f"Failed to initialize SQLite cache: {e}") from e

    def get(self, key: str) -> Optional[Any]:
        """Get value from SQLite cache."""
        try:
            with sqlite3.connect(str(self.cache_file)) as conn:
                cursor = conn.execute(
                    "SELECT value, expires_at FROM cache_entries WHERE key = ?", (key,)
                )
                row = cursor.fetchone()

                if row is None:
                    self._misses += 1
                    return None

                value_blob, expires_at = row

                # Check TTL
                if expires_at and time.time() > expires_at:
                    conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                    conn.commit()
                    self._misses += 1
                    return None

                # Update access statistics
                conn.execute(
                    """
                    UPDATE cache_entries
                    SET access_count = access_count + 1, last_accessed = ?
                    WHERE key = ?
                """,
                    (time.time(), key),
                )
                conn.commit()

                # Deserialize and return value
                value = pickle.loads(value_blob)
                self._hits += 1
                return value

        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self._misses += 1
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in SQLite cache."""
        try:
            # Serialize value
            value_blob = pickle.dumps(value)

            # Calculate expiration time
            expires_at = None
            if ttl is not None:
                expires_at = time.time() + ttl

            current_time = time.time()

            with sqlite3.connect(str(self.cache_file)) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cache_entries
                    (key, value, created_at, expires_at, access_count, last_accessed)
                    VALUES (?, ?, ?, ?, 0, ?)
                """,
                    (key, value_blob, current_time, expires_at, current_time),
                )
                conn.commit()

        except Exception as e:
            raise CacheError(f"Failed to set cache entry: {e}") from e

    def delete(self, key: str) -> bool:
        """Delete value from SQLite cache."""
        try:
            with sqlite3.connect(str(self.cache_file)) as conn:
                cursor = conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False

    def clear(self) -> None:
        """Clear all SQLite cache entries."""
        try:
            with sqlite3.connect(str(self.cache_file)) as conn:
                conn.execute("DELETE FROM cache_entries")
                conn.commit()

            self._hits = 0
            self._misses = 0
        except Exception as e:
            raise CacheError(f"Failed to clear cache: {e}") from e

    def cleanup_expired(self) -> int:
        """Remove expired cache entries.

        Returns:
            Number of entries removed
        """
        try:
            current_time = time.time()
            with sqlite3.connect(str(self.cache_file)) as conn:
                cursor = conn.execute(
                    "DELETE FROM cache_entries WHERE expires_at IS NOT NULL AND expires_at < ?",
                    (current_time,),
                )
                conn.commit()
                return cursor.rowcount
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")
            return 0

    def stats(self) -> Dict[str, Any]:
        """Get SQLite cache statistics."""
        try:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

            with sqlite3.connect(str(self.cache_file)) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM cache_entries")
                total_entries = cursor.fetchone()[0]

                cursor = conn.execute(
                    """
                    SELECT COUNT(*) FROM cache_entries
                    WHERE expires_at IS NOT NULL AND expires_at < ?
                """,
                    (time.time(),),
                )
                expired_entries = cursor.fetchone()[0]

                # Get file size
                file_size = self.cache_file.stat().st_size if self.cache_file.exists() else 0

                return {
                    "backend": "sqlite",
                    "size": total_entries,
                    "expired": expired_entries,
                    "file_size_bytes": file_size,
                    "hits": self._hits,
                    "misses": self._misses,
                    "hit_rate": hit_rate,
                }
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {"backend": "sqlite", "error": str(e)}


class EvaluationCache:
    """High-level cache for evaluation results with key generation."""

    def __init__(
        self,
        backend: Optional[CacheBackend] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        default_ttl: int = 86400,  # 24 hours
        enabled: bool = True,
    ):
        """Initialize evaluation cache.

        Args:
            backend: Cache backend to use
            cache_dir: Directory for cache files (used if backend is None)
            default_ttl: Default TTL in seconds
            enabled: Whether caching is enabled
        """
        self.default_ttl = default_ttl
        self.enabled = enabled

        if not enabled:
            self.backend = None
        elif backend is not None:
            self.backend = backend
        else:
            # Create default SQLite backend
            cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / "evaluation_cache"
            cache_file = cache_dir / "evaluations.db"
            self.backend = SQLiteCache(cache_file)

    def get_evaluation_result(
        self,
        provider: str,
        metric: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        context: Optional[list] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Get cached evaluation result.

        Args:
            provider: Provider name
            metric: Metric name
            input_data: Input data
            output_data: Output data
            context: Optional context data
            config: Optional configuration

        Returns:
            Cached result or None if not found
        """
        if not self.enabled or not self.backend:
            return None

        cache_key = self._generate_cache_key(
            provider, metric, input_data, output_data, context, config
        )

        return self.backend.get(cache_key)

    def set_evaluation_result(
        self,
        provider: str,
        metric: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        result: Any,
        context: Optional[list] = None,
        config: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> None:
        """Cache evaluation result.

        Args:
            provider: Provider name
            metric: Metric name
            input_data: Input data
            output_data: Output data
            result: Evaluation result to cache
            context: Optional context data
            config: Optional configuration
            ttl: TTL in seconds (uses default if None)
        """
        if not self.enabled or not self.backend:
            return

        cache_key = self._generate_cache_key(
            provider, metric, input_data, output_data, context, config
        )

        ttl = ttl if ttl is not None else self.default_ttl
        self.backend.set(cache_key, result, ttl)

    def _generate_cache_key(
        self,
        provider: str,
        metric: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        context: Optional[list] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate deterministic cache key.

        Args:
            provider: Provider name
            metric: Metric name
            input_data: Input data
            output_data: Output data
            context: Optional context data
            config: Optional configuration

        Returns:
            Cache key string
        """
        # Create content dictionary for hashing
        content = {
            "provider": provider,
            "metric": metric,
            "input": input_data,
            "output": output_data,
            "context": context,
            "config": config or {},
        }

        # Convert to deterministic JSON string
        content_str = json.dumps(content, sort_keys=True, default=str)

        # Generate SHA-256 hash
        return hashlib.sha256(content_str.encode("utf-8")).hexdigest()

    def clear(self) -> None:
        """Clear all cached results."""
        if self.backend:
            self.backend.clear()

    def cleanup_expired(self) -> int:
        """Remove expired cache entries.

        Returns:
            Number of entries removed (if supported by backend)
        """
        if hasattr(self.backend, "cleanup_expired"):
            return self.backend.cleanup_expired()
        return 0

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.backend:
            return {"enabled": False}

        stats = self.backend.stats()
        stats["enabled"] = self.enabled
        stats["default_ttl"] = self.default_ttl
        return stats

    def is_enabled(self) -> bool:
        """Check if caching is enabled."""
        return self.enabled and self.backend is not None


# Global cache instance
_global_cache: Optional[EvaluationCache] = None


def get_evaluation_cache(
    cache_dir: Optional[Union[str, Path]] = None, enabled: bool = True
) -> EvaluationCache:
    """Get global evaluation cache instance.

    Args:
        cache_dir: Cache directory (only used on first call)
        enabled: Whether caching should be enabled

    Returns:
        EvaluationCache instance
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = EvaluationCache(cache_dir=cache_dir, enabled=enabled)
    return _global_cache


def configure_cache(
    backend_type: str = "sqlite",
    cache_dir: Optional[Union[str, Path]] = None,
    default_ttl: int = 86400,
    enabled: bool = True,
    **backend_kwargs,
) -> EvaluationCache:
    """Configure global evaluation cache.

    Args:
        backend_type: Type of cache backend ('memory' or 'sqlite')
        cache_dir: Directory for cache files
        default_ttl: Default TTL in seconds
        enabled: Whether caching is enabled
        **backend_kwargs: Additional backend-specific arguments

    Returns:
        Configured EvaluationCache instance
    """
    global _global_cache

    # Create backend
    if backend_type == "memory":
        max_size = backend_kwargs.get("max_size", 1000)
        backend = MemoryCache(max_size=max_size)
    elif backend_type == "sqlite":
        cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / "evaluation_cache"
        cache_file = cache_dir / "evaluations.db"
        backend = SQLiteCache(cache_file)
    else:
        raise ValueError(f"Unsupported cache backend type: {backend_type}")

    _global_cache = EvaluationCache(backend=backend, default_ttl=default_ttl, enabled=enabled)

    return _global_cache
