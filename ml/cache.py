"""
BIBLOS v2 - ML Caching Module

Provides optimized caching for ML inference:
- O(1) LRU cache with bounded size
- TTL-based expiration
- Memory-aware eviction
- Async-compatible interface
- OpenTelemetry metrics

Fixes the unbounded cache growth issue identified in the original implementation.
"""

from __future__ import annotations

import asyncio
import hashlib
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Optional,
    TypeVar,
)
from functools import wraps

import numpy as np

from opentelemetry import trace

tracer = trace.get_tracer(__name__)

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with metadata for eviction decisions."""

    value: T
    created_at: float
    last_accessed: float
    access_count: int = 0
    size_bytes: int = 0


@dataclass
class CacheStats:
    """Statistics for cache monitoring."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    entry_count: int = 0
    avg_access_time_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "evictions": self.evictions,
            "total_size_bytes": self.total_size_bytes,
            "entry_count": self.entry_count,
            "avg_access_time_ms": self.avg_access_time_ms,
        }


class LRUCache(Generic[T]):
    """
    O(1) LRU Cache with bounded size and TTL.

    Features:
    - O(1) get, put, and eviction operations using OrderedDict
    - Memory-based size limiting
    - TTL-based expiration
    - Thread-safe operations

    Usage:
        cache = LRUCache[np.ndarray](max_size=1000, max_memory_mb=512)
        cache.put("key", embedding)
        result = cache.get("key")
    """

    def __init__(
        self,
        max_size: int = 10000,
        max_memory_mb: int = 512,
        ttl_seconds: Optional[float] = 3600.0,
    ):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.ttl_seconds = ttl_seconds

        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats()
        self._total_access_time = 0.0
        self._access_count = 0

    def _estimate_size(self, value: T) -> int:
        """Estimate memory size of a value."""
        if isinstance(value, np.ndarray):
            return value.nbytes
        elif hasattr(value, "__sizeof__"):
            return value.__sizeof__()
        else:
            return sys.getsizeof(value)

    def _is_expired(self, entry: CacheEntry[T]) -> bool:
        """Check if an entry has expired."""
        if self.ttl_seconds is None:
            return False
        return time.monotonic() - entry.created_at > self.ttl_seconds

    def _evict_oldest(self) -> None:
        """Evict the least recently used entry. Must be called with lock held."""
        if self._cache:
            key, entry = self._cache.popitem(last=False)
            self._stats.total_size_bytes -= entry.size_bytes
            self._stats.entry_count -= 1
            self._stats.evictions += 1

    def _evict_until_fits(self, new_size: int) -> None:
        """Evict entries until new item fits. Must be called with lock held."""
        # Evict expired entries first
        expired_keys = [
            key for key, entry in self._cache.items()
            if self._is_expired(entry)
        ]
        for key in expired_keys:
            entry = self._cache.pop(key)
            self._stats.total_size_bytes -= entry.size_bytes
            self._stats.entry_count -= 1
            self._stats.evictions += 1

        # Evict by count limit
        while len(self._cache) >= self.max_size:
            self._evict_oldest()

        # Evict by memory limit
        while self._stats.total_size_bytes + new_size > self.max_memory_bytes and self._cache:
            self._evict_oldest()

    def get(self, key: str) -> Optional[T]:
        """
        Get a value from the cache.

        O(1) operation that also updates access order.
        """
        start_time = time.monotonic()

        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats.misses += 1
                return None

            if self._is_expired(entry):
                # Expired entry - remove and return miss
                self._cache.pop(key)
                self._stats.total_size_bytes -= entry.size_bytes
                self._stats.entry_count -= 1
                self._stats.misses += 1
                return None

            # Move to end (most recently used) - O(1) with OrderedDict
            self._cache.move_to_end(key)

            # Update access metadata
            entry.last_accessed = time.monotonic()
            entry.access_count += 1

            self._stats.hits += 1

            # Track access time for stats
            access_time = (time.monotonic() - start_time) * 1000
            self._total_access_time += access_time
            self._access_count += 1
            self._stats.avg_access_time_ms = self._total_access_time / self._access_count

            return entry.value

    def put(self, key: str, value: T) -> None:
        """
        Put a value into the cache.

        O(1) operation for insertion, may trigger eviction.
        """
        size = self._estimate_size(value)
        now = time.monotonic()

        with self._lock:
            # Update existing entry
            if key in self._cache:
                old_entry = self._cache.pop(key)
                self._stats.total_size_bytes -= old_entry.size_bytes

            # Evict if necessary
            self._evict_until_fits(size)

            # Add new entry
            entry = CacheEntry(
                value=value,
                created_at=now,
                last_accessed=now,
                size_bytes=size,
            )
            self._cache[key] = entry
            self._stats.total_size_bytes += size
            self._stats.entry_count = len(self._cache)

    def contains(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        with self._lock:
            if key not in self._cache:
                return False
            if self._is_expired(self._cache[key]):
                return False
            return True

    def delete(self, key: str) -> bool:
        """Remove a key from the cache."""
        with self._lock:
            if key in self._cache:
                entry = self._cache.pop(key)
                self._stats.total_size_bytes -= entry.size_bytes
                self._stats.entry_count -= 1
                return True
            return False

    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            self._cache.clear()
            self._stats = CacheStats()

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                total_size_bytes=self._stats.total_size_bytes,
                entry_count=self._stats.entry_count,
                avg_access_time_ms=self._stats.avg_access_time_ms,
            )


class AsyncLRUCache(Generic[T]):
    """
    Async-compatible LRU cache wrapper.

    Provides async interface over the thread-safe LRUCache.
    """

    def __init__(
        self,
        max_size: int = 10000,
        max_memory_mb: int = 512,
        ttl_seconds: Optional[float] = 3600.0,
    ):
        self._cache = LRUCache[T](max_size, max_memory_mb, ttl_seconds)
        self._async_lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[T]:
        """Get a value asynchronously."""
        async with self._async_lock:
            return self._cache.get(key)

    async def put(self, key: str, value: T) -> None:
        """Put a value asynchronously."""
        async with self._async_lock:
            self._cache.put(key, value)

    async def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], T],
    ) -> T:
        """Get from cache or compute and store."""
        result = await self.get(key)
        if result is not None:
            return result

        # Compute outside the lock
        result = compute_fn()
        await self.put(key, result)
        return result

    async def get_or_compute_async(
        self,
        key: str,
        compute_fn: Callable[[], Any],
    ) -> T:
        """Get from cache or compute asynchronously and store."""
        result = await self.get(key)
        if result is not None:
            return result

        # Compute outside the lock
        result = await compute_fn()
        await self.put(key, result)
        return result

    async def clear(self) -> None:
        """Clear the cache."""
        async with self._async_lock:
            self._cache.clear()

    def get_stats(self) -> CacheStats:
        """Get cache statistics (synchronous)."""
        return self._cache.get_stats()


def embedding_cache_key(text: str, model_name: Optional[str] = None) -> str:
    """
    Generate a cache key for an embedding.

    Uses SHA256 hash for consistent key length and collision resistance.
    """
    content = text
    if model_name:
        content = f"{model_name}:{text}"
    return hashlib.sha256(content.encode()).hexdigest()


def cached_embedding(
    cache: LRUCache[np.ndarray],
    model_name: Optional[str] = None,
) -> Callable:
    """
    Decorator for caching embedding functions.

    Usage:
        @cached_embedding(embedding_cache, "mpnet")
        def embed(text: str) -> np.ndarray:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(text: str, *args, **kwargs) -> np.ndarray:
            key = embedding_cache_key(text, model_name)

            with tracer.start_as_current_span("cache.embedding_lookup") as span:
                result = cache.get(key)
                if result is not None:
                    span.set_attribute("cache.hit", True)
                    return result
                span.set_attribute("cache.hit", False)

            with tracer.start_as_current_span("cache.embedding_compute"):
                result = func(text, *args, **kwargs)

            cache.put(key, result)
            return result

        return wrapper

    return decorator


def cached_embedding_async(
    cache: AsyncLRUCache[np.ndarray],
    model_name: Optional[str] = None,
) -> Callable:
    """
    Async decorator for caching embedding functions.

    Usage:
        @cached_embedding_async(embedding_cache, "mpnet")
        async def embed(text: str) -> np.ndarray:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(text: str, *args, **kwargs) -> np.ndarray:
            key = embedding_cache_key(text, model_name)

            with tracer.start_as_current_span("cache.embedding_lookup") as span:
                result = await cache.get(key)
                if result is not None:
                    span.set_attribute("cache.hit", True)
                    return result
                span.set_attribute("cache.hit", False)

            with tracer.start_as_current_span("cache.embedding_compute"):
                result = await func(text, *args, **kwargs)

            await cache.put(key, result)
            return result

        return wrapper

    return decorator


# Global embedding cache instance
_embedding_cache: Optional[AsyncLRUCache[np.ndarray]] = None


def get_embedding_cache(
    max_size: int = 10000,
    max_memory_mb: int = 512,
) -> AsyncLRUCache[np.ndarray]:
    """Get or create the global embedding cache."""
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = AsyncLRUCache[np.ndarray](
            max_size=max_size,
            max_memory_mb=max_memory_mb,
            ttl_seconds=3600.0,
        )
    return _embedding_cache
