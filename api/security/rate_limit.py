"""
BIBLOS v2 - Rate Limiting Module

Provides configurable rate limiting:
- Per-IP rate limiting
- Per-user rate limiting
- Per-endpoint rate limiting
- Redis backend for distributed deployments
- In-memory fallback

Features:
- Sliding window algorithm
- Configurable limits per endpoint
- Burst allowance
- Retry-After header support
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Tuple,
)
import functools

from fastapi import HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from opentelemetry import trace

tracer = trace.get_tracer(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""

    # Default limits (requests per window)
    default_limit: int = 100
    default_window_seconds: int = 60

    # Endpoint-specific limits: path -> (limit, window_seconds)
    endpoint_limits: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    # User-based limits
    authenticated_multiplier: float = 2.0  # Authenticated users get 2x limit

    # Response headers
    include_headers: bool = True
    header_prefix: str = "X-RateLimit"

    # Redis settings for distributed limiting
    redis_url: Optional[str] = None
    redis_key_prefix: str = "biblos:ratelimit:"

    # Exempt paths
    exempt_paths: set = field(default_factory=lambda: {"/health", "/metrics"})


class RateLimitBackend(ABC):
    """Abstract backend for rate limit storage."""

    @abstractmethod
    async def is_allowed(
        self,
        key: str,
        limit: int,
        window_seconds: int,
    ) -> Tuple[bool, int, int, int]:
        """
        Check if request is allowed.

        Returns:
            (allowed, remaining, reset_time, retry_after)
        """
        pass


class InMemoryBackend(RateLimitBackend):
    """In-memory rate limit backend using sliding window."""

    def __init__(self):
        # key -> list of timestamps
        self._requests: Dict[str, list] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def is_allowed(
        self,
        key: str,
        limit: int,
        window_seconds: int,
    ) -> Tuple[bool, int, int, int]:
        """Check if request is allowed using sliding window."""
        async with self._lock:
            now = time.time()
            window_start = now - window_seconds

            # Clean old entries
            self._requests[key] = [
                ts for ts in self._requests[key]
                if ts > window_start
            ]

            current_count = len(self._requests[key])
            remaining = max(0, limit - current_count - 1)
            reset_time = int(now + window_seconds)

            if current_count >= limit:
                # Calculate retry after
                oldest = min(self._requests[key]) if self._requests[key] else now
                retry_after = int(oldest + window_seconds - now) + 1
                return False, 0, reset_time, retry_after

            # Add this request
            self._requests[key].append(now)
            return True, remaining, reset_time, 0


class RedisBackend(RateLimitBackend):
    """Redis rate limit backend for distributed deployments."""

    def __init__(self, redis_url: str, key_prefix: str = "ratelimit:"):
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self._redis = None

    async def _get_redis(self):
        """Get or create Redis connection."""
        if self._redis is None:
            try:
                import redis.asyncio as aioredis
                self._redis = await aioredis.from_url(self.redis_url)
            except ImportError:
                raise RuntimeError("redis package required for Redis backend")
        return self._redis

    async def is_allowed(
        self,
        key: str,
        limit: int,
        window_seconds: int,
    ) -> Tuple[bool, int, int, int]:
        """Check if request is allowed using Redis sliding window."""
        redis = await self._get_redis()
        full_key = f"{self.key_prefix}{key}"
        now = time.time()
        window_start = now - window_seconds

        # Use Redis transaction for atomic operations
        async with redis.pipeline(transaction=True) as pipe:
            # Remove old entries and count current
            pipe.zremrangebyscore(full_key, 0, window_start)
            pipe.zcard(full_key)
            pipe.zadd(full_key, {str(now): now})
            pipe.expire(full_key, window_seconds + 1)
            results = await pipe.execute()

        current_count = results[1]
        remaining = max(0, limit - current_count - 1)
        reset_time = int(now + window_seconds)

        if current_count >= limit:
            # Get oldest timestamp
            oldest_entries = await redis.zrange(full_key, 0, 0, withscores=True)
            if oldest_entries:
                oldest = oldest_entries[0][1]
                retry_after = int(oldest + window_seconds - now) + 1
            else:
                retry_after = window_seconds
            return False, 0, reset_time, retry_after

        return True, remaining, reset_time, 0


class RateLimiter:
    """
    Rate limiter with configurable backend.

    Usage:
        limiter = RateLimiter(config)

        # Check if allowed
        allowed, remaining, reset_time, retry_after = await limiter.check(
            request,
            user_id="user123"
        )
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()

        # Initialize backend
        if self.config.redis_url:
            self._backend = RedisBackend(
                self.config.redis_url,
                self.config.redis_key_prefix,
            )
        else:
            self._backend = InMemoryBackend()

    def _get_limit_for_endpoint(
        self,
        path: str,
        authenticated: bool = False,
    ) -> Tuple[int, int]:
        """Get rate limit for an endpoint."""
        if path in self.config.endpoint_limits:
            limit, window = self.config.endpoint_limits[path]
        else:
            limit = self.config.default_limit
            window = self.config.default_window_seconds

        if authenticated:
            limit = int(limit * self.config.authenticated_multiplier)

        return limit, window

    def _get_client_key(
        self,
        request: Request,
        user_id: Optional[str] = None,
    ) -> str:
        """Generate rate limit key for client."""
        if user_id:
            return f"user:{user_id}"

        # Use forwarded IP if behind proxy
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"

        return f"ip:{client_ip}"

    async def check(
        self,
        request: Request,
        user_id: Optional[str] = None,
    ) -> Tuple[bool, int, int, int]:
        """
        Check if request is allowed.

        Returns:
            (allowed, remaining, reset_time, retry_after)
        """
        path = request.url.path

        # Skip exempt paths
        if path in self.config.exempt_paths:
            return True, -1, 0, 0

        client_key = self._get_client_key(request, user_id)
        endpoint_key = f"{client_key}:{path}"

        limit, window = self._get_limit_for_endpoint(
            path,
            authenticated=user_id is not None,
        )

        with tracer.start_as_current_span("ratelimit.check") as span:
            span.set_attribute("ratelimit.key", endpoint_key)
            span.set_attribute("ratelimit.limit", limit)
            span.set_attribute("ratelimit.window", window)

            allowed, remaining, reset_time, retry_after = await self._backend.is_allowed(
                endpoint_key,
                limit,
                window,
            )

            span.set_attribute("ratelimit.allowed", allowed)
            span.set_attribute("ratelimit.remaining", remaining)

            return allowed, remaining, reset_time, retry_after

    def add_headers(
        self,
        response: Response,
        limit: int,
        remaining: int,
        reset_time: int,
    ) -> None:
        """Add rate limit headers to response."""
        if not self.config.include_headers:
            return

        prefix = self.config.header_prefix
        response.headers[f"{prefix}-Limit"] = str(limit)
        response.headers[f"{prefix}-Remaining"] = str(remaining)
        response.headers[f"{prefix}-Reset"] = str(reset_time)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""

    def __init__(self, app, config: Optional[RateLimitConfig] = None):
        super().__init__(app)
        self.limiter = RateLimiter(config)
        self.config = config or RateLimitConfig()

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with rate limiting."""
        # Get user ID from request state if authenticated
        user_id = getattr(request.state, "user_id", None)

        allowed, remaining, reset_time, retry_after = await self.limiter.check(
            request,
            user_id,
        )

        if not allowed:
            with tracer.start_as_current_span("ratelimit.exceeded") as span:
                span.set_attribute("ratelimit.retry_after", retry_after)

            response = Response(
                content='{"detail": "Rate limit exceeded"}',
                status_code=429,
                media_type="application/json",
            )
            response.headers["Retry-After"] = str(retry_after)
            self.limiter.add_headers(
                response,
                self.config.default_limit,
                0,
                reset_time,
            )
            return response

        response = await call_next(request)

        # Add rate limit headers
        limit, _ = self.limiter._get_limit_for_endpoint(
            request.url.path,
            authenticated=user_id is not None,
        )
        self.limiter.add_headers(response, limit, remaining, reset_time)

        return response


def rate_limit(
    limit: int = 100,
    window_seconds: int = 60,
) -> Callable:
    """
    Decorator for endpoint-specific rate limiting.

    Usage:
        @app.get("/expensive")
        @rate_limit(limit=10, window_seconds=60)
        async def expensive_endpoint():
            ...
    """
    limiter = RateLimiter(RateLimitConfig(
        default_limit=limit,
        default_window_seconds=window_seconds,
    ))

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            user_id = getattr(request.state, "user_id", None)

            allowed, remaining, reset_time, retry_after = await limiter.check(
                request,
                user_id,
            )

            if not allowed:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded",
                    headers={"Retry-After": str(retry_after)},
                )

            return await func(request, *args, **kwargs)

        return wrapper

    return decorator
