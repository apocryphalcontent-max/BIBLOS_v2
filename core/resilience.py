"""
BIBLOS v2 - Resilience Patterns

Implements robust resilience patterns for fault-tolerant operation:
- Circuit Breaker: Prevents cascade failures
- Retry Policy: Configurable retry with backoff
- Bulkhead: Resource isolation and limiting
- Combined resilient decorator

All patterns integrate with OpenTelemetry for observability.
"""

from __future__ import annotations

import asyncio
import functools
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
    ParamSpec,
)

from opentelemetry import trace

from core.errors import (
    BiblosError,
    BiblosResourceError,
    BiblosTimeoutError,
    ErrorContext,
)

T = TypeVar("T")
P = ParamSpec("P")

tracer = trace.get_tracer(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: float = 30.0
    excluded_exceptions: Set[Type[Exception]] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if self.success_threshold < 1:
            raise ValueError("success_threshold must be >= 1")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.

    Prevents cascade failures by tracking failures and
    temporarily blocking calls when failure rate is high.

    States:
    - CLOSED: Normal operation, tracking failures
    - OPEN: Too many failures, rejecting all calls
    - HALF_OPEN: Testing if service recovered

    Usage:
        breaker = CircuitBreaker("my_service")

        async with breaker:
            await some_risky_operation()
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Current circuit state with automatic timeout check."""
        if self._state == CircuitState.OPEN:
            if self._should_attempt_reset():
                return CircuitState.HALF_OPEN
        return self._state

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if not self._last_failure_time:
            return True
        elapsed = datetime.now(timezone.utc) - self._last_failure_time
        return elapsed.total_seconds() >= self.config.timeout_seconds

    async def __aenter__(self) -> "CircuitBreaker":
        """Context manager entry - check if call is allowed."""
        async with self._lock:
            state = self.state

            if state == CircuitState.OPEN:
                raise BiblosResourceError(
                    message=f"Circuit breaker '{self.name}' is OPEN",
                    resource_type="circuit_breaker",
                    context=ErrorContext.from_current_span(
                        operation="circuit_check",
                        component=self.name,
                    ),
                )

            if state == CircuitState.HALF_OPEN:
                # Allow single test request
                self._state = CircuitState.HALF_OPEN

        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> bool:
        """Context manager exit - record success/failure."""
        async with self._lock:
            if exc_type is None:
                await self._record_success()
            elif exc_type and not issubclass(exc_type, tuple(self.config.excluded_exceptions)):
                await self._record_failure()

        return False  # Don't suppress exceptions

    async def _record_success(self) -> None:
        """Record successful call."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.success_threshold:
                self._reset()
        elif self._state == CircuitState.CLOSED:
            # Decay failure count on success
            self._failure_count = max(0, self._failure_count - 1)

    async def _record_failure(self) -> None:
        """Record failed call."""
        self._failure_count += 1
        self._last_failure_time = datetime.now(timezone.utc)
        self._success_count = 0

        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
        elif self._failure_count >= self.config.failure_threshold:
            self._state = CircuitState.OPEN

    def _reset(self) -> None:
        """Reset circuit to closed state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None

    def get_metrics(self) -> Dict[str, Any]:
        """Get current circuit metrics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure": self._last_failure_time.isoformat() if self._last_failure_time else None,
        }


@dataclass
class RetryConfig:
    """Configuration for retry policy."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: Set[Type[Exception]] = field(
        default_factory=lambda: {Exception}
    )
    non_retryable_exceptions: Set[Type[Exception]] = field(default_factory=set)


class RetryPolicy:
    """
    Configurable retry policy with exponential backoff.

    Features:
    - Exponential backoff with optional jitter
    - Configurable retryable exceptions
    - Maximum delay cap
    - OpenTelemetry tracing

    Usage:
        policy = RetryPolicy(max_attempts=3)

        @policy.wrap
        async def my_function():
            ...
    """

    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        delay = min(
            self.config.base_delay * (self.config.exponential_base ** attempt),
            self.config.max_delay
        )

        if self.config.jitter:
            delay *= (0.5 + random.random())

        return delay

    def is_retryable(self, exception: Exception) -> bool:
        """Check if exception should trigger retry."""
        exc_type = type(exception)

        # Check non-retryable first
        if any(issubclass(exc_type, t) for t in self.config.non_retryable_exceptions):
            return False

        # Then check retryable
        return any(issubclass(exc_type, t) for t in self.config.retryable_exceptions)

    def wrap(
        self,
        func: Callable[P, T],
    ) -> Callable[P, T]:
        """Wrap function with retry logic."""

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Optional[Exception] = None

            for attempt in range(self.config.max_attempts):
                with tracer.start_as_current_span(
                    f"retry.attempt_{attempt}",
                ) as span:
                    span.set_attribute("retry.attempt", attempt)
                    span.set_attribute("retry.max_attempts", self.config.max_attempts)

                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        span.set_attribute("retry.exception", type(e).__name__)

                        if not self.is_retryable(e):
                            raise

                        if attempt < self.config.max_attempts - 1:
                            delay = self.calculate_delay(attempt)
                            span.set_attribute("retry.delay_seconds", delay)
                            await asyncio.sleep(delay)

            raise last_exception  # type: ignore

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Optional[Exception] = None

            for attempt in range(self.config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if not self.is_retryable(e):
                        raise

                    if attempt < self.config.max_attempts - 1:
                        delay = self.calculate_delay(attempt)
                        time.sleep(delay)

            raise last_exception  # type: ignore

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore


@dataclass
class BulkheadConfig:
    """Configuration for bulkhead pattern."""

    max_concurrent: int = 10
    max_waiting: int = 100
    timeout_seconds: float = 30.0


class Bulkhead:
    """
    Bulkhead pattern for resource isolation.

    Limits concurrent executions to prevent resource exhaustion
    and provides isolation between different operations.

    Usage:
        bulkhead = Bulkhead("database_ops", max_concurrent=10)

        async with bulkhead:
            await database_operation()
    """

    def __init__(
        self,
        name: str,
        config: Optional[BulkheadConfig] = None,
    ):
        self.name = name
        self.config = config or BulkheadConfig()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
        self._waiting = 0
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> "Bulkhead":
        """Acquire bulkhead slot."""
        async with self._lock:
            if self._waiting >= self.config.max_waiting:
                raise BiblosResourceError(
                    message=f"Bulkhead '{self.name}' queue is full",
                    resource_type="bulkhead",
                    current_usage=float(self._waiting),
                    limit=float(self.config.max_waiting),
                )
            self._waiting += 1

        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self.config.timeout_seconds,
            )
        except asyncio.TimeoutError:
            async with self._lock:
                self._waiting -= 1
            raise BiblosTimeoutError(
                message=f"Bulkhead '{self.name}' acquisition timeout",
                timeout_seconds=self.config.timeout_seconds,
            )

        async with self._lock:
            self._waiting -= 1

        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> bool:
        """Release bulkhead slot."""
        self._semaphore.release()
        return False

    def get_metrics(self) -> Dict[str, Any]:
        """Get current bulkhead metrics."""
        return {
            "name": self.name,
            "max_concurrent": self.config.max_concurrent,
            "available": self._semaphore._value,  # type: ignore
            "waiting": self._waiting,
        }


# Convenience decorators

def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    retryable_exceptions: Optional[Set[Type[Exception]]] = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for adding retry logic to a function.

    Usage:
        @with_retry(max_attempts=3)
        async def my_function():
            ...
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        retryable_exceptions=retryable_exceptions or {Exception},
    )
    policy = RetryPolicy(config)
    return policy.wrap


def with_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    timeout_seconds: float = 30.0,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for adding circuit breaker to a function.

    Usage:
        @with_circuit_breaker("my_service")
        async def my_function():
            ...
    """
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        timeout_seconds=timeout_seconds,
    )
    breaker = CircuitBreaker(name, config)

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            async with breaker:
                return await func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def with_bulkhead(
    name: str,
    max_concurrent: int = 10,
    timeout_seconds: float = 30.0,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for adding bulkhead to a function.

    Usage:
        @with_bulkhead("database", max_concurrent=10)
        async def my_function():
            ...
    """
    config = BulkheadConfig(
        max_concurrent=max_concurrent,
        timeout_seconds=timeout_seconds,
    )
    bulkhead = Bulkhead(name, config)

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            async with bulkhead:
                return await func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def resilient(
    circuit_breaker_name: Optional[str] = None,
    max_retries: int = 3,
    bulkhead_name: Optional[str] = None,
    bulkhead_max_concurrent: int = 10,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Combined resilience decorator with circuit breaker, retry, and bulkhead.

    Usage:
        @resilient(
            circuit_breaker_name="my_service",
            max_retries=3,
            bulkhead_name="my_pool",
            bulkhead_max_concurrent=10,
        )
        async def my_function():
            ...
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        wrapped = func

        # Apply retry (innermost)
        if max_retries > 0:
            wrapped = with_retry(max_attempts=max_retries)(wrapped)

        # Apply circuit breaker
        if circuit_breaker_name:
            wrapped = with_circuit_breaker(circuit_breaker_name)(wrapped)

        # Apply bulkhead (outermost)
        if bulkhead_name:
            wrapped = with_bulkhead(
                bulkhead_name,
                max_concurrent=bulkhead_max_concurrent,
            )(wrapped)

        return wrapped

    return decorator


# =============================================================================
# Synchronous Circuit Breaker (for connection pool management)
# =============================================================================


@dataclass
class SyncCircuitBreakerConfig:
    """Configuration for synchronous circuit breaker."""

    failure_threshold: int = 5
    success_threshold: int = 3
    recovery_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if self.success_threshold < 1:
            raise ValueError("success_threshold must be >= 1")
        if self.recovery_timeout_seconds <= 0:
            raise ValueError("recovery_timeout_seconds must be > 0")


class SyncCircuitBreaker:
    """
    Synchronous circuit breaker for connection pool management.

    Unlike the async CircuitBreaker, this uses simple methods instead
    of context managers, making it suitable for embedding in dataclasses
    and for use in connection pools where the check-before-execute pattern
    is preferred.

    Usage:
        breaker = SyncCircuitBreaker("database")

        if breaker.can_execute():
            try:
                result = do_operation()
                breaker.record_success()
            except Exception:
                breaker.record_failure()
                raise
    """

    def __init__(
        self,
        name: str = "",
        config: Optional[SyncCircuitBreakerConfig] = None,
    ):
        self.name = name
        self.config = config or SyncCircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0.0

    @property
    def state(self) -> CircuitState:
        """Get current circuit state with automatic timeout check."""
        if self._state == CircuitState.OPEN:
            if self._should_attempt_reset():
                return CircuitState.HALF_OPEN
        return self._state

    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        return self._failure_count

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self._last_failure_time == 0:
            return True
        elapsed = time.time() - self._last_failure_time
        return elapsed >= self.config.recovery_timeout_seconds

    def can_execute(self) -> bool:
        """Check if requests can be executed."""
        state = self.state  # Use property to trigger timeout check

        if state == CircuitState.CLOSED:
            return True

        if state == CircuitState.OPEN:
            return False

        # HALF_OPEN - allow test request
        return True

    def record_success(self) -> None:
        """Record a successful call."""
        self._success_count += 1

        if self._state == CircuitState.HALF_OPEN:
            if self._success_count >= self.config.success_threshold:
                self._reset()
        else:
            # Decay failure count on success
            self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self) -> None:
        """Record a failed call."""
        self._failure_count += 1
        self._success_count = 0
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
        elif self._failure_count >= self.config.failure_threshold:
            self._state = CircuitState.OPEN

    def _reset(self) -> None:
        """Reset circuit to closed state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0

    def get_metrics(self) -> Dict[str, Any]:
        """Get current circuit metrics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure": self._last_failure_time if self._last_failure_time > 0 else None,
        }
