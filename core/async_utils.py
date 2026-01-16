"""
BIBLOS v2 - Async Utilities

Provides advanced async patterns for efficient concurrent operations:
- Task groups with structured concurrency
- Batch processing with configurable concurrency
- Rate limiting and throttling
- Timeout with cleanup support
- Cancel scopes for graceful cancellation

All utilities integrate with OpenTelemetry for observability.
"""

from __future__ import annotations

import asyncio
import functools
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    TypeVar,
    ParamSpec,
)

from opentelemetry import trace

from core.errors import (
    BiblosTimeoutError,
    BiblosResourceError,
    ErrorContext,
)

T = TypeVar("T")
P = ParamSpec("P")

tracer = trace.get_tracer(__name__)


@dataclass
class TaskGroupConfig:
    """Configuration for async task group."""

    max_concurrency: int = 10
    timeout_seconds: Optional[float] = None
    fail_fast: bool = False
    collect_exceptions: bool = True


class AsyncTaskGroup:
    """
    Structured concurrency task group with cancellation support.

    Features:
    - Controlled concurrency with semaphore
    - Automatic cancellation on failure (optional)
    - Exception collection and reporting
    - OpenTelemetry tracing

    Usage:
        async with AsyncTaskGroup(max_concurrency=5) as group:
            group.create_task(async_operation1())
            group.create_task(async_operation2())
        results = group.results
    """

    def __init__(self, config: Optional[TaskGroupConfig] = None):
        self.config = config or TaskGroupConfig()
        self._tasks: List[asyncio.Task] = []
        self._results: List[Any] = []
        self._exceptions: List[Exception] = []
        self._semaphore = asyncio.Semaphore(self.config.max_concurrency)
        self._cancelled = False

    @property
    def results(self) -> List[Any]:
        """Results from completed tasks."""
        return self._results

    @property
    def exceptions(self) -> List[Exception]:
        """Exceptions from failed tasks."""
        return self._exceptions

    def create_task(
        self,
        coro: Awaitable[T],
        name: Optional[str] = None,
    ) -> asyncio.Task:
        """Create and track a new task."""
        async def wrapped() -> T:
            async with self._semaphore:
                if self._cancelled:
                    raise asyncio.CancelledError()
                return await coro

        task = asyncio.create_task(wrapped(), name=name)
        self._tasks.append(task)
        return task

    async def __aenter__(self) -> "AsyncTaskGroup":
        """Enter task group context."""
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> bool:
        """Exit task group, waiting for all tasks."""
        if exc_type and self.config.fail_fast:
            self._cancelled = True
            for task in self._tasks:
                if not task.done():
                    task.cancel()

        # Wait for all tasks with optional timeout
        if self._tasks:
            if self.config.timeout_seconds:
                done, pending = await asyncio.wait(
                    self._tasks,
                    timeout=self.config.timeout_seconds,
                )
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            else:
                done = set(await asyncio.gather(
                    *self._tasks,
                    return_exceptions=True,
                ))

            # Collect results and exceptions
            for task in self._tasks:
                try:
                    if task.done() and not task.cancelled():
                        result = task.result()
                        if isinstance(result, Exception):
                            self._exceptions.append(result)
                        else:
                            self._results.append(result)
                except Exception as e:
                    self._exceptions.append(e)

        return False  # Don't suppress exceptions


@dataclass
class BatcherConfig:
    """Configuration for async batcher."""

    batch_size: int = 100
    max_concurrency: int = 10
    timeout_seconds: Optional[float] = None
    collect_exceptions: bool = True


class AsyncBatcher(Generic[T]):
    """
    Async batch processor with controlled concurrency.

    Processes items in batches with configurable parallelism.

    Usage:
        async def process_batch(items: List[str]) -> List[Result]:
            ...

        batcher = AsyncBatcher(process_batch, batch_size=100)
        results = await batcher.process(all_items)
    """

    def __init__(
        self,
        processor: Callable[[List[T]], Awaitable[List[Any]]],
        config: Optional[BatcherConfig] = None,
    ):
        self.processor = processor
        self.config = config or BatcherConfig()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrency)

    async def process(
        self,
        items: List[T],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Any]:
        """Process all items in batches."""
        with tracer.start_as_current_span("batch.process") as span:
            span.set_attribute("batch.total_items", len(items))
            span.set_attribute("batch.batch_size", self.config.batch_size)

            results: List[Any] = []
            exceptions: List[Exception] = []

            # Create batches
            batches = [
                items[i:i + self.config.batch_size]
                for i in range(0, len(items), self.config.batch_size)
            ]

            span.set_attribute("batch.batch_count", len(batches))

            async def process_batch(batch_idx: int, batch: List[T]) -> List[Any]:
                async with self._semaphore:
                    with tracer.start_as_current_span(f"batch.chunk_{batch_idx}") as batch_span:
                        batch_span.set_attribute("batch.chunk_size", len(batch))
                        batch_results = await self.processor(batch)

                        if progress_callback:
                            processed = min((batch_idx + 1) * self.config.batch_size, len(items))
                            progress_callback(processed, len(items))

                        return batch_results

            # Process all batches
            tasks = [
                process_batch(idx, batch)
                for idx, batch in enumerate(batches)
            ]

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, Exception):
                    if self.config.collect_exceptions:
                        exceptions.append(result)
                    else:
                        raise result
                else:
                    results.extend(result)

            span.set_attribute("batch.results_count", len(results))
            span.set_attribute("batch.exceptions_count", len(exceptions))

            if exceptions and not self.config.collect_exceptions:
                raise exceptions[0]

            return results


@dataclass
class ThrottlerConfig:
    """Configuration for rate limiter."""

    requests_per_second: float = 10.0
    burst_size: int = 10
    timeout_seconds: float = 30.0


class AsyncThrottler:
    """
    Token bucket rate limiter for async operations.

    Features:
    - Configurable rate limiting
    - Burst allowance
    - Timeout support

    Usage:
        throttler = AsyncThrottler(requests_per_second=10)

        async with throttler:
            await rate_limited_operation()
    """

    def __init__(self, config: Optional[ThrottlerConfig] = None):
        self.config = config or ThrottlerConfig()
        self._tokens = float(self.config.burst_size)
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary."""
        start_time = time.monotonic()

        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_update
                self._last_update = now

                # Add tokens based on elapsed time
                self._tokens = min(
                    self.config.burst_size,
                    self._tokens + elapsed * self.config.requests_per_second,
                )

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return

            # Check timeout
            if time.monotonic() - start_time > self.config.timeout_seconds:
                raise BiblosTimeoutError(
                    message="Rate limiter timeout",
                    timeout_seconds=self.config.timeout_seconds,
                )

            # Wait before retry
            await asyncio.sleep(1.0 / self.config.requests_per_second)

    async def __aenter__(self) -> "AsyncThrottler":
        await self.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> bool:
        return False


async def gather_with_concurrency(
    *coros: Awaitable[T],
    max_concurrency: int = 10,
    return_exceptions: bool = False,
) -> List[T]:
    """
    Like asyncio.gather but with controlled concurrency.

    Usage:
        results = await gather_with_concurrency(
            fetch(url1),
            fetch(url2),
            fetch(url3),
            max_concurrency=5,
        )
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def bounded_coro(coro: Awaitable[T]) -> T:
        async with semaphore:
            return await coro

    return await asyncio.gather(
        *[bounded_coro(coro) for coro in coros],
        return_exceptions=return_exceptions,
    )


@asynccontextmanager
async def timeout_with_cleanup(
    seconds: float,
    cleanup_fn: Optional[Callable[[], Awaitable[None]]] = None,
) -> AsyncIterator[None]:
    """
    Timeout context manager with cleanup on timeout.

    Usage:
        async with timeout_with_cleanup(30.0, cleanup_database):
            await long_running_operation()
    """
    try:
        async with asyncio.timeout(seconds):
            yield
    except asyncio.TimeoutError:
        if cleanup_fn:
            try:
                await cleanup_fn()
            except Exception:
                pass  # Ignore cleanup errors
        raise BiblosTimeoutError(
            message=f"Operation timed out after {seconds} seconds",
            timeout_seconds=seconds,
        )


class CancelScope:
    """
    Cancellation scope for structured cancellation.

    Usage:
        scope = CancelScope()

        async def worker():
            while not scope.cancelled:
                await do_work()

        # Later
        scope.cancel()
    """

    def __init__(self):
        self._cancelled = False
        self._shield = False

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        """Request cancellation."""
        if not self._shield:
            self._cancelled = True

    @asynccontextmanager
    async def shield(self) -> AsyncIterator[None]:
        """Temporarily shield from cancellation."""
        self._shield = True
        try:
            yield
        finally:
            self._shield = False

    def check(self) -> None:
        """Check for cancellation and raise if cancelled."""
        if self._cancelled:
            raise asyncio.CancelledError()


@asynccontextmanager
async def cancel_scope() -> AsyncIterator[CancelScope]:
    """
    Create a cancel scope context manager.

    Usage:
        async with cancel_scope() as scope:
            task = asyncio.create_task(worker(scope))
            await asyncio.sleep(10)
            scope.cancel()
    """
    scope = CancelScope()
    try:
        yield scope
    finally:
        scope.cancel()


def async_cached(
    ttl_seconds: float = 300.0,
    max_size: int = 1000,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """
    Decorator for caching async function results.

    Usage:
        @async_cached(ttl_seconds=60)
        async def fetch_data(key: str) -> Data:
            ...
    """
    cache: Dict[str, tuple[T, float]] = {}

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Create cache key
            key = str((args, tuple(sorted(kwargs.items()))))
            now = time.monotonic()

            # Check cache
            if key in cache:
                result, timestamp = cache[key]
                if now - timestamp < ttl_seconds:
                    return result

            # Execute and cache
            result = await func(*args, **kwargs)

            # Evict if at max size
            if len(cache) >= max_size:
                oldest_key = min(cache.keys(), key=lambda k: cache[k][1])
                del cache[oldest_key]

            cache[key] = (result, now)
            return result

        return wrapper

    return decorator


# =============================================================================
# STREAMING & MEMORY-EFFICIENT ITERATION
# =============================================================================


async def async_chunked_iter(
    items: List[T],
    chunk_size: int = 100,
) -> AsyncIterator[List[T]]:
    """
    Async iterator that yields items in chunks.

    Memory-efficient for processing large lists.

    Usage:
        async for chunk in async_chunked_iter(items, chunk_size=100):
            await process_batch(chunk)
    """
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]
        # Allow other tasks to run between chunks
        await asyncio.sleep(0)


async def async_buffered_iter(
    source: AsyncIterator[T],
    buffer_size: int = 10,
) -> AsyncIterator[T]:
    """
    Async iterator with buffered prefetching.

    Prefetches items to reduce latency in processing pipelines.

    Usage:
        async for item in async_buffered_iter(slow_source(), buffer_size=5):
            process(item)
    """
    buffer: asyncio.Queue[T | None] = asyncio.Queue(maxsize=buffer_size)
    done = asyncio.Event()

    async def producer() -> None:
        try:
            async for item in source:
                await buffer.put(item)
            await buffer.put(None)  # Sentinel
        finally:
            done.set()

    producer_task = asyncio.create_task(producer())

    try:
        while True:
            item = await buffer.get()
            if item is None:
                break
            yield item
    finally:
        producer_task.cancel()
        try:
            await producer_task
        except asyncio.CancelledError:
            pass


# =============================================================================
# LAZY INITIALIZATION
# =============================================================================


class LazyAsync(Generic[T]):
    """
    Lazy async initialization wrapper.

    Defers expensive initialization until first access.

    Usage:
        db = LazyAsync(create_database_connection)

        # Later, when needed:
        connection = await db.get()
    """

    def __init__(
        self,
        factory: Callable[[], Awaitable[T]],
        cleanup: Optional[Callable[[T], Awaitable[None]]] = None,
    ):
        self._factory = factory
        self._cleanup = cleanup
        self._value: Optional[T] = None
        self._lock = asyncio.Lock()
        self._initialized = False

    async def get(self) -> T:
        """Get or create the value."""
        if self._initialized:
            return self._value  # type: ignore

        async with self._lock:
            # Double-check after acquiring lock
            if self._initialized:
                return self._value  # type: ignore

            self._value = await self._factory()
            self._initialized = True
            return self._value

    async def reset(self) -> None:
        """Reset and optionally cleanup the value."""
        async with self._lock:
            if self._initialized and self._cleanup and self._value is not None:
                await self._cleanup(self._value)
            self._value = None
            self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if value has been initialized."""
        return self._initialized


# =============================================================================
# PRIORITY TASK EXECUTION
# =============================================================================


@dataclass(order=True)
class PrioritizedTask:
    """Task with priority for queue ordering."""

    priority: int
    task_id: str = field(compare=False)
    coro: Awaitable[Any] = field(compare=False)


class PriorityTaskQueue:
    """
    Priority-based async task queue.

    Higher priority (lower number) tasks execute first.

    Usage:
        queue = PriorityTaskQueue(max_workers=5)
        await queue.start()

        await queue.submit(coro1, priority=1)  # High priority
        await queue.submit(coro2, priority=10)  # Low priority

        await queue.stop()
    """

    def __init__(self, max_workers: int = 5):
        self.max_workers = max_workers
        self._queue: asyncio.PriorityQueue[PrioritizedTask] = asyncio.PriorityQueue()
        self._workers: List[asyncio.Task] = []
        self._running = False
        self._task_counter = 0

    async def start(self) -> None:
        """Start worker tasks."""
        if self._running:
            return

        self._running = True
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(i))
            self._workers.append(worker)

    async def stop(self, timeout: float = 10.0) -> None:
        """Stop all workers gracefully."""
        self._running = False

        # Cancel all workers
        for worker in self._workers:
            worker.cancel()

        # Wait for workers to finish
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    async def submit(
        self,
        coro: Awaitable[T],
        priority: int = 5,
    ) -> None:
        """Submit a task with priority (lower = higher priority)."""
        self._task_counter += 1
        task = PrioritizedTask(
            priority=priority,
            task_id=f"task_{self._task_counter}",
            coro=coro,
        )
        await self._queue.put(task)

    async def _worker(self, worker_id: int) -> None:
        """Worker coroutine that processes tasks from queue."""
        while self._running:
            try:
                task = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0,
                )
                try:
                    await task.coro
                except Exception as e:
                    # Log error but continue processing
                    pass
                finally:
                    self._queue.task_done()
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break


# =============================================================================
# DEBOUNCE & COALESCE
# =============================================================================


def debounce(
    wait_seconds: float,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[Optional[T]]]]:
    """
    Debounce async function calls.

    Only executes after wait_seconds of no calls.

    Usage:
        @debounce(0.5)
        async def save_draft(content: str):
            await save_to_db(content)
    """
    def decorator(
        func: Callable[P, Awaitable[T]],
    ) -> Callable[P, Awaitable[Optional[T]]]:
        pending_task: Optional[asyncio.Task] = None
        lock = asyncio.Lock()

        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> Optional[T]:
            nonlocal pending_task

            async with lock:
                if pending_task and not pending_task.done():
                    pending_task.cancel()
                    try:
                        await pending_task
                    except asyncio.CancelledError:
                        pass

                async def delayed_call() -> T:
                    await asyncio.sleep(wait_seconds)
                    return await func(*args, **kwargs)

                pending_task = asyncio.create_task(delayed_call())

            try:
                return await pending_task
            except asyncio.CancelledError:
                return None

        return wrapper

    return decorator


def coalesce(
    window_seconds: float,
    key_fn: Optional[Callable[..., str]] = None,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """
    Coalesce multiple calls within a time window.

    Returns same result for identical calls within window.

    Usage:
        @coalesce(1.0, key_fn=lambda user_id: str(user_id))
        async def fetch_user(user_id: int) -> User:
            return await db.get_user(user_id)
    """
    pending: Dict[str, asyncio.Task[T]] = {}

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Generate key
            if key_fn:
                key = key_fn(*args, **kwargs)
            else:
                key = str((args, tuple(sorted(kwargs.items()))))

            # Check if there's a pending call for this key
            if key in pending and not pending[key].done():
                return await pending[key]

            # Create new task
            async def execute() -> T:
                try:
                    return await func(*args, **kwargs)
                finally:
                    # Clean up after window expires
                    await asyncio.sleep(window_seconds)
                    pending.pop(key, None)

            task = asyncio.create_task(execute())
            pending[key] = task
            return await task

        return wrapper

    return decorator


# =============================================================================
# ASYNC CONTEXT STACK
# =============================================================================


class AsyncContextStack:
    """
    Manage multiple async context managers as a stack.

    Useful for managing multiple resources that need coordinated cleanup.

    Usage:
        async with AsyncContextStack() as stack:
            db = await stack.enter(database_connection())
            cache = await stack.enter(cache_connection())
            # Both cleaned up automatically
    """

    def __init__(self) -> None:
        self._contexts: List[Any] = []
        self._values: List[Any] = []

    async def enter(self, context_manager: Any) -> Any:
        """Enter a context manager and track it."""
        value = await context_manager.__aenter__()
        self._contexts.append(context_manager)
        self._values.append(value)
        return value

    async def __aenter__(self) -> "AsyncContextStack":
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> bool:
        """Exit all context managers in reverse order."""
        exceptions: List[Exception] = []

        for context in reversed(self._contexts):
            try:
                await context.__aexit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                exceptions.append(e)

        self._contexts.clear()
        self._values.clear()

        if exceptions:
            raise exceptions[0]

        return False
