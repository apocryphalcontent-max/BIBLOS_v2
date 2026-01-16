"""
BIBLOS v2 - Mediator Pattern Implementation

The mediator acts as the central nervous system of the application,
routing commands and queries through a pipeline of behaviors that
provide cross-cutting concerns like validation, logging, and transactions.

Architecture:
    - Commands: Write operations that change state (emits domain events)
    - Queries: Read operations that return data (no side effects)
    - Pipeline Behaviors: Middleware that wraps handler execution
    - Notifications: Domain events broadcast to multiple handlers

The mediator decouples senders from receivers, enabling:
    - Clean separation of concerns
    - Composable middleware pipeline
    - Easy testing and extension
    - Audit logging and observability

Usage:
    from domain.mediator import Mediator, Command, Query

    # Define a command
    @dataclass
    class ProcessVerseCommand(Command[ProcessVerseResult]):
        verse_id: str
        pipeline_id: str

    # Define handler
    class ProcessVerseHandler(ICommandHandler[ProcessVerseCommand, ProcessVerseResult]):
        async def handle(self, command: ProcessVerseCommand) -> ProcessVerseResult:
            ...

    # Execute through mediator
    result = await mediator.send(ProcessVerseCommand(verse_id="GEN.1.1"))
"""
from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Type,
    TypeVar,
)
from uuid import UUID, uuid4

from domain.entities import DomainEvent


logger = logging.getLogger("biblos.mediator")


# =============================================================================
# BASE TYPES FOR CQRS
# =============================================================================


TResult = TypeVar("TResult")
TRequest = TypeVar("TRequest", bound="IRequest")
TCommand = TypeVar("TCommand", bound="Command")
TQuery = TypeVar("TQuery", bound="Query")
TNotification = TypeVar("TNotification", bound="INotification")


class IRequest(ABC, Generic[TResult]):
    """
    Base interface for all requests (commands and queries).

    Requests are the messages that flow through the mediator.
    They are processed by handlers and may pass through pipeline behaviors.
    """

    @property
    def request_id(self) -> UUID:
        """Unique identifier for this request."""
        return getattr(self, "_request_id", uuid4())

    @property
    def timestamp(self) -> datetime:
        """When this request was created."""
        return getattr(self, "_timestamp", datetime.now(timezone.utc))


class Command(IRequest[TResult], Generic[TResult]):
    """
    Base class for commands.

    Commands represent intent to change system state. They:
    - Have side effects (modify data)
    - Should be validated before execution
    - Emit domain events on success
    - Are typically processed by a single handler

    Example:
        @dataclass
        class CreateVerseCommand(Command[Verse]):
            reference: str
            text_original: str
            text_english: str

    Note: Subclasses should use @dataclass decorator. The request_id
    and timestamp are generated automatically via properties.
    """

    @property
    def request_id(self) -> UUID:
        if not hasattr(self, "_request_id"):
            object.__setattr__(self, "_request_id", uuid4())
        return self._request_id  # type: ignore

    @property
    def timestamp(self) -> datetime:
        if not hasattr(self, "_timestamp"):
            object.__setattr__(self, "_timestamp", datetime.now(timezone.utc))
        return self._timestamp  # type: ignore

    @property
    def correlation_id(self) -> Optional[str]:
        return getattr(self, "_correlation_id", None)

    @property
    def causation_id(self) -> Optional[str]:
        return getattr(self, "_causation_id", None)


class Query(IRequest[TResult], Generic[TResult]):
    """
    Base class for queries.

    Queries represent requests for data. They:
    - Have no side effects
    - Are idempotent (same query = same result)
    - Can be cached
    - Are typically processed by a single handler

    Example:
        @dataclass
        class GetVerseQuery(Query[Optional[Verse]]):
            verse_id: str

    Note: Subclasses should use @dataclass decorator. The request_id
    and timestamp are generated automatically via properties.
    """

    @property
    def request_id(self) -> UUID:
        if not hasattr(self, "_request_id"):
            object.__setattr__(self, "_request_id", uuid4())
        return self._request_id  # type: ignore

    @property
    def timestamp(self) -> datetime:
        if not hasattr(self, "_timestamp"):
            object.__setattr__(self, "_timestamp", datetime.now(timezone.utc))
        return self._timestamp  # type: ignore


class INotification(ABC):
    """
    Base interface for notifications.

    Notifications are broadcast to multiple handlers and represent
    events that have occurred. Unlike commands/queries, multiple
    handlers can process the same notification.
    """

    @property
    @abstractmethod
    def notification_id(self) -> UUID:
        """Unique identifier for this notification."""
        pass


@dataclass
class DomainEventNotification(INotification):
    """
    Notification wrapper for domain events.

    Allows domain events to flow through the mediator's notification
    system, enabling loose coupling between event producers and consumers.
    """
    event: DomainEvent
    _notification_id: UUID = field(default_factory=uuid4)

    @property
    def notification_id(self) -> UUID:
        return self._notification_id

    @property
    def event_type(self) -> str:
        return self.event.event_type


# =============================================================================
# HANDLER INTERFACES
# =============================================================================


class IRequestHandler(ABC, Generic[TRequest, TResult]):
    """
    Base interface for request handlers.

    Handlers contain the business logic for processing requests.
    Each request type should have exactly one handler.
    """

    @abstractmethod
    async def handle(self, request: TRequest) -> TResult:
        """Handle the request and return a result."""
        pass


class ICommandHandler(IRequestHandler[TCommand, TResult], Generic[TCommand, TResult]):
    """Handler for commands."""
    pass


class IQueryHandler(IRequestHandler[TQuery, TResult], Generic[TQuery, TResult]):
    """Handler for queries."""
    pass


class INotificationHandler(ABC, Generic[TNotification]):
    """
    Handler for notifications.

    Multiple handlers can process the same notification type.
    """

    @abstractmethod
    async def handle(self, notification: TNotification) -> None:
        """Handle the notification."""
        pass


# =============================================================================
# PIPELINE BEHAVIORS
# =============================================================================


class IPipelineBehavior(ABC, Generic[TRequest, TResult]):
    """
    Pipeline behavior for cross-cutting concerns.

    Behaviors wrap around handler execution, forming a middleware pipeline.
    They can perform actions before and after handler execution.

    The pipeline looks like:
        Behavior1 -> Behavior2 -> Behavior3 -> Handler -> Behavior3 -> Behavior2 -> Behavior1

    Usage:
        class LoggingBehavior(IPipelineBehavior[TRequest, TResult]):
            async def handle(
                self,
                request: TRequest,
                next: RequestHandlerDelegate[TResult]
            ) -> TResult:
                logger.info(f"Handling {request}")
                result = await next()
                logger.info(f"Handled {request}")
                return result
    """

    @abstractmethod
    async def handle(
        self,
        request: TRequest,
        next: "RequestHandlerDelegate[TResult]"
    ) -> TResult:
        """
        Handle the request in the pipeline.

        Args:
            request: The request being processed
            next: Delegate to call the next behavior or handler

        Returns:
            The result from the handler or modified result
        """
        pass


# Type for the next delegate in the pipeline
# Using Awaitable for broader compatibility with coroutines and futures
RequestHandlerDelegate = Callable[[], Awaitable[TResult]]


# =============================================================================
# CONCRETE PIPELINE BEHAVIORS
# =============================================================================


class LoggingBehavior(IPipelineBehavior[TRequest, TResult], Generic[TRequest, TResult]):
    """
    Pipeline behavior that logs request handling.

    Logs the request type, duration, and any errors.
    """

    def __init__(self, log_level: int = logging.INFO) -> None:
        self._log_level = log_level

    async def handle(
        self,
        request: TRequest,
        next: RequestHandlerDelegate[TResult]
    ) -> TResult:
        request_type = type(request).__name__
        request_id = getattr(request, "request_id", "unknown")

        logger.log(
            self._log_level,
            f"[{request_id}] Starting {request_type}"
        )
        start_time = time.perf_counter()

        try:
            result = await next()
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.log(
                self._log_level,
                f"[{request_id}] Completed {request_type} in {duration_ms:.2f}ms"
            )
            return result
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"[{request_id}] Failed {request_type} after {duration_ms:.2f}ms: {e}"
            )
            raise


class ValidationBehavior(IPipelineBehavior[TRequest, TResult], Generic[TRequest, TResult]):
    """
    Pipeline behavior that validates requests before handling.

    Calls validate() method on requests that have it.
    """

    async def handle(
        self,
        request: TRequest,
        next: RequestHandlerDelegate[TResult]
    ) -> TResult:
        # Check if request has a validate method
        validate_method = getattr(request, "validate", None)
        if callable(validate_method):
            validation_result = validate_method()
            if validation_result:
                # Ensure we have a list of strings
                if isinstance(validation_result, list):
                    raise ValidationError(validation_result)
                elif isinstance(validation_result, str):
                    raise ValidationError([validation_result])
                else:
                    raise ValidationError([str(validation_result)])

        return await next()


class ValidationError(Exception):
    """Raised when request validation fails."""

    def __init__(self, errors: List[str]) -> None:
        self.errors = errors
        super().__init__(f"Validation failed: {', '.join(errors)}")


class TransactionBehavior(IPipelineBehavior[TCommand, TResult], Generic[TCommand, TResult]):
    """
    Pipeline behavior that wraps commands in a database transaction.

    Commits on success, rolls back on failure.
    """

    def __init__(self, unit_of_work_factory: Callable[[], Any]) -> None:
        self._uow_factory = unit_of_work_factory

    async def handle(
        self,
        request: TCommand,  # noqa: ARG002 - Transaction behavior doesn't inspect request
        next: RequestHandlerDelegate[TResult]
    ) -> TResult:
        del request  # Unused but required by interface
        uow = self._uow_factory()
        async with uow:
            try:
                result = await next()
                await uow.commit()
                return result
            except Exception:
                await uow.rollback()
                raise


class PerformanceMonitoringBehavior(
    IPipelineBehavior[TRequest, TResult], Generic[TRequest, TResult]
):
    """
    Pipeline behavior that monitors performance metrics.

    Records execution time, memory usage, and other metrics.
    """

    def __init__(
        self,
        metrics_collector: Optional[Any] = None,
        slow_threshold_ms: float = 1000.0
    ) -> None:
        self._metrics = metrics_collector
        self._slow_threshold_ms = slow_threshold_ms

    async def handle(
        self,
        request: TRequest,
        next: RequestHandlerDelegate[TResult]
    ) -> TResult:
        request_type = type(request).__name__
        start_time = time.perf_counter()

        result = await next()

        duration_ms = (time.perf_counter() - start_time) * 1000

        if duration_ms > self._slow_threshold_ms:
            logger.warning(
                f"Slow request: {request_type} took {duration_ms:.2f}ms"
            )

        if self._metrics:
            self._metrics.record_duration(request_type, duration_ms)

        return result


class RetryBehavior(IPipelineBehavior[TRequest, TResult], Generic[TRequest, TResult]):
    """
    Pipeline behavior that retries failed requests.

    Implements exponential backoff with configurable retry count.
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay_seconds: float = 0.1,
        exponential_base: float = 2.0,
        retryable_exceptions: Optional[tuple] = None
    ) -> None:
        self._max_retries = max_retries
        self._base_delay = base_delay_seconds
        self._exponential_base = exponential_base
        self._retryable = retryable_exceptions or (Exception,)

    async def handle(
        self,
        request: TRequest,
        next: RequestHandlerDelegate[TResult]
    ) -> TResult:
        last_exception: Optional[Exception] = None

        for attempt in range(self._max_retries + 1):
            try:
                return await next()
            except self._retryable as e:
                last_exception = e
                if attempt < self._max_retries:
                    delay = self._base_delay * (self._exponential_base ** attempt)
                    logger.warning(
                        f"Retry {attempt + 1}/{self._max_retries} "
                        f"for {type(request).__name__} after {delay:.2f}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    raise

        # Should never reach here, but satisfy type checker
        raise last_exception  # type: ignore


class CachingBehavior(IPipelineBehavior[TQuery, TResult], Generic[TQuery, TResult]):
    """
    Pipeline behavior that caches query results.

    Only applies to Query types, not Commands.
    """

    def __init__(
        self,
        cache: Optional[Dict[str, Any]] = None,
        ttl_seconds: float = 300.0
    ) -> None:
        self._cache: Dict[str, tuple[Any, float]] = cache or {}
        self._ttl = ttl_seconds

    def _make_cache_key(self, request: TQuery) -> str:
        """Generate cache key from request."""
        request_dict = {
            k: v for k, v in vars(request).items()
            if not k.startswith("_")
        }
        return f"{type(request).__name__}:{hash(frozenset(request_dict.items()))}"

    async def handle(
        self,
        request: TQuery,
        next: RequestHandlerDelegate[TResult]
    ) -> TResult:
        cache_key = self._make_cache_key(request)
        current_time = time.time()

        # Check cache
        if cache_key in self._cache:
            cached_value, cached_time = self._cache[cache_key]
            if current_time - cached_time < self._ttl:
                logger.debug(f"Cache hit for {cache_key}")
                return cached_value

        # Execute and cache
        result = await next()
        self._cache[cache_key] = (result, current_time)
        return result


# =============================================================================
# MEDIATOR IMPLEMENTATION
# =============================================================================


class Mediator:
    """
    Central mediator for routing requests to handlers.

    The mediator is the nervous system of the application, coordinating
    all communication between components without them knowing about each other.

    Usage:
        mediator = Mediator()

        # Register handler
        mediator.register_handler(ProcessVerseCommand, ProcessVerseHandler())

        # Add pipeline behaviors (in order of execution)
        mediator.add_behavior(LoggingBehavior())
        mediator.add_behavior(ValidationBehavior())
        mediator.add_behavior(TransactionBehavior(uow_factory))

        # Send command
        result = await mediator.send(ProcessVerseCommand(verse_id="GEN.1.1"))
    """

    def __init__(self) -> None:
        self._handlers: Dict[Type, IRequestHandler] = {}
        self._notification_handlers: Dict[Type, List[INotificationHandler]] = {}
        self._behaviors: List[IPipelineBehavior] = []
        self._command_behaviors: List[IPipelineBehavior] = []
        self._query_behaviors: List[IPipelineBehavior] = []

    def register_handler(
        self,
        request_type: Type[TRequest],
        handler: IRequestHandler[TRequest, Any]
    ) -> None:
        """Register a handler for a request type."""
        self._handlers[request_type] = handler
        logger.debug(f"Registered handler for {request_type.__name__}")

    def register_notification_handler(
        self,
        notification_type: Type[TNotification],
        handler: INotificationHandler[TNotification]
    ) -> None:
        """Register a handler for a notification type."""
        if notification_type not in self._notification_handlers:
            self._notification_handlers[notification_type] = []
        self._notification_handlers[notification_type].append(handler)
        logger.debug(f"Registered notification handler for {notification_type.__name__}")

    def add_behavior(self, behavior: IPipelineBehavior) -> None:
        """Add a pipeline behavior that applies to all requests."""
        self._behaviors.append(behavior)

    def add_command_behavior(self, behavior: IPipelineBehavior) -> None:
        """Add a pipeline behavior that applies only to commands."""
        self._command_behaviors.append(behavior)

    def add_query_behavior(self, behavior: IPipelineBehavior) -> None:
        """Add a pipeline behavior that applies only to queries."""
        self._query_behaviors.append(behavior)

    async def send(self, request: IRequest[TResult]) -> TResult:
        """
        Send a request through the mediator.

        The request passes through the pipeline behaviors before
        reaching the handler.

        Args:
            request: Command or Query to process

        Returns:
            Result from the handler

        Raises:
            ValueError: If no handler is registered for the request type
        """
        request_type = type(request)
        handler = self._handlers.get(request_type)

        if handler is None:
            raise ValueError(f"No handler registered for {request_type.__name__}")

        # Build behavior pipeline
        behaviors = list(self._behaviors)
        if isinstance(request, Command):
            behaviors.extend(self._command_behaviors)
        elif isinstance(request, Query):
            behaviors.extend(self._query_behaviors)

        # Build the pipeline chain
        async def final_handler() -> TResult:
            return await handler.handle(request)

        pipeline = self._build_pipeline(request, behaviors, final_handler)
        return await pipeline()

    def _build_pipeline(
        self,
        request: Any,
        behaviors: List[IPipelineBehavior],
        handler: Callable[[], Awaitable[TResult]]
    ) -> Callable[[], Awaitable[TResult]]:
        """Build the pipeline of behaviors wrapping the handler."""
        current: Callable[[], Awaitable[Any]] = handler

        # Build from inside out (last behavior wraps handler first)
        for behavior in reversed(behaviors):
            # Capture behavior and next in closure properly
            captured_behavior = behavior
            captured_next = current
            captured_request = request

            async def make_step(
                b: IPipelineBehavior,
                n: Callable[[], Awaitable[Any]],
                r: Any
            ) -> Any:
                return await b.handle(r, n)

            # Create closure that captures current values
            current = (
                lambda b=captured_behavior, n=captured_next, r=captured_request:
                make_step(b, n, r)
            )  # type: ignore

        return current  # type: ignore

    async def publish(self, notification: INotification) -> None:
        """
        Publish a notification to all registered handlers.

        All handlers are executed concurrently.

        Args:
            notification: The notification to publish
        """
        notification_type = type(notification)
        handlers = self._notification_handlers.get(notification_type, [])

        if not handlers:
            logger.debug(f"No handlers for notification {notification_type.__name__}")
            return

        # Execute all handlers concurrently
        await asyncio.gather(
            *(handler.handle(notification) for handler in handlers),
            return_exceptions=True
        )

    async def publish_domain_event(self, event: DomainEvent) -> None:
        """
        Publish a domain event as a notification.

        Wraps the domain event in a notification and publishes it.
        """
        notification = DomainEventNotification(event=event)
        await self.publish(notification)

    async def publish_domain_events(self, events: List[DomainEvent]) -> None:
        """Publish multiple domain events."""
        for event in events:
            await self.publish_domain_event(event)


# =============================================================================
# MEDIATOR BUILDER
# =============================================================================


class MediatorBuilder:
    """
    Builder for constructing a configured Mediator.

    Usage:
        mediator = (MediatorBuilder()
            .with_logging()
            .with_validation()
            .with_transaction(uow_factory)
            .with_performance_monitoring()
            .register_handler(ProcessVerseCommand, ProcessVerseHandler())
            .build())
    """

    def __init__(self) -> None:
        self._mediator = Mediator()

    def with_logging(self, log_level: int = logging.INFO) -> "MediatorBuilder":
        """Add logging behavior."""
        self._mediator.add_behavior(LoggingBehavior(log_level))
        return self

    def with_validation(self) -> "MediatorBuilder":
        """Add validation behavior."""
        self._mediator.add_behavior(ValidationBehavior())
        return self

    def with_transaction(
        self, unit_of_work_factory: Callable[[], Any]
    ) -> "MediatorBuilder":
        """Add transaction behavior for commands."""
        self._mediator.add_command_behavior(
            TransactionBehavior(unit_of_work_factory)
        )
        return self

    def with_performance_monitoring(
        self,
        metrics_collector: Optional[Any] = None,
        slow_threshold_ms: float = 1000.0
    ) -> "MediatorBuilder":
        """Add performance monitoring behavior."""
        self._mediator.add_behavior(
            PerformanceMonitoringBehavior(metrics_collector, slow_threshold_ms)
        )
        return self

    def with_retry(
        self,
        max_retries: int = 3,
        base_delay_seconds: float = 0.1,
        retryable_exceptions: Optional[tuple] = None
    ) -> "MediatorBuilder":
        """Add retry behavior."""
        self._mediator.add_behavior(
            RetryBehavior(
                max_retries=max_retries,
                base_delay_seconds=base_delay_seconds,
                retryable_exceptions=retryable_exceptions
            )
        )
        return self

    def with_query_caching(
        self,
        ttl_seconds: float = 300.0
    ) -> "MediatorBuilder":
        """Add caching behavior for queries."""
        self._mediator.add_query_behavior(CachingBehavior(ttl_seconds=ttl_seconds))
        return self

    def with_behavior(self, behavior: IPipelineBehavior) -> "MediatorBuilder":
        """Add a custom behavior."""
        self._mediator.add_behavior(behavior)
        return self

    def register_handler(
        self,
        request_type: Type[TRequest],
        handler: IRequestHandler[TRequest, Any]
    ) -> "MediatorBuilder":
        """Register a request handler."""
        self._mediator.register_handler(request_type, handler)
        return self

    def register_notification_handler(
        self,
        notification_type: Type[TNotification],
        handler: INotificationHandler[TNotification]
    ) -> "MediatorBuilder":
        """Register a notification handler."""
        self._mediator.register_notification_handler(notification_type, handler)
        return self

    def build(self) -> Mediator:
        """Build and return the configured mediator."""
        return self._mediator


# =============================================================================
# COMMON COMMAND/QUERY DEFINITIONS
# =============================================================================


@dataclass
class ProcessVerseCommand(Command[Dict[str, Any]]):
    """Command to process a verse through the extraction pipeline."""
    verse_id: str
    pipeline_id: str = field(default_factory=lambda: str(uuid4()))
    force_reprocess: bool = False


@dataclass
class DiscoverCrossReferencesCommand(Command[List[Dict[str, Any]]]):
    """Command to discover cross-references for a verse."""
    verse_id: str
    top_k: int = 10
    min_confidence: float = 0.5
    connection_types: Optional[List[str]] = None


@dataclass
class VerifyCrossReferenceCommand(Command[bool]):
    """Command to verify a cross-reference."""
    crossref_id: str
    verification_type: str  # "human", "patristic", "ml"
    verifier: str
    notes: Optional[str] = None


@dataclass
class CertifyGoldenRecordCommand(Command[Dict[str, Any]]):
    """Command to certify a golden record."""
    verse_id: str
    force_certification: bool = False


@dataclass
class GetVerseQuery(Query[Optional[Dict[str, Any]]]):
    """Query to get a verse by ID."""
    verse_id: str
    include_extractions: bool = False
    include_cross_references: bool = False


@dataclass
class GetCrossReferencesQuery(Query[List[Dict[str, Any]]]):
    """Query to get cross-references for a verse."""
    verse_id: str
    direction: str = "both"  # "outgoing", "incoming", "both"
    connection_types: Optional[List[str]] = None
    min_confidence: float = 0.0
    include_unverified: bool = True


@dataclass
class SearchVersesQuery(Query[List[Dict[str, Any]]]):
    """Query to search verses by text."""
    search_text: str
    book_code: Optional[str] = None
    testament: Optional[str] = None
    limit: int = 10
    offset: int = 0


@dataclass
class GetGoldenRecordQuery(Query[Optional[Dict[str, Any]]]):
    """Query to get a golden record."""
    verse_id: str
    include_all_data: bool = True


@dataclass
class GetPipelineStatusQuery(Query[Dict[str, Any]]):
    """Query to get pipeline processing status."""
    pipeline_id: Optional[str] = None
    verse_id: Optional[str] = None
