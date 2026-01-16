"""
BIBLOS v2 - Event Store Implementation

The event store is the memory of the system - an append-only ledger that captures
every significant occurrence in the domain. Like the recording angel, it faithfully
preserves the complete history of all changes.

Features:
    - Optimistic concurrency control via aggregate versioning
    - Event schema versioning with automatic upcasting
    - Aggregate snapshots for fast rehydration
    - Multiple stream types (aggregate, category, all)
    - Catch-up and persistent subscriptions
    - Correlation/causation tracking for distributed tracing
    - Dead letter queue for failed event processing

Event Sourcing Principles:
    - Events are immutable facts about what happened
    - Current state is derived by replaying events
    - The event log is the single source of truth
    - Schema changes are handled through upcasting, never mutation

Architecture:
    - IEventStore: Core interface for event persistence
    - ISnapshotStore: Interface for aggregate snapshots
    - IEventUpcaster: Interface for schema migration
    - EventStore: PostgreSQL implementation
    - SnapshotStore: Snapshot management
    - EventStream: Cursor-based event streaming

Usage:
    # Store events
    await event_store.append(event, expected_version=0)

    # Rehydrate aggregate with snapshots
    snapshot = await snapshot_store.get_latest_snapshot("verse-GEN.1.1")
    events = await event_store.get_events_by_aggregate(
        "verse-GEN.1.1",
        from_version=snapshot.version + 1 if snapshot else 0
    )

    # Subscribe to events
    subscription = event_store.subscribe_persistent(
        "projection-golden-records",
        lambda e: process_event(e)
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)
from uuid import UUID, uuid4

import asyncpg

from db.events import BaseEvent, deserialize_event, EventType

logger = logging.getLogger(__name__)

T = TypeVar("T")
TEvent = TypeVar("TEvent", bound=BaseEvent)


# =============================================================================
# EXCEPTIONS
# =============================================================================


class EventStoreError(Exception):
    """Base exception for event store errors."""
    pass


class ConcurrencyError(EventStoreError):
    """Raised when optimistic concurrency check fails."""

    def __init__(self, aggregate_id: str, expected: int, actual: int):
        self.aggregate_id = aggregate_id
        self.expected_version = expected
        self.actual_version = actual
        super().__init__(
            f"Concurrency conflict for aggregate {aggregate_id}: "
            f"expected version {expected}, found {actual}"
        )


class EventNotFoundError(EventStoreError):
    """Raised when an expected event is not found."""
    pass


class SnapshotError(EventStoreError):
    """Raised when snapshot operations fail."""
    pass


class UpcastingError(EventStoreError):
    """Raised when event upcasting fails."""
    pass


class SubscriptionError(EventStoreError):
    """Raised when subscription operations fail."""
    pass


# =============================================================================
# VALUE OBJECTS
# =============================================================================


class StreamCategory(Enum):
    """Categories of event streams."""
    AGGREGATE = "aggregate"      # Events for a single aggregate
    CATEGORY = "category"        # Events for a category of aggregates (e.g., all verses)
    ALL = "all"                  # All events in the store
    CORRELATION = "correlation"  # Events sharing a correlation ID


@dataclass(frozen=True, slots=True)
class StreamPosition:
    """
    Position in an event stream.

    The position is the address of an event in the stream - like a page number
    in the book of history.
    """
    value: int

    @classmethod
    def start(cls) -> "StreamPosition":
        """Start of stream."""
        return cls(0)

    @classmethod
    def end(cls) -> "StreamPosition":
        """End of stream (for subscriptions that start from now)."""
        return cls(-1)

    def __add__(self, other: int) -> "StreamPosition":
        return StreamPosition(self.value + other)


@dataclass(frozen=True, slots=True)
class EventMetadata:
    """
    Metadata attached to every event.

    This is the event's passport - containing its identity, lineage,
    and processing information.
    """
    event_id: UUID
    event_type: str
    aggregate_id: str
    aggregate_type: str
    aggregate_version: int
    timestamp: datetime
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    schema_version: int = 1
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    stream_position: Optional[int] = None
    custom: Dict[str, Any] = field(default_factory=dict)

    def with_position(self, position: int) -> "EventMetadata":
        """Create new metadata with stream position set."""
        return EventMetadata(
            event_id=self.event_id,
            event_type=self.event_type,
            aggregate_id=self.aggregate_id,
            aggregate_type=self.aggregate_type,
            aggregate_version=self.aggregate_version,
            timestamp=self.timestamp,
            correlation_id=self.correlation_id,
            causation_id=self.causation_id,
            schema_version=self.schema_version,
            user_id=self.user_id,
            tenant_id=self.tenant_id,
            stream_position=position,
            custom=self.custom,
        )


@dataclass(frozen=True, slots=True)
class StoredEvent:
    """
    An event as stored in the event store.

    Contains both the event payload and its metadata.
    """
    metadata: EventMetadata
    payload: Dict[str, Any]
    stored_at: datetime

    def to_domain_event(self) -> BaseEvent:
        """Convert to domain event."""
        event_data = {
            "event_id": self.metadata.event_id,
            "event_type": self.metadata.event_type,
            "aggregate_id": self.metadata.aggregate_id,
            "aggregate_version": self.metadata.aggregate_version,
            "timestamp": self.metadata.timestamp,
            "correlation_id": self.metadata.correlation_id,
            "causation_id": self.metadata.causation_id,
            "metadata": self.metadata.custom,
            **self.payload,
        }
        return deserialize_event(event_data)


@dataclass(frozen=True, slots=True)
class Snapshot(Generic[T]):
    """
    A snapshot of aggregate state.

    Snapshots are checkpoints in the aggregate's history - milestones that
    allow us to skip replaying ancient events during rehydration.
    """
    aggregate_id: str
    aggregate_type: str
    version: int
    state: T
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SubscriptionCheckpoint:
    """
    Checkpoint for a subscription's progress.

    This is the bookmark in the event stream - telling us where
    a subscription left off reading.
    """
    subscription_id: str
    stream_name: str
    position: StreamPosition
    last_processed_at: datetime


# =============================================================================
# INTERFACES
# =============================================================================


class IEventUpcaster(ABC):
    """
    Interface for event schema migration (upcasting).

    When event schemas evolve, upcasters transform old events to the new format.
    This ensures backward compatibility without mutating historical data.

    The upcasting chain transforms events through each version:
    v1 -> v2 -> v3 -> current

    Usage:
        class VerseCreatedUpaster(IEventUpcaster):
            @property
            def event_type(self) -> str:
                return "verse_created"

            @property
            def from_version(self) -> int:
                return 1

            @property
            def to_version(self) -> int:
                return 2

            def upcast(self, payload: Dict, metadata: EventMetadata) -> Dict:
                # Add new required field with default
                payload["text_language"] = payload.get("text_language", "hebrew")
                return payload
    """

    @property
    @abstractmethod
    def event_type(self) -> str:
        """Event type this upcaster handles."""
        pass

    @property
    @abstractmethod
    def from_version(self) -> int:
        """Source schema version."""
        pass

    @property
    @abstractmethod
    def to_version(self) -> int:
        """Target schema version."""
        pass

    @abstractmethod
    def upcast(self, payload: Dict[str, Any], metadata: EventMetadata) -> Dict[str, Any]:
        """
        Transform event payload from old to new schema version.

        Args:
            payload: Event payload in old format
            metadata: Event metadata

        Returns:
            Payload in new format
        """
        pass


class ISnapshotStore(ABC):
    """
    Interface for aggregate snapshot storage.

    Snapshots accelerate aggregate rehydration by providing a starting point
    closer to the current state than the beginning of time.
    """

    @abstractmethod
    async def save_snapshot(self, snapshot: Snapshot[Any]) -> None:
        """
        Save an aggregate snapshot.

        Args:
            snapshot: Snapshot to save
        """
        pass

    @abstractmethod
    async def get_latest_snapshot(
        self,
        aggregate_id: str,
        aggregate_type: str,
    ) -> Optional[Snapshot[Any]]:
        """
        Get the latest snapshot for an aggregate.

        Args:
            aggregate_id: Aggregate identifier
            aggregate_type: Type of aggregate

        Returns:
            Latest snapshot or None if no snapshots exist
        """
        pass

    @abstractmethod
    async def get_snapshot_at_version(
        self,
        aggregate_id: str,
        aggregate_type: str,
        version: int,
    ) -> Optional[Snapshot[Any]]:
        """
        Get snapshot at or before a specific version.

        Args:
            aggregate_id: Aggregate identifier
            aggregate_type: Type of aggregate
            version: Target version

        Returns:
            Snapshot at or before the specified version
        """
        pass

    @abstractmethod
    async def delete_snapshots_before(
        self,
        aggregate_id: str,
        aggregate_type: str,
        version: int,
    ) -> int:
        """
        Delete old snapshots before a version (cleanup).

        Args:
            aggregate_id: Aggregate identifier
            aggregate_type: Type of aggregate
            version: Version threshold

        Returns:
            Number of snapshots deleted
        """
        pass


@runtime_checkable
class ISubscriptionHandler(Protocol):
    """Protocol for subscription event handlers."""

    async def handle(self, event: StoredEvent) -> None:
        """Handle an event from the subscription."""
        ...


class IEventStore(ABC):
    """
    Interface for event store operations.

    The event store is the persistent memory of all domain changes.
    """

    @abstractmethod
    async def append(
        self,
        event: BaseEvent,
        expected_version: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StreamPosition:
        """
        Append an event with optimistic concurrency control.

        Args:
            event: Event to append
            expected_version: Expected aggregate version for concurrency check
            metadata: Additional metadata to attach

        Returns:
            Position of the appended event

        Raises:
            ConcurrencyError: If version check fails
        """
        pass

    @abstractmethod
    async def append_batch(
        self,
        events: List[BaseEvent],
        expected_version: Optional[int] = None,
    ) -> List[StreamPosition]:
        """
        Append multiple events atomically.

        Args:
            events: Events to append (must be for same aggregate)
            expected_version: Expected starting version

        Returns:
            Positions of appended events
        """
        pass

    @abstractmethod
    async def get_events_by_aggregate(
        self,
        aggregate_id: str,
        from_version: int = 0,
        to_version: Optional[int] = None,
    ) -> List[StoredEvent]:
        """Get events for an aggregate within a version range."""
        pass

    @abstractmethod
    async def get_events_by_stream(
        self,
        stream_name: str,
        from_position: StreamPosition = StreamPosition.start(),
        limit: int = 1000,
    ) -> List[StoredEvent]:
        """Get events from a named stream."""
        pass

    @abstractmethod
    async def get_aggregate_version(self, aggregate_id: str) -> int:
        """Get current version of an aggregate."""
        pass

    @abstractmethod
    async def stream_events(
        self,
        from_position: StreamPosition = StreamPosition.start(),
        batch_size: int = 100,
    ) -> AsyncIterator[StoredEvent]:
        """Stream all events from a position."""
        pass


# =============================================================================
# UPCASTER REGISTRY
# =============================================================================


class UpcasterRegistry:
    """
    Registry for event upcasters.

    Maintains a chain of upcasters for each event type, allowing events
    to be transformed through multiple schema versions.

    The registry is the translator that brings old events into the
    current dialect of the domain language.
    """

    def __init__(self) -> None:
        self._upcasters: Dict[str, Dict[int, IEventUpcaster]] = {}
        self._current_versions: Dict[str, int] = {}

    def register(self, upcaster: IEventUpcaster) -> None:
        """
        Register an upcaster.

        Args:
            upcaster: Upcaster to register
        """
        event_type = upcaster.event_type
        if event_type not in self._upcasters:
            self._upcasters[event_type] = {}

        self._upcasters[event_type][upcaster.from_version] = upcaster

        # Track highest version
        current = self._current_versions.get(event_type, 1)
        self._current_versions[event_type] = max(current, upcaster.to_version)

        logger.debug(
            f"Registered upcaster for {event_type} v{upcaster.from_version} -> v{upcaster.to_version}"
        )

    def upcast(
        self,
        event_type: str,
        payload: Dict[str, Any],
        metadata: EventMetadata,
    ) -> Tuple[Dict[str, Any], int]:
        """
        Upcast an event payload to the current schema version.

        Args:
            event_type: Type of event
            payload: Event payload
            metadata: Event metadata (contains schema_version)

        Returns:
            Tuple of (upcasted payload, final version)
        """
        current_version = metadata.schema_version
        target_version = self._current_versions.get(event_type, 1)

        if current_version >= target_version:
            return payload, current_version

        upcasters = self._upcasters.get(event_type, {})

        while current_version < target_version:
            upcaster = upcasters.get(current_version)
            if upcaster is None:
                raise UpcastingError(
                    f"No upcaster found for {event_type} v{current_version} -> v{current_version + 1}"
                )

            try:
                payload = upcaster.upcast(payload, metadata)
                current_version = upcaster.to_version
            except Exception as e:
                raise UpcastingError(
                    f"Failed to upcast {event_type} from v{current_version}: {e}"
                ) from e

        return payload, current_version


# =============================================================================
# SNAPSHOT STORE IMPLEMENTATION
# =============================================================================


class SnapshotStore(ISnapshotStore):
    """
    PostgreSQL-backed snapshot store.

    Snapshots are stored as JSON blobs with metadata for efficient retrieval.
    """

    def __init__(self, connection_pool: asyncpg.Pool) -> None:
        self.pool = connection_pool
        self._serializers: Dict[str, Callable[[Any], Dict[str, Any]]] = {}
        self._deserializers: Dict[str, Callable[[Dict[str, Any]], Any]] = {}

    def register_serializer(
        self,
        aggregate_type: str,
        serializer: Callable[[Any], Dict[str, Any]],
        deserializer: Callable[[Dict[str, Any]], Any],
    ) -> None:
        """
        Register serialization functions for an aggregate type.

        Args:
            aggregate_type: Type of aggregate
            serializer: Function to convert state to dict
            deserializer: Function to convert dict to state
        """
        self._serializers[aggregate_type] = serializer
        self._deserializers[aggregate_type] = deserializer

    async def initialize(self) -> None:
        """Initialize snapshot schema."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    aggregate_id VARCHAR(200) NOT NULL,
                    aggregate_type VARCHAR(100) NOT NULL,
                    version INTEGER NOT NULL,
                    state JSONB NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE(aggregate_id, aggregate_type, version)
                )
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_snapshots_aggregate
                ON snapshots(aggregate_id, aggregate_type, version DESC)
            """)

            logger.info("Snapshot store schema initialized")

    async def save_snapshot(self, snapshot: Snapshot[Any]) -> None:
        """Save an aggregate snapshot."""
        serializer = self._serializers.get(snapshot.aggregate_type)
        state_dict = serializer(snapshot.state) if serializer else snapshot.state

        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO snapshots (
                    aggregate_id, aggregate_type, version, state, metadata, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (aggregate_id, aggregate_type, version)
                DO UPDATE SET state = $4, metadata = $5, created_at = $6
            """,
                snapshot.aggregate_id,
                snapshot.aggregate_type,
                snapshot.version,
                json.dumps(state_dict),
                json.dumps(snapshot.metadata) if snapshot.metadata else None,
                snapshot.created_at,
            )

        logger.debug(
            f"Saved snapshot for {snapshot.aggregate_type}/{snapshot.aggregate_id} "
            f"at version {snapshot.version}"
        )

    async def get_latest_snapshot(
        self,
        aggregate_id: str,
        aggregate_type: str,
    ) -> Optional[Snapshot[Any]]:
        """Get the latest snapshot for an aggregate."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT version, state, metadata, created_at
                FROM snapshots
                WHERE aggregate_id = $1 AND aggregate_type = $2
                ORDER BY version DESC
                LIMIT 1
            """, aggregate_id, aggregate_type)

        if row is None:
            return None

        return self._row_to_snapshot(aggregate_id, aggregate_type, row)

    async def get_snapshot_at_version(
        self,
        aggregate_id: str,
        aggregate_type: str,
        version: int,
    ) -> Optional[Snapshot[Any]]:
        """Get snapshot at or before a specific version."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT version, state, metadata, created_at
                FROM snapshots
                WHERE aggregate_id = $1 AND aggregate_type = $2 AND version <= $3
                ORDER BY version DESC
                LIMIT 1
            """, aggregate_id, aggregate_type, version)

        if row is None:
            return None

        return self._row_to_snapshot(aggregate_id, aggregate_type, row)

    async def delete_snapshots_before(
        self,
        aggregate_id: str,
        aggregate_type: str,
        version: int,
    ) -> int:
        """Delete old snapshots before a version."""
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM snapshots
                WHERE aggregate_id = $1 AND aggregate_type = $2 AND version < $3
            """, aggregate_id, aggregate_type, version)

        count = int(result.split()[-1])
        logger.debug(f"Deleted {count} old snapshots for {aggregate_type}/{aggregate_id}")
        return count

    def _row_to_snapshot(
        self,
        aggregate_id: str,
        aggregate_type: str,
        row: asyncpg.Record,
    ) -> Snapshot[Any]:
        """Convert database row to snapshot."""
        state_dict = row["state"]
        deserializer = self._deserializers.get(aggregate_type)
        state = deserializer(state_dict) if deserializer else state_dict

        return Snapshot(
            aggregate_id=aggregate_id,
            aggregate_type=aggregate_type,
            version=row["version"],
            state=state,
            created_at=row["created_at"],
            metadata=row["metadata"] or {},
        )


# =============================================================================
# SUBSCRIPTION MANAGEMENT
# =============================================================================


class SubscriptionState(Enum):
    """State of a subscription."""
    STOPPED = "stopped"
    RUNNING = "running"
    CATCHING_UP = "catching_up"
    LIVE = "live"
    FAILED = "failed"


@dataclass
class PersistentSubscription:
    """
    A persistent subscription that tracks its position.

    Persistent subscriptions remember where they left off, allowing
    them to resume after restarts without missing events.
    """
    subscription_id: str
    stream_name: str
    handler: Callable[[StoredEvent], Any]
    state: SubscriptionState = SubscriptionState.STOPPED
    position: StreamPosition = field(default_factory=StreamPosition.start)
    last_error: Optional[Exception] = None
    _task: Optional[asyncio.Task[None]] = field(default=None, repr=False)


class SubscriptionManager:
    """
    Manages event subscriptions.

    The subscription manager is the nervous system - distributing event
    signals to all interested parties.
    """

    def __init__(
        self,
        event_store: "EventStore",
        connection_pool: asyncpg.Pool,
    ) -> None:
        self._event_store = event_store
        self.pool = connection_pool
        self._subscriptions: Dict[str, PersistentSubscription] = {}
        self._transient_handlers: Dict[str, List[Callable[[StoredEvent], Any]]] = {}
        self._running = False

    async def initialize(self) -> None:
        """Initialize subscription checkpoint table."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS subscription_checkpoints (
                    subscription_id VARCHAR(200) PRIMARY KEY,
                    stream_name VARCHAR(200) NOT NULL,
                    position BIGINT NOT NULL DEFAULT 0,
                    last_processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    metadata JSONB
                )
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS dead_letter_events (
                    id SERIAL PRIMARY KEY,
                    subscription_id VARCHAR(200) NOT NULL,
                    event_id UUID NOT NULL,
                    event_type VARCHAR(100) NOT NULL,
                    error_message TEXT NOT NULL,
                    payload JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    retry_count INTEGER NOT NULL DEFAULT 0
                )
            """)

        logger.info("Subscription manager initialized")

    async def create_persistent_subscription(
        self,
        subscription_id: str,
        stream_name: str,
        handler: Callable[[StoredEvent], Any],
        start_from: StreamPosition = StreamPosition.start(),
    ) -> PersistentSubscription:
        """
        Create a persistent subscription.

        Args:
            subscription_id: Unique identifier for the subscription
            stream_name: Name of stream to subscribe to
            handler: Async function to handle events
            start_from: Position to start from

        Returns:
            The created subscription
        """
        # Load existing checkpoint
        checkpoint = await self._load_checkpoint(subscription_id)
        if checkpoint:
            start_from = checkpoint.position

        subscription = PersistentSubscription(
            subscription_id=subscription_id,
            stream_name=stream_name,
            handler=handler,
            position=start_from,
        )

        self._subscriptions[subscription_id] = subscription
        logger.info(f"Created persistent subscription {subscription_id} from position {start_from.value}")
        return subscription

    async def start_subscription(self, subscription_id: str) -> None:
        """Start a persistent subscription."""
        subscription = self._subscriptions.get(subscription_id)
        if subscription is None:
            raise SubscriptionError(f"Subscription {subscription_id} not found")

        if subscription.state == SubscriptionState.RUNNING:
            return

        subscription.state = SubscriptionState.CATCHING_UP
        subscription._task = asyncio.create_task(
            self._run_subscription(subscription)
        )
        logger.info(f"Started subscription {subscription_id}")

    async def stop_subscription(self, subscription_id: str) -> None:
        """Stop a persistent subscription."""
        subscription = self._subscriptions.get(subscription_id)
        if subscription is None:
            return

        subscription.state = SubscriptionState.STOPPED
        if subscription._task:
            subscription._task.cancel()
            try:
                await subscription._task
            except asyncio.CancelledError:
                pass
            subscription._task = None

        logger.info(f"Stopped subscription {subscription_id}")

    def subscribe_transient(
        self,
        event_type: str,
        handler: Callable[[StoredEvent], Any],
    ) -> Callable[[], None]:
        """
        Subscribe transiently to events of a type.

        Returns a function to unsubscribe.
        """
        if event_type not in self._transient_handlers:
            self._transient_handlers[event_type] = []
        self._transient_handlers[event_type].append(handler)

        def unsubscribe() -> None:
            self._transient_handlers[event_type].remove(handler)

        return unsubscribe

    async def notify_transient(self, event: StoredEvent) -> None:
        """Notify transient subscribers of a new event."""
        handlers = self._transient_handlers.get(event.metadata.event_type, [])
        for handler in handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Error in transient handler: {e}", exc_info=True)

    async def _run_subscription(self, subscription: PersistentSubscription) -> None:
        """Run the subscription loop."""
        try:
            # Catch-up phase
            async for event in self._event_store.stream_events(
                from_position=subscription.position,
                batch_size=100,
            ):
                if subscription.state == SubscriptionState.STOPPED:
                    break

                await self._process_event(subscription, event)

            # Switch to live mode
            subscription.state = SubscriptionState.LIVE

            # TODO: Implement live polling or LISTEN/NOTIFY
            while subscription.state != SubscriptionState.STOPPED:
                await asyncio.sleep(1)  # Poll interval

        except Exception as e:
            subscription.state = SubscriptionState.FAILED
            subscription.last_error = e
            logger.error(f"Subscription {subscription.subscription_id} failed: {e}", exc_info=True)

    async def _process_event(
        self,
        subscription: PersistentSubscription,
        event: StoredEvent,
    ) -> None:
        """Process a single event in a subscription."""
        try:
            result = subscription.handler(event)
            if asyncio.iscoroutine(result):
                await result

            # Update checkpoint
            new_position = StreamPosition(event.metadata.stream_position or 0)
            subscription.position = new_position + 1
            await self._save_checkpoint(subscription)

        except Exception as e:
            logger.error(
                f"Error processing event {event.metadata.event_id} "
                f"in subscription {subscription.subscription_id}: {e}",
                exc_info=True,
            )
            await self._move_to_dead_letter(subscription, event, e)

    async def _save_checkpoint(self, subscription: PersistentSubscription) -> None:
        """Save subscription checkpoint."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO subscription_checkpoints (subscription_id, stream_name, position, last_processed_at)
                VALUES ($1, $2, $3, NOW())
                ON CONFLICT (subscription_id)
                DO UPDATE SET position = $3, last_processed_at = NOW()
            """,
                subscription.subscription_id,
                subscription.stream_name,
                subscription.position.value,
            )

    async def _load_checkpoint(self, subscription_id: str) -> Optional[SubscriptionCheckpoint]:
        """Load subscription checkpoint."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT stream_name, position, last_processed_at
                FROM subscription_checkpoints
                WHERE subscription_id = $1
            """, subscription_id)

        if row is None:
            return None

        return SubscriptionCheckpoint(
            subscription_id=subscription_id,
            stream_name=row["stream_name"],
            position=StreamPosition(row["position"]),
            last_processed_at=row["last_processed_at"],
        )

    async def _move_to_dead_letter(
        self,
        subscription: PersistentSubscription,
        event: StoredEvent,
        error: Exception,
    ) -> None:
        """Move failed event to dead letter queue."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO dead_letter_events (
                    subscription_id, event_id, event_type, error_message, payload
                ) VALUES ($1, $2, $3, $4, $5)
            """,
                subscription.subscription_id,
                event.metadata.event_id,
                event.metadata.event_type,
                str(error),
                json.dumps(event.payload),
            )


# =============================================================================
# EVENT STORE IMPLEMENTATION
# =============================================================================


class EventStore(IEventStore):
    """
    PostgreSQL-backed event store with comprehensive features.

    The event store is the heart of the event sourcing architecture -
    an append-only log that captures every meaningful change in the domain.

    Features:
    - Optimistic concurrency control
    - Event schema versioning with upcasting
    - Correlation/causation tracking
    - Efficient streaming and querying
    - Transient and persistent subscriptions
    """

    def __init__(
        self,
        connection_pool: asyncpg.Pool,
        upcaster_registry: Optional[UpcasterRegistry] = None,
    ) -> None:
        """
        Initialize event store.

        Args:
            connection_pool: PostgreSQL connection pool
            upcaster_registry: Registry of event upcasters
        """
        self.pool = connection_pool
        self._upcaster_registry = upcaster_registry or UpcasterRegistry()
        self._subscription_manager: Optional[SubscriptionManager] = None
        self._snapshot_store: Optional[SnapshotStore] = None
        self._global_position_counter = 0

    @property
    def subscriptions(self) -> SubscriptionManager:
        """Get subscription manager."""
        if self._subscription_manager is None:
            raise RuntimeError("Subscription manager not initialized")
        return self._subscription_manager

    @property
    def snapshots(self) -> SnapshotStore:
        """Get snapshot store."""
        if self._snapshot_store is None:
            raise RuntimeError("Snapshot store not initialized")
        return self._snapshot_store

    async def initialize(self) -> None:
        """
        Initialize database schema.

        Creates events table with indexes for efficient querying.
        """
        async with self.pool.acquire() as conn:
            # Create events table with enhanced schema
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    global_position BIGSERIAL,
                    event_id UUID PRIMARY KEY,
                    event_type VARCHAR(100) NOT NULL,
                    aggregate_id VARCHAR(200) NOT NULL,
                    aggregate_type VARCHAR(100) NOT NULL,
                    aggregate_version INTEGER NOT NULL,
                    schema_version INTEGER NOT NULL DEFAULT 1,
                    timestamp TIMESTAMPTZ NOT NULL,
                    correlation_id VARCHAR(100),
                    causation_id VARCHAR(100),
                    user_id VARCHAR(100),
                    tenant_id VARCHAR(100),
                    payload JSONB NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE(aggregate_id, aggregate_version)
                )
            """)

            # Indexes for efficient querying
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_global_position
                ON events(global_position)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_aggregate
                ON events(aggregate_id, aggregate_version)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_aggregate_type
                ON events(aggregate_type)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_type
                ON events(event_type)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_correlation
                ON events(correlation_id)
                WHERE correlation_id IS NOT NULL
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_timestamp
                ON events(timestamp DESC)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_tenant
                ON events(tenant_id)
                WHERE tenant_id IS NOT NULL
            """)

            # Get current max position
            max_pos = await conn.fetchval("""
                SELECT COALESCE(MAX(global_position), 0) FROM events
            """)
            self._global_position_counter = max_pos

        # Initialize related stores
        self._snapshot_store = SnapshotStore(self.pool)
        await self._snapshot_store.initialize()

        self._subscription_manager = SubscriptionManager(self, self.pool)
        await self._subscription_manager.initialize()

        logger.info("Event store initialized successfully")

    async def append(
        self,
        event: BaseEvent,
        expected_version: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StreamPosition:
        """
        Append event to the store with optimistic concurrency control.

        Args:
            event: Event to append
            expected_version: Expected current version of aggregate
            metadata: Additional metadata

        Returns:
            Global position of the appended event
        """
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Concurrency check
                if expected_version is not None:
                    current_version = await conn.fetchval("""
                        SELECT MAX(aggregate_version)
                        FROM events
                        WHERE aggregate_id = $1
                    """, event.aggregate_id)

                    current_version = current_version or 0
                    if current_version != expected_version:
                        raise ConcurrencyError(
                            event.aggregate_id,
                            expected_version,
                            current_version,
                        )

                # Extract payload (non-metadata fields)
                event_dict = event.to_dict()
                payload = {
                    k: v for k, v in event_dict.items()
                    if k not in [
                        "event_id", "event_type", "aggregate_id",
                        "aggregate_version", "timestamp", "correlation_id",
                        "causation_id", "metadata"
                    ]
                }

                # Determine aggregate type from event type or metadata
                aggregate_type = (
                    metadata.get("aggregate_type")
                    if metadata else event.event_type.split("_")[0]
                )

                # Insert event
                global_position = await conn.fetchval("""
                    INSERT INTO events (
                        event_id, event_type, aggregate_id, aggregate_type,
                        aggregate_version, schema_version, timestamp,
                        correlation_id, causation_id, user_id, tenant_id,
                        payload, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    RETURNING global_position
                """,
                    event.event_id,
                    event.event_type,
                    event.aggregate_id,
                    aggregate_type,
                    event.aggregate_version,
                    metadata.get("schema_version", 1) if metadata else 1,
                    event.timestamp,
                    event.correlation_id,
                    event.causation_id,
                    metadata.get("user_id") if metadata else None,
                    metadata.get("tenant_id") if metadata else None,
                    json.dumps(payload),
                    json.dumps(event.metadata) if event.metadata else None,
                )

        logger.debug(
            f"Appended event {event.event_type} for {event.aggregate_id} "
            f"v{event.aggregate_version} at position {global_position}"
        )

        # Create stored event for notifications
        stored_event = StoredEvent(
            metadata=EventMetadata(
                event_id=event.event_id,
                event_type=event.event_type,
                aggregate_id=event.aggregate_id,
                aggregate_type=aggregate_type,
                aggregate_version=event.aggregate_version,
                timestamp=event.timestamp,
                correlation_id=event.correlation_id,
                causation_id=event.causation_id,
                stream_position=global_position,
            ),
            payload=payload,
            stored_at=datetime.utcnow(),
        )

        # Notify transient subscribers
        if self._subscription_manager:
            await self._subscription_manager.notify_transient(stored_event)

        return StreamPosition(global_position)

    async def append_batch(
        self,
        events: List[BaseEvent],
        expected_version: Optional[int] = None,
    ) -> List[StreamPosition]:
        """Append multiple events atomically."""
        if not events:
            return []

        # All events must be for the same aggregate
        aggregate_ids = {e.aggregate_id for e in events}
        if len(aggregate_ids) > 1:
            raise EventStoreError("Batch append requires all events for same aggregate")

        positions: List[StreamPosition] = []

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Concurrency check
                if expected_version is not None:
                    current_version = await conn.fetchval("""
                        SELECT MAX(aggregate_version)
                        FROM events
                        WHERE aggregate_id = $1
                    """, events[0].aggregate_id)

                    current_version = current_version or 0
                    if current_version != expected_version:
                        raise ConcurrencyError(
                            events[0].aggregate_id,
                            expected_version,
                            current_version,
                        )

                for event in events:
                    event_dict = event.to_dict()
                    payload = {
                        k: v for k, v in event_dict.items()
                        if k not in [
                            "event_id", "event_type", "aggregate_id",
                            "aggregate_version", "timestamp", "correlation_id",
                            "causation_id", "metadata"
                        ]
                    }

                    aggregate_type = event.event_type.split("_")[0]

                    global_position = await conn.fetchval("""
                        INSERT INTO events (
                            event_id, event_type, aggregate_id, aggregate_type,
                            aggregate_version, timestamp, correlation_id,
                            causation_id, payload, metadata
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        RETURNING global_position
                    """,
                        event.event_id,
                        event.event_type,
                        event.aggregate_id,
                        aggregate_type,
                        event.aggregate_version,
                        event.timestamp,
                        event.correlation_id,
                        event.causation_id,
                        json.dumps(payload),
                        json.dumps(event.metadata) if event.metadata else None,
                    )
                    positions.append(StreamPosition(global_position))

        logger.info(f"Appended batch of {len(events)} events")

        # Notify subscribers
        if self._subscription_manager:
            for event in events:
                stored_event = StoredEvent(
                    metadata=EventMetadata(
                        event_id=event.event_id,
                        event_type=event.event_type,
                        aggregate_id=event.aggregate_id,
                        aggregate_type=event.event_type.split("_")[0],
                        aggregate_version=event.aggregate_version,
                        timestamp=event.timestamp,
                        correlation_id=event.correlation_id,
                        causation_id=event.causation_id,
                    ),
                    payload={},  # Simplified
                    stored_at=datetime.utcnow(),
                )
                await self._subscription_manager.notify_transient(stored_event)

        return positions

    async def get_events_by_aggregate(
        self,
        aggregate_id: str,
        from_version: int = 0,
        to_version: Optional[int] = None,
    ) -> List[StoredEvent]:
        """Get all events for an aggregate."""
        query = """
            SELECT global_position, event_id, event_type, aggregate_id,
                   aggregate_type, aggregate_version, schema_version,
                   timestamp, correlation_id, causation_id, user_id,
                   tenant_id, payload, metadata, created_at
            FROM events
            WHERE aggregate_id = $1 AND aggregate_version >= $2
        """
        params: List[Any] = [aggregate_id, from_version]

        if to_version is not None:
            query += " AND aggregate_version <= $3"
            params.append(to_version)

        query += " ORDER BY aggregate_version ASC"

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        return [self._row_to_stored_event(row) for row in rows]

    async def get_events_by_stream(
        self,
        stream_name: str,
        from_position: StreamPosition = StreamPosition.start(),
        limit: int = 1000,
    ) -> List[StoredEvent]:
        """Get events from a named stream (category, type, etc)."""
        # Parse stream name to determine query type
        if stream_name.startswith("$ce-"):
            # Category stream
            category = stream_name[4:]
            query = """
                SELECT global_position, event_id, event_type, aggregate_id,
                       aggregate_type, aggregate_version, schema_version,
                       timestamp, correlation_id, causation_id, user_id,
                       tenant_id, payload, metadata, created_at
                FROM events
                WHERE aggregate_type = $1 AND global_position > $2
                ORDER BY global_position ASC
                LIMIT $3
            """
            params = [category, from_position.value, limit]
        elif stream_name.startswith("$et-"):
            # Event type stream
            event_type = stream_name[4:]
            query = """
                SELECT global_position, event_id, event_type, aggregate_id,
                       aggregate_type, aggregate_version, schema_version,
                       timestamp, correlation_id, causation_id, user_id,
                       tenant_id, payload, metadata, created_at
                FROM events
                WHERE event_type = $1 AND global_position > $2
                ORDER BY global_position ASC
                LIMIT $3
            """
            params = [event_type, from_position.value, limit]
        else:
            # All events
            query = """
                SELECT global_position, event_id, event_type, aggregate_id,
                       aggregate_type, aggregate_version, schema_version,
                       timestamp, correlation_id, causation_id, user_id,
                       tenant_id, payload, metadata, created_at
                FROM events
                WHERE global_position > $1
                ORDER BY global_position ASC
                LIMIT $2
            """
            params = [from_position.value, limit]

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        return [self._row_to_stored_event(row) for row in rows]

    async def get_events_by_type(
        self,
        event_type: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[StoredEvent]:
        """Get events by type."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT global_position, event_id, event_type, aggregate_id,
                       aggregate_type, aggregate_version, schema_version,
                       timestamp, correlation_id, causation_id, user_id,
                       tenant_id, payload, metadata, created_at
                FROM events
                WHERE event_type = $1
                ORDER BY timestamp DESC
                LIMIT $2 OFFSET $3
            """, event_type, limit, offset)

        return [self._row_to_stored_event(row) for row in rows]

    async def get_events_by_correlation(
        self,
        correlation_id: str,
    ) -> List[StoredEvent]:
        """Get all events with a specific correlation ID."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT global_position, event_id, event_type, aggregate_id,
                       aggregate_type, aggregate_version, schema_version,
                       timestamp, correlation_id, causation_id, user_id,
                       tenant_id, payload, metadata, created_at
                FROM events
                WHERE correlation_id = $1
                ORDER BY timestamp ASC
            """, correlation_id)

        return [self._row_to_stored_event(row) for row in rows]

    async def get_events_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        event_types: Optional[List[str]] = None,
        limit: int = 1000,
    ) -> List[StoredEvent]:
        """Get events within a time range."""
        query = """
            SELECT global_position, event_id, event_type, aggregate_id,
                   aggregate_type, aggregate_version, schema_version,
                   timestamp, correlation_id, causation_id, user_id,
                   tenant_id, payload, metadata, created_at
            FROM events
            WHERE timestamp >= $1 AND timestamp <= $2
        """
        params: List[Any] = [start_time, end_time]

        if event_types:
            query += " AND event_type = ANY($3)"
            params.append(event_types)
            query += " ORDER BY timestamp ASC LIMIT $4"
            params.append(limit)
        else:
            query += " ORDER BY timestamp ASC LIMIT $3"
            params.append(limit)

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        return [self._row_to_stored_event(row) for row in rows]

    async def stream_events(
        self,
        from_position: StreamPosition = StreamPosition.start(),
        batch_size: int = 100,
    ) -> AsyncIterator[StoredEvent]:
        """Stream all events from a position."""
        current_position = from_position.value

        while True:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT global_position, event_id, event_type, aggregate_id,
                           aggregate_type, aggregate_version, schema_version,
                           timestamp, correlation_id, causation_id, user_id,
                           tenant_id, payload, metadata, created_at
                    FROM events
                    WHERE global_position > $1
                    ORDER BY global_position ASC
                    LIMIT $2
                """, current_position, batch_size)

            if not rows:
                break

            for row in rows:
                event = self._row_to_stored_event(row)
                yield event
                current_position = row["global_position"]

    async def get_aggregate_version(self, aggregate_id: str) -> int:
        """Get current version of an aggregate."""
        async with self.pool.acquire() as conn:
            version = await conn.fetchval("""
                SELECT MAX(aggregate_version)
                FROM events
                WHERE aggregate_id = $1
            """, aggregate_id)

        return version if version is not None else 0

    def subscribe(self, event_type: str, callback: Callable[[BaseEvent], Any]) -> Callable[[], None]:
        """
        Subscribe to events of a specific type (transient).

        Args:
            event_type: Event type to subscribe to
            callback: Async function to call when event occurs

        Returns:
            Function to unsubscribe
        """
        async def wrapper(stored_event: StoredEvent) -> None:
            domain_event = stored_event.to_domain_event()
            result = callback(domain_event)
            if asyncio.iscoroutine(result):
                await result

        if self._subscription_manager:
            return self._subscription_manager.subscribe_transient(event_type, wrapper)
        else:
            # Fallback to basic subscription if manager not initialized
            logger.warning("Subscription manager not initialized, subscription may not work")
            return lambda: None

    def _row_to_stored_event(self, row: asyncpg.Record) -> StoredEvent:
        """Convert database row to stored event."""
        payload = row["payload"] if row["payload"] else {}

        # Apply upcasting if needed
        schema_version = row.get("schema_version", 1) or 1
        metadata = EventMetadata(
            event_id=row["event_id"],
            event_type=row["event_type"],
            aggregate_id=row["aggregate_id"],
            aggregate_type=row["aggregate_type"],
            aggregate_version=row["aggregate_version"],
            timestamp=row["timestamp"],
            correlation_id=row["correlation_id"],
            causation_id=row["causation_id"],
            schema_version=schema_version,
            user_id=row.get("user_id"),
            tenant_id=row.get("tenant_id"),
            stream_position=row["global_position"],
            custom=row["metadata"] if row["metadata"] else {},
        )

        # Upcast if necessary
        if self._upcaster_registry:
            payload, _ = self._upcaster_registry.upcast(
                row["event_type"],
                payload,
                metadata,
            )

        return StoredEvent(
            metadata=metadata,
            payload=payload,
            stored_at=row["created_at"],
        )

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self._subscription_manager:
            for sub_id in list(self._subscription_manager._subscriptions.keys()):
                await self._subscription_manager.stop_subscription(sub_id)
        logger.info("Event store cleaned up")
