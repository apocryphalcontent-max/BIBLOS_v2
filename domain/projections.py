"""
BIBLOS v2 - Event Projections

The perceptive organs of the BIBLOS organism - they observe the event stream
and build materialized views for efficient querying without corrupting the source.

This module implements the CQRS read model:
    - Event handlers transform events into query-optimized views
    - Checkpointing enables resumable processing after restart
    - Rebuild capability allows full reconstruction from event history
    - Concurrent projections process events in parallel

Architecture:
    EventStore → Subscription → Projector → Projection → QueryStore

Projections are seraphic in nature:
    - They observe without modifying the source (pure light)
    - They partake only in what they need (right abundances)
    - They are resilient to failure (persistent)
    - They work in harmony with the event store (harmonious yet individual)

Usage:
    # Define a projection
    class VerseCountByBookProjection(ProjectionBase):
        async def apply(self, event: StoredEvent) -> None:
            if event.event_type == "VerseCreated":
                book = event.data["verse_reference"][:3]
                self._state.setdefault(book, 0)
                self._state[book] += 1

    # Register and start
    manager = ProjectionManager(event_store, checkpoint_store)
    manager.register(VerseCountByBookProjection())
    await manager.start_all()
"""
from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    runtime_checkable,
)
from uuid import UUID

logger = logging.getLogger("biblos.projections")

T = TypeVar("T")
TState = TypeVar("TState")


# =============================================================================
# EVENT TYPES (MINIMAL - TO AVOID CIRCULAR IMPORTS)
# =============================================================================


@dataclass(frozen=True, slots=True)
class ProjectedEvent:
    """
    Minimal event representation for projection processing.

    Contains only what projections need - no more, no less.
    This is seraphic: partaking only in right abundances.
    """
    event_id: UUID
    event_type: str
    aggregate_id: str
    aggregate_type: str
    stream_name: str
    stream_position: int
    global_position: int
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: float
    schema_version: int = 1


# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================


class CheckpointStrategy(Enum):
    """Strategies for when to persist checkpoints."""
    AFTER_EACH_EVENT = auto()      # Maximum durability, lowest performance
    AFTER_BATCH = auto()           # Balance of durability and performance
    PERIODIC = auto()              # Time-based checkpointing
    MANUAL = auto()                # Application-controlled checkpointing


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """
    Immutable checkpoint record.

    Marks the position up to which a projection has processed events,
    enabling resumable processing after restart.
    """
    projection_name: str
    stream_name: str
    position: int
    global_position: int
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def initial(cls, projection_name: str, stream_name: str = "$all") -> "Checkpoint":
        """Create an initial checkpoint at position 0."""
        return cls(
            projection_name=projection_name,
            stream_name=stream_name,
            position=0,
            global_position=0,
            timestamp=time.time(),
        )


@runtime_checkable
class ICheckpointStore(Protocol):
    """
    Interface for checkpoint persistence.

    Implementations may store checkpoints in PostgreSQL, Redis, or files.
    """

    async def get_checkpoint(
        self,
        projection_name: str,
        stream_name: str = "$all",
    ) -> Optional[Checkpoint]:
        """Get the latest checkpoint for a projection."""
        ...

    async def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Save a checkpoint."""
        ...

    async def delete_checkpoint(
        self,
        projection_name: str,
        stream_name: str = "$all",
    ) -> None:
        """Delete a checkpoint (used during rebuild)."""
        ...


class InMemoryCheckpointStore:
    """
    In-memory checkpoint store for development and testing.

    Not suitable for production as checkpoints are lost on restart.
    """

    def __init__(self):
        self._checkpoints: Dict[str, Checkpoint] = {}

    def _key(self, projection_name: str, stream_name: str) -> str:
        return f"{projection_name}:{stream_name}"

    async def get_checkpoint(
        self,
        projection_name: str,
        stream_name: str = "$all",
    ) -> Optional[Checkpoint]:
        return self._checkpoints.get(self._key(projection_name, stream_name))

    async def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        key = self._key(checkpoint.projection_name, checkpoint.stream_name)
        self._checkpoints[key] = checkpoint

    async def delete_checkpoint(
        self,
        projection_name: str,
        stream_name: str = "$all",
    ) -> None:
        key = self._key(projection_name, stream_name)
        self._checkpoints.pop(key, None)


class PostgresCheckpointStore:
    """
    PostgreSQL-backed checkpoint store for production use.

    Provides durable checkpoint storage with transactional guarantees.
    """

    def __init__(self, pool: Any):
        self._pool = pool
        self._initialized = False

    async def initialize(self) -> None:
        """Create the checkpoints table if it doesn't exist."""
        if self._initialized:
            return

        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS projection_checkpoints (
                    projection_name TEXT NOT NULL,
                    stream_name TEXT NOT NULL,
                    position BIGINT NOT NULL,
                    global_position BIGINT NOT NULL,
                    timestamp DOUBLE PRECISION NOT NULL,
                    metadata JSONB DEFAULT '{}',
                    PRIMARY KEY (projection_name, stream_name)
                )
            """)

        self._initialized = True

    async def get_checkpoint(
        self,
        projection_name: str,
        stream_name: str = "$all",
    ) -> Optional[Checkpoint]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT projection_name, stream_name, position,
                       global_position, timestamp, metadata
                FROM projection_checkpoints
                WHERE projection_name = $1 AND stream_name = $2
                """,
                projection_name,
                stream_name,
            )

            if row is None:
                return None

            return Checkpoint(
                projection_name=row["projection_name"],
                stream_name=row["stream_name"],
                position=row["position"],
                global_position=row["global_position"],
                timestamp=row["timestamp"],
                metadata=row["metadata"] or {},
            )

    async def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO projection_checkpoints
                    (projection_name, stream_name, position, global_position, timestamp, metadata)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (projection_name, stream_name)
                DO UPDATE SET
                    position = EXCLUDED.position,
                    global_position = EXCLUDED.global_position,
                    timestamp = EXCLUDED.timestamp,
                    metadata = EXCLUDED.metadata
                """,
                checkpoint.projection_name,
                checkpoint.stream_name,
                checkpoint.position,
                checkpoint.global_position,
                checkpoint.timestamp,
                checkpoint.metadata,
            )

    async def delete_checkpoint(
        self,
        projection_name: str,
        stream_name: str = "$all",
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                DELETE FROM projection_checkpoints
                WHERE projection_name = $1 AND stream_name = $2
                """,
                projection_name,
                stream_name,
            )


# =============================================================================
# PROJECTION INTERFACES AND BASE CLASSES
# =============================================================================


class ProjectionStatus(Enum):
    """Status of a projection."""
    STOPPED = auto()
    STARTING = auto()
    CATCHING_UP = auto()  # Processing historical events
    LIVE = auto()          # Processing live events
    REBUILDING = auto()
    FAULTED = auto()


@dataclass
class ProjectionStats:
    """Statistics for a projection."""
    projection_name: str
    status: ProjectionStatus
    current_position: int
    events_processed: int
    events_per_second: float
    last_event_timestamp: Optional[float]
    last_error: Optional[str]
    uptime_seconds: float


@runtime_checkable
class IProjection(Protocol):
    """
    Interface for event projections.

    A projection transforms a stream of events into a query-optimized view.
    """

    @property
    def name(self) -> str:
        """Unique projection name."""
        ...

    @property
    def stream_filter(self) -> Optional[str]:
        """Stream name filter (None = all streams)."""
        ...

    @property
    def event_types(self) -> Optional[Set[str]]:
        """Event types to handle (None = all types)."""
        ...

    async def apply(self, event: ProjectedEvent) -> None:
        """Apply an event to update the projection state."""
        ...

    async def reset(self) -> None:
        """Reset the projection state (for rebuilds)."""
        ...


class ProjectionBase(ABC, Generic[TState]):
    """
    Base class for event projections with typed state.

    Provides common functionality:
        - Event filtering by type
        - State management
        - Statistics tracking
        - Error handling

    Usage:
        class VerseCountProjection(ProjectionBase[Dict[str, int]]):
            def __init__(self):
                super().__init__("VerseCount")

            def initial_state(self) -> Dict[str, int]:
                return {}

            async def apply(self, event: ProjectedEvent) -> None:
                if event.event_type == "VerseCreated":
                    book = event.data["verse_reference"][:3]
                    self._state.setdefault(book, 0)
                    self._state[book] += 1
    """

    def __init__(
        self,
        name: str,
        stream_filter: Optional[str] = None,
        event_types: Optional[Set[str]] = None,
    ):
        self._name = name
        self._stream_filter = stream_filter
        self._event_types = event_types
        self._state: TState = self.initial_state()
        self._events_processed = 0
        self._start_time = time.time()
        self._last_event_time: Optional[float] = None
        self._last_error: Optional[str] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def stream_filter(self) -> Optional[str]:
        return self._stream_filter

    @property
    def event_types(self) -> Optional[Set[str]]:
        return self._event_types

    @property
    def state(self) -> TState:
        """Get the current projection state."""
        return self._state

    @property
    def events_processed(self) -> int:
        return self._events_processed

    @abstractmethod
    def initial_state(self) -> TState:
        """Return the initial state for this projection."""
        ...

    def should_handle(self, event: ProjectedEvent) -> bool:
        """Check if this projection should handle the given event."""
        # Check stream filter
        if self._stream_filter and not event.stream_name.startswith(self._stream_filter):
            return False

        # Check event type filter
        if self._event_types and event.event_type not in self._event_types:
            return False

        return True

    @abstractmethod
    async def apply(self, event: ProjectedEvent) -> None:
        """Apply an event to update the projection state."""
        ...

    async def handle(self, event: ProjectedEvent) -> None:
        """Handle an event with filtering and tracking."""
        if not self.should_handle(event):
            return

        try:
            await self.apply(event)
            self._events_processed += 1
            self._last_event_time = time.time()
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"Projection {self._name} failed on event {event.event_id}: {e}")
            raise

    async def reset(self) -> None:
        """Reset the projection state for rebuilds."""
        self._state = self.initial_state()
        self._events_processed = 0
        self._last_event_time = None
        self._last_error = None

    def get_stats(self, status: ProjectionStatus, position: int) -> ProjectionStats:
        """Get projection statistics."""
        uptime = time.time() - self._start_time
        eps = self._events_processed / uptime if uptime > 0 else 0.0

        return ProjectionStats(
            projection_name=self._name,
            status=status,
            current_position=position,
            events_processed=self._events_processed,
            events_per_second=eps,
            last_event_timestamp=self._last_event_time,
            last_error=self._last_error,
            uptime_seconds=uptime,
        )


# =============================================================================
# SPECIALIZED PROJECTION TYPES
# =============================================================================


class AggregateProjection(ProjectionBase[Dict[str, Any]]):
    """
    Projection that maintains state per aggregate.

    Useful for building read models that mirror aggregate state
    but optimized for querying.
    """

    def __init__(
        self,
        name: str,
        aggregate_type: str,
    ):
        super().__init__(name)
        self._aggregate_type = aggregate_type

    def initial_state(self) -> Dict[str, Any]:
        return {}

    def get_aggregate_state(self, aggregate_id: str) -> Optional[Dict[str, Any]]:
        """Get the projected state for a specific aggregate."""
        return self._state.get(aggregate_id)

    async def apply(self, event: ProjectedEvent) -> None:
        if event.aggregate_type != self._aggregate_type:
            return

        aggregate_id = event.aggregate_id
        if aggregate_id not in self._state:
            self._state[aggregate_id] = {}

        await self.apply_to_aggregate(aggregate_id, event)

    @abstractmethod
    async def apply_to_aggregate(
        self,
        aggregate_id: str,
        event: ProjectedEvent,
    ) -> None:
        """Apply event to a specific aggregate's state."""
        ...


class CountingProjection(ProjectionBase[Dict[str, int]]):
    """
    Projection that counts events by a grouping key.

    Useful for dashboards, analytics, and monitoring.
    """

    def __init__(
        self,
        name: str,
        key_extractor: Callable[[ProjectedEvent], Optional[str]],
        event_types: Optional[Set[str]] = None,
    ):
        super().__init__(name, event_types=event_types)
        self._key_extractor = key_extractor

    def initial_state(self) -> Dict[str, int]:
        return {}

    async def apply(self, event: ProjectedEvent) -> None:
        key = self._key_extractor(event)
        if key is not None:
            self._state.setdefault(key, 0)
            self._state[key] += 1

    def get_count(self, key: str) -> int:
        """Get the count for a specific key."""
        return self._state.get(key, 0)

    def get_total(self) -> int:
        """Get the total count across all keys."""
        return sum(self._state.values())

    def get_top(self, n: int = 10) -> List[Tuple[str, int]]:
        """Get the top N keys by count."""
        return sorted(
            self._state.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:n]


class TimeSeriesProjection(ProjectionBase[Dict[str, List[Tuple[float, Any]]]]):
    """
    Projection that maintains time-series data.

    Useful for tracking metrics over time.
    """

    def __init__(
        self,
        name: str,
        series_key_extractor: Callable[[ProjectedEvent], Optional[str]],
        value_extractor: Callable[[ProjectedEvent], Any],
        max_points: int = 10000,
    ):
        super().__init__(name)
        self._series_key_extractor = series_key_extractor
        self._value_extractor = value_extractor
        self._max_points = max_points

    def initial_state(self) -> Dict[str, List[Tuple[float, Any]]]:
        return {}

    async def apply(self, event: ProjectedEvent) -> None:
        series_key = self._series_key_extractor(event)
        if series_key is None:
            return

        value = self._value_extractor(event)
        timestamp = event.timestamp

        if series_key not in self._state:
            self._state[series_key] = []

        series = self._state[series_key]
        series.append((timestamp, value))

        # Trim to max points
        if len(series) > self._max_points:
            self._state[series_key] = series[-self._max_points:]

    def get_series(
        self,
        key: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[Tuple[float, Any]]:
        """Get time series data with optional time filtering."""
        series = self._state.get(key, [])

        if start_time is not None:
            series = [(t, v) for t, v in series if t >= start_time]
        if end_time is not None:
            series = [(t, v) for t, v in series if t <= end_time]

        return series


class CrossReferenceGraphProjection(ProjectionBase[Dict[str, Set[str]]]):
    """
    Projection that builds a graph of cross-references.

    Maintains adjacency lists for efficient graph traversal queries.
    """

    def __init__(self):
        super().__init__(
            "CrossReferenceGraph",
            event_types={"CrossReferenceCreated", "CrossReferenceVerified", "CrossReferenceDiscovered"},
        )
        self._reverse_index: Dict[str, Set[str]] = {}

    def initial_state(self) -> Dict[str, Set[str]]:
        return {}

    async def reset(self) -> None:
        await super().reset()
        self._reverse_index.clear()

    async def apply(self, event: ProjectedEvent) -> None:
        source = event.data.get("source_ref")
        target = event.data.get("target_ref")

        if not source or not target:
            return

        # Forward index: source -> targets
        if source not in self._state:
            self._state[source] = set()
        self._state[source].add(target)

        # Reverse index: target -> sources
        if target not in self._reverse_index:
            self._reverse_index[target] = set()
        self._reverse_index[target].add(source)

    def get_outgoing(self, verse_ref: str) -> Set[str]:
        """Get all verses this verse references."""
        return self._state.get(verse_ref, set()).copy()

    def get_incoming(self, verse_ref: str) -> Set[str]:
        """Get all verses that reference this verse."""
        return self._reverse_index.get(verse_ref, set()).copy()

    def get_connected(self, verse_ref: str) -> Set[str]:
        """Get all verses connected to this verse (in or out)."""
        return self.get_outgoing(verse_ref) | self.get_incoming(verse_ref)

    def get_degree(self, verse_ref: str) -> Tuple[int, int]:
        """Get the in-degree and out-degree for a verse."""
        return (
            len(self.get_incoming(verse_ref)),
            len(self.get_outgoing(verse_ref)),
        )


class VerseProcessingStatusProjection(ProjectionBase[Dict[str, Dict[str, Any]]]):
    """
    Projection that tracks verse processing status.

    Provides a dashboard view of which verses have been processed
    and their current status.
    """

    def __init__(self):
        super().__init__(
            "VerseProcessingStatus",
            event_types={
                "VerseCreated",
                "VerseProcessingStarted",
                "VerseProcessingCompleted",
                "GoldenRecordCertified",
            },
        )

    def initial_state(self) -> Dict[str, Dict[str, Any]]:
        return {}

    async def apply(self, event: ProjectedEvent) -> None:
        verse_ref = event.data.get("verse_reference") or event.aggregate_id

        if verse_ref not in self._state:
            self._state[verse_ref] = {
                "status": "unknown",
                "created_at": None,
                "processed_at": None,
                "certified_at": None,
                "quality_tier": None,
            }

        status = self._state[verse_ref]

        if event.event_type == "VerseCreated":
            status["status"] = "created"
            status["created_at"] = event.timestamp
        elif event.event_type == "VerseProcessingStarted":
            status["status"] = "processing"
        elif event.event_type == "VerseProcessingCompleted":
            status["status"] = "processed"
            status["processed_at"] = event.timestamp
        elif event.event_type == "GoldenRecordCertified":
            status["status"] = "certified"
            status["certified_at"] = event.timestamp
            status["quality_tier"] = event.data.get("quality_tier")

    def get_by_status(self, status: str) -> List[str]:
        """Get all verses with a specific status."""
        return [
            ref for ref, data in self._state.items()
            if data["status"] == status
        ]

    def get_pending_count(self) -> int:
        """Get count of verses awaiting processing."""
        return len(self.get_by_status("created"))

    def get_certified_count(self) -> int:
        """Get count of certified verses."""
        return len(self.get_by_status("certified"))


# =============================================================================
# PROJECTION MANAGER
# =============================================================================


class ProjectionManager:
    """
    Manages multiple projections and their lifecycle.

    Responsibilities:
        - Register and track projections
        - Start/stop projection processing
        - Coordinate checkpoint persistence
        - Handle rebuilds
        - Monitor projection health

    Usage:
        manager = ProjectionManager(event_store, checkpoint_store)
        manager.register(VerseCountProjection())
        manager.register(CrossReferenceGraphProjection())

        await manager.start_all()
        # ... application runs ...
        await manager.stop_all()
    """

    def __init__(
        self,
        event_store: Any,  # IEventStore
        checkpoint_store: ICheckpointStore,
        batch_size: int = 100,
        checkpoint_strategy: CheckpointStrategy = CheckpointStrategy.AFTER_BATCH,
        checkpoint_interval_events: int = 100,
        checkpoint_interval_seconds: float = 30.0,
    ):
        self._event_store = event_store
        self._checkpoint_store = checkpoint_store
        self._batch_size = batch_size
        self._checkpoint_strategy = checkpoint_strategy
        self._checkpoint_interval_events = checkpoint_interval_events
        self._checkpoint_interval_seconds = checkpoint_interval_seconds

        self._projections: Dict[str, IProjection] = {}
        self._statuses: Dict[str, ProjectionStatus] = {}
        self._positions: Dict[str, int] = {}
        self._tasks: Dict[str, asyncio.Task[None]] = {}
        self._cancellation: Dict[str, asyncio.Event] = {}

    def register(self, projection: IProjection) -> "ProjectionManager":
        """Register a projection."""
        if projection.name in self._projections:
            raise ValueError(f"Projection {projection.name} already registered")

        self._projections[projection.name] = projection
        self._statuses[projection.name] = ProjectionStatus.STOPPED
        self._positions[projection.name] = 0
        self._cancellation[projection.name] = asyncio.Event()

        logger.info(f"Registered projection: {projection.name}")
        return self

    def unregister(self, name: str) -> None:
        """Unregister a projection."""
        if name in self._tasks and not self._tasks[name].done():
            raise RuntimeError(f"Cannot unregister running projection: {name}")

        self._projections.pop(name, None)
        self._statuses.pop(name, None)
        self._positions.pop(name, None)
        self._tasks.pop(name, None)
        self._cancellation.pop(name, None)

    async def start(self, name: str) -> None:
        """Start a specific projection."""
        if name not in self._projections:
            raise ValueError(f"Unknown projection: {name}")

        if self._statuses[name] not in (ProjectionStatus.STOPPED, ProjectionStatus.FAULTED):
            return

        projection = self._projections[name]
        cancellation = self._cancellation[name]
        cancellation.clear()

        self._statuses[name] = ProjectionStatus.STARTING

        # Load checkpoint
        checkpoint = await self._checkpoint_store.get_checkpoint(name)
        start_position = checkpoint.global_position if checkpoint else 0
        self._positions[name] = start_position

        # Create processing task
        self._tasks[name] = asyncio.create_task(
            self._run_projection(projection, start_position, cancellation),
            name=f"projection:{name}",
        )

        logger.info(f"Started projection {name} from position {start_position}")

    async def stop(self, name: str) -> None:
        """Stop a specific projection."""
        if name not in self._projections:
            raise ValueError(f"Unknown projection: {name}")

        cancellation = self._cancellation.get(name)
        if cancellation:
            cancellation.set()

        task = self._tasks.get(name)
        if task and not task.done():
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except asyncio.TimeoutError:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._statuses[name] = ProjectionStatus.STOPPED
        logger.info(f"Stopped projection {name}")

    async def start_all(self) -> None:
        """Start all registered projections."""
        for name in self._projections:
            await self.start(name)

    async def stop_all(self) -> None:
        """Stop all running projections."""
        tasks = []
        for name in self._projections:
            tasks.append(self.stop(name))
        await asyncio.gather(*tasks, return_exceptions=True)

    async def rebuild(self, name: str) -> None:
        """Rebuild a projection from the beginning."""
        if name not in self._projections:
            raise ValueError(f"Unknown projection: {name}")

        # Stop if running
        await self.stop(name)

        self._statuses[name] = ProjectionStatus.REBUILDING

        # Reset projection state
        projection = self._projections[name]
        await projection.reset()

        # Delete checkpoint
        await self._checkpoint_store.delete_checkpoint(name)
        self._positions[name] = 0

        # Restart from beginning
        await self.start(name)
        logger.info(f"Rebuilding projection {name} from position 0")

    async def rebuild_all(self) -> None:
        """Rebuild all projections from the beginning."""
        for name in self._projections:
            await self.rebuild(name)

    async def _run_projection(
        self,
        projection: IProjection,
        start_position: int,
        cancellation: asyncio.Event,
    ) -> None:
        """Run the projection processing loop."""
        position = start_position
        events_since_checkpoint = 0
        last_checkpoint_time = time.time()

        self._statuses[projection.name] = ProjectionStatus.CATCHING_UP

        try:
            while not cancellation.is_set():
                # Fetch batch of events
                events = await self._fetch_events(
                    projection,
                    position,
                    self._batch_size,
                )

                if not events:
                    # Caught up - switch to live mode
                    self._statuses[projection.name] = ProjectionStatus.LIVE
                    # Wait for new events
                    try:
                        await asyncio.wait_for(
                            cancellation.wait(),
                            timeout=1.0,
                        )
                        break  # Cancellation requested
                    except asyncio.TimeoutError:
                        continue

                # Process events
                for event in events:
                    if cancellation.is_set():
                        break

                    await projection.apply(event)
                    position = event.global_position + 1
                    self._positions[projection.name] = position
                    events_since_checkpoint += 1

                    # Checkpoint after each event if configured
                    if self._checkpoint_strategy == CheckpointStrategy.AFTER_EACH_EVENT:
                        await self._save_checkpoint(projection.name, position, event)

                # Checkpoint after batch if configured
                if self._checkpoint_strategy == CheckpointStrategy.AFTER_BATCH:
                    await self._save_checkpoint(projection.name, position, events[-1])
                    events_since_checkpoint = 0

                # Periodic checkpointing
                elif self._checkpoint_strategy == CheckpointStrategy.PERIODIC:
                    current_time = time.time()
                    should_checkpoint = (
                        events_since_checkpoint >= self._checkpoint_interval_events or
                        current_time - last_checkpoint_time >= self._checkpoint_interval_seconds
                    )
                    if should_checkpoint:
                        await self._save_checkpoint(projection.name, position, events[-1])
                        events_since_checkpoint = 0
                        last_checkpoint_time = current_time

        except Exception as e:
            logger.error(f"Projection {projection.name} faulted: {e}")
            self._statuses[projection.name] = ProjectionStatus.FAULTED
            raise

    async def _fetch_events(
        self,
        projection: IProjection,
        from_position: int,
        limit: int,
    ) -> List[ProjectedEvent]:
        """Fetch events for a projection."""
        # This would integrate with the event store
        # For now, return empty list
        try:
            events = await self._event_store.get_events_from_position(
                from_position,
                limit,
            )
            return [self._to_projected_event(e) for e in events]
        except AttributeError:
            # Event store doesn't have the method yet
            return []

    def _to_projected_event(self, stored_event: Any) -> ProjectedEvent:
        """Convert stored event to projected event."""
        return ProjectedEvent(
            event_id=stored_event.event_id,
            event_type=stored_event.event_type,
            aggregate_id=stored_event.aggregate_id,
            aggregate_type=stored_event.aggregate_type,
            stream_name=f"{stored_event.aggregate_type}-{stored_event.aggregate_id}",
            stream_position=stored_event.stream_position,
            global_position=stored_event.global_position,
            data=stored_event.data,
            metadata=stored_event.metadata,
            timestamp=stored_event.timestamp,
            schema_version=stored_event.schema_version,
        )

    async def _save_checkpoint(
        self,
        projection_name: str,
        position: int,
        last_event: ProjectedEvent,
    ) -> None:
        """Save a checkpoint for a projection."""
        checkpoint = Checkpoint(
            projection_name=projection_name,
            stream_name="$all",
            position=last_event.stream_position,
            global_position=position,
            timestamp=time.time(),
            metadata={"last_event_id": str(last_event.event_id)},
        )
        await self._checkpoint_store.save_checkpoint(checkpoint)

    def get_status(self, name: str) -> ProjectionStatus:
        """Get the status of a projection."""
        return self._statuses.get(name, ProjectionStatus.STOPPED)

    def get_position(self, name: str) -> int:
        """Get the current position of a projection."""
        return self._positions.get(name, 0)

    def get_all_stats(self) -> Dict[str, ProjectionStats]:
        """Get statistics for all projections."""
        stats = {}
        for name, projection in self._projections.items():
            if isinstance(projection, ProjectionBase):
                stats[name] = projection.get_stats(
                    self._statuses[name],
                    self._positions[name],
                )
        return stats

    def get_projection(self, name: str) -> Optional[IProjection]:
        """Get a projection by name."""
        return self._projections.get(name)


# =============================================================================
# PROJECTION BUILDER
# =============================================================================


class ProjectionBuilder:
    """
    Fluent builder for creating projections.

    Usage:
        projection = (
            ProjectionBuilder("VerseCounts")
            .filter_events({"VerseCreated"})
            .with_handler(lambda e: ...)
            .with_state_type(dict)
            .build()
        )
    """

    def __init__(self, name: str):
        self._name = name
        self._stream_filter: Optional[str] = None
        self._event_types: Optional[Set[str]] = None
        self._handler: Optional[Callable[[ProjectedEvent, Any], Awaitable[None]]] = None
        self._initial_state_factory: Callable[[], Any] = dict

    def filter_stream(self, prefix: str) -> "ProjectionBuilder":
        """Filter by stream name prefix."""
        self._stream_filter = prefix
        return self

    def filter_events(self, event_types: Set[str]) -> "ProjectionBuilder":
        """Filter by event types."""
        self._event_types = event_types
        return self

    def with_handler(
        self,
        handler: Callable[[ProjectedEvent, Any], Awaitable[None]],
    ) -> "ProjectionBuilder":
        """Set the event handler."""
        self._handler = handler
        return self

    def with_initial_state(self, factory: Callable[[], Any]) -> "ProjectionBuilder":
        """Set the initial state factory."""
        self._initial_state_factory = factory
        return self

    def build(self) -> IProjection:
        """Build the projection."""
        if self._handler is None:
            raise ValueError("Handler is required")

        builder_name = self._name
        builder_stream_filter = self._stream_filter
        builder_event_types = self._event_types
        builder_handler = self._handler
        builder_initial_state = self._initial_state_factory

        class DynamicProjection(ProjectionBase[Any]):
            def __init__(self) -> None:
                super().__init__(
                    builder_name,
                    stream_filter=builder_stream_filter,
                    event_types=builder_event_types,
                )

            def initial_state(self) -> Any:
                return builder_initial_state()

            async def apply(self, event: ProjectedEvent) -> None:
                await builder_handler(event, self._state)

        return DynamicProjection()


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Event Types
    "ProjectedEvent",
    # Checkpoint
    "CheckpointStrategy",
    "Checkpoint",
    "ICheckpointStore",
    "InMemoryCheckpointStore",
    "PostgresCheckpointStore",
    # Projection
    "ProjectionStatus",
    "ProjectionStats",
    "IProjection",
    "ProjectionBase",
    # Specialized Projections
    "AggregateProjection",
    "CountingProjection",
    "TimeSeriesProjection",
    "CrossReferenceGraphProjection",
    "VerseProcessingStatusProjection",
    # Manager
    "ProjectionManager",
    # Builder
    "ProjectionBuilder",
]
