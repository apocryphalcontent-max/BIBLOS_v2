"""
CQRS: Read Model Projections

Projections build denormalized read models from events.
They enable efficient querying without replaying the entire event stream.
"""
import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from uuid import UUID
import asyncpg

from db.events import (
    BaseEvent,
    EventType,
    VerseProcessingStarted,
    VerseProcessingCompleted,
    VerseProcessingFailed,
    CrossReferenceDiscovered,
    CrossReferenceValidated,
    CrossReferenceRejected,
)
from db.event_store import EventStore


logger = logging.getLogger(__name__)


class ProjectionBase:
    """
    Base class for event-sourced projections.

    Projections listen to events and build denormalized read models
    optimized for specific query patterns.
    """

    def __init__(self, db_pool: asyncpg.Pool, event_store: EventStore):
        """
        Initialize projection.

        Args:
            db_pool: Database connection pool
            event_store: Event store to subscribe to
        """
        self.db_pool = db_pool
        self.event_store = event_store
        self.projection_name = self.__class__.__name__
        self._last_position = 0
        self._is_running = False

    async def initialize(self) -> None:
        """
        Initialize projection schema.

        Subclasses must implement this to create tables/indexes.
        """
        raise NotImplementedError()

    async def handle_event(self, event: BaseEvent) -> None:
        """
        Handle a single event.

        Subclasses implement this to update the read model.
        """
        raise NotImplementedError()

    async def rebuild(self, from_position: int = 0) -> None:
        """
        Rebuild projection from event stream.

        Args:
            from_position: Starting position in event stream
        """
        logger.info(f"Rebuilding {self.projection_name} from position {from_position}")

        # Clear existing data
        await self._clear_projection()

        # Process all events
        async for event in self.event_store.stream_events(from_position):
            try:
                await self.handle_event(event)
                self._last_position += 1

                if self._last_position % 1000 == 0:
                    logger.info(
                        f"{self.projection_name}: processed {self._last_position} events"
                    )
            except Exception as e:
                logger.error(
                    f"Error handling event in {self.projection_name}: {e}",
                    exc_info=True
                )

        logger.info(f"{self.projection_name} rebuild complete")

    async def start(self) -> None:
        """Start projection subscription to live events."""
        self._is_running = True
        logger.info(f"Started {self.projection_name} subscription")

    async def stop(self) -> None:
        """Stop projection subscription."""
        self._is_running = False
        logger.info(f"Stopped {self.projection_name} subscription")

    async def _clear_projection(self) -> None:
        """Clear projection data. Subclasses should override."""
        pass


class VerseStatusProjection(ProjectionBase):
    """
    Projection tracking the current status of verse processing.

    Provides fast lookup of:
    - Which verses have been processed
    - Which phases completed for each verse
    - Processing duration and quality tier
    - Latest errors/failures
    """

    async def initialize(self) -> None:
        """Create verse_status table."""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS verse_status (
                    verse_id VARCHAR(50) PRIMARY KEY,
                    status VARCHAR(20) NOT NULL,
                    phases_completed TEXT[],
                    total_duration_ms FLOAT,
                    cross_reference_count INTEGER DEFAULT 0,
                    quality_tier VARCHAR(20),
                    last_error TEXT,
                    last_error_phase VARCHAR(50),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_verse_status_status
                ON verse_status(status)
            """)

    async def handle_event(self, event: BaseEvent) -> None:
        """Update verse status based on event."""
        if isinstance(event, VerseProcessingStarted):
            await self._handle_processing_started(event)
        elif isinstance(event, VerseProcessingCompleted):
            await self._handle_processing_completed(event)
        elif isinstance(event, VerseProcessingFailed):
            await self._handle_processing_failed(event)

    async def _handle_processing_started(self, event: VerseProcessingStarted) -> None:
        """Mark verse as in progress."""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO verse_status (verse_id, status, phases_completed, updated_at)
                VALUES ($1, 'processing', ARRAY[]::text[], $2)
                ON CONFLICT (verse_id)
                DO UPDATE SET
                    status = 'processing',
                    phases_completed = ARRAY[]::text[],
                    updated_at = $2
            """, event.verse_id, event.timestamp)

    async def _handle_processing_completed(
        self,
        event: VerseProcessingCompleted
    ) -> None:
        """Mark verse as completed."""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO verse_status (
                    verse_id, status, phases_completed, total_duration_ms,
                    cross_reference_count, quality_tier, updated_at
                )
                VALUES ($1, 'completed', $2, $3, $4, $5, $6)
                ON CONFLICT (verse_id)
                DO UPDATE SET
                    status = 'completed',
                    phases_completed = $2,
                    total_duration_ms = $3,
                    cross_reference_count = $4,
                    quality_tier = $5,
                    updated_at = $6
            """,
                event.verse_id,
                event.phases_completed,
                event.total_duration_ms,
                event.cross_reference_count,
                event.quality_tier,
                event.timestamp
            )

    async def _handle_processing_failed(self, event: VerseProcessingFailed) -> None:
        """Mark verse as failed."""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO verse_status (
                    verse_id, status, last_error, last_error_phase, updated_at
                )
                VALUES ($1, 'failed', $2, $3, $4)
                ON CONFLICT (verse_id)
                DO UPDATE SET
                    status = 'failed',
                    last_error = $2,
                    last_error_phase = $3,
                    updated_at = $4
            """,
                event.verse_id,
                event.error_message,
                event.failed_phase,
                event.timestamp
            )

    async def _clear_projection(self) -> None:
        """Clear verse status table."""
        async with self.db_pool.acquire() as conn:
            await conn.execute("TRUNCATE TABLE verse_status")


class CrossReferenceProjection(ProjectionBase):
    """
    Projection tracking validated cross-references.

    Provides fast lookup of:
    - All cross-references for a verse
    - Cross-references by type
    - Rejected references with reasons
    - Confidence scores and theological validation
    """

    async def initialize(self) -> None:
        """Create cross_references table."""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS cross_references (
                    id SERIAL PRIMARY KEY,
                    source_ref VARCHAR(50) NOT NULL,
                    target_ref VARCHAR(50) NOT NULL,
                    connection_type VARCHAR(50) NOT NULL,
                    status VARCHAR(20) NOT NULL DEFAULT 'discovered',
                    confidence FLOAT NOT NULL,
                    theological_score FLOAT,
                    discovered_by VARCHAR(100),
                    validators TEXT[],
                    rejection_reason TEXT,
                    violated_constraints TEXT[],
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE(source_ref, target_ref, connection_type)
                )
            """)

            # Indexes for efficient querying
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_crossref_source
                ON cross_references(source_ref)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_crossref_target
                ON cross_references(target_ref)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_crossref_type
                ON cross_references(connection_type)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_crossref_status
                ON cross_references(status)
            """)

    async def handle_event(self, event: BaseEvent) -> None:
        """Update cross-references based on event."""
        if isinstance(event, CrossReferenceDiscovered):
            await self._handle_discovered(event)
        elif isinstance(event, CrossReferenceValidated):
            await self._handle_validated(event)
        elif isinstance(event, CrossReferenceRejected):
            await self._handle_rejected(event)

    async def _handle_discovered(self, event: CrossReferenceDiscovered) -> None:
        """Record discovered cross-reference."""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO cross_references (
                    source_ref, target_ref, connection_type, status,
                    confidence, discovered_by, created_at, updated_at
                )
                VALUES ($1, $2, $3, 'discovered', $4, $5, $6, $6)
                ON CONFLICT (source_ref, target_ref, connection_type)
                DO UPDATE SET
                    status = 'discovered',
                    confidence = $4,
                    discovered_by = $5,
                    updated_at = $6
            """,
                event.source_ref,
                event.target_ref,
                event.connection_type,
                event.confidence,
                event.discovered_by,
                event.timestamp
            )

    async def _handle_validated(self, event: CrossReferenceValidated) -> None:
        """Mark cross-reference as validated."""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE cross_references
                SET
                    status = 'validated',
                    confidence = $4,
                    theological_score = $5,
                    validators = $6,
                    updated_at = $7
                WHERE source_ref = $1 AND target_ref = $2 AND connection_type = $3
            """,
                event.source_ref,
                event.target_ref,
                event.connection_type,
                event.final_confidence,
                event.theological_score,
                event.validators,
                event.timestamp
            )

    async def _handle_rejected(self, event: CrossReferenceRejected) -> None:
        """Mark cross-reference as rejected."""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE cross_references
                SET
                    status = 'rejected',
                    rejection_reason = $4,
                    violated_constraints = $5,
                    updated_at = $6
                WHERE source_ref = $1 AND target_ref = $2 AND connection_type = $3
            """,
                event.source_ref,
                event.target_ref,
                event.connection_type,
                event.rejection_reason,
                event.violated_constraints,
                event.timestamp
            )

    async def _clear_projection(self) -> None:
        """Clear cross-references table."""
        async with self.db_pool.acquire() as conn:
            await conn.execute("TRUNCATE TABLE cross_references")


class ProjectionManager:
    """
    Manages multiple projections and coordinates rebuilds.

    Provides single interface for starting/stopping all projections
    and monitoring their health.
    """

    def __init__(self, db_pool: asyncpg.Pool, event_store: EventStore):
        """
        Initialize projection manager.

        Args:
            db_pool: Database connection pool
            event_store: Event store
        """
        self.db_pool = db_pool
        self.event_store = event_store
        self.projections: List[ProjectionBase] = []

        # Register default projections
        self.register(VerseStatusProjection(db_pool, event_store))
        self.register(CrossReferenceProjection(db_pool, event_store))

    def register(self, projection: ProjectionBase) -> None:
        """Register a projection."""
        self.projections.append(projection)
        logger.info(f"Registered projection: {projection.projection_name}")

    async def initialize_all(self) -> None:
        """Initialize all projections."""
        logger.info("Initializing all projections...")
        for projection in self.projections:
            await projection.initialize()
        logger.info("All projections initialized")

    async def rebuild_all(self, from_position: int = 0) -> None:
        """Rebuild all projections from events."""
        logger.info("Rebuilding all projections...")
        for projection in self.projections:
            await projection.rebuild(from_position)
        logger.info("All projections rebuilt")

    async def start_all(self) -> None:
        """Start all projection subscriptions."""
        logger.info("Starting all projections...")
        for projection in self.projections:
            await projection.start()
        logger.info("All projections started")

    async def stop_all(self) -> None:
        """Stop all projection subscriptions."""
        logger.info("Stopping all projections...")
        for projection in self.projections:
            await projection.stop()
        logger.info("All projections stopped")
