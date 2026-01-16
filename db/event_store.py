"""
Event Sourcing: Event Store Implementation

PostgreSQL-backed event store for persisting all domain events.
Provides append-only event log with optimistic concurrency control.
"""
import json
import logging
from typing import List, Optional, AsyncIterator, Dict, Any
from datetime import datetime
from uuid import UUID
import asyncpg

from db.events import BaseEvent, deserialize_event, EventType


logger = logging.getLogger(__name__)


class ConcurrencyError(Exception):
    """Raised when optimistic concurrency check fails."""
    pass


class EventStore:
    """
    Append-only event store backed by PostgreSQL.

    Features:
    - Optimistic concurrency control via aggregate versioning
    - Correlation ID tracking for distributed tracing
    - Causation ID tracking for command/event chains
    - Efficient querying by aggregate, type, correlation, and time range
    """

    def __init__(self, connection_pool: asyncpg.Pool):
        """
        Initialize event store.

        Args:
            connection_pool: PostgreSQL connection pool
        """
        self.pool = connection_pool
        self._subscription_callbacks: Dict[str, List[callable]] = {}

    async def initialize(self) -> None:
        """
        Initialize database schema.

        Creates events table with indexes for efficient querying.
        """
        async with self.pool.acquire() as conn:
            # Create events table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id UUID PRIMARY KEY,
                    event_type VARCHAR(100) NOT NULL,
                    aggregate_id VARCHAR(200) NOT NULL,
                    aggregate_version INTEGER NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    correlation_id VARCHAR(100),
                    causation_id VARCHAR(100),
                    payload JSONB NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE(aggregate_id, aggregate_version)
                )
            """)

            # Create indexes for efficient querying
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_aggregate
                ON events(aggregate_id, aggregate_version)
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

            # Create event stream view for ordered reading
            await conn.execute("""
                CREATE OR REPLACE VIEW event_stream AS
                SELECT
                    event_id,
                    event_type,
                    aggregate_id,
                    aggregate_version,
                    timestamp,
                    correlation_id,
                    causation_id,
                    payload
                FROM events
                ORDER BY created_at ASC, aggregate_version ASC
            """)

            logger.info("Event store schema initialized successfully")

    async def append(
        self,
        event: BaseEvent,
        expected_version: Optional[int] = None
    ) -> None:
        """
        Append event to the store with optimistic concurrency control.

        Args:
            event: Event to append
            expected_version: Expected current version of aggregate (for concurrency check)

        Raises:
            ConcurrencyError: If expected_version doesn't match actual version
        """
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Check current version if expected_version is provided
                if expected_version is not None:
                    current_version = await conn.fetchval("""
                        SELECT MAX(aggregate_version)
                        FROM events
                        WHERE aggregate_id = $1
                    """, event.aggregate_id)

                    if current_version is None:
                        current_version = 0

                    if current_version != expected_version:
                        raise ConcurrencyError(
                            f"Concurrency conflict for aggregate {event.aggregate_id}: "
                            f"expected version {expected_version}, found {current_version}"
                        )

                # Prepare event data
                event_dict = event.to_dict()
                payload = {k: v for k, v in event_dict.items()
                          if k not in ['event_id', 'event_type', 'aggregate_id',
                                      'aggregate_version', 'timestamp', 'correlation_id',
                                      'causation_id', 'metadata']}

                # Insert event
                await conn.execute("""
                    INSERT INTO events (
                        event_id, event_type, aggregate_id, aggregate_version,
                        timestamp, correlation_id, causation_id, payload, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                    event.event_id,
                    event.event_type,
                    event.aggregate_id,
                    event.aggregate_version,
                    event.timestamp,
                    event.correlation_id,
                    event.causation_id,
                    json.dumps(payload),
                    json.dumps(event.metadata) if event.metadata else None
                )

                logger.debug(
                    f"Appended event {event.event_type} for aggregate {event.aggregate_id} "
                    f"v{event.aggregate_version}"
                )

        # Notify subscribers
        await self._notify_subscribers(event)

    async def append_batch(self, events: List[BaseEvent]) -> None:
        """
        Append multiple events atomically.

        Args:
            events: List of events to append
        """
        if not events:
            return

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                for event in events:
                    event_dict = event.to_dict()
                    payload = {k: v for k, v in event_dict.items()
                              if k not in ['event_id', 'event_type', 'aggregate_id',
                                          'aggregate_version', 'timestamp', 'correlation_id',
                                          'causation_id', 'metadata']}

                    await conn.execute("""
                        INSERT INTO events (
                            event_id, event_type, aggregate_id, aggregate_version,
                            timestamp, correlation_id, causation_id, payload, metadata
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,
                        event.event_id,
                        event.event_type,
                        event.aggregate_id,
                        event.aggregate_version,
                        event.timestamp,
                        event.correlation_id,
                        event.causation_id,
                        json.dumps(payload),
                        json.dumps(event.metadata) if event.metadata else None
                    )

        logger.info(f"Appended batch of {len(events)} events")

        # Notify subscribers
        for event in events:
            await self._notify_subscribers(event)

    async def get_events_by_aggregate(
        self,
        aggregate_id: str,
        from_version: int = 0,
        to_version: Optional[int] = None
    ) -> List[BaseEvent]:
        """
        Get all events for an aggregate.

        Args:
            aggregate_id: Aggregate identifier
            from_version: Starting version (inclusive)
            to_version: Ending version (inclusive), None for all

        Returns:
            List of events in order
        """
        query = """
            SELECT event_type, payload, event_id, aggregate_id, aggregate_version,
                   timestamp, correlation_id, causation_id, metadata
            FROM events
            WHERE aggregate_id = $1 AND aggregate_version >= $2
        """
        params = [aggregate_id, from_version]

        if to_version is not None:
            query += " AND aggregate_version <= $3"
            params.append(to_version)

        query += " ORDER BY aggregate_version ASC"

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        return [self._row_to_event(row) for row in rows]

    async def get_events_by_type(
        self,
        event_type: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[BaseEvent]:
        """
        Get events by type.

        Args:
            event_type: Event type to filter by
            limit: Maximum number of events to return
            offset: Number of events to skip

        Returns:
            List of events
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT event_type, payload, event_id, aggregate_id, aggregate_version,
                       timestamp, correlation_id, causation_id, metadata
                FROM events
                WHERE event_type = $1
                ORDER BY timestamp DESC
                LIMIT $2 OFFSET $3
            """, event_type, limit, offset)

        return [self._row_to_event(row) for row in rows]

    async def get_events_by_correlation(
        self,
        correlation_id: str
    ) -> List[BaseEvent]:
        """
        Get all events with a specific correlation ID.

        Useful for tracing a complete workflow or request.

        Args:
            correlation_id: Correlation identifier

        Returns:
            List of events in chronological order
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT event_type, payload, event_id, aggregate_id, aggregate_version,
                       timestamp, correlation_id, causation_id, metadata
                FROM events
                WHERE correlation_id = $1
                ORDER BY timestamp ASC
            """, correlation_id)

        return [self._row_to_event(row) for row in rows]

    async def get_events_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        event_types: Optional[List[str]] = None,
        limit: int = 1000
    ) -> List[BaseEvent]:
        """
        Get events within a time range.

        Args:
            start_time: Start of time range
            end_time: End of time range
            event_types: Optional list of event types to filter by
            limit: Maximum number of events to return

        Returns:
            List of events in chronological order
        """
        query = """
            SELECT event_type, payload, event_id, aggregate_id, aggregate_version,
                   timestamp, correlation_id, causation_id, metadata
            FROM events
            WHERE timestamp >= $1 AND timestamp <= $2
        """
        params = [start_time, end_time]

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

        return [self._row_to_event(row) for row in rows]

    async def stream_events(
        self,
        from_position: int = 0,
        batch_size: int = 100
    ) -> AsyncIterator[BaseEvent]:
        """
        Stream all events from a position.

        Args:
            from_position: Starting position (row number)
            batch_size: Number of events to fetch per batch

        Yields:
            Events in order
        """
        offset = from_position
        while True:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT event_type, payload, event_id, aggregate_id, aggregate_version,
                           timestamp, correlation_id, causation_id, metadata
                    FROM event_stream
                    LIMIT $1 OFFSET $2
                """, batch_size, offset)

            if not rows:
                break

            for row in rows:
                yield self._row_to_event(row)

            offset += len(rows)

    def subscribe(self, event_type: str, callback: callable) -> None:
        """
        Subscribe to events of a specific type.

        Args:
            event_type: Event type to subscribe to
            callback: Async function to call when event occurs
        """
        if event_type not in self._subscription_callbacks:
            self._subscription_callbacks[event_type] = []
        self._subscription_callbacks[event_type].append(callback)
        logger.info(f"Subscribed to {event_type} events")

    async def _notify_subscribers(self, event: BaseEvent) -> None:
        """Notify subscribers of a new event."""
        callbacks = self._subscription_callbacks.get(event.event_type, [])
        for callback in callbacks:
            try:
                await callback(event)
            except Exception as e:
                logger.error(f"Error in event subscriber: {e}", exc_info=True)

    def _row_to_event(self, row: asyncpg.Record) -> BaseEvent:
        """Convert database row to event object."""
        # Reconstruct full event data
        event_data = {
            'event_id': row['event_id'],
            'event_type': row['event_type'],
            'aggregate_id': row['aggregate_id'],
            'aggregate_version': row['aggregate_version'],
            'timestamp': row['timestamp'],
            'correlation_id': row['correlation_id'],
            'causation_id': row['causation_id'],
            'metadata': row['metadata'] if row['metadata'] else {},
        }

        # Merge payload
        payload = row['payload'] if row['payload'] else {}
        event_data.update(payload)

        return deserialize_event(event_data)

    async def get_aggregate_version(self, aggregate_id: str) -> int:
        """
        Get current version of an aggregate.

        Args:
            aggregate_id: Aggregate identifier

        Returns:
            Current version (0 if aggregate doesn't exist)
        """
        async with self.pool.acquire() as conn:
            version = await conn.fetchval("""
                SELECT MAX(aggregate_version)
                FROM events
                WHERE aggregate_id = $1
            """, aggregate_id)

        return version if version is not None else 0

    async def cleanup(self) -> None:
        """Cleanup resources."""
        # Close connection pool (if managed by event store)
        pass
