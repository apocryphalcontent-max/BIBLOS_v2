"""
BIBLOS v2 - Redis Streams Event Bus

Event-driven message bus using Redis Streams for decoupled agent communication.
Provides at-least-once delivery, consumer groups, and message replay capabilities.
"""
import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import redis.asyncio as aioredis
from redis.asyncio import Redis
from redis.exceptions import ResponseError, ConnectionError as RedisConnectionError

from config import get_config

# Import core error types for specific exception handling
from core.errors import (
    BiblosError,
    BiblosPipelineError,
    BiblosResourceError,
)


logger = logging.getLogger("biblos.pipeline.event_bus")


# =============================================================================
# STREAM TOPICS - Canonical topic definitions
# =============================================================================

class StreamTopic(str, Enum):
    """Stream topics for pipeline events."""
    # Verse ingestion
    VERSE_INGESTED = "stream:verse:ingested"
    VERSE_COMPLETED = "stream:verse:completed"
    VERSE_FAILED = "stream:verse:failed"

    # Phase requests and completions
    PHASE_LINGUISTIC_REQUEST = "stream:phase:linguistic:request"
    PHASE_LINGUISTIC_COMPLETE = "stream:phase:linguistic:complete"
    PHASE_THEOLOGICAL_REQUEST = "stream:phase:theological:request"
    PHASE_THEOLOGICAL_COMPLETE = "stream:phase:theological:complete"
    PHASE_INTERTEXTUAL_REQUEST = "stream:phase:intertextual:request"
    PHASE_INTERTEXTUAL_COMPLETE = "stream:phase:intertextual:complete"
    PHASE_VALIDATION_REQUEST = "stream:phase:validation:request"
    PHASE_VALIDATION_COMPLETE = "stream:phase:validation:complete"
    PHASE_FINALIZATION_REQUEST = "stream:phase:finalization:request"
    PHASE_FINALIZATION_COMPLETE = "stream:phase:finalization:complete"

    # Error handling
    DEAD_LETTER_QUEUE = "stream:dlq"
    RETRY_QUEUE = "stream:retry"

    # Coordination
    CHECKPOINT = "stream:checkpoint"
    HEARTBEAT = "stream:heartbeat"


# Topic mapping for convenience
TOPICS: Dict[str, str] = {topic.name.lower(): topic.value for topic in StreamTopic}


# Phase topic mapping
PHASE_TOPICS = {
    "linguistic": {
        "request": StreamTopic.PHASE_LINGUISTIC_REQUEST,
        "complete": StreamTopic.PHASE_LINGUISTIC_COMPLETE,
    },
    "theological": {
        "request": StreamTopic.PHASE_THEOLOGICAL_REQUEST,
        "complete": StreamTopic.PHASE_THEOLOGICAL_COMPLETE,
    },
    "intertextual": {
        "request": StreamTopic.PHASE_INTERTEXTUAL_REQUEST,
        "complete": StreamTopic.PHASE_INTERTEXTUAL_COMPLETE,
    },
    "validation": {
        "request": StreamTopic.PHASE_VALIDATION_REQUEST,
        "complete": StreamTopic.PHASE_VALIDATION_COMPLETE,
    },
    "finalization": {
        "request": StreamTopic.PHASE_FINALIZATION_REQUEST,
        "complete": StreamTopic.PHASE_FINALIZATION_COMPLETE,
    },
}


# =============================================================================
# MESSAGE TYPES
# =============================================================================

@dataclass
class EventMessage:
    """
    Base event message for the event bus.

    All messages include:
    - message_id: Unique identifier (set by Redis on publish)
    - event_type: Type of event
    - timestamp: When the event occurred
    - correlation_id: For tracing related events
    - payload: Event-specific data
    """
    event_type: str
    payload: Dict[str, Any]
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    message_id: Optional[str] = None  # Set by Redis on publish
    retry_count: int = 0
    source: str = "biblos"
    version: str = "2.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Redis storage."""
        return {
            "event_type": self.event_type,
            "payload": json.dumps(self.payload),
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
            "retry_count": str(self.retry_count),
            "source": self.source,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], message_id: Optional[str] = None) -> "EventMessage":
        """Create from Redis stream entry."""
        payload = data.get("payload", "{}")
        if isinstance(payload, str):
            payload = json.loads(payload)

        return cls(
            event_type=data.get("event_type", "unknown"),
            payload=payload,
            correlation_id=data.get("correlation_id", str(uuid.uuid4())),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            message_id=message_id,
            retry_count=int(data.get("retry_count", "0")),
            source=data.get("source", "biblos"),
            version=data.get("version", "2.0.0"),
        )


@dataclass
class VerseEvent(EventMessage):
    """Event for verse processing."""

    @classmethod
    def create(
        cls,
        verse_id: str,
        text: str,
        event_type: str = "verse_ingested",
        metadata: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> "VerseEvent":
        """Factory method to create a verse event."""
        return cls(
            event_type=event_type,
            payload={
                "verse_id": verse_id,
                "text": text,
                "metadata": metadata or {},
            },
            correlation_id=correlation_id or str(uuid.uuid4()),
        )


@dataclass
class PhaseRequestEvent(EventMessage):
    """Event requesting phase execution."""

    @classmethod
    def create(
        cls,
        phase_name: str,
        verse_id: str,
        text: str,
        context: Dict[str, Any],
        correlation_id: str,
    ) -> "PhaseRequestEvent":
        """Factory method to create a phase request event."""
        return cls(
            event_type=f"phase_{phase_name}_request",
            payload={
                "phase_name": phase_name,
                "verse_id": verse_id,
                "text": text,
                "context": context,
            },
            correlation_id=correlation_id,
        )


@dataclass
class PhaseCompleteEvent(EventMessage):
    """Event signaling phase completion."""

    @classmethod
    def create(
        cls,
        phase_name: str,
        verse_id: str,
        status: str,
        result: Dict[str, Any],
        correlation_id: str,
        error: Optional[str] = None,
    ) -> "PhaseCompleteEvent":
        """Factory method to create a phase completion event."""
        return cls(
            event_type=f"phase_{phase_name}_complete",
            payload={
                "phase_name": phase_name,
                "verse_id": verse_id,
                "status": status,
                "result": result,
                "error": error,
            },
            correlation_id=correlation_id,
        )


@dataclass
class DeadLetterEvent(EventMessage):
    """Event for failed message handling."""

    @classmethod
    def create(
        cls,
        original_message: EventMessage,
        error: str,
        source_topic: str,
        retry_count: int,
    ) -> "DeadLetterEvent":
        """Factory method to create a dead letter event."""
        return cls(
            event_type="dead_letter",
            payload={
                "original_event_type": original_message.event_type,
                "original_payload": original_message.payload,
                "original_message_id": original_message.message_id,
                "error": error,
                "source_topic": source_topic,
                "final_retry_count": retry_count,
            },
            correlation_id=original_message.correlation_id,
        )


# =============================================================================
# EVENT BUS CONFIGURATION
# =============================================================================

@dataclass
class EventBusConfig:
    """Configuration for the event bus."""
    # Redis connection
    redis_url: str = field(default_factory=lambda: get_config().database.redis_url)
    max_connections: int = 20

    # Stream settings
    max_stream_length: int = 100000  # Max entries per stream
    block_timeout_ms: int = 5000  # Blocking read timeout
    batch_size: int = 10  # Messages per read

    # Retry settings
    max_retries: int = 3
    retry_delay_base: float = 1.0  # Base delay in seconds
    retry_delay_max: float = 60.0  # Max delay in seconds

    # Dead letter settings
    enable_dlq: bool = True
    dlq_retention_hours: int = 168  # 7 days

    # Consumer group settings
    consumer_prefix: str = "biblos"
    claim_min_idle_time_ms: int = 60000  # Claim pending after 60s

    # Checkpointing
    checkpoint_interval: int = 100  # Checkpoint every N messages

    # Health check
    heartbeat_interval: int = 30  # Heartbeat every 30s


# =============================================================================
# EVENT BUS IMPLEMENTATION
# =============================================================================

class EventBus:
    """
    Redis Streams-based event bus for BIBLOS pipeline.

    Provides:
    - Publish/subscribe with at-least-once delivery
    - Consumer groups for horizontal scaling
    - Message acknowledgment
    - Dead letter queue for failed messages
    - Message replay capability
    - Automatic retry with exponential backoff
    """

    def __init__(self, config: Optional[EventBusConfig] = None):
        self.config = config or EventBusConfig()
        self.logger = logging.getLogger("biblos.event_bus")
        self._redis: Optional[Redis] = None
        self._initialized = False
        self._consumer_id = f"{self.config.consumer_prefix}-{uuid.uuid4().hex[:8]}"
        self._subscriptions: Dict[str, asyncio.Task] = {}
        self._shutdown_event = asyncio.Event()

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def initialize(self) -> None:
        """Initialize the event bus connection."""
        if self._initialized:
            return

        self.logger.info(f"Initializing event bus with consumer ID: {self._consumer_id}")

        try:
            self._redis = await aioredis.from_url(
                self.config.redis_url,
                max_connections=self.config.max_connections,
                decode_responses=True,
            )
            await self._redis.ping()
            self._initialized = True
            self.logger.info("Event bus initialized successfully")
        except RedisConnectionError as e:
            self.logger.error(f"Redis connection error during initialization: {e}")
            raise BiblosPipelineError(f"Failed to connect to Redis: {e}", error_code="EVT_CONN_001")
        except ResponseError as e:
            self.logger.error(f"Redis response error during initialization: {e}")
            raise BiblosPipelineError(f"Redis error: {e}", error_code="EVT_RESP_001")
        except asyncio.TimeoutError as e:
            self.logger.error("Event bus initialization timed out")
            raise BiblosPipelineError("Redis connection timeout", error_code="EVT_TIMEOUT_001")
        except (MemoryError, BiblosResourceError) as e:
            self.logger.critical(f"Resource exhaustion during event bus initialization: {e}")
            raise
        except BiblosError as e:
            self.logger.error(f"BIBLOS error during event bus initialization: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error initializing event bus: {e} ({type(e).__name__})")
            raise BiblosPipelineError(f"Failed to initialize event bus: {e}", error_code="EVT_INIT_001")

    async def shutdown(self) -> None:
        """Gracefully shutdown the event bus."""
        self.logger.info("Shutting down event bus...")
        self._shutdown_event.set()

        # Cancel all subscriptions
        for topic, task in self._subscriptions.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            self.logger.debug(f"Cancelled subscription to {topic}")

        self._subscriptions.clear()

        # Close Redis connection
        if self._redis:
            await self._redis.close()
            self._redis = None

        self._initialized = False
        self.logger.info("Event bus shutdown complete")

    def _ensure_initialized(self) -> None:
        """Ensure the event bus is initialized."""
        if not self._initialized or not self._redis:
            raise RuntimeError("Event bus not initialized. Call initialize() first.")

    # -------------------------------------------------------------------------
    # Publishing
    # -------------------------------------------------------------------------

    async def publish(
        self,
        topic: Union[str, StreamTopic],
        message: EventMessage,
        max_len: Optional[int] = None,
    ) -> str:
        """
        Publish a message to a stream.

        Args:
            topic: Stream topic (string or StreamTopic enum)
            message: Event message to publish
            max_len: Maximum stream length (trims old messages)

        Returns:
            Message ID assigned by Redis
        """
        self._ensure_initialized()

        stream_name = topic.value if isinstance(topic, StreamTopic) else topic
        max_len = max_len or self.config.max_stream_length

        try:
            message_dict = message.to_dict()
            message_id = await self._redis.xadd(
                stream_name,
                message_dict,
                maxlen=max_len,
                approximate=True,
            )

            self.logger.debug(
                f"Published message {message_id} to {stream_name}: "
                f"type={message.event_type}, correlation={message.correlation_id}"
            )

            return message_id

        except RedisConnectionError as e:
            self.logger.error(f"Redis connection error publishing to {stream_name}: {e}")
            raise BiblosPipelineError(f"Redis connection error: {e}", error_code="EVT_PUB_CONN")
        except ResponseError as e:
            self.logger.error(f"Redis response error publishing to {stream_name}: {e}")
            raise BiblosPipelineError(f"Redis error: {e}", error_code="EVT_PUB_RESP")
        except asyncio.TimeoutError:
            self.logger.error(f"Timeout publishing to {stream_name}")
            raise BiblosPipelineError("Publish timeout", error_code="EVT_PUB_TIMEOUT")
        except (MemoryError, BiblosResourceError) as e:
            self.logger.critical(f"Resource exhaustion publishing to {stream_name}: {e}")
            raise
        except BiblosError as e:
            self.logger.error(f"BIBLOS error publishing to {stream_name}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error publishing to {stream_name}: {e} ({type(e).__name__})")
            raise BiblosPipelineError(f"Failed to publish: {e}", error_code="EVT_PUB_ERR")

    async def publish_batch(
        self,
        topic: Union[str, StreamTopic],
        messages: List[EventMessage],
    ) -> List[str]:
        """
        Publish multiple messages to a stream.

        Args:
            topic: Stream topic
            messages: List of messages to publish

        Returns:
            List of message IDs
        """
        self._ensure_initialized()

        stream_name = topic.value if isinstance(topic, StreamTopic) else topic
        message_ids = []

        async with self._redis.pipeline() as pipe:
            for message in messages:
                pipe.xadd(
                    stream_name,
                    message.to_dict(),
                    maxlen=self.config.max_stream_length,
                    approximate=True,
                )

            results = await pipe.execute()
            message_ids = [r for r in results if isinstance(r, str)]

        self.logger.info(f"Published {len(message_ids)} messages to {stream_name}")
        return message_ids

    # -------------------------------------------------------------------------
    # Consumer Groups
    # -------------------------------------------------------------------------

    async def create_consumer_group(
        self,
        topic: Union[str, StreamTopic],
        group_name: str,
        start_id: str = "0",
        mkstream: bool = True,
    ) -> bool:
        """
        Create a consumer group for a stream.

        Args:
            topic: Stream topic
            group_name: Name of the consumer group
            start_id: Starting message ID ("0" for all, "$" for new only)
            mkstream: Create stream if it doesn't exist

        Returns:
            True if created, False if already exists
        """
        self._ensure_initialized()

        stream_name = topic.value if isinstance(topic, StreamTopic) else topic

        try:
            await self._redis.xgroup_create(
                stream_name,
                group_name,
                id=start_id,
                mkstream=mkstream,
            )
            self.logger.info(f"Created consumer group '{group_name}' for {stream_name}")
            return True

        except ResponseError as e:
            if "BUSYGROUP" in str(e):
                self.logger.debug(f"Consumer group '{group_name}' already exists for {stream_name}")
                return False
            raise

    async def ensure_consumer_group(
        self,
        topic: Union[str, StreamTopic],
        group_name: str,
        start_id: str = "0",
    ) -> None:
        """Ensure a consumer group exists, creating if necessary."""
        await self.create_consumer_group(topic, group_name, start_id, mkstream=True)

    # -------------------------------------------------------------------------
    # Subscribing / Consuming
    # -------------------------------------------------------------------------

    async def subscribe(
        self,
        topic: Union[str, StreamTopic],
        group_name: str,
        consumer_name: Optional[str] = None,
        handler: Optional[Callable[[EventMessage], Any]] = None,
        start_new: bool = True,
    ) -> AsyncIterator[Tuple[str, EventMessage]]:
        """
        Subscribe to a stream using consumer groups.

        Args:
            topic: Stream topic
            group_name: Consumer group name
            consumer_name: Consumer name (defaults to instance consumer ID)
            handler: Optional async handler for each message
            start_new: If True, read new messages; if False, read pending first

        Yields:
            Tuples of (message_id, EventMessage)
        """
        self._ensure_initialized()

        stream_name = topic.value if isinstance(topic, StreamTopic) else topic
        consumer_name = consumer_name or self._consumer_id

        # Ensure consumer group exists
        await self.ensure_consumer_group(topic, group_name)

        # Start reading position (> for new messages, 0 for pending)
        read_id = ">" if start_new else "0"

        self.logger.info(
            f"Subscribing to {stream_name} as {group_name}/{consumer_name}"
        )

        while not self._shutdown_event.is_set():
            try:
                # Read from stream
                messages = await self._redis.xreadgroup(
                    groupname=group_name,
                    consumername=consumer_name,
                    streams={stream_name: read_id},
                    count=self.config.batch_size,
                    block=self.config.block_timeout_ms,
                )

                if not messages:
                    continue

                # Process messages
                for stream, entries in messages:
                    for message_id, data in entries:
                        try:
                            event = EventMessage.from_dict(data, message_id)

                            if handler:
                                await handler(event)

                            yield (message_id, event)

                        except Exception as e:
                            self.logger.error(
                                f"Error processing message {message_id}: {e}"
                            )
                            # Continue to next message

                # Switch to reading new messages after processing pending
                if read_id == "0":
                    read_id = ">"

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in subscription loop: {e}")
                await asyncio.sleep(1)  # Brief pause before retry

    async def read_stream(
        self,
        topic: Union[str, StreamTopic],
        start_id: str = "0",
        end_id: str = "+",
        count: Optional[int] = None,
    ) -> List[Tuple[str, EventMessage]]:
        """
        Read messages from a stream (non-blocking, without consumer group).

        Args:
            topic: Stream topic
            start_id: Starting message ID (exclusive)
            end_id: Ending message ID (inclusive, "+" for latest)
            count: Maximum number of messages to return

        Returns:
            List of (message_id, EventMessage) tuples
        """
        self._ensure_initialized()

        stream_name = topic.value if isinstance(topic, StreamTopic) else topic

        entries = await self._redis.xrange(
            stream_name,
            min=start_id,
            max=end_id,
            count=count,
        )

        return [
            (msg_id, EventMessage.from_dict(data, msg_id))
            for msg_id, data in entries
        ]

    # -------------------------------------------------------------------------
    # Acknowledgment
    # -------------------------------------------------------------------------

    async def acknowledge(
        self,
        topic: Union[str, StreamTopic],
        group_name: str,
        message_id: str,
    ) -> int:
        """
        Acknowledge a message (mark as processed).

        Args:
            topic: Stream topic
            group_name: Consumer group name
            message_id: Message ID to acknowledge

        Returns:
            Number of messages acknowledged (0 or 1)
        """
        self._ensure_initialized()

        stream_name = topic.value if isinstance(topic, StreamTopic) else topic

        count = await self._redis.xack(stream_name, group_name, message_id)

        if count > 0:
            self.logger.debug(f"Acknowledged message {message_id} in {stream_name}")

        return count

    async def acknowledge_batch(
        self,
        topic: Union[str, StreamTopic],
        group_name: str,
        message_ids: List[str],
    ) -> int:
        """
        Acknowledge multiple messages.

        Args:
            topic: Stream topic
            group_name: Consumer group name
            message_ids: List of message IDs to acknowledge

        Returns:
            Number of messages acknowledged
        """
        self._ensure_initialized()

        stream_name = topic.value if isinstance(topic, StreamTopic) else topic

        count = await self._redis.xack(stream_name, group_name, *message_ids)

        self.logger.debug(f"Acknowledged {count} messages in {stream_name}")
        return count

    # -------------------------------------------------------------------------
    # Pending Messages & Recovery
    # -------------------------------------------------------------------------

    async def get_pending(
        self,
        topic: Union[str, StreamTopic],
        group_name: str,
        count: int = 100,
        consumer_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get pending messages for a consumer group.

        Args:
            topic: Stream topic
            group_name: Consumer group name
            count: Maximum number of pending messages
            consumer_name: Filter by consumer (optional)

        Returns:
            List of pending message info
        """
        self._ensure_initialized()

        stream_name = topic.value if isinstance(topic, StreamTopic) else topic

        # Get pending summary first
        pending_info = await self._redis.xpending(stream_name, group_name)

        if pending_info["pending"] == 0:
            return []

        # Get detailed pending list
        if consumer_name:
            pending_range = await self._redis.xpending_range(
                stream_name,
                group_name,
                min="-",
                max="+",
                count=count,
                consumername=consumer_name,
            )
        else:
            pending_range = await self._redis.xpending_range(
                stream_name,
                group_name,
                min="-",
                max="+",
                count=count,
            )

        return [
            {
                "message_id": entry["message_id"],
                "consumer": entry["consumer"],
                "time_since_delivered": entry["time_since_delivered"],
                "times_delivered": entry["times_delivered"],
            }
            for entry in pending_range
        ]

    async def claim_pending(
        self,
        topic: Union[str, StreamTopic],
        group_name: str,
        consumer_name: str,
        min_idle_time_ms: Optional[int] = None,
        count: int = 10,
    ) -> List[Tuple[str, EventMessage]]:
        """
        Claim idle pending messages from other consumers.

        Args:
            topic: Stream topic
            group_name: Consumer group name
            consumer_name: Consumer to claim messages for
            min_idle_time_ms: Minimum idle time before claiming
            count: Maximum messages to claim

        Returns:
            List of claimed (message_id, EventMessage) tuples
        """
        self._ensure_initialized()

        stream_name = topic.value if isinstance(topic, StreamTopic) else topic
        min_idle = min_idle_time_ms or self.config.claim_min_idle_time_ms

        # Get pending messages
        pending = await self.get_pending(topic, group_name, count)

        if not pending:
            return []

        # Filter by idle time
        claimable = [
            p["message_id"]
            for p in pending
            if p["time_since_delivered"] >= min_idle
        ]

        if not claimable:
            return []

        # Claim the messages
        claimed = await self._redis.xclaim(
            stream_name,
            group_name,
            consumer_name,
            min_idle_time=min_idle,
            message_ids=claimable,
        )

        result = [
            (msg_id, EventMessage.from_dict(data, msg_id))
            for msg_id, data in claimed
        ]

        if result:
            self.logger.info(
                f"Claimed {len(result)} pending messages for {consumer_name}"
            )

        return result

    # -------------------------------------------------------------------------
    # Replay
    # -------------------------------------------------------------------------

    async def replay(
        self,
        topic: Union[str, StreamTopic],
        start_id: str = "0",
        end_id: str = "+",
        handler: Optional[Callable[[EventMessage], Any]] = None,
    ) -> AsyncIterator[Tuple[str, EventMessage]]:
        """
        Replay messages from a specific point in the stream.

        Args:
            topic: Stream topic
            start_id: Starting message ID (exclusive)
            end_id: Ending message ID (inclusive)
            handler: Optional async handler for each message

        Yields:
            Tuples of (message_id, EventMessage)
        """
        self._ensure_initialized()

        stream_name = topic.value if isinstance(topic, StreamTopic) else topic

        self.logger.info(f"Replaying {stream_name} from {start_id} to {end_id}")

        current_id = start_id

        while not self._shutdown_event.is_set():
            # Read a batch
            entries = await self._redis.xrange(
                stream_name,
                min=f"({current_id}" if current_id != "0" else "-",
                max=end_id,
                count=self.config.batch_size,
            )

            if not entries:
                break

            for message_id, data in entries:
                event = EventMessage.from_dict(data, message_id)

                if handler:
                    await handler(event)

                yield (message_id, event)
                current_id = message_id

        self.logger.info(f"Replay complete for {stream_name}")

    async def replay_from_checkpoint(
        self,
        topic: Union[str, StreamTopic],
        checkpoint_key: str,
    ) -> AsyncIterator[Tuple[str, EventMessage]]:
        """
        Replay from a saved checkpoint.

        Args:
            topic: Stream topic
            checkpoint_key: Redis key where checkpoint is stored

        Yields:
            Tuples of (message_id, EventMessage)
        """
        self._ensure_initialized()

        # Get checkpoint
        start_id = await self._redis.get(checkpoint_key) or "0"

        self.logger.info(f"Resuming replay from checkpoint: {start_id}")

        async for entry in self.replay(topic, start_id):
            yield entry

    async def save_checkpoint(
        self,
        checkpoint_key: str,
        message_id: str,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """
        Save a checkpoint for replay recovery.

        Args:
            checkpoint_key: Redis key to store checkpoint
            message_id: Last processed message ID
            ttl_seconds: Optional TTL for checkpoint
        """
        self._ensure_initialized()

        if ttl_seconds:
            await self._redis.setex(checkpoint_key, ttl_seconds, message_id)
        else:
            await self._redis.set(checkpoint_key, message_id)

        self.logger.debug(f"Saved checkpoint {checkpoint_key}: {message_id}")

    # -------------------------------------------------------------------------
    # Dead Letter Queue
    # -------------------------------------------------------------------------

    async def send_to_dlq(
        self,
        original_message: EventMessage,
        error: str,
        source_topic: Union[str, StreamTopic],
    ) -> str:
        """
        Send a failed message to the dead letter queue.

        Args:
            original_message: The failed message
            error: Error description
            source_topic: Original stream topic

        Returns:
            DLQ message ID
        """
        if not self.config.enable_dlq:
            self.logger.warning("DLQ disabled, dropping failed message")
            return ""

        source_name = source_topic.value if isinstance(source_topic, StreamTopic) else source_topic

        dlq_event = DeadLetterEvent.create(
            original_message=original_message,
            error=error,
            source_topic=source_name,
            retry_count=original_message.retry_count,
        )

        message_id = await self.publish(StreamTopic.DEAD_LETTER_QUEUE, dlq_event)

        self.logger.warning(
            f"Sent message to DLQ: {message_id}, "
            f"original_id={original_message.message_id}, error={error}"
        )

        return message_id

    async def get_dlq_messages(
        self,
        count: int = 100,
        start_id: str = "-",
        end_id: str = "+",
    ) -> List[Tuple[str, EventMessage]]:
        """
        Get messages from the dead letter queue.

        Args:
            count: Maximum messages to retrieve
            start_id: Starting ID
            end_id: Ending ID

        Returns:
            List of DLQ messages
        """
        return await self.read_stream(
            StreamTopic.DEAD_LETTER_QUEUE,
            start_id=start_id,
            end_id=end_id,
            count=count,
        )

    async def reprocess_dlq_message(
        self,
        dlq_message_id: str,
        target_topic: Union[str, StreamTopic],
    ) -> Optional[str]:
        """
        Reprocess a message from the dead letter queue.

        Args:
            dlq_message_id: DLQ message ID
            target_topic: Topic to republish to

        Returns:
            New message ID if successful
        """
        self._ensure_initialized()

        # Read the DLQ message
        entries = await self._redis.xrange(
            StreamTopic.DEAD_LETTER_QUEUE.value,
            min=dlq_message_id,
            max=dlq_message_id,
        )

        if not entries:
            self.logger.warning(f"DLQ message not found: {dlq_message_id}")
            return None

        _, data = entries[0]
        dlq_event = EventMessage.from_dict(data, dlq_message_id)

        # Reconstruct original message
        payload = dlq_event.payload
        original_event = EventMessage(
            event_type=payload.get("original_event_type", "unknown"),
            payload=payload.get("original_payload", {}),
            correlation_id=dlq_event.correlation_id,
            retry_count=0,  # Reset retry count
        )

        # Republish
        new_id = await self.publish(target_topic, original_event)

        self.logger.info(
            f"Reprocessed DLQ message {dlq_message_id} -> {new_id} on {target_topic}"
        )

        return new_id

    # -------------------------------------------------------------------------
    # Retry Logic
    # -------------------------------------------------------------------------

    def calculate_retry_delay(self, retry_count: int) -> float:
        """
        Calculate retry delay with exponential backoff and jitter.

        Args:
            retry_count: Current retry attempt number

        Returns:
            Delay in seconds
        """
        import random

        # Exponential backoff
        delay = self.config.retry_delay_base * (2 ** retry_count)

        # Add jitter (0.5 to 1.5 multiplier)
        jitter = 0.5 + random.random()
        delay *= jitter

        # Cap at max delay
        return min(delay, self.config.retry_delay_max)

    async def schedule_retry(
        self,
        message: EventMessage,
        target_topic: Union[str, StreamTopic],
        error: str,
    ) -> Optional[str]:
        """
        Schedule a message for retry.

        Args:
            message: Message to retry
            target_topic: Topic to retry on
            error: Error that caused the failure

        Returns:
            Retry message ID, or None if max retries exceeded
        """
        if message.retry_count >= self.config.max_retries:
            self.logger.warning(
                f"Max retries ({self.config.max_retries}) exceeded for "
                f"message {message.message_id}"
            )
            # Send to DLQ
            await self.send_to_dlq(message, error, target_topic)
            return None

        # Calculate delay
        delay = self.calculate_retry_delay(message.retry_count)

        self.logger.info(
            f"Scheduling retry {message.retry_count + 1}/{self.config.max_retries} "
            f"for message {message.message_id} after {delay:.1f}s"
        )

        # Wait for delay
        await asyncio.sleep(delay)

        # Create retry message with incremented count
        retry_message = EventMessage(
            event_type=message.event_type,
            payload=message.payload,
            correlation_id=message.correlation_id,
            retry_count=message.retry_count + 1,
            source=message.source,
            version=message.version,
        )

        # Republish
        return await self.publish(target_topic, retry_message)

    # -------------------------------------------------------------------------
    # Stream Management
    # -------------------------------------------------------------------------

    async def get_stream_info(
        self,
        topic: Union[str, StreamTopic],
    ) -> Dict[str, Any]:
        """
        Get information about a stream.

        Args:
            topic: Stream topic

        Returns:
            Stream info dictionary
        """
        self._ensure_initialized()

        stream_name = topic.value if isinstance(topic, StreamTopic) else topic

        try:
            info = await self._redis.xinfo_stream(stream_name)
            return {
                "length": info["length"],
                "first_entry": info.get("first-entry"),
                "last_entry": info.get("last-entry"),
                "groups": info.get("groups", 0),
            }
        except ResponseError:
            return {"length": 0, "exists": False}

    async def get_consumer_group_info(
        self,
        topic: Union[str, StreamTopic],
        group_name: str,
    ) -> Dict[str, Any]:
        """
        Get information about a consumer group.

        Args:
            topic: Stream topic
            group_name: Consumer group name

        Returns:
            Consumer group info dictionary
        """
        self._ensure_initialized()

        stream_name = topic.value if isinstance(topic, StreamTopic) else topic

        try:
            groups = await self._redis.xinfo_groups(stream_name)

            for group in groups:
                if group["name"] == group_name:
                    return {
                        "name": group["name"],
                        "consumers": group["consumers"],
                        "pending": group["pending"],
                        "last_delivered_id": group.get("last-delivered-id"),
                    }

            return {"exists": False}

        except ResponseError:
            return {"exists": False}

    async def trim_stream(
        self,
        topic: Union[str, StreamTopic],
        max_len: int,
        approximate: bool = True,
    ) -> int:
        """
        Trim a stream to a maximum length.

        Args:
            topic: Stream topic
            max_len: Maximum entries to keep
            approximate: Use approximate trimming (faster)

        Returns:
            Number of entries removed
        """
        self._ensure_initialized()

        stream_name = topic.value if isinstance(topic, StreamTopic) else topic

        # Get current length
        info = await self.get_stream_info(topic)
        current_len = info.get("length", 0)

        if current_len <= max_len:
            return 0

        # Trim
        if approximate:
            await self._redis.xtrim(stream_name, maxlen=max_len, approximate=True)
        else:
            await self._redis.xtrim(stream_name, maxlen=max_len, approximate=False)

        trimmed = current_len - max_len
        self.logger.info(f"Trimmed {trimmed} entries from {stream_name}")

        return trimmed

    async def delete_stream(
        self,
        topic: Union[str, StreamTopic],
    ) -> bool:
        """
        Delete a stream entirely.

        Args:
            topic: Stream topic

        Returns:
            True if deleted
        """
        self._ensure_initialized()

        stream_name = topic.value if isinstance(topic, StreamTopic) else topic

        result = await self._redis.delete(stream_name)

        if result:
            self.logger.info(f"Deleted stream: {stream_name}")

        return bool(result)

    # -------------------------------------------------------------------------
    # Health & Monitoring
    # -------------------------------------------------------------------------

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the event bus.

        Returns:
            Health status dictionary
        """
        try:
            if not self._initialized or not self._redis:
                return {"healthy": False, "error": "Not initialized"}

            # Ping Redis
            latency_start = time.time()
            await self._redis.ping()
            latency = (time.time() - latency_start) * 1000

            # Get stream stats
            stream_stats = {}
            for topic in StreamTopic:
                info = await self.get_stream_info(topic)
                if info.get("length", 0) > 0:
                    stream_stats[topic.value] = info.get("length", 0)

            return {
                "healthy": True,
                "consumer_id": self._consumer_id,
                "latency_ms": round(latency, 2),
                "active_streams": len(stream_stats),
                "stream_lengths": stream_stats,
            }

        except Exception as e:
            return {"healthy": False, "error": str(e)}

    async def publish_heartbeat(self) -> str:
        """
        Publish a heartbeat message.

        Returns:
            Heartbeat message ID
        """
        heartbeat = EventMessage(
            event_type="heartbeat",
            payload={
                "consumer_id": self._consumer_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        return await self.publish(StreamTopic.HEARTBEAT, heartbeat)


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_event_bus: Optional[EventBus] = None


async def get_event_bus() -> EventBus:
    """Get or create the global event bus instance."""
    global _event_bus

    if _event_bus is None:
        _event_bus = EventBus()
        await _event_bus.initialize()

    return _event_bus


async def shutdown_event_bus() -> None:
    """Shutdown the global event bus instance."""
    global _event_bus

    if _event_bus:
        await _event_bus.shutdown()
        _event_bus = None
