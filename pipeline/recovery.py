"""
BIBLOS v2 - Pipeline Recovery Service

Provides recovery mechanisms for the stream-based pipeline including:
- Dead letter queue management
- Message reprocessing
- Checkpoint-based recovery
- Stale message cleanup
"""
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from pipeline.event_bus import (
    EventBus,
    EventBusConfig,
    EventMessage,
    StreamTopic,
    PHASE_TOPICS,
    get_event_bus,
)


logger = logging.getLogger("biblos.pipeline.recovery")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RecoveryConfig:
    """Configuration for recovery service."""
    # DLQ settings
    dlq_retention_hours: int = 168  # 7 days
    dlq_batch_size: int = 50
    auto_reprocess_enabled: bool = False
    auto_reprocess_delay_minutes: int = 60

    # Checkpoint settings
    checkpoint_retention_hours: int = 24
    checkpoint_prefix: str = "biblos:checkpoint:"

    # Stale message settings
    stale_threshold_minutes: int = 30
    stale_cleanup_interval_minutes: int = 10

    # Recovery settings
    max_recovery_batch: int = 100
    recovery_cooldown_seconds: int = 5


# =============================================================================
# DEAD LETTER QUEUE ENTRY
# =============================================================================

@dataclass
class DLQEntry:
    """Represents an entry in the dead letter queue."""
    message_id: str
    original_event_type: str
    original_payload: Dict[str, Any]
    original_message_id: Optional[str]
    error: str
    source_topic: str
    final_retry_count: int
    correlation_id: str
    timestamp: str

    @classmethod
    def from_event(cls, message_id: str, event: EventMessage) -> "DLQEntry":
        """Create from DLQ event message."""
        payload = event.payload
        return cls(
            message_id=message_id,
            original_event_type=payload.get("original_event_type", "unknown"),
            original_payload=payload.get("original_payload", {}),
            original_message_id=payload.get("original_message_id"),
            error=payload.get("error", "Unknown error"),
            source_topic=payload.get("source_topic", "unknown"),
            final_retry_count=payload.get("final_retry_count", 0),
            correlation_id=event.correlation_id,
            timestamp=event.timestamp,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message_id": self.message_id,
            "original_event_type": self.original_event_type,
            "original_payload": self.original_payload,
            "original_message_id": self.original_message_id,
            "error": self.error,
            "source_topic": self.source_topic,
            "final_retry_count": self.final_retry_count,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
        }


# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================

@dataclass
class Checkpoint:
    """Represents a pipeline checkpoint."""
    checkpoint_id: str
    consumer_id: str
    topic: str
    last_message_id: str
    processed_count: int
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "consumer_id": self.consumer_id,
            "topic": self.topic,
            "last_message_id": self.last_message_id,
            "processed_count": self.processed_count,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Checkpoint":
        """Create from dictionary."""
        return cls(
            checkpoint_id=data["checkpoint_id"],
            consumer_id=data["consumer_id"],
            topic=data["topic"],
            last_message_id=data["last_message_id"],
            processed_count=data["processed_count"],
            timestamp=data["timestamp"],
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# RECOVERY SERVICE
# =============================================================================

class RecoveryService:
    """
    Manages recovery operations for the stream-based pipeline.

    Features:
    - DLQ browsing, filtering, and reprocessing
    - Checkpoint creation and recovery
    - Stale message detection and cleanup
    - Metrics and health reporting
    """

    def __init__(
        self,
        config: Optional[RecoveryConfig] = None,
        event_bus: Optional[EventBus] = None,
    ):
        self.config = config or RecoveryConfig()
        self.logger = logging.getLogger("biblos.recovery")
        self._event_bus = event_bus
        self._initialized = False
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()

        # Metrics
        self._metrics = {
            "dlq_entries_total": 0,
            "dlq_entries_reprocessed": 0,
            "dlq_entries_discarded": 0,
            "checkpoints_created": 0,
            "recoveries_performed": 0,
            "stale_messages_cleaned": 0,
        }

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def initialize(self) -> None:
        """Initialize the recovery service."""
        if self._initialized:
            return

        self.logger.info("Initializing recovery service")

        if self._event_bus is None:
            self._event_bus = await get_event_bus()

        self._initialized = True
        self.logger.info("Recovery service initialized")

    async def start(self) -> None:
        """Start background recovery tasks."""
        if self._running:
            return

        if not self._initialized:
            await self.initialize()

        self._running = True
        self._shutdown_event.clear()

        self.logger.info("Starting recovery service background tasks")

        # Start stale cleanup loop
        self._tasks.append(
            asyncio.create_task(
                self._stale_cleanup_loop(),
                name="stale_cleanup"
            )
        )

        # Start auto-reprocess loop if enabled
        if self.config.auto_reprocess_enabled:
            self._tasks.append(
                asyncio.create_task(
                    self._auto_reprocess_loop(),
                    name="auto_reprocess"
                )
            )

    async def stop(self) -> None:
        """Stop the recovery service."""
        if not self._running:
            return

        self.logger.info("Stopping recovery service")
        self._shutdown_event.set()
        self._running = False

        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._tasks.clear()
        self.logger.info("Recovery service stopped")

    # -------------------------------------------------------------------------
    # Dead Letter Queue Management
    # -------------------------------------------------------------------------

    async def get_dlq_entries(
        self,
        count: int = 100,
        start_id: str = "-",
        end_id: str = "+",
        filter_error: Optional[str] = None,
        filter_topic: Optional[str] = None,
    ) -> List[DLQEntry]:
        """
        Get entries from the dead letter queue.

        Args:
            count: Maximum entries to retrieve
            start_id: Starting message ID
            end_id: Ending message ID
            filter_error: Filter by error message substring
            filter_topic: Filter by source topic

        Returns:
            List of DLQ entries
        """
        await self._ensure_initialized()

        entries = await self._event_bus.read_stream(
            StreamTopic.DEAD_LETTER_QUEUE,
            start_id=start_id,
            end_id=end_id,
            count=count,
        )

        dlq_entries = []
        for message_id, event in entries:
            entry = DLQEntry.from_event(message_id, event)

            # Apply filters
            if filter_error and filter_error.lower() not in entry.error.lower():
                continue
            if filter_topic and filter_topic != entry.source_topic:
                continue

            dlq_entries.append(entry)

        self._metrics["dlq_entries_total"] = len(dlq_entries)
        return dlq_entries

    async def get_dlq_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for the dead letter queue.

        Returns:
            Summary dictionary with counts by error type and source topic
        """
        await self._ensure_initialized()

        # Get all DLQ entries
        entries = await self.get_dlq_entries(count=1000)

        # Group by error type
        by_error: Dict[str, int] = {}
        by_topic: Dict[str, int] = {}
        by_hour: Dict[str, int] = {}

        for entry in entries:
            # Extract error type (first line of error)
            error_type = entry.error.split("\n")[0][:100]
            by_error[error_type] = by_error.get(error_type, 0) + 1

            # Count by source topic
            by_topic[entry.source_topic] = by_topic.get(entry.source_topic, 0) + 1

            # Count by hour
            try:
                dt = datetime.fromisoformat(entry.timestamp)
                hour_key = dt.strftime("%Y-%m-%d %H:00")
                by_hour[hour_key] = by_hour.get(hour_key, 0) + 1
            except (ValueError, TypeError):
                pass

        return {
            "total_entries": len(entries),
            "by_error_type": dict(sorted(by_error.items(), key=lambda x: -x[1])[:10]),
            "by_source_topic": by_topic,
            "by_hour": dict(sorted(by_hour.items())[-24:]),  # Last 24 hours
            "oldest_entry": entries[0].timestamp if entries else None,
            "newest_entry": entries[-1].timestamp if entries else None,
        }

    async def reprocess_dlq_entry(
        self,
        message_id: str,
        target_topic: Optional[str] = None,
    ) -> Optional[str]:
        """
        Reprocess a single DLQ entry.

        Args:
            message_id: DLQ message ID
            target_topic: Override target topic (uses original if None)

        Returns:
            New message ID if successful
        """
        await self._ensure_initialized()

        # Get the DLQ entry
        entries = await self._event_bus.read_stream(
            StreamTopic.DEAD_LETTER_QUEUE,
            start_id=message_id,
            end_id=message_id,
        )

        if not entries:
            self.logger.warning(f"DLQ entry not found: {message_id}")
            return None

        _, event = entries[0]
        dlq_entry = DLQEntry.from_event(message_id, event)

        # Determine target topic
        topic = target_topic or dlq_entry.source_topic

        # Reconstruct and publish original message
        new_id = await self._event_bus.reprocess_dlq_message(message_id, topic)

        if new_id:
            self._metrics["dlq_entries_reprocessed"] += 1
            self.logger.info(
                f"Reprocessed DLQ entry {message_id} -> {new_id} on {topic}"
            )

        return new_id

    async def reprocess_dlq_batch(
        self,
        message_ids: List[str],
        target_topic: Optional[str] = None,
        delay_seconds: float = 0.5,
    ) -> Dict[str, Optional[str]]:
        """
        Reprocess multiple DLQ entries.

        Args:
            message_ids: List of DLQ message IDs
            target_topic: Override target topic
            delay_seconds: Delay between reprocessing

        Returns:
            Dictionary mapping old IDs to new IDs
        """
        results = {}

        for message_id in message_ids:
            new_id = await self.reprocess_dlq_entry(message_id, target_topic)
            results[message_id] = new_id

            if delay_seconds > 0:
                await asyncio.sleep(delay_seconds)

        return results

    async def reprocess_dlq_by_filter(
        self,
        filter_error: Optional[str] = None,
        filter_topic: Optional[str] = None,
        max_count: int = 100,
        target_topic: Optional[str] = None,
    ) -> int:
        """
        Reprocess DLQ entries matching filters.

        Args:
            filter_error: Filter by error substring
            filter_topic: Filter by source topic
            max_count: Maximum entries to reprocess
            target_topic: Override target topic

        Returns:
            Number of entries reprocessed
        """
        entries = await self.get_dlq_entries(
            count=max_count,
            filter_error=filter_error,
            filter_topic=filter_topic,
        )

        count = 0
        for entry in entries:
            result = await self.reprocess_dlq_entry(
                entry.message_id,
                target_topic or entry.source_topic,
            )
            if result:
                count += 1

            await asyncio.sleep(self.config.recovery_cooldown_seconds)

        return count

    async def discard_dlq_entry(self, message_id: str) -> bool:
        """
        Discard a DLQ entry (mark as handled without reprocessing).

        For now, we track discards in metrics but don't remove from stream.
        Stream trimming handles actual cleanup.
        """
        self._metrics["dlq_entries_discarded"] += 1
        self.logger.info(f"Discarded DLQ entry: {message_id}")
        return True

    async def purge_old_dlq_entries(
        self,
        older_than_hours: Optional[int] = None,
    ) -> int:
        """
        Purge DLQ entries older than specified time.

        Args:
            older_than_hours: Age threshold (defaults to config value)

        Returns:
            Number of entries purged
        """
        await self._ensure_initialized()

        hours = older_than_hours or self.config.dlq_retention_hours
        threshold = datetime.now(timezone.utc) - timedelta(hours=hours)

        # Get all entries
        entries = await self.get_dlq_entries(count=10000)

        # Find entries to purge
        to_purge = []
        for entry in entries:
            try:
                dt = datetime.fromisoformat(entry.timestamp)
                if dt < threshold:
                    to_purge.append(entry.message_id)
            except (ValueError, TypeError):
                pass

        # Trim stream to remove old entries
        if to_purge:
            # Get the newest entry to keep
            info = await self._event_bus.get_stream_info(StreamTopic.DEAD_LETTER_QUEUE)
            current_len = info.get("length", 0)
            new_len = current_len - len(to_purge)

            if new_len > 0:
                await self._event_bus.trim_stream(
                    StreamTopic.DEAD_LETTER_QUEUE,
                    max_len=max(new_len, 1),
                )

        self.logger.info(f"Purged {len(to_purge)} old DLQ entries")
        return len(to_purge)

    # -------------------------------------------------------------------------
    # Checkpoint Management
    # -------------------------------------------------------------------------

    async def create_checkpoint(
        self,
        consumer_id: str,
        topic: str,
        last_message_id: str,
        processed_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Checkpoint:
        """
        Create a checkpoint for a consumer.

        Args:
            consumer_id: Consumer identifier
            topic: Stream topic
            last_message_id: Last processed message ID
            processed_count: Number of messages processed
            metadata: Additional metadata

        Returns:
            Created checkpoint
        """
        await self._ensure_initialized()

        checkpoint = Checkpoint(
            checkpoint_id=f"{consumer_id}:{topic}",
            consumer_id=consumer_id,
            topic=topic,
            last_message_id=last_message_id,
            processed_count=processed_count,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {},
        )

        # Store in Redis
        key = f"{self.config.checkpoint_prefix}{checkpoint.checkpoint_id}"
        await self._event_bus._redis.set(
            key,
            json.dumps(checkpoint.to_dict()),
            ex=self.config.checkpoint_retention_hours * 3600,
        )

        self._metrics["checkpoints_created"] += 1
        self.logger.debug(f"Created checkpoint: {checkpoint.checkpoint_id}")

        return checkpoint

    async def get_checkpoint(
        self,
        consumer_id: str,
        topic: str,
    ) -> Optional[Checkpoint]:
        """
        Get checkpoint for a consumer/topic pair.

        Args:
            consumer_id: Consumer identifier
            topic: Stream topic

        Returns:
            Checkpoint if found
        """
        await self._ensure_initialized()

        checkpoint_id = f"{consumer_id}:{topic}"
        key = f"{self.config.checkpoint_prefix}{checkpoint_id}"

        data = await self._event_bus._redis.get(key)

        if data:
            return Checkpoint.from_dict(json.loads(data))
        return None

    async def list_checkpoints(
        self,
        consumer_id: Optional[str] = None,
    ) -> List[Checkpoint]:
        """
        List all checkpoints, optionally filtered by consumer.

        Args:
            consumer_id: Filter by consumer (optional)

        Returns:
            List of checkpoints
        """
        await self._ensure_initialized()

        pattern = f"{self.config.checkpoint_prefix}*"
        if consumer_id:
            pattern = f"{self.config.checkpoint_prefix}{consumer_id}:*"

        checkpoints = []
        cursor = 0

        while True:
            cursor, keys = await self._event_bus._redis.scan(
                cursor=cursor,
                match=pattern,
                count=100,
            )

            for key in keys:
                data = await self._event_bus._redis.get(key)
                if data:
                    checkpoints.append(Checkpoint.from_dict(json.loads(data)))

            if cursor == 0:
                break

        return checkpoints

    async def delete_checkpoint(
        self,
        consumer_id: str,
        topic: str,
    ) -> bool:
        """
        Delete a checkpoint.

        Args:
            consumer_id: Consumer identifier
            topic: Stream topic

        Returns:
            True if deleted
        """
        await self._ensure_initialized()

        checkpoint_id = f"{consumer_id}:{topic}"
        key = f"{self.config.checkpoint_prefix}{checkpoint_id}"

        result = await self._event_bus._redis.delete(key)
        return bool(result)

    # -------------------------------------------------------------------------
    # Stale Message Cleanup
    # -------------------------------------------------------------------------

    async def find_stale_messages(
        self,
        topic: str,
        group_name: str,
        threshold_minutes: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find messages that have been pending for too long.

        Args:
            topic: Stream topic
            group_name: Consumer group
            threshold_minutes: Age threshold (defaults to config)

        Returns:
            List of stale message info
        """
        await self._ensure_initialized()

        threshold = (threshold_minutes or self.config.stale_threshold_minutes) * 60 * 1000

        pending = await self._event_bus.get_pending(
            topic,
            group_name,
            count=1000,
        )

        stale = [
            p for p in pending
            if p["time_since_delivered"] > threshold
        ]

        return stale

    async def cleanup_stale_messages(
        self,
        topic: str,
        group_name: str,
        claim_consumer: str,
    ) -> int:
        """
        Claim and reprocess stale messages.

        Args:
            topic: Stream topic
            group_name: Consumer group
            claim_consumer: Consumer to claim messages for

        Returns:
            Number of messages claimed
        """
        await self._ensure_initialized()

        threshold_ms = self.config.stale_threshold_minutes * 60 * 1000

        claimed = await self._event_bus.claim_pending(
            topic,
            group_name,
            claim_consumer,
            min_idle_time_ms=threshold_ms,
            count=self.config.max_recovery_batch,
        )

        if claimed:
            self._metrics["stale_messages_cleaned"] += len(claimed)
            self.logger.info(f"Claimed {len(claimed)} stale messages from {topic}")

        return len(claimed)

    async def cleanup_all_stale(
        self,
        group_name: str = "biblos-phases",
        claim_consumer: str = "recovery",
    ) -> Dict[str, int]:
        """
        Cleanup stale messages across all phase topics.

        Args:
            group_name: Consumer group
            claim_consumer: Consumer to claim for

        Returns:
            Dictionary of topic -> messages claimed
        """
        results = {}

        for phase_name, topics in PHASE_TOPICS.items():
            request_topic = topics["request"]

            count = await self.cleanup_stale_messages(
                request_topic.value if hasattr(request_topic, "value") else request_topic,
                group_name,
                claim_consumer,
            )

            if count > 0:
                results[f"{phase_name}_request"] = count

        return results

    # -------------------------------------------------------------------------
    # Background Tasks
    # -------------------------------------------------------------------------

    async def _stale_cleanup_loop(self) -> None:
        """Periodic stale message cleanup."""
        interval = self.config.stale_cleanup_interval_minutes * 60

        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(interval)

                results = await self.cleanup_all_stale()

                if results:
                    self.logger.info(f"Stale cleanup results: {results}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Stale cleanup error: {e}")

    async def _auto_reprocess_loop(self) -> None:
        """Automatic DLQ reprocessing."""
        interval = self.config.auto_reprocess_delay_minutes * 60

        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(interval)

                # Get old DLQ entries
                entries = await self.get_dlq_entries(
                    count=self.config.dlq_batch_size
                )

                # Only reprocess entries older than the delay
                threshold = datetime.now(timezone.utc) - timedelta(
                    minutes=self.config.auto_reprocess_delay_minutes
                )

                to_reprocess = []
                for entry in entries:
                    try:
                        dt = datetime.fromisoformat(entry.timestamp)
                        if dt < threshold:
                            to_reprocess.append(entry.message_id)
                    except (ValueError, TypeError):
                        pass

                if to_reprocess:
                    self.logger.info(
                        f"Auto-reprocessing {len(to_reprocess)} DLQ entries"
                    )
                    await self.reprocess_dlq_batch(
                        to_reprocess,
                        delay_seconds=self.config.recovery_cooldown_seconds,
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Auto-reprocess error: {e}")

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    async def _ensure_initialized(self) -> None:
        """Ensure service is initialized."""
        if not self._initialized:
            await self.initialize()

    def get_metrics(self) -> Dict[str, Any]:
        """Get recovery service metrics."""
        return {
            **self._metrics,
            "running": self._running,
            "initialized": self._initialized,
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        dlq_info = {}

        if self._initialized:
            try:
                dlq_info = await self.get_dlq_summary()
            except Exception:
                pass

        return {
            "healthy": self._initialized,
            "running": self._running,
            "metrics": self._metrics,
            "dlq_summary": dlq_info,
        }


# =============================================================================
# SINGLETON AND FACTORY
# =============================================================================

_recovery_service: Optional[RecoveryService] = None


async def get_recovery_service() -> RecoveryService:
    """Get or create global recovery service."""
    global _recovery_service

    if _recovery_service is None:
        _recovery_service = RecoveryService()
        await _recovery_service.initialize()

    return _recovery_service


async def shutdown_recovery_service() -> None:
    """Shutdown global recovery service."""
    global _recovery_service

    if _recovery_service:
        await _recovery_service.stop()
        _recovery_service = None
