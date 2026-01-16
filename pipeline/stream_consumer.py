"""
BIBLOS v2 - Stream Consumer Base Class

Base class for stream-based phase consumers that process messages
from Redis Streams with automatic acknowledgment, retry, and error handling.
"""
import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, Union

from pipeline.event_bus import (
    EventBus,
    EventBusConfig,
    EventMessage,
    PhaseCompleteEvent,
    PhaseRequestEvent,
    StreamTopic,
    PHASE_TOPICS,
    get_event_bus,
)
from pipeline.base import (
    BasePipelinePhase,
    PhaseConfig,
    PhaseResult,
    PhaseStatus,
)


logger = logging.getLogger("biblos.pipeline.stream_consumer")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class StreamConsumerConfig:
    """Configuration for a stream consumer."""
    # Consumer identity
    consumer_group: str = "biblos-phases"
    consumer_prefix: str = "phase"

    # Processing settings
    batch_size: int = 5
    parallel_processing: int = 2
    timeout_seconds: int = 300

    # Retry settings
    max_retries: int = 3
    retry_delay_base: float = 1.0
    retry_delay_max: float = 60.0

    # Health check
    heartbeat_interval: int = 30

    # Checkpointing
    checkpoint_enabled: bool = True
    checkpoint_interval: int = 10


# =============================================================================
# BASE STREAM CONSUMER
# =============================================================================

class BaseStreamConsumer(ABC):
    """
    Base class for stream-based phase consumers.

    Provides:
    - Automatic message acknowledgment
    - Retry logic with exponential backoff
    - Dead letter queue integration
    - Consumer group coordination
    - Health monitoring
    """

    def __init__(
        self,
        phase_name: str,
        config: Optional[StreamConsumerConfig] = None,
        event_bus: Optional[EventBus] = None,
    ):
        self.phase_name = phase_name
        self.config = config or StreamConsumerConfig()
        self.logger = logging.getLogger(f"biblos.consumer.{phase_name}")
        self._event_bus = event_bus
        self._initialized = False
        self._running = False

        # Consumer ID
        self._consumer_id = f"{self.config.consumer_prefix}-{phase_name}-{uuid.uuid4().hex[:8]}"

        # Topics
        self._request_topic = PHASE_TOPICS[phase_name]["request"]
        self._complete_topic = PHASE_TOPICS[phase_name]["complete"]

        # Background tasks
        self._tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()

        # Processing semaphore for parallel limit
        self._semaphore = asyncio.Semaphore(self.config.parallel_processing)

        # Metrics
        self._metrics = {
            "messages_received": 0,
            "messages_processed": 0,
            "messages_failed": 0,
            "messages_retried": 0,
            "avg_processing_time": 0.0,
            "processing_times": [],
        }

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def initialize(self) -> None:
        """Initialize the stream consumer."""
        if self._initialized:
            return

        self.logger.info(f"Initializing stream consumer: {self._consumer_id}")

        # Get or create event bus
        if self._event_bus is None:
            self._event_bus = await get_event_bus()

        # Create consumer group
        await self._event_bus.ensure_consumer_group(
            self._request_topic,
            self.config.consumer_group,
        )

        # Initialize phase-specific resources
        await self._initialize_phase()

        self._initialized = True
        self.logger.info(f"Stream consumer initialized: {self._consumer_id}")

    @abstractmethod
    async def _initialize_phase(self) -> None:
        """Initialize phase-specific resources (agents, models, etc.)."""
        pass

    async def start(self) -> None:
        """Start the consumer event loop."""
        if self._running:
            return

        if not self._initialized:
            await self.initialize()

        self._running = True
        self._shutdown_event.clear()

        self.logger.info(f"Starting stream consumer: {self._consumer_id}")

        # Start main consumer loop
        self._tasks.append(
            asyncio.create_task(
                self._consume_loop(),
                name=f"consume_loop_{self.phase_name}"
            )
        )

        # Start pending message recovery
        self._tasks.append(
            asyncio.create_task(
                self._pending_recovery_loop(),
                name=f"pending_recovery_{self.phase_name}"
            )
        )

        # Start heartbeat
        if self.config.heartbeat_interval > 0:
            self._tasks.append(
                asyncio.create_task(
                    self._heartbeat_loop(),
                    name=f"heartbeat_{self.phase_name}"
                )
            )

        self.logger.info(f"Started {len(self._tasks)} consumer tasks")

    async def stop(self) -> None:
        """Stop the consumer gracefully."""
        if not self._running:
            return

        self.logger.info(f"Stopping stream consumer: {self._consumer_id}")
        self._shutdown_event.set()
        self._running = False

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._tasks.clear()

        # Cleanup phase resources
        await self._cleanup_phase()

        self.logger.info(f"Stream consumer stopped: {self._consumer_id}")

    @abstractmethod
    async def _cleanup_phase(self) -> None:
        """Cleanup phase-specific resources."""
        pass

    async def shutdown(self) -> None:
        """Full shutdown."""
        await self.stop()
        self._initialized = False

    # -------------------------------------------------------------------------
    # Message Processing
    # -------------------------------------------------------------------------

    async def _consume_loop(self) -> None:
        """Main consumer loop for processing messages."""
        self.logger.info(f"Starting consume loop for {self.phase_name}")

        async for message_id, event in self._event_bus.subscribe(
            self._request_topic,
            self.config.consumer_group,
            self._consumer_id,
        ):
            if self._shutdown_event.is_set():
                break

            # Process with semaphore for parallel limit
            asyncio.create_task(
                self._process_with_semaphore(message_id, event)
            )

    async def _process_with_semaphore(
        self,
        message_id: str,
        event: EventMessage,
    ) -> None:
        """Process a message with semaphore limiting."""
        async with self._semaphore:
            await self._process_message(message_id, event)

    async def _process_message(
        self,
        message_id: str,
        event: EventMessage,
    ) -> None:
        """Process a single message with error handling."""
        self._metrics["messages_received"] += 1
        start_time = time.time()

        payload = event.payload
        verse_id = payload.get("verse_id", "unknown")
        text = payload.get("text", "")
        context = payload.get("context", {})

        self.logger.info(
            f"Processing {self.phase_name} for {verse_id} "
            f"(message_id: {message_id})"
        )

        try:
            # Execute phase processing
            result = await asyncio.wait_for(
                self._execute_phase(verse_id, text, context),
                timeout=self.config.timeout_seconds,
            )

            # Record processing time
            processing_time = time.time() - start_time
            self._record_processing_time(processing_time)

            # Publish completion event
            await self._publish_completion(
                event=event,
                status="completed",
                result=result,
            )

            # Acknowledge message
            await self._event_bus.acknowledge(
                self._request_topic,
                self.config.consumer_group,
                message_id,
            )

            self._metrics["messages_processed"] += 1
            self.logger.info(
                f"Completed {self.phase_name} for {verse_id} "
                f"in {processing_time:.2f}s"
            )

        except asyncio.TimeoutError:
            await self._handle_failure(
                message_id,
                event,
                f"Timeout after {self.config.timeout_seconds}s",
            )

        except Exception as e:
            await self._handle_failure(message_id, event, str(e))

    @abstractmethod
    async def _execute_phase(
        self,
        verse_id: str,
        text: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute the phase processing.

        Args:
            verse_id: Verse identifier
            text: Verse text
            context: Pipeline context with previous results

        Returns:
            Dictionary with phase results
        """
        pass

    async def _publish_completion(
        self,
        event: EventMessage,
        status: str,
        result: Dict[str, Any],
        error: Optional[str] = None,
    ) -> None:
        """Publish phase completion event."""
        payload = event.payload
        verse_id = payload.get("verse_id", "unknown")

        complete_event = PhaseCompleteEvent.create(
            phase_name=self.phase_name,
            verse_id=verse_id,
            status=status,
            result=result,
            correlation_id=event.correlation_id,
            error=error,
        )

        await self._event_bus.publish(self._complete_topic, complete_event)

    async def _handle_failure(
        self,
        message_id: str,
        event: EventMessage,
        error: str,
    ) -> None:
        """Handle message processing failure."""
        self._metrics["messages_failed"] += 1
        verse_id = event.payload.get("verse_id", "unknown")

        self.logger.error(
            f"Failed to process {self.phase_name} for {verse_id}: {error}"
        )

        # Check retry count
        if event.retry_count < self.config.max_retries:
            # Schedule retry
            self._metrics["messages_retried"] += 1
            self.logger.info(
                f"Scheduling retry {event.retry_count + 1}/{self.config.max_retries} "
                f"for {verse_id}"
            )

            await self._event_bus.schedule_retry(
                event,
                self._request_topic,
                error,
            )

            # Acknowledge original to prevent double processing
            await self._event_bus.acknowledge(
                self._request_topic,
                self.config.consumer_group,
                message_id,
            )
        else:
            # Max retries exceeded - send completion with failure
            await self._publish_completion(
                event=event,
                status="failed",
                result={},
                error=error,
            )

            # Acknowledge
            await self._event_bus.acknowledge(
                self._request_topic,
                self.config.consumer_group,
                message_id,
            )

            # Send to DLQ (already done by schedule_retry when max exceeded)

    # -------------------------------------------------------------------------
    # Pending Message Recovery
    # -------------------------------------------------------------------------

    async def _pending_recovery_loop(self) -> None:
        """Periodically claim and process pending messages."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Check every minute

                # Claim pending messages from other consumers
                claimed = await self._event_bus.claim_pending(
                    self._request_topic,
                    self.config.consumer_group,
                    self._consumer_id,
                    min_idle_time_ms=60000,  # 60 seconds idle
                    count=5,
                )

                for message_id, event in claimed:
                    self.logger.info(
                        f"Recovered pending message {message_id}"
                    )
                    asyncio.create_task(
                        self._process_with_semaphore(message_id, event)
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Pending recovery error: {e}")

    # -------------------------------------------------------------------------
    # Heartbeat
    # -------------------------------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        """Periodic heartbeat."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.config.heartbeat_interval)

                # Publish heartbeat with consumer stats
                heartbeat = EventMessage(
                    event_type="consumer_heartbeat",
                    payload={
                        "consumer_id": self._consumer_id,
                        "phase_name": self.phase_name,
                        "metrics": self._metrics,
                    },
                )

                await self._event_bus.publish(StreamTopic.HEARTBEAT, heartbeat)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    def _record_processing_time(self, duration: float) -> None:
        """Record processing time for metrics."""
        times = self._metrics["processing_times"]
        times.append(duration)

        # Keep last 100 times
        if len(times) > 100:
            times = times[-100:]
            self._metrics["processing_times"] = times

        # Update average
        self._metrics["avg_processing_time"] = sum(times) / len(times)

    def get_metrics(self) -> Dict[str, Any]:
        """Get consumer metrics."""
        return {
            "consumer_id": self._consumer_id,
            "phase_name": self.phase_name,
            "running": self._running,
            "initialized": self._initialized,
            **{k: v for k, v in self._metrics.items() if k != "processing_times"},
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            "healthy": self._running and self._initialized,
            **self.get_metrics(),
        }


# =============================================================================
# PHASE CONSUMER ADAPTER
# =============================================================================

class PhaseStreamConsumer(BaseStreamConsumer):
    """
    Stream consumer that wraps an existing BasePipelinePhase.

    Allows existing phase implementations to be used in the
    stream-based architecture without modification.
    """

    def __init__(
        self,
        phase: BasePipelinePhase,
        config: Optional[StreamConsumerConfig] = None,
        event_bus: Optional[EventBus] = None,
    ):
        self._phase = phase
        super().__init__(
            phase_name=phase.config.name,
            config=config,
            event_bus=event_bus,
        )

    async def _initialize_phase(self) -> None:
        """Initialize the wrapped phase."""
        await self._phase.initialize()

    async def _cleanup_phase(self) -> None:
        """Cleanup the wrapped phase."""
        await self._phase.cleanup()

    async def _execute_phase(
        self,
        verse_id: str,
        text: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute the wrapped phase."""
        result = await self._phase.execute(verse_id, text, context)

        return {
            "status": result.status.value,
            "agent_results": result.agent_results,
            "metrics": result.metrics,
            "duration": result.duration,
            "error": result.error,
        }


# =============================================================================
# CONSUMER MANAGER
# =============================================================================

class StreamConsumerManager:
    """
    Manages multiple stream consumers for all pipeline phases.

    Provides centralized lifecycle management, health monitoring,
    and metrics collection.
    """

    def __init__(
        self,
        config: Optional[StreamConsumerConfig] = None,
    ):
        self.config = config or StreamConsumerConfig()
        self.logger = logging.getLogger("biblos.consumer_manager")
        self._consumers: Dict[str, BaseStreamConsumer] = {}
        self._initialized = False
        self._running = False

    def register_consumer(
        self,
        phase_name: str,
        consumer: BaseStreamConsumer,
    ) -> None:
        """Register a consumer for a phase."""
        self._consumers[phase_name] = consumer
        self.logger.info(f"Registered consumer for phase: {phase_name}")

    def register_phase(
        self,
        phase: BasePipelinePhase,
        config: Optional[StreamConsumerConfig] = None,
    ) -> None:
        """Register a phase as a stream consumer."""
        consumer = PhaseStreamConsumer(
            phase=phase,
            config=config or self.config,
        )
        self.register_consumer(phase.config.name, consumer)

    async def initialize(self) -> None:
        """Initialize all registered consumers."""
        if self._initialized:
            return

        self.logger.info(f"Initializing {len(self._consumers)} consumers")

        for phase_name, consumer in self._consumers.items():
            try:
                await consumer.initialize()
                self.logger.info(f"Initialized consumer: {phase_name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize {phase_name}: {e}")

        self._initialized = True

    async def start(self) -> None:
        """Start all registered consumers."""
        if self._running:
            return

        if not self._initialized:
            await self.initialize()

        self.logger.info(f"Starting {len(self._consumers)} consumers")

        for phase_name, consumer in self._consumers.items():
            try:
                await consumer.start()
                self.logger.info(f"Started consumer: {phase_name}")
            except Exception as e:
                self.logger.error(f"Failed to start {phase_name}: {e}")

        self._running = True

    async def stop(self) -> None:
        """Stop all registered consumers."""
        if not self._running:
            return

        self.logger.info(f"Stopping {len(self._consumers)} consumers")

        for phase_name, consumer in self._consumers.items():
            try:
                await consumer.stop()
                self.logger.info(f"Stopped consumer: {phase_name}")
            except Exception as e:
                self.logger.error(f"Failed to stop {phase_name}: {e}")

        self._running = False

    async def shutdown(self) -> None:
        """Full shutdown."""
        await self.stop()
        self._consumers.clear()
        self._initialized = False

    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics from all consumers."""
        return {
            phase_name: consumer.get_metrics()
            for phase_name, consumer in self._consumers.items()
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all consumers."""
        consumer_health = {}

        for phase_name, consumer in self._consumers.items():
            consumer_health[phase_name] = await consumer.health_check()

        all_healthy = all(
            h.get("healthy", False)
            for h in consumer_health.values()
        )

        return {
            "healthy": all_healthy and self._running,
            "running": self._running,
            "initialized": self._initialized,
            "consumer_count": len(self._consumers),
            "consumers": consumer_health,
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

async def create_phase_consumer(
    phase: BasePipelinePhase,
    config: Optional[StreamConsumerConfig] = None,
    auto_start: bool = False,
) -> PhaseStreamConsumer:
    """
    Create a stream consumer for a pipeline phase.

    Args:
        phase: The pipeline phase to wrap
        config: Consumer configuration
        auto_start: Whether to automatically start the consumer

    Returns:
        Initialized PhaseStreamConsumer
    """
    consumer = PhaseStreamConsumer(phase=phase, config=config)
    await consumer.initialize()

    if auto_start:
        await consumer.start()

    return consumer


async def create_consumer_manager(
    phases: List[BasePipelinePhase],
    config: Optional[StreamConsumerConfig] = None,
    auto_start: bool = False,
) -> StreamConsumerManager:
    """
    Create a consumer manager for multiple phases.

    Args:
        phases: List of pipeline phases
        config: Consumer configuration
        auto_start: Whether to automatically start consumers

    Returns:
        Initialized StreamConsumerManager
    """
    manager = StreamConsumerManager(config=config)

    for phase in phases:
        manager.register_phase(phase)

    await manager.initialize()

    if auto_start:
        await manager.start()

    return manager
