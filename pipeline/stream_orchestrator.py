"""
BIBLOS v2 - Stream-based Pipeline Orchestrator

Event-driven orchestrator using Redis Streams for decoupled, resilient
pipeline execution with recovery and horizontal scaling capabilities.
"""
import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

from pipeline.event_bus import (
    EventBus,
    EventBusConfig,
    EventMessage,
    PhaseCompleteEvent,
    PhaseRequestEvent,
    StreamTopic,
    VerseEvent,
    PHASE_TOPICS,
    get_event_bus,
)
from pipeline.base import (
    PhaseResult,
    PhaseStatus,
    PipelineContext,
)
from data.schemas import (
    ProcessingStatus,
    GoldenRecordSchema,
    PipelineResultSchema,
    normalize_verse_id,
)


logger = logging.getLogger("biblos.pipeline.stream_orchestrator")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class StreamOrchestratorConfig:
    """Configuration for the stream-based orchestrator."""
    # Phase execution order
    phases: List[str] = field(default_factory=lambda: [
        "linguistic", "theological", "intertextual", "validation", "finalization"
    ])

    # Phase dependencies
    phase_dependencies: Dict[str, List[str]] = field(default_factory=lambda: {
        "linguistic": [],
        "theological": ["linguistic"],
        "intertextual": ["linguistic", "theological"],
        "validation": ["linguistic", "theological", "intertextual"],
        "finalization": ["validation"],
    })

    # Consumer group configuration
    consumer_group: str = "biblos-orchestrator"
    consumer_prefix: str = "orchestrator"

    # Timeout settings
    phase_timeout_seconds: int = 300
    verse_timeout_seconds: int = 1800  # 30 minutes for entire verse

    # Batch processing
    batch_size: int = 10
    parallel_verses: int = 4

    # Recovery settings
    enable_recovery: bool = True
    checkpoint_interval: int = 10  # Checkpoint every N completed verses
    stale_verse_timeout: int = 600  # Mark verse stale after 10 minutes

    # Monitoring
    heartbeat_interval: int = 30
    metrics_interval: int = 60


# =============================================================================
# VERSE STATE TRACKING
# =============================================================================

@dataclass
class VerseState:
    """
    Tracks the state of a verse being processed through the pipeline.

    Stored in Redis for crash recovery.
    """
    verse_id: str
    text: str
    correlation_id: str
    status: str = "pending"  # pending, processing, completed, failed

    # Phase tracking
    completed_phases: List[str] = field(default_factory=list)
    current_phase: Optional[str] = None
    phase_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Timing
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    # Error tracking
    errors: List[str] = field(default_factory=list)
    retry_count: int = 0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "verse_id": self.verse_id,
            "text": self.text,
            "correlation_id": self.correlation_id,
            "status": self.status,
            "completed_phases": self.completed_phases,
            "current_phase": self.current_phase,
            "phase_results": self.phase_results,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "errors": self.errors,
            "retry_count": self.retry_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerseState":
        """Create from dictionary."""
        return cls(
            verse_id=data.get("verse_id", ""),
            text=data.get("text", ""),
            correlation_id=data.get("correlation_id", ""),
            status=data.get("status", "pending"),
            completed_phases=data.get("completed_phases", []),
            current_phase=data.get("current_phase"),
            phase_results=data.get("phase_results", {}),
            started_at=data.get("started_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            completed_at=data.get("completed_at"),
            errors=data.get("errors", []),
            retry_count=data.get("retry_count", 0),
            metadata=data.get("metadata", {}),
        )

    def is_phase_complete(self, phase_name: str) -> bool:
        """Check if a phase is complete."""
        return phase_name in self.completed_phases

    def are_dependencies_met(
        self,
        phase_name: str,
        dependencies: Dict[str, List[str]],
    ) -> bool:
        """Check if all dependencies for a phase are met."""
        required = dependencies.get(phase_name, [])
        return all(dep in self.completed_phases for dep in required)

    def get_next_phase(
        self,
        phases: List[str],
        dependencies: Dict[str, List[str]],
    ) -> Optional[str]:
        """Get the next phase to execute."""
        for phase in phases:
            if phase not in self.completed_phases:
                if self.are_dependencies_met(phase, dependencies):
                    return phase
        return None

    def to_context(self) -> Dict[str, Any]:
        """Convert to context dictionary for phase execution."""
        return {
            "verse_id": self.verse_id,
            "text": self.text,
            "completed_phases": self.completed_phases,
            "agent_results": self._flatten_agent_results(),
            "metadata": self.metadata,
        }

    def _flatten_agent_results(self) -> Dict[str, Any]:
        """Flatten all agent results from completed phases."""
        results = {}
        for phase_name, phase_result in self.phase_results.items():
            agent_results = phase_result.get("agent_results", {})
            results.update(agent_results)
        return results


# =============================================================================
# STREAM ORCHESTRATOR
# =============================================================================

class StreamOrchestrator:
    """
    Event-driven pipeline orchestrator using Redis Streams.

    Features:
    - Decoupled phase execution via publish/subscribe
    - Crash recovery via checkpointing
    - Horizontal scaling via consumer groups
    - Automatic retry with exponential backoff
    - Dead letter queue for failed messages
    - Real-time progress tracking
    """

    def __init__(
        self,
        config: Optional[StreamOrchestratorConfig] = None,
        event_bus: Optional[EventBus] = None,
    ):
        self.config = config or StreamOrchestratorConfig()
        self.logger = logging.getLogger("biblos.stream_orchestrator")
        self._event_bus = event_bus
        self._initialized = False
        self._running = False

        # State tracking (in-memory cache, backed by Redis)
        self._verse_states: Dict[str, VerseState] = {}

        # Consumer ID for this orchestrator instance
        self._consumer_id = f"{self.config.consumer_prefix}-{uuid.uuid4().hex[:8]}"

        # Background tasks
        self._tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()

        # Metrics
        self._metrics = {
            "verses_received": 0,
            "verses_completed": 0,
            "verses_failed": 0,
            "phases_completed": 0,
            "current_processing": 0,
        }

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def initialize(self) -> None:
        """Initialize the stream orchestrator."""
        if self._initialized:
            return

        self.logger.info(f"Initializing stream orchestrator: {self._consumer_id}")

        # Get or create event bus
        if self._event_bus is None:
            self._event_bus = await get_event_bus()

        # Create consumer groups for all phase completion topics
        await self._setup_consumer_groups()

        # Load any in-progress verse states from Redis
        if self.config.enable_recovery:
            await self._recover_state()

        self._initialized = True
        self.logger.info("Stream orchestrator initialized")

    async def _setup_consumer_groups(self) -> None:
        """Set up consumer groups for orchestrator topics."""
        # Listen to verse ingested events
        await self._event_bus.ensure_consumer_group(
            StreamTopic.VERSE_INGESTED,
            self.config.consumer_group,
        )

        # Listen to all phase completion events
        for phase_name, topics in PHASE_TOPICS.items():
            await self._event_bus.ensure_consumer_group(
                topics["complete"],
                self.config.consumer_group,
            )

        self.logger.info("Consumer groups configured")

    async def start(self) -> None:
        """Start the orchestrator event loop."""
        if self._running:
            return

        if not self._initialized:
            await self.initialize()

        self._running = True
        self._shutdown_event.clear()

        self.logger.info("Starting stream orchestrator event loops")

        # Start verse ingestion consumer
        self._tasks.append(
            asyncio.create_task(
                self._verse_ingestion_loop(),
                name="verse_ingestion_loop"
            )
        )

        # Start phase completion consumers
        for phase_name in self.config.phases:
            self._tasks.append(
                asyncio.create_task(
                    self._phase_completion_loop(phase_name),
                    name=f"phase_completion_{phase_name}"
                )
            )

        # Start monitoring tasks
        if self.config.heartbeat_interval > 0:
            self._tasks.append(
                asyncio.create_task(
                    self._heartbeat_loop(),
                    name="heartbeat_loop"
                )
            )

        # Start stale verse cleanup
        self._tasks.append(
            asyncio.create_task(
                self._stale_verse_cleanup_loop(),
                name="stale_cleanup_loop"
            )
        )

        self.logger.info(f"Started {len(self._tasks)} background tasks")

    async def stop(self) -> None:
        """Stop the orchestrator gracefully."""
        if not self._running:
            return

        self.logger.info("Stopping stream orchestrator...")
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

        # Checkpoint all in-progress states
        await self._checkpoint_all_states()

        self.logger.info("Stream orchestrator stopped")

    async def shutdown(self) -> None:
        """Full shutdown including event bus."""
        await self.stop()
        self._initialized = False

    # -------------------------------------------------------------------------
    # Verse Ingestion
    # -------------------------------------------------------------------------

    async def ingest_verse(
        self,
        verse_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Ingest a verse for processing.

        Args:
            verse_id: Verse identifier
            text: Verse text
            metadata: Optional metadata

        Returns:
            Correlation ID for tracking
        """
        if not self._initialized:
            await self.initialize()

        # Normalize verse ID
        verse_id = normalize_verse_id(verse_id)
        correlation_id = str(uuid.uuid4())

        # Create verse event
        event = VerseEvent.create(
            verse_id=verse_id,
            text=text,
            event_type="verse_ingested",
            metadata=metadata,
            correlation_id=correlation_id,
        )

        # Publish to verse ingested stream
        await self._event_bus.publish(StreamTopic.VERSE_INGESTED, event)

        self.logger.info(
            f"Ingested verse {verse_id} with correlation_id {correlation_id}"
        )

        self._metrics["verses_received"] += 1

        return correlation_id

    async def ingest_batch(
        self,
        verses: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Ingest multiple verses for processing.

        Args:
            verses: List of verse dictionaries with verse_id, text, metadata

        Returns:
            List of correlation IDs
        """
        correlation_ids = []

        for verse in verses:
            cid = await self.ingest_verse(
                verse_id=verse["verse_id"],
                text=verse["text"],
                metadata=verse.get("metadata"),
            )
            correlation_ids.append(cid)

        return correlation_ids

    async def _verse_ingestion_loop(self) -> None:
        """Consumer loop for verse ingestion events."""
        self.logger.info("Starting verse ingestion loop")

        async for message_id, event in self._event_bus.subscribe(
            StreamTopic.VERSE_INGESTED,
            self.config.consumer_group,
            self._consumer_id,
        ):
            if self._shutdown_event.is_set():
                break

            try:
                await self._handle_verse_ingested(event)

                # Acknowledge the message
                await self._event_bus.acknowledge(
                    StreamTopic.VERSE_INGESTED,
                    self.config.consumer_group,
                    message_id,
                )

            except Exception as e:
                self.logger.error(f"Error handling verse ingestion: {e}")
                # Schedule retry
                await self._event_bus.schedule_retry(
                    event,
                    StreamTopic.VERSE_INGESTED,
                    str(e),
                )

    async def _handle_verse_ingested(self, event: EventMessage) -> None:
        """Handle a verse ingested event."""
        payload = event.payload
        verse_id = payload["verse_id"]
        text = payload["text"]
        metadata = payload.get("metadata", {})

        self.logger.info(f"Processing verse ingestion: {verse_id}")

        # Create verse state
        state = VerseState(
            verse_id=verse_id,
            text=text,
            correlation_id=event.correlation_id,
            status="processing",
            metadata=metadata,
        )

        # Store state
        self._verse_states[event.correlation_id] = state
        await self._save_verse_state(state)

        self._metrics["current_processing"] += 1

        # Start first phase
        await self._dispatch_next_phase(state)

    # -------------------------------------------------------------------------
    # Phase Coordination
    # -------------------------------------------------------------------------

    async def _dispatch_next_phase(self, state: VerseState) -> None:
        """Dispatch the next phase for a verse."""
        next_phase = state.get_next_phase(
            self.config.phases,
            self.config.phase_dependencies,
        )

        if next_phase is None:
            # All phases complete
            await self._complete_verse(state)
            return

        self.logger.info(
            f"Dispatching phase '{next_phase}' for verse {state.verse_id}"
        )

        # Update state
        state.current_phase = next_phase
        state.updated_at = time.time()
        await self._save_verse_state(state)

        # Get phase topics
        phase_topics = PHASE_TOPICS.get(next_phase)
        if not phase_topics:
            self.logger.error(f"Unknown phase: {next_phase}")
            return

        # Create phase request event
        request_event = PhaseRequestEvent.create(
            phase_name=next_phase,
            verse_id=state.verse_id,
            text=state.text,
            context=state.to_context(),
            correlation_id=state.correlation_id,
        )

        # Publish to phase request stream
        await self._event_bus.publish(phase_topics["request"], request_event)

    async def _phase_completion_loop(self, phase_name: str) -> None:
        """Consumer loop for phase completion events."""
        self.logger.info(f"Starting phase completion loop for '{phase_name}'")

        phase_topics = PHASE_TOPICS.get(phase_name)
        if not phase_topics:
            self.logger.error(f"Unknown phase: {phase_name}")
            return

        complete_topic = phase_topics["complete"]

        async for message_id, event in self._event_bus.subscribe(
            complete_topic,
            self.config.consumer_group,
            self._consumer_id,
        ):
            if self._shutdown_event.is_set():
                break

            try:
                await self._handle_phase_complete(event, phase_name)

                # Acknowledge
                await self._event_bus.acknowledge(
                    complete_topic,
                    self.config.consumer_group,
                    message_id,
                )

            except Exception as e:
                self.logger.error(f"Error handling phase completion: {e}")
                # Don't retry phase completions - let the phase handle retry

    async def _handle_phase_complete(
        self,
        event: EventMessage,
        phase_name: str,
    ) -> None:
        """Handle a phase completion event."""
        payload = event.payload
        verse_id = payload["verse_id"]
        status = payload["status"]
        result = payload.get("result", {})
        error = payload.get("error")

        correlation_id = event.correlation_id

        self.logger.info(
            f"Phase '{phase_name}' complete for {verse_id}: status={status}"
        )

        # Get verse state
        state = self._verse_states.get(correlation_id)
        if not state:
            # Try to recover from Redis
            state = await self._load_verse_state(correlation_id)
            if not state:
                self.logger.warning(
                    f"No state found for correlation_id {correlation_id}"
                )
                return
            self._verse_states[correlation_id] = state

        # Update state based on phase result
        if status == "completed":
            state.completed_phases.append(phase_name)
            state.phase_results[phase_name] = result
            state.current_phase = None
            self._metrics["phases_completed"] += 1
        elif status == "failed":
            state.errors.append(f"{phase_name}: {error or 'Unknown error'}")

            # Check retry count
            if state.retry_count < 3:
                state.retry_count += 1
                self.logger.info(
                    f"Retrying phase '{phase_name}' for {verse_id} "
                    f"(attempt {state.retry_count})"
                )
                # Re-dispatch the same phase
                await self._dispatch_next_phase(state)
                return
            else:
                # Mark verse as failed
                await self._fail_verse(state, f"Phase '{phase_name}' failed after retries")
                return

        state.updated_at = time.time()
        await self._save_verse_state(state)

        # Dispatch next phase
        await self._dispatch_next_phase(state)

    # -------------------------------------------------------------------------
    # Verse Completion
    # -------------------------------------------------------------------------

    async def _complete_verse(self, state: VerseState) -> None:
        """Mark a verse as complete and emit completion event."""
        state.status = "completed"
        state.completed_at = time.time()
        state.updated_at = time.time()

        self.logger.info(
            f"Verse {state.verse_id} completed in "
            f"{state.completed_at - state.started_at:.2f}s"
        )

        # Build golden record
        golden_record = self._build_golden_record(state)

        # Emit completion event
        complete_event = EventMessage(
            event_type="verse_completed",
            payload={
                "verse_id": state.verse_id,
                "correlation_id": state.correlation_id,
                "golden_record": golden_record,
                "processing_time": state.completed_at - state.started_at,
                "phases_executed": state.completed_phases,
            },
            correlation_id=state.correlation_id,
        )

        await self._event_bus.publish(StreamTopic.VERSE_COMPLETED, complete_event)

        # Update metrics
        self._metrics["verses_completed"] += 1
        self._metrics["current_processing"] -= 1

        # Cleanup state
        await self._cleanup_verse_state(state.correlation_id)

        # Checkpoint if needed
        if self._metrics["verses_completed"] % self.config.checkpoint_interval == 0:
            await self._checkpoint_all_states()

    async def _fail_verse(self, state: VerseState, error: str) -> None:
        """Mark a verse as failed."""
        state.status = "failed"
        state.completed_at = time.time()
        state.updated_at = time.time()
        state.errors.append(error)

        self.logger.error(f"Verse {state.verse_id} failed: {error}")

        # Emit failure event
        fail_event = EventMessage(
            event_type="verse_failed",
            payload={
                "verse_id": state.verse_id,
                "correlation_id": state.correlation_id,
                "error": error,
                "errors": state.errors,
                "completed_phases": state.completed_phases,
                "processing_time": state.completed_at - state.started_at,
            },
            correlation_id=state.correlation_id,
        )

        await self._event_bus.publish(StreamTopic.VERSE_FAILED, fail_event)

        # Update metrics
        self._metrics["verses_failed"] += 1
        self._metrics["current_processing"] -= 1

        # Send to DLQ
        original_event = EventMessage(
            event_type="verse_processing",
            payload=state.to_dict(),
            correlation_id=state.correlation_id,
        )
        await self._event_bus.send_to_dlq(
            original_event,
            error,
            StreamTopic.VERSE_INGESTED,
        )

        # Cleanup
        await self._cleanup_verse_state(state.correlation_id)

    def _build_golden_record(self, state: VerseState) -> Dict[str, Any]:
        """Build a golden record from completed verse state."""
        # Calculate confidence
        confidences = []
        for phase_result in state.phase_results.values():
            metrics = phase_result.get("metrics", {})
            if "phase_confidence" in metrics:
                confidences.append(metrics["phase_confidence"])

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Determine certification level
        if avg_confidence >= 0.9:
            level = "gold"
        elif avg_confidence >= 0.75:
            level = "silver"
        elif avg_confidence >= 0.5:
            level = "bronze"
        else:
            level = "provisional"

        # Flatten agent results
        agent_results = state._flatten_agent_results()

        return {
            "verse_id": state.verse_id,
            "text": state.text,
            "certification": {
                "level": level,
                "score": avg_confidence,
                "validation_passed": "validation" in state.completed_phases,
                "quality_passed": avg_confidence >= 0.7,
            },
            "data": {
                "structural": {
                    k: v for k, v in agent_results.items()
                    if k in ["grammateus", "syntaktikos"]
                },
                "morphological": agent_results.get("morphologos", {}),
                "semantic": agent_results.get("semantikos", {}),
                "theological": {
                    k: v for k, v in agent_results.items()
                    if k in ["patrologos", "typologos", "theologos", "liturgikos", "dogmatikos"]
                },
                "cross_references": {
                    k: v for k, v in agent_results.items()
                    if k in ["syndesmos", "harmonikos", "allographos", "paradeigma", "topos"]
                },
            },
            "phases_executed": state.completed_phases,
            "agent_count": len(agent_results),
            "total_processing_time": (state.completed_at or time.time()) - state.started_at,
        }

    # -------------------------------------------------------------------------
    # State Persistence (Redis-backed)
    # -------------------------------------------------------------------------

    async def _save_verse_state(self, state: VerseState) -> None:
        """Save verse state to Redis."""
        key = f"biblos:verse_state:{state.correlation_id}"

        import json
        await self._event_bus._redis.set(
            key,
            json.dumps(state.to_dict()),
            ex=3600,  # 1 hour TTL
        )

    async def _load_verse_state(self, correlation_id: str) -> Optional[VerseState]:
        """Load verse state from Redis."""
        key = f"biblos:verse_state:{correlation_id}"

        import json
        data = await self._event_bus._redis.get(key)

        if data:
            return VerseState.from_dict(json.loads(data))
        return None

    async def _cleanup_verse_state(self, correlation_id: str) -> None:
        """Remove verse state from Redis and memory."""
        key = f"biblos:verse_state:{correlation_id}"

        await self._event_bus._redis.delete(key)
        self._verse_states.pop(correlation_id, None)

    async def _checkpoint_all_states(self) -> None:
        """Checkpoint all in-progress verse states."""
        self.logger.info(f"Checkpointing {len(self._verse_states)} verse states")

        for state in self._verse_states.values():
            if state.status == "processing":
                await self._save_verse_state(state)

    async def _recover_state(self) -> None:
        """Recover in-progress verses from Redis."""
        self.logger.info("Recovering verse states from Redis")

        pattern = "biblos:verse_state:*"
        cursor = 0
        recovered = 0

        while True:
            cursor, keys = await self._event_bus._redis.scan(
                cursor=cursor,
                match=pattern,
                count=100,
            )

            for key in keys:
                import json
                data = await self._event_bus._redis.get(key)
                if data:
                    state = VerseState.from_dict(json.loads(data))

                    # Only recover processing states
                    if state.status == "processing":
                        self._verse_states[state.correlation_id] = state
                        recovered += 1

                        # Resume processing
                        await self._dispatch_next_phase(state)

            if cursor == 0:
                break

        if recovered > 0:
            self.logger.info(f"Recovered {recovered} in-progress verses")

    # -------------------------------------------------------------------------
    # Background Tasks
    # -------------------------------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        """Periodic heartbeat publishing."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                await self._event_bus.publish_heartbeat()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")

    async def _stale_verse_cleanup_loop(self) -> None:
        """Cleanup stale verses that have been processing too long."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Check every minute

                now = time.time()
                stale_ids = []

                for correlation_id, state in self._verse_states.items():
                    if state.status == "processing":
                        age = now - state.updated_at
                        if age > self.config.stale_verse_timeout:
                            stale_ids.append(correlation_id)

                for correlation_id in stale_ids:
                    state = self._verse_states[correlation_id]
                    self.logger.warning(
                        f"Verse {state.verse_id} is stale, failing"
                    )
                    await self._fail_verse(state, "Processing timeout")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Stale cleanup error: {e}")

    # -------------------------------------------------------------------------
    # Monitoring & Queries
    # -------------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Get current orchestrator metrics."""
        return {
            **self._metrics,
            "in_memory_states": len(self._verse_states),
            "consumer_id": self._consumer_id,
            "running": self._running,
        }

    async def get_verse_status(
        self,
        correlation_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get the current status of a verse by correlation ID."""
        state = self._verse_states.get(correlation_id)
        if not state:
            state = await self._load_verse_state(correlation_id)

        if state:
            return state.to_dict()
        return None

    async def list_processing_verses(self) -> List[Dict[str, Any]]:
        """List all currently processing verses."""
        return [
            state.to_dict()
            for state in self._verse_states.values()
            if state.status == "processing"
        ]

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        event_bus_health = await self._event_bus.health_check()

        return {
            "healthy": event_bus_health.get("healthy", False) and self._running,
            "running": self._running,
            "initialized": self._initialized,
            "consumer_id": self._consumer_id,
            "active_tasks": len(self._tasks),
            "processing_verses": len([
                s for s in self._verse_states.values()
                if s.status == "processing"
            ]),
            "metrics": self._metrics,
            "event_bus": event_bus_health,
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

async def create_stream_orchestrator(
    config: Optional[StreamOrchestratorConfig] = None,
) -> StreamOrchestrator:
    """Create and initialize a stream orchestrator."""
    orchestrator = StreamOrchestrator(config)
    await orchestrator.initialize()
    return orchestrator


# =============================================================================
# RESULT TYPE FOR SYNC INTERFACE
# =============================================================================

@dataclass
class StreamPipelineResult:
    """Result from stream-based pipeline processing."""
    verse_id: str
    correlation_id: str
    status: str
    golden_record: Optional[Dict[str, Any]]
    processing_time: float
    phases_executed: List[str]
    errors: List[str]

    @classmethod
    def from_verse_state(cls, state: VerseState) -> "StreamPipelineResult":
        """Create from verse state."""
        return cls(
            verse_id=state.verse_id,
            correlation_id=state.correlation_id,
            status=state.status,
            golden_record=None,  # Built separately
            processing_time=(state.completed_at or time.time()) - state.started_at,
            phases_executed=state.completed_phases,
            errors=state.errors,
        )
