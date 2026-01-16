"""
Unified Orchestrator

Central orchestrator integrating all BIBLOS v2 components.
Manages phase execution with circuit breakers and metrics.
"""
import time
import logging
from typing import Dict, List, Optional, Set, Any, Coroutine
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from uuid import uuid4

from pipeline.context import ProcessingContext, PhaseState
from pipeline.golden_record import GoldenRecord, GoldenRecordBuilder
from pipeline.phases.linguistic import LinguisticPhase
from pipeline.phases.theological import TheologicalPhase
from pipeline.phases.intertextual import IntertextualPhase
from pipeline.phases.cross_reference import CrossReferencePhase
from pipeline.phases.validation import ValidationPhase


logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states for component health."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery

    @property
    def allows_requests(self) -> bool:
        return self in {CircuitState.CLOSED, CircuitState.HALF_OPEN}


@dataclass
class CircuitBreaker:
    """Circuit breaker for component failure isolation."""
    component_name: str
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    failure_threshold: int = 5
    reset_timeout_seconds: int = 60
    last_failure_time: Optional[datetime] = None

    def record_failure(self) -> None:
        """Record a failure and potentially open circuit."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

    def record_success(self) -> None:
        """Record success and close circuit if half-open."""
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
        self.failure_count = 0

    def should_allow_request(self) -> bool:
        """Check if request should be allowed."""
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            # Check if reset timeout has passed
            if self.last_failure_time:
                elapsed = (datetime.utcnow() - self.last_failure_time).total_seconds()
                if elapsed >= self.reset_timeout_seconds:
                    self.state = CircuitState.HALF_OPEN
                    return True
            return False
        return True  # HALF_OPEN allows one request


class OrchestratorMetric(Enum):
    """Metrics tracked by orchestrator."""
    VERSES_PROCESSED = "verses_processed"
    PHASES_COMPLETED = "phases_completed"
    ORACLE_INVOCATIONS = "oracle_invocations"
    CROSS_REFS_DISCOVERED = "cross_refs_discovered"
    VALIDATION_REJECTIONS = "validation_rejections"
    AVG_PHASE_DURATION_MS = "avg_phase_duration_ms"

    @property
    def aggregation_type(self) -> str:
        """How to aggregate this metric."""
        return {
            OrchestratorMetric.VERSES_PROCESSED: "counter",
            OrchestratorMetric.PHASES_COMPLETED: "counter",
            OrchestratorMetric.ORACLE_INVOCATIONS: "counter",
            OrchestratorMetric.CROSS_REFS_DISCOVERED: "counter",
            OrchestratorMetric.VALIDATION_REJECTIONS: "counter",
            OrchestratorMetric.AVG_PHASE_DURATION_MS: "gauge",
        }[self]


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class UnifiedOrchestrator:
    """
    Central orchestrator integrating all BIBLOS v2 components.
    Manages phase execution with circuit breakers and metrics.
    """

    def __init__(
        self,
        # Event infrastructure
        event_store=None,
        command_handler=None,
        event_publisher=None,

        # Database clients
        postgres_client=None,
        neo4j_client=None,
        vector_store=None,
        redis_client=None,

        # Oracle engines (Sessions 03-07)
        omni_resolver=None,
        necessity_calculator=None,
        lxx_extractor=None,
        typology_engine=None,
        prophetic_prover=None,

        # Core ML components
        mutual_transformation=None,
        theological_validator=None,
        gnn_model=None,
        inference_pipeline=None,

        # Configuration
        config=None
    ):
        self.event_store = event_store
        self.command_handler = command_handler
        self.event_publisher = event_publisher
        self.postgres = postgres_client
        self.neo4j = neo4j_client
        self.vector_store = vector_store
        self.redis = redis_client

        self.omni_resolver = omni_resolver
        self.necessity_calculator = necessity_calculator
        self.lxx_extractor = lxx_extractor
        self.typology_engine = typology_engine
        self.prophetic_prover = prophetic_prover

        self.mutual_transformation = mutual_transformation
        self.theological_validator = theological_validator
        self.gnn_model = gnn_model
        self.inference_pipeline = inference_pipeline

        self.config = config

        # Phase executors with dependency order
        self.phases = [
            LinguisticPhase(self),
            TheologicalPhase(self),
            IntertextualPhase(self),
            CrossReferencePhase(self),
            ValidationPhase(self)
        ]

        # Circuit breakers for each component
        self._circuit_breakers: Dict[str, CircuitBreaker] = {
            "neo4j": CircuitBreaker("neo4j"),
            "postgres": CircuitBreaker("postgres"),
            "vector_store": CircuitBreaker("vector_store"),
            "omni_resolver": CircuitBreaker("omni_resolver"),
            "lxx_extractor": CircuitBreaker("lxx_extractor"),
            "typology_engine": CircuitBreaker("typology_engine"),
            "gnn_model": CircuitBreaker("gnn_model"),
        }

        # Metrics tracking
        self._metrics: Dict[OrchestratorMetric, float] = {
            m: 0.0 for m in OrchestratorMetric
        }
        self._phase_durations: Dict[str, List[float]] = defaultdict(list)

        # Golden record builder
        self._golden_record_builder = GoldenRecordBuilder(
            db_client=postgres_client,
            neo4j_client=neo4j_client
        )

    def get_circuit_breaker(self, component: str) -> CircuitBreaker:
        """Get circuit breaker for a component."""
        return self._circuit_breakers.get(component, CircuitBreaker(component))

    async def _execute_with_circuit_breaker(
        self,
        component: str,
        coro: Coroutine
    ) -> Any:
        """Execute coroutine with circuit breaker protection."""
        breaker = self.get_circuit_breaker(component)
        if not breaker.should_allow_request():
            raise CircuitOpenError(f"Circuit open for {component}")
        try:
            result = await coro
            breaker.record_success()
            return result
        except Exception as e:
            breaker.record_failure()
            raise

    async def process_verse(
        self,
        verse_id: str,
        correlation_id: Optional[str] = None,
        skip_phases: Optional[Set[str]] = None
    ) -> GoldenRecord:
        """
        Process a single verse through all phases.
        Supports phase skipping for partial reprocessing.

        Args:
            verse_id: Verse identifier (e.g., "GEN.1.1")
            correlation_id: Optional correlation ID for tracking
            skip_phases: Set of phase names to skip

        Returns:
            Complete GoldenRecord for the verse
        """
        correlation_id = correlation_id or str(uuid4())
        skip_phases = skip_phases or set()

        logger.info(f"Processing verse {verse_id} with correlation_id {correlation_id}")

        # Emit processing started event
        if self.event_publisher:
            from db.events import VerseProcessingStarted
            await self.event_publisher.publish(VerseProcessingStarted(
                aggregate_id=verse_id,
                correlation_id=correlation_id,
                verse_id=verse_id,
                phase_plan=[p.name for p in self.phases if p.name not in skip_phases]
            ))

        try:
            # Execute all phases with state tracking
            context = ProcessingContext(
                verse_id=verse_id,
                correlation_id=correlation_id,
                phase_states={p.name: PhaseState.PENDING for p in self.phases}
            )

            for phase in self.phases:
                if phase.name in skip_phases:
                    context.phase_states[phase.name] = PhaseState.SKIPPED
                    logger.debug(f"Skipping phase {phase.name}")
                    continue

                context.phase_states[phase.name] = PhaseState.RUNNING
                phase_start_time = time.time()

                try:
                    logger.debug(f"Executing phase {phase.name}")
                    context = await phase.execute(context)
                    context.phase_states[phase.name] = PhaseState.COMPLETED
                    duration_ms = (time.time() - phase_start_time) * 1000
                    context.phase_durations[phase.name] = duration_ms
                    self._phase_durations[phase.name].append(duration_ms)

                    logger.debug(f"Phase {phase.name} completed in {duration_ms:.0f}ms")
                except Exception as phase_error:
                    context.phase_states[phase.name] = PhaseState.FAILED
                    context.add_error(phase.name, phase_error)
                    logger.error(f"Phase {phase.name} failed: {phase_error}", exc_info=True)

                    if phase.is_critical:
                        raise  # Critical phases stop pipeline

            # Build and return Golden Record
            golden_record = await self._build_golden_record(context)
            self._metrics[OrchestratorMetric.VERSES_PROCESSED] += 1

            # Emit processing completed event
            if self.event_publisher:
                from db.events import VerseProcessingCompleted
                await self.event_publisher.publish(VerseProcessingCompleted(
                    aggregate_id=verse_id,
                    correlation_id=correlation_id,
                    verse_id=verse_id,
                    quality_tier=self._determine_quality_tier(context),
                    cross_reference_count=len(context.validated_cross_references),
                    phases_completed=list(context.phase_states.keys())
                ))

            logger.info(f"Successfully processed verse {verse_id}")
            return golden_record

        except Exception as e:
            # Emit failure event
            if self.event_publisher:
                from db.events import VerseProcessingFailed
                await self.event_publisher.publish(VerseProcessingFailed(
                    aggregate_id=verse_id,
                    correlation_id=correlation_id,
                    verse_id=verse_id,
                    error_message=str(e),
                    failed_phase=self._get_failed_phase(context) if 'context' in locals() else "unknown"
                ))

            logger.error(f"Failed to process verse {verse_id}: {e}", exc_info=True)
            raise

    async def _build_golden_record(self, context: ProcessingContext) -> GoldenRecord:
        """Build Golden Record from processing context."""
        return await self._golden_record_builder.build(context)

    def _determine_quality_tier(self, context: ProcessingContext) -> int:
        """Determine quality tier based on completeness."""
        if context.completeness.value == "full":
            return 5
        elif context.completeness.value == "partial":
            return 3
        elif context.completeness.value == "minimal":
            return 1
        else:
            return 0

    def _get_failed_phase(self, context: ProcessingContext) -> Optional[str]:
        """Find which phase failed."""
        for phase_name, state in context.phase_states.items():
            if state == PhaseState.FAILED:
                return phase_name
        return None

    def get_metrics(self) -> Dict[str, Any]:
        """Return current orchestrator metrics."""
        metrics = {m.value: v for m, v in self._metrics.items()}

        # Calculate average phase durations
        for phase_name, durations in self._phase_durations.items():
            if durations:
                metrics[f"avg_{phase_name}_duration_ms"] = sum(durations) / len(durations)

        return metrics

    def get_circuit_breaker_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers."""
        status = {}
        for component, breaker in self._circuit_breakers.items():
            status[component] = {
                "state": breaker.state.value,
                "failure_count": breaker.failure_count,
                "last_failure": breaker.last_failure_time.isoformat() if breaker.last_failure_time else None
            }
        return status
