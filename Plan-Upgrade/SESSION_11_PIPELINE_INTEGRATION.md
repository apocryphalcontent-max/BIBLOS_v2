# SESSION 11: PIPELINE INTEGRATION & ORCHESTRATION

## Session Overview

**Objective**: Integrate all components from Sessions 01-10 into a unified, event-driven pipeline that orchestrates the complete BIBLOS v2 processing flow. This session brings together the Five Impossible Oracles, theological constraints, event sourcing, graph database, and multi-vector search into a cohesive system.

**Estimated Duration**: 1 Claude session (120-150 minutes of focused implementation)

**Prerequisites**:
- ALL previous sessions (01-10) must be complete
- Understanding of existing `pipeline/stream_orchestrator.py`
- Familiarity with LangGraph for agent orchestration
- Understanding of event-driven architecture

---

## Part 1: Integrated Architecture Overview

### System Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           COMMAND INTERFACE                              │
│              (CLI / API / Batch Processor / Event Trigger)               │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          COMMAND HANDLER                                 │
│                    (Issues Events, Manages Transactions)                 │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
┌─────────────────────────────────┐ ┌─────────────────────────────────────┐
│         EVENT STORE              │ │          EVENT PUBLISHER             │
│      (PostgreSQL, Append-Only)   │ │        (Redis Streams)               │
└─────────────────────────────────┘ └───────────────┬─────────────────────┘
                                                    │
                    ┌───────────────────────────────┤
                    ▼                               ▼
┌─────────────────────────────────┐ ┌─────────────────────────────────────┐
│      READ MODEL PROJECTIONS      │ │      REAL-TIME SUBSCRIBERS          │
│  ┌─────────────────────────┐    │ │  ┌─────────────────────────────┐   │
│  │ PostgreSQL Read Models  │    │ │  │  Oracle Engine Triggers      │   │
│  ├─────────────────────────┤    │ │  ├─────────────────────────────┤   │
│  │ Neo4j Graph Projection  │    │ │  │  Validation Triggers         │   │
│  ├─────────────────────────┤    │ │  ├─────────────────────────────┤   │
│  │ Vector Store Projection │    │ │  │  Notification Service        │   │
│  └─────────────────────────┘    │ │  └─────────────────────────────┘   │
└─────────────────────────────────┘ └─────────────────────────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         ORACLE ORCHESTRATOR                              │
│  ┌────────────────────────────────────────────────────────────────────┐│
│  │                    PHASE 1: LINGUISTIC                             ││
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    ││
│  │  │ OmniContextual  │  │   Morphology    │  │   Syntax        │    ││
│  │  │ Resolver        │  │   Analysis      │  │   Analysis      │    ││
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘    ││
│  └────────────────────────────────────────────────────────────────────┘│
│  ┌────────────────────────────────────────────────────────────────────┐│
│  │                   PHASE 2: THEOLOGICAL                             ││
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    ││
│  │  │ Theological     │  │   Patristic     │  │   LXX           │    ││
│  │  │ Constraint Val  │  │   Integration   │  │   Extractor     │    ││
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘    ││
│  └────────────────────────────────────────────────────────────────────┘│
│  ┌────────────────────────────────────────────────────────────────────┐│
│  │                  PHASE 3: INTERTEXTUAL                             ││
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    ││
│  │  │ Fractal         │  │   Necessity     │  │   Prophetic     │    ││
│  │  │ Typology        │  │   Calculator    │  │   Prover        │    ││
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘    ││
│  └────────────────────────────────────────────────────────────────────┘│
│  ┌────────────────────────────────────────────────────────────────────┐│
│  │                   PHASE 4: CROSS-REFERENCE                         ││
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    ││
│  │  │ Multi-Vector    │  │   GNN           │  │   Mutual        │    ││
│  │  │ Search          │  │   Refinement    │  │   Transformation│    ││
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘    ││
│  └────────────────────────────────────────────────────────────────────┘│
│  ┌────────────────────────────────────────────────────────────────────┐│
│  │                   PHASE 5: VALIDATION                              ││
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    ││
│  │  │ Prosecutor      │  │   Witness       │  │   Final         │    ││
│  │  │ Challenges      │  │   Defense       │  │   Judgment      │    ││
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘    ││
│  └────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          GOLDEN RECORD OUTPUT                            │
│      (Complete analysis with all oracle insights, validated)             │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Part 2: Unified Orchestrator Implementation

### File: `pipeline/unified_orchestrator.py`

**Phase State Machine and Circuit Breaker Enums**:

```python
class PhaseState(Enum):
    """State machine for phase execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"

    @property
    def is_terminal(self) -> bool:
        """Check if state is terminal (no further transitions)."""
        return self in {PhaseState.COMPLETED, PhaseState.FAILED, PhaseState.SKIPPED}

    @property
    def allows_retry(self) -> bool:
        """Check if state allows retry."""
        return self == PhaseState.FAILED

    def can_transition_to(self, target: "PhaseState") -> bool:
        """Validate state transition."""
        valid_transitions = {
            PhaseState.PENDING: {PhaseState.RUNNING, PhaseState.SKIPPED},
            PhaseState.RUNNING: {PhaseState.COMPLETED, PhaseState.FAILED},
            PhaseState.FAILED: {PhaseState.RETRYING, PhaseState.SKIPPED},
            PhaseState.RETRYING: {PhaseState.RUNNING},
            PhaseState.COMPLETED: set(),  # Terminal
            PhaseState.SKIPPED: set(),    # Terminal
        }
        return target in valid_transitions.get(self, set())


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
```

**Core Orchestrator Class**:

```python
class UnifiedOrchestrator:
    """
    Central orchestrator integrating all BIBLOS v2 components.
    Manages phase execution with circuit breakers and metrics.
    """

    def __init__(
        self,
        # Event infrastructure
        event_store: EventStore,
        command_handler: CommandHandler,
        event_publisher: EventPublisher,

        # Database clients
        postgres_client: PostgresClient,
        neo4j_client: Neo4jGraphClient,
        vector_store: MultiVectorStore,
        redis_client: RedisClient,

        # Oracle engines (Sessions 03-07)
        omni_resolver: OmniContextualResolver,
        necessity_calculator: InterVerseNecessityCalculator,
        lxx_extractor: LXXChristologicalExtractor,
        typology_engine: HyperFractalTypologyEngine,
        prophetic_prover: PropheticNecessityProver,

        # Core ML components
        mutual_transformation: MutualTransformationMetric,
        theological_validator: TheologicalConstraintValidator,
        gnn_model: CrossRefGNN,
        inference_pipeline: InferencePipeline,

        # Configuration
        config: BiblosConfig
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
        correlation_id: str = None,
        skip_phases: Optional[Set[str]] = None
    ) -> GoldenRecord:
        """
        Process a single verse through all phases.
        Supports phase skipping for partial reprocessing.
        """
        correlation_id = correlation_id or str(uuid4())
        skip_phases = skip_phases or set()

        # Emit processing started event
        await self.event_publisher.publish(VerseProcessingStartedEvent(
            event_id=str(uuid4()),
            aggregate_id=verse_id,
            correlation_id=correlation_id,
            enabled_phases=[p.name for p in self.phases if p.name not in skip_phases]
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
                    continue

                context.phase_states[phase.name] = PhaseState.RUNNING
                start_time = time.time()

                try:
                    context = await phase.execute(context)
                    context.phase_states[phase.name] = PhaseState.COMPLETED
                    duration_ms = (time.time() - start_time) * 1000
                    context.phase_durations[phase.name] = duration_ms
                    self._phase_durations[phase.name].append(duration_ms)
                except Exception as phase_error:
                    context.phase_states[phase.name] = PhaseState.FAILED
                    context.add_error(phase.name, phase_error)
                    if phase.is_critical:
                        raise  # Critical phases stop pipeline

                # Emit phase completion event
                await self.event_publisher.publish(PhaseCompletedEvent(
                    phase_name=phase.name,
                    verse_id=verse_id,
                    correlation_id=correlation_id,
                    state=context.phase_states[phase.name].value,
                    duration_ms=context.phase_durations.get(phase.name, 0)
                ))

            # Build and return Golden Record
            golden_record = await self._build_golden_record(context)
            self._metrics[OrchestratorMetric.VERSES_PROCESSED] += 1

            # Emit processing completed event
            await self.event_publisher.publish(VerseProcessingCompletedEvent(
                verse_id=verse_id,
                correlation_id=correlation_id,
                success=True,
                cross_ref_count=len(context.validated_cross_references)
            ))

            return golden_record

        except Exception as e:
            # Emit failure event
            await self.event_publisher.publish(VerseProcessingFailedEvent(
                verse_id=verse_id,
                correlation_id=correlation_id,
                error_type=type(e).__name__,
                error_message=str(e),
                failed_phase=self._get_failed_phase(context)
            ))
            raise

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
```

---

## Part 3: Phase Implementations

### File: `pipeline/phases/base.py`

**Phase Base Class with Dependency Management**:

```python
class PhasePriority(Enum):
    """Execution priority for phases."""
    CRITICAL = 1      # Must complete, blocks pipeline
    HIGH = 2          # Important but gracefully degradable
    NORMAL = 3        # Standard processing
    OPTIONAL = 4      # Can be skipped without impact

    @property
    def timeout_multiplier(self) -> float:
        """Timeout scaling based on priority."""
        return {
            PhasePriority.CRITICAL: 2.0,
            PhasePriority.HIGH: 1.5,
            PhasePriority.NORMAL: 1.0,
            PhasePriority.OPTIONAL: 0.5,
        }[self]


class PhaseCategory(Enum):
    """Category classification for phases."""
    LINGUISTIC = "linguistic"
    THEOLOGICAL = "theological"
    INTERTEXTUAL = "intertextual"
    CROSS_REFERENCE = "cross_reference"
    VALIDATION = "validation"

    @property
    def typical_oracles(self) -> List[str]:
        """Which oracles typically run in this category."""
        return {
            PhaseCategory.LINGUISTIC: ["omni_resolver"],
            PhaseCategory.THEOLOGICAL: ["lxx_extractor"],
            PhaseCategory.INTERTEXTUAL: ["typology_engine", "necessity_calculator", "prophetic_prover"],
            PhaseCategory.CROSS_REFERENCE: ["gnn_model", "vector_store"],
            PhaseCategory.VALIDATION: ["theological_validator"],
        }[self]


@dataclass
class PhaseDependency:
    """Declares a dependency between phases."""
    phase_name: str
    required_outputs: List[str]  # Fields required from that phase
    is_hard: bool = True  # Hard dependency blocks; soft allows degradation


class Phase(ABC):
    """
    Abstract base class for pipeline phases.
    Provides dependency tracking and execution lifecycle.
    """
    name: str
    category: PhaseCategory
    priority: PhasePriority = PhasePriority.NORMAL
    is_critical: bool = False
    base_timeout_seconds: float = 30.0

    def __init__(self, orchestrator: "UnifiedOrchestrator"):
        self.orchestrator = orchestrator

    @property
    def effective_timeout(self) -> float:
        """Calculate timeout based on priority."""
        return self.base_timeout_seconds * self.priority.timeout_multiplier

    @property
    @abstractmethod
    def dependencies(self) -> List[PhaseDependency]:
        """Declare phase dependencies."""
        pass

    @property
    @abstractmethod
    def outputs(self) -> List[str]:
        """Declare what this phase produces in context."""
        pass

    def check_dependencies_satisfied(self, context: ProcessingContext) -> bool:
        """Check if all hard dependencies are satisfied."""
        for dep in self.dependencies:
            if not dep.is_hard:
                continue
            phase_state = context.phase_states.get(dep.phase_name)
            if phase_state != PhaseState.COMPLETED:
                return False
            # Check required outputs exist
            for output in dep.required_outputs:
                if not hasattr(context, output) or getattr(context, output) is None:
                    return False
        return True

    @abstractmethod
    async def execute(self, context: ProcessingContext) -> ProcessingContext:
        """Execute the phase logic."""
        pass

    async def execute_with_timeout(self, context: ProcessingContext) -> ProcessingContext:
        """Execute with timeout protection."""
        return await asyncio.wait_for(
            self.execute(context),
            timeout=self.effective_timeout
        )
```

### File: `pipeline/phases/`

#### Linguistic Phase
```python
class LinguisticPhase(Phase):
    """
    Phase 1: Linguistic analysis including OmniContextual resolution.
    """
    name = "linguistic"
    category = PhaseCategory.LINGUISTIC
    priority = PhasePriority.CRITICAL
    is_critical = True
    base_timeout_seconds = 45.0

    @property
    def dependencies(self) -> List[PhaseDependency]:
        return []  # First phase - no dependencies

    @property
    def outputs(self) -> List[str]:
        return ["linguistic_analysis", "resolved_meanings"]

    async def execute(self, context: ProcessingContext) -> ProcessingContext:
        start_time = time.time()

        # Get verse text
        verse = await self.orchestrator.postgres.get_verse(context.verse_id)

        # Run OmniContextual Resolver for polysemous words
        language = "hebrew" if verse.testament == "OT" else "greek"
        words = await self.orchestrator.corpus.get_verse_words(context.verse_id)

        resolved_meanings = {}
        for word in words:
            if self._is_polysemous(word.lemma, language):
                result = await self.orchestrator.omni_resolver.resolve_absolute_meaning(
                    word=word.surface,
                    verse_id=context.verse_id,
                    language=language
                )
                resolved_meanings[word.position] = result

                # Emit resolution event
                await self.orchestrator.event_publisher.publish(
                    OmniContextualResolutionEvent(
                        word=word.surface,
                        verse_id=context.verse_id,
                        resolved_meaning=result.primary_meaning,
                        confidence=result.confidence
                    )
                )

        # Store in context
        context.linguistic_analysis = {
            "words": words,
            "resolved_meanings": resolved_meanings,
            "morphology": await self._get_morphology(context.verse_id),
            "syntax": await self._get_syntax(context.verse_id)
        }

        context.phase_durations[self.name] = (time.time() - start_time) * 1000
        return context
```

#### Theological Phase
```python
class TheologicalPhase(Phase):
    """
    Phase 2: Theological analysis including LXX extraction and patristic integration.
    """
    name = "theological"
    category = PhaseCategory.THEOLOGICAL
    priority = PhasePriority.HIGH
    is_critical = False  # Can degrade gracefully
    base_timeout_seconds = 60.0

    @property
    def dependencies(self) -> List[PhaseDependency]:
        return [
            PhaseDependency(
                phase_name="linguistic",
                required_outputs=["linguistic_analysis"],
                is_hard=True
            )
        ]

    @property
    def outputs(self) -> List[str]:
        return ["lxx_analysis", "patristic_witness", "embeddings"]

    async def execute(self, context: ProcessingContext) -> ProcessingContext:
        start_time = time.time()

        # Determine testament from verse ID
        context.testament = self._determine_testament(context.verse_id)

        # LXX Christological Analysis (OT only)
        if context.testament == "OT":
            try:
                lxx_result = await self.orchestrator._execute_with_circuit_breaker(
                    "lxx_extractor",
                    self.orchestrator.lxx_extractor.extract_christological_content(
                        context.verse_id
                    )
                )
                context.lxx_analysis = lxx_result

                if lxx_result.christological_divergence_count > 0:
                    await self.orchestrator.event_publisher.publish(
                        LXXDivergenceDetectedEvent(
                            verse_id=context.verse_id,
                            divergence_count=lxx_result.christological_divergence_count,
                            primary_insight=lxx_result.primary_christological_insight
                        )
                    )
            except CircuitOpenError:
                context.add_warning(self.name, "LXX extractor circuit open, skipping")

        # Gather patristic interpretations with fallback
        try:
            patristic = await self.orchestrator.patristic_db.get_interpretations(context.verse_id)
            context.patristic_witness = patristic
        except Exception as e:
            context.add_warning(self.name, f"Patristic lookup failed: {e}")
            context.patristic_witness = []

        # Generate theological embeddings (multi-domain)
        if context.patristic_witness:
            patristic_embedding = await self.orchestrator.vector_store.generate_patristic_embedding(
                context.verse_id, context.patristic_witness
            )
            context.embeddings["patristic"] = patristic_embedding

        # Also generate liturgical embedding if relevant
        liturgical_refs = await self._get_liturgical_references(context.verse_id)
        if liturgical_refs:
            liturgical_embedding = await self.orchestrator.vector_store.generate_liturgical_embedding(
                context.verse_id, liturgical_refs
            )
            context.embeddings["liturgical"] = liturgical_embedding

        context.phase_durations[self.name] = (time.time() - start_time) * 1000
        return context

    def _determine_testament(self, verse_id: str) -> str:
        """Determine testament from verse ID."""
        book = verse_id.split(".")[0]
        ot_books = {"GEN", "EXO", "LEV", "NUM", "DEU", "JOS", "JDG", "RUT", "1SA", "2SA",
                    "1KI", "2KI", "1CH", "2CH", "EZR", "NEH", "EST", "JOB", "PSA", "PRO",
                    "ECC", "SNG", "ISA", "JER", "LAM", "EZK", "DAN", "HOS", "JOL", "AMO",
                    "OBA", "JON", "MIC", "NAM", "HAB", "ZEP", "HAG", "ZEC", "MAL"}
        return "OT" if book in ot_books else "NT"

    async def _get_liturgical_references(self, verse_id: str) -> List[Dict]:
        """Check for liturgical usage of this verse."""
        return await self.orchestrator.postgres.query(
            "SELECT * FROM liturgical_readings WHERE verse_id = $1",
            verse_id
        )
```

#### Intertextual Phase
```python
class IntertextualPhase(Phase):
    """
    Phase 3: Intertextual analysis with typology, necessity, and prophetic proving.
    Coordinates the three Impossible Oracles: Typology, Necessity, and Prophetic.
    """
    name = "intertextual"
    category = PhaseCategory.INTERTEXTUAL
    priority = PhasePriority.HIGH
    is_critical = False
    base_timeout_seconds = 90.0  # Longest phase due to multiple oracle invocations

    # Threshold for necessity calculation
    TYPOLOGY_STRENGTH_THRESHOLD = 0.7

    @property
    def dependencies(self) -> List[PhaseDependency]:
        return [
            PhaseDependency(
                phase_name="linguistic",
                required_outputs=["linguistic_analysis"],
                is_hard=True
            ),
            PhaseDependency(
                phase_name="theological",
                required_outputs=["lxx_analysis"],
                is_hard=False  # Soft - can proceed without LXX
            )
        ]

    @property
    def outputs(self) -> List[str]:
        return ["typological_connections", "prophetic_analysis"]

    async def execute(self, context: ProcessingContext) -> ProcessingContext:
        start_time = time.time()

        # Track oracle invocations for metrics
        oracle_invocations = 0

        # Fractal Typology Analysis with circuit breaker
        try:
            typology_results = await self.orchestrator._execute_with_circuit_breaker(
                "typology_engine",
                self.orchestrator.typology_engine.discover_fractal_patterns(
                    context.verse_id
                )
            )
            oracle_invocations += 1
            context.typological_connections = typology_results

            # Emit events for significant discoveries
            for result in typology_results:
                if result.composite_strength > 0.5:  # Only notable connections
                    await self.orchestrator.event_publisher.publish(
                        TypologicalConnectionIdentifiedEvent(
                            type_ref=result.type_reference,
                            antitype_ref=result.antitype_reference,
                            fractal_depth=result.fractal_depth,
                            composite_strength=result.composite_strength,
                            pattern_type=result.pattern_signature
                        )
                    )
        except CircuitOpenError:
            context.add_warning(self.name, "Typology engine circuit open")
            context.typological_connections = []

        # Necessity Calculation for strong typological connections
        # Run in parallel for efficiency
        necessity_tasks = []
        for result in context.typological_connections:
            if result.composite_strength > self.TYPOLOGY_STRENGTH_THRESHOLD:
                target_verse = (
                    result.antitype_reference if context.testament == "OT"
                    else result.type_reference
                )
                necessity_tasks.append(
                    self._calculate_necessity_with_tracking(
                        context.verse_id,
                        target_verse,
                        result
                    )
                )

        if necessity_tasks:
            await asyncio.gather(*necessity_tasks, return_exceptions=True)
            oracle_invocations += len(necessity_tasks)

        # Prophetic Analysis (for prophetic passages only)
        if await self._is_prophetic_passage(context.verse_id):
            try:
                prophecy_data = await self.orchestrator.prophetic_prover.analyze_prophecy(
                    context.verse_id
                )
                context.prophetic_analysis = prophecy_data
                oracle_invocations += 1

                if prophecy_data.posterior_probability > 0.8:
                    await self.orchestrator.event_publisher.publish(
                        PropheticFulfillmentIdentifiedEvent(
                            prophecy_verse=context.verse_id,
                            fulfillment_verses=prophecy_data.fulfillment_references,
                            posterior_probability=prophecy_data.posterior_probability
                        )
                    )
            except Exception as e:
                context.add_warning(self.name, f"Prophetic analysis failed: {e}")

        # Update metrics
        self.orchestrator._metrics[OrchestratorMetric.ORACLE_INVOCATIONS] += oracle_invocations

        context.phase_durations[self.name] = (time.time() - start_time) * 1000
        return context

    async def _calculate_necessity_with_tracking(
        self,
        source_verse: str,
        target_verse: str,
        typology_result
    ) -> None:
        """Calculate necessity and attach to typology result."""
        try:
            necessity = await self.orchestrator.necessity_calculator.calculate_necessity(
                source_verse,
                target_verse
            )
            typology_result.necessity_score = necessity.necessity_score

            await self.orchestrator.event_publisher.publish(
                NecessityCalculatedEvent(
                    source_verse=source_verse,
                    target_verse=target_verse,
                    necessity_score=necessity.necessity_score,
                    layer_scores=necessity.layer_breakdown
                )
            )
        except Exception as e:
            typology_result.necessity_score = 0.0  # Fallback

    async def _is_prophetic_passage(self, verse_id: str) -> bool:
        """Check if verse is in a prophetic section."""
        book = verse_id.split(".")[0]
        prophetic_books = {"ISA", "JER", "EZK", "DAN", "HOS", "JOL", "AMO",
                          "OBA", "JON", "MIC", "NAM", "HAB", "ZEP", "HAG", "ZEC", "MAL"}
        if book in prophetic_books:
            return True
        # Also check for prophetic passages in non-prophetic books
        prophetic_sections = await self.orchestrator.postgres.query(
            "SELECT 1 FROM prophetic_sections WHERE $1 BETWEEN start_verse AND end_verse",
            verse_id
        )
        return bool(prophetic_sections)
```

#### Cross-Reference Phase
```python
class CrossRefStrategy(Enum):
    """Strategy for cross-reference discovery."""
    SEMANTIC = "semantic"
    TYPOLOGICAL = "typological"
    PROPHETIC = "prophetic"
    LITURGICAL = "liturgical"
    BALANCED = "balanced"

    @property
    def domain_weights(self) -> Dict[str, float]:
        """Get embedding domain weights for this strategy."""
        return {
            CrossRefStrategy.SEMANTIC: {"semantic": 0.8, "patristic": 0.2},
            CrossRefStrategy.TYPOLOGICAL: {"typological": 0.5, "semantic": 0.3, "patristic": 0.2},
            CrossRefStrategy.PROPHETIC: {"prophetic": 0.5, "semantic": 0.3, "typological": 0.2},
            CrossRefStrategy.LITURGICAL: {"liturgical": 0.6, "semantic": 0.3, "patristic": 0.1},
            CrossRefStrategy.BALANCED: {"semantic": 0.25, "patristic": 0.25, "typological": 0.25, "prophetic": 0.25},
        }[self]


class CrossReferencePhase(Phase):
    """
    Phase 4: Cross-reference discovery with multi-vector search and GNN refinement.
    Integrates vector similarity, graph structure, and mutual transformation analysis.
    """
    name = "cross_reference"
    category = PhaseCategory.CROSS_REFERENCE
    priority = PhasePriority.NORMAL
    is_critical = False
    base_timeout_seconds = 60.0

    # Configuration
    INITIAL_CANDIDATES = 50
    GNN_REFINED_TOP_K = 20
    MIN_MUTUAL_INFLUENCE = 0.3

    @property
    def dependencies(self) -> List[PhaseDependency]:
        return [
            PhaseDependency(
                phase_name="linguistic",
                required_outputs=["linguistic_analysis"],
                is_hard=True
            ),
            PhaseDependency(
                phase_name="theological",
                required_outputs=["embeddings"],
                is_hard=False
            ),
            PhaseDependency(
                phase_name="intertextual",
                required_outputs=["typological_connections"],
                is_hard=False
            )
        ]

    @property
    def outputs(self) -> List[str]:
        return ["cross_references"]

    async def execute(self, context: ProcessingContext) -> ProcessingContext:
        start_time = time.time()

        # Select strategy based on context
        strategy = self._select_query_strategy(context)
        query_vectors = await self._build_query_vectors(context)

        # Multi-vector hybrid search with strategy-specific weights
        try:
            candidates = await self.orchestrator._execute_with_circuit_breaker(
                "vector_store",
                self.orchestrator.vector_store.hybrid_search(
                    query_vectors=query_vectors,
                    weights=strategy.domain_weights,
                    top_k=self.INITIAL_CANDIDATES
                )
            )
        except CircuitOpenError:
            context.add_warning(self.name, "Vector store circuit open, using graph-only")
            candidates = await self._fallback_graph_candidates(context.verse_id)

        # Capture pre-GNN embeddings for mutual transformation
        pre_gnn_embeddings = await self._capture_embeddings(context.verse_id, candidates)

        # GNN refinement with circuit breaker
        try:
            refined = await self.orchestrator._execute_with_circuit_breaker(
                "gnn_model",
                self.orchestrator.gnn_model.refine_candidates(
                    source_verse=context.verse_id,
                    candidates=candidates,
                    top_k=self.GNN_REFINED_TOP_K
                )
            )
        except CircuitOpenError:
            context.add_warning(self.name, "GNN circuit open, using unrefined candidates")
            refined = candidates[:self.GNN_REFINED_TOP_K]

        # Capture post-GNN embeddings
        post_gnn_embeddings = await self._capture_embeddings(context.verse_id, refined)

        # Calculate mutual transformation in parallel
        mt_tasks = [
            self._calculate_mutual_transformation(
                context.verse_id,
                candidate,
                pre_gnn_embeddings,
                post_gnn_embeddings
            )
            for candidate in refined
        ]
        await asyncio.gather(*mt_tasks, return_exceptions=True)

        # Filter by minimum mutual influence
        context.cross_references = [
            c for c in refined
            if c.mutual_influence >= self.MIN_MUTUAL_INFLUENCE
        ]

        # Emit discovery events for significant finds
        for candidate in context.cross_references:
            await self.orchestrator.event_publisher.publish(
                CrossReferenceDiscoveredEvent(
                    source_ref=context.verse_id,
                    target_ref=candidate.target_ref,
                    confidence=candidate.confidence,
                    mutual_influence=candidate.mutual_influence,
                    transformation_type=candidate.transformation_type,
                    strategy_used=strategy.value
                )
            )

        self.orchestrator._metrics[OrchestratorMetric.CROSS_REFS_DISCOVERED] += len(context.cross_references)
        context.phase_durations[self.name] = (time.time() - start_time) * 1000
        return context

    def _select_query_strategy(self, context: ProcessingContext) -> CrossRefStrategy:
        """Select optimal query strategy based on context."""
        # Prophetic passages use prophetic strategy
        if context.prophetic_analysis:
            return CrossRefStrategy.PROPHETIC

        # Strong typological connections use typological strategy
        if context.typological_connections:
            strong_typo = any(t.composite_strength > 0.7 for t in context.typological_connections)
            if strong_typo:
                return CrossRefStrategy.TYPOLOGICAL

        # Liturgical embedding present suggests liturgical strategy
        if "liturgical" in context.embeddings:
            return CrossRefStrategy.LITURGICAL

        # Default to balanced
        return CrossRefStrategy.BALANCED

    async def _build_query_vectors(self, context: ProcessingContext) -> Dict[str, np.ndarray]:
        """Build query vectors from available embeddings."""
        vectors = {}

        # Use pre-computed embeddings from context
        for domain, embedding in context.embeddings.items():
            vectors[domain] = embedding

        # Generate semantic embedding if missing
        if "semantic" not in vectors:
            verse = await self.orchestrator.postgres.get_verse(context.verse_id)
            vectors["semantic"] = await self.orchestrator.vector_store.generate_semantic_embedding(
                verse.text
            )

        return vectors

    async def _capture_embeddings(
        self,
        source_id: str,
        candidates: List
    ) -> Dict[str, np.ndarray]:
        """Capture embeddings for source and all candidates."""
        verse_ids = [source_id] + [c.target_ref for c in candidates]
        return await self.orchestrator.vector_store.batch_get_embeddings(verse_ids)

    async def _calculate_mutual_transformation(
        self,
        source_verse: str,
        candidate,
        pre_embeddings: Dict,
        post_embeddings: Dict
    ) -> None:
        """Calculate and attach mutual transformation metrics."""
        try:
            mt_result = await self.orchestrator.mutual_transformation.measure_transformation(
                source_verse=source_verse,
                target_verse=candidate.target_ref,
                source_before=pre_embeddings.get(source_verse),
                source_after=post_embeddings.get(source_verse),
                target_before=pre_embeddings.get(candidate.target_ref),
                target_after=post_embeddings.get(candidate.target_ref)
            )
            candidate.mutual_influence = mt_result.mutual_influence
            candidate.transformation_type = mt_result.transformation_type
        except Exception:
            candidate.mutual_influence = 0.0
            candidate.transformation_type = "unknown"

    async def _fallback_graph_candidates(self, verse_id: str) -> List:
        """Fallback to graph-based candidates when vector store unavailable."""
        return await self.orchestrator.neo4j.execute("""
            MATCH (v:Verse {id: $verse_id})-[r:CROSS_REFERENCES|SHARES_LEMMA|TYPOLOGICALLY_LINKED]-(t:Verse)
            RETURN t.id AS target_ref, type(r) AS connection_type, 0.5 AS confidence
            LIMIT 50
        """, verse_id=verse_id)
```

#### Validation Phase
```python
class ValidationVerdict(Enum):
    """Possible validation verdicts."""
    APPROVED = "approved"
    REJECTED = "rejected"
    DEFERRED = "deferred"  # Needs human review
    CHALLENGED = "challenged"  # Partially valid

    @property
    def allows_inclusion(self) -> bool:
        """Whether this verdict allows inclusion in golden record."""
        return self in {ValidationVerdict.APPROVED, ValidationVerdict.CHALLENGED}


@dataclass
class ChallengeResult:
    """Result from prosecutor challenge."""
    challenge_type: str
    severity: float  # 0-1, how serious the challenge is
    description: str
    supporting_evidence: List[str]

    @property
    def is_blocking(self) -> bool:
        """Challenge severe enough to block acceptance."""
        return self.severity > 0.7


@dataclass
class DefenseResult:
    """Result from witness defense."""
    defense_type: str
    strength: float  # 0-1, how strong the defense is
    description: str
    supporting_sources: List[str]

    @property
    def overcomes_challenge(self) -> bool:
        """Defense strong enough to overcome challenge."""
        return self.strength > 0.6


class ValidationPhase(Phase):
    """
    Phase 5: Validation with theological constraints and prosecutor/witness pattern.
    Implements adversarial validation where challenges must be defended.
    """
    name = "validation"
    category = PhaseCategory.VALIDATION
    priority = PhasePriority.NORMAL
    is_critical = False
    base_timeout_seconds = 45.0

    # Scoring weights for final judgment
    CONSTRAINT_WEIGHT = 0.4
    PROSECUTOR_PENALTY_WEIGHT = 0.3
    WITNESS_BONUS_WEIGHT = 0.3

    @property
    def dependencies(self) -> List[PhaseDependency]:
        return [
            PhaseDependency(
                phase_name="cross_reference",
                required_outputs=["cross_references"],
                is_hard=True
            ),
            PhaseDependency(
                phase_name="theological",
                required_outputs=["patristic_witness"],
                is_hard=False  # Patristic data enhances but isn't required
            )
        ]

    @property
    def outputs(self) -> List[str]:
        return ["validated_cross_references"]

    async def execute(self, context: ProcessingContext) -> ProcessingContext:
        start_time = time.time()

        validated_refs = []
        rejection_count = 0

        for ref in context.cross_references:
            # Apply theological constraints first
            constraint_result = await self.orchestrator.theological_validator.validate(
                source_verse=context.verse_id,
                target_verse=ref.target_ref,
                connection_type=ref.connection_type,
                confidence=ref.confidence
            )

            ref.constraint_validations = constraint_result.validations
            ref.theological_confidence = constraint_result.overall_score

            # Skip adversarial check if constraints already reject
            if constraint_result.overall_score < 0.3:
                ref.final_confidence = constraint_result.overall_score
                ref.verdict = ValidationVerdict.REJECTED
                rejection_count += 1
                continue

            # Prosecutor challenges
            challenges = await self._run_prosecutor(ref, context)
            ref.challenges = challenges

            # Witness defense (if challenged)
            defenses = []
            if challenges:
                defenses = await self._run_witness(ref, challenges, context)
                ref.defenses = defenses

            # Final judgment using weighted scoring
            final_score, verdict = self._calculate_final_judgment(
                ref, constraint_result, challenges, defenses
            )
            ref.final_confidence = final_score
            ref.verdict = verdict

            if verdict.allows_inclusion:
                validated_refs.append(ref)

                await self.orchestrator.event_publisher.publish(
                    CrossReferenceValidatedEvent(
                        cross_ref_id=f"{context.verse_id}:{ref.target_ref}",
                        validation_result=verdict.value,
                        final_confidence=final_score,
                        challenge_count=len(challenges),
                        defense_count=len(defenses)
                    )
                )
            else:
                rejection_count += 1
                await self.orchestrator.event_publisher.publish(
                    CrossReferenceValidatedEvent(
                        cross_ref_id=f"{context.verse_id}:{ref.target_ref}",
                        validation_result=verdict.value,
                        final_confidence=final_score,
                        rejection_reason=self._summarize_rejection(challenges, constraint_result)
                    )
                )

        context.validated_cross_references = validated_refs
        self.orchestrator._metrics[OrchestratorMetric.VALIDATION_REJECTIONS] += rejection_count
        context.phase_durations[self.name] = (time.time() - start_time) * 1000
        return context

    async def _run_prosecutor(
        self,
        ref,
        context: ProcessingContext
    ) -> List[ChallengeResult]:
        """Run prosecutor to challenge the cross-reference."""
        challenges = []

        # Challenge 1: Chronological implausibility
        if await self._check_chronological_issue(ref):
            challenges.append(ChallengeResult(
                challenge_type="chronological",
                severity=0.8,
                description="Source text post-dates target, making direct reference impossible",
                supporting_evidence=["date analysis"]
            ))

        # Challenge 2: Genre mismatch
        if await self._check_genre_mismatch(ref, context):
            challenges.append(ChallengeResult(
                challenge_type="genre_mismatch",
                severity=0.5,
                description="Source and target genres suggest coincidental similarity",
                supporting_evidence=["genre analysis"]
            ))

        # Challenge 3: Low mutual transformation
        if ref.mutual_influence < 0.4:
            challenges.append(ChallengeResult(
                challenge_type="weak_mutual_influence",
                severity=0.6,
                description="Texts do not significantly illuminate each other",
                supporting_evidence=[f"MI score: {ref.mutual_influence:.2f}"]
            ))

        # Challenge 4: No patristic support
        if not await self._has_patristic_support(ref, context):
            challenges.append(ChallengeResult(
                challenge_type="no_tradition",
                severity=0.4,
                description="No Church Father connects these passages",
                supporting_evidence=["patristic survey"]
            ))

        return challenges

    async def _run_witness(
        self,
        ref,
        challenges: List[ChallengeResult],
        context: ProcessingContext
    ) -> List[DefenseResult]:
        """Run witness to defend against challenges."""
        defenses = []

        for challenge in challenges:
            defense = await self._generate_defense(challenge, ref, context)
            if defense:
                defenses.append(defense)

        return defenses

    async def _generate_defense(
        self,
        challenge: ChallengeResult,
        ref,
        context: ProcessingContext
    ) -> Optional[DefenseResult]:
        """Generate defense for a specific challenge."""
        if challenge.challenge_type == "chronological":
            # Defense: typological connections transcend chronology
            if context.typological_connections:
                return DefenseResult(
                    defense_type="typological_transcendence",
                    strength=0.7,
                    description="Typological patterns operate outside strict chronology",
                    supporting_sources=["patristic hermeneutics"]
                )

        elif challenge.challenge_type == "no_tradition":
            # Defense: novel discovery with strong semantic/structural support
            if ref.confidence > 0.8:
                return DefenseResult(
                    defense_type="strong_structural_evidence",
                    strength=0.6,
                    description="Linguistic and structural evidence warrants new discovery",
                    supporting_sources=["semantic analysis", "syntactic parallels"]
                )

        elif challenge.challenge_type == "weak_mutual_influence":
            # Defense: unidirectional influence is still valid
            return DefenseResult(
                defense_type="unidirectional_validity",
                strength=0.5,
                description="One-way illumination is theologically meaningful",
                supporting_sources=["hermeneutical theory"]
            )

        return None

    def _calculate_final_judgment(
        self,
        ref,
        constraint_result,
        challenges: List[ChallengeResult],
        defenses: List[DefenseResult]
    ) -> Tuple[float, ValidationVerdict]:
        """Calculate final confidence score and verdict."""
        # Base from constraints
        base_score = constraint_result.overall_score * self.CONSTRAINT_WEIGHT

        # Prosecutor penalty
        total_penalty = sum(c.severity for c in challenges)
        max_penalty = len(challenges) if challenges else 1
        penalty_ratio = total_penalty / max_penalty
        penalty = penalty_ratio * self.PROSECUTOR_PENALTY_WEIGHT

        # Witness bonus (only for defended challenges)
        total_defense = sum(d.strength for d in defenses)
        max_defense = len(challenges) if challenges else 1
        defense_ratio = total_defense / max_defense if challenges else 0
        bonus = defense_ratio * self.WITNESS_BONUS_WEIGHT

        # Final score
        final_score = base_score - penalty + bonus + (ref.confidence * 0.2)
        final_score = max(0.0, min(1.0, final_score))

        # Determine verdict
        if final_score >= 0.7:
            verdict = ValidationVerdict.APPROVED
        elif final_score >= 0.5:
            verdict = ValidationVerdict.CHALLENGED
        elif final_score >= 0.3:
            verdict = ValidationVerdict.DEFERRED
        else:
            verdict = ValidationVerdict.REJECTED

        return final_score, verdict

    def _summarize_rejection(
        self,
        challenges: List[ChallengeResult],
        constraint_result
    ) -> str:
        """Summarize why a cross-reference was rejected."""
        if constraint_result.overall_score < 0.3:
            return f"Failed theological constraints: {constraint_result.failed_constraints}"
        if challenges:
            blocking = [c for c in challenges if c.is_blocking]
            if blocking:
                return f"Blocking challenges: {[c.challenge_type for c in blocking]}"
        return "Insufficient overall evidence"

    async def _check_chronological_issue(self, ref) -> bool:
        """Check for chronological implausibility."""
        # Implementation would check authorship dates
        return False

    async def _check_genre_mismatch(self, ref, context) -> bool:
        """Check for problematic genre mismatch."""
        # Implementation would compare genres
        return False

    async def _has_patristic_support(self, ref, context) -> bool:
        """Check if any Father connects these passages."""
        if not context.patristic_witness:
            return False
        # Check if any interpretation mentions the target
        for interp in context.patristic_witness:
            if ref.target_ref in str(interp):
                return True
        return False
```

---

## Part 4: Processing Context

### File: `pipeline/context.py`

**Context Completeness and Health Tracking**:

```python
class ContextCompleteness(Enum):
    """Level of context completeness."""
    FULL = "full"           # All phases completed successfully
    PARTIAL = "partial"     # Some phases skipped or degraded
    MINIMAL = "minimal"     # Only critical phases completed
    FAILED = "failed"       # Critical phase failed

    @property
    def is_usable(self) -> bool:
        """Whether this context can produce a golden record."""
        return self in {ContextCompleteness.FULL, ContextCompleteness.PARTIAL, ContextCompleteness.MINIMAL}


@dataclass
class ProcessingContext:
    """
    Carries state through all processing phases.
    Provides computed properties for context health monitoring.
    """
    # Identification
    verse_id: str
    correlation_id: str
    testament: str = None

    # Phase state tracking
    phase_states: Dict[str, PhaseState] = field(default_factory=dict)

    # Phase results
    linguistic_analysis: Dict = field(default_factory=dict)
    theological_analysis: Dict = field(default_factory=dict)
    lxx_analysis: Optional[LXXAnalysisResult] = None
    patristic_witness: List[PatristicInterpretation] = field(default_factory=list)
    typological_connections: List[FractalTypologyResult] = field(default_factory=list)
    prophetic_analysis: Optional[PropheticProofResult] = None
    cross_references: List[CrossReferenceCandidate] = field(default_factory=list)
    validated_cross_references: List[CrossReferenceCandidate] = field(default_factory=list)

    # Embeddings
    embeddings: Dict[str, np.ndarray] = field(default_factory=dict)

    # Metrics
    phase_durations: Dict[str, float] = field(default_factory=dict)
    total_duration_ms: float = 0

    # Errors and warnings
    errors: List[Dict] = field(default_factory=list)
    warnings: List[Dict] = field(default_factory=list)

    # Computed properties
    @property
    def completeness(self) -> ContextCompleteness:
        """Determine overall context completeness."""
        if not self.phase_states:
            return ContextCompleteness.FAILED

        failed = sum(1 for s in self.phase_states.values() if s == PhaseState.FAILED)
        completed = sum(1 for s in self.phase_states.values() if s == PhaseState.COMPLETED)
        skipped = sum(1 for s in self.phase_states.values() if s == PhaseState.SKIPPED)

        if failed > 0:
            # Check if any critical phase failed
            if self.phase_states.get("linguistic") == PhaseState.FAILED:
                return ContextCompleteness.FAILED
            return ContextCompleteness.MINIMAL

        if skipped > 0:
            return ContextCompleteness.PARTIAL

        if completed == len(self.phase_states):
            return ContextCompleteness.FULL

        return ContextCompleteness.PARTIAL

    @property
    def has_rich_theological_data(self) -> bool:
        """Check if context has rich theological analysis."""
        return (
            self.lxx_analysis is not None or
            len(self.patristic_witness) > 0 or
            len(self.typological_connections) > 0
        )

    @property
    def oracle_coverage(self) -> Dict[str, bool]:
        """Which oracles contributed to this context."""
        return {
            "omni_resolver": bool(self.linguistic_analysis.get("resolved_meanings")),
            "lxx_extractor": self.lxx_analysis is not None,
            "typology_engine": len(self.typological_connections) > 0,
            "necessity_calculator": any(
                hasattr(t, "necessity_score") and t.necessity_score > 0
                for t in self.typological_connections
            ),
            "prophetic_prover": self.prophetic_analysis is not None,
        }

    @property
    def oracle_coverage_ratio(self) -> float:
        """Ratio of oracles that contributed."""
        coverage = self.oracle_coverage
        return sum(coverage.values()) / len(coverage) if coverage else 0.0

    @property
    def embedding_domains(self) -> List[str]:
        """List of embedding domains present."""
        return list(self.embeddings.keys())

    @property
    def cross_ref_stats(self) -> Dict[str, Any]:
        """Statistics about cross-references."""
        if not self.validated_cross_references:
            return {"count": 0, "avg_confidence": 0.0, "types": []}

        types = set(r.connection_type for r in self.validated_cross_references if hasattr(r, "connection_type"))
        confidences = [r.final_confidence for r in self.validated_cross_references if hasattr(r, "final_confidence")]

        return {
            "count": len(self.validated_cross_references),
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
            "types": list(types),
            "by_verdict": self._count_by_verdict(),
        }

    def _count_by_verdict(self) -> Dict[str, int]:
        """Count cross-refs by verdict."""
        counts = defaultdict(int)
        for ref in self.validated_cross_references:
            if hasattr(ref, "verdict"):
                counts[ref.verdict.value] += 1
        return dict(counts)

    @property
    def total_processing_time_ms(self) -> float:
        """Total time across all phases."""
        return sum(self.phase_durations.values())

    @property
    def slowest_phase(self) -> Optional[Tuple[str, float]]:
        """Identify the slowest phase."""
        if not self.phase_durations:
            return None
        slowest = max(self.phase_durations.items(), key=lambda x: x[1])
        return slowest

    def add_error(self, phase: str, error: Exception):
        """Record an error from a phase."""
        self.errors.append({
            "phase": phase,
            "type": type(error).__name__,
            "message": str(error),
            "timestamp": datetime.utcnow().isoformat(),
            "traceback": traceback.format_exc()
        })

    def add_warning(self, phase: str, message: str):
        """Record a warning from a phase."""
        self.warnings.append({
            "phase": phase,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        })

    def to_summary(self) -> Dict[str, Any]:
        """Generate a summary of this context for logging."""
        return {
            "verse_id": self.verse_id,
            "correlation_id": self.correlation_id,
            "completeness": self.completeness.value,
            "oracle_coverage": self.oracle_coverage_ratio,
            "cross_refs": self.cross_ref_stats,
            "total_time_ms": self.total_processing_time_ms,
            "slowest_phase": self.slowest_phase,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
        }
```

---

## Part 5: Golden Record Builder

### File: `pipeline/golden_record.py`

```python
@dataclass
class GoldenRecord:
    """
    Complete output of BIBLOS v2 processing.
    The authoritative record for a verse.
    """
    # Identification
    verse_id: str
    book: str
    chapter: int
    verse: int
    testament: str

    # Text
    text_hebrew: Optional[str] = None
    text_greek: Optional[str] = None
    text_english: str = ""

    # Linguistic Analysis
    words: List[WordAnalysis] = field(default_factory=list)
    resolved_meanings: Dict[int, AbsoluteMeaningResult] = field(default_factory=dict)

    # Theological Analysis
    lxx_divergences: List[LXXDivergence] = field(default_factory=list)
    patristic_interpretations: List[PatristicInterpretation] = field(default_factory=list)
    patristic_consensus: float = 0.0
    theological_themes: List[str] = field(default_factory=list)

    # Intertextual Analysis
    typological_connections: List[TypologicalConnection] = field(default_factory=list)
    covenant_position: Optional[CovenantPosition] = None
    prophetic_data: Optional[PropheticData] = None

    # Cross-References
    cross_references: List[ValidatedCrossReference] = field(default_factory=list)
    centrality_score: float = 0.0

    # Oracle Insights
    oracle_insights: OracleInsights = field(default_factory=OracleInsights)

    # Metadata
    processing_timestamp: datetime = field(default_factory=datetime.utcnow)
    pipeline_version: str = "2.0.0"
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    processing_duration_ms: float = 0.0


@dataclass
class OracleInsights:
    """
    Aggregated insights from the Five Impossible Oracles.
    """
    # From OmniContextual Resolver
    absolute_meanings: Dict[str, str] = field(default_factory=dict)
    meaning_confidence: Dict[str, float] = field(default_factory=dict)

    # From Necessity Calculator
    essential_connections: List[str] = field(default_factory=list)
    necessity_scores: Dict[str, float] = field(default_factory=dict)

    # From LXX Extractor
    christological_insights: List[str] = field(default_factory=list)
    oldest_manuscript_support: Dict[str, str] = field(default_factory=dict)

    # From Fractal Typology
    typological_layers: Dict[str, List[str]] = field(default_factory=dict)
    pattern_signatures: List[str] = field(default_factory=list)

    # From Prophetic Prover
    prophetic_significance: float = 0.0
    fulfillment_evidence: List[str] = field(default_factory=list)
    bayesian_conclusion: Optional[str] = None


class GoldenRecordBuilder:
    """
    Builds GoldenRecord from ProcessingContext.
    """

    async def build(self, context: ProcessingContext) -> GoldenRecord:
        verse = await self.get_verse_data(context.verse_id)

        return GoldenRecord(
            verse_id=context.verse_id,
            book=verse.book,
            chapter=verse.chapter,
            verse=verse.verse,
            testament=verse.testament,
            text_hebrew=verse.text_hebrew,
            text_greek=verse.text_greek,
            text_english=verse.text_english,

            words=self._build_word_analysis(context.linguistic_analysis),
            resolved_meanings=context.linguistic_analysis.get("resolved_meanings", {}),

            lxx_divergences=self._extract_lxx_divergences(context.lxx_analysis),
            patristic_interpretations=context.patristic_witness,
            patristic_consensus=self._calculate_consensus(context.patristic_witness),
            theological_themes=self._extract_themes(context),

            typological_connections=self._build_typological(context.typological_connections),
            covenant_position=self._build_covenant_position(context),
            prophetic_data=self._build_prophetic_data(context.prophetic_analysis),

            cross_references=self._build_cross_refs(context.validated_cross_references),
            centrality_score=await self._get_centrality(context.verse_id),

            oracle_insights=self._build_oracle_insights(context),

            processing_timestamp=datetime.utcnow(),
            pipeline_version="2.0.0",
            confidence_scores=self._aggregate_confidence(context),
            processing_duration_ms=sum(context.phase_durations.values())
        )
```

---

## Part 6: Batch Processing

### File: `pipeline/batch_processor.py`

**Batch Processing Strategy and Backpressure**:

```python
class BatchStrategy(Enum):
    """Strategy for batch processing."""
    SEQUENTIAL = "sequential"      # One at a time (safest)
    CHUNKED = "chunked"            # Process in fixed chunks
    ADAPTIVE = "adaptive"          # Adjust concurrency based on performance
    PRIORITY = "priority"          # Process high-value verses first

    @property
    def base_concurrency(self) -> int:
        """Default concurrency level."""
        return {
            BatchStrategy.SEQUENTIAL: 1,
            BatchStrategy.CHUNKED: 10,
            BatchStrategy.ADAPTIVE: 5,
            BatchStrategy.PRIORITY: 8,
        }[self]


@dataclass
class BackpressureState:
    """Track backpressure metrics for adaptive processing."""
    current_concurrency: int
    success_rate: float = 1.0
    avg_latency_ms: float = 0.0
    error_streak: int = 0
    latency_samples: List[float] = field(default_factory=list)

    MAX_LATENCY_SAMPLES = 100
    ERROR_THRESHOLD = 3
    LATENCY_INCREASE_THRESHOLD = 1.5

    def record_success(self, latency_ms: float) -> None:
        """Record a successful processing."""
        self.latency_samples.append(latency_ms)
        if len(self.latency_samples) > self.MAX_LATENCY_SAMPLES:
            self.latency_samples.pop(0)
        self.avg_latency_ms = sum(self.latency_samples) / len(self.latency_samples)
        self.error_streak = 0

    def record_failure(self) -> None:
        """Record a failed processing."""
        self.error_streak += 1

    @property
    def should_reduce_concurrency(self) -> bool:
        """Check if we should reduce concurrency due to backpressure."""
        if self.error_streak >= self.ERROR_THRESHOLD:
            return True
        if len(self.latency_samples) >= 10:
            recent_avg = sum(self.latency_samples[-10:]) / 10
            if recent_avg > self.avg_latency_ms * self.LATENCY_INCREASE_THRESHOLD:
                return True
        return False

    @property
    def should_increase_concurrency(self) -> bool:
        """Check if we can safely increase concurrency."""
        return (
            self.error_streak == 0 and
            len(self.latency_samples) >= 20 and
            sum(self.latency_samples[-10:]) / 10 < self.avg_latency_ms * 0.8
        )


class BatchProcessor:
    """
    Process multiple verses efficiently with adaptive concurrency.
    """

    def __init__(self, orchestrator: UnifiedOrchestrator, config: BatchConfig):
        self.orchestrator = orchestrator
        self.config = config
        self.strategy = config.strategy or BatchStrategy.CHUNKED
        self._backpressure = BackpressureState(
            current_concurrency=self.strategy.base_concurrency
        )
        self._semaphore = asyncio.Semaphore(self._backpressure.current_concurrency)
        self._start_time: Optional[float] = None

    async def process_book(self, book_id: str) -> BatchResult:
        """Process all verses in a book."""
        self._start_time = time.time()
        verses = await self.orchestrator.corpus.get_book_verses(book_id)

        batch_id = str(uuid4())
        results = []
        errors = []

        # Emit batch started event
        await self.orchestrator.event_publisher.publish(
            BatchProcessingStartedEvent(
                batch_id=batch_id,
                book_id=book_id,
                verse_count=len(verses),
                strategy=self.strategy.value
            )
        )

        # Optionally prioritize verses
        if self.strategy == BatchStrategy.PRIORITY:
            verses = await self._prioritize_verses(verses)

        # Process in chunks with adaptive sizing
        chunk_size = self._calculate_chunk_size()
        for chunk in self._chunk_verses(verses, chunk_size):
            chunk_results = await self._process_chunk(chunk, batch_id)

            for verse_id, result in chunk_results:
                if isinstance(result, Exception):
                    errors.append({
                        "verse_id": verse_id,
                        "error": str(result),
                        "error_type": type(result).__name__
                    })
                else:
                    results.append(result)

            # Adaptive concurrency adjustment
            if self.strategy == BatchStrategy.ADAPTIVE:
                await self._adjust_concurrency()

            # Progress update with ETA
            processed = len(results) + len(errors)
            eta_seconds = self._calculate_eta(processed, len(verses))

            await self.orchestrator.event_publisher.publish(
                BatchProgressEvent(
                    batch_id=batch_id,
                    processed=processed,
                    total=len(verses),
                    success_count=len(results),
                    error_count=len(errors),
                    eta_seconds=eta_seconds,
                    current_concurrency=self._backpressure.current_concurrency
                )
            )

        # Calculate final statistics
        duration_ms = (time.time() - self._start_time) * 1000

        # Emit batch completed event
        await self.orchestrator.event_publisher.publish(
            BatchProcessingCompletedEvent(
                batch_id=batch_id,
                success_count=len(results),
                error_count=len(errors),
                duration_ms=duration_ms,
                avg_verse_ms=duration_ms / len(verses) if verses else 0
            )
        )

        return BatchResult(
            batch_id=batch_id,
            book_id=book_id,
            results=results,
            errors=errors,
            duration_ms=duration_ms,
            throughput_per_second=len(verses) / (duration_ms / 1000) if duration_ms > 0 else 0
        )

    async def _process_chunk(
        self,
        chunk: List[str],
        batch_id: str
    ) -> List[Tuple[str, Union[GoldenRecord, Exception]]]:
        """Process a chunk of verses with semaphore control."""
        tasks = [
            self._process_with_semaphore(verse_id, batch_id)
            for verse_id in chunk
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return list(zip(chunk, results))

    async def _process_with_semaphore(
        self,
        verse_id: str,
        batch_id: str
    ) -> GoldenRecord:
        """Process single verse with semaphore and backpressure tracking."""
        async with self._semaphore:
            start = time.time()
            try:
                result = await self.orchestrator.process_verse(
                    verse_id,
                    correlation_id=f"{batch_id}:{verse_id}"
                )
                latency = (time.time() - start) * 1000
                self._backpressure.record_success(latency)
                return result
            except Exception as e:
                self._backpressure.record_failure()
                raise

    async def _adjust_concurrency(self) -> None:
        """Adjust concurrency based on backpressure state."""
        if self._backpressure.should_reduce_concurrency:
            new_concurrency = max(1, self._backpressure.current_concurrency - 2)
            self._backpressure.current_concurrency = new_concurrency
            self._semaphore = asyncio.Semaphore(new_concurrency)
        elif self._backpressure.should_increase_concurrency:
            new_concurrency = min(
                self.config.max_concurrency,
                self._backpressure.current_concurrency + 1
            )
            self._backpressure.current_concurrency = new_concurrency
            self._semaphore = asyncio.Semaphore(new_concurrency)

    def _calculate_chunk_size(self) -> int:
        """Calculate optimal chunk size based on strategy."""
        if self.strategy == BatchStrategy.SEQUENTIAL:
            return 1
        return min(self.config.chunk_size, self._backpressure.current_concurrency * 2)

    def _calculate_eta(self, processed: int, total: int) -> float:
        """Calculate estimated time to completion."""
        if processed == 0 or self._start_time is None:
            return 0.0
        elapsed = time.time() - self._start_time
        rate = processed / elapsed
        remaining = total - processed
        return remaining / rate if rate > 0 else 0.0

    async def _prioritize_verses(self, verses: List[str]) -> List[str]:
        """Prioritize verses for processing (high-value first)."""
        # Get centrality scores from graph
        centralities = await self.orchestrator.neo4j.execute("""
            MATCH (v:Verse)
            WHERE v.id IN $verses
            RETURN v.id AS verse_id, v.pagerank AS centrality
            ORDER BY v.pagerank DESC
        """, verses=verses)

        scored = {r["verse_id"]: r["centrality"] or 0 for r in centralities}
        return sorted(verses, key=lambda v: scored.get(v, 0), reverse=True)

    def _chunk_verses(self, verses: List[str], chunk_size: int) -> Iterator[List[str]]:
        """Yield chunks of verses."""
        for i in range(0, len(verses), chunk_size):
            yield verses[i:i + chunk_size]

    async def process_canon(self) -> CanonResult:
        """Process entire biblical canon with graph updates between books."""
        books = self.orchestrator.config.book_order

        book_results = []
        for book in books:
            result = await self.process_book(book)
            book_results.append(result)

            # Run graph algorithms after each book for incremental updates
            await self.orchestrator.neo4j.calculate_verse_centrality()
            await self.orchestrator.neo4j.detect_communities()

        return CanonResult(
            book_results=book_results,
            total_verses=sum(r.success_count + r.error_count for r in book_results),
            total_errors=sum(r.error_count for r in book_results),
            total_duration_ms=sum(r.duration_ms for r in book_results)
        )
```

---

## Part 7: Query Interface

### File: `pipeline/query_interface.py`

```python
class QueryInterface:
    """
    High-level query interface for processed data.
    """

    def __init__(self, orchestrator: UnifiedOrchestrator):
        self.orchestrator = orchestrator

    async def get_verse_analysis(self, verse_id: str) -> GoldenRecord:
        """Get complete analysis for a verse."""
        # Check cache first
        cached = await self.orchestrator.redis.get(f"golden:{verse_id}")
        if cached:
            return GoldenRecord.from_json(cached)

        # Load from database
        return await self._load_golden_record(verse_id)

    async def find_cross_references(
        self,
        verse_id: str,
        min_confidence: float = 0.5,
        connection_types: List[str] = None
    ) -> List[ValidatedCrossReference]:
        """Find cross-references for a verse."""
        return await self.orchestrator.neo4j.execute("""
            MATCH (v:Verse {id: $verse_id})-[r:CROSS_REFERENCES]->(t:Verse)
            WHERE r.confidence >= $min_conf
            AND ($types IS NULL OR r.connection_type IN $types)
            RETURN t.id AS target,
                   r.connection_type AS type,
                   r.confidence AS confidence,
                   r.mutual_influence AS mutual_influence
            ORDER BY r.confidence DESC
        """, verse_id=verse_id, min_conf=min_confidence, types=connection_types)

    async def find_typological_chain(
        self,
        verse_id: str,
        pattern: str = None
    ) -> List[TypologicalConnection]:
        """Find typological chain for a verse."""
        return await self.orchestrator.typology_engine.discover_fractal_patterns(
            verse_id
        )

    async def get_patristic_consensus(
        self,
        verse_id: str,
        traditions: List[str] = None
    ) -> PatristicConsensus:
        """Get patristic consensus on a verse."""
        interpretations = await self.orchestrator.patristic_db.get_interpretations(
            verse_id,
            traditions=traditions
        )

        return PatristicConsensus(
            verse_id=verse_id,
            interpretations=interpretations,
            consensus_score=self._calculate_consensus(interpretations),
            dominant_sense=self._determine_dominant_sense(interpretations)
        )

    async def prove_prophecy(
        self,
        prophecy_verses: List[str],
        prior: float = 0.5
    ) -> PropheticProofResult:
        """Run prophetic proof calculation."""
        return await self.orchestrator.prophetic_prover.prove_prophetic_necessity(
            prophecy_verses,
            prior_supernatural=prior
        )

    async def semantic_search(
        self,
        query: str,
        strategy: str = "theological",
        top_k: int = 10
    ) -> List[SearchResult]:
        """Semantic search across the corpus."""
        strategy_obj = self._get_strategy(strategy)
        query_vector = await self._embed_query(query)

        return await self.orchestrator.vector_store.hybrid_search(
            query_vectors={"semantic": query_vector},
            weights=strategy_obj.get_weights(),
            top_k=top_k
        )
```

---

## Part 8: Testing Specification

### Unit Tests: `tests/pipeline/test_unified_orchestrator.py`

**Test 1: `test_single_verse_processing`**
- Process GEN.1.1
- Verify all phases executed
- Verify Golden Record complete

**Test 2: `test_phase_events_emitted`**
- Process verse
- Verify all phase events emitted
- Verify correct ordering

**Test 3: `test_oracle_integration`**
- Process polysemous verse
- Verify OmniContextual Resolver called
- Verify meaning resolved correctly

**Test 4: `test_lxx_integration`**
- Process ISA.7.14
- Verify LXX Extractor called
- Verify παρθένος divergence detected

**Test 5: `test_typology_integration`**
- Process GEN.22.1
- Verify Typology Engine called
- Verify Isaac/Christ connection found

**Test 6: `test_batch_processing`**
- Process Genesis 1
- Verify all verses processed
- Verify no errors

**Test 7: `test_cross_reference_validation`**
- Process verse with cross-refs
- Verify theological constraints applied
- Verify prosecutor/witness pattern

**Test 8: `test_golden_record_completeness`**
- Process complex verse
- Verify all GoldenRecord fields populated
- Verify oracle insights present

---

## Part 9: Configuration

### Add to `config.py`

**New Configuration Dataclass**: `UnifiedOrchestratorConfig`

Fields:
- `max_concurrency: int = 10`
- `chunk_size: int = 50`
- `min_final_confidence: float = 0.5`
- `enable_oracle_engines: bool = True`
- `oracle_engines: List[str] = ["omni", "necessity", "lxx", "typology", "prophetic"]`
- `enable_patristic_integration: bool = True`
- `enable_liturgical_integration: bool = True`
- `cache_golden_records: bool = True`
- `cache_ttl_hours: int = 24`
- `emit_all_events: bool = True`
- `validation_mode: str = "strict"` # "strict", "lenient", "disabled"

---

## Part 10: Success Criteria

### Functional Requirements
- [ ] All phases execute correctly
- [ ] All oracle engines integrated
- [ ] Events emitted for all operations
- [ ] Golden Record complete and accurate
- [ ] Batch processing functional
- [ ] Query interface working

### Performance Requirements
- [ ] Single verse processing: < 5 seconds
- [ ] Batch (100 verses): < 2 minutes
- [ ] Book processing (Genesis, 1533 verses): < 30 minutes
- [ ] Full canon: < 24 hours

### Integration Requirements
- [ ] Event sourcing working (Session 08)
- [ ] Neo4j projections updating (Session 09)
- [ ] Vector store updating (Session 10)
- [ ] All oracle engines invoked correctly

---

## Part 11: Detailed Implementation Order

1. **Create `ProcessingContext`** dataclass
2. **Create phase base class**
3. **Implement LinguisticPhase** with OmniContextual integration
4. **Implement TheologicalPhase** with LXX integration
5. **Implement IntertextualPhase** with Typology integration
6. **Implement CrossReferencePhase** with multi-vector search
7. **Implement ValidationPhase** with theological constraints
8. **Create `GoldenRecord`** and builder
9. **Create `UnifiedOrchestrator`** main class
10. **Integrate event publishing** throughout
11. **Create `BatchProcessor`**
12. **Create `QueryInterface`**
13. **Add configuration to `config.py`**
14. **Write unit tests**
15. **Run integration tests**
16. **Benchmark performance**

---

## Part 12: Dependencies on Other Sessions

### Depends On
- ALL Sessions 01-10

### Depended On By
- SESSION 12: Testing & Final Integration

### External Dependencies
- All previous session implementations
- LangGraph for complex orchestration
- Redis for caching and streams

---

## Session Completion Checklist

```markdown
- [ ] `pipeline/unified_orchestrator.py` implemented
- [ ] All five phase classes implemented
- [ ] `ProcessingContext` complete
- [ ] `GoldenRecord` and builder complete
- [ ] All oracle engines integrated
- [ ] Event publishing throughout
- [ ] `BatchProcessor` working
- [ ] `QueryInterface` working
- [ ] Configuration added to config.py
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] Single verse benchmarked
- [ ] Batch processing benchmarked
- [ ] Documentation complete
```

**Next Session**: SESSION 12: Testing, Validation & Final Integration
