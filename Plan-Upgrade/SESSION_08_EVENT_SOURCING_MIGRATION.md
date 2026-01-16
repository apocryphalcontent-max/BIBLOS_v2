# SESSION 08: EVENT SOURCING MIGRATION

## Session Overview

**Objective**: Migrate BIBLOS v2 from traditional CRUD patterns to an Event Sourcing architecture with CQRS (Command Query Responsibility Segregation). This enables complete audit trails, temporal queries, and reproducible state reconstruction.

**Estimated Duration**: 1 Claude session (90-120 minutes of focused implementation)

**Prerequisites**:
- Understanding of existing pipeline architecture (`pipeline/stream_orchestrator.py`)
- Familiarity with Redis Streams (already used for event streaming)
- Knowledge of PostgreSQL for event storage
- Understanding of CQRS patterns

---

## Part 1: Understanding Event Sourcing

### Core Concept
Instead of storing current state (CRUD), store all events that led to current state:
- **Events are immutable**: Once written, never modified
- **State is derived**: Current state = replay of all events
- **Complete history**: Every change is tracked with full context
- **Temporal queries**: "What was the state at time T?"

### Why Event Sourcing for BIBLOS

1. **Audit Trail**: Track every cross-reference discovery, agent decision, theological validation
2. **Reproducibility**: Replay events to reproduce any analysis
3. **Debugging**: Trace exactly what happened and why
4. **Versioning**: Compare analysis results across system versions
5. **Learning**: Analyze historical patterns for ML improvement

### Current vs. Target Architecture

**Current (CRUD)**:
```
Agent extracts data → Update PostgreSQL → State is current only
```

**Target (Event Sourcing)**:
```
Agent extracts data → Emit Event → Store in Event Log → Project to Read Models
                                                      → Update PostgreSQL (read model)
                                                      → Update Neo4j (read model)
                                                      → Update Redis cache
```

---

## Part 2: Event Schema Specification

### File: `events/schemas.py`

**Location**: Create new directory `events/`

**Base Event Classes**:

#### 1. `EventBase` (Dataclass)
```python
@dataclass
class EventBase:
    event_id: str                    # UUID for this event
    event_type: str                  # e.g., "VerseProcessingStarted"
    aggregate_id: str                # e.g., verse_id or cross_ref_id
    aggregate_type: str              # e.g., "Verse", "CrossReference"
    timestamp: datetime              # When event occurred
    version: int                     # Aggregate version (for optimistic concurrency)
    correlation_id: str              # Links related events
    causation_id: Optional[str]      # Event that caused this one
    actor: str                       # Who/what triggered (agent name, user, system)
    metadata: Dict[str, Any]         # Additional context
```

#### 2. Domain Events

**Verse Processing Events**:
```python
@dataclass
class VerseProcessingStartedEvent(EventBase):
    verse_id: str
    book: str
    chapter: int
    verse: int
    text_content: str
    phase: str  # "linguistic", "theological", "intertextual", "validation"

@dataclass
class VerseProcessingCompletedEvent(EventBase):
    verse_id: str
    phase: str
    duration_ms: int
    agent_count: int
    result_summary: Dict[str, Any]

@dataclass
class VerseProcessingFailedEvent(EventBase):
    verse_id: str
    phase: str
    error_type: str
    error_message: str
    stack_trace: Optional[str]
    retry_count: int
```

**Cross-Reference Events**:
```python
@dataclass
class CrossReferenceDiscoveredEvent(EventBase):
    source_ref: str
    target_ref: str
    connection_type: str
    initial_confidence: float
    discovery_method: str  # "embedding", "gnn", "keyword", "agent"
    features: Dict[str, float]

@dataclass
class CrossReferenceValidatedEvent(EventBase):
    cross_ref_id: str
    validator_agent: str
    validation_result: str  # "approved", "rejected", "modified"
    confidence_adjustment: float
    reasoning: str

@dataclass
class CrossReferenceRefinedEvent(EventBase):
    cross_ref_id: str
    refinement_type: str  # "mutual_transformation", "theological", "typological"
    old_values: Dict[str, Any]
    new_values: Dict[str, Any]
    reason: str
```

**Theological Events**:
```python
@dataclass
class TheologicalConstraintAppliedEvent(EventBase):
    target_id: str  # verse or cross-ref
    constraint_type: str  # "patristic_escalation", "christological_coherence", etc.
    constraint_result: str  # "passed", "failed", "warning"
    details: Dict[str, Any]

@dataclass
class PatristicWitnessAddedEvent(EventBase):
    target_id: str
    father_name: str
    work: str
    citation: str
    interpretation_summary: str

@dataclass
class TypologicalConnectionIdentifiedEvent(EventBase):
    type_ref: str
    antitype_ref: str
    typology_layer: str
    fractal_depth: int
    composite_strength: float
```

**Oracle Engine Events**:
```python
@dataclass
class OmniContextualResolutionEvent(EventBase):
    word: str
    verse_id: str
    language: str
    resolved_meaning: str
    eliminated_meanings: List[str]
    total_occurrences: int
    confidence: float

@dataclass
class NecessityCalculatedEvent(EventBase):
    source_verse: str
    target_verse: str
    necessity_score: float
    necessity_type: str
    semantic_gaps_filled: int

@dataclass
class LXXDivergenceDetectedEvent(EventBase):
    verse_id: str
    divergence_type: str
    christological_category: str
    oldest_witness: str
    manuscript_support: float

@dataclass
class PropheticProofComputedEvent(EventBase):
    prophecy_set: List[str]
    compound_probability: float
    bayesian_posterior: float
    prior_used: float
    interpretation: str
```

---

## Part 3: Event Store Implementation

### File: `events/event_store.py`

**PostgreSQL Event Store Schema**:
```sql
CREATE TABLE events (
    event_id UUID PRIMARY KEY,
    event_type VARCHAR(255) NOT NULL,
    aggregate_id VARCHAR(255) NOT NULL,
    aggregate_type VARCHAR(100) NOT NULL,
    version INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    correlation_id UUID,
    causation_id UUID,
    actor VARCHAR(255),
    data JSONB NOT NULL,
    metadata JSONB,

    -- Indexes for common queries
    INDEX idx_aggregate (aggregate_type, aggregate_id, version),
    INDEX idx_timestamp (timestamp),
    INDEX idx_correlation (correlation_id),
    INDEX idx_event_type (event_type),

    -- Ensure version ordering per aggregate
    UNIQUE (aggregate_type, aggregate_id, version)
);

CREATE TABLE event_snapshots (
    aggregate_id VARCHAR(255) NOT NULL,
    aggregate_type VARCHAR(100) NOT NULL,
    version INTEGER NOT NULL,
    snapshot_data JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (aggregate_type, aggregate_id, version)
);
```

**EventStore Class**:
```python
class EventStore:
    """
    Append-only event store with optimistic concurrency.
    """

    async def append(
        self,
        events: List[EventBase],
        expected_version: Optional[int] = None
    ) -> None:
        """
        Append events to store with optimistic concurrency check.
        """

    async def get_events(
        self,
        aggregate_type: str,
        aggregate_id: str,
        from_version: int = 0,
        to_version: Optional[int] = None
    ) -> List[EventBase]:
        """
        Get events for an aggregate.
        """

    async def get_events_by_correlation(
        self,
        correlation_id: str
    ) -> List[EventBase]:
        """
        Get all events sharing a correlation ID.
        """

    async def get_events_by_type(
        self,
        event_type: str,
        from_timestamp: Optional[datetime] = None,
        to_timestamp: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[EventBase]:
        """
        Get events of a specific type.
        """

    async def get_all_events(
        self,
        from_position: int = 0,
        batch_size: int = 1000
    ) -> AsyncIterator[EventBase]:
        """
        Stream all events for replay.
        """

    async def save_snapshot(
        self,
        aggregate_type: str,
        aggregate_id: str,
        version: int,
        snapshot_data: Dict
    ) -> None:
        """
        Save aggregate snapshot for faster hydration.
        """

    async def get_latest_snapshot(
        self,
        aggregate_type: str,
        aggregate_id: str
    ) -> Optional[Tuple[int, Dict]]:
        """
        Get latest snapshot for aggregate.
        """
```

---

## Part 4: Aggregate Root Implementation

### File: `events/aggregates.py`

**Base Aggregate**:
```python
class AggregateRoot:
    """
    Base class for event-sourced aggregates.
    """

    def __init__(self, aggregate_id: str):
        self.aggregate_id = aggregate_id
        self.version = 0
        self._pending_events: List[EventBase] = []

    def apply(self, event: EventBase) -> None:
        """Apply event to update state."""
        handler = getattr(self, f"_apply_{event.event_type}", None)
        if handler:
            handler(event)
        self.version = event.version

    def record(self, event: EventBase) -> None:
        """Record new event (pending commit)."""
        event.version = self.version + 1
        self._pending_events.append(event)
        self.apply(event)

    def get_pending_events(self) -> List[EventBase]:
        """Get events pending commit."""
        return self._pending_events

    def clear_pending_events(self) -> None:
        """Clear pending events after commit."""
        self._pending_events = []
```

**VerseAggregate**:
```python
class VerseAggregate(AggregateRoot):
    """
    Aggregate for verse processing state.
    """

    def __init__(self, verse_id: str):
        super().__init__(verse_id)
        self.verse_id = verse_id
        self.text_content: Optional[str] = None
        self.processing_status: Dict[str, str] = {}
        self.agent_results: Dict[str, Dict] = {}
        self.cross_references: List[str] = []
        self.theological_validations: List[Dict] = []

    def start_processing(self, text: str, phase: str, actor: str) -> None:
        self.record(VerseProcessingStartedEvent(
            event_id=str(uuid4()),
            event_type="VerseProcessingStarted",
            aggregate_id=self.verse_id,
            aggregate_type="Verse",
            timestamp=datetime.utcnow(),
            version=0,  # Will be set by record()
            correlation_id=str(uuid4()),
            causation_id=None,
            actor=actor,
            metadata={},
            verse_id=self.verse_id,
            book=parse_book(self.verse_id),
            chapter=parse_chapter(self.verse_id),
            verse=parse_verse(self.verse_id),
            text_content=text,
            phase=phase
        ))

    def _apply_VerseProcessingStarted(self, event: VerseProcessingStartedEvent) -> None:
        self.text_content = event.text_content
        self.processing_status[event.phase] = "in_progress"

    def complete_processing(self, phase: str, duration_ms: int, result: Dict, actor: str) -> None:
        # Record completion event...
        pass

    def _apply_VerseProcessingCompleted(self, event: VerseProcessingCompletedEvent) -> None:
        self.processing_status[event.phase] = "completed"
        self.agent_results[event.phase] = event.result_summary
```

**CrossReferenceAggregate**:
```python
class CrossReferenceAggregate(AggregateRoot):
    """
    Aggregate for cross-reference lifecycle.
    """

    def __init__(self, cross_ref_id: str):
        super().__init__(cross_ref_id)
        self.cross_ref_id = cross_ref_id
        self.source_ref: Optional[str] = None
        self.target_ref: Optional[str] = None
        self.connection_type: Optional[str] = None
        self.confidence: float = 0.0
        self.validation_history: List[Dict] = []
        self.refinements: List[Dict] = []
        self.status: str = "discovered"

    def discover(
        self,
        source_ref: str,
        target_ref: str,
        connection_type: str,
        confidence: float,
        method: str,
        features: Dict,
        actor: str
    ) -> None:
        self.record(CrossReferenceDiscoveredEvent(
            # ... event fields
        ))

    def validate(self, validator: str, result: str, adjustment: float, reasoning: str) -> None:
        self.record(CrossReferenceValidatedEvent(
            # ... event fields
        ))

    def refine(self, refinement_type: str, old_values: Dict, new_values: Dict, reason: str) -> None:
        self.record(CrossReferenceRefinedEvent(
            # ... event fields
        ))
```

---

## Part 5: Read Model Projections

### File: `events/projections.py`

**Projection Base**:
```python
class ProjectionBase:
    """
    Base class for read model projections.
    """

    def __init__(self, event_store: EventStore):
        self.event_store = event_store
        self.last_processed_position = 0

    async def project(self, event: EventBase) -> None:
        """Project a single event to read model."""
        handler = getattr(self, f"_handle_{event.event_type}", None)
        if handler:
            await handler(event)

    async def rebuild(self) -> None:
        """Rebuild projection from all events."""
        async for event in self.event_store.get_all_events():
            await self.project(event)

    async def catch_up(self) -> None:
        """Process events since last position."""
        async for event in self.event_store.get_all_events(
            from_position=self.last_processed_position
        ):
            await self.project(event)
            self.last_processed_position += 1
```

**PostgreSQL Read Model Projection**:
```python
class PostgresReadModelProjection(ProjectionBase):
    """
    Projects events to PostgreSQL read model tables.
    """

    def __init__(self, event_store: EventStore, postgres_client):
        super().__init__(event_store)
        self.postgres = postgres_client

    async def _handle_CrossReferenceDiscoveredEvent(
        self,
        event: CrossReferenceDiscoveredEvent
    ) -> None:
        await self.postgres.execute("""
            INSERT INTO cross_references (
                id, source_ref, target_ref, connection_type,
                confidence, discovery_method, features, status, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, 'discovered', $8)
            ON CONFLICT (id) DO UPDATE SET
                confidence = EXCLUDED.confidence,
                features = EXCLUDED.features
        """, event.aggregate_id, event.source_ref, event.target_ref,
             event.connection_type, event.initial_confidence,
             event.discovery_method, event.features, event.timestamp)

    async def _handle_CrossReferenceValidatedEvent(
        self,
        event: CrossReferenceValidatedEvent
    ) -> None:
        # Update confidence and status based on validation
        if event.validation_result == "approved":
            new_status = "validated"
        elif event.validation_result == "rejected":
            new_status = "rejected"
        else:
            new_status = "pending_review"

        await self.postgres.execute("""
            UPDATE cross_references
            SET confidence = confidence + $1,
                status = $2,
                validated_at = $3,
                validator = $4
            WHERE id = $5
        """, event.confidence_adjustment, new_status,
             event.timestamp, event.validator_agent, event.cross_ref_id)
```

**Neo4j Graph Projection**:
```python
class Neo4jGraphProjection(ProjectionBase):
    """
    Projects events to Neo4j graph.
    """

    def __init__(self, event_store: EventStore, neo4j_client):
        super().__init__(event_store)
        self.neo4j = neo4j_client

    async def _handle_CrossReferenceDiscoveredEvent(
        self,
        event: CrossReferenceDiscoveredEvent
    ) -> None:
        await self.neo4j.execute("""
            MERGE (s:Verse {id: $source_ref})
            MERGE (t:Verse {id: $target_ref})
            CREATE (s)-[:CROSS_REFERENCE {
                id: $cross_ref_id,
                type: $connection_type,
                confidence: $confidence,
                discovery_method: $method,
                discovered_at: $timestamp
            }]->(t)
        """, source_ref=event.source_ref, target_ref=event.target_ref,
             cross_ref_id=event.aggregate_id, connection_type=event.connection_type,
             confidence=event.initial_confidence, method=event.discovery_method,
             timestamp=event.timestamp.isoformat())

    async def _handle_TypologicalConnectionIdentifiedEvent(
        self,
        event: TypologicalConnectionIdentifiedEvent
    ) -> None:
        await self.neo4j.execute("""
            MERGE (t:Verse {id: $type_ref})
            MERGE (a:Verse {id: $antitype_ref})
            CREATE (t)-[:TYPIFIES {
                layer: $layer,
                fractal_depth: $depth,
                strength: $strength,
                identified_at: $timestamp
            }]->(a)
        """, type_ref=event.type_ref, antitype_ref=event.antitype_ref,
             layer=event.typology_layer, depth=event.fractal_depth,
             strength=event.composite_strength, timestamp=event.timestamp.isoformat())
```

---

## Part 6: Command Handlers

### File: `events/commands.py`

**Command Pattern Implementation**:
```python
@dataclass
class Command:
    """Base command class."""
    command_id: str = field(default_factory=lambda: str(uuid4()))
    correlation_id: Optional[str] = None
    actor: str = "system"

@dataclass
class ProcessVerseCommand(Command):
    verse_id: str
    text_content: str
    phases: List[str] = field(default_factory=lambda: ["linguistic", "theological", "intertextual", "validation"])

@dataclass
class DiscoverCrossReferencesCommand(Command):
    verse_id: str
    discovery_methods: List[str] = field(default_factory=lambda: ["embedding", "gnn"])
    min_confidence: float = 0.5

@dataclass
class ValidateCrossReferenceCommand(Command):
    cross_ref_id: str
    validator: str

@dataclass
class ApplyTheologicalConstraintsCommand(Command):
    target_id: str
    constraint_types: List[str]
```

**Command Handler**:
```python
class CommandHandler:
    """
    Handles commands by loading aggregates, executing operations, and saving events.
    """

    def __init__(self, event_store: EventStore, aggregate_repository):
        self.event_store = event_store
        self.repository = aggregate_repository

    async def handle(self, command: Command) -> None:
        handler = getattr(self, f"_handle_{type(command).__name__}", None)
        if not handler:
            raise ValueError(f"No handler for {type(command).__name__}")
        await handler(command)

    async def _handle_ProcessVerseCommand(self, command: ProcessVerseCommand) -> None:
        # Load or create aggregate
        aggregate = await self.repository.get_or_create(
            VerseAggregate, command.verse_id
        )

        # Execute command
        for phase in command.phases:
            aggregate.start_processing(
                text=command.text_content,
                phase=phase,
                actor=command.actor
            )

        # Save events
        await self.event_store.append(
            aggregate.get_pending_events(),
            expected_version=aggregate.version - len(aggregate.get_pending_events())
        )
        aggregate.clear_pending_events()

    async def _handle_DiscoverCrossReferencesCommand(
        self,
        command: DiscoverCrossReferencesCommand
    ) -> None:
        # Run discovery pipeline
        candidates = await self.discovery_pipeline.discover(
            command.verse_id,
            methods=command.discovery_methods
        )

        for candidate in candidates:
            if candidate.confidence >= command.min_confidence:
                aggregate = CrossReferenceAggregate(
                    cross_ref_id=f"{command.verse_id}:{candidate.target_ref}"
                )
                aggregate.discover(
                    source_ref=command.verse_id,
                    target_ref=candidate.target_ref,
                    connection_type=candidate.connection_type,
                    confidence=candidate.confidence,
                    method=candidate.discovery_method,
                    features=candidate.features,
                    actor=command.actor
                )
                await self.event_store.append(aggregate.get_pending_events())
```

---

## Part 7: Integration with Existing Pipeline

### Modifications to `pipeline/stream_orchestrator.py`

**Current Architecture**:
```python
class StreamOrchestrator:
    async def process_verse(self, verse_id: str):
        # Direct state mutations
        await self.update_verse_state(verse_id, "processing")
        # ...
```

**New Event-Sourced Architecture**:
```python
class EventSourcedOrchestrator:
    def __init__(
        self,
        event_store: EventStore,
        command_handler: CommandHandler,
        projections: List[ProjectionBase]
    ):
        self.event_store = event_store
        self.command_handler = command_handler
        self.projections = projections

    async def process_verse(self, verse_id: str, text: str) -> str:
        """
        Process verse using event sourcing.
        Returns correlation_id for tracking.
        """
        correlation_id = str(uuid4())

        # Issue command (will emit events)
        command = ProcessVerseCommand(
            verse_id=verse_id,
            text_content=text,
            correlation_id=correlation_id,
            actor="pipeline"
        )
        await self.command_handler.handle(command)

        # Events are automatically projected to read models
        return correlation_id

    async def get_verse_history(self, verse_id: str) -> List[EventBase]:
        """Get complete event history for a verse."""
        return await self.event_store.get_events(
            aggregate_type="Verse",
            aggregate_id=verse_id
        )

    async def replay_to_point(
        self,
        aggregate_id: str,
        target_timestamp: datetime
    ) -> AggregateRoot:
        """Replay events up to a specific point in time."""
        events = await self.event_store.get_events(
            aggregate_type="Verse",
            aggregate_id=aggregate_id
        )

        aggregate = VerseAggregate(aggregate_id)
        for event in events:
            if event.timestamp <= target_timestamp:
                aggregate.apply(event)

        return aggregate
```

---

## Part 8: Event Publishing & Subscriptions

### File: `events/publisher.py`

**Event Publisher**:
```python
class EventPublisher:
    """
    Publishes events to Redis Streams for real-time subscriptions.
    """

    def __init__(self, redis_client):
        self.redis = redis_client

    async def publish(self, event: EventBase) -> None:
        stream_key = f"events:{event.aggregate_type}"
        await self.redis.xadd(stream_key, {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "aggregate_id": event.aggregate_id,
            "data": json.dumps(event.__dict__, default=str)
        })

    async def subscribe(
        self,
        aggregate_type: str,
        handler: Callable[[EventBase], Awaitable[None]],
        from_id: str = "$"
    ) -> None:
        stream_key = f"events:{aggregate_type}"
        while True:
            messages = await self.redis.xread(
                {stream_key: from_id},
                block=5000
            )
            for stream, entries in messages:
                for entry_id, data in entries:
                    event = self.deserialize_event(data)
                    await handler(event)
                    from_id = entry_id
```

---

## Part 9: Testing Specification

### Unit Tests: `tests/events/test_event_sourcing.py`

**Test 1: `test_event_append_and_retrieve`**
- Append events to store
- Retrieve by aggregate ID
- Verify order and content

**Test 2: `test_optimistic_concurrency`**
- Attempt concurrent updates
- Verify version conflict detected
- Handle retry logic

**Test 3: `test_aggregate_hydration`**
- Create aggregate from events
- Verify state matches expected
- Test with snapshots

**Test 4: `test_projection_consistency`**
- Emit events
- Verify all projections updated
- Check read model accuracy

**Test 5: `test_temporal_query`**
- Replay to specific timestamp
- Verify historical state correct

**Test 6: `test_correlation_tracking`**
- Emit related events with same correlation_id
- Query by correlation
- Verify complete chain returned

**Test 7: `test_snapshot_performance`**
- Create aggregate with 1000 events
- Create snapshot
- Verify hydration faster with snapshot

---

## Part 10: Configuration

### Add to `config.py`

**New Configuration Dataclass**: `EventSourcingConfig`

Fields:
- `event_store_table: str = "events"`
- `snapshot_table: str = "event_snapshots"`
- `snapshot_threshold: int = 100` - Create snapshot every N events
- `enable_projections: bool = True`
- `projection_batch_size: int = 100`
- `enable_redis_publishing: bool = True`
- `redis_stream_prefix: str = "events:"`
- `retention_days: int = 365` - How long to keep events
- `enable_compression: bool = True` - Compress old events

---

## Part 11: Migration Strategy

### Phase 1: Dual-Write Period
1. Keep existing CRUD operations
2. Also emit events for all changes
3. Build projections alongside existing data
4. Verify consistency

### Phase 2: Read from Projections
1. Switch reads to use projected read models
2. Keep CRUD writes as backup
3. Monitor for discrepancies

### Phase 3: Event-First
1. All writes go through commands/events
2. CRUD becomes projection-only
3. Enable temporal queries
4. Deprecate direct CRUD

### Migration Script Structure
```python
async def migrate_to_event_sourcing():
    """
    Migrate existing data to event sourced format.
    """
    # 1. Create event tables
    await create_event_tables()

    # 2. Generate historical events from existing data
    async for verse in get_all_verses():
        event = VerseProcessingCompletedEvent(
            # ... reconstruct from existing data
            metadata={"migrated": True, "original_created_at": verse.created_at}
        )
        await event_store.append([event])

    # 3. Rebuild projections
    for projection in projections:
        await projection.rebuild()

    # 4. Verify consistency
    await verify_migration_consistency()
```

---

## Part 12: Success Criteria

### Functional Requirements
- [ ] Event store appends events immutably
- [ ] Aggregates hydrate correctly from events
- [ ] Projections update in real-time
- [ ] Temporal queries return correct historical state
- [ ] Correlation tracking works across related events

### Performance Requirements
- [ ] Event append: < 10ms
- [ ] Aggregate hydration (100 events): < 100ms
- [ ] Aggregate hydration with snapshot: < 20ms
- [ ] Projection update: < 50ms per event
- [ ] Full replay (10,000 events): < 5 minutes

### Reliability Requirements
- [ ] Optimistic concurrency prevents conflicts
- [ ] Projection failures are recoverable
- [ ] Event ordering is guaranteed
- [ ] No data loss on failures

---

## Part 13: Detailed Implementation Order

1. **Create `events/` directory structure**
2. **Define event schemas** in `events/schemas.py`
3. **Create PostgreSQL event tables**
4. **Implement `EventStore` class**
5. **Implement `AggregateRoot` base class**
6. **Create `VerseAggregate`** and `CrossReferenceAggregate`
7. **Implement projection base class**
8. **Create PostgreSQL projection**
9. **Create Neo4j projection**
10. **Implement command handlers**
11. **Add Redis event publishing**
12. **Integrate with `StreamOrchestrator`**
13. **Add configuration to `config.py`**
14. **Write migration scripts**
15. **Write unit tests**
16. **Run dual-write verification**

---

## Part 14: Dependencies on Other Sessions

### Depends On
- None (infrastructure session)

### Depended On By
- SESSION 09: Neo4j Graph-First Architecture (uses event projections)
- SESSION 10: Vector DB Enhancement (subscribes to events)
- SESSION 11: Pipeline Integration (orchestrates via commands)

### External Dependencies
- PostgreSQL for event storage
- Redis Streams for real-time publishing
- Neo4j for graph projections

---

## Session Completion Checklist

```markdown
- [ ] `events/__init__.py` created
- [ ] `events/schemas.py` with all event types
- [ ] `events/event_store.py` implemented
- [ ] `events/aggregates.py` with base and domain aggregates
- [ ] `events/projections.py` with PostgreSQL and Neo4j projections
- [ ] `events/commands.py` with command handlers
- [ ] `events/publisher.py` for Redis publishing
- [ ] PostgreSQL migration for event tables
- [ ] Integration with StreamOrchestrator
- [ ] Configuration added to config.py
- [ ] Unit tests passing
- [ ] Temporal query tests passing
- [ ] Projection consistency verified
- [ ] Migration script created
- [ ] Documentation complete
```

**Next Session**: SESSION 09: Neo4j Graph-First Architecture
