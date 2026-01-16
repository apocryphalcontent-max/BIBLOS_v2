# SESSION 08: EVENT SOURCING MIGRATION

## Session Overview

**Objective**: Migrate BIBLOS v2 from traditional CRUD patterns to an Event Sourcing architecture with CQRS (Command Query Responsibility Segregation). This enables complete audit trails, temporal queries, and reproducible state reconstruction.

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

### Core Event Infrastructure

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

class EventCategory(Enum):
    """High-level event categorization for routing and filtering."""
    PROCESSING = "processing"      # Verse/batch processing lifecycle
    DISCOVERY = "discovery"        # Cross-reference, pattern discovery
    VALIDATION = "validation"      # Theological, patristic validation
    ORACLE = "oracle"              # Five Impossible Oracles events
    SYSTEM = "system"              # Infrastructure events

    @property
    def retention_days(self) -> int:
        """Category-specific retention policies."""
        return {
            EventCategory.PROCESSING: 90,
            EventCategory.DISCOVERY: 365,
            EventCategory.VALIDATION: 365,
            EventCategory.ORACLE: 730,  # 2 years for research
            EventCategory.SYSTEM: 30
        }[self]


@dataclass
class EventMetadata:
    """Rich metadata for event context and debugging."""
    source_system: str = "biblos_v2"
    pipeline_version: str = ""
    agent_versions: Dict[str, str] = field(default_factory=dict)
    session_id: Optional[str] = None
    batch_id: Optional[str] = None
    environment: str = "production"
    trace_id: Optional[str] = None
    span_id: Optional[str] = None


@dataclass
class EventBase:
    """Base class for all domain events."""
    event_id: str                         # UUID for this event
    event_type: str                       # Discriminator for deserialization
    event_category: EventCategory
    aggregate_id: str                     # Entity this event belongs to
    aggregate_type: str                   # e.g., "Verse", "CrossReference"
    timestamp: datetime
    version: int                          # Aggregate version (optimistic concurrency)
    correlation_id: str                   # Links related events across aggregates
    causation_id: Optional[str] = None    # Event that caused this one
    actor: str = "system"                 # Who/what triggered
    metadata: EventMetadata = field(default_factory=EventMetadata)

    def __post_init__(self):
        if not self.event_id:
            self.event_id = str(uuid.uuid4())
        if not self.correlation_id:
            self.correlation_id = str(uuid.uuid4())

    @property
    def stream_key(self) -> str:
        """Redis stream key for publishing."""
        return f"events:{self.aggregate_type.lower()}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "event_category": self.event_category.value,
            "aggregate_id": self.aggregate_id,
            "aggregate_type": self.aggregate_type,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "actor": self.actor,
            "metadata": self.metadata.__dict__,
            **self._payload_dict()
        }

    def _payload_dict(self) -> Dict[str, Any]:
        """Override in subclasses for event-specific payload."""
        return {}
```

### Domain Events

```python
# Verse Processing Events
@dataclass
class VerseProcessingStartedEvent(EventBase):
    verse_id: str
    book: str
    chapter: int
    verse: int
    text_content: str
    phase: str  # "linguistic", "theological", "intertextual", "validation"
    event_type: str = "VerseProcessingStarted"
    event_category: EventCategory = EventCategory.PROCESSING


@dataclass
class VerseProcessingCompletedEvent(EventBase):
    verse_id: str
    phase: str
    duration_ms: int
    agent_results: Dict[str, Dict[str, Any]]  # agent_name → results
    quality_score: float
    event_type: str = "VerseProcessingCompleted"
    event_category: EventCategory = EventCategory.PROCESSING


@dataclass
class VerseProcessingFailedEvent(EventBase):
    verse_id: str
    phase: str
    error_type: str
    error_message: str
    stack_trace: Optional[str] = None
    retry_count: int = 0
    will_retry: bool = False
    event_type: str = "VerseProcessingFailed"
    event_category: EventCategory = EventCategory.PROCESSING


# Cross-Reference Events
@dataclass
class CrossReferenceDiscoveredEvent(EventBase):
    source_ref: str
    target_ref: str
    connection_type: str
    initial_confidence: float
    discovery_method: str  # "embedding", "gnn", "keyword", "agent"
    feature_vector: List[float]
    semantic_similarity: float
    event_type: str = "CrossReferenceDiscovered"
    event_category: EventCategory = EventCategory.DISCOVERY


@dataclass
class CrossReferenceValidatedEvent(EventBase):
    cross_ref_id: str
    validator_agent: str
    validation_result: str  # "approved", "rejected", "modified", "escalated"
    confidence_before: float
    confidence_after: float
    reasoning: str
    theological_checks: Dict[str, bool]
    event_type: str = "CrossReferenceValidated"
    event_category: EventCategory = EventCategory.VALIDATION


@dataclass
class CrossReferenceRefinedEvent(EventBase):
    cross_ref_id: str
    refinement_type: str  # "mutual_transformation", "necessity", "typology"
    refinement_source: str  # Which Oracle/engine
    old_confidence: float
    new_confidence: float
    enrichments: Dict[str, Any]
    event_type: str = "CrossReferenceRefined"
    event_category: EventCategory = EventCategory.DISCOVERY


# Oracle Engine Events
@dataclass
class NecessityCalculatedEvent(EventBase):
    source_verse: str
    target_verse: str
    necessity_score: float
    necessity_type: str
    necessity_strength: str
    semantic_gaps_filled: int
    presuppositions_detected: List[str]
    event_type: str = "NecessityCalculated"
    event_category: EventCategory = EventCategory.ORACLE


@dataclass
class TypologicalConnectionIdentifiedEvent(EventBase):
    type_ref: str
    antitype_ref: str
    typology_layer: str
    relation_type: str
    fractal_depth: int
    composite_strength: float
    layer_strengths: Dict[str, float]
    event_type: str = "TypologicalConnectionIdentified"
    event_category: EventCategory = EventCategory.ORACLE


@dataclass
class LXXDivergenceDetectedEvent(EventBase):
    verse_id: str
    divergence_type: str
    lxx_reading: str
    mt_reading: str
    christological_category: Optional[str]
    oldest_witness: str
    manuscript_confidence: float
    event_type: str = "LXXDivergenceDetected"
    event_category: EventCategory = EventCategory.ORACLE


@dataclass
class PropheticProofComputedEvent(EventBase):
    prophecy_ids: List[str]
    compound_probability: float
    log_probability: float
    bayesian_posterior: float
    prior_used: float
    bayes_factor: float
    interpretation: str
    event_type: str = "PropheticProofComputed"
    event_category: EventCategory = EventCategory.ORACLE
```

---

## Part 3: Event Store Implementation

### PostgreSQL Schema

```sql
-- Main event log (append-only)
CREATE TABLE events (
    sequence_number BIGSERIAL PRIMARY KEY,
    event_id UUID NOT NULL UNIQUE,
    event_type VARCHAR(255) NOT NULL,
    event_category VARCHAR(50) NOT NULL,
    aggregate_id VARCHAR(255) NOT NULL,
    aggregate_type VARCHAR(100) NOT NULL,
    version INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    correlation_id UUID NOT NULL,
    causation_id UUID,
    actor VARCHAR(255) NOT NULL,
    data JSONB NOT NULL,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for efficient queries
CREATE INDEX idx_events_aggregate ON events (aggregate_type, aggregate_id, version);
CREATE INDEX idx_events_timestamp ON events (timestamp);
CREATE INDEX idx_events_correlation ON events (correlation_id);
CREATE INDEX idx_events_type ON events (event_type);
CREATE INDEX idx_events_category ON events (event_category);
CREATE INDEX idx_events_sequence ON events (sequence_number);

-- Unique constraint for optimistic concurrency
CREATE UNIQUE INDEX idx_events_aggregate_version
    ON events (aggregate_type, aggregate_id, version);

-- Snapshots for faster aggregate hydration
CREATE TABLE event_snapshots (
    snapshot_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    aggregate_id VARCHAR(255) NOT NULL,
    aggregate_type VARCHAR(100) NOT NULL,
    version INTEGER NOT NULL,
    state_data JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (aggregate_type, aggregate_id, version)
);

-- Projection checkpoints for resumable projections
CREATE TABLE projection_checkpoints (
    projection_name VARCHAR(255) PRIMARY KEY,
    last_sequence_number BIGINT NOT NULL DEFAULT 0,
    last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

### EventStore Class

```python
from typing import AsyncIterator, Tuple

class EventStore:
    """Append-only event store with optimistic concurrency."""

    def __init__(self, postgres_pool, config: EventSourcingConfig):
        self.pool = postgres_pool
        self.config = config
        self._event_registry: Dict[str, type] = {}

    def register_event(self, event_class: type) -> None:
        """Register event type for deserialization."""
        self._event_registry[event_class.__name__] = event_class

    async def append(
        self,
        events: List[EventBase],
        expected_version: Optional[int] = None
    ) -> int:
        """
        Append events atomically with optimistic concurrency.
        Returns the new aggregate version.
        Raises ConcurrencyError if version mismatch.
        """
        if not events:
            return expected_version or 0

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Check current version if expected_version provided
                if expected_version is not None:
                    current = await conn.fetchval("""
                        SELECT MAX(version) FROM events
                        WHERE aggregate_type = $1 AND aggregate_id = $2
                    """, events[0].aggregate_type, events[0].aggregate_id)

                    if (current or 0) != expected_version:
                        raise ConcurrencyError(
                            f"Expected version {expected_version}, found {current}"
                        )

                # Append all events
                for event in events:
                    await conn.execute("""
                        INSERT INTO events (
                            event_id, event_type, event_category,
                            aggregate_id, aggregate_type, version,
                            timestamp, correlation_id, causation_id,
                            actor, data, metadata
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    """,
                        event.event_id, event.event_type, event.event_category.value,
                        event.aggregate_id, event.aggregate_type, event.version,
                        event.timestamp, event.correlation_id, event.causation_id,
                        event.actor, json.dumps(event.to_dict()),
                        json.dumps(event.metadata.__dict__)
                    )

                return events[-1].version

    async def get_events(
        self,
        aggregate_type: str,
        aggregate_id: str,
        from_version: int = 0,
        to_version: Optional[int] = None
    ) -> List[EventBase]:
        """Get events for an aggregate."""
        async with self.pool.acquire() as conn:
            query = """
                SELECT data, event_type FROM events
                WHERE aggregate_type = $1 AND aggregate_id = $2
                AND version >= $3
            """
            params = [aggregate_type, aggregate_id, from_version]

            if to_version is not None:
                query += " AND version <= $4"
                params.append(to_version)

            query += " ORDER BY version ASC"
            rows = await conn.fetch(query, *params)

            return [self._deserialize(row['data'], row['event_type']) for row in rows]

    async def get_all_events(
        self,
        from_sequence: int = 0,
        batch_size: int = 1000
    ) -> AsyncIterator[Tuple[int, EventBase]]:
        """Stream all events for replay/projection."""
        async with self.pool.acquire() as conn:
            current_seq = from_sequence
            while True:
                rows = await conn.fetch("""
                    SELECT sequence_number, data, event_type FROM events
                    WHERE sequence_number > $1
                    ORDER BY sequence_number ASC
                    LIMIT $2
                """, current_seq, batch_size)

                if not rows:
                    break

                for row in rows:
                    event = self._deserialize(row['data'], row['event_type'])
                    yield row['sequence_number'], event
                    current_seq = row['sequence_number']

    async def save_snapshot(
        self,
        aggregate_type: str,
        aggregate_id: str,
        version: int,
        state_data: Dict
    ) -> None:
        """Save aggregate snapshot."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO event_snapshots (aggregate_id, aggregate_type, version, state_data)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (aggregate_type, aggregate_id, version)
                DO UPDATE SET state_data = EXCLUDED.state_data, created_at = NOW()
            """, aggregate_id, aggregate_type, version, json.dumps(state_data))

    async def get_latest_snapshot(
        self,
        aggregate_type: str,
        aggregate_id: str
    ) -> Optional[Tuple[int, Dict]]:
        """Get latest snapshot for faster hydration."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT version, state_data FROM event_snapshots
                WHERE aggregate_type = $1 AND aggregate_id = $2
                ORDER BY version DESC LIMIT 1
            """, aggregate_type, aggregate_id)

            if row:
                return row['version'], json.loads(row['state_data'])
            return None

    def _deserialize(self, data: str, event_type: str) -> EventBase:
        """Deserialize event from JSON."""
        event_class = self._event_registry.get(event_type)
        if not event_class:
            raise ValueError(f"Unknown event type: {event_type}")
        return event_class(**json.loads(data))


class ConcurrencyError(Exception):
    """Raised when optimistic concurrency check fails."""
    pass
```

---

## Part 4: Aggregate Implementation

```python
from abc import ABC, abstractmethod

class AggregateRoot(ABC):
    """Base class for event-sourced aggregates."""

    def __init__(self, aggregate_id: str):
        self.aggregate_id = aggregate_id
        self.aggregate_type = self.__class__.__name__
        self.version = 0
        self._pending_events: List[EventBase] = []
        self._correlation_id: Optional[str] = None

    def apply(self, event: EventBase) -> None:
        """Apply event to update state (used during hydration)."""
        handler_name = f"_apply_{event.event_type}"
        handler = getattr(self, handler_name, None)
        if handler:
            handler(event)
        self.version = event.version

    def record(self, event: EventBase) -> None:
        """Record new event (used during command execution)."""
        event.version = self.version + 1
        event.correlation_id = self._correlation_id or str(uuid.uuid4())
        self._pending_events.append(event)
        self.apply(event)

    def set_correlation_id(self, correlation_id: str) -> None:
        """Set correlation ID for all events in this operation."""
        self._correlation_id = correlation_id

    def get_pending_events(self) -> List[EventBase]:
        return self._pending_events

    def clear_pending_events(self) -> None:
        self._pending_events = []

    @abstractmethod
    def get_snapshot_state(self) -> Dict[str, Any]:
        """Return state for snapshotting."""
        pass

    @abstractmethod
    def restore_from_snapshot(self, state: Dict[str, Any]) -> None:
        """Restore state from snapshot."""
        pass


class CrossReferenceAggregate(AggregateRoot):
    """Aggregate for cross-reference lifecycle."""

    def __init__(self, cross_ref_id: str):
        super().__init__(cross_ref_id)
        self.source_ref: Optional[str] = None
        self.target_ref: Optional[str] = None
        self.connection_type: Optional[str] = None
        self.confidence: float = 0.0
        self.status: str = "pending"
        self.validation_history: List[Dict] = []
        self.enrichments: Dict[str, Any] = {}

    def discover(
        self,
        source_ref: str,
        target_ref: str,
        connection_type: str,
        confidence: float,
        method: str,
        features: List[float],
        similarity: float,
        actor: str,
        metadata: EventMetadata
    ) -> None:
        """Record discovery of a new cross-reference."""
        self.record(CrossReferenceDiscoveredEvent(
            event_id=str(uuid.uuid4()),
            aggregate_id=self.aggregate_id,
            aggregate_type="CrossReference",
            timestamp=datetime.utcnow(),
            version=0,
            correlation_id="",
            actor=actor,
            metadata=metadata,
            source_ref=source_ref,
            target_ref=target_ref,
            connection_type=connection_type,
            initial_confidence=confidence,
            discovery_method=method,
            feature_vector=features,
            semantic_similarity=similarity
        ))

    def _apply_CrossReferenceDiscovered(self, event: CrossReferenceDiscoveredEvent) -> None:
        self.source_ref = event.source_ref
        self.target_ref = event.target_ref
        self.connection_type = event.connection_type
        self.confidence = event.initial_confidence
        self.status = "discovered"

    def validate(
        self,
        validator: str,
        result: str,
        reasoning: str,
        theological_checks: Dict[str, bool],
        actor: str
    ) -> None:
        """Record validation of cross-reference."""
        confidence_delta = self._calculate_validation_delta(result, theological_checks)
        new_confidence = max(0.0, min(1.0, self.confidence + confidence_delta))

        self.record(CrossReferenceValidatedEvent(
            event_id=str(uuid.uuid4()),
            aggregate_id=self.aggregate_id,
            aggregate_type="CrossReference",
            timestamp=datetime.utcnow(),
            version=0,
            correlation_id="",
            actor=actor,
            metadata=EventMetadata(),
            cross_ref_id=self.aggregate_id,
            validator_agent=validator,
            validation_result=result,
            confidence_before=self.confidence,
            confidence_after=new_confidence,
            reasoning=reasoning,
            theological_checks=theological_checks
        ))

    def _apply_CrossReferenceValidated(self, event: CrossReferenceValidatedEvent) -> None:
        self.confidence = event.confidence_after
        self.validation_history.append({
            "validator": event.validator_agent,
            "result": event.validation_result,
            "timestamp": event.timestamp.isoformat()
        })
        if event.validation_result == "approved":
            self.status = "validated"
        elif event.validation_result == "rejected":
            self.status = "rejected"

    def _calculate_validation_delta(self, result: str, checks: Dict[str, bool]) -> float:
        base_delta = {"approved": 0.1, "rejected": -0.2, "modified": 0.0, "escalated": 0.05}
        delta = base_delta.get(result, 0.0)
        # Bonus/penalty for theological checks
        passed = sum(1 for v in checks.values() if v)
        check_bonus = (passed / len(checks) - 0.5) * 0.1 if checks else 0
        return delta + check_bonus

    def get_snapshot_state(self) -> Dict[str, Any]:
        return {
            "source_ref": self.source_ref,
            "target_ref": self.target_ref,
            "connection_type": self.connection_type,
            "confidence": self.confidence,
            "status": self.status,
            "validation_history": self.validation_history,
            "enrichments": self.enrichments
        }

    def restore_from_snapshot(self, state: Dict[str, Any]) -> None:
        self.source_ref = state["source_ref"]
        self.target_ref = state["target_ref"]
        self.connection_type = state["connection_type"]
        self.confidence = state["confidence"]
        self.status = state["status"]
        self.validation_history = state["validation_history"]
        self.enrichments = state["enrichments"]
```

---

## Part 5: Read Model Projections

```python
class ProjectionBase(ABC):
    """Base class for read model projections."""

    def __init__(self, event_store: EventStore, projection_name: str):
        self.event_store = event_store
        self.projection_name = projection_name
        self.last_sequence = 0

    @abstractmethod
    async def project(self, event: EventBase) -> None:
        """Project a single event to read model."""
        pass

    async def rebuild(self) -> None:
        """Rebuild projection from all events."""
        await self._reset_read_model()
        async for seq, event in self.event_store.get_all_events():
            await self.project(event)
            self.last_sequence = seq
        await self._save_checkpoint()

    async def catch_up(self) -> int:
        """Process events since last checkpoint. Returns count processed."""
        count = 0
        async for seq, event in self.event_store.get_all_events(self.last_sequence):
            await self.project(event)
            self.last_sequence = seq
            count += 1
        await self._save_checkpoint()
        return count

    @abstractmethod
    async def _reset_read_model(self) -> None:
        """Clear read model for rebuild."""
        pass

    async def _save_checkpoint(self) -> None:
        """Save projection checkpoint."""
        async with self.event_store.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO projection_checkpoints (projection_name, last_sequence_number)
                VALUES ($1, $2)
                ON CONFLICT (projection_name) DO UPDATE
                SET last_sequence_number = EXCLUDED.last_sequence_number,
                    last_updated = NOW()
            """, self.projection_name, self.last_sequence)


class Neo4jGraphProjection(ProjectionBase):
    """Projects discovery and typology events to Neo4j."""

    def __init__(self, event_store: EventStore, neo4j_client):
        super().__init__(event_store, "neo4j_graph")
        self.neo4j = neo4j_client

    async def project(self, event: EventBase) -> None:
        handler = getattr(self, f"_handle_{event.event_type}", None)
        if handler:
            await handler(event)

    async def _handle_CrossReferenceDiscovered(self, event: CrossReferenceDiscoveredEvent) -> None:
        await self.neo4j.execute("""
            MERGE (s:Verse {id: $source_ref})
            MERGE (t:Verse {id: $target_ref})
            CREATE (s)-[:CROSS_REFERENCE {
                id: $cross_ref_id,
                type: $connection_type,
                confidence: $confidence,
                method: $method,
                discovered_at: datetime($timestamp)
            }]->(t)
        """, source_ref=event.source_ref, target_ref=event.target_ref,
             cross_ref_id=event.aggregate_id, connection_type=event.connection_type,
             confidence=event.initial_confidence, method=event.discovery_method,
             timestamp=event.timestamp.isoformat())

    async def _handle_CrossReferenceValidated(self, event: CrossReferenceValidatedEvent) -> None:
        await self.neo4j.execute("""
            MATCH ()-[r:CROSS_REFERENCE {id: $cross_ref_id}]->()
            SET r.confidence = $confidence,
                r.status = $status,
                r.validated_at = datetime($timestamp)
        """, cross_ref_id=event.cross_ref_id, confidence=event.confidence_after,
             status=event.validation_result, timestamp=event.timestamp.isoformat())

    async def _handle_TypologicalConnectionIdentified(self, event: TypologicalConnectionIdentifiedEvent) -> None:
        await self.neo4j.execute("""
            MERGE (t:Verse {id: $type_ref})
            MERGE (a:Verse {id: $antitype_ref})
            CREATE (t)-[:TYPIFIES {
                layer: $layer,
                relation: $relation,
                fractal_depth: $depth,
                strength: $strength,
                identified_at: datetime($timestamp)
            }]->(a)
        """, type_ref=event.type_ref, antitype_ref=event.antitype_ref,
             layer=event.typology_layer, relation=event.relation_type,
             depth=event.fractal_depth, strength=event.composite_strength,
             timestamp=event.timestamp.isoformat())

    async def _reset_read_model(self) -> None:
        await self.neo4j.execute("MATCH ()-[r:CROSS_REFERENCE]->() DELETE r")
        await self.neo4j.execute("MATCH ()-[r:TYPIFIES]->() DELETE r")
```

---

## Part 6: Command Handlers

```python
@dataclass
class Command:
    """Base command class."""
    command_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: Optional[str] = None
    actor: str = "system"
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DiscoverCrossReferencesCommand(Command):
    verse_id: str
    verse_text: str
    discovery_methods: List[str] = field(default_factory=lambda: ["embedding", "gnn"])
    min_confidence: float = 0.5


class CommandHandler:
    """Executes commands by orchestrating aggregates and events."""

    def __init__(
        self,
        event_store: EventStore,
        discovery_pipeline,
        projections: List[ProjectionBase]
    ):
        self.event_store = event_store
        self.discovery = discovery_pipeline
        self.projections = projections

    async def handle(self, command: Command) -> Dict[str, Any]:
        """Route command to appropriate handler."""
        handler = getattr(self, f"_handle_{type(command).__name__}", None)
        if not handler:
            raise ValueError(f"No handler for {type(command).__name__}")

        result = await handler(command)

        # Project events to read models
        for projection in self.projections:
            await projection.catch_up()

        return result

    async def _handle_DiscoverCrossReferencesCommand(
        self, command: DiscoverCrossReferencesCommand
    ) -> Dict[str, Any]:
        """Execute cross-reference discovery."""
        candidates = await self.discovery.discover(
            command.verse_id,
            command.verse_text,
            methods=command.discovery_methods
        )

        discovered = []
        for candidate in candidates:
            if candidate.confidence >= command.min_confidence:
                cross_ref_id = f"{command.verse_id}:{candidate.target_ref}"

                aggregate = CrossReferenceAggregate(cross_ref_id)
                aggregate.set_correlation_id(command.correlation_id or command.command_id)

                aggregate.discover(
                    source_ref=command.verse_id,
                    target_ref=candidate.target_ref,
                    connection_type=candidate.connection_type,
                    confidence=candidate.confidence,
                    method=candidate.method,
                    features=candidate.features,
                    similarity=candidate.similarity,
                    actor=command.actor,
                    metadata=EventMetadata(session_id=command.command_id)
                )

                await self.event_store.append(aggregate.get_pending_events())
                discovered.append(cross_ref_id)

        return {"discovered_count": len(discovered), "cross_ref_ids": discovered}
```

---

## Part 7: Event Publishing (Redis Streams)

```python
class EventPublisher:
    """Publishes events to Redis Streams for real-time consumers."""

    def __init__(self, redis_client, config: EventSourcingConfig):
        self.redis = redis_client
        self.config = config

    async def publish(self, event: EventBase) -> str:
        """Publish event to appropriate stream. Returns message ID."""
        stream_key = f"{self.config.redis_stream_prefix}{event.aggregate_type.lower()}"

        message_id = await self.redis.xadd(stream_key, {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "aggregate_id": event.aggregate_id,
            "correlation_id": event.correlation_id,
            "timestamp": event.timestamp.isoformat(),
            "data": json.dumps(event.to_dict())
        }, maxlen=self.config.stream_max_length)

        return message_id

    async def subscribe(
        self,
        aggregate_types: List[str],
        consumer_group: str,
        consumer_name: str,
        handler: Callable[[EventBase], Awaitable[None]]
    ) -> None:
        """Subscribe to event streams with consumer group."""
        streams = {
            f"{self.config.redis_stream_prefix}{t.lower()}": ">"
            for t in aggregate_types
        }

        # Create consumer groups if needed
        for stream in streams:
            try:
                await self.redis.xgroup_create(stream, consumer_group, id="0", mkstream=True)
            except Exception:
                pass  # Group already exists

        while True:
            messages = await self.redis.xreadgroup(
                consumer_group, consumer_name, streams, count=100, block=5000
            )

            for stream, entries in messages:
                for entry_id, data in entries:
                    try:
                        event = self._deserialize(data["data"])
                        await handler(event)
                        await self.redis.xack(stream, consumer_group, entry_id)
                    except Exception as e:
                        # Handle failure - message remains pending
                        logging.error(f"Failed to process {entry_id}: {e}")
```

---

## Part 8: Configuration

```python
@dataclass
class EventSourcingConfig:
    """Configuration for event sourcing infrastructure."""

    # PostgreSQL tables
    event_table: str = "events"
    snapshot_table: str = "event_snapshots"
    checkpoint_table: str = "projection_checkpoints"

    # Snapshot policy
    snapshot_threshold: int = 100       # Create snapshot every N events
    snapshot_retention: int = 10        # Keep N most recent snapshots

    # Projection settings
    projection_batch_size: int = 100
    projection_catch_up_interval_ms: int = 1000

    # Redis publishing
    enable_redis_publishing: bool = True
    redis_stream_prefix: str = "events:"
    stream_max_length: int = 100000

    # Retention
    retention_days: int = 365
    compression_after_days: int = 90

    # Concurrency
    max_retries_on_conflict: int = 3
    retry_delay_ms: int = 100
```

---

## Part 9: Migration Strategy

### Phase 1: Dual-Write (Week 1-2)
1. Deploy event tables alongside existing CRUD
2. Emit events after every CRUD write
3. Build projections in parallel
4. Verify read model consistency

### Phase 2: Event-First Writes (Week 3)
1. All writes go through commands → events
2. CRUD becomes projection-only
3. Monitor for discrepancies
4. Rollback capability via event replay

### Phase 3: Full Event Sourcing (Week 4+)
1. Remove CRUD write paths
2. Enable temporal queries
3. Archive/compress old events
4. Performance optimization

---

## Part 10: Testing Specification

### Test Cases: `tests/events/test_event_sourcing.py`

**Test 1: `test_event_append_optimistic_concurrency`**
- Append events with expected version
- Verify ConcurrencyError on mismatch

**Test 2: `test_aggregate_hydration_from_events`**
- Create aggregate from event stream
- Verify final state correct

**Test 3: `test_aggregate_hydration_with_snapshot`**
- Create snapshot at version 50
- Hydrate from snapshot + remaining events
- Verify faster than full replay

**Test 4: `test_projection_catch_up`**
- Emit 100 events
- Run projection catch-up
- Verify read model complete

**Test 5: `test_temporal_replay`**
- Emit events over time
- Replay to specific timestamp
- Verify historical state

---

## Part 11: Success Criteria

### Functional
- [ ] Events append atomically with optimistic concurrency
- [ ] Aggregates hydrate correctly from events
- [ ] Snapshots accelerate hydration
- [ ] Projections update consistently
- [ ] Temporal queries work

### Performance
- [ ] Event append: < 10ms
- [ ] Hydration (100 events): < 100ms
- [ ] Hydration with snapshot: < 20ms
- [ ] Projection per event: < 50ms

---

## Session Completion Checklist

```markdown
- [ ] `events/schemas.py` with all event types
- [ ] `events/event_store.py` implemented
- [ ] `events/aggregates.py` with domain aggregates
- [ ] `events/projections.py` with Neo4j projection
- [ ] `events/commands.py` with handlers
- [ ] `events/publisher.py` for Redis
- [ ] PostgreSQL migration scripts
- [ ] Configuration in config.py
- [ ] Unit tests passing
- [ ] Migration scripts ready
```

**Next Session**: SESSION 09: Neo4j Graph-First Architecture
