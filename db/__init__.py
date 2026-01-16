"""
BIBLOS v2 - Database Layer (The Memory and Organs)

The database layer represents the organism's vital organs and memory systems:
- PostgreSQL: The muscle tissue - storing and transforming data
- Neo4j: The neural network - mapping relationships (SPIDERWEB)
- Qdrant: The pattern recognition - finding similar patterns
- Redis: The short-term memory - caching recent activity
- Event Store: The hippocampus - recording what happened

This layer follows Dependency Inversion Principle with abstract interfaces
that can be implemented by different backends.

Event Sourcing Philosophy:
    Rather than storing current state (which is mutable truth), we store
    the complete history of what happened (immutable facts). The current
    state is derived by replaying events - like remembering by recalling
    experiences rather than looking at a snapshot.

Architectural Role:
    The database layer doesn't decide what to store - it faithfully
    persists what it's given and retrieves what it's asked for. Like
    organs that don't question why blood flows through them, these
    components serve without judgment or opinion.

Usage:
    from db import (
        # Event Sourcing
        EventStore, StoredEvent, Snapshot,
        # Repositories
        IVerseRepository, ICrossReferenceRepository,
        # Clients
        PostgresClient, Neo4jClient, QdrantVectorStore,
    )

    # Append events
    event_store = EventStore(postgres_client)
    await event_store.append(
        stream_name="verse-GEN.1.1",
        events=[VerseCreated(verse_id="GEN.1.1", text="...")],
        expected_version=0
    )
"""

# ============================================================================
# Models - The Cellular Structure
# ============================================================================
# ORM models define how data is physically stored. Like cells have specific
# structures for their functions, these models define the shape of our data.
# ============================================================================

from db.models_optimized import (
    Base,
    Book,
    Verse,
    CrossReference,
    PatristicCitation,
    ExtractionResult,
)

# ============================================================================
# Database Clients - The Organ Systems
# ============================================================================
# Each client is a specialized organ system. PostgreSQL is the liver
# (processing and storage), Neo4j is the brain's connectome (relationships),
# Qdrant is the pattern cortex (similarity), Redis is working memory.
# ============================================================================

from db.postgres_optimized import PostgresClient, get_db_session
from db.neo4j_optimized import Neo4jClient
from db.qdrant_client import QdrantVectorStore
from db.connection_pool_optimized import ConnectionManager

# ============================================================================
# Event Sourcing - The Hippocampus (Memory Formation)
# ============================================================================
# Events are immutable facts - things that happened. By storing events
# rather than state, we gain complete auditability, time-travel debugging,
# and the ability to rebuild any projection from the source of truth.
# ============================================================================

# Core Events - Now with Seraphic Awareness
# Events know their own aggregate types, schema versions, and subscribers
from db.events import (
    # Seraphic Infrastructure - Events Know Their Nature
    SeraphicEventRegistry,
    EventAffinity,
    UpcastSpec,
    SubscriptionAffinity,
    # Seraphic Decorators
    event,
    subscribes,
    upcasts_from,
    self_aware_event,
    # Event type enumeration
    EventType,
    # Base event with seraphic awareness
    BaseEvent,
    # Verse lifecycle events
    VerseProcessingStarted,
    VerseProcessingCompleted,
    VerseProcessingFailed,
    # Cross-reference discovery events
    CrossReferenceDiscovered,
    CrossReferenceValidated,
    CrossReferenceRejected,
    # Oracle/Analysis events
    OmniResolutionComputed,
    NecessityCalculated,
    LXXDivergenceDetected,
    TypologyDiscovered,
    PropheticProofComputed,
    # Phase lifecycle events
    PhaseStarted,
    PhaseCompleted,
    PhaseFailed,
    # Word analysis events
    WordAnalyzed,
    SemanticFieldMapped,
    # Guardrail events - anonymous wisdom, no named witnesses
    PatristicWitnessAdded,  # DEPRECATED - use InterpretiveGuardrailApplied
    ConsensusCalculated,
    InterpretiveGuardrailApplied,
    HeresyRejected,
    # Constraint events
    TheologicalConstraintApplied,
    ConstraintViolationDetected,
    # Deserialization
    deserialize_event,
    EVENT_REGISTRY,
)

# Event Store - The Enhanced Hippocampus
from db.event_store import (
    # Error hierarchy (specific pain signals)
    EventStoreError,
    ConcurrencyError,
    EventNotFoundError,
    SnapshotError,
    UpcastingError,
    SubscriptionError,
    # Stream categorization
    StreamCategory,
    # Position tracking
    StreamPosition,
    # Event metadata (contextual information)
    EventMetadata,
    # Stored event (immutable fact)
    StoredEvent,
    # Snapshots (memory consolidation)
    Snapshot,
    # Subscription checkpoints
    SubscriptionCheckpoint,
    SubscriptionState,
    PersistentSubscription,
    # Event upcasting (schema evolution)
    IEventUpcaster,
    UpcasterRegistry,
    # Snapshot storage
    ISnapshotStore,
    SnapshotStore,
    # Subscription handling
    ISubscriptionHandler,
    SubscriptionManager,
    # Core interface and implementation
    IEventStore,
    EventStore,
    EventStreamReader,
)

# Commands (intentions to change state)
from db.commands import (
    Command,
    ProcessVerseCommand,
    DiscoverCrossReferencesCommand,
    ValidateCrossReferenceCommand,
    ComputeTypologyCommand,
    ProveProhecyFulfillmentCommand,
    AddPatristicWitnessCommand,
)

# Command handlers (execute intentions)
from db.command_handlers import CommandHandler, CommandDispatcher

# ============================================================================
# Projections - The Senses (Derived Views)
# ============================================================================
# Projections are read models derived from events. Like our senses create
# different views of reality (sight, sound, touch), projections create
# different views of our event stream optimized for specific queries.
# ============================================================================

from db.projections import (
    ProjectionBase,
    CrossReferenceProjection,
    VerseSummaryProjection,
    PatristicConsensusProjection,
    TypologyNetworkProjection,
)
from db.graph_projection import GraphProjection

# ============================================================================
# Neo4j Schema and Algorithms - The Neural Architecture
# ============================================================================
# Neo4j stores the SPIDERWEB - the network of relationships between verses.
# Like neurons form specific connection patterns, our graph has defined
# node types and relationship types.
# ============================================================================

from db.neo4j_schema import (
    NodeLabel,
    RelationshipType,
    GraphMetricType,
    CommunityAlgorithm,
    PathAlgorithm,
    GraphStatistics,
    NodeSchema,
    RelationshipSchema,
    generate_create_indexes_cypher,
    generate_merge_node_cypher,
    generate_create_relationship_cypher,
)

from db.neo4j_algorithms import (
    AlgorithmConfig,
    ALGORITHM_CONFIG,
    ProjectionManager,
    CentralityAlgorithms,
    CommunityAlgorithms,
    PathAlgorithms,
    SimilarityAlgorithms,
    GraphAnalytics,
    Neo4jAlgorithmsClient,
)

# ============================================================================
# Abstract Interfaces - The Contracts
# ============================================================================
# These interfaces define WHAT can be done, not HOW. Any implementation
# that fulfills the contract can be substituted - this is the Dependency
# Inversion Principle in action.
# ============================================================================

from db.interfaces import (
    # Repository pattern interfaces
    IRepository,
    IVerseRepository,
    ICrossReferenceRepository,
    IExtractionResultRepository,
    IPatristicRepository,
    IGraphRepository,
    IVectorStore,
    # Unit of Work pattern
    IUnitOfWork,
    # Generic database client
    IDatabaseClient,
    # Capability interfaces
    Connectable,
    Initializable,
    HealthCheckable,
)


__all__ = [
    # ========================================================================
    # MODELS - Cellular Structure
    # ========================================================================
    "Base",
    "Book",
    "Verse",
    "CrossReference",
    "PatristicCitation",
    "ExtractionResult",

    # ========================================================================
    # DATABASE CLIENTS - Organ Systems
    # ========================================================================
    "PostgresClient",
    "get_db_session",
    "Neo4jClient",
    "QdrantVectorStore",
    "ConnectionManager",

    # ========================================================================
    # EVENT SOURCING - Memory Formation (Seraphic Awareness)
    # ========================================================================
    # Seraphic Infrastructure - Events Know Their Nature
    "SeraphicEventRegistry",
    "EventAffinity",
    "UpcastSpec",
    "SubscriptionAffinity",
    # Seraphic Decorators
    "event",
    "subscribes",
    "upcasts_from",
    "self_aware_event",
    # Event Types
    "EventType",
    "BaseEvent",
    # Verse Events
    "VerseProcessingStarted",
    "VerseProcessingCompleted",
    "VerseProcessingFailed",
    # Cross-Reference Events
    "CrossReferenceDiscovered",
    "CrossReferenceValidated",
    "CrossReferenceRejected",
    # Oracle Events
    "OmniResolutionComputed",
    "NecessityCalculated",
    "LXXDivergenceDetected",
    "TypologyDiscovered",
    "PropheticProofComputed",
    # Phase Events
    "PhaseStarted",
    "PhaseCompleted",
    "PhaseFailed",
    # Word Events
    "WordAnalyzed",
    "SemanticFieldMapped",
    # Guardrail Events - Anonymous Wisdom Encoded as Rules
    "PatristicWitnessAdded",  # DEPRECATED
    "ConsensusCalculated",
    "InterpretiveGuardrailApplied",
    "HeresyRejected",
    # Constraint Events
    "TheologicalConstraintApplied",
    "ConstraintViolationDetected",
    # Deserialization
    "deserialize_event",
    "EVENT_REGISTRY",
    # Event Store Errors
    "EventStoreError",
    "ConcurrencyError",
    "EventNotFoundError",
    "SnapshotError",
    "UpcastingError",
    "SubscriptionError",
    # Stream Management
    "StreamCategory",
    "StreamPosition",
    "EventMetadata",
    "StoredEvent",
    # Snapshots
    "Snapshot",
    "ISnapshotStore",
    "SnapshotStore",
    # Subscriptions
    "SubscriptionCheckpoint",
    "SubscriptionState",
    "PersistentSubscription",
    "ISubscriptionHandler",
    "SubscriptionManager",
    # Upcasting
    "IEventUpcaster",
    "UpcasterRegistry",
    # Core Event Store
    "IEventStore",
    "EventStore",
    "EventStreamReader",
    # Commands
    "Command",
    "ProcessVerseCommand",
    "DiscoverCrossReferencesCommand",
    "ValidateCrossReferenceCommand",
    "ComputeTypologyCommand",
    "ProveProhecyFulfillmentCommand",
    "AddPatristicWitnessCommand",
    "CommandHandler",
    "CommandDispatcher",

    # ========================================================================
    # PROJECTIONS - Derived Senses
    # ========================================================================
    "ProjectionBase",
    "CrossReferenceProjection",
    "VerseSummaryProjection",
    "PatristicConsensusProjection",
    "TypologyNetworkProjection",
    "GraphProjection",

    # ========================================================================
    # NEO4J GRAPH - Neural Architecture
    # ========================================================================
    "NodeLabel",
    "RelationshipType",
    "GraphMetricType",
    "CommunityAlgorithm",
    "PathAlgorithm",
    "GraphStatistics",
    "NodeSchema",
    "RelationshipSchema",
    "generate_create_indexes_cypher",
    "generate_merge_node_cypher",
    "generate_create_relationship_cypher",
    # Algorithms
    "AlgorithmConfig",
    "ALGORITHM_CONFIG",
    "ProjectionManager",
    "CentralityAlgorithms",
    "CommunityAlgorithms",
    "PathAlgorithms",
    "SimilarityAlgorithms",
    "GraphAnalytics",
    "Neo4jAlgorithmsClient",

    # ========================================================================
    # ABSTRACT INTERFACES - Contracts
    # ========================================================================
    "IRepository",
    "IVerseRepository",
    "ICrossReferenceRepository",
    "IExtractionResultRepository",
    "IPatristicRepository",
    "IGraphRepository",
    "IVectorStore",
    "IUnitOfWork",
    "IDatabaseClient",
    "Connectable",
    "Initializable",
    "HealthCheckable",
]
