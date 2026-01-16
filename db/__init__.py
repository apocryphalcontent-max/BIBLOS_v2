"""
BIBLOS v2 - Database Layer

Provides unified access to:
- PostgreSQL (relational data, pgvector embeddings)
- Neo4j (SPIDERWEB graph relationships)
- Qdrant (vector similarity search)
- Redis (caching layer)

Follows Dependency Inversion Principle with abstract interfaces
in db.interfaces that can be implemented by different backends.
"""
# Use optimized versions of core database components
from db.models_optimized import Base, Book, Verse, CrossReference, PatristicCitation, ExtractionResult
from db.postgres_optimized import PostgresClient, get_db_session
from db.neo4j_optimized import Neo4jClient
from db.qdrant_client import QdrantVectorStore
from db.connection_pool_optimized import ConnectionManager

# Event Sourcing components
from db.events import (
    EventBase, VerseProcessingStarted, VerseProcessingCompleted,
    VerseProcessingFailed, CrossReferenceDiscovered, CrossReferenceValidated,
    CrossReferenceRejected, CrossReferenceRefined, TypologicalConnectionIdentified,
    ProphecyFulfillmentProved, PatristicWitnessAdded, OmniResolutionComputed,
    VerseAggregate, CrossReferenceAggregate
)
from db.event_store import EventStore, EventStreamReader
from db.commands import (
    Command, ProcessVerseCommand, DiscoverCrossReferencesCommand,
    ValidateCrossReferenceCommand, ComputeTypologyCommand,
    ProveProhecyFulfillmentCommand, AddPatristicWitnessCommand
)
from db.command_handlers import CommandHandler, CommandDispatcher
from db.projections import (
    ProjectionBase, CrossReferenceProjection, VerseSummaryProjection,
    PatristicConsensusProjection, TypologyNetworkProjection
)
from db.graph_projection import GraphProjection

# Neo4j Schema and Algorithms
from db.neo4j_schema import (
    NodeLabel, RelationshipType, GraphMetricType, CommunityAlgorithm,
    PathAlgorithm, GraphStatistics, NodeSchema, RelationshipSchema,
    generate_create_indexes_cypher, generate_merge_node_cypher,
    generate_create_relationship_cypher
)
from db.neo4j_algorithms import (
    AlgorithmConfig, ALGORITHM_CONFIG, ProjectionManager,
    CentralityAlgorithms, CommunityAlgorithms, PathAlgorithms,
    SimilarityAlgorithms, GraphAnalytics, Neo4jAlgorithmsClient
)

# Abstract interfaces for dependency injection
from db.interfaces import (
    IRepository,
    IVerseRepository,
    ICrossReferenceRepository,
    IExtractionResultRepository,
    IPatristicRepository,
    IGraphRepository,
    IVectorStore,
    IUnitOfWork,
    IDatabaseClient,
    Connectable,
    Initializable,
    HealthCheckable,
)

__all__ = [
    # Models
    "Base", "Book", "Verse", "CrossReference", "PatristicCitation", "ExtractionResult",
    # Concrete implementations
    "PostgresClient", "get_db_session",
    "Neo4jClient",
    "QdrantVectorStore",
    "ConnectionManager",
    # Event Sourcing - Events
    "EventBase", "VerseProcessingStarted", "VerseProcessingCompleted",
    "VerseProcessingFailed", "CrossReferenceDiscovered", "CrossReferenceValidated",
    "CrossReferenceRejected", "CrossReferenceRefined", "TypologicalConnectionIdentified",
    "ProphecyFulfillmentProved", "PatristicWitnessAdded", "OmniResolutionComputed",
    "VerseAggregate", "CrossReferenceAggregate",
    # Event Sourcing - Store and Commands
    "EventStore", "EventStreamReader",
    "Command", "ProcessVerseCommand", "DiscoverCrossReferencesCommand",
    "ValidateCrossReferenceCommand", "ComputeTypologyCommand",
    "ProveProhecyFulfillmentCommand", "AddPatristicWitnessCommand",
    "CommandHandler", "CommandDispatcher",
    # Event Sourcing - Projections
    "ProjectionBase", "CrossReferenceProjection", "VerseSummaryProjection",
    "PatristicConsensusProjection", "TypologyNetworkProjection", "GraphProjection",
    # Neo4j Schema
    "NodeLabel", "RelationshipType", "GraphMetricType", "CommunityAlgorithm",
    "PathAlgorithm", "GraphStatistics", "NodeSchema", "RelationshipSchema",
    "generate_create_indexes_cypher", "generate_merge_node_cypher",
    "generate_create_relationship_cypher",
    # Neo4j Algorithms
    "AlgorithmConfig", "ALGORITHM_CONFIG", "ProjectionManager",
    "CentralityAlgorithms", "CommunityAlgorithms", "PathAlgorithms",
    "SimilarityAlgorithms", "GraphAnalytics", "Neo4jAlgorithmsClient",
    # Abstract interfaces (for DI)
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
