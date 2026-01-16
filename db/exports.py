"""
BIBLOS v2 - Database Module Exports

Central export point for all database clients and utilities.
Provides a unified interface for database access across the system.

Usage:
    from db.exports import (
        # Clients
        Neo4jClient, QdrantVectorStore, PostgresClient,
        # Models
        Base, Book, Verse, CrossReference, PatristicQuote,
        # Utilities
        get_session, get_engine, init_db,
    )
"""

# Core database models
from db.models import (
    Base,
    Book,
    Verse,
    Word,
    CrossReference,
    PatristicQuote,
    ExtractionResult,
    GoldenRecord,
    PipelineRun,
)

# Neo4j graph client
from db.neo4j_client import (
    Neo4jClient,
    GraphNode,
    GraphRelationship,
)

# Qdrant vector store
from db.qdrant_client import (
    QdrantVectorStore,
    SearchResult,
)

# Postgres client (if optimized version exists)
try:
    from db.postgres_optimized import (
        PostgresClient,
        ConnectionPool,
    )
except ImportError:
    from db.postgres import (
        PostgresClient,
    )
    ConnectionPool = None

# Connection pool utilities
try:
    from db.connection_pool_optimized import (
        OptimizedConnectionPool,
        PoolMetrics,
        ConnectionState,
    )
except ImportError:
    OptimizedConnectionPool = None
    PoolMetrics = None
    ConnectionState = None

__all__ = [
    # Models
    "Base",
    "Book",
    "Verse",
    "Word",
    "CrossReference",
    "PatristicQuote",
    "ExtractionResult",
    "GoldenRecord",
    "PipelineRun",
    # Clients
    "Neo4jClient",
    "GraphNode",
    "GraphRelationship",
    "QdrantVectorStore",
    "SearchResult",
    "PostgresClient",
    "ConnectionPool",
    "OptimizedConnectionPool",
    "PoolMetrics",
    "ConnectionState",
]
