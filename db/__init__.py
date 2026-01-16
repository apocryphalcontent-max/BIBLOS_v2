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
from db.models import Base, Book, Verse, CrossReference, PatristicCitation, ExtractionResult
from db.postgres import PostgresClient, get_db_session
from db.neo4j_client import Neo4jClient
from db.qdrant_client import QdrantVectorStore
from db.connection_pool import ConnectionManager

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
