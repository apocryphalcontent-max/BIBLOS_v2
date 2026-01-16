"""
BIBLOS v2 - Database Abstraction Interfaces

Provides abstract interfaces for database operations following the
Dependency Inversion Principle (DIP). Concrete implementations can
be swapped without changing dependent code.

Features:
- Repository pattern for domain-driven design
- Protocol classes for structural typing
- Unit of Work pattern for transaction management
- Database-agnostic interfaces

Usage:
    from db.interfaces import VerseRepository, IUnitOfWork

    class AgentWithDI:
        def __init__(self, verse_repo: VerseRepository):
            self._verses = verse_repo

        async def process(self, verse_id: str):
            verse = await self._verses.get_by_id(verse_id)
            ...
"""
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

# Type variables for generic repository
T = TypeVar("T")
ID = TypeVar("ID")


# =============================================================================
# BASE REPOSITORY INTERFACE
# =============================================================================


class IRepository(ABC, Generic[T, ID]):
    """
    Abstract base repository interface.

    Defines standard CRUD operations that all repositories must implement.
    Generic over entity type T and identifier type ID.
    """

    @abstractmethod
    async def get_by_id(self, id: ID) -> Optional[T]:
        """Retrieve entity by identifier."""
        pass

    @abstractmethod
    async def get_all(self, limit: int = 100, offset: int = 0) -> List[T]:
        """Retrieve all entities with pagination."""
        pass

    @abstractmethod
    async def add(self, entity: T) -> T:
        """Add a new entity."""
        pass

    @abstractmethod
    async def update(self, entity: T) -> T:
        """Update an existing entity."""
        pass

    @abstractmethod
    async def delete(self, id: ID) -> bool:
        """Delete entity by identifier. Returns True if deleted."""
        pass

    @abstractmethod
    async def exists(self, id: ID) -> bool:
        """Check if entity exists."""
        pass


# =============================================================================
# DOMAIN-SPECIFIC REPOSITORY INTERFACES
# =============================================================================


class IVerseRepository(ABC):
    """
    Repository interface for Bible verse operations.

    Provides verse-specific query methods beyond basic CRUD.
    """

    @abstractmethod
    async def get_by_reference(self, reference: str) -> Optional[Dict[str, Any]]:
        """Get verse by canonical reference (e.g., 'GEN.1.1')."""
        pass

    @abstractmethod
    async def get_by_book(self, book_code: str) -> List[Dict[str, Any]]:
        """Get all verses in a book."""
        pass

    @abstractmethod
    async def get_by_chapter(
        self, book_code: str, chapter: int
    ) -> List[Dict[str, Any]]:
        """Get all verses in a chapter."""
        pass

    @abstractmethod
    async def get_range(
        self,
        start_ref: str,
        end_ref: str,
    ) -> List[Dict[str, Any]]:
        """Get verses in a range."""
        pass

    @abstractmethod
    async def upsert(self, verse_data: Dict[str, Any]) -> Dict[str, Any]:
        """Insert or update verse."""
        pass

    @abstractmethod
    async def batch_upsert(self, verses: List[Dict[str, Any]]) -> int:
        """Batch insert or update verses. Returns count affected."""
        pass

    @abstractmethod
    async def search_text(
        self, query: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Full-text search in verse content."""
        pass


class ICrossReferenceRepository(ABC):
    """
    Repository interface for cross-reference operations.
    """

    @abstractmethod
    async def get_by_source(
        self, source_ref: str
    ) -> List[Dict[str, Any]]:
        """Get all cross-references from a source verse."""
        pass

    @abstractmethod
    async def get_by_target(
        self, target_ref: str
    ) -> List[Dict[str, Any]]:
        """Get all cross-references to a target verse."""
        pass

    @abstractmethod
    async def get_by_type(
        self, connection_type: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get cross-references by connection type."""
        pass

    @abstractmethod
    async def get_bidirectional(
        self, verse_ref: str
    ) -> List[Dict[str, Any]]:
        """Get all cross-references involving a verse (as source or target)."""
        pass

    @abstractmethod
    async def add(self, cross_ref: Dict[str, Any]) -> Dict[str, Any]:
        """Add a new cross-reference."""
        pass

    @abstractmethod
    async def batch_add(self, cross_refs: List[Dict[str, Any]]) -> int:
        """Batch add cross-references. Returns count added."""
        pass

    @abstractmethod
    async def delete_by_source(self, source_ref: str) -> int:
        """Delete all cross-references from a source. Returns count deleted."""
        pass


class IExtractionResultRepository(ABC):
    """
    Repository interface for extraction result storage.
    """

    @abstractmethod
    async def get_by_verse_and_agent(
        self, verse_id: str, agent_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get extraction result for a verse from a specific agent."""
        pass

    @abstractmethod
    async def get_all_for_verse(
        self, verse_id: str
    ) -> List[Dict[str, Any]]:
        """Get all extraction results for a verse."""
        pass

    @abstractmethod
    async def save(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Save extraction result (insert or update)."""
        pass

    @abstractmethod
    async def batch_save(self, results: List[Dict[str, Any]]) -> int:
        """Batch save extraction results. Returns count saved."""
        pass

    @abstractmethod
    async def get_latest_by_agent(
        self, agent_name: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get latest extraction results from an agent."""
        pass


class IPatristicRepository(ABC):
    """
    Repository interface for patristic citation operations.
    """

    @abstractmethod
    async def get_by_verse(
        self, verse_ref: str
    ) -> List[Dict[str, Any]]:
        """Get all patristic citations for a verse."""
        pass

    @abstractmethod
    async def get_by_father(
        self, father_name: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get citations by a Church Father."""
        pass

    @abstractmethod
    async def search(
        self, query: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search patristic citations."""
        pass

    @abstractmethod
    async def add(self, citation: Dict[str, Any]) -> Dict[str, Any]:
        """Add a patristic citation."""
        pass


# =============================================================================
# GRAPH DATABASE INTERFACE
# =============================================================================


class IGraphRepository(ABC):
    """
    Repository interface for graph database operations (Neo4j, etc.).

    Provides methods for node and relationship operations in the
    SPIDERWEB schema.
    """

    @abstractmethod
    async def create_verse_node(
        self, verse_ref: str, properties: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a verse node in the graph."""
        pass

    @abstractmethod
    async def create_relationship(
        self,
        source_ref: str,
        target_ref: str,
        rel_type: str,
        properties: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create a relationship between verses."""
        pass

    @abstractmethod
    async def get_connected_verses(
        self,
        verse_ref: str,
        rel_type: Optional[str] = None,
        direction: str = "both",
        max_depth: int = 1,
    ) -> List[Dict[str, Any]]:
        """Get verses connected to a given verse."""
        pass

    @abstractmethod
    async def get_path(
        self, source_ref: str, target_ref: str, max_depth: int = 5
    ) -> Optional[List[Dict[str, Any]]]:
        """Find shortest path between two verses."""
        pass

    @abstractmethod
    async def get_subgraph(
        self, verse_refs: List[str], include_relationships: bool = True
    ) -> Dict[str, Any]:
        """Get a subgraph containing the specified verses."""
        pass


# =============================================================================
# VECTOR STORE INTERFACE
# =============================================================================


class IVectorStore(ABC):
    """
    Interface for vector similarity operations.

    Used for embedding-based cross-reference discovery.
    """

    @abstractmethod
    async def add_embedding(
        self,
        id: str,
        embedding: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        """Add an embedding with metadata."""
        pass

    @abstractmethod
    async def batch_add_embeddings(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> int:
        """Batch add embeddings. Returns count added."""
        pass

    @abstractmethod
    async def search_similar(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar embeddings."""
        pass

    @abstractmethod
    async def delete(self, id: str) -> bool:
        """Delete an embedding by ID."""
        pass


# =============================================================================
# UNIT OF WORK INTERFACE
# =============================================================================


class IUnitOfWork(ABC):
    """
    Unit of Work pattern for managing transactions.

    Ensures atomic operations across multiple repositories.
    """

    # Repository properties (implemented by concrete UoW)
    verses: IVerseRepository
    cross_references: ICrossReferenceRepository
    extraction_results: IExtractionResultRepository
    patristic: IPatristicRepository

    @abstractmethod
    async def __aenter__(self) -> "IUnitOfWork":
        """Enter the unit of work context."""
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the unit of work context."""
        pass

    @abstractmethod
    async def commit(self) -> None:
        """Commit all changes in this unit of work."""
        pass

    @abstractmethod
    async def rollback(self) -> None:
        """Rollback all changes in this unit of work."""
        pass


# =============================================================================
# PROTOCOL CLASSES FOR STRUCTURAL TYPING
# =============================================================================


@runtime_checkable
class Connectable(Protocol):
    """Protocol for database clients that can connect/disconnect."""

    async def connect(self) -> None:
        """Establish connection."""
        ...

    async def disconnect(self) -> None:
        """Close connection."""
        ...

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        ...


@runtime_checkable
class Initializable(Protocol):
    """Protocol for components that require initialization."""

    async def initialize(self) -> None:
        """Initialize the component."""
        ...

    async def close(self) -> None:
        """Close and cleanup the component."""
        ...


@runtime_checkable
class HealthCheckable(Protocol):
    """Protocol for components that support health checks."""

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check.

        Returns dict with at least:
            - healthy: bool
            - message: str
        """
        ...


# =============================================================================
# DATABASE CLIENT INTERFACE
# =============================================================================


class IDatabaseClient(ABC):
    """
    Abstract interface for database clients.

    Provides common lifecycle methods that all database clients
    must implement.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize database connection and resources."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close database connection and cleanup resources."""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on database connection.

        Returns:
            Dictionary with health status information
        """
        pass

    @abstractmethod
    @asynccontextmanager
    async def session(self) -> AsyncGenerator[Any, None]:
        """
        Get a database session/transaction context.

        Yields a session that auto-commits on success and
        rolls back on exception.
        """
        pass
