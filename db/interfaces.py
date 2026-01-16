"""
BIBLOS v2 - Database Abstraction Interfaces

Enterprise-grade database interfaces implementing:
- Specification Pattern for composable, type-safe queries
- Repository Pattern with read/write separation (ISP)
- Unit of Work Pattern with domain event dispatch
- Optimistic Concurrency with version tracking
- Aggregate Root enforcement
- Pagination with total count tracking

Architecture Principles:
    - Interface Segregation: Read-only and Write-only repository interfaces
    - Dependency Inversion: All concrete implementations depend on abstractions
    - Open/Closed: Extensible via Specification without modifying repositories
    - Type Safety: Full generic typing with domain entities, not Dict[str, Any]

Usage:
    from db.interfaces import (
        IReadRepository, IWriteRepository, ISpecification,
        IUnitOfWork, Page, VerseSpec
    )

    class AgentWithDI:
        def __init__(self, verses: IReadRepository[Verse, str]):
            self._verses = verses

        async def process(self, verse_id: str):
            # Type-safe query with specification
            spec = VerseSpec.by_book("GEN").and_(VerseSpec.in_chapter(1))
            verses = await self._verses.find(spec)
            for verse in verses:  # verse is typed as Verse, not Dict
                ...
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    ClassVar,
    Dict,
    FrozenSet,
    Generic,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)
from uuid import UUID, uuid4


# =============================================================================
# TYPE VARIABLES AND CONSTRAINTS
# =============================================================================

# Entity type variable - represents domain entities
T = TypeVar("T", bound="IEntity")
T_co = TypeVar("T_co", covariant=True, bound="IEntity")

# Identifier type variable - represents entity IDs
ID = TypeVar("ID")

# Specification type variable
TSpec = TypeVar("TSpec", bound="ISpecification")

# Domain event type variable
TEvent = TypeVar("TEvent", bound="IDomainEvent")

# Query/Command result types
TResult = TypeVar("TResult")
TQuery = TypeVar("TQuery")

# Unbound type variable for pagination (allows mapping to any type)
U = TypeVar("U")

# Type variable for Page items (unbound to allow any type in pagination)
PageItemT = TypeVar("PageItemT")


# =============================================================================
# DOMAIN PRIMITIVES - BASE INTERFACES
# =============================================================================


class IEntity(Protocol):
    """
    Protocol for domain entities.

    All entities must have an identity that distinguishes them.
    """

    @property
    def id(self) -> Any:
        """Unique identifier for this entity."""
        ...


class IVersionedEntity(IEntity, Protocol):
    """
    Protocol for entities with optimistic concurrency support.

    Version is incremented on each update to detect concurrent modifications.
    """

    @property
    def version(self) -> int:
        """Current version number for optimistic concurrency."""
        ...

    def increment_version(self) -> None:
        """Increment version after successful update."""
        ...


class IAggregateRoot(IVersionedEntity, Protocol):
    """
    Protocol for aggregate roots in DDD.

    Aggregate roots are the entry point to a cluster of domain objects.
    They maintain invariants and emit domain events.
    """

    @property
    def domain_events(self) -> List["IDomainEvent"]:
        """Pending domain events to be dispatched."""
        ...

    def add_domain_event(self, event: "IDomainEvent") -> None:
        """Add a domain event to be dispatched on commit."""
        ...

    def clear_domain_events(self) -> List["IDomainEvent"]:
        """Clear and return all pending domain events."""
        ...


class IDomainEvent(Protocol):
    """
    Protocol for domain events.

    Domain events represent something that happened in the domain
    that domain experts care about.
    """

    @property
    def event_id(self) -> UUID:
        """Unique identifier for this event."""
        ...

    @property
    def event_type(self) -> str:
        """Type name of this event."""
        ...

    @property
    def occurred_at(self) -> datetime:
        """When this event occurred."""
        ...

    @property
    def aggregate_id(self) -> Optional[str]:
        """ID of the aggregate that emitted this event."""
        ...


# =============================================================================
# VALUE OBJECTS
# =============================================================================


@dataclass(frozen=True, slots=True)
class VerseId:
    """
    Value object for verse identifiers.

    Immutable and validated verse reference following BOOK.CHAPTER.VERSE format.
    """
    book: str
    chapter: int
    verse: int

    def __post_init__(self) -> None:
        if not self.book or len(self.book) != 3:
            raise ValueError(f"Invalid book code: {self.book}")
        if self.chapter < 1:
            raise ValueError(f"Invalid chapter: {self.chapter}")
        if self.verse < 1:
            raise ValueError(f"Invalid verse: {self.verse}")

    def __str__(self) -> str:
        return f"{self.book}.{self.chapter}.{self.verse}"

    @classmethod
    def parse(cls, reference: str) -> "VerseId":
        """Parse a verse reference string."""
        parts = reference.split(".")
        if len(parts) != 3:
            raise ValueError(f"Invalid verse reference format: {reference}")
        return cls(book=parts[0], chapter=int(parts[1]), verse=int(parts[2]))

    def to_tuple(self) -> Tuple[str, int, int]:
        """Convert to tuple for sorting/comparison."""
        return (self.book, self.chapter, self.verse)


@dataclass(frozen=True, slots=True)
class ConnectionType:
    """Value object for cross-reference connection types."""

    TYPOLOGICAL: ClassVar[str] = "typological"
    PROPHETIC: ClassVar[str] = "prophetic"
    VERBAL: ClassVar[str] = "verbal"
    THEMATIC: ClassVar[str] = "thematic"
    CONCEPTUAL: ClassVar[str] = "conceptual"
    HISTORICAL: ClassVar[str] = "historical"
    LITURGICAL: ClassVar[str] = "liturgical"
    NARRATIVE: ClassVar[str] = "narrative"
    GENEALOGICAL: ClassVar[str] = "genealogical"
    GEOGRAPHICAL: ClassVar[str] = "geographical"

    value: str

    _VALID_TYPES: ClassVar[FrozenSet[str]] = frozenset({
        "typological", "prophetic", "verbal", "thematic", "conceptual",
        "historical", "liturgical", "narrative", "genealogical", "geographical"
    })

    def __post_init__(self) -> None:
        if self.value not in self._VALID_TYPES:
            raise ValueError(f"Invalid connection type: {self.value}")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class Confidence:
    """Value object for confidence scores with validation."""

    value: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1: {self.value}")

    def __float__(self) -> float:
        return self.value

    @classmethod
    def high(cls) -> "Confidence":
        return cls(0.9)

    @classmethod
    def medium(cls) -> "Confidence":
        return cls(0.7)

    @classmethod
    def low(cls) -> "Confidence":
        return cls(0.5)


# =============================================================================
# PAGINATION
# =============================================================================


@dataclass(frozen=True, slots=True)
class Page(Generic[PageItemT]):
    """
    Immutable pagination result wrapper.

    Provides access to a page of results along with pagination metadata.
    Uses unbound type variable to support any item type (entities, DTOs, etc.).
    """
    items: Tuple[PageItemT, ...]
    total: int
    page: int
    page_size: int

    @property
    def total_pages(self) -> int:
        """Calculate total number of pages."""
        if self.page_size <= 0:
            return 0
        return (self.total + self.page_size - 1) // self.page_size

    @property
    def has_next(self) -> bool:
        """Check if there is a next page."""
        return self.page < self.total_pages

    @property
    def has_previous(self) -> bool:
        """Check if there is a previous page."""
        return self.page > 1

    @property
    def is_empty(self) -> bool:
        """Check if page is empty."""
        return len(self.items) == 0

    def map(self, func: Callable[[PageItemT], U]) -> "Page[U]":
        """Transform items in the page to a different type."""
        return Page(
            items=tuple(func(item) for item in self.items),
            total=self.total,
            page=self.page,
            page_size=self.page_size
        )

    def filter(self, predicate: Callable[[PageItemT], bool]) -> "Page[PageItemT]":
        """Filter items in the page (note: total count unchanged)."""
        filtered = tuple(item for item in self.items if predicate(item))
        return Page(
            items=filtered,
            total=self.total,  # Original total preserved for pagination math
            page=self.page,
            page_size=self.page_size
        )

    @classmethod
    def empty(cls, page: int = 1, page_size: int = 10) -> "Page[PageItemT]":
        """Create an empty page."""
        return cls(items=(), total=0, page=page, page_size=page_size)

    @classmethod
    def of(
        cls, items: Sequence[PageItemT], total: int, page: int, page_size: int
    ) -> "Page[PageItemT]":
        """Create a page from a sequence."""
        return cls(items=tuple(items), total=total, page=page, page_size=page_size)

    @classmethod
    def from_list(
        cls, all_items: Sequence[PageItemT], page: int = 1, page_size: int = 10
    ) -> "Page[PageItemT]":
        """Create a page by slicing from a full list (useful for in-memory pagination)."""
        total = len(all_items)
        offset = (page - 1) * page_size
        items = tuple(all_items[offset:offset + page_size])
        return cls(items=items, total=total, page=page, page_size=page_size)


@dataclass(frozen=True, slots=True)
class PageRequest:
    """Request parameters for pagination."""
    page: int = 1
    page_size: int = 10

    def __post_init__(self) -> None:
        if self.page < 1:
            raise ValueError(f"Page must be >= 1: {self.page}")
        if self.page_size < 1 or self.page_size > 1000:
            raise ValueError(f"Page size must be 1-1000: {self.page_size}")

    @property
    def offset(self) -> int:
        """Calculate offset for database query."""
        return (self.page - 1) * self.page_size

    @property
    def limit(self) -> int:
        """Alias for page_size for database queries."""
        return self.page_size


# =============================================================================
# SPECIFICATION PATTERN
# =============================================================================


class ISpecification(ABC, Generic[T]):
    """
    Specification pattern for composable, type-safe queries.

    Specifications encapsulate query logic that can be combined
    using logical operators (and, or, not).

    Usage:
        # Simple specification
        spec = VerseSpec.by_book("GEN")

        # Combined specification
        spec = (
            VerseSpec.by_book("GEN")
            .and_(VerseSpec.in_chapter(1))
            .and_(VerseSpec.has_text_containing("beginning"))
        )

        # Use with repository
        verses = await repo.find(spec)
    """

    @abstractmethod
    def is_satisfied_by(self, entity: T) -> bool:
        """
        Check if an entity satisfies this specification.

        Used for in-memory filtering.
        """
        pass

    @abstractmethod
    def to_query_params(self) -> Dict[str, Any]:
        """
        Convert specification to query parameters.

        Returns a dictionary that repositories can use to build queries.
        The exact format depends on the repository implementation.
        """
        pass

    def and_(self, other: "ISpecification[T]") -> "ISpecification[T]":
        """Combine with AND logic."""
        return AndSpecification(self, other)

    def or_(self, other: "ISpecification[T]") -> "ISpecification[T]":
        """Combine with OR logic."""
        return OrSpecification(self, other)

    def not_(self) -> "ISpecification[T]":
        """Negate this specification."""
        return NotSpecification(self)

    def __and__(self, other: "ISpecification[T]") -> "ISpecification[T]":
        """Support & operator for AND."""
        return self.and_(other)

    def __or__(self, other: "ISpecification[T]") -> "ISpecification[T]":
        """Support | operator for OR."""
        return self.or_(other)

    def __invert__(self) -> "ISpecification[T]":
        """Support ~ operator for NOT."""
        return self.not_()


class AndSpecification(ISpecification[T]):
    """Specification that combines two specs with AND logic."""

    def __init__(self, left: ISpecification[T], right: ISpecification[T]) -> None:
        self._left = left
        self._right = right

    def is_satisfied_by(self, entity: T) -> bool:
        return self._left.is_satisfied_by(entity) and self._right.is_satisfied_by(entity)

    def to_query_params(self) -> Dict[str, Any]:
        return {
            "$and": [
                self._left.to_query_params(),
                self._right.to_query_params()
            ]
        }


class OrSpecification(ISpecification[T]):
    """Specification that combines two specs with OR logic."""

    def __init__(self, left: ISpecification[T], right: ISpecification[T]) -> None:
        self._left = left
        self._right = right

    def is_satisfied_by(self, entity: T) -> bool:
        return self._left.is_satisfied_by(entity) or self._right.is_satisfied_by(entity)

    def to_query_params(self) -> Dict[str, Any]:
        return {
            "$or": [
                self._left.to_query_params(),
                self._right.to_query_params()
            ]
        }


class NotSpecification(ISpecification[T]):
    """Specification that negates another spec."""

    def __init__(self, spec: ISpecification[T]) -> None:
        self._spec = spec

    def is_satisfied_by(self, entity: T) -> bool:
        return not self._spec.is_satisfied_by(entity)

    def to_query_params(self) -> Dict[str, Any]:
        return {"$not": self._spec.to_query_params()}


class TrueSpecification(ISpecification[T]):
    """Specification that always matches (useful as base for building)."""

    def is_satisfied_by(self, entity: T) -> bool:
        del entity  # Unused but required by interface
        return True

    def to_query_params(self) -> Dict[str, Any]:
        return {}


class FalseSpecification(ISpecification[T]):
    """Specification that never matches."""

    def is_satisfied_by(self, entity: T) -> bool:
        del entity  # Unused but required by interface
        return False

    def to_query_params(self) -> Dict[str, Any]:
        return {"$false": True}


# =============================================================================
# REPOSITORY INTERFACES - READ SIDE (ISP)
# =============================================================================


class IReadRepository(ABC, Generic[T, ID]):
    """
    Read-only repository interface following Interface Segregation Principle.

    Provides query operations without any mutation capabilities.
    Use this interface when code only needs to read data.
    """

    @abstractmethod
    async def get_by_id(self, id: ID) -> Optional[T]:
        """
        Retrieve entity by identifier.

        Returns None if not found.
        """
        pass

    @abstractmethod
    async def get_by_ids(self, ids: Iterable[ID]) -> List[T]:
        """
        Retrieve multiple entities by identifiers.

        Returns list in same order as input IDs. Missing entities are omitted.
        """
        pass

    @abstractmethod
    async def find(self, spec: ISpecification[T]) -> List[T]:
        """
        Find all entities matching a specification.

        Returns empty list if none match.
        """
        pass

    @abstractmethod
    async def find_one(self, spec: ISpecification[T]) -> Optional[T]:
        """
        Find first entity matching a specification.

        Returns None if none match.
        """
        pass

    @abstractmethod
    async def find_page(
        self,
        spec: ISpecification[T],
        page_request: PageRequest
    ) -> Page[T]:
        """
        Find entities matching a specification with pagination.
        """
        pass

    @abstractmethod
    async def count(self, spec: Optional[ISpecification[T]] = None) -> int:
        """
        Count entities matching a specification.

        If spec is None, counts all entities.
        """
        pass

    @abstractmethod
    async def exists(self, spec: ISpecification[T]) -> bool:
        """
        Check if any entity matches the specification.
        """
        pass

    @abstractmethod
    async def exists_by_id(self, id: ID) -> bool:
        """
        Check if an entity with the given ID exists.
        """
        pass


class IWriteRepository(ABC, Generic[T, ID]):
    """
    Write-only repository interface following Interface Segregation Principle.

    Provides mutation operations without any query capabilities.
    Use this interface when code only needs to write data.
    """

    @abstractmethod
    async def add(self, entity: T) -> T:
        """
        Add a new entity.

        Returns the added entity (potentially with generated ID).
        Raises if entity with same ID already exists.
        """
        pass

    @abstractmethod
    async def add_range(self, entities: Iterable[T]) -> List[T]:
        """
        Add multiple entities.

        Returns list of added entities.
        """
        pass

    @abstractmethod
    async def update(self, entity: T) -> T:
        """
        Update an existing entity.

        Returns the updated entity.
        Raises if entity does not exist.
        For versioned entities, checks version for optimistic concurrency.
        """
        pass

    @abstractmethod
    async def update_range(self, entities: Iterable[T]) -> List[T]:
        """
        Update multiple entities.

        Returns list of updated entities.
        """
        pass

    @abstractmethod
    async def remove(self, entity: T) -> None:
        """
        Remove an entity.

        Raises if entity does not exist.
        """
        pass

    @abstractmethod
    async def remove_by_id(self, id: ID) -> bool:
        """
        Remove entity by identifier.

        Returns True if removed, False if not found.
        """
        pass

    @abstractmethod
    async def remove_range(self, entities: Iterable[T]) -> int:
        """
        Remove multiple entities.

        Returns count of entities removed.
        """
        pass

    @abstractmethod
    async def remove_by_spec(self, spec: ISpecification[T]) -> int:
        """
        Remove all entities matching a specification.

        Returns count of entities removed.
        """
        pass


class IRepository(IReadRepository[T, ID], IWriteRepository[T, ID], ABC):
    """
    Full repository interface combining read and write operations.

    Use this interface when code needs both read and write capabilities.
    Prefer using IReadRepository or IWriteRepository when possible.
    """
    pass


# =============================================================================
# DOMAIN-SPECIFIC REPOSITORY INTERFACES
# =============================================================================


# Forward declare Verse for type hints (actual implementation in domain layer)
class Verse(Protocol):
    """Protocol for Verse entity."""
    @property
    def id(self) -> str: ...
    @property
    def reference(self) -> str: ...
    @property
    def book_code(self) -> str: ...
    @property
    def chapter(self) -> int: ...
    @property
    def verse_num(self) -> int: ...
    @property
    def text_original(self) -> Optional[str]: ...
    @property
    def text_english(self) -> Optional[str]: ...


class CrossReference(Protocol):
    """Protocol for CrossReference entity."""
    @property
    def id(self) -> str: ...
    @property
    def source_ref(self) -> str: ...
    @property
    def target_ref(self) -> str: ...
    @property
    def connection_type(self) -> str: ...
    @property
    def confidence(self) -> float: ...
    @property
    def strength(self) -> Optional[str]: ...


class ExtractionResult(Protocol):
    """Protocol for ExtractionResult entity."""
    @property
    def id(self) -> str: ...
    @property
    def verse_id(self) -> str: ...
    @property
    def agent_name(self) -> str: ...
    @property
    def extraction_type(self) -> str: ...
    @property
    def data(self) -> Dict[str, Any]: ...
    @property
    def confidence(self) -> float: ...


class PatristicCitation(Protocol):
    """Protocol for PatristicCitation entity."""
    @property
    def id(self) -> str: ...
    @property
    def verse_ref(self) -> str: ...
    @property
    def father_name(self) -> str: ...
    @property
    def work_title(self) -> str: ...
    @property
    def citation_text(self) -> str: ...


class IVerseRepository(IRepository[Verse, str], ABC):
    """
    Repository interface for Bible verse operations.

    Extends base repository with verse-specific query methods.
    """

    @abstractmethod
    async def get_by_reference(self, reference: str) -> Optional[Verse]:
        """Get verse by canonical reference (e.g., 'GEN.1.1')."""
        pass

    @abstractmethod
    async def get_by_book(self, book_code: str) -> List[Verse]:
        """Get all verses in a book."""
        pass

    @abstractmethod
    async def get_by_chapter(self, book_code: str, chapter: int) -> List[Verse]:
        """Get all verses in a chapter."""
        pass

    @abstractmethod
    async def get_range(self, start_ref: str, end_ref: str) -> List[Verse]:
        """Get verses in a range (inclusive)."""
        pass

    @abstractmethod
    async def search_text(self, query: str, limit: int = 10) -> List[Verse]:
        """Full-text search in verse content."""
        pass

    @abstractmethod
    async def get_with_embeddings(self, reference: str) -> Optional[Verse]:
        """Get verse with its embedding data loaded."""
        pass


class ICrossReferenceRepository(IRepository[CrossReference, str], ABC):
    """
    Repository interface for cross-reference operations.
    """

    @abstractmethod
    async def get_by_source(self, source_ref: str) -> List[CrossReference]:
        """Get all cross-references from a source verse."""
        pass

    @abstractmethod
    async def get_by_target(self, target_ref: str) -> List[CrossReference]:
        """Get all cross-references to a target verse."""
        pass

    @abstractmethod
    async def get_by_type(
        self,
        connection_type: str,
        page_request: Optional[PageRequest] = None
    ) -> Union[List[CrossReference], Page[CrossReference]]:
        """Get cross-references by connection type."""
        pass

    @abstractmethod
    async def get_bidirectional(self, verse_ref: str) -> List[CrossReference]:
        """Get all cross-references involving a verse (as source or target)."""
        pass

    @abstractmethod
    async def get_network(
        self,
        verse_ref: str,
        depth: int = 1,
        connection_types: Optional[List[str]] = None
    ) -> List[CrossReference]:
        """Get cross-reference network up to specified depth."""
        pass


class IExtractionResultRepository(IRepository[ExtractionResult, str], ABC):
    """
    Repository interface for extraction result storage.
    """

    @abstractmethod
    async def get_by_verse_and_agent(
        self,
        verse_id: str,
        agent_name: str
    ) -> Optional[ExtractionResult]:
        """Get extraction result for a verse from a specific agent."""
        pass

    @abstractmethod
    async def get_all_for_verse(self, verse_id: str) -> List[ExtractionResult]:
        """Get all extraction results for a verse."""
        pass

    @abstractmethod
    async def get_latest_by_agent(
        self,
        agent_name: str,
        limit: int = 100
    ) -> List[ExtractionResult]:
        """Get latest extraction results from an agent."""
        pass

    @abstractmethod
    async def get_by_extraction_type(
        self,
        extraction_type: str,
        page_request: Optional[PageRequest] = None
    ) -> Union[List[ExtractionResult], Page[ExtractionResult]]:
        """Get extraction results by type."""
        pass


class IPatristicRepository(IRepository[PatristicCitation, str], ABC):
    """
    Repository interface for patristic citation operations.
    """

    @abstractmethod
    async def get_by_verse(self, verse_ref: str) -> List[PatristicCitation]:
        """Get all patristic citations for a verse."""
        pass

    @abstractmethod
    async def get_by_father(
        self,
        father_name: str,
        page_request: Optional[PageRequest] = None
    ) -> Union[List[PatristicCitation], Page[PatristicCitation]]:
        """Get citations by a Church Father."""
        pass

    @abstractmethod
    async def search(self, query: str, limit: int = 10) -> List[PatristicCitation]:
        """Full-text search in patristic citations."""
        pass

    @abstractmethod
    async def get_fathers(self) -> List[str]:
        """Get list of all Church Fathers in the database."""
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
        self,
        verse_ref: str,
        properties: Dict[str, Any]
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
        self,
        source_ref: str,
        target_ref: str,
        max_depth: int = 5
    ) -> Optional[List[Dict[str, Any]]]:
        """Find shortest path between two verses."""
        pass

    @abstractmethod
    async def get_subgraph(
        self,
        verse_refs: List[str],
        include_relationships: bool = True
    ) -> Dict[str, Any]:
        """Get a subgraph containing the specified verses."""
        pass

    @abstractmethod
    async def get_centrality(
        self,
        verse_ref: str,
        algorithm: str = "degree"
    ) -> float:
        """Calculate centrality score for a verse."""
        pass

    @abstractmethod
    async def get_clusters(
        self,
        min_size: int = 3
    ) -> List[List[str]]:
        """Detect clusters/communities in the graph."""
        pass


# =============================================================================
# VECTOR STORE INTERFACE
# =============================================================================


@dataclass(frozen=True, slots=True)
class VectorSearchResult(Generic[T]):
    """Result from vector similarity search."""
    entity: T
    score: float
    distance: float


class IVectorStore(ABC, Generic[T, ID]):
    """
    Interface for vector similarity operations.

    Used for embedding-based cross-reference discovery.
    """

    @abstractmethod
    async def add_embedding(
        self,
        id: ID,
        embedding: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        """Add an embedding with metadata."""
        pass

    @abstractmethod
    async def batch_add_embeddings(
        self,
        items: List[Tuple[ID, List[float], Dict[str, Any]]],
    ) -> int:
        """Batch add embeddings. Returns count added."""
        pass

    @abstractmethod
    async def search_similar(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0,
    ) -> List[VectorSearchResult[T]]:
        """Search for similar embeddings."""
        pass

    @abstractmethod
    async def search_by_text(
        self,
        text: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult[T]]:
        """Search by text (embedding computed internally)."""
        pass

    @abstractmethod
    async def get_embedding(self, id: ID) -> Optional[List[float]]:
        """Get embedding for an ID."""
        pass

    @abstractmethod
    async def delete(self, id: ID) -> bool:
        """Delete an embedding by ID."""
        pass

    @abstractmethod
    async def delete_by_filter(self, filters: Dict[str, Any]) -> int:
        """Delete embeddings matching filter. Returns count deleted."""
        pass


# =============================================================================
# UNIT OF WORK INTERFACE WITH DOMAIN EVENT DISPATCH
# =============================================================================


class IDomainEventDispatcher(ABC):
    """
    Interface for dispatching domain events.
    """

    @abstractmethod
    async def dispatch(self, event: IDomainEvent) -> None:
        """Dispatch a single event."""
        pass

    @abstractmethod
    async def dispatch_all(self, events: Iterable[IDomainEvent]) -> None:
        """Dispatch multiple events."""
        pass


class IUnitOfWork(ABC):
    """
    Unit of Work pattern for managing transactions with domain event dispatch.

    Ensures atomic operations across multiple repositories and handles
    domain event collection and dispatch on commit.

    Usage:
        async with uow:
            verse = await uow.verses.get_by_reference("GEN.1.1")
            verse.add_domain_event(VerseUpdated(verse.id))
            await uow.verses.update(verse)
            await uow.commit()  # Events dispatched after successful commit
    """

    # Repository properties (implemented by concrete UoW)
    @property
    @abstractmethod
    def verses(self) -> IVerseRepository:
        """Verse repository."""
        pass

    @property
    @abstractmethod
    def cross_references(self) -> ICrossReferenceRepository:
        """Cross-reference repository."""
        pass

    @property
    @abstractmethod
    def extraction_results(self) -> IExtractionResultRepository:
        """Extraction result repository."""
        pass

    @property
    @abstractmethod
    def patristic(self) -> IPatristicRepository:
        """Patristic citation repository."""
        pass

    @abstractmethod
    async def __aenter__(self) -> "IUnitOfWork":
        """Enter the unit of work context."""
        pass

    @abstractmethod
    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any]
    ) -> None:
        """Exit the unit of work context."""
        pass

    @abstractmethod
    async def commit(self) -> None:
        """
        Commit all changes in this unit of work.

        After successful commit, dispatches all collected domain events.
        """
        pass

    @abstractmethod
    async def rollback(self) -> None:
        """
        Rollback all changes in this unit of work.

        Clears any collected domain events without dispatching.
        """
        pass

    @abstractmethod
    def register_domain_event(self, event: IDomainEvent) -> None:
        """
        Register a domain event to be dispatched on commit.

        Events are dispatched in order of registration after successful commit.
        """
        pass

    @abstractmethod
    def get_pending_events(self) -> List[IDomainEvent]:
        """Get all pending domain events."""
        pass

    @abstractmethod
    def track_aggregate(self, aggregate: IAggregateRoot) -> None:
        """
        Track an aggregate root for domain event collection.

        On commit, events from tracked aggregates are collected and dispatched.
        """
        pass


# =============================================================================
# OUTBOX PATTERN FOR RELIABLE EVENT PUBLISHING
# =============================================================================


class OutboxMessageStatus(Enum):
    """Status of an outbox message."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class OutboxMessage:
    """
    Outbox message for reliable event publishing.

    Ensures at-least-once delivery of domain events to external systems.
    """
    id: UUID
    event_type: str
    payload: Dict[str, Any]
    status: OutboxMessageStatus
    created_at: datetime
    processed_at: Optional[datetime] = None
    retry_count: int = 0
    error_message: Optional[str] = None

    @classmethod
    def from_event(cls, event: IDomainEvent) -> "OutboxMessage":
        """Create outbox message from domain event."""
        return cls(
            id=uuid4(),
            event_type=event.event_type,
            payload={
                "event_id": str(event.event_id),
                "occurred_at": event.occurred_at.isoformat(),
                "aggregate_id": event.aggregate_id,
                # Additional event data should be added by concrete implementation
            },
            status=OutboxMessageStatus.PENDING,
            created_at=datetime.now(timezone.utc)
        )


class IOutboxRepository(ABC):
    """
    Repository for outbox pattern message storage.
    """

    @abstractmethod
    async def add(self, message: OutboxMessage) -> None:
        """Add a message to the outbox."""
        pass

    @abstractmethod
    async def get_pending(self, limit: int = 100) -> List[OutboxMessage]:
        """Get pending messages for processing."""
        pass

    @abstractmethod
    async def mark_processing(self, message_id: UUID) -> bool:
        """Mark message as being processed. Returns False if already processing."""
        pass

    @abstractmethod
    async def mark_completed(self, message_id: UUID) -> None:
        """Mark message as successfully processed."""
        pass

    @abstractmethod
    async def mark_failed(
        self,
        message_id: UUID,
        error: str,
        retry: bool = True
    ) -> None:
        """Mark message as failed, optionally allowing retry."""
        pass

    @abstractmethod
    async def delete_completed(self, older_than: datetime) -> int:
        """Delete completed messages older than specified time."""
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

    async def health_check(self) -> "HealthCheckResult":
        """Perform health check."""
        ...


@dataclass(frozen=True, slots=True)
class HealthCheckResult:
    """Result of a health check."""
    healthy: bool
    message: str
    component: str
    latency_ms: float
    details: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def healthy_result(
        cls,
        component: str,
        latency_ms: float,
        message: str = "OK"
    ) -> "HealthCheckResult":
        return cls(
            healthy=True,
            message=message,
            component=component,
            latency_ms=latency_ms
        )

    @classmethod
    def unhealthy_result(
        cls,
        component: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> "HealthCheckResult":
        return cls(
            healthy=False,
            message=message,
            component=component,
            latency_ms=0.0,
            details=details or {}
        )


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
    async def health_check(self) -> HealthCheckResult:
        """
        Perform health check on database connection.

        Returns structured health check result.
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
        yield  # type: ignore

    @abstractmethod
    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[Any, None]:
        """
        Get an explicit transaction context.

        Unlike session(), does not auto-commit - must call commit explicitly.
        """
        yield  # type: ignore

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if client is initialized and ready."""
        pass


# =============================================================================
# CONCURRENCY CONTROL
# =============================================================================


class ConcurrencyException(Exception):
    """Raised when optimistic concurrency check fails."""

    def __init__(
        self,
        entity_type: str,
        entity_id: Any,
        expected_version: int,
        actual_version: int
    ):
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.expected_version = expected_version
        self.actual_version = actual_version
        super().__init__(
            f"Concurrency conflict: {entity_type} {entity_id} "
            f"expected version {expected_version}, actual {actual_version}"
        )


class EntityNotFoundException(Exception):
    """Raised when an entity is not found."""

    def __init__(self, entity_type: str, entity_id: Any):
        self.entity_type = entity_type
        self.entity_id = entity_id
        super().__init__(f"{entity_type} not found: {entity_id}")


class DuplicateEntityException(Exception):
    """Raised when trying to add an entity that already exists."""

    def __init__(self, entity_type: str, entity_id: Any):
        self.entity_type = entity_type
        self.entity_id = entity_id
        super().__init__(f"{entity_type} already exists: {entity_id}")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Domain Primitives
    "IEntity",
    "IVersionedEntity",
    "IAggregateRoot",
    "IDomainEvent",

    # Value Objects
    "VerseId",
    "ConnectionType",
    "Confidence",

    # Pagination
    "Page",
    "PageRequest",

    # Specification Pattern
    "ISpecification",
    "AndSpecification",
    "OrSpecification",
    "NotSpecification",
    "TrueSpecification",
    "FalseSpecification",

    # Repository Interfaces (ISP)
    "IReadRepository",
    "IWriteRepository",
    "IRepository",

    # Domain Repositories
    "Verse",
    "CrossReference",
    "ExtractionResult",
    "PatristicCitation",
    "IVerseRepository",
    "ICrossReferenceRepository",
    "IExtractionResultRepository",
    "IPatristicRepository",

    # Graph & Vector
    "IGraphRepository",
    "IVectorStore",
    "VectorSearchResult",

    # Unit of Work
    "IUnitOfWork",
    "IDomainEventDispatcher",

    # Outbox Pattern
    "OutboxMessage",
    "OutboxMessageStatus",
    "IOutboxRepository",

    # Protocols
    "Connectable",
    "Initializable",
    "HealthCheckable",
    "HealthCheckResult",

    # Database Client
    "IDatabaseClient",

    # Exceptions
    "ConcurrencyException",
    "EntityNotFoundException",
    "DuplicateEntityException",
]
