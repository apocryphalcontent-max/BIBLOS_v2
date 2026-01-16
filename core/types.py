"""
BIBLOS v2 - Centralized Type Definitions

Provides type aliases, TypedDicts, and Protocol classes for
consistent typing throughout the system.

Usage:
    from core.types import VerseId, ConnectionType, AgentResult

    def process_verse(verse_id: VerseId) -> AgentResult:
        ...
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Protocol,
    TypedDict,
    TypeVar,
    Union,
    runtime_checkable,
)

# =============================================================================
# TYPE ALIASES - Simple type shortcuts
# =============================================================================

# Verse identification
VerseId = str  # Format: "BOOK.CHAPTER.VERSE" (e.g., "GEN.1.1")
BookCode = str  # 3-letter code (e.g., "GEN", "MAT")
WordId = str  # Format: "BOOK.CHAPTER.VERSE.WORD" (e.g., "GEN.1.1.1")

# Cross-reference types
ConnectionTypeLiteral = Literal[
    "thematic", "verbal", "conceptual", "historical",
    "typological", "prophetic", "liturgical", "narrative",
    "genealogical", "geographical"
]
StrengthLiteral = Literal["strong", "moderate", "weak"]
StatusLiteral = Literal["pending", "in_progress", "completed", "failed", "skipped"]
CertificationLiteral = Literal["gold", "silver", "bronze", "provisional"]

# Processing types
Confidence = float  # 0.0 to 1.0
MillisecondsDuration = float
SecondsDuration = float

# Generic callable types
T = TypeVar("T")
R = TypeVar("R")
AsyncCallable = Callable[..., Awaitable[T]]


# =============================================================================
# TYPED DICTS - Structured dictionaries with type hints
# =============================================================================


class VerseDict(TypedDict, total=False):
    """Dictionary representation of a verse."""
    verse_id: str
    book: str
    book_name: str
    chapter: int
    verse: int
    text: str
    original_text: str
    testament: Literal["OT", "NT"]
    language: Literal["hebrew", "aramaic", "greek"]
    metadata: Dict[str, Any]


class MorphologyDict(TypedDict, total=False):
    """Dictionary representation of morphological analysis."""
    part_of_speech: str
    person: Optional[str]
    number: Optional[str]
    gender: Optional[str]
    case: Optional[str]
    tense: Optional[str]
    voice: Optional[str]
    mood: Optional[str]
    stem: Optional[str]
    raw_code: str


class WordDict(TypedDict, total=False):
    """Dictionary representation of a word."""
    word_id: str
    verse_id: str
    surface_form: str
    lemma: str
    position: int
    language: str
    morphology: MorphologyDict
    transliteration: str
    gloss: str
    strongs: str


class CrossReferenceDict(TypedDict, total=False):
    """Dictionary representation of a cross-reference."""
    source_ref: str
    target_ref: str
    connection_type: ConnectionTypeLiteral
    strength: StrengthLiteral
    confidence: float
    bidirectional: bool
    notes: List[str]
    sources: List[str]
    verified: bool
    patristic_support: bool
    metadata: Dict[str, Any]


class ExtractionResultDict(TypedDict, total=False):
    """Dictionary representation of an extraction result."""
    agent_name: str
    extraction_type: str
    verse_id: str
    status: StatusLiteral
    confidence: float
    data: Dict[str, Any]
    error: Optional[str]
    processing_time: float


class GoldenRecordDict(TypedDict, total=False):
    """Dictionary representation of a golden record."""
    verse_id: str
    text: str
    certification: Dict[str, Any]
    data: Dict[str, Any]
    phases_executed: List[str]
    agent_count: int
    total_processing_time: float


class InferenceCandidateDict(TypedDict, total=False):
    """Dictionary representation of an inference candidate."""
    source_verse: str
    target_verse: str
    connection_type: ConnectionTypeLiteral
    confidence: float
    embedding_similarity: float
    semantic_similarity: float
    features: Dict[str, float]
    evidence: List[str]


# =============================================================================
# PROTOCOLS - Interface definitions for duck typing
# =============================================================================


@runtime_checkable
class Validatable(Protocol):
    """Protocol for objects that can be validated."""

    def validate(self) -> List[str]:
        """Return list of validation errors (empty if valid)."""
        ...


@runtime_checkable
class Serializable(Protocol):
    """Protocol for objects that can be serialized."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        ...

    def to_json(self) -> str:
        """Convert to JSON string."""
        ...


@runtime_checkable
class Identifiable(Protocol):
    """Protocol for objects with an ID."""

    @property
    def id(self) -> str:
        """Return unique identifier."""
        ...


@runtime_checkable
class AsyncContextManager(Protocol[T]):
    """Protocol for async context managers."""

    async def __aenter__(self) -> T:
        ...

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> Optional[bool]:
        ...


@runtime_checkable
class ExtractionAgent(Protocol):
    """Protocol for extraction agents."""

    @property
    def name(self) -> str:
        """Agent name."""
        ...

    @property
    def extraction_type(self) -> str:
        """Type of extraction performed."""
        ...

    async def extract(
        self,
        verse_id: str,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExtractionResultDict:
        """Perform extraction on a verse."""
        ...

    def validate(self, result: ExtractionResultDict) -> bool:
        """Validate extraction result."""
        ...


@runtime_checkable
class DatabaseClient(Protocol):
    """Protocol for database clients."""

    async def connect(self) -> None:
        """Establish connection."""
        ...

    async def disconnect(self) -> None:
        """Close connection."""
        ...

    async def is_connected(self) -> bool:
        """Check if connected."""
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Protocol for vector stores."""

    async def add_vectors(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add vectors to the store."""
        ...

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        ...


@runtime_checkable
class EmbeddingModel(Protocol):
    """Protocol for embedding models."""

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        ...

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        ...


@runtime_checkable
class PipelinePhase(Protocol):
    """Protocol for pipeline phases."""

    @property
    def name(self) -> str:
        """Phase name."""
        ...

    async def execute(
        self,
        verse_id: str,
        text: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute the phase."""
        ...


@runtime_checkable
class MetricsRecorder(Protocol):
    """Protocol for metrics recording."""

    def record_counter(
        self,
        name: str,
        value: int,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record counter metric."""
        ...

    def record_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record histogram metric."""
        ...


@runtime_checkable
class Logger(Protocol):
    """Protocol for loggers."""

    def debug(self, msg: str, **kwargs: Any) -> None:
        ...

    def info(self, msg: str, **kwargs: Any) -> None:
        ...

    def warning(self, msg: str, **kwargs: Any) -> None:
        ...

    def error(self, msg: str, **kwargs: Any) -> None:
        ...


# =============================================================================
# RESULT TYPES - Explicit success/error handling
# =============================================================================


class Result(Generic[T]):
    """
    Explicit success/error result type.

    Usage:
        result = await process_verse(verse_id)
        if result.is_success:
            print(result.value)
        else:
            print(result.error)
    """

    def __init__(
        self,
        value: Optional[T] = None,
        error: Optional[str] = None,
        exception: Optional[Exception] = None,
    ):
        self._value = value
        self._error = error
        self._exception = exception

    @property
    def is_success(self) -> bool:
        return self._error is None and self._exception is None

    @property
    def is_failure(self) -> bool:
        return not self.is_success

    @property
    def value(self) -> T:
        if self.is_failure:
            raise ValueError(f"Cannot get value from failed result: {self._error}")
        return self._value  # type: ignore

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def exception(self) -> Optional[Exception]:
        return self._exception

    def unwrap(self) -> T:
        """Get value or raise exception."""
        if self._exception:
            raise self._exception
        if self._error:
            raise ValueError(self._error)
        return self._value  # type: ignore

    def unwrap_or(self, default: T) -> T:
        """Get value or return default."""
        if self.is_failure:
            return default
        return self._value  # type: ignore

    def map(self, fn: Callable[[T], R]) -> "Result[R]":
        """Transform value if successful."""
        if self.is_failure:
            return Result(error=self._error, exception=self._exception)
        return Result(value=fn(self._value))  # type: ignore

    @classmethod
    def success(cls, value: T) -> "Result[T]":
        """Create successful result."""
        return cls(value=value)

    @classmethod
    def failure(cls, error: str) -> "Result[T]":
        """Create failed result with error message."""
        return cls(error=error)

    @classmethod
    def from_exception(cls, exception: Exception) -> "Result[T]":
        """Create failed result from exception."""
        return cls(error=str(exception), exception=exception)


# =============================================================================
# TYPE GUARDS - Runtime type checking helpers
# =============================================================================


def is_verse_id(value: Any) -> bool:
    """Check if value is a valid verse ID format."""
    if not isinstance(value, str):
        return False
    parts = value.split(".")
    if len(parts) < 3:
        return False
    try:
        int(parts[1])
        int(parts[2])
        return True
    except (ValueError, IndexError):
        return False


def is_connection_type(value: Any) -> bool:
    """Check if value is a valid connection type."""
    valid = {
        "thematic", "verbal", "conceptual", "historical",
        "typological", "prophetic", "liturgical", "narrative",
        "genealogical", "geographical"
    }
    return isinstance(value, str) and value in valid


def is_strength(value: Any) -> bool:
    """Check if value is a valid strength."""
    return isinstance(value, str) and value in {"strong", "moderate", "weak"}


def is_confidence(value: Any) -> bool:
    """Check if value is a valid confidence score."""
    return isinstance(value, (int, float)) and 0.0 <= value <= 1.0


# =============================================================================
# SENTINEL TYPES - Special marker values
# =============================================================================


class _Missing:
    """Sentinel for missing values (distinct from None)."""

    _instance: Optional["_Missing"] = None

    def __new__(cls) -> "_Missing":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "MISSING"

    def __bool__(self) -> bool:
        return False


MISSING = _Missing()


class _Unset:
    """Sentinel for unset configuration values."""

    _instance: Optional["_Unset"] = None

    def __new__(cls) -> "_Unset":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNSET"

    def __bool__(self) -> bool:
        return False


UNSET = _Unset()
