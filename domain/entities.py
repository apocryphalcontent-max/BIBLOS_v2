"""
BIBLOS v2 - Domain Entities

Rich domain model implementing aggregate roots, entities, value objects,
and domain events for biblical scholarship data.

This module embodies the living heart of the system - each aggregate
maintains its invariants, emits events on state changes, and encapsulates
the complex rules of biblical cross-reference analysis.

Design Principles:
    - Aggregates are consistency boundaries
    - Value objects are immutable and self-validating
    - Domain events capture all significant state changes
    - Business rules are encoded in the domain, not services
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    ClassVar,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
)
from uuid import UUID, uuid4


# =============================================================================
# VALUE OBJECTS - Immutable, self-validating domain primitives
# =============================================================================


@dataclass(frozen=True, slots=True)
class VerseReference:
    """
    Value object for verse references.

    Immutable, self-validating reference following canonical format.
    The verse reference is the fundamental identifier in biblical scholarship.

    Format: BOOK.CHAPTER.VERSE (e.g., GEN.1.1, JHN.3.16)
    """
    book: str
    chapter: int
    verse: int

    # Valid book codes (3-letter canonical codes)
    VALID_BOOKS: ClassVar[FrozenSet[str]] = frozenset({
        # Old Testament (39 books)
        "GEN", "EXO", "LEV", "NUM", "DEU", "JOS", "JDG", "RUT", "1SA", "2SA",
        "1KI", "2KI", "1CH", "2CH", "EZR", "NEH", "EST", "JOB", "PSA", "PRO",
        "ECC", "SNG", "ISA", "JER", "LAM", "EZK", "DAN", "HOS", "JOL", "AMO",
        "OBA", "JON", "MIC", "NAM", "HAB", "ZEP", "HAG", "ZEC", "MAL",
        # New Testament (27 books)
        "MAT", "MRK", "LUK", "JHN", "ACT", "ROM", "1CO", "2CO", "GAL", "EPH",
        "PHP", "COL", "1TH", "2TH", "1TI", "2TI", "TIT", "PHM", "HEB", "JAS",
        "1PE", "2PE", "1JN", "2JN", "3JN", "JUD", "REV",
        # Deuterocanonical (Orthodox Canon)
        "TOB", "JDT", "WIS", "SIR", "BAR", "1MA", "2MA", "3MA", "4MA",
        "1ES", "2ES", "PRM", "PSS", "ODE", "SUS", "BEL", "LJE",
    })

    OT_BOOKS: ClassVar[FrozenSet[str]] = frozenset({
        "GEN", "EXO", "LEV", "NUM", "DEU", "JOS", "JDG", "RUT", "1SA", "2SA",
        "1KI", "2KI", "1CH", "2CH", "EZR", "NEH", "EST", "JOB", "PSA", "PRO",
        "ECC", "SNG", "ISA", "JER", "LAM", "EZK", "DAN", "HOS", "JOL", "AMO",
        "OBA", "JON", "MIC", "NAM", "HAB", "ZEP", "HAG", "ZEC", "MAL",
        # Deuterocanonical
        "TOB", "JDT", "WIS", "SIR", "BAR", "1MA", "2MA", "3MA", "4MA",
        "1ES", "2ES", "PRM", "PSS", "ODE", "SUS", "BEL", "LJE",
    })

    NT_BOOKS: ClassVar[FrozenSet[str]] = frozenset({
        "MAT", "MRK", "LUK", "JHN", "ACT", "ROM", "1CO", "2CO", "GAL", "EPH",
        "PHP", "COL", "1TH", "2TH", "1TI", "2TI", "TIT", "PHM", "HEB", "JAS",
        "1PE", "2PE", "1JN", "2JN", "3JN", "JUD", "REV",
    })

    def __post_init__(self) -> None:
        """Validate the verse reference."""
        if self.book.upper() not in self.VALID_BOOKS:
            raise ValueError(f"Invalid book code: {self.book}")
        if self.chapter < 1:
            raise ValueError(f"Chapter must be >= 1: {self.chapter}")
        if self.verse < 1:
            raise ValueError(f"Verse must be >= 1: {self.verse}")
        # Normalize book to uppercase (frozen dataclass trick)
        object.__setattr__(self, "book", self.book.upper())

    def __str__(self) -> str:
        return f"{self.book}.{self.chapter}.{self.verse}"

    def __hash__(self) -> int:
        return hash((self.book, self.chapter, self.verse))

    @classmethod
    def parse(cls, reference: str) -> "VerseReference":
        """Parse a verse reference string."""
        normalized = reference.upper().replace(":", ".").replace(" ", ".")
        parts = normalized.split(".")
        if len(parts) < 3:
            raise ValueError(f"Invalid verse reference format: {reference}")
        return cls(
            book=parts[0],
            chapter=int(parts[1]),
            verse=int(parts[2])
        )

    @property
    def testament(self) -> str:
        """Return 'OT' or 'NT' based on book."""
        return "OT" if self.book in self.OT_BOOKS else "NT"

    @property
    def is_old_testament(self) -> bool:
        return self.book in self.OT_BOOKS

    @property
    def is_new_testament(self) -> bool:
        return self.book in self.NT_BOOKS

    def to_tuple(self) -> Tuple[str, int, int]:
        """Convert to tuple for sorting."""
        return (self.book, self.chapter, self.verse)

    def chapter_reference(self) -> str:
        """Get chapter-level reference (e.g., GEN.1)."""
        return f"{self.book}.{self.chapter}"

    def book_reference(self) -> str:
        """Get book-level reference (e.g., GEN)."""
        return self.book


class ConnectionTypeEnum(str, Enum):
    """Connection types for cross-references."""
    TYPOLOGICAL = "typological"
    PROPHETIC = "prophetic"
    VERBAL = "verbal"
    THEMATIC = "thematic"
    CONCEPTUAL = "conceptual"
    HISTORICAL = "historical"
    LITURGICAL = "liturgical"
    NARRATIVE = "narrative"
    GENEALOGICAL = "genealogical"
    GEOGRAPHICAL = "geographical"


@dataclass(frozen=True, slots=True)
class ConnectionStrength:
    """
    Value object for connection strength with semantic meaning.

    Represents how strongly two verses are connected:
    - STRONG: Clear, direct connection (explicit quotation, prophecy fulfillment)
    - MODERATE: Recognizable thematic or verbal parallel
    - WEAK: Subtle connection requiring interpretation
    """
    STRONG: ClassVar[str] = "strong"
    MODERATE: ClassVar[str] = "moderate"
    WEAK: ClassVar[str] = "weak"

    value: str

    _VALID_STRENGTHS: ClassVar[FrozenSet[str]] = frozenset({"strong", "moderate", "weak"})
    _WEIGHTS: ClassVar[Dict[str, float]] = {
        "strong": 1.0,
        "moderate": 0.7,
        "weak": 0.4,
    }

    def __post_init__(self) -> None:
        if self.value not in self._VALID_STRENGTHS:
            raise ValueError(f"Invalid strength: {self.value}")

    def __str__(self) -> str:
        return self.value

    @property
    def weight(self) -> float:
        """Numeric weight for calculations."""
        return self._WEIGHTS[self.value]

    @classmethod
    def strong(cls) -> "ConnectionStrength":
        return cls(cls.STRONG)

    @classmethod
    def moderate(cls) -> "ConnectionStrength":
        return cls(cls.MODERATE)

    @classmethod
    def weak(cls) -> "ConnectionStrength":
        return cls(cls.WEAK)

    def is_stronger_than(self, other: "ConnectionStrength") -> bool:
        """Compare connection strengths."""
        return self.weight > other.weight


@dataclass(frozen=True, slots=True)
class ConfidenceScore:
    """
    Value object for confidence scores with validation and semantics.

    Confidence represents the ML/analytical certainty of a determination.
    """
    value: float

    # Thresholds for classification
    HIGH_THRESHOLD: ClassVar[float] = 0.85
    MEDIUM_THRESHOLD: ClassVar[float] = 0.65
    LOW_THRESHOLD: ClassVar[float] = 0.45

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"Confidence must be 0-1: {self.value}")

    def __float__(self) -> float:
        return self.value

    def __str__(self) -> str:
        return f"{self.value:.2%}"

    @property
    def is_high(self) -> bool:
        return self.value >= self.HIGH_THRESHOLD

    @property
    def is_medium(self) -> bool:
        return self.MEDIUM_THRESHOLD <= self.value < self.HIGH_THRESHOLD

    @property
    def is_low(self) -> bool:
        return self.LOW_THRESHOLD <= self.value < self.MEDIUM_THRESHOLD

    @property
    def is_very_low(self) -> bool:
        return self.value < self.LOW_THRESHOLD

    @property
    def tier(self) -> str:
        """Return human-readable tier."""
        if self.is_high:
            return "high"
        if self.is_medium:
            return "medium"
        if self.is_low:
            return "low"
        return "very_low"

    @classmethod
    def high(cls) -> "ConfidenceScore":
        return cls(0.9)

    @classmethod
    def medium(cls) -> "ConfidenceScore":
        return cls(0.7)

    @classmethod
    def low(cls) -> "ConfidenceScore":
        return cls(0.5)

    @classmethod
    def zero(cls) -> "ConfidenceScore":
        return cls(0.0)

    def combine_with(self, other: "ConfidenceScore", weight: float = 0.5) -> "ConfidenceScore":
        """Combine two confidence scores with weighted average."""
        combined = (self.value * weight) + (other.value * (1 - weight))
        return ConfidenceScore(min(1.0, max(0.0, combined)))


class QualityTier(str, Enum):
    """Quality certification tiers for golden records."""
    GOLD = "gold"
    SILVER = "silver"
    BRONZE = "bronze"
    PROVISIONAL = "provisional"
    REJECTED = "rejected"

    @property
    def numeric_value(self) -> int:
        """Numeric value for comparison."""
        return {
            self.GOLD: 4,
            self.SILVER: 3,
            self.BRONZE: 2,
            self.PROVISIONAL: 1,
            self.REJECTED: 0,
        }[self]

    def is_certified(self) -> bool:
        """Check if this tier represents certification."""
        return self in {self.GOLD, self.SILVER, self.BRONZE}


class ExtractionType(str, Enum):
    """Types of extraction performed by agents."""
    STRUCTURAL = "structural"
    MORPHOLOGICAL = "morphological"
    SYNTACTIC = "syntactic"
    SEMANTIC = "semantic"
    LEXICAL = "lexical"
    ETYMOLOGICAL = "etymological"
    PHONOLOGICAL = "phonological"
    THEOLOGICAL = "theological"
    TYPOLOGICAL = "typological"
    PATRISTIC = "patristic"
    LITURGICAL = "liturgical"
    INTERTEXTUAL = "intertextual"
    CROSS_REFERENCE = "cross_reference"


class InterpretationType(str, Enum):
    """Patristic interpretation types (fourfold sense)."""
    LITERAL = "literal"
    ALLEGORICAL = "allegorical"
    TROPOLOGICAL = "tropological"  # Moral sense
    ANAGOGICAL = "anagogical"  # Eschatological sense


# =============================================================================
# DOMAIN EVENTS - Signals of significant state changes
# =============================================================================


@dataclass(frozen=True)
class DomainEvent:
    """
    Base class for domain events.

    Domain events are immutable records of significant occurrences
    in the domain. They propagate through the system like neural signals,
    triggering reactions in other parts of the organism.
    """
    event_id: UUID = field(default_factory=uuid4)
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    aggregate_id: Optional[str] = None

    @property
    def event_type(self) -> str:
        """Return the event type name."""
        return self.__class__.__name__

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_id": str(self.event_id),
            "event_type": self.event_type,
            "occurred_at": self.occurred_at.isoformat(),
            "aggregate_id": self.aggregate_id,
            "data": self._event_data(),
        }

    def _event_data(self) -> Dict[str, Any]:
        """Override in subclasses to provide event-specific data."""
        return {}


# Verse Events
@dataclass(frozen=True)
class VerseCreated(DomainEvent):
    """Emitted when a new verse is created in the system."""
    reference: str = ""
    book: str = ""
    chapter: int = 0
    verse: int = 0
    testament: str = ""

    def _event_data(self) -> Dict[str, Any]:
        return {
            "reference": self.reference,
            "book": self.book,
            "chapter": self.chapter,
            "verse": self.verse,
            "testament": self.testament,
        }


@dataclass(frozen=True)
class VerseTextUpdated(DomainEvent):
    """Emitted when verse text is modified."""
    reference: str = ""
    field_updated: str = ""
    old_hash: str = ""
    new_hash: str = ""

    def _event_data(self) -> Dict[str, Any]:
        return {
            "reference": self.reference,
            "field_updated": self.field_updated,
            "old_hash": self.old_hash,
            "new_hash": self.new_hash,
        }


@dataclass(frozen=True)
class VerseProcessingStarted(DomainEvent):
    """Emitted when verse processing begins."""
    reference: str = ""
    pipeline_id: str = ""

    def _event_data(self) -> Dict[str, Any]:
        return {
            "reference": self.reference,
            "pipeline_id": self.pipeline_id,
        }


@dataclass(frozen=True)
class VerseProcessingCompleted(DomainEvent):
    """Emitted when verse processing completes."""
    reference: str = ""
    pipeline_id: str = ""
    success: bool = True
    agents_executed: int = 0
    duration_ms: float = 0.0

    def _event_data(self) -> Dict[str, Any]:
        return {
            "reference": self.reference,
            "pipeline_id": self.pipeline_id,
            "success": self.success,
            "agents_executed": self.agents_executed,
            "duration_ms": self.duration_ms,
        }


# Cross-Reference Events
@dataclass(frozen=True)
class CrossReferenceCreated(DomainEvent):
    """Emitted when a cross-reference is created."""
    source_ref: str = ""
    target_ref: str = ""
    connection_type: str = ""
    strength: str = ""
    confidence: float = 0.0

    def _event_data(self) -> Dict[str, Any]:
        return {
            "source_ref": self.source_ref,
            "target_ref": self.target_ref,
            "connection_type": self.connection_type,
            "strength": self.strength,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class CrossReferenceStrengthUpdated(DomainEvent):
    """Emitted when cross-reference strength is modified."""
    crossref_id: str = ""
    old_strength: str = ""
    new_strength: str = ""
    reason: str = ""

    def _event_data(self) -> Dict[str, Any]:
        return {
            "crossref_id": self.crossref_id,
            "old_strength": self.old_strength,
            "new_strength": self.new_strength,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CrossReferenceVerified(DomainEvent):
    """Emitted when a cross-reference is verified (human or patristic)."""
    crossref_id: str = ""
    verification_type: str = ""  # "human", "patristic", "ml"
    verifier: str = ""

    def _event_data(self) -> Dict[str, Any]:
        return {
            "crossref_id": self.crossref_id,
            "verification_type": self.verification_type,
            "verifier": self.verifier,
        }


@dataclass(frozen=True)
class CrossReferenceDiscovered(DomainEvent):
    """Emitted when ML inference discovers a new cross-reference candidate."""
    source_ref: str = ""
    target_ref: str = ""
    connection_type: str = ""
    confidence: float = 0.0
    model_name: str = ""
    features: Dict[str, float] = field(default_factory=dict)

    def _event_data(self) -> Dict[str, Any]:
        return {
            "source_ref": self.source_ref,
            "target_ref": self.target_ref,
            "connection_type": self.connection_type,
            "confidence": self.confidence,
            "model_name": self.model_name,
            "features": self.features,
        }


# Extraction Events
@dataclass(frozen=True)
class ExtractionResultCreated(DomainEvent):
    """Emitted when an agent produces an extraction result."""
    verse_id: str = ""
    agent_name: str = ""
    extraction_type: str = ""
    confidence: float = 0.0

    def _event_data(self) -> Dict[str, Any]:
        return {
            "verse_id": self.verse_id,
            "agent_name": self.agent_name,
            "extraction_type": self.extraction_type,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class ExtractionResultUpdated(DomainEvent):
    """Emitted when an extraction result is updated."""
    extraction_id: str = ""
    agent_name: str = ""
    old_confidence: float = 0.0
    new_confidence: float = 0.0

    def _event_data(self) -> Dict[str, Any]:
        return {
            "extraction_id": self.extraction_id,
            "agent_name": self.agent_name,
            "old_confidence": self.old_confidence,
            "new_confidence": self.new_confidence,
        }


# Golden Record Events
@dataclass(frozen=True)
class GoldenRecordCertified(DomainEvent):
    """Emitted when a golden record achieves certification."""
    verse_id: str = ""
    quality_tier: str = ""
    score: float = 0.0
    agents_contributing: int = 0

    def _event_data(self) -> Dict[str, Any]:
        return {
            "verse_id": self.verse_id,
            "quality_tier": self.quality_tier,
            "score": self.score,
            "agents_contributing": self.agents_contributing,
        }


@dataclass(frozen=True)
class GoldenRecordDecertified(DomainEvent):
    """Emitted when a golden record loses certification."""
    verse_id: str = ""
    previous_tier: str = ""
    reason: str = ""

    def _event_data(self) -> Dict[str, Any]:
        return {
            "verse_id": self.verse_id,
            "previous_tier": self.previous_tier,
            "reason": self.reason,
        }


# Patristic Events
@dataclass(frozen=True)
class PatristicCitationLinked(DomainEvent):
    """Emitted when a patristic citation is linked to a verse."""
    verse_ref: str = ""
    father: str = ""
    work: str = ""
    interpretation_type: str = ""

    def _event_data(self) -> Dict[str, Any]:
        return {
            "verse_ref": self.verse_ref,
            "father": self.father,
            "work": self.work,
            "interpretation_type": self.interpretation_type,
        }


# =============================================================================
# BASE AGGREGATE AND ENTITY CLASSES
# =============================================================================
#
# Seraphic Architecture Principle:
#     These base classes embody the organism's nature at the cellular level.
#     Each entity doesn't need to be told how to participate in the whole -
#     it KNOWS how because that knowledge is inherent in its structure.
#
#     Like a seraph's wing that IS light, not merely carries it, these
#     entities ARE aspects of the organism, not merely parts of it.
#
# =============================================================================


class Entity(ABC):
    """
    Base class for domain entities.

    Entities have identity that persists over time, distinguishing them
    from value objects which are defined solely by their attributes.

    Seraphic Nature:
        An entity knows it exists as part of a larger whole. Its identity
        is not just a unique ID but a place in the organism's structure.
        The entity can reflect on its own nature and report its state.
    """

    # =========================================================================
    # IDENTITY - The entity's essential nature
    # =========================================================================

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique identifier for this entity."""
        pass

    @property
    def entity_type(self) -> str:
        """Return the type name of this entity."""
        return self.__class__.__name__

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    def __repr__(self) -> str:
        return f"{self.entity_type}(id={self.id!r})"

    # =========================================================================
    # SELF-AWARENESS - The entity can reflect on its own nature
    # =========================================================================

    def introspect(self) -> Dict[str, Any]:
        """
        Introspect the entity's current state.

        This is not serialization - it's the entity looking at itself
        and reporting what it sees. Used for diagnostics, debugging,
        and health monitoring.
        """
        return {
            "entity_type": self.entity_type,
            "id": self.id,
        }


class AggregateRoot(Entity, ABC):
    """
    Base class for aggregate roots.

    Aggregate roots are the entry point to a cluster of domain objects.
    They maintain invariants across the aggregate boundary and emit
    domain events when significant state changes occur.

    Seraphic Architecture:
        An aggregate root is not just a container of data - it's a living
        aspect of the organism. It:

        1. NATURALLY emits events when its state changes (like a nerve
           naturally fires when stimulated)
        2. INHERENTLY tracks its version (like cells count their divisions)
        3. INTRINSICALLY knows its health (like tissue senses damage)
        4. ORGANICALLY participates in the whole (like an organ doesn't
           need to be told it's part of a body)

        The aggregate doesn't need an external coordinator to tell it
        what to do - it knows its purpose and acts accordingly.

    Event Sourcing:
        Aggregates are the source of truth. Events flow FROM them,
        not TO them. When an aggregate changes, it produces events
        as a natural consequence, like a heart producing heartbeats.
    """

    # =========================================================================
    # LIFECYCLE - Birth and maturation
    # =========================================================================

    def __init__(self) -> None:
        """
        Initialize the aggregate root.

        This is the aggregate's birth - it comes into existence with
        the inherent capacity for events, versioning, and self-monitoring.
        """
        self._domain_events: List[DomainEvent] = []
        self._version: int = 0
        self._created_at: datetime = datetime.now(timezone.utc)
        self._last_event_at: Optional[datetime] = None
        self._invariant_violations: List[str] = []

    # =========================================================================
    # EVENT SOURCING - Natural event emission
    # =========================================================================

    @property
    def version(self) -> int:
        """Current version for optimistic concurrency."""
        return self._version

    @property
    def domain_events(self) -> List[DomainEvent]:
        """Pending domain events to be dispatched."""
        return list(self._domain_events)

    @property
    def has_pending_events(self) -> bool:
        """Check if there are events waiting to be dispatched."""
        return len(self._domain_events) > 0

    @property
    def event_count(self) -> int:
        """Number of pending events."""
        return len(self._domain_events)

    def add_domain_event(self, event: DomainEvent) -> None:
        """
        Add a domain event to be dispatched on commit.

        This is the aggregate's voice - when something significant happens,
        the aggregate speaks by emitting an event. This isn't optional
        behavior; it's the aggregate's natural response to change.
        """
        self._domain_events.append(event)
        self._last_event_at = datetime.now(timezone.utc)

    def clear_domain_events(self) -> List[DomainEvent]:
        """
        Clear and return all pending domain events.

        Events are cleared when they've been persisted to the event store.
        This is like exhaling - the aggregate releases what it has produced.
        """
        events = self._domain_events
        self._domain_events = []
        return events

    def increment_version(self) -> None:
        """
        Increment version after successful update.

        Each version represents a distinct state in the aggregate's history.
        Like rings in a tree, versions accumulate as the aggregate evolves.
        """
        self._version += 1

    # =========================================================================
    # SELF-MONITORING - Intrinsic health awareness
    # =========================================================================

    @property
    def is_healthy(self) -> bool:
        """
        Check if the aggregate is in a healthy state.

        The aggregate knows its own health - it doesn't need an external
        monitor to tell it something is wrong. This is intrinsic awareness.
        """
        return len(self._invariant_violations) == 0

    @property
    def invariant_violations(self) -> List[str]:
        """Any invariants that are currently violated."""
        return list(self._invariant_violations)

    def _validate_invariants(self) -> None:
        """
        Validate all invariants and populate violations list.

        Override in subclasses to add domain-specific invariant checks.
        This is the aggregate examining itself for consistency.
        """
        self._invariant_violations = []
        # Subclasses add their own invariant checks here

    def _add_invariant_violation(self, violation: str) -> None:
        """Record an invariant violation."""
        if violation not in self._invariant_violations:
            self._invariant_violations.append(violation)

    # =========================================================================
    # INTROSPECTION - Self-awareness and reflection
    # =========================================================================

    def introspect(self) -> Dict[str, Any]:
        """
        Introspect the aggregate's current state.

        This provides a comprehensive view of the aggregate for diagnostics.
        It's the aggregate looking at itself and reporting what it sees.
        """
        self._validate_invariants()
        return {
            "entity_type": self.entity_type,
            "id": self.id,
            "version": self._version,
            "is_healthy": self.is_healthy,
            "invariant_violations": self._invariant_violations,
            "pending_events": self.event_count,
            "created_at": self._created_at.isoformat(),
            "last_event_at": self._last_event_at.isoformat() if self._last_event_at else None,
        }

    # =========================================================================
    # RECONSTITUTION - Rebuilding from history (Event Sourcing support)
    # =========================================================================

    @classmethod
    def _reconstitute(cls, id: str, events: List[DomainEvent]) -> "AggregateRoot":
        """
        Reconstitute an aggregate from its event history.

        This is the aggregate being born anew from its memories.
        Each event is replayed to rebuild the current state.

        Override in subclasses to handle domain-specific events.
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement _reconstitute to support event sourcing"
        )

    def _apply_event(self, event: DomainEvent) -> None:
        """
        Apply an event to update aggregate state.

        This is the aggregate learning from its history.
        Override in subclasses to handle specific event types.
        """
        # Base implementation just tracks the event time
        self._last_event_at = event.occurred_at

    # =========================================================================
    # ORGANISM PARTICIPATION - Natural membership in the whole
    # =========================================================================

    @property
    def aggregate_type(self) -> str:
        """
        The type of this aggregate in the domain model.

        Used for stream naming in event sourcing and for routing
        in the mediator. This is intrinsic identity.
        """
        return self.__class__.__name__

    @property
    def stream_name(self) -> str:
        """
        The event stream name for this aggregate.

        Following event sourcing conventions: {type}-{id}
        This is how the aggregate identifies itself to the event store.
        """
        return f"{self.aggregate_type.lower()}-{self.id}"

    def to_snapshot(self) -> Dict[str, Any]:
        """
        Create a snapshot of the aggregate's current state.

        Snapshots are performance optimization - instead of replaying
        all events, we can start from a recent snapshot. This is the
        aggregate's memory consolidation.

        Override in subclasses to capture domain-specific state.
        """
        return {
            "aggregate_type": self.aggregate_type,
            "id": self.id,
            "version": self._version,
            "created_at": self._created_at.isoformat(),
        }

    @classmethod
    def from_snapshot(cls, snapshot: Dict[str, Any]) -> "AggregateRoot":
        """
        Restore an aggregate from a snapshot.

        Override in subclasses to restore domain-specific state.
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement from_snapshot to support snapshots"
        )


# =============================================================================
# AGGREGATE ROOTS - The vital organs of the system
# =============================================================================


class VerseAggregate(AggregateRoot):
    """
    Aggregate root for Bible verses.

    The verse is the fundamental unit of biblical text. This aggregate
    maintains the integrity of verse data and tracks all processing state.

    Invariants:
        - Reference must be valid (valid book, chapter >= 1, verse >= 1)
        - If processed, must have at least one extraction result
        - Text hashes must match stored hashes (immutability check)
    """

    def __init__(
        self,
        id: str,
        reference: VerseReference,
        text_original: str = "",
        text_english: str = "",
        text_lxx: str = "",
        language: str = "hebrew",
    ) -> None:
        super().__init__()
        self._id = id
        self._reference = reference
        self._text_original = text_original
        self._text_english = text_english
        self._text_lxx = text_lxx
        self._language = language
        self._text_hash = self._compute_text_hash()
        self._is_processed = False
        self._processing_status = "pending"
        self._extraction_ids: Set[str] = set()
        self._cross_reference_ids: Set[str] = set()
        self._metadata: Dict[str, Any] = {}
        self._created_at = datetime.now(timezone.utc)
        self._updated_at = self._created_at

    @property
    def id(self) -> str:
        return self._id

    @property
    def reference(self) -> VerseReference:
        return self._reference

    @property
    def text_original(self) -> str:
        return self._text_original

    @property
    def text_english(self) -> str:
        return self._text_english

    @property
    def text_lxx(self) -> str:
        return self._text_lxx

    @property
    def language(self) -> str:
        return self._language

    @property
    def is_processed(self) -> bool:
        return self._is_processed

    @property
    def processing_status(self) -> str:
        return self._processing_status

    @property
    def extraction_ids(self) -> FrozenSet[str]:
        return frozenset(self._extraction_ids)

    @property
    def cross_reference_ids(self) -> FrozenSet[str]:
        return frozenset(self._cross_reference_ids)

    def _compute_text_hash(self) -> str:
        """Compute hash of all text fields for change detection."""
        combined = f"{self._text_original}|{self._text_english}|{self._text_lxx}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    @classmethod
    def create(
        cls,
        reference: VerseReference,
        text_original: str = "",
        text_english: str = "",
        text_lxx: str = "",
        language: str = "hebrew",
    ) -> "VerseAggregate":
        """Factory method to create a new verse."""
        verse_id = str(reference)
        verse = cls(
            id=verse_id,
            reference=reference,
            text_original=text_original,
            text_english=text_english,
            text_lxx=text_lxx,
            language=language,
        )
        verse.add_domain_event(VerseCreated(
            aggregate_id=verse_id,
            reference=str(reference),
            book=reference.book,
            chapter=reference.chapter,
            verse=reference.verse,
            testament=reference.testament,
        ))
        return verse

    def update_text(
        self,
        text_original: Optional[str] = None,
        text_english: Optional[str] = None,
        text_lxx: Optional[str] = None,
    ) -> None:
        """Update verse text fields."""
        old_hash = self._text_hash

        if text_original is not None:
            self._text_original = text_original
        if text_english is not None:
            self._text_english = text_english
        if text_lxx is not None:
            self._text_lxx = text_lxx

        new_hash = self._compute_text_hash()
        if new_hash != old_hash:
            self._text_hash = new_hash
            self._updated_at = datetime.now(timezone.utc)
            self.increment_version()
            self.add_domain_event(VerseTextUpdated(
                aggregate_id=self._id,
                reference=str(self._reference),
                field_updated="text",
                old_hash=old_hash,
                new_hash=new_hash,
            ))

    def start_processing(self, pipeline_id: str) -> None:
        """Mark verse as being processed."""
        if self._processing_status == "in_progress":
            raise ValueError(f"Verse {self._id} is already being processed")

        self._processing_status = "in_progress"
        self._updated_at = datetime.now(timezone.utc)
        self.add_domain_event(VerseProcessingStarted(
            aggregate_id=self._id,
            reference=str(self._reference),
            pipeline_id=pipeline_id,
        ))

    def complete_processing(
        self,
        pipeline_id: str,
        success: bool,
        agents_executed: int,
        duration_ms: float,
    ) -> None:
        """Mark verse processing as complete."""
        self._processing_status = "completed" if success else "failed"
        self._is_processed = success
        self._updated_at = datetime.now(timezone.utc)
        self.increment_version()
        self.add_domain_event(VerseProcessingCompleted(
            aggregate_id=self._id,
            reference=str(self._reference),
            pipeline_id=pipeline_id,
            success=success,
            agents_executed=agents_executed,
            duration_ms=duration_ms,
        ))

    def add_extraction(self, extraction_id: str) -> None:
        """Link an extraction result to this verse."""
        self._extraction_ids.add(extraction_id)
        self._updated_at = datetime.now(timezone.utc)

    def add_cross_reference(self, crossref_id: str) -> None:
        """Link a cross-reference to this verse."""
        self._cross_reference_ids.add(crossref_id)
        self._updated_at = datetime.now(timezone.utc)

    # =========================================================================
    # SERAPHIC SELF-AWARENESS - Intrinsic knowledge of self
    # =========================================================================

    def _validate_invariants(self) -> None:
        """
        Validate all invariants for this verse aggregate.

        The verse knows its own rules and can detect when they're violated.
        This is intrinsic awareness, not external validation.
        """
        self._invariant_violations = []

        # Invariant 1: Reference must be valid
        if not self._reference:
            self._add_invariant_violation("Reference is required")

        # Invariant 2: If processed, must have extractions
        if self._is_processed and len(self._extraction_ids) == 0:
            self._add_invariant_violation(
                "Processed verse must have at least one extraction result"
            )

        # Invariant 3: Processing status must be consistent with is_processed
        if self._is_processed and self._processing_status != "completed":
            self._add_invariant_violation(
                f"Inconsistent state: is_processed=True but status={self._processing_status}"
            )

    def introspect(self) -> Dict[str, Any]:
        """
        Introspect the verse's current state.

        The verse looks at itself and reports what it sees.
        """
        base = super().introspect()
        base.update({
            "reference": str(self._reference),
            "book": self._reference.book,
            "chapter": self._reference.chapter,
            "verse": self._reference.verse,
            "testament": self._reference.testament,
            "language": self._language,
            "is_processed": self._is_processed,
            "processing_status": self._processing_status,
            "extraction_count": len(self._extraction_ids),
            "cross_reference_count": len(self._cross_reference_ids),
            "text_hash": self._text_hash,
            "updated_at": self._updated_at.isoformat(),
        })
        return base

    def to_snapshot(self) -> Dict[str, Any]:
        """
        Create a snapshot of the verse's current state.

        This is the verse remembering itself for later reconstitution.
        """
        base = super().to_snapshot()
        base.update({
            "reference": str(self._reference),
            "text_original": self._text_original,
            "text_english": self._text_english,
            "text_lxx": self._text_lxx,
            "language": self._language,
            "text_hash": self._text_hash,
            "is_processed": self._is_processed,
            "processing_status": self._processing_status,
            "extraction_ids": list(self._extraction_ids),
            "cross_reference_ids": list(self._cross_reference_ids),
            "metadata": self._metadata,
            "updated_at": self._updated_at.isoformat(),
        })
        return base

    @classmethod
    def from_snapshot(cls, snapshot: Dict[str, Any]) -> "VerseAggregate":
        """
        Restore a verse from a snapshot.

        The verse is reborn from its memories.
        """
        reference = VerseReference.parse(snapshot["reference"])
        verse = cls(
            id=snapshot["id"],
            reference=reference,
            text_original=snapshot.get("text_original", ""),
            text_english=snapshot.get("text_english", ""),
            text_lxx=snapshot.get("text_lxx", ""),
            language=snapshot.get("language", "hebrew"),
        )
        verse._version = snapshot.get("version", 0)
        verse._is_processed = snapshot.get("is_processed", False)
        verse._processing_status = snapshot.get("processing_status", "pending")
        verse._extraction_ids = set(snapshot.get("extraction_ids", []))
        verse._cross_reference_ids = set(snapshot.get("cross_reference_ids", []))
        verse._metadata = snapshot.get("metadata", {})

        if "updated_at" in snapshot:
            verse._updated_at = datetime.fromisoformat(snapshot["updated_at"])
        if "created_at" in snapshot:
            verse._created_at = datetime.fromisoformat(snapshot["created_at"])

        return verse

    @classmethod
    def _reconstitute(cls, id: str, events: List[DomainEvent]) -> "VerseAggregate":
        """
        Reconstitute a verse from its event history.

        The verse is reborn by reliving its memories.
        """
        # Find the creation event to get initial state
        verse: Optional["VerseAggregate"] = None

        for event in events:
            if isinstance(event, VerseCreated):
                reference = VerseReference.parse(event.reference)
                verse = cls(
                    id=id,
                    reference=reference,
                )
                verse._domain_events = []  # Don't re-emit creation event
            elif verse is not None:
                verse._apply_event(event)

        if verse is None:
            raise ValueError(f"Cannot reconstitute verse {id}: no VerseCreated event found")

        return verse

    def _apply_event(self, event: DomainEvent) -> None:
        """
        Apply an event to update verse state.

        The verse learns from its history.
        """
        super()._apply_event(event)

        if isinstance(event, VerseTextUpdated):
            # Text was updated - we don't have the actual text, just the hash
            pass  # State already updated in original operation
        elif isinstance(event, VerseProcessingStarted):
            self._processing_status = "in_progress"
        elif isinstance(event, VerseProcessingCompleted):
            self._processing_status = "completed" if event.success else "failed"
            self._is_processed = event.success
        # Add more event handlers as needed


class CrossReferenceAggregate(AggregateRoot):
    """
    Aggregate root for cross-references.

    Cross-references are the connections that weave Scripture together
    into a unified tapestry. This aggregate maintains the integrity of
    these sacred connections.

    Invariants:
        - Source and target must be valid verse references
        - Source and target cannot be the same verse
        - Connection type must be from valid enumeration
        - Confidence must be between 0 and 1
        - If verified by patristic source, patristic_support must be True
    """

    def __init__(
        self,
        id: str,
        source_ref: VerseReference,
        target_ref: VerseReference,
        connection_type: ConnectionTypeEnum,
        strength: ConnectionStrength,
        confidence: ConfidenceScore,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self._validate_refs(source_ref, target_ref)

        self._id = id
        self._source_ref = source_ref
        self._target_ref = target_ref
        self._connection_type = connection_type
        self._strength = strength
        self._confidence = confidence
        self._bidirectional = bidirectional
        self._verified = False
        self._verification_type: Optional[str] = None
        self._verifier: Optional[str] = None
        self._patristic_support = False
        self._patristic_sources: List[str] = []
        self._notes: List[str] = []
        self._sources: List[str] = []
        self._metadata: Dict[str, Any] = {}
        self._created_at = datetime.now(timezone.utc)
        self._updated_at = self._created_at

    def _validate_refs(self, source: VerseReference, target: VerseReference) -> None:
        """Validate that source and target are different."""
        if str(source) == str(target):
            raise ValueError("Source and target cannot be the same verse")

    @property
    def id(self) -> str:
        return self._id

    @property
    def source_ref(self) -> VerseReference:
        return self._source_ref

    @property
    def target_ref(self) -> VerseReference:
        return self._target_ref

    @property
    def connection_type(self) -> ConnectionTypeEnum:
        return self._connection_type

    @property
    def strength(self) -> ConnectionStrength:
        return self._strength

    @property
    def confidence(self) -> ConfidenceScore:
        return self._confidence

    @property
    def is_verified(self) -> bool:
        return self._verified

    @property
    def has_patristic_support(self) -> bool:
        return self._patristic_support

    @property
    def is_typological(self) -> bool:
        return self._connection_type == ConnectionTypeEnum.TYPOLOGICAL

    @property
    def spans_testaments(self) -> bool:
        """Check if this connection spans OT/NT boundary."""
        return self._source_ref.testament != self._target_ref.testament

    @classmethod
    def create(
        cls,
        source_ref: VerseReference,
        target_ref: VerseReference,
        connection_type: ConnectionTypeEnum,
        strength: ConnectionStrength,
        confidence: ConfidenceScore,
        bidirectional: bool = False,
    ) -> "CrossReferenceAggregate":
        """Factory method to create a new cross-reference."""
        # Generate deterministic ID from source/target/type
        id_input = f"{source_ref}|{target_ref}|{connection_type.value}"
        crossref_id = hashlib.sha256(id_input.encode()).hexdigest()[:16]

        crossref = cls(
            id=crossref_id,
            source_ref=source_ref,
            target_ref=target_ref,
            connection_type=connection_type,
            strength=strength,
            confidence=confidence,
            bidirectional=bidirectional,
        )
        crossref.add_domain_event(CrossReferenceCreated(
            aggregate_id=crossref_id,
            source_ref=str(source_ref),
            target_ref=str(target_ref),
            connection_type=connection_type.value,
            strength=strength.value,
            confidence=confidence.value,
        ))
        return crossref

    @classmethod
    def from_discovery(
        cls,
        source_ref: VerseReference,
        target_ref: VerseReference,
        connection_type: ConnectionTypeEnum,
        confidence: ConfidenceScore,
        model_name: str,
        features: Dict[str, float],
    ) -> "CrossReferenceAggregate":
        """Create from ML discovery with appropriate initial strength."""
        # Determine strength based on confidence
        if confidence.is_high:
            strength = ConnectionStrength.strong()
        elif confidence.is_medium:
            strength = ConnectionStrength.moderate()
        else:
            strength = ConnectionStrength.weak()

        crossref = cls.create(
            source_ref=source_ref,
            target_ref=target_ref,
            connection_type=connection_type,
            strength=strength,
            confidence=confidence,
        )
        crossref._sources.append(f"ML:{model_name}")
        crossref.add_domain_event(CrossReferenceDiscovered(
            aggregate_id=crossref._id,
            source_ref=str(source_ref),
            target_ref=str(target_ref),
            connection_type=connection_type.value,
            confidence=confidence.value,
            model_name=model_name,
            features=features,
        ))
        return crossref

    def update_strength(self, new_strength: ConnectionStrength, reason: str) -> None:
        """Update the connection strength."""
        old_strength = self._strength
        self._strength = new_strength
        self._updated_at = datetime.now(timezone.utc)
        self.increment_version()
        self.add_domain_event(CrossReferenceStrengthUpdated(
            aggregate_id=self._id,
            crossref_id=self._id,
            old_strength=old_strength.value,
            new_strength=new_strength.value,
            reason=reason,
        ))

    def verify(self, verification_type: str, verifier: str) -> None:
        """Mark the cross-reference as verified."""
        self._verified = True
        self._verification_type = verification_type
        self._verifier = verifier
        self._updated_at = datetime.now(timezone.utc)
        self.increment_version()

        if verification_type == "patristic":
            self._patristic_support = True
            self._patristic_sources.append(verifier)

        self.add_domain_event(CrossReferenceVerified(
            aggregate_id=self._id,
            crossref_id=self._id,
            verification_type=verification_type,
            verifier=verifier,
        ))

    def add_patristic_source(self, father_name: str) -> None:
        """Add patristic attestation for this connection."""
        if father_name not in self._patristic_sources:
            self._patristic_sources.append(father_name)
            self._patristic_support = True
            self._updated_at = datetime.now(timezone.utc)

    def add_note(self, note: str) -> None:
        """Add an explanatory note."""
        self._notes.append(note)
        self._updated_at = datetime.now(timezone.utc)

    # =========================================================================
    # SERAPHIC SELF-AWARENESS - Intrinsic knowledge of self
    # =========================================================================

    def _validate_invariants(self) -> None:
        """
        Validate all invariants for this cross-reference aggregate.

        The connection knows its own rules and can detect violations.
        """
        self._invariant_violations = []

        # Invariant 1: Source and target must be different
        if str(self._source_ref) == str(self._target_ref):
            self._add_invariant_violation("Source and target cannot be identical")

        # Invariant 2: Confidence must be valid
        if not (0.0 <= self._confidence.value <= 1.0):
            self._add_invariant_violation(
                f"Confidence must be 0-1: {self._confidence.value}"
            )

        # Invariant 3: If verified by patristic source, must have support flag
        if self._verification_type == "patristic" and not self._patristic_support:
            self._add_invariant_violation(
                "Patristic verification requires patristic_support=True"
            )

        # Invariant 4: Verified connections should have a verifier
        if self._verified and not self._verifier:
            self._add_invariant_violation(
                "Verified connection must have a verifier recorded"
            )

    def introspect(self) -> Dict[str, Any]:
        """
        Introspect the cross-reference's current state.

        The connection looks at itself and reports what it sees.
        """
        base = super().introspect()
        base.update({
            "source_ref": str(self._source_ref),
            "target_ref": str(self._target_ref),
            "connection_type": self._connection_type.value,
            "strength": self._strength.value,
            "confidence": self._confidence.value,
            "confidence_tier": self._confidence.tier,
            "bidirectional": self._bidirectional,
            "verified": self._verified,
            "verification_type": self._verification_type,
            "has_patristic_support": self._patristic_support,
            "patristic_sources": list(self._patristic_sources),
            "spans_testaments": self.spans_testaments,
            "is_typological": self.is_typological,
            "notes_count": len(self._notes),
            "sources": list(self._sources),
            "updated_at": self._updated_at.isoformat(),
        })
        return base

    def to_snapshot(self) -> Dict[str, Any]:
        """
        Create a snapshot of the cross-reference's current state.

        The connection remembers itself for later reconstitution.
        """
        base = super().to_snapshot()
        base.update({
            "source_ref": str(self._source_ref),
            "target_ref": str(self._target_ref),
            "connection_type": self._connection_type.value,
            "strength": self._strength.value,
            "confidence": self._confidence.value,
            "bidirectional": self._bidirectional,
            "verified": self._verified,
            "verification_type": self._verification_type,
            "verifier": self._verifier,
            "patristic_support": self._patristic_support,
            "patristic_sources": list(self._patristic_sources),
            "notes": list(self._notes),
            "sources": list(self._sources),
            "metadata": self._metadata,
            "updated_at": self._updated_at.isoformat(),
        })
        return base

    @classmethod
    def from_snapshot(cls, snapshot: Dict[str, Any]) -> "CrossReferenceAggregate":
        """
        Restore a cross-reference from a snapshot.

        The connection is reborn from its memories.
        """
        source_ref = VerseReference.parse(snapshot["source_ref"])
        target_ref = VerseReference.parse(snapshot["target_ref"])
        connection_type = ConnectionTypeEnum(snapshot["connection_type"])
        strength = ConnectionStrength(snapshot["strength"])
        confidence = ConfidenceScore(snapshot["confidence"])

        crossref = cls(
            id=snapshot["id"],
            source_ref=source_ref,
            target_ref=target_ref,
            connection_type=connection_type,
            strength=strength,
            confidence=confidence,
            bidirectional=snapshot.get("bidirectional", False),
        )
        crossref._version = snapshot.get("version", 0)
        crossref._verified = snapshot.get("verified", False)
        crossref._verification_type = snapshot.get("verification_type")
        crossref._verifier = snapshot.get("verifier")
        crossref._patristic_support = snapshot.get("patristic_support", False)
        crossref._patristic_sources = list(snapshot.get("patristic_sources", []))
        crossref._notes = list(snapshot.get("notes", []))
        crossref._sources = list(snapshot.get("sources", []))
        crossref._metadata = snapshot.get("metadata", {})

        if "updated_at" in snapshot:
            crossref._updated_at = datetime.fromisoformat(snapshot["updated_at"])
        if "created_at" in snapshot:
            crossref._created_at = datetime.fromisoformat(snapshot["created_at"])

        return crossref

    @classmethod
    def _reconstitute(cls, id: str, events: List[DomainEvent]) -> "CrossReferenceAggregate":
        """
        Reconstitute a cross-reference from its event history.

        The connection is reborn by reliving its memories.
        """
        crossref: Optional["CrossReferenceAggregate"] = None

        for event in events:
            if isinstance(event, CrossReferenceCreated):
                source_ref = VerseReference.parse(event.source_ref)
                target_ref = VerseReference.parse(event.target_ref)
                connection_type = ConnectionTypeEnum(event.connection_type)
                strength = ConnectionStrength(event.strength)
                confidence = ConfidenceScore(event.confidence)

                crossref = cls(
                    id=id,
                    source_ref=source_ref,
                    target_ref=target_ref,
                    connection_type=connection_type,
                    strength=strength,
                    confidence=confidence,
                )
                crossref._domain_events = []  # Don't re-emit creation event
            elif crossref is not None:
                crossref._apply_event(event)

        if crossref is None:
            raise ValueError(
                f"Cannot reconstitute cross-reference {id}: no CrossReferenceCreated event found"
            )

        return crossref

    def _apply_event(self, event: DomainEvent) -> None:
        """
        Apply an event to update cross-reference state.

        The connection learns from its history.
        """
        super()._apply_event(event)

        if isinstance(event, CrossReferenceStrengthUpdated):
            self._strength = ConnectionStrength(event.new_strength)
        elif isinstance(event, CrossReferenceVerified):
            self._verified = True
            self._verification_type = event.verification_type
            self._verifier = event.verifier
            if event.verification_type == "patristic":
                self._patristic_support = True
                if event.verifier not in self._patristic_sources:
                    self._patristic_sources.append(event.verifier)


class ExtractionResultAggregate(AggregateRoot):
    """
    Aggregate root for extraction results.

    Each extraction result represents the output of a single agent's
    analysis of a verse. Multiple agents contribute their specialized
    knowledge, like organs contributing to the body's function.

    Invariants:
        - Agent name must be from registered agents
        - Confidence must be between 0 and 1
        - Status must follow valid state transitions
    """

    VALID_STATUSES: ClassVar[FrozenSet[str]] = frozenset({
        "pending", "in_progress", "completed", "failed", "skipped"
    })

    def __init__(
        self,
        id: str,
        verse_id: str,
        agent_name: str,
        extraction_type: ExtractionType,
        confidence: ConfidenceScore,
        data: Dict[str, Any],
    ) -> None:
        super().__init__()
        self._id = id
        self._verse_id = verse_id
        self._agent_name = agent_name
        self._extraction_type = extraction_type
        self._confidence = confidence
        self._data = data
        self._status = "completed"
        self._error: Optional[str] = None
        self._processing_time_ms: float = 0.0
        self._created_at = datetime.now(timezone.utc)
        self._updated_at = self._created_at

    @property
    def id(self) -> str:
        return self._id

    @property
    def verse_id(self) -> str:
        return self._verse_id

    @property
    def agent_name(self) -> str:
        return self._agent_name

    @property
    def extraction_type(self) -> ExtractionType:
        return self._extraction_type

    @property
    def confidence(self) -> ConfidenceScore:
        return self._confidence

    @property
    def data(self) -> Dict[str, Any]:
        return dict(self._data)

    @property
    def status(self) -> str:
        return self._status

    @classmethod
    def create(
        cls,
        verse_id: str,
        agent_name: str,
        extraction_type: ExtractionType,
        confidence: ConfidenceScore,
        data: Dict[str, Any],
        processing_time_ms: float = 0.0,
    ) -> "ExtractionResultAggregate":
        """Factory method to create a new extraction result."""
        result_id = f"{verse_id}_{agent_name}_{uuid4().hex[:8]}"

        result = cls(
            id=result_id,
            verse_id=verse_id,
            agent_name=agent_name,
            extraction_type=extraction_type,
            confidence=confidence,
            data=data,
        )
        result._processing_time_ms = processing_time_ms
        result.add_domain_event(ExtractionResultCreated(
            aggregate_id=result_id,
            verse_id=verse_id,
            agent_name=agent_name,
            extraction_type=extraction_type.value,
            confidence=confidence.value,
        ))
        return result

    def update_confidence(self, new_confidence: ConfidenceScore) -> None:
        """Update the confidence score."""
        old_confidence = self._confidence
        self._confidence = new_confidence
        self._updated_at = datetime.now(timezone.utc)
        self.increment_version()
        self.add_domain_event(ExtractionResultUpdated(
            aggregate_id=self._id,
            extraction_id=self._id,
            agent_name=self._agent_name,
            old_confidence=old_confidence.value,
            new_confidence=new_confidence.value,
        ))

    def update_data(self, new_data: Dict[str, Any]) -> None:
        """Update the extraction data."""
        self._data = dict(new_data)
        self._updated_at = datetime.now(timezone.utc)
        self.increment_version()


class PatristicCitationAggregate(AggregateRoot):
    """
    Aggregate root for patristic citations.

    Patristic citations are the voice of the Church Fathers interpreting
    Scripture. They provide authoritative witness to the meaning of texts
    and validate typological/thematic connections.

    Invariants:
        - Father name must not be empty
        - Work title must not be empty
        - At least one verse reference required
    """

    def __init__(
        self,
        id: str,
        father: str,
        work: str,
        verse_refs: List[VerseReference],
        interpretation_type: InterpretationType,
        quote: str,
    ) -> None:
        super().__init__()
        self._validate(father, work, verse_refs)

        self._id = id
        self._father = father
        self._work = work
        self._verse_refs = list(verse_refs)
        self._interpretation_type = interpretation_type
        self._quote = quote
        self._book: str = ""
        self._chapter: str = ""
        self._section: str = ""
        self._summary: str = ""
        self._themes: List[str] = []
        self._language: str = "greek"
        self._created_at = datetime.now(timezone.utc)

    def _validate(
        self, father: str, work: str, verse_refs: List[VerseReference]
    ) -> None:
        if not father:
            raise ValueError("Father name is required")
        if not work:
            raise ValueError("Work title is required")
        if not verse_refs:
            raise ValueError("At least one verse reference is required")

    @property
    def id(self) -> str:
        return self._id

    @property
    def father(self) -> str:
        return self._father

    @property
    def work(self) -> str:
        return self._work

    @property
    def verse_refs(self) -> List[VerseReference]:
        return list(self._verse_refs)

    @property
    def interpretation_type(self) -> InterpretationType:
        return self._interpretation_type

    @classmethod
    def create(
        cls,
        father: str,
        work: str,
        verse_refs: List[VerseReference],
        interpretation_type: InterpretationType,
        quote: str,
    ) -> "PatristicCitationAggregate":
        """Factory method to create a new patristic citation."""
        citation_id = f"{father}_{work}_{uuid4().hex[:8]}".replace(" ", "_").lower()

        citation = cls(
            id=citation_id,
            father=father,
            work=work,
            verse_refs=verse_refs,
            interpretation_type=interpretation_type,
            quote=quote,
        )

        for ref in verse_refs:
            citation.add_domain_event(PatristicCitationLinked(
                aggregate_id=citation_id,
                verse_ref=str(ref),
                father=father,
                work=work,
                interpretation_type=interpretation_type.value,
            ))

        return citation

    def add_verse_reference(self, ref: VerseReference) -> None:
        """Link an additional verse to this citation."""
        if ref not in self._verse_refs:
            self._verse_refs.append(ref)
            self.add_domain_event(PatristicCitationLinked(
                aggregate_id=self._id,
                verse_ref=str(ref),
                father=self._father,
                work=self._work,
                interpretation_type=self._interpretation_type.value,
            ))


class GoldenRecordAggregate(AggregateRoot):
    """
    Aggregate root for golden records.

    The golden record represents the fully processed, certified output
    for a verse - the culmination of all agent extractions synthesized
    into a canonical record.

    Invariants:
        - Cannot be certified without passing quality gates
        - Certification tier must match score thresholds
        - Must have contributions from minimum required agents
    """

    MIN_AGENTS_FOR_CERTIFICATION: ClassVar[int] = 3

    TIER_THRESHOLDS: ClassVar[Dict[QualityTier, float]] = {
        QualityTier.GOLD: 0.90,
        QualityTier.SILVER: 0.75,
        QualityTier.BRONZE: 0.60,
        QualityTier.PROVISIONAL: 0.0,
    }

    def __init__(
        self,
        id: str,
        verse_id: str,
        text: str,
    ) -> None:
        super().__init__()
        self._id = id
        self._verse_id = verse_id
        self._text = text
        self._quality_tier = QualityTier.PROVISIONAL
        self._quality_score: float = 0.0
        self._data: Dict[str, Dict[str, Any]] = {}
        self._extraction_ids: List[str] = []
        self._phases_executed: List[str] = []
        self._certified = False
        self._certification_date: Optional[datetime] = None
        self._created_at = datetime.now(timezone.utc)
        self._updated_at = self._created_at

    @property
    def id(self) -> str:
        return self._id

    @property
    def verse_id(self) -> str:
        return self._verse_id

    @property
    def quality_tier(self) -> QualityTier:
        return self._quality_tier

    @property
    def quality_score(self) -> float:
        return self._quality_score

    @property
    def is_certified(self) -> bool:
        return self._certified

    @property
    def agent_count(self) -> int:
        return len(self._extraction_ids)

    @classmethod
    def create(cls, verse_id: str, text: str) -> "GoldenRecordAggregate":
        """Factory method to create a new golden record."""
        return cls(
            id=f"gr_{verse_id}",
            verse_id=verse_id,
            text=text,
        )

    def add_extraction_data(
        self,
        extraction_type: str,
        data: Dict[str, Any],
        extraction_id: str,
    ) -> None:
        """Add extraction data from an agent."""
        self._data[extraction_type] = data
        if extraction_id not in self._extraction_ids:
            self._extraction_ids.append(extraction_id)
        self._updated_at = datetime.now(timezone.utc)

    def record_phase_execution(self, phase_name: str) -> None:
        """Record that a pipeline phase was executed."""
        if phase_name not in self._phases_executed:
            self._phases_executed.append(phase_name)

    def certify(self, quality_score: float) -> None:
        """
        Certify the golden record if it meets quality gates.

        Raises:
            ValueError: If certification requirements not met
        """
        if self.agent_count < self.MIN_AGENTS_FOR_CERTIFICATION:
            raise ValueError(
                f"Cannot certify: need {self.MIN_AGENTS_FOR_CERTIFICATION} agents, "
                f"have {self.agent_count}"
            )

        self._quality_score = quality_score

        # Determine tier based on score
        for tier, threshold in sorted(
            self.TIER_THRESHOLDS.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if quality_score >= threshold:
                self._quality_tier = tier
                break

        if self._quality_tier.is_certified():
            self._certified = True
            self._certification_date = datetime.now(timezone.utc)
            self.increment_version()
            self.add_domain_event(GoldenRecordCertified(
                aggregate_id=self._id,
                verse_id=self._verse_id,
                quality_tier=self._quality_tier.value,
                score=quality_score,
                agents_contributing=self.agent_count,
            ))

    def decertify(self, reason: str) -> None:
        """Remove certification from this golden record."""
        if not self._certified:
            return

        previous_tier = self._quality_tier
        self._certified = False
        self._quality_tier = QualityTier.PROVISIONAL
        self._certification_date = None
        self.increment_version()
        self.add_domain_event(GoldenRecordDecertified(
            aggregate_id=self._id,
            verse_id=self._verse_id,
            previous_tier=previous_tier.value,
            reason=reason,
        ))
