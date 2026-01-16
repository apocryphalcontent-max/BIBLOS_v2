"""
BIBLOS v2 - Domain Specifications

Type-safe, composable query specifications for domain entities.
These specifications implement the Specification Pattern, allowing
complex queries to be built from simple, reusable predicates.

The specifications act like the sensory neurons of the system -
they query and filter domain entities based on precise criteria.

Usage:
    from domain.specifications import (
        VerseByBookSpec, VerseByChapterSpec, CrossRefHighConfidenceSpec
    )

    # Simple specification
    spec = VerseByBookSpec("GEN")

    # Composed specification
    spec = (
        VerseByBookSpec("GEN")
        .and_(VerseByChapterSpec(1))
        .and_(VerseProcessedSpec())
    )

    # Use with repository
    verses = await verse_repo.find(spec)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from db.interfaces import ISpecification

from domain.entities import (
    VerseAggregate,
    CrossReferenceAggregate,
    ExtractionResultAggregate,
    VerseReference,
    ConnectionTypeEnum,
    ConnectionStrength,
    ExtractionType,
)


# =============================================================================
# VERSE SPECIFICATIONS
# =============================================================================


@dataclass
class VerseByBookSpec(ISpecification[VerseAggregate]):
    """Specification for verses in a specific book."""
    book_code: str

    def is_satisfied_by(self, entity: VerseAggregate) -> bool:
        return entity.reference.book == self.book_code.upper()

    def to_query_params(self) -> Dict[str, Any]:
        return {"book": self.book_code.upper()}


@dataclass
class VerseByChapterSpec(ISpecification[VerseAggregate]):
    """Specification for verses in a specific chapter."""
    book_code: str
    chapter: int

    def __init__(self, chapter: int, book_code: str = ""):
        self.chapter = chapter
        self.book_code = book_code

    def is_satisfied_by(self, entity: VerseAggregate) -> bool:
        if self.book_code:
            return (
                entity.reference.book == self.book_code.upper()
                and entity.reference.chapter == self.chapter
            )
        return entity.reference.chapter == self.chapter

    def to_query_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {"chapter": self.chapter}
        if self.book_code:
            params["book"] = self.book_code.upper()
        return params


@dataclass
class VerseByReferenceSpec(ISpecification[VerseAggregate]):
    """Specification for a verse by exact reference."""
    reference: VerseReference

    def is_satisfied_by(self, entity: VerseAggregate) -> bool:
        return entity.reference == self.reference

    def to_query_params(self) -> Dict[str, Any]:
        return {
            "book": self.reference.book,
            "chapter": self.reference.chapter,
            "verse": self.reference.verse,
        }

    @classmethod
    def from_string(cls, ref_string: str) -> "VerseByReferenceSpec":
        """Create from verse reference string."""
        return cls(reference=VerseReference.parse(ref_string))


@dataclass
class VerseWithTextContainingSpec(ISpecification[VerseAggregate]):
    """Specification for verses containing specific text."""
    search_text: str
    case_sensitive: bool = False

    def is_satisfied_by(self, entity: VerseAggregate) -> bool:
        if self.case_sensitive:
            return (
                self.search_text in entity.text_original
                or self.search_text in entity.text_english
                or self.search_text in entity.text_lxx
            )
        search_lower = self.search_text.lower()
        return (
            search_lower in entity.text_original.lower()
            or search_lower in entity.text_english.lower()
            or search_lower in entity.text_lxx.lower()
        )

    def to_query_params(self) -> Dict[str, Any]:
        return {
            "$text_search": self.search_text,
            "case_sensitive": self.case_sensitive,
        }


@dataclass
class VerseProcessedSpec(ISpecification[VerseAggregate]):
    """Specification for verses that have been processed."""

    def is_satisfied_by(self, entity: VerseAggregate) -> bool:
        return entity.is_processed

    def to_query_params(self) -> Dict[str, Any]:
        return {"is_processed": True}


@dataclass
class VerseNeedsProcessingSpec(ISpecification[VerseAggregate]):
    """Specification for verses that need processing."""

    def is_satisfied_by(self, entity: VerseAggregate) -> bool:
        return not entity.is_processed and entity.processing_status != "in_progress"

    def to_query_params(self) -> Dict[str, Any]:
        return {
            "$and": [
                {"is_processed": False},
                {"processing_status": {"$ne": "in_progress"}},
            ]
        }


@dataclass
class VerseByTestamentSpec(ISpecification[VerseAggregate]):
    """Specification for verses in a specific testament."""
    testament: str  # "OT" or "NT"

    def is_satisfied_by(self, entity: VerseAggregate) -> bool:
        return entity.reference.testament == self.testament.upper()

    def to_query_params(self) -> Dict[str, Any]:
        return {"testament": self.testament.upper()}


@dataclass
class VerseInRangeSpec(ISpecification[VerseAggregate]):
    """Specification for verses in a reference range."""
    start_ref: VerseReference
    end_ref: VerseReference

    def is_satisfied_by(self, entity: VerseAggregate) -> bool:
        ref = entity.reference
        # Simple comparison assuming same book
        if ref.book != self.start_ref.book:
            return False
        start = (self.start_ref.chapter, self.start_ref.verse)
        end = (self.end_ref.chapter, self.end_ref.verse)
        current = (ref.chapter, ref.verse)
        return start <= current <= end

    def to_query_params(self) -> Dict[str, Any]:
        return {
            "book": self.start_ref.book,
            "$range": {
                "start_chapter": self.start_ref.chapter,
                "start_verse": self.start_ref.verse,
                "end_chapter": self.end_ref.chapter,
                "end_verse": self.end_ref.verse,
            }
        }


# =============================================================================
# CROSS-REFERENCE SPECIFICATIONS
# =============================================================================


@dataclass
class CrossRefBySourceSpec(ISpecification[CrossReferenceAggregate]):
    """Specification for cross-references from a source verse."""
    source_ref: VerseReference

    def is_satisfied_by(self, entity: CrossReferenceAggregate) -> bool:
        return entity.source_ref == self.source_ref

    def to_query_params(self) -> Dict[str, Any]:
        return {"source_ref": str(self.source_ref)}

    @classmethod
    def from_string(cls, ref_string: str) -> "CrossRefBySourceSpec":
        return cls(source_ref=VerseReference.parse(ref_string))


@dataclass
class CrossRefByTargetSpec(ISpecification[CrossReferenceAggregate]):
    """Specification for cross-references to a target verse."""
    target_ref: VerseReference

    def is_satisfied_by(self, entity: CrossReferenceAggregate) -> bool:
        return entity.target_ref == self.target_ref

    def to_query_params(self) -> Dict[str, Any]:
        return {"target_ref": str(self.target_ref)}

    @classmethod
    def from_string(cls, ref_string: str) -> "CrossRefByTargetSpec":
        return cls(target_ref=VerseReference.parse(ref_string))


@dataclass
class CrossRefByTypeSpec(ISpecification[CrossReferenceAggregate]):
    """Specification for cross-references by connection type."""
    connection_type: ConnectionTypeEnum

    def is_satisfied_by(self, entity: CrossReferenceAggregate) -> bool:
        return entity.connection_type == self.connection_type

    def to_query_params(self) -> Dict[str, Any]:
        return {"connection_type": self.connection_type.value}


@dataclass
class CrossRefByStrengthSpec(ISpecification[CrossReferenceAggregate]):
    """Specification for cross-references by strength level."""
    min_strength: ConnectionStrength

    def is_satisfied_by(self, entity: CrossReferenceAggregate) -> bool:
        return not self.min_strength.is_stronger_than(entity.strength)

    def to_query_params(self) -> Dict[str, Any]:
        return {"strength_min_weight": self.min_strength.weight}


@dataclass
class CrossRefVerifiedSpec(ISpecification[CrossReferenceAggregate]):
    """Specification for verified cross-references."""
    verification_type: Optional[str] = None  # "human", "patristic", "ml", or None for any

    def is_satisfied_by(self, entity: CrossReferenceAggregate) -> bool:
        if not entity.is_verified:
            return False
        if self.verification_type:
            return entity._verification_type == self.verification_type
        return True

    def to_query_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {"verified": True}
        if self.verification_type:
            params["verification_type"] = self.verification_type
        return params


@dataclass
class CrossRefHighConfidenceSpec(ISpecification[CrossReferenceAggregate]):
    """Specification for high-confidence cross-references."""
    min_confidence: float = 0.85

    def is_satisfied_by(self, entity: CrossReferenceAggregate) -> bool:
        return entity.confidence.value >= self.min_confidence

    def to_query_params(self) -> Dict[str, Any]:
        return {"confidence_gte": self.min_confidence}


@dataclass
class CrossRefTypologicalSpec(ISpecification[CrossReferenceAggregate]):
    """Specification for typological cross-references (OT type -> NT antitype)."""

    def is_satisfied_by(self, entity: CrossReferenceAggregate) -> bool:
        return (
            entity.is_typological
            and entity.source_ref.is_old_testament
            and entity.target_ref.is_new_testament
        )

    def to_query_params(self) -> Dict[str, Any]:
        return {
            "connection_type": "typological",
            "source_testament": "OT",
            "target_testament": "NT",
        }


@dataclass
class CrossRefPatristicSupportedSpec(ISpecification[CrossReferenceAggregate]):
    """Specification for cross-references with patristic attestation."""

    def is_satisfied_by(self, entity: CrossReferenceAggregate) -> bool:
        return entity.has_patristic_support

    def to_query_params(self) -> Dict[str, Any]:
        return {"patristic_support": True}


@dataclass
class CrossRefBidirectionalSpec(ISpecification[CrossReferenceAggregate]):
    """Specification for cross-references involving a verse (source or target)."""
    verse_ref: VerseReference

    def is_satisfied_by(self, entity: CrossReferenceAggregate) -> bool:
        return entity.source_ref == self.verse_ref or entity.target_ref == self.verse_ref

    def to_query_params(self) -> Dict[str, Any]:
        return {
            "$or": [
                {"source_ref": str(self.verse_ref)},
                {"target_ref": str(self.verse_ref)},
            ]
        }


@dataclass
class CrossRefSpansTestamentsSpec(ISpecification[CrossReferenceAggregate]):
    """Specification for cross-references that span OT/NT boundary."""

    def is_satisfied_by(self, entity: CrossReferenceAggregate) -> bool:
        return entity.spans_testaments

    def to_query_params(self) -> Dict[str, Any]:
        return {"spans_testaments": True}


# =============================================================================
# EXTRACTION RESULT SPECIFICATIONS
# =============================================================================


@dataclass
class ExtractionByVerseSpec(ISpecification[ExtractionResultAggregate]):
    """Specification for extractions for a specific verse."""
    verse_id: str

    def is_satisfied_by(self, entity: ExtractionResultAggregate) -> bool:
        return entity.verse_id == self.verse_id

    def to_query_params(self) -> Dict[str, Any]:
        return {"verse_id": self.verse_id}


@dataclass
class ExtractionByAgentSpec(ISpecification[ExtractionResultAggregate]):
    """Specification for extractions from a specific agent."""
    agent_name: str

    def is_satisfied_by(self, entity: ExtractionResultAggregate) -> bool:
        return entity.agent_name.lower() == self.agent_name.lower()

    def to_query_params(self) -> Dict[str, Any]:
        return {"agent_name": self.agent_name}


@dataclass
class ExtractionByTypeSpec(ISpecification[ExtractionResultAggregate]):
    """Specification for extractions by extraction type."""
    extraction_type: ExtractionType

    def is_satisfied_by(self, entity: ExtractionResultAggregate) -> bool:
        return entity.extraction_type == self.extraction_type

    def to_query_params(self) -> Dict[str, Any]:
        return {"extraction_type": self.extraction_type.value}


@dataclass
class ExtractionCompletedSpec(ISpecification[ExtractionResultAggregate]):
    """Specification for completed extractions."""

    def is_satisfied_by(self, entity: ExtractionResultAggregate) -> bool:
        return entity.status == "completed"

    def to_query_params(self) -> Dict[str, Any]:
        return {"status": "completed"}


@dataclass
class ExtractionHighConfidenceSpec(ISpecification[ExtractionResultAggregate]):
    """Specification for high-confidence extractions."""
    min_confidence: float = 0.85

    def is_satisfied_by(self, entity: ExtractionResultAggregate) -> bool:
        return entity.confidence.value >= self.min_confidence

    def to_query_params(self) -> Dict[str, Any]:
        return {"confidence_gte": self.min_confidence}


@dataclass
class ExtractionByVerseAndAgentSpec(ISpecification[ExtractionResultAggregate]):
    """Specification for a specific verse/agent combination."""
    verse_id: str
    agent_name: str

    def is_satisfied_by(self, entity: ExtractionResultAggregate) -> bool:
        return (
            entity.verse_id == self.verse_id
            and entity.agent_name.lower() == self.agent_name.lower()
        )

    def to_query_params(self) -> Dict[str, Any]:
        return {
            "verse_id": self.verse_id,
            "agent_name": self.agent_name,
        }


# =============================================================================
# COMPOSITE SPECIFICATION BUILDERS
# =============================================================================


class VerseSpecBuilder:
    """
    Fluent builder for composing verse specifications.

    Usage:
        spec = (VerseSpecBuilder()
            .in_book("GEN")
            .in_chapter(1)
            .processed()
            .build())
    """

    def __init__(self) -> None:
        self._specs: List[ISpecification[VerseAggregate]] = []

    def in_book(self, book_code: str) -> "VerseSpecBuilder":
        self._specs.append(VerseByBookSpec(book_code))
        return self

    def in_chapter(self, chapter: int, book_code: str = "") -> "VerseSpecBuilder":
        self._specs.append(VerseByChapterSpec(chapter, book_code))
        return self

    def at_reference(self, ref: VerseReference) -> "VerseSpecBuilder":
        self._specs.append(VerseByReferenceSpec(ref))
        return self

    def containing_text(
        self, text: str, case_sensitive: bool = False
    ) -> "VerseSpecBuilder":
        self._specs.append(VerseWithTextContainingSpec(text, case_sensitive))
        return self

    def processed(self) -> "VerseSpecBuilder":
        self._specs.append(VerseProcessedSpec())
        return self

    def needs_processing(self) -> "VerseSpecBuilder":
        self._specs.append(VerseNeedsProcessingSpec())
        return self

    def in_testament(self, testament: str) -> "VerseSpecBuilder":
        self._specs.append(VerseByTestamentSpec(testament))
        return self

    def build(self) -> ISpecification[VerseAggregate]:
        """Build the composite specification."""
        if not self._specs:
            from db.interfaces import TrueSpecification
            return TrueSpecification()

        result = self._specs[0]
        for spec in self._specs[1:]:
            result = result.and_(spec)
        return result


class CrossRefSpecBuilder:
    """
    Fluent builder for composing cross-reference specifications.

    Usage:
        spec = (CrossRefSpecBuilder()
            .from_verse("GEN.1.1")
            .of_type(ConnectionTypeEnum.TYPOLOGICAL)
            .high_confidence()
            .build())
    """

    def __init__(self) -> None:
        self._specs: List[ISpecification[CrossReferenceAggregate]] = []

    def from_verse(self, ref: str | VerseReference) -> "CrossRefSpecBuilder":
        if isinstance(ref, str):
            ref = VerseReference.parse(ref)
        self._specs.append(CrossRefBySourceSpec(ref))
        return self

    def to_verse(self, ref: str | VerseReference) -> "CrossRefSpecBuilder":
        if isinstance(ref, str):
            ref = VerseReference.parse(ref)
        self._specs.append(CrossRefByTargetSpec(ref))
        return self

    def involving_verse(self, ref: str | VerseReference) -> "CrossRefSpecBuilder":
        if isinstance(ref, str):
            ref = VerseReference.parse(ref)
        self._specs.append(CrossRefBidirectionalSpec(ref))
        return self

    def of_type(self, connection_type: ConnectionTypeEnum) -> "CrossRefSpecBuilder":
        self._specs.append(CrossRefByTypeSpec(connection_type))
        return self

    def min_strength(self, strength: ConnectionStrength) -> "CrossRefSpecBuilder":
        self._specs.append(CrossRefByStrengthSpec(strength))
        return self

    def verified(self, verification_type: Optional[str] = None) -> "CrossRefSpecBuilder":
        self._specs.append(CrossRefVerifiedSpec(verification_type))
        return self

    def high_confidence(self, min_conf: float = 0.85) -> "CrossRefSpecBuilder":
        self._specs.append(CrossRefHighConfidenceSpec(min_conf))
        return self

    def typological(self) -> "CrossRefSpecBuilder":
        self._specs.append(CrossRefTypologicalSpec())
        return self

    def patristic_supported(self) -> "CrossRefSpecBuilder":
        self._specs.append(CrossRefPatristicSupportedSpec())
        return self

    def spans_testaments(self) -> "CrossRefSpecBuilder":
        self._specs.append(CrossRefSpansTestamentsSpec())
        return self

    def build(self) -> ISpecification[CrossReferenceAggregate]:
        """Build the composite specification."""
        if not self._specs:
            from db.interfaces import TrueSpecification
            return TrueSpecification()

        result = self._specs[0]
        for spec in self._specs[1:]:
            result = result.and_(spec)
        return result
