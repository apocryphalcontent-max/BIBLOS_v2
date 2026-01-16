"""
BIBLOS v2 - Data Schemas

Normalized schemas for all data types ensuring system-wide uniformity.
All data in the system should conform to these schemas.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from enum import Enum
from datetime import datetime, timezone
import json


# =============================================================================
# ENUMS - Standard values across the system
# =============================================================================

class Testament(str, Enum):
    """Testament designation."""
    OLD_TESTAMENT = "OT"
    NEW_TESTAMENT = "NT"


class Language(str, Enum):
    """Biblical languages."""
    HEBREW = "hebrew"
    ARAMAIC = "aramaic"
    GREEK = "greek"


class ConnectionType(str, Enum):
    """Cross-reference connection types."""
    THEMATIC = "thematic"
    VERBAL = "verbal"
    CONCEPTUAL = "conceptual"
    HISTORICAL = "historical"
    TYPOLOGICAL = "typological"
    PROPHETIC = "prophetic"
    LITURGICAL = "liturgical"
    NARRATIVE = "narrative"
    GENEALOGICAL = "genealogical"
    GEOGRAPHICAL = "geographical"


class ConnectionStrength(str, Enum):
    """Connection strength levels."""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


# Alias for backwards compatibility
Strength = ConnectionStrength
StrengthLevel = ConnectionStrength


class ProcessingStatus(str, Enum):
    """Processing status values."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"  # Alias: PROCESSING
    PROCESSING = "processing"    # Legacy alias
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    NEEDS_REVIEW = "needs_review"


class CertificationLevel(str, Enum):
    """Quality certification levels."""
    GOLD = "gold"
    SILVER = "silver"
    BRONZE = "bronze"
    PROVISIONAL = "provisional"


# =============================================================================
# BASE SCHEMA - Common fields
# =============================================================================

@dataclass
class BaseSchema:
    """Base schema with common metadata fields."""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    version: str = "2.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseSchema":
        """Create from dictionary."""
        return cls(**data)


# =============================================================================
# VERSE SCHEMA
# =============================================================================

@dataclass
class VerseSchema(BaseSchema):
    """
    Normalized schema for Bible verses.

    Example:
    {
        "verse_id": "GEN.1.1",
        "book": "GEN",
        "book_name": "Genesis",
        "chapter": 1,
        "verse": 1,
        "text": "In the beginning God created...",
        "original_text": "בְּרֵאשִׁית בָּרָא...",
        "testament": "OT",
        "language": "hebrew"
    }
    """
    verse_id: str = ""
    book: str = ""
    book_name: str = ""
    chapter: int = 0
    verse: int = 0
    text: str = ""
    original_text: str = ""
    testament: str = "OT"
    language: str = "hebrew"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> List[str]:
        """Validate the verse schema."""
        errors = []
        if not self.verse_id:
            errors.append("verse_id is required")
        if not self.book:
            errors.append("book is required")
        if self.chapter < 1:
            errors.append("chapter must be >= 1")
        if self.verse < 1:
            errors.append("verse must be >= 1")
        if self.testament not in ["OT", "NT"]:
            errors.append("testament must be OT or NT")
        return errors


# =============================================================================
# WORD SCHEMA
# =============================================================================

@dataclass
class MorphologySchema(BaseSchema):
    """
    Normalized schema for word morphology.

    Example:
    {
        "part_of_speech": "verb",
        "person": "3",
        "number": "singular",
        "gender": "masculine",
        "tense": "perfect",
        "raw_code": "HVqp3ms"
    }
    """
    part_of_speech: str = "unknown"
    person: Optional[str] = None
    number: Optional[str] = None
    gender: Optional[str] = None
    case: Optional[str] = None
    tense: Optional[str] = None
    voice: Optional[str] = None
    mood: Optional[str] = None
    stem: Optional[str] = None
    state: Optional[str] = None
    pattern: Optional[str] = None
    raw_code: str = ""


@dataclass
class WordSchema(BaseSchema):
    """
    Normalized schema for words within verses.

    Example:
    {
        "word_id": "GEN.1.1.1",
        "verse_id": "GEN.1.1",
        "surface_form": "בְּרֵאשִׁית",
        "lemma": "רֵאשִׁית",
        "position": 0,
        "morphology": {...}
    }
    """
    word_id: str = ""
    verse_id: str = ""
    surface_form: str = ""
    lemma: str = ""
    position: int = 0
    language: str = "hebrew"
    morphology: Dict[str, Any] = field(default_factory=dict)
    transliteration: str = ""
    gloss: str = ""
    strongs: str = ""
    syntax_role: str = ""
    clause_id: str = ""
    phrase_id: str = ""


# =============================================================================
# CROSS-REFERENCE SCHEMA
# =============================================================================

@dataclass
class CrossReferenceSchema(BaseSchema):
    """
    Normalized schema for cross-references.

    Example:
    {
        "source_ref": "GEN.1.1",
        "target_ref": "JHN.1.1",
        "connection_type": "typological",
        "strength": "strong",
        "confidence": 0.95,
        "notes": ["In the beginning - verbal parallel"],
        "sources": ["Treasury of Scripture Knowledge"],
        "verified": true
    }
    """
    source_ref: str = ""
    target_ref: str = ""
    connection_type: str = "thematic"
    strength: str = "moderate"
    confidence: float = 1.0
    bidirectional: bool = False
    notes: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    verified: bool = False
    patristic_support: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Mutual transformation metrics
    mutual_influence_score: float = 0.0  # Harmonic mean of bidirectional shift
    source_semantic_shift: float = 0.0   # How much source verse meaning changed
    target_semantic_shift: float = 0.0   # How much target verse meaning changed
    transformation_type: str = "MINIMAL" # RADICAL, MODERATE, MINIMAL

    def validate(self) -> List[str]:
        """Validate the cross-reference schema."""
        errors = []
        if not self.source_ref:
            errors.append("source_ref is required")
        if not self.target_ref:
            errors.append("target_ref is required")
        if self.connection_type not in [e.value for e in ConnectionType]:
            errors.append(f"Invalid connection_type: {self.connection_type}")
        if self.strength not in [e.value for e in ConnectionStrength]:
            errors.append(f"Invalid strength: {self.strength}")
        if not 0 <= self.confidence <= 1:
            errors.append("confidence must be between 0 and 1")
        return errors


# =============================================================================
# EXTRACTION RESULT SCHEMA
# =============================================================================

@dataclass
class ExtractionResultSchema(BaseSchema):
    """
    Normalized schema for agent extraction results.

    Example:
    {
        "agent_name": "grammateus",
        "extraction_type": "structural",
        "verse_id": "GEN.1.1",
        "status": "completed",
        "confidence": 0.95,
        "data": {...}
    }
    """
    agent_name: str = ""
    extraction_type: str = ""
    verse_id: str = ""
    status: str = "pending"
    confidence: float = 0.0
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    processing_time: float = 0.0


# =============================================================================
# GOLDEN RECORD SCHEMA
# =============================================================================

@dataclass
class GoldenRecordSchema(BaseSchema):
    """
    Normalized schema for golden records (final pipeline output).

    Example:
    {
        "verse_id": "GEN.1.1",
        "text": "In the beginning...",
        "certification": {
            "level": "gold",
            "score": 0.95
        },
        "data": {
            "structural": {...},
            "morphological": {...},
            "theological": {...},
            "cross_references": [...]
        }
    }
    """
    verse_id: str = ""
    text: str = ""
    certification: Dict[str, Any] = field(default_factory=lambda: {
        "level": "provisional",
        "score": 0.0,
        "validation_passed": False,
        "quality_passed": False
    })
    data: Dict[str, Any] = field(default_factory=dict)
    phases_executed: List[str] = field(default_factory=list)
    agent_count: int = 0
    total_processing_time: float = 0.0


# =============================================================================
# PATRISTIC TEXT SCHEMA (Full Works)
# =============================================================================

@dataclass
class PatristicTextSchema(BaseSchema):
    """
    Normalized schema for full patristic text documents.

    Represents complete works from Church Fathers, structured into
    logical sections for indexing and cross-referencing.

    Example:
    {
        "text_id": "damascenus-fide-orthodoxa",
        "author": "John of Damascus",
        "author_dates": "675-749",
        "title": "De Fide Orthodoxa",
        "title_english": "An Exact Exposition of the Orthodox Faith",
        "book": "1",
        "chapter": "1",
        "chapter_title": "That the Deity is incomprehensible",
        "content": "...",
        "language": "greek",
        "translation_language": "english",
        "source_collection": "NPNF-2",
        "verse_references": ["GEN.1.1", "JHN.1.1"]
    }
    """
    text_id: str = ""
    author: str = ""
    author_dates: str = ""
    title: str = ""
    title_english: str = ""
    book: str = ""
    chapter: str = ""
    chapter_title: str = ""
    section: str = ""
    content: str = ""
    content_clean: str = ""
    language: str = "greek"
    translation_language: str = "english"
    source_collection: str = ""
    source_file: str = ""
    verse_references: List[str] = field(default_factory=list)
    themes: List[str] = field(default_factory=list)
    footnotes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> List[str]:
        """Validate the patristic text schema."""
        errors = []
        if not self.author:
            errors.append("author is required")
        if not self.title:
            errors.append("title is required")
        if not self.content:
            errors.append("content is required")
        return errors

    def is_garbage(self) -> bool:
        """Check if content is likely garbage/metadata."""
        if len(self.content.strip()) < 50:
            return True
        garbage_markers = [
            "file:///",
            "http://",
            ".htm",
            ".pdf",
            "2006-06-01",
            "Index",
            "Table of Contents"
        ]
        lower_content = self.content.lower()
        return any(marker.lower() in lower_content for marker in garbage_markers)


# =============================================================================
# PATRISTIC CITATION SCHEMA
# =============================================================================

@dataclass
class PatristicCitationSchema(BaseSchema):
    """
    Normalized schema for patristic citations.

    Example:
    {
        "father": "Augustine",
        "work": "Confessions",
        "book": "11",
        "section": "14",
        "verse_refs": ["GEN.1.1"],
        "interpretation_type": "allegorical",
        "quote": "..."
    }
    """
    father: str = ""
    work: str = ""
    book: str = ""
    chapter: str = ""
    section: str = ""
    verse_refs: List[str] = field(default_factory=list)
    interpretation_type: str = "literal"
    quote: str = ""
    summary: str = ""
    themes: List[str] = field(default_factory=list)
    language: str = "latin"


# =============================================================================
# TYPOLOGICAL CONNECTION SCHEMA
# =============================================================================

@dataclass
class TypologicalConnectionSchema(BaseSchema):
    """
    Normalized schema for typological connections (OT type -> NT antitype).

    Example:
    {
        "type_ref": "GEN.22.2",
        "antitype_ref": "JHN.3.16",
        "type_category": "sacrifice",
        "description": "Isaac as type of Christ",
        "patristic_sources": ["Augustine", "Chrysostom"]
    }
    """
    type_ref: str = ""
    antitype_ref: str = ""
    type_category: str = ""
    description: str = ""
    confidence: float = 0.0
    patristic_sources: List[str] = field(default_factory=list)
    liturgical_connections: List[str] = field(default_factory=list)


# =============================================================================
# PIPELINE RESULT SCHEMA
# =============================================================================

@dataclass
class PhaseResultSchema(BaseSchema):
    """
    Normalized schema for pipeline phase results.

    Example:
    {
        "phase_name": "linguistic",
        "status": "completed",
        "agent_results": {...},
        "metrics": {...}
    }
    """
    phase_name: str = ""
    status: str = "pending"
    agent_results: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    start_time: float = 0.0
    end_time: float = 0.0
    error: Optional[str] = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class PipelineResultSchema(BaseSchema):
    """
    Normalized schema for complete pipeline results.

    Example:
    {
        "verse_id": "GEN.1.1",
        "status": "completed",
        "phase_results": {...},
        "golden_record": {...}
    }
    """
    verse_id: str = ""
    status: str = "pending"
    phase_results: Dict[str, PhaseResultSchema] = field(default_factory=dict)
    golden_record: Optional[GoldenRecordSchema] = None
    start_time: float = 0.0
    end_time: float = 0.0
    errors: List[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


# =============================================================================
# INFERENCE RESULT SCHEMA
# =============================================================================

@dataclass
class InferenceCandidateSchema(BaseSchema):
    """
    Normalized schema for inference candidates.

    Example:
    {
        "source_verse": "GEN.1.1",
        "target_verse": "JHN.1.1",
        "connection_type": "typological",
        "confidence": 0.85,
        "features": {...}
    }
    """
    source_verse: str = ""
    target_verse: str = ""
    connection_type: str = "thematic"
    confidence: float = 0.0
    embedding_similarity: float = 0.0
    semantic_similarity: float = 0.0
    features: Dict[str, float] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)


@dataclass
class InferenceResultSchema(BaseSchema):
    """
    Normalized schema for inference results.

    Example:
    {
        "verse_id": "GEN.1.1",
        "candidates": [...],
        "processing_time": 1.5
    }
    """
    verse_id: str = ""
    candidates: List[InferenceCandidateSchema] = field(default_factory=list)
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_verse_id(verse_id: str) -> bool:
    """Validate verse ID format."""
    if not verse_id or not isinstance(verse_id, str):
        return False
    parts = verse_id.upper().replace(" ", ".").replace(":", ".").split(".")
    if len(parts) < 3:
        return False
    try:
        int(parts[1])
        int(parts[2])
        return True
    except ValueError:
        return False


def normalize_verse_id(verse_id: str) -> str:
    """Normalize verse ID to standard format (BOOK.CHAPTER.VERSE)."""
    return verse_id.upper().replace(" ", ".").replace(":", ".")


def validate_connection_type(conn_type: str) -> bool:
    """Validate connection type."""
    return conn_type in [e.value for e in ConnectionType]


def validate_strength(strength: str) -> bool:
    """Validate connection strength."""
    return strength in [e.value for e in ConnectionStrength]


# =============================================================================
# SCHEMA REGISTRY
# =============================================================================

SCHEMA_REGISTRY = {
    "verse": VerseSchema,
    "word": WordSchema,
    "morphology": MorphologySchema,
    "cross_reference": CrossReferenceSchema,
    "extraction_result": ExtractionResultSchema,
    "golden_record": GoldenRecordSchema,
    "patristic_text": PatristicTextSchema,
    "patristic_citation": PatristicCitationSchema,
    "typological_connection": TypologicalConnectionSchema,
    "phase_result": PhaseResultSchema,
    "pipeline_result": PipelineResultSchema,
    "inference_candidate": InferenceCandidateSchema,
    "inference_result": InferenceResultSchema
}


def get_schema(schema_type: str) -> type:
    """Get schema class by type name."""
    return SCHEMA_REGISTRY.get(schema_type)


def create_from_dict(schema_type: str, data: Dict[str, Any]) -> BaseSchema:
    """Create schema instance from dictionary."""
    schema_class = get_schema(schema_type)
    if schema_class:
        return schema_class(**data)
    raise ValueError(f"Unknown schema type: {schema_type}")
