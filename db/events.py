"""
Event Sourcing: Event Definitions for BIBLOS v2

All domain events that can occur in the system. Events are immutable
records of facts that have happened.
"""
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum
from uuid import UUID, uuid4


class EventType(Enum):
    """Types of domain events in the system."""
    # Verse Processing Events
    VERSE_PROCESSING_STARTED = "VerseProcessingStarted"
    VERSE_PROCESSING_COMPLETED = "VerseProcessingCompleted"
    VERSE_PROCESSING_FAILED = "VerseProcessingFailed"

    # Cross-Reference Events
    CROSS_REFERENCE_DISCOVERED = "CrossReferenceDiscovered"
    CROSS_REFERENCE_VALIDATED = "CrossReferenceValidated"
    CROSS_REFERENCE_REJECTED = "CrossReferenceRejected"

    # Oracle Events
    OMNI_RESOLUTION_COMPUTED = "OmniResolutionComputed"
    NECESSITY_CALCULATED = "NecessityCalculated"
    LXX_DIVERGENCE_DETECTED = "LXXDivergenceDetected"
    TYPOLOGY_DISCOVERED = "TypologyDiscovered"
    PROPHETIC_PROOF_COMPUTED = "PropheticProofComputed"

    # Phase Events
    PHASE_STARTED = "PhaseStarted"
    PHASE_COMPLETED = "PhaseCompleted"
    PHASE_FAILED = "PhaseFailed"

    # Word Analysis Events
    WORD_ANALYZED = "WordAnalyzed"
    SEMANTIC_FIELD_MAPPED = "SemanticFieldMapped"

    # Patristic Events
    PATRISTIC_WITNESS_ADDED = "PatristicWitnessAdded"
    CONSENSUS_CALCULATED = "ConsensusCalculated"

    # Constraint Events
    THEOLOGICAL_CONSTRAINT_APPLIED = "TheologicalConstraintApplied"
    CONSTRAINT_VIOLATION_DETECTED = "ConstraintViolationDetected"


@dataclass(frozen=True)
class BaseEvent:
    """
    Base class for all events.

    Events are immutable and represent facts that have occurred.
    """
    event_id: UUID = field(default_factory=uuid4)
    event_type: str = field(init=False)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    aggregate_id: str = ""
    aggregate_version: int = 0
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        data = asdict(self)
        data['event_id'] = str(self.event_id)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseEvent':
        """Reconstruct event from dictionary."""
        data = data.copy()
        data['event_id'] = UUID(data['event_id'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ==================== Verse Processing Events ====================

@dataclass(frozen=True)
class VerseProcessingStarted(BaseEvent):
    """Event emitted when verse processing begins."""
    event_type: str = field(default=EventType.VERSE_PROCESSING_STARTED.value, init=False)
    verse_id: str = ""
    phase_plan: List[str] = field(default_factory=list)
    user_id: Optional[str] = None


@dataclass(frozen=True)
class VerseProcessingCompleted(BaseEvent):
    """Event emitted when verse processing completes successfully."""
    event_type: str = field(default=EventType.VERSE_PROCESSING_COMPLETED.value, init=False)
    verse_id: str = ""
    phases_completed: List[str] = field(default_factory=list)
    total_duration_ms: float = 0.0
    cross_reference_count: int = 0
    quality_tier: str = ""  # gold, silver, bronze


@dataclass(frozen=True)
class VerseProcessingFailed(BaseEvent):
    """Event emitted when verse processing fails."""
    event_type: str = field(default=EventType.VERSE_PROCESSING_FAILED.value, init=False)
    verse_id: str = ""
    failed_phase: str = ""
    error_message: str = ""
    error_type: str = ""
    retry_count: int = 0


# ==================== Cross-Reference Events ====================

@dataclass(frozen=True)
class CrossReferenceDiscovered(BaseEvent):
    """Event emitted when a cross-reference is discovered."""
    event_type: str = field(default=EventType.CROSS_REFERENCE_DISCOVERED.value, init=False)
    source_ref: str = ""
    target_ref: str = ""
    connection_type: str = ""
    confidence: float = 0.0
    discovered_by: str = ""  # Which component discovered it
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CrossReferenceValidated(BaseEvent):
    """Event emitted when a cross-reference passes validation."""
    event_type: str = field(default=EventType.CROSS_REFERENCE_VALIDATED.value, init=False)
    source_ref: str = ""
    target_ref: str = ""
    connection_type: str = ""
    final_confidence: float = 0.0
    validators: List[str] = field(default_factory=list)
    theological_score: float = 0.0


@dataclass(frozen=True)
class CrossReferenceRejected(BaseEvent):
    """Event emitted when a cross-reference is rejected."""
    event_type: str = field(default=EventType.CROSS_REFERENCE_REJECTED.value, init=False)
    source_ref: str = ""
    target_ref: str = ""
    connection_type: str = ""
    rejection_reason: str = ""
    violated_constraints: List[str] = field(default_factory=list)


# ==================== Oracle Events ====================

@dataclass(frozen=True)
class OmniResolutionComputed(BaseEvent):
    """Event emitted when OmniContextual resolution is computed."""
    event_type: str = field(default=EventType.OMNI_RESOLUTION_COMPUTED.value, init=False)
    verse_id: str = ""
    word: str = ""
    language: str = ""
    primary_meaning: str = ""
    total_occurrences: int = 0
    confidence: float = 0.0
    semantic_field_map: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class NecessityCalculated(BaseEvent):
    """Event emitted when inter-verse necessity is calculated."""
    event_type: str = field(default=EventType.NECESSITY_CALCULATED.value, init=False)
    verse_a: str = ""
    verse_b: str = ""
    necessity_score: float = 0.0
    necessity_strength: str = ""  # absolute, strong, moderate, weak, none
    semantic_gap_count: int = 0
    gap_severity: str = ""


@dataclass(frozen=True)
class LXXDivergenceDetected(BaseEvent):
    """Event emitted when LXX/MT divergence is detected."""
    event_type: str = field(default=EventType.LXX_DIVERGENCE_DETECTED.value, init=False)
    verse_id: str = ""
    lxx_text: str = ""
    mt_text: str = ""
    divergence_type: str = ""
    christological_significance: float = 0.0
    manuscript_witnesses: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class TypologyDiscovered(BaseEvent):
    """Event emitted when typological connection is discovered."""
    event_type: str = field(default=EventType.TYPOLOGY_DISCOVERED.value, init=False)
    type_ref: str = ""
    antitype_ref: str = ""
    typology_layers: List[str] = field(default_factory=list)
    composite_strength: float = 0.0
    pattern_type: str = ""


@dataclass(frozen=True)
class PropheticProofComputed(BaseEvent):
    """Event emitted when prophetic proof is computed."""
    event_type: str = field(default=EventType.PROPHETIC_PROOF_COMPUTED.value, init=False)
    prophecy_id: str = ""
    natural_probability: float = 0.0
    posterior_supernatural: float = 0.0
    evidence_count: int = 0


# ==================== Phase Events ====================

@dataclass(frozen=True)
class PhaseStarted(BaseEvent):
    """Event emitted when a pipeline phase starts."""
    event_type: str = field(default=EventType.PHASE_STARTED.value, init=False)
    verse_id: str = ""
    phase_name: str = ""
    agent_count: int = 0


@dataclass(frozen=True)
class PhaseCompleted(BaseEvent):
    """Event emitted when a pipeline phase completes."""
    event_type: str = field(default=EventType.PHASE_COMPLETED.value, init=False)
    verse_id: str = ""
    phase_name: str = ""
    duration_ms: float = 0.0
    success_count: int = 0
    warning_count: int = 0


@dataclass(frozen=True)
class PhaseFailed(BaseEvent):
    """Event emitted when a pipeline phase fails."""
    event_type: str = field(default=EventType.PHASE_FAILED.value, init=False)
    verse_id: str = ""
    phase_name: str = ""
    error_message: str = ""
    failed_agents: List[str] = field(default_factory=list)


# ==================== Word Analysis Events ====================

@dataclass(frozen=True)
class WordAnalyzed(BaseEvent):
    """Event emitted when a word is analyzed."""
    event_type: str = field(default=EventType.WORD_ANALYZED.value, init=False)
    verse_id: str = ""
    word_index: int = 0
    lemma: str = ""
    morph_code: str = ""
    part_of_speech: str = ""
    gloss: str = ""


@dataclass(frozen=True)
class SemanticFieldMapped(BaseEvent):
    """Event emitted when semantic field is mapped."""
    event_type: str = field(default=EventType.SEMANTIC_FIELD_MAPPED.value, init=False)
    word: str = ""
    language: str = ""
    semantic_domains: List[str] = field(default_factory=list)
    primary_domain: str = ""
    confidence: float = 0.0


# ==================== Patristic Events ====================

@dataclass(frozen=True)
class PatristicWitnessAdded(BaseEvent):
    """Event emitted when patristic witness is added."""
    event_type: str = field(default=EventType.PATRISTIC_WITNESS_ADDED.value, init=False)
    verse_id: str = ""
    father_name: str = ""
    authority_level: str = ""  # ecumenical, great, major, minor
    interpretation: str = ""
    source_reference: str = ""


@dataclass(frozen=True)
class ConsensusCalculated(BaseEvent):
    """Event emitted when patristic consensus is calculated."""
    event_type: str = field(default=EventType.CONSENSUS_CALCULATED.value, init=False)
    verse_id: str = ""
    consensus_score: float = 0.0
    witness_count: int = 0
    dominant_interpretation: str = ""


# ==================== Constraint Events ====================

@dataclass(frozen=True)
class TheologicalConstraintApplied(BaseEvent):
    """Event emitted when theological constraint is applied."""
    event_type: str = field(default=EventType.THEOLOGICAL_CONSTRAINT_APPLIED.value, init=False)
    cross_reference_id: str = ""
    constraint_name: str = ""
    passed: bool = False
    confidence_modifier: float = 1.0


@dataclass(frozen=True)
class ConstraintViolationDetected(BaseEvent):
    """Event emitted when constraint violation is detected."""
    event_type: str = field(default=EventType.CONSTRAINT_VIOLATION_DETECTED.value, init=False)
    cross_reference_id: str = ""
    constraint_name: str = ""
    violation_severity: str = ""  # impossible, critical, soft, warning
    explanation: str = ""


# Event type registry for deserialization
EVENT_REGISTRY: Dict[str, type] = {
    EventType.VERSE_PROCESSING_STARTED.value: VerseProcessingStarted,
    EventType.VERSE_PROCESSING_COMPLETED.value: VerseProcessingCompleted,
    EventType.VERSE_PROCESSING_FAILED.value: VerseProcessingFailed,
    EventType.CROSS_REFERENCE_DISCOVERED.value: CrossReferenceDiscovered,
    EventType.CROSS_REFERENCE_VALIDATED.value: CrossReferenceValidated,
    EventType.CROSS_REFERENCE_REJECTED.value: CrossReferenceRejected,
    EventType.OMNI_RESOLUTION_COMPUTED.value: OmniResolutionComputed,
    EventType.NECESSITY_CALCULATED.value: NecessityCalculated,
    EventType.LXX_DIVERGENCE_DETECTED.value: LXXDivergenceDetected,
    EventType.TYPOLOGY_DISCOVERED.value: TypologyDiscovered,
    EventType.PROPHETIC_PROOF_COMPUTED.value: PropheticProofComputed,
    EventType.PHASE_STARTED.value: PhaseStarted,
    EventType.PHASE_COMPLETED.value: PhaseCompleted,
    EventType.PHASE_FAILED.value: PhaseFailed,
    EventType.WORD_ANALYZED.value: WordAnalyzed,
    EventType.SEMANTIC_FIELD_MAPPED.value: SemanticFieldMapped,
    EventType.PATRISTIC_WITNESS_ADDED.value: PatristicWitnessAdded,
    EventType.CONSENSUS_CALCULATED.value: ConsensusCalculated,
    EventType.THEOLOGICAL_CONSTRAINT_APPLIED.value: TheologicalConstraintApplied,
    EventType.CONSTRAINT_VIOLATION_DETECTED.value: ConstraintViolationDetected,
}


def deserialize_event(data: Dict[str, Any]) -> BaseEvent:
    """
    Deserialize event from dictionary.

    Args:
        data: Event data dictionary

    Returns:
        Reconstructed event object

    Raises:
        ValueError: If event type is unknown
    """
    event_type = data.get('event_type')
    if not event_type:
        raise ValueError("Event data missing 'event_type' field")

    event_class = EVENT_REGISTRY.get(event_type)
    if not event_class:
        raise ValueError(f"Unknown event type: {event_type}")

    return event_class.from_dict(data)
