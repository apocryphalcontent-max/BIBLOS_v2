"""
Event Sourcing: Event Definitions for BIBLOS v2

All domain events that can occur in the system. Events are immutable
records of facts that have happened.

Seraphic Architecture - Events Know Their Own Nature:
    In the seraphic paradigm, events are not passive data structures that require
    external configuration. Each event type intrinsically KNOWS:
    - Its aggregate type (which domain concept it belongs to)
    - Its schema version and evolution path
    - Who subscribes to it (intrinsic listener affinity)
    - How it transforms to newer versions (upcasting DNA)

    The event store doesn't "configure" events - it perceives their nature
    through the SeraphicEventRegistry, where all event types ARE known simply
    by existing.

Usage:
    @event(aggregate="verse", version=1)
    @subscribes(ProjectionHandler, AnalyticsHandler)
    class VerseProcessingStarted(BaseEvent):
        verse_id: str = ""
        phase_plan: List[str] = field(default_factory=list)
"""
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Type, FrozenSet, Callable, TypeVar, Set
from enum import Enum
from uuid import UUID, uuid4
import threading

T = TypeVar("T")


# =============================================================================
# SERAPHIC EVENT INFRASTRUCTURE
# =============================================================================


@dataclass(frozen=True)
class UpcastSpec:
    """
    Specification for how an event transforms between schema versions.

    This is the event's evolutionary DNA - encoded knowledge of how
    it has changed and how to bring old instances into the present.
    """
    from_version: int
    to_version: int
    transform: Callable[[Dict[str, Any]], Dict[str, Any]]


@dataclass(frozen=True)
class SubscriptionAffinity:
    """
    A declared affinity between an event and its listeners.

    Events know who cares about them. Subscriptions are not external
    configuration but intrinsic relationships.
    """
    handler_type: Type
    handler_method: str = "handle"
    filter_predicate: Optional[Callable[[Any], bool]] = None


@dataclass
class EventAffinity:
    """
    The intrinsic nature of an event type.

    Every event type HAS an affinity - its essential characteristics.
    The affinity is not assigned externally but declared by the event
    itself through decorators or discovered through introspection.
    """
    event_class: Type
    event_type_name: str
    aggregate_type: str
    schema_version: int = 1
    upcasters: FrozenSet[UpcastSpec] = field(default_factory=frozenset)
    subscribers: FrozenSet[SubscriptionAffinity] = field(default_factory=frozenset)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def with_upcaster(self, spec: UpcastSpec) -> "EventAffinity":
        """Add an upcaster to this event's evolution chain."""
        return EventAffinity(
            event_class=self.event_class,
            event_type_name=self.event_type_name,
            aggregate_type=self.aggregate_type,
            schema_version=max(self.schema_version, spec.to_version),
            upcasters=self.upcasters | frozenset([spec]),
            subscribers=self.subscribers,
            metadata=self.metadata,
        )

    def with_subscriber(self, affinity: SubscriptionAffinity) -> "EventAffinity":
        """Add a subscriber to this event's listener set."""
        return EventAffinity(
            event_class=self.event_class,
            event_type_name=self.event_type_name,
            aggregate_type=self.aggregate_type,
            schema_version=self.schema_version,
            upcasters=self.upcasters,
            subscribers=self.subscribers | frozenset([affinity]),
            metadata=self.metadata,
        )


class SeraphicEventRegistry:
    """
    The Well of Event Memory.

    All event types exist here - not because they were "registered" but
    because they ARE. The registry doesn't manage events; it provides
    the space where events and their listeners discover each other.

    This is a singleton that holds the collective knowledge of all
    event types in the system.
    """
    _instance: Optional["SeraphicEventRegistry"] = None
    _lock: threading.Lock = threading.Lock()

    # Class-level type declarations for singleton attributes
    _affinities: Dict[Type, EventAffinity]
    _type_name_to_class: Dict[str, Type]
    _aggregate_to_events: Dict[str, Set[Type]]
    _subscriber_to_events: Dict[Type, Set[Type]]

    def __new__(cls) -> "SeraphicEventRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._affinities = {}
                    instance._type_name_to_class = {}
                    instance._aggregate_to_events = {}
                    instance._subscriber_to_events = {}
                    cls._instance = instance
        return cls._instance

    @classmethod
    def get_instance(cls) -> "SeraphicEventRegistry":
        """Get the singleton instance."""
        return cls()

    def register_affinity(self, affinity: EventAffinity) -> None:
        """
        Register an event's intrinsic nature.

        This is called automatically by the @event decorator - events
        declare their nature, and the registry simply acknowledges it.
        """
        self._affinities[affinity.event_class] = affinity
        self._type_name_to_class[affinity.event_type_name] = affinity.event_class

        # Index by aggregate
        if affinity.aggregate_type not in self._aggregate_to_events:
            self._aggregate_to_events[affinity.aggregate_type] = set()
        self._aggregate_to_events[affinity.aggregate_type].add(affinity.event_class)

        # Index by subscriber
        for sub in affinity.subscribers:
            if sub.handler_type not in self._subscriber_to_events:
                self._subscriber_to_events[sub.handler_type] = set()
            self._subscriber_to_events[sub.handler_type].add(affinity.event_class)

    def get_affinity(self, event_class: Type) -> Optional[EventAffinity]:
        """Get the intrinsic nature of an event type."""
        return self._affinities.get(event_class)

    def get_event_class(self, type_name: str) -> Optional[Type]:
        """Find event class by its type name."""
        return self._type_name_to_class.get(type_name)

    def get_events_for_aggregate(self, aggregate_type: str) -> Set[Type]:
        """Get all events that belong to an aggregate type."""
        return self._aggregate_to_events.get(aggregate_type, set())

    def get_events_for_subscriber(self, handler_type: Type) -> Set[Type]:
        """Get all events that a handler type subscribes to."""
        return self._subscriber_to_events.get(handler_type, set())

    def get_upcaster_chain(self, event_class: Type, from_version: int) -> List[UpcastSpec]:
        """
        Get the chain of upcasters needed to bring an event to current version.

        Returns an ordered list of transformations to apply.
        """
        affinity = self._affinities.get(event_class)
        if not affinity:
            return []

        chain: List[UpcastSpec] = []
        current_version = from_version
        upcasters_by_from = {u.from_version: u for u in affinity.upcasters}

        while current_version < affinity.schema_version:
            upcaster = upcasters_by_from.get(current_version)
            if not upcaster:
                break  # Gap in chain
            chain.append(upcaster)
            current_version = upcaster.to_version

        return chain

    def introspect(self) -> Dict[str, Any]:
        """
        Reveal the registry's current state.

        Introspection is a seraphic capability - the system knows itself.
        """
        return {
            "event_count": len(self._affinities),
            "aggregate_types": list(self._aggregate_to_events.keys()),
            "events": {
                name: {
                    "aggregate": aff.aggregate_type,
                    "version": aff.schema_version,
                    "subscriber_count": len(aff.subscribers),
                }
                for cls, aff in self._affinities.items()
                for name in [aff.event_type_name]
            },
        }


# =============================================================================
# SERAPHIC DECORATORS - Events Declare Their Nature
# =============================================================================


def event(
    aggregate: str,
    version: int = 1,
    **metadata: Any,
) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator for an event to declare its intrinsic nature.

    Usage:
        @event(aggregate="verse", version=1)
        class VerseProcessingStarted(BaseEvent):
            verse_id: str = ""

    The event now KNOWS:
    - Its aggregate type ("verse")
    - Its schema version (1)
    - Any additional metadata

    This information is not external configuration - it IS the event.
    """
    def decorator(cls: Type[T]) -> Type[T]:
        # Infer event type name from class
        event_type_name = getattr(cls, "event_type", None)
        if event_type_name is None:
            # Convert CamelCase to event type name
            event_type_name = cls.__name__

        affinity = EventAffinity(
            event_class=cls,
            event_type_name=event_type_name,
            aggregate_type=aggregate,
            schema_version=version,
            metadata=metadata,
        )

        # Register with the seraphic registry
        SeraphicEventRegistry.get_instance().register_affinity(affinity)

        # Attach affinity to the class for introspection
        cls._seraphic_affinity = affinity  # type: ignore

        return cls

    return decorator


def subscribes(*handlers: Type, method: str = "handle") -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to declare which handlers subscribe to this event.

    Usage:
        @event(aggregate="verse", version=1)
        @subscribes(GoldenRecordProjection, AnalyticsHandler)
        class VerseProcessingCompleted(BaseEvent):
            ...

    The event now knows who cares about it.
    """
    def decorator(cls: Type[T]) -> Type[T]:
        registry = SeraphicEventRegistry.get_instance()
        affinity = registry.get_affinity(cls)

        if affinity:
            for handler_type in handlers:
                subscription = SubscriptionAffinity(
                    handler_type=handler_type,
                    handler_method=method,
                )
                affinity = affinity.with_subscriber(subscription)
            registry.register_affinity(affinity)

        return cls

    return decorator


def upcasts_from(
    from_version: int,
    to_version: int,
    transform: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to declare how an event evolves between schema versions.

    Usage:
        @event(aggregate="verse", version=2)
        @upcasts_from(1, 2, lambda d: {**d, "new_field": d.get("old_field", "default")})
        class VerseProcessingStarted(BaseEvent):
            ...

    The event now carries its own evolutionary history.
    """
    def decorator(cls: Type[T]) -> Type[T]:
        registry = SeraphicEventRegistry.get_instance()
        affinity = registry.get_affinity(cls)

        if affinity:
            spec = UpcastSpec(
                from_version=from_version,
                to_version=to_version,
                transform=transform,
            )
            affinity = affinity.with_upcaster(spec)
            registry.register_affinity(affinity)

        return cls

    return decorator


def self_aware_event(cls: Type[T]) -> Type[T]:
    """
    Mark an event class as fully self-aware with introspection capabilities.

    Adds methods for the event to reveal its own nature.
    """
    def get_affinity(self) -> Optional[EventAffinity]:
        """Get this event's intrinsic affinity."""
        return SeraphicEventRegistry.get_instance().get_affinity(type(self))

    def get_aggregate_type(self) -> str:
        """Get the aggregate type this event belongs to."""
        affinity = self.get_affinity()
        return affinity.aggregate_type if affinity else "unknown"

    def get_schema_version(self) -> int:
        """Get the current schema version of this event type."""
        affinity = self.get_affinity()
        return affinity.schema_version if affinity else 1

    cls.get_affinity = get_affinity  # type: ignore
    cls.get_aggregate_type = get_aggregate_type  # type: ignore
    cls.get_schema_version = get_schema_version  # type: ignore

    return cls


def _utcnow() -> datetime:
    """Get current UTC time with timezone awareness."""
    return datetime.now(timezone.utc)


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

    Seraphic Awareness:
        In the seraphic architecture, events are not passive data structures.
        Each event instance can perceive its own nature through the registry.
        The event knows its aggregate type, its schema version, and who
        subscribes to it - this knowledge is intrinsic, not configured.
    """
    event_id: UUID = field(default_factory=uuid4)
    event_type: str = field(init=False)
    timestamp: datetime = field(default_factory=_utcnow)
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
        if 'event_id' in data:
            data['event_id'] = UUID(data['event_id']) if isinstance(data['event_id'], str) else data['event_id']
        if 'timestamp' in data:
            data['timestamp'] = datetime.fromisoformat(data['timestamp']) if isinstance(data['timestamp'], str) else data['timestamp']
        # Filter to only include fields that are init=True in the dataclass
        init_fields = {
            name for name, field_obj in cls.__dataclass_fields__.items()
            if field_obj.init
        }
        return cls(**{k: v for k, v in data.items() if k in init_fields})

    # ==========================================================================
    # SERAPHIC INTROSPECTION - The event knows itself
    # ==========================================================================

    def get_affinity(self) -> Optional[EventAffinity]:
        """
        Get this event's intrinsic affinity from the seraphic registry.

        The event perceives its own nature - aggregate type, schema version,
        subscribers - through the collective memory of all events.
        """
        return SeraphicEventRegistry.get_instance().get_affinity(type(self))

    def get_aggregate_type(self) -> str:
        """
        Get the aggregate type this event belongs to.

        Returns the aggregate type from the seraphic registry if available,
        otherwise infers it from the event_type field.
        """
        affinity = self.get_affinity()
        if affinity:
            return affinity.aggregate_type
        # Fallback: infer from event_type
        return self.event_type.split("_")[0].lower() if self.event_type else "unknown"

    def get_schema_version(self) -> int:
        """Get the current schema version of this event type."""
        affinity = self.get_affinity()
        return affinity.schema_version if affinity else 1

    def get_subscribers(self) -> FrozenSet[SubscriptionAffinity]:
        """Get the handlers that subscribe to this event type."""
        affinity = self.get_affinity()
        return affinity.subscribers if affinity else frozenset()


# ==================== Verse Processing Events ====================
# Events declare their aggregate type intrinsically - the store perceives this.

@event(aggregate="verse", version=1)
@dataclass(frozen=True)
class VerseProcessingStarted(BaseEvent):
    """Event emitted when verse processing begins."""
    event_type: str = field(default=EventType.VERSE_PROCESSING_STARTED.value, init=False)
    verse_id: str = ""
    phase_plan: List[str] = field(default_factory=list)
    user_id: Optional[str] = None


@event(aggregate="verse", version=1)
@dataclass(frozen=True)
class VerseProcessingCompleted(BaseEvent):
    """Event emitted when verse processing completes successfully."""
    event_type: str = field(default=EventType.VERSE_PROCESSING_COMPLETED.value, init=False)
    verse_id: str = ""
    phases_completed: List[str] = field(default_factory=list)
    total_duration_ms: float = 0.0
    cross_reference_count: int = 0
    quality_tier: str = ""  # gold, silver, bronze


@event(aggregate="verse", version=1)
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
# The SPIDERWEB's heartbeat - each connection discovery is witnessed.

@event(aggregate="cross_reference", version=1)
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


@event(aggregate="cross_reference", version=1)
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


@event(aggregate="cross_reference", version=1)
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
# The oracles of BIBLOS - deep analysis revelations.

@event(aggregate="oracle", version=1)
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


@event(aggregate="oracle", version=1)
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


@event(aggregate="textual", version=1)
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


@event(aggregate="typology", version=1)
@dataclass(frozen=True)
class TypologyDiscovered(BaseEvent):
    """Event emitted when typological connection is discovered."""
    event_type: str = field(default=EventType.TYPOLOGY_DISCOVERED.value, init=False)
    type_ref: str = ""
    antitype_ref: str = ""
    typology_layers: List[str] = field(default_factory=list)
    composite_strength: float = 0.0
    pattern_type: str = ""


@event(aggregate="prophecy", version=1)
@dataclass(frozen=True)
class PropheticProofComputed(BaseEvent):
    """Event emitted when prophetic proof is computed."""
    event_type: str = field(default=EventType.PROPHETIC_PROOF_COMPUTED.value, init=False)
    prophecy_id: str = ""
    natural_probability: float = 0.0
    posterior_supernatural: float = 0.0
    evidence_count: int = 0


# ==================== Phase Events ====================
# Pipeline phase lifecycle - the rhythm of processing.

@event(aggregate="pipeline", version=1)
@dataclass(frozen=True)
class PhaseStarted(BaseEvent):
    """Event emitted when a pipeline phase starts."""
    event_type: str = field(default=EventType.PHASE_STARTED.value, init=False)
    verse_id: str = ""
    phase_name: str = ""
    agent_count: int = 0


@event(aggregate="pipeline", version=1)
@dataclass(frozen=True)
class PhaseCompleted(BaseEvent):
    """Event emitted when a pipeline phase completes."""
    event_type: str = field(default=EventType.PHASE_COMPLETED.value, init=False)
    verse_id: str = ""
    phase_name: str = ""
    duration_ms: float = 0.0
    success_count: int = 0
    warning_count: int = 0


@event(aggregate="pipeline", version=1)
@dataclass(frozen=True)
class PhaseFailed(BaseEvent):
    """Event emitted when a pipeline phase fails."""
    event_type: str = field(default=EventType.PHASE_FAILED.value, init=False)
    verse_id: str = ""
    phase_name: str = ""
    error_message: str = ""
    failed_agents: List[str] = field(default_factory=list)


# ==================== Word Analysis Events ====================
# Linguistic analysis - the seraph perceives each word.

@event(aggregate="word", version=1)
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


@event(aggregate="word", version=1)
@dataclass(frozen=True)
class SemanticFieldMapped(BaseEvent):
    """Event emitted when semantic field is mapped."""
    event_type: str = field(default=EventType.SEMANTIC_FIELD_MAPPED.value, init=False)
    word: str = ""
    language: str = ""
    semantic_domains: List[str] = field(default_factory=list)
    primary_domain: str = ""
    confidence: float = 0.0


# ==================== Theological Guardrail Events ====================
# Anonymous wisdom encoded as rules - the Seraph's intrinsic constraints.
# No fathers named, no sources quoted - only logical principles distilled.

@event(aggregate="guardrail", version=1)
@dataclass(frozen=True)
class PatristicWitnessAdded(BaseEvent):
    """
    DEPRECATED: Use InterpretiveGuardrailApplied instead.
    Kept for backward compatibility - event represents an anonymous
    interpretive principle, not a named witness.
    """
    event_type: str = field(default=EventType.PATRISTIC_WITNESS_ADDED.value, init=False)
    verse_id: str = ""
    father_name: str = ""  # Should be empty - principle is anonymous
    authority_level: str = ""  # ecumenical, dogmatic, consensus
    interpretation: str = ""  # The principle itself, not a quote
    source_reference: str = ""  # Empty - no external attribution


@event(aggregate="guardrail", version=1)
@dataclass(frozen=True)
class ConsensusCalculated(BaseEvent):
    """
    Event emitted when interpretive consensus is calculated.
    Represents the distilled agreement of the tradition, not
    a count of named witnesses.
    """
    event_type: str = field(default=EventType.CONSENSUS_CALCULATED.value, init=False)
    verse_id: str = ""
    consensus_score: float = 0.0  # Strength of consensus (0.0-1.0)
    witness_count: int = 0  # Number of anonymous principles supporting
    dominant_interpretation: str = ""  # The objective interpretation


@event(aggregate="guardrail", version=1)
@dataclass(frozen=True)
class InterpretiveGuardrailApplied(BaseEvent):
    """
    Event emitted when a theological guardrail constrains interpretation.

    The Seraph, curled in its wings, bears intra-biblical rules that
    prevent heretical, lax, or progressive readings. These guardrails
    are the distilled wisdom of tradition - unnamed, attributed to no
    mortal, but encoded as objective logical constraints.
    """
    event_type: str = field(default="InterpretiveGuardrailApplied", init=False)
    verse_id: str = ""
    guardrail_category: str = ""  # christological, trinitarian, soteriological, etc.
    rule_applied: str = ""  # The logical principle enforced
    interpretation_before: str = ""  # What was proposed
    interpretation_after: str = ""  # What the guardrail produces
    confidence: float = 0.0  # How strongly the rule constrains
    rejected_heresy: Optional[str] = None  # If a heretical reading was blocked


@event(aggregate="guardrail", version=1)
@dataclass(frozen=True)
class HeresyRejected(BaseEvent):
    """
    Event emitted when an interpretation is rejected as heretical.

    The Seraph, from within its sphere of biblical isolation, declares
    what is NOT the case. This is anti-hallucination, anti-heresy,
    anti-progressivism in action - objective exclusion of falsehood.
    """
    event_type: str = field(default="HeresyRejected", init=False)
    verse_id: str = ""
    proposed_interpretation: str = ""  # What was rejected
    heresy_category: str = ""  # arianism, nestorianism, pelagianism, modernism, etc.
    rejection_reason: str = ""  # Why it contradicts intra-biblical truth
    severity: str = ""  # impossible, contrary, suspect
    corrective_principle: str = ""  # The right reading's foundation


# ==================== Constraint Events ====================
# Theological guardrails - protecting against heresy.

@event(aggregate="constraint", version=1)
@dataclass(frozen=True)
class TheologicalConstraintApplied(BaseEvent):
    """Event emitted when theological constraint is applied."""
    event_type: str = field(default=EventType.THEOLOGICAL_CONSTRAINT_APPLIED.value, init=False)
    cross_reference_id: str = ""
    constraint_name: str = ""
    passed: bool = False
    confidence_modifier: float = 1.0


@event(aggregate="constraint", version=1)
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

    Seraphic Awareness:
        The deserialization first consults the SeraphicEventRegistry -
        the collective memory of all event types. If not found there,
        falls back to the static EVENT_REGISTRY for backward compatibility.

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

    # First, consult the seraphic registry (intrinsic knowledge)
    registry = SeraphicEventRegistry.get_instance()
    event_class = registry.get_event_class(event_type)

    # Fall back to static registry for backward compatibility
    if not event_class:
        event_class = EVENT_REGISTRY.get(event_type)

    if not event_class:
        raise ValueError(f"Unknown event type: {event_type}")

    return event_class.from_dict(data)
