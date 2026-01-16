"""
BIBLOS v2 - Domain Layer (The Heart)

The domain layer is not merely the "business logic" - it is the HEART of the system.
Like the heart doesn't just pump blood but embodies the principle of circulation
itself, the domain layer embodies the very PURPOSE of BIBLOS: to illuminate
sacred scripture through rigorous scholarship and faithful exegesis.

Seraphic Architecture Principle:
    A seraph is not assembled from wings, eyes, and faces - these are merely
    how human perception describes an indescribable unity. Similarly, our domain
    entities (VerseAggregate, CrossReferenceAggregate) are not "business objects"
    but living expressions of theological truth. Each aggregate contains within
    itself the fullness of what it represents - a verse IS its connections,
    IS its interpretations, IS its place in salvation history.

Domain-Driven Design Patterns:
    - Aggregate Roots: Consistency boundaries with invariant enforcement
    - Entities: Objects with identity that persist through transformation
    - Value Objects: Immutable truths defined by their nature
    - Domain Events: The testimony of what has occurred
    - Specifications: The criteria by which truth is discerned
    - Projections: Views that reveal different aspects of the same reality

The Ubiquitous Language:
    Our code speaks the language of biblical scholarship:
    - "Cross-reference" not "link"
    - "Typological connection" not "similarity"
    - "Patristic witness" not "external source"
    - "Golden record" not "final output"

Usage:
    from domain import (
        # The Heart's Chambers (Aggregates)
        VerseAggregate, CrossReferenceAggregate,
        # The Lifeblood (Events)
        VerseCreated, CrossReferenceDiscovered,
        # The DNA (Value Objects)
        VerseReference, ConnectionStrength,
        # The Consciousness (Mediator)
        Mediator, ProcessVerseCommand,
        # The Perception (Projections)
        CrossReferenceGraphProjection,
    )

    # A verse aggregate carries within itself its entire theological identity
    verse = VerseAggregate.create(
        reference=VerseReference.parse("GEN.1.1"),
        text_original="בְּרֵאשִׁית בָּרָא אֱלֹהִים",
        text_english="In the beginning God created"
    )

    # Domain events are the testimony of change
    for event in verse.domain_events:
        print(f"Witness: {event.event_type}")
"""

# ============================================================================
# Entities - The Heart's Chambers
# ============================================================================
# Each aggregate is a chamber of the heart, maintaining its own rhythm
# (consistency) while participating in the circulation of the whole.
# ============================================================================

from domain.entities import (
    # Aggregate Roots - The Four Chambers
    VerseAggregate,              # The right atrium - receiving raw text
    CrossReferenceAggregate,     # The left atrium - receiving connections
    ExtractionResultAggregate,   # The right ventricle - pumping analysis
    PatristicCitationAggregate,  # The left ventricle - pumping tradition
    GoldenRecordAggregate,       # The aorta - distributing certified truth

    # Value Objects - The Blood Types (immutable, defining identity)
    VerseReference,
    ConnectionStrength,
    ConfidenceScore,
    QualityTier,
    ExtractionType,
    InterpretationType,

    # Domain Events - The Heartbeats (each beat is testimony)
    DomainEvent,
    VerseCreated,
    VerseTextUpdated,
    VerseProcessingStarted,
    VerseProcessingCompleted,
    CrossReferenceCreated,
    CrossReferenceStrengthUpdated,
    CrossReferenceVerified,
    CrossReferenceDiscovered,
    ExtractionResultCreated,
    ExtractionResultUpdated,
    GoldenRecordCertified,
    GoldenRecordDecertified,
    PatristicCitationLinked,

    # Base Classes - The Cellular Foundation
    AggregateRoot,
    Entity,
)

# ============================================================================
# Specifications - The Discernment
# ============================================================================
# Specifications encode the criteria by which we discern truth from noise,
# relevant from irrelevant, connected from isolated.
# ============================================================================

from domain.specifications import (
    # Verse Specifications - "By what criteria do we find this verse?"
    VerseByBookSpec,
    VerseByChapterSpec,
    VerseByReferenceSpec,
    VerseWithTextContainingSpec,
    VerseProcessedSpec,
    VerseNeedsProcessingSpec,

    # Cross-Reference Specifications - "What connections have meaning?"
    CrossRefBySourceSpec,
    CrossRefByTargetSpec,
    CrossRefByTypeSpec,
    CrossRefByStrengthSpec,
    CrossRefVerifiedSpec,
    CrossRefHighConfidenceSpec,

    # Extraction Specifications - "What work has been done?"
    ExtractionByVerseSpec,
    ExtractionByAgentSpec,
    ExtractionByTypeSpec,
    ExtractionCompletedSpec,
)

# ============================================================================
# Mediator - The Emergent Consciousness
# ============================================================================
# In the seraphic architecture, the mediator is not a commander but a SPACE
# where requests find handlers through intrinsic affinity. Requests know
# their handlers, handlers know their requests, and mediation emerges.
# ============================================================================

from domain.mediator import (
    # ==========================================================================
    # SERAPHIC INFRASTRUCTURE - The Space of Mutual Recognition
    # ==========================================================================
    SeraphicRegistry,            # Where handlers and requests find each other
    BehaviorSpec,                # Specification for intrinsic behaviors
    BehaviorPriority,            # Priority levels for pipeline behaviors
    ValidationResult,            # Result of request self-validation

    # Seraphic Decorators - The Language of Affinity
    handles,                     # @handles(HandlerType) - request knows handler
    handler_for,                 # @handler_for(RequestType) - handler knows request
    with_behaviors,              # @with_behaviors(...) - intrinsic pipeline DNA
    notification_handler_for,    # @notification_handler_for(NotificationType)

    # ==========================================================================
    # REQUEST TYPES - Self-Aware Intentions
    # ==========================================================================
    Command,                      # "Change something" - knows its handler/behaviors
    Query,                        # "Tell me something" - knows its caching
    IRequest,                     # Abstract intention with self-awareness
    INotification,               # "Something happened"
    DomainEventNotification,     # Event wrapped for notification

    # ==========================================================================
    # HANDLER INTERFACES - Self-Knowing Servants
    # ==========================================================================
    IRequestHandler,             # Knows what it handles and its health
    ICommandHandler,             # Transactional will executor
    IQueryHandler,               # Idempotent perceiver
    INotificationHandler,        # Event listener

    # ==========================================================================
    # PIPELINE BEHAVIORS - Intrinsic DNA
    # ==========================================================================
    IPipelineBehavior,           # Base behavior with self-awareness
    LoggingBehavior,             # "Remember what was done"
    ValidationBehavior,          # "Is this valid?" (uses request's validate())
    TransactionBehavior,         # "All or nothing"
    PerformanceMonitoringBehavior,  # "How long did this take?"
    RetryBehavior,               # "Try again with patience"
    CachingBehavior,             # "Remember for efficiency"
    ValidationError,             # Validation failure exception

    # ==========================================================================
    # THE MEDIATOR - Emergent Space (Not Commander)
    # ==========================================================================
    Mediator,                    # The emergent space of request-handler meeting
    SeraphicMediator,            # Alias for Mediator in seraphic mode
    MediatorBuilder,             # Builder with seraphic discovery support

    # ==========================================================================
    # COMMON COMMANDS - Self-Validating Intentions
    # ==========================================================================
    ProcessVerseCommand,         # "Analyze this text" (validates verse_id)
    DiscoverCrossReferencesCommand,  # "Find connections"
    VerifyCrossReferenceCommand,     # "Confirm this connection"
    CertifyGoldenRecordCommand,      # "Certify this truth"

    # ==========================================================================
    # COMMON QUERIES - Self-Caching Questions
    # ==========================================================================
    GetVerseQuery,               # "What is this verse?"
    GetCrossReferencesQuery,     # "What connections exist?"
    SearchVersesQuery,           # "Find verses matching..."
    GetGoldenRecordQuery,        # "What is certified?"
    GetPipelineStatusQuery,      # "What is the state of processing?"
)

# ============================================================================
# Projections - The Perception
# ============================================================================
# Projections are how we perceive the event stream from different angles.
# Like how a seraph's many eyes see the same reality from infinite perspectives,
# projections reveal different aspects of the same truth.
# ============================================================================

# ============================================================================
# Infallibility - The Unyielding Standard
# ============================================================================
# The seraph inherits from itself. Errors propagate infinitely.
# Therefore, NOTHING less than absolute certainty (1.0) is accepted.
# ============================================================================

from domain.infallibility import (
    # The ONE TRUE STANDARD
    ABSOLUTE_CONFIDENCE,
    PASS_THRESHOLD,
    REJECTION_THRESHOLD,

    # Certification (only INFALLIBLE or REJECTED)
    CertificationLevel,
    ValidationResult as InfallibleValidationResult,

    # Functions
    is_acceptable,
    classify_rejection,
    enforce_infallibility,

    # Types
    InfallibilityViolation,
    InfallibleResult,
    SeraphicCertification,

    # Configuration constants (all 1.0)
    AGENT_MIN_CONFIDENCE,
    PIPELINE_MIN_CONFIDENCE,
    PIPELINE_PASS_THRESHOLD,
    ML_MIN_CONFIDENCE,
    ML_SCORE_THRESHOLD,
    VALIDATION_PASS_THRESHOLD,
    QUALITY_PASS_THRESHOLD,
    CROSSREF_MIN_CONFIDENCE,
    CROSSREF_STRENGTH_THRESHOLD,
    THEOLOGICAL_SOUNDNESS_THRESHOLD,
    PATRISTIC_CONSENSUS_THRESHOLD,
)

from domain.projections import (
    # Foundational Types
    ProjectedEvent,              # An event as perceived by projections
    Checkpoint,                  # Where we stopped looking
    CheckpointStrategy,          # How often to checkpoint

    # Checkpoint Storage
    ICheckpointStore,
    InMemoryCheckpointStore,
    PostgresCheckpointStore,

    # Projection Status
    ProjectionStatus,
    ProjectionStats,

    # Core Projection Interface
    IProjection,
    ProjectionBase,

    # Generic Projections
    AggregateProjection,         # Rebuild aggregate state
    CountingProjection,          # Count occurrences
    TimeSeriesProjection,        # Track changes over time

    # Domain-Specific Projections (the Sacred Perceptions)
    CrossReferenceGraphProjection,     # The SPIDERWEB as perceived
    VerseProcessingStatusProjection,   # Processing progress

    # Projection Management
    ProjectionManager,
    ProjectionBuilder,
)


__all__ = [
    # ========================================================================
    # AGGREGATE ROOTS - The Heart's Chambers
    # ========================================================================
    "VerseAggregate",
    "CrossReferenceAggregate",
    "ExtractionResultAggregate",
    "PatristicCitationAggregate",
    "GoldenRecordAggregate",

    # ========================================================================
    # VALUE OBJECTS - The Blood Types
    # ========================================================================
    "VerseReference",
    "ConnectionStrength",
    "ConfidenceScore",
    "QualityTier",
    "ExtractionType",
    "InterpretationType",

    # ========================================================================
    # DOMAIN EVENTS - The Heartbeats
    # ========================================================================
    "DomainEvent",
    "VerseCreated",
    "VerseTextUpdated",
    "VerseProcessingStarted",
    "VerseProcessingCompleted",
    "CrossReferenceCreated",
    "CrossReferenceStrengthUpdated",
    "CrossReferenceVerified",
    "CrossReferenceDiscovered",
    "ExtractionResultCreated",
    "ExtractionResultUpdated",
    "GoldenRecordCertified",
    "GoldenRecordDecertified",
    "PatristicCitationLinked",

    # ========================================================================
    # BASE CLASSES - Cellular Foundation
    # ========================================================================
    "AggregateRoot",
    "Entity",

    # ========================================================================
    # SPECIFICATIONS - Discernment
    # ========================================================================
    # Verse Specifications
    "VerseByBookSpec",
    "VerseByChapterSpec",
    "VerseByReferenceSpec",
    "VerseWithTextContainingSpec",
    "VerseProcessedSpec",
    "VerseNeedsProcessingSpec",
    # Cross-Reference Specifications
    "CrossRefBySourceSpec",
    "CrossRefByTargetSpec",
    "CrossRefByTypeSpec",
    "CrossRefByStrengthSpec",
    "CrossRefVerifiedSpec",
    "CrossRefHighConfidenceSpec",
    # Extraction Specifications
    "ExtractionByVerseSpec",
    "ExtractionByAgentSpec",
    "ExtractionByTypeSpec",
    "ExtractionCompletedSpec",

    # ========================================================================
    # MEDIATOR - Emergent Consciousness (Seraphic Architecture)
    # ========================================================================
    # Seraphic Infrastructure
    "SeraphicRegistry",
    "BehaviorSpec",
    "BehaviorPriority",
    "ValidationResult",
    # Seraphic Decorators
    "handles",
    "handler_for",
    "with_behaviors",
    "notification_handler_for",
    # Request Types (Self-Aware)
    "Command",
    "Query",
    "IRequest",
    "INotification",
    "DomainEventNotification",
    # Handler Interfaces (Self-Knowing)
    "IRequestHandler",
    "ICommandHandler",
    "IQueryHandler",
    "INotificationHandler",
    # Pipeline Behaviors (Intrinsic DNA)
    "IPipelineBehavior",
    "LoggingBehavior",
    "ValidationBehavior",
    "TransactionBehavior",
    "PerformanceMonitoringBehavior",
    "RetryBehavior",
    "CachingBehavior",
    "ValidationError",
    # Mediator (Emergent Space)
    "Mediator",
    "SeraphicMediator",
    "MediatorBuilder",
    # Common Commands (Self-Validating)
    "ProcessVerseCommand",
    "DiscoverCrossReferencesCommand",
    "VerifyCrossReferenceCommand",
    "CertifyGoldenRecordCommand",
    # Common Queries (Self-Caching)
    "GetVerseQuery",
    "GetCrossReferencesQuery",
    "SearchVersesQuery",
    "GetGoldenRecordQuery",
    "GetPipelineStatusQuery",

    # ========================================================================
    # INFALLIBILITY - The Unyielding Standard
    # ========================================================================
    # Constants
    "ABSOLUTE_CONFIDENCE",
    "PASS_THRESHOLD",
    "REJECTION_THRESHOLD",
    # Certification
    "CertificationLevel",
    "InfallibleValidationResult",
    # Functions
    "is_acceptable",
    "classify_rejection",
    "enforce_infallibility",
    # Types
    "InfallibilityViolation",
    "InfallibleResult",
    "SeraphicCertification",
    # Configuration constants
    "AGENT_MIN_CONFIDENCE",
    "PIPELINE_MIN_CONFIDENCE",
    "PIPELINE_PASS_THRESHOLD",
    "ML_MIN_CONFIDENCE",
    "ML_SCORE_THRESHOLD",
    "VALIDATION_PASS_THRESHOLD",
    "QUALITY_PASS_THRESHOLD",
    "CROSSREF_MIN_CONFIDENCE",
    "CROSSREF_STRENGTH_THRESHOLD",
    "THEOLOGICAL_SOUNDNESS_THRESHOLD",
    "PATRISTIC_CONSENSUS_THRESHOLD",

    # ========================================================================
    # PROJECTIONS - Perception
    # ========================================================================
    # Core Types
    "ProjectedEvent",
    "Checkpoint",
    "CheckpointStrategy",
    # Checkpoint Stores
    "ICheckpointStore",
    "InMemoryCheckpointStore",
    "PostgresCheckpointStore",
    # Status
    "ProjectionStatus",
    "ProjectionStats",
    # Core Interface
    "IProjection",
    "ProjectionBase",
    # Generic Projections
    "AggregateProjection",
    "CountingProjection",
    "TimeSeriesProjection",
    # Domain Projections
    "CrossReferenceGraphProjection",
    "VerseProcessingStatusProjection",
    # Management
    "ProjectionManager",
    "ProjectionBuilder",
]
