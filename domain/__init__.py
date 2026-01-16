"""
BIBLOS v2 - Domain Layer

The domain layer represents the heart of the system - the vital organs that
process, transform, and maintain the integrity of biblical scholarship data.

This layer implements Domain-Driven Design patterns:
    - Aggregate Roots: Consistency boundaries with invariant enforcement
    - Entities: Objects with identity that persist over time
    - Value Objects: Immutable objects defined by their attributes
    - Domain Events: Record of significant occurrences in the domain
    - Domain Services: Operations that don't belong to any entity

Architectural Principles:
    - Rich Domain Model: Business logic lives in the domain, not services
    - Ubiquitous Language: Code reflects the language of biblical scholarship
    - Aggregate Boundaries: Each aggregate maintains its own consistency
    - Event-Driven: All significant changes emit domain events

Usage:
    from domain import (
        VerseAggregate, CrossReferenceAggregate,
        VerseCreated, CrossReferenceDiscovered,
        VerseReference, ConnectionStrength
    )

    # Create a new verse aggregate
    verse = VerseAggregate.create(
        reference=VerseReference.parse("GEN.1.1"),
        text_original="בְּרֵאשִׁית בָּרָא אֱלֹהִים",
        text_english="In the beginning God created"
    )

    # Check domain events
    for event in verse.domain_events:
        print(f"Event: {event.event_type}")
"""
from domain.entities import (
    # Aggregate Roots
    VerseAggregate,
    CrossReferenceAggregate,
    ExtractionResultAggregate,
    PatristicCitationAggregate,
    GoldenRecordAggregate,
    # Value Objects
    VerseReference,
    ConnectionStrength,
    ConfidenceScore,
    QualityTier,
    ExtractionType,
    InterpretationType,
    # Domain Events
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
    # Base Classes
    AggregateRoot,
    Entity,
)
from domain.specifications import (
    # Verse Specifications
    VerseByBookSpec,
    VerseByChapterSpec,
    VerseByReferenceSpec,
    VerseWithTextContainingSpec,
    VerseProcessedSpec,
    VerseNeedsProcessingSpec,
    # Cross-Reference Specifications
    CrossRefBySourceSpec,
    CrossRefByTargetSpec,
    CrossRefByTypeSpec,
    CrossRefByStrengthSpec,
    CrossRefVerifiedSpec,
    CrossRefHighConfidenceSpec,
    # Extraction Specifications
    ExtractionByVerseSpec,
    ExtractionByAgentSpec,
    ExtractionByTypeSpec,
    ExtractionCompletedSpec,
)
from domain.mediator import (
    # Base Types
    Command,
    Query,
    IRequest,
    INotification,
    DomainEventNotification,
    # Handler Interfaces
    IRequestHandler,
    ICommandHandler,
    IQueryHandler,
    INotificationHandler,
    # Pipeline Behaviors
    IPipelineBehavior,
    LoggingBehavior,
    ValidationBehavior,
    TransactionBehavior,
    PerformanceMonitoringBehavior,
    RetryBehavior,
    CachingBehavior,
    ValidationError,
    # Mediator
    Mediator,
    MediatorBuilder,
    # Common Commands
    ProcessVerseCommand,
    DiscoverCrossReferencesCommand,
    VerifyCrossReferenceCommand,
    CertifyGoldenRecordCommand,
    # Common Queries
    GetVerseQuery,
    GetCrossReferencesQuery,
    SearchVersesQuery,
    GetGoldenRecordQuery,
    GetPipelineStatusQuery,
)

__all__ = [
    # Aggregate Roots
    "VerseAggregate",
    "CrossReferenceAggregate",
    "ExtractionResultAggregate",
    "PatristicCitationAggregate",
    "GoldenRecordAggregate",
    # Value Objects
    "VerseReference",
    "ConnectionStrength",
    "ConfidenceScore",
    "QualityTier",
    "ExtractionType",
    "InterpretationType",
    # Domain Events
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
    # Base Classes
    "AggregateRoot",
    "Entity",
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
]
