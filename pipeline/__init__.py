"""
BIBLOS v2 - Pipeline Module

Pipeline phases for SDES extraction workflow:
- LinguisticPhase: Linguistic analysis (morphology, syntax, semantics)
- TheologicalPhase: Theological analysis (patristics, typology)
- IntertextualPhase: Cross-reference and connection analysis
- ValidationPhase: Quality validation and harmonization
- FinalizationPhase: Golden record creation and export

Stream-based components for event-driven architecture:
- EventBus: Redis Streams event bus for decoupled communication
- StreamOrchestrator: Event-driven pipeline orchestrator
- StreamConsumer: Base class for stream-consuming phases
- RecoveryService: Dead letter queue and checkpoint management
"""

# Core pipeline components
from pipeline.base import (
    BasePipelinePhase,
    PhaseConfig,
    PhaseResult,
    PhaseStatus,
    PipelineContext,
)
from pipeline.linguistic import LinguisticPhase
from pipeline.theological import TheologicalPhase
from pipeline.intertextual import IntertextualPhase
from pipeline.validation import ValidationPhase
from pipeline.finalization import FinalizationPhase
from pipeline.orchestrator import PipelineOrchestrator, PipelineConfig, PipelineResult

# Stream-based event-driven components
from pipeline.event_bus import (
    EventBus,
    EventBusConfig,
    EventMessage,
    StreamTopic,
    TOPICS,
    PHASE_TOPICS,
    VerseEvent,
    PhaseRequestEvent,
    PhaseCompleteEvent,
    get_event_bus,
    shutdown_event_bus,
)
from pipeline.stream_orchestrator import (
    StreamOrchestrator,
    StreamOrchestratorConfig,
    VerseState,
    StreamPipelineResult,
    create_stream_orchestrator,
)
from pipeline.stream_consumer import (
    BaseStreamConsumer,
    StreamConsumerConfig,
    PhaseStreamConsumer,
    StreamConsumerManager,
    create_phase_consumer,
    create_consumer_manager,
)
from pipeline.stream_phases import (
    StreamEnabledPhase,
    StreamLinguisticPhase,
    StreamPhaseFactory,
    start_all_phase_consumers,
    start_phase_consumer,
)
from pipeline.recovery import (
    RecoveryService,
    RecoveryConfig,
    DLQEntry,
    Checkpoint,
    get_recovery_service,
    shutdown_recovery_service,
)

__all__ = [
    # Base components
    "BasePipelinePhase",
    "PhaseConfig",
    "PhaseResult",
    "PhaseStatus",
    "PipelineContext",
    # Phases
    "LinguisticPhase",
    "TheologicalPhase",
    "IntertextualPhase",
    "ValidationPhase",
    "FinalizationPhase",
    # Original orchestrator
    "PipelineOrchestrator",
    "PipelineConfig",
    "PipelineResult",
    # Event bus
    "EventBus",
    "EventBusConfig",
    "EventMessage",
    "StreamTopic",
    "TOPICS",
    "PHASE_TOPICS",
    "VerseEvent",
    "PhaseRequestEvent",
    "PhaseCompleteEvent",
    "get_event_bus",
    "shutdown_event_bus",
    # Stream orchestrator
    "StreamOrchestrator",
    "StreamOrchestratorConfig",
    "VerseState",
    "StreamPipelineResult",
    "create_stream_orchestrator",
    # Stream consumers
    "BaseStreamConsumer",
    "StreamConsumerConfig",
    "PhaseStreamConsumer",
    "StreamConsumerManager",
    "create_phase_consumer",
    "create_consumer_manager",
    # Stream-enabled phases
    "StreamEnabledPhase",
    "StreamLinguisticPhase",
    "StreamPhaseFactory",
    "start_all_phase_consumers",
    "start_phase_consumer",
    # Recovery
    "RecoveryService",
    "RecoveryConfig",
    "DLQEntry",
    "Checkpoint",
    "get_recovery_service",
    "shutdown_recovery_service",
]
