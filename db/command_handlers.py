"""
CQRS: Command Handlers

Handles commands by executing business logic and emitting events.
Each handler validates the command and produces zero or more events.
"""
import logging
from typing import List, Optional
from datetime import datetime, timezone
from uuid import UUID

from db.events import (
    BaseEvent,
    VerseProcessingStarted,
    CrossReferenceDiscovered,
    CrossReferenceValidated,
    CrossReferenceRejected,
    OmniResolutionComputed,
    NecessityCalculated,
    LXXDivergenceDetected,
    TypologyDiscovered,
    PropheticProofComputed,
    PatristicWitnessAdded,
    ConsensusCalculated,
    TheologicalConstraintApplied,
    ConstraintViolationDetected,
)
from db.commands import (
    BaseCommand,
    ProcessVerseCommand,
    DiscoverCrossReferencesCommand,
    ValidateCrossReferenceCommand,
    RejectCrossReferenceCommand,
    ResolveWordMeaningCommand,
    CalculateNecessityCommand,
    ExtractLXXContentCommand,
    AnalyzeTypologyCommand,
    ProvePropheticNecessityCommand,
    AddPatristicWitnessCommand,
    CalculateConsensusCommand,
    ApplyTheologicalConstraintsCommand,
)
from db.event_store import EventStore
from core.validation import validate_verse_id


logger = logging.getLogger(__name__)


class CommandHandler:
    """
    Base command handler with common functionality.

    Subclasses implement handle() method for specific command types.
    """

    def __init__(self, event_store: EventStore):
        """
        Initialize command handler.

        Args:
            event_store: Event store for persisting events
        """
        self.event_store = event_store

    async def execute(self, command: BaseCommand) -> List[BaseEvent]:
        """
        Execute a command and return emitted events.

        Args:
            command: Command to execute

        Returns:
            List of emitted events

        Raises:
            ValueError: If command is invalid
        """
        # Validate command
        self.validate(command)

        # Handle command logic
        events = await self.handle(command)

        # Persist events
        if events:
            await self.event_store.append_batch(events)
            logger.info(
                f"Executed {command.__class__.__name__}, emitted {len(events)} events"
            )

        return events

    def validate(self, command: BaseCommand) -> None:
        """
        Validate command before execution.

        Args:
            command: Command to validate

        Raises:
            ValueError: If command is invalid
        """
        # Base validation - subclasses can override
        if not command.command_id:
            raise ValueError("Command must have command_id")

    async def handle(self, command: BaseCommand) -> List[BaseEvent]:
        """
        Handle command logic and produce events.

        Args:
            command: Command to handle

        Returns:
            List of events to emit
        """
        raise NotImplementedError("Subclasses must implement handle()")


class ProcessVerseCommandHandler(CommandHandler):
    """Handler for ProcessVerseCommand."""

    def validate(self, command: ProcessVerseCommand) -> None:
        """Validate verse ID format."""
        super().validate(command)
        validate_verse_id(command.verse_id)

    async def handle(self, command: ProcessVerseCommand) -> List[BaseEvent]:
        """Start verse processing workflow."""
        # Determine phase plan
        all_phases = ["linguistic", "theological", "intertextual", "cross_reference", "validation"]
        phase_plan = [p for p in all_phases if p not in (command.skip_phases or [])]

        # Get current aggregate version
        version = await self.event_store.get_aggregate_version(command.verse_id)

        # Create event
        event = VerseProcessingStarted(
            aggregate_id=command.verse_id,
            aggregate_version=version + 1,
            verse_id=command.verse_id,
            phase_plan=phase_plan,
            user_id=command.user_id,
            correlation_id=command.correlation_id,
            causation_id=str(command.command_id),
            timestamp=datetime.now(timezone.utc)
        )

        return [event]


class DiscoverCrossReferencesCommandHandler(CommandHandler):
    """Handler for DiscoverCrossReferencesCommand."""

    def __init__(
        self,
        event_store: EventStore,
        discovery_engine: Optional[Any] = None  # Injected ML component
    ):
        super().__init__(event_store)
        self.discovery_engine = discovery_engine

    def validate(self, command: DiscoverCrossReferencesCommand) -> None:
        """Validate verse ID and parameters."""
        super().validate(command)
        validate_verse_id(command.verse_id)

        if command.top_k <= 0:
            raise ValueError("top_k must be positive")
        if not (0.0 <= command.min_confidence <= 1.0):
            raise ValueError("min_confidence must be between 0 and 1")

    async def handle(self, command: DiscoverCrossReferencesCommand) -> List[BaseEvent]:
        """Discover cross-references using ML inference."""
        events = []

        # Mock discovery for now - real implementation would use ML model
        # In production, this would call the InferencePipeline
        candidates = await self._discover_candidates(command)

        version = await self.event_store.get_aggregate_version(command.verse_id)

        for idx, candidate in enumerate(candidates):
            event = CrossReferenceDiscovered(
                aggregate_id=command.verse_id,
                aggregate_version=version + idx + 1,
                source_ref=command.verse_id,
                target_ref=candidate['target_ref'],
                connection_type=candidate['connection_type'],
                confidence=candidate['confidence'],
                discovered_by="inference_pipeline",
                evidence=candidate.get('evidence', {}),
                correlation_id=command.correlation_id,
                causation_id=str(command.command_id),
                timestamp=datetime.now(timezone.utc)
            )
            events.append(event)

        return events

    async def _discover_candidates(self, command: DiscoverCrossReferencesCommand) -> List[dict]:
        """Placeholder for ML-based discovery."""
        # In production, this would use the InferencePipeline from ml/inference/
        # For now, return empty list
        return []


class ValidateCrossReferenceCommandHandler(CommandHandler):
    """Handler for ValidateCrossReferenceCommand."""

    def validate(self, command: ValidateCrossReferenceCommand) -> None:
        """Validate verse IDs."""
        super().validate(command)
        validate_verse_id(command.source_ref)
        validate_verse_id(command.target_ref)

    async def handle(self, command: ValidateCrossReferenceCommand) -> List[BaseEvent]:
        """Validate cross-reference through theological constraints."""
        # Mock validation - real implementation would check constraints
        passed = await self._validate_reference(command)

        version = await self.event_store.get_aggregate_version(command.source_ref)

        if passed:
            event = CrossReferenceValidated(
                aggregate_id=command.source_ref,
                aggregate_version=version + 1,
                source_ref=command.source_ref,
                target_ref=command.target_ref,
                connection_type=command.connection_type,
                final_confidence=command.discovered_confidence,
                validators=["theological_constraint_validator"],
                theological_score=0.9,
                correlation_id=command.correlation_id,
                causation_id=str(command.command_id),
                timestamp=datetime.now(timezone.utc)
            )
        else:
            event = CrossReferenceRejected(
                aggregate_id=command.source_ref,
                aggregate_version=version + 1,
                source_ref=command.source_ref,
                target_ref=command.target_ref,
                connection_type=command.connection_type,
                rejection_reason="Failed theological constraints",
                violated_constraints=["chronological_priority"],
                correlation_id=command.correlation_id,
                causation_id=str(command.command_id),
                timestamp=datetime.now(timezone.utc)
            )

        return [event]

    async def _validate_reference(self, command: ValidateCrossReferenceCommand) -> bool:
        """Placeholder for validation logic."""
        # In production, would call TheologicalConstraintValidator
        return True


class ResolveWordMeaningCommandHandler(CommandHandler):
    """Handler for ResolveWordMeaningCommand."""

    def validate(self, command: ResolveWordMeaningCommand) -> None:
        """Validate parameters."""
        super().validate(command)
        validate_verse_id(command.verse_id)

        if not command.word:
            raise ValueError("word cannot be empty")
        if command.language not in ["hebrew", "greek", "aramaic"]:
            raise ValueError("language must be hebrew, greek, or aramaic")

    async def handle(self, command: ResolveWordMeaningCommand) -> List[BaseEvent]:
        """Resolve word meaning using OmniContextual oracle."""
        # Mock resolution - real implementation would use OmniContextualResolver
        result = await self._resolve_meaning(command)

        version = await self.event_store.get_aggregate_version(command.verse_id)

        event = OmniResolutionComputed(
            aggregate_id=command.verse_id,
            aggregate_version=version + 1,
            verse_id=command.verse_id,
            word=command.word,
            language=command.language,
            primary_meaning=result.get("primary_meaning", ""),
            total_occurrences=result.get("total_occurrences", 0),
            confidence=result.get("confidence", 0.0),
            semantic_field_map=result.get("semantic_field_map", {}),
            correlation_id=command.correlation_id,
            causation_id=str(command.command_id),
            timestamp=datetime.now(timezone.utc)
        )

        return [event]

    async def _resolve_meaning(self, command: ResolveWordMeaningCommand) -> dict:
        """Placeholder for OmniContextual resolution."""
        return {
            "primary_meaning": "spirit",
            "total_occurrences": 389,
            "confidence": 0.85,
            "semantic_field_map": {"breath": 0.3, "wind": 0.2, "spirit": 0.5}
        }


# ==================== Command Handler Registry ====================

class CommandBus:
    """
    Central command bus for routing commands to handlers.

    Provides single entry point for command execution.
    """

    def __init__(self, event_store: EventStore):
        """
        Initialize command bus.

        Args:
            event_store: Event store for handlers
        """
        self.event_store = event_store
        self.handlers: dict = {}
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """Register default command handlers."""
        self.register(ProcessVerseCommand, ProcessVerseCommandHandler(self.event_store))
        self.register(DiscoverCrossReferencesCommand, DiscoverCrossReferencesCommandHandler(self.event_store))
        self.register(ValidateCrossReferenceCommand, ValidateCrossReferenceCommandHandler(self.event_store))
        self.register(ResolveWordMeaningCommand, ResolveWordMeaningCommandHandler(self.event_store))
        # Add more handlers as needed

    def register(self, command_type: type, handler: CommandHandler) -> None:
        """
        Register a command handler.

        Args:
            command_type: Command class
            handler: Handler instance
        """
        self.handlers[command_type] = handler
        logger.info(f"Registered handler for {command_type.__name__}")

    async def dispatch(self, command: BaseCommand) -> List[BaseEvent]:
        """
        Dispatch command to appropriate handler.

        Args:
            command: Command to dispatch

        Returns:
            List of emitted events

        Raises:
            ValueError: If no handler registered for command type
        """
        command_type = type(command)

        if command_type not in self.handlers:
            raise ValueError(f"No handler registered for {command_type.__name__}")

        handler = self.handlers[command_type]
        return await handler.execute(command)
