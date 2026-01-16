"""
CQRS: Command Definitions

Commands represent intentions to change system state.
They are validated and executed by command handlers.
"""
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4


@dataclass
class BaseCommand:
    """Base class for all commands."""
    command_id: UUID
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None


# ==================== Verse Processing Commands ====================

@dataclass
class ProcessVerseCommand(BaseCommand):
    """
    Command to process a verse through the complete pipeline.

    Args:
        verse_id: Verse to process (e.g., "GEN.1.1")
        skip_phases: Optional list of phases to skip
        force_reprocess: Force reprocessing even if already processed
    """
    verse_id: str
    skip_phases: Optional[List[str]] = None
    force_reprocess: bool = False


@dataclass
class BatchProcessCommand(BaseCommand):
    """
    Command to batch process multiple verses.

    Args:
        verse_ids: List of verses to process
        parallel: Whether to process in parallel
        max_concurrency: Maximum parallel workers
    """
    verse_ids: List[str]
    parallel: bool = True
    max_concurrency: int = 4


# ==================== Cross-Reference Commands ====================

@dataclass
class DiscoverCrossReferencesCommand(BaseCommand):
    """
    Command to discover cross-references for a verse.

    Args:
        verse_id: Source verse
        top_k: Number of candidates to discover
        min_confidence: Minimum confidence threshold
        connection_types: Filter by connection types
    """
    verse_id: str
    top_k: int = 10
    min_confidence: float = 0.7
    connection_types: Optional[List[str]] = None


@dataclass
class ValidateCrossReferenceCommand(BaseCommand):
    """
    Command to validate a cross-reference.

    Args:
        source_ref: Source verse
        target_ref: Target verse
        connection_type: Type of connection
        discovered_confidence: Initial confidence from discovery
    """
    source_ref: str
    target_ref: str
    connection_type: str
    discovered_confidence: float


@dataclass
class RejectCrossReferenceCommand(BaseCommand):
    """
    Command to reject a cross-reference.

    Args:
        source_ref: Source verse
        target_ref: Target verse
        connection_type: Type of connection
        rejection_reason: Why it was rejected
    """
    source_ref: str
    target_ref: str
    connection_type: str
    rejection_reason: str


# ==================== Oracle Commands ====================

@dataclass
class ResolveWordMeaningCommand(BaseCommand):
    """
    Command to resolve absolute word meaning.

    Args:
        verse_id: Verse containing the word
        word: Word to resolve
        language: Language code (hebrew, greek, aramaic)
    """
    verse_id: str
    word: str
    language: str


@dataclass
class CalculateNecessityCommand(BaseCommand):
    """
    Command to calculate inter-verse necessity.

    Args:
        verse_a: First verse
        verse_b: Second verse
    """
    verse_a: str
    verse_b: str


@dataclass
class ExtractLXXContentCommand(BaseCommand):
    """
    Command to extract LXX Christological content.

    Args:
        verse_id: OT verse to analyze
    """
    verse_id: str


@dataclass
class AnalyzeTypologyCommand(BaseCommand):
    """
    Command to analyze typological connections.

    Args:
        type_ref: Type verse (usually OT)
        antitype_ref: Antitype verse (usually NT)
    """
    type_ref: str
    antitype_ref: str


@dataclass
class ProvePropheticNecessityCommand(BaseCommand):
    """
    Command to compute prophetic necessity proof.

    Args:
        prophecy_ids: List of prophecy identifiers
        prior_supernatural: Prior probability of supernatural
    """
    prophecy_ids: List[str]
    prior_supernatural: float = 0.5


# ==================== Patristic Commands ====================

@dataclass
class AddPatristicWitnessCommand(BaseCommand):
    """
    Command to add a patristic witness.

    Args:
        verse_id: Verse being interpreted
        father_name: Name of Church Father
        authority_level: Authority level
        interpretation: The interpretation
        source_reference: Citation
    """
    verse_id: str
    father_name: str
    authority_level: str
    interpretation: str
    source_reference: str


@dataclass
class CalculateConsensusCommand(BaseCommand):
    """
    Command to calculate patristic consensus.

    Args:
        verse_id: Verse to calculate consensus for
    """
    verse_id: str


# ==================== Constraint Commands ====================

@dataclass
class ApplyTheologicalConstraintsCommand(BaseCommand):
    """
    Command to apply theological constraints to a cross-reference.

    Args:
        source_ref: Source verse
        target_ref: Target verse
        connection_type: Type of connection
        initial_confidence: Pre-constraint confidence
    """
    source_ref: str
    target_ref: str
    connection_type: str
    initial_confidence: float


# ==================== Data Migration Commands ====================

@dataclass
class MigrateVerseDataCommand(BaseCommand):
    """
    Command to migrate verse data from legacy format.

    Args:
        verse_id: Verse to migrate
        source_format: Source data format
        data: Raw data to migrate
    """
    verse_id: str
    source_format: str
    data: Dict[str, Any]


@dataclass
class RebuildProjectionCommand(BaseCommand):
    """
    Command to rebuild a read model projection.

    Args:
        projection_name: Name of projection to rebuild
        from_event_id: Optional starting event ID
    """
    projection_name: str
    from_event_id: Optional[UUID] = None


# Factory for creating commands with auto-generated IDs
def create_command(command_class: type, **kwargs) -> BaseCommand:
    """
    Create a command with auto-generated command_id.

    Args:
        command_class: Command class to instantiate
        **kwargs: Command parameters

    Returns:
        Instantiated command
    """
    if 'command_id' not in kwargs:
        kwargs['command_id'] = uuid4()
    return command_class(**kwargs)
