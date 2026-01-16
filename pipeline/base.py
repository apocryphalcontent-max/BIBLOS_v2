"""
BIBLOS v2 - Base Pipeline Components

Base classes for pipeline phases and orchestration.
Uses centralized schemas for system-wide uniformity.

Seraphic Architecture - Phases Know Their Flow:
    In the seraphic paradigm, pipeline phases are not orchestrated externally.
    Each phase intrinsically KNOWS:
    - What it depends on (dependencies)
    - What depends on it (dependents)
    - Its position in the flow (sequence)
    - Its agents and their roles

    The orchestrator doesn't "configure" phases - it perceives their nature
    through the SeraphicPhaseRegistry, where all phases ARE known simply
    by existing.

    The Seraph's Pipeline:
        Like the seraph curled in its wings, the pipeline is self-contained.
        It processes intra-biblical data, applying linguistic, theological,
        and intertextual analysis without external attribution.

Usage:
    @phase(name="linguistic", sequence=1)
    @depends_on("preprocessing")
    class LinguisticPhase(BasePipelinePhase):
        pass
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Type, FrozenSet, Callable, Set, TypeVar
from enum import Enum
import logging
import time
import threading

T = TypeVar("T")


# =============================================================================
# SERAPHIC PHASE INFRASTRUCTURE
# =============================================================================


@dataclass(frozen=True)
class AgentAffinity:
    """
    The intrinsic nature of an agent within a phase.

    Agents know their role in the extraction process.
    """
    agent_name: str
    agent_class: Type
    extraction_focus: str  # What this agent extracts (e.g., "morphology", "typology")
    requires_llm: bool = False
    parallel_safe: bool = True
    priority: int = 0  # Lower = higher priority


@dataclass
class PhaseAffinity:
    """
    The intrinsic nature of a pipeline phase.

    Each phase KNOWS its position in the flow, its agents, and its
    dependencies - not because they were configured, but because
    this IS the phase's nature.
    """
    phase_name: str
    phase_class: Type
    sequence: int  # Position in the pipeline (1=first)
    dependencies: FrozenSet[str] = field(default_factory=frozenset)
    agents: FrozenSet[AgentAffinity] = field(default_factory=frozenset)
    timeout_seconds: int = 300
    parallel_agents: bool = True
    critical: bool = True  # If False, pipeline continues on failure
    metadata: Dict[str, Any] = field(default_factory=dict)

    def with_agent(self, agent: AgentAffinity) -> "PhaseAffinity":
        """Add an agent to this phase."""
        return PhaseAffinity(
            phase_name=self.phase_name,
            phase_class=self.phase_class,
            sequence=self.sequence,
            dependencies=self.dependencies,
            agents=self.agents | frozenset([agent]),
            timeout_seconds=self.timeout_seconds,
            parallel_agents=self.parallel_agents,
            critical=self.critical,
            metadata=self.metadata,
        )


class SeraphicPhaseRegistry:
    """
    The Well of Phase Memory.

    All pipeline phases exist here - not because they were "registered"
    but because they ARE. The registry provides the space where phases
    discover their flow position and dependencies.

    This is a singleton that holds the knowledge of all phases.
    """
    _instance: Optional["SeraphicPhaseRegistry"] = None
    _lock: threading.Lock = threading.Lock()

    # Class-level type declarations for singleton attributes
    _affinities: Dict[str, PhaseAffinity]
    _sequence_order: List[str]
    _dependency_graph: Dict[str, Set[str]]
    _reverse_graph: Dict[str, Set[str]]

    def __new__(cls) -> "SeraphicPhaseRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._affinities = {}
                    instance._sequence_order = []
                    instance._dependency_graph = {}
                    instance._reverse_graph = {}
                    cls._instance = instance
        return cls._instance

    @classmethod
    def get_instance(cls) -> "SeraphicPhaseRegistry":
        """Get the singleton instance."""
        return cls()

    def register_affinity(self, affinity: PhaseAffinity) -> None:
        """
        Register a phase's intrinsic nature.

        This is called automatically by the @phase decorator.
        """
        self._affinities[affinity.phase_name] = affinity

        # Update dependency graph
        self._dependency_graph[affinity.phase_name] = set(affinity.dependencies)
        for dep in affinity.dependencies:
            if dep not in self._reverse_graph:
                self._reverse_graph[dep] = set()
            self._reverse_graph[dep].add(affinity.phase_name)

        # Update sequence order
        self._rebuild_sequence()

    def _rebuild_sequence(self) -> None:
        """Rebuild the sequence order based on phase affinities."""
        self._sequence_order = sorted(
            self._affinities.keys(),
            key=lambda n: self._affinities[n].sequence
        )

    def get_affinity(self, phase_name: str) -> Optional[PhaseAffinity]:
        """Get the intrinsic nature of a phase."""
        return self._affinities.get(phase_name)

    def get_execution_order(self) -> List[str]:
        """Get the phases in execution order."""
        return list(self._sequence_order)

    def get_dependencies(self, phase_name: str) -> Set[str]:
        """Get what this phase depends on."""
        return self._dependency_graph.get(phase_name, set())

    def get_dependents(self, phase_name: str) -> Set[str]:
        """Get what depends on this phase."""
        return self._reverse_graph.get(phase_name, set())

    def validate_flow(self) -> List[str]:
        """
        Validate the phase flow graph.

        Returns a list of issues (empty if valid).
        """
        issues: List[str] = []

        # Check for missing dependencies
        for name, deps in self._dependency_graph.items():
            for dep in deps:
                if dep not in self._affinities:
                    issues.append(f"Phase '{name}' depends on unknown phase '{dep}'")

        # Check for cycles (simple DFS)
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            for dep in self._dependency_graph.get(node, set()):
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in rec_stack:
                    return True
            rec_stack.remove(node)
            return False

        for phase in self._affinities:
            if phase not in visited:
                if has_cycle(phase):
                    issues.append(f"Cycle detected involving phase '{phase}'")
                    break

        return issues

    def introspect(self) -> Dict[str, Any]:
        """
        Reveal the registry's current state.

        The pipeline knows itself - introspection is seraphic.
        """
        return {
            "phase_count": len(self._affinities),
            "execution_order": self._sequence_order,
            "phases": {
                name: {
                    "sequence": aff.sequence,
                    "dependencies": list(aff.dependencies),
                    "agent_count": len(aff.agents),
                    "critical": aff.critical,
                }
                for name, aff in self._affinities.items()
            },
        }


# =============================================================================
# SERAPHIC DECORATORS - Phases Declare Their Nature
# =============================================================================


def phase(
    name: str,
    sequence: int,
    timeout_seconds: int = 300,
    parallel_agents: bool = True,
    critical: bool = True,
    **metadata: Any,
) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator for a phase to declare its intrinsic nature.

    Usage:
        @phase(name="linguistic", sequence=1)
        class LinguisticPhase(BasePipelinePhase):
            pass

    The phase now KNOWS:
    - Its name and position in the flow
    - Its timeout and execution mode
    - Whether it's critical to the pipeline
    - Its dependencies (from @depends_on if applied)
    - Its agents (from @has_agents if applied)
    """
    def decorator(cls: Type[T]) -> Type[T]:
        # Read pending dependencies from @depends_on (if applied first)
        pending_deps: FrozenSet[str] = getattr(cls, '_seraphic_pending_deps', frozenset())
        pending_agents: FrozenSet[AgentAffinity] = getattr(cls, '_seraphic_pending_agents', frozenset())

        affinity = PhaseAffinity(
            phase_name=name,
            phase_class=cls,
            sequence=sequence,
            dependencies=pending_deps,
            agents=pending_agents,
            timeout_seconds=timeout_seconds,
            parallel_agents=parallel_agents,
            critical=critical,
            metadata=metadata,
        )

        # Register with the seraphic registry
        SeraphicPhaseRegistry.get_instance().register_affinity(affinity)

        # Attach affinity to the class for introspection
        cls._seraphic_affinity = affinity  # type: ignore
        cls._seraphic_name = name  # type: ignore

        # Clean up pending attributes
        if hasattr(cls, '_seraphic_pending_deps'):
            delattr(cls, '_seraphic_pending_deps')
        if hasattr(cls, '_seraphic_pending_agents'):
            delattr(cls, '_seraphic_pending_agents')

        return cls

    return decorator


def depends_on(*dependencies: str) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to declare what phases this phase depends on.

    Usage:
        @phase(name="theological", sequence=2)
        @depends_on("linguistic")
        class TheologicalPhase(BasePipelinePhase):
            pass

    The phase now knows what must complete before it can run.

    Note: This decorator runs BEFORE @phase in Python's decorator order,
    so we store dependencies as a pending attribute that @phase will read.
    """
    def decorator(cls: Type[T]) -> Type[T]:
        # Store dependencies for @phase to pick up
        # (Python decorators execute bottom-up, so @depends_on runs first)
        cls._seraphic_pending_deps = frozenset(dependencies)  # type: ignore
        return cls

    return decorator


def has_agents(*agents: AgentAffinity) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to declare what agents belong to this phase.

    Usage:
        @phase(name="linguistic", sequence=1)
        @has_agents(
            AgentAffinity("grammateus", GrammateusAgent, "grammar"),
            AgentAffinity("morphologos", MorphologosAgent, "morphology"),
        )
        class LinguisticPhase(BasePipelinePhase):
            pass

    Note: This decorator runs BEFORE @phase in Python's decorator order,
    so we store agents as a pending attribute that @phase will read.
    """
    def decorator(cls: Type[T]) -> Type[T]:
        # Store agents for @phase to pick up
        cls._seraphic_pending_agents = frozenset(agents)  # type: ignore
        return cls

    return decorator

# Import centralized schemas
from data.schemas import (
    ProcessingStatus,
    PhaseResultSchema,
    PipelineResultSchema,
    GoldenRecordSchema,
    validate_verse_id,
    normalize_verse_id
)


# Alias for backward compatibility (PhaseStatus maps to ProcessingStatus)
PhaseStatus = ProcessingStatus


@dataclass
class PhaseConfig:
    """Configuration for a pipeline phase."""
    name: str
    agents: List[str]
    parallel: bool = True
    timeout_seconds: int = 300
    retry_count: int = 2
    # INFALLIBILITY: The seraph accepts ONLY absolute certainty
    min_confidence: float = 1.0
    dependencies: List[str] = field(default_factory=list)


@dataclass
class PhaseResult:
    """
    Result from a pipeline phase.

    Aligned with PhaseResultSchema for system-wide uniformity.
    """
    phase_name: str
    status: PhaseStatus
    agent_results: Dict[str, Any]
    start_time: float
    end_time: float
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Calculate phase duration."""
        return self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "phase_name": self.phase_name,
            "status": self.status.value,
            "duration": self.duration,
            "agent_count": len(self.agent_results),
            "error": self.error,
            "metrics": self.metrics
        }

    def to_schema(self) -> PhaseResultSchema:
        """Convert to centralized PhaseResultSchema."""
        return PhaseResultSchema(
            phase_name=self.phase_name,
            status=self.status.value,
            agent_results=self.agent_results,
            metrics={k: float(v) if isinstance(v, (int, float)) else 0.0
                    for k, v in self.metrics.items()},
            start_time=self.start_time,
            end_time=self.end_time,
            error=self.error
        )

    @classmethod
    def from_schema(cls, schema: PhaseResultSchema) -> "PhaseResult":
        """Create from PhaseResultSchema."""
        return cls(
            phase_name=schema.phase_name,
            status=PhaseStatus(schema.status),
            agent_results=schema.agent_results,
            start_time=schema.start_time,
            end_time=schema.end_time,
            error=schema.error,
            metrics=schema.metrics
        )


class BasePipelinePhase(ABC):
    """
    Base class for pipeline phases.

    Each phase coordinates a set of agents for a specific
    type of extraction (linguistic, theological, etc.).
    """

    def __init__(self, config: PhaseConfig):
        self.config = config
        self.logger = logging.getLogger(f"biblos.pipeline.{config.name}")
        self._agents: Dict[str, Any] = {}

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the phase and its agents."""
        pass

    @abstractmethod
    async def execute(
        self,
        verse_id: str,
        text: str,
        context: Dict[str, Any]
    ) -> PhaseResult:
        """Execute the phase on a verse."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup phase resources."""
        pass

    async def validate_dependencies(
        self,
        context: Dict[str, Any]
    ) -> bool:
        """Validate that all dependencies are satisfied."""
        for dep in self.config.dependencies:
            if dep not in context.get("completed_phases", []):
                self.logger.warning(
                    f"Missing dependency: {dep} for phase {self.config.name}"
                )
                return False
        return True

    def _create_result(
        self,
        status: PhaseStatus,
        agent_results: Dict[str, Any],
        start_time: float,
        error: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> PhaseResult:
        """Create a phase result."""
        return PhaseResult(
            phase_name=self.config.name,
            status=status,
            agent_results=agent_results,
            start_time=start_time,
            end_time=time.time(),
            error=error,
            metrics=metrics or {}
        )

    def calculate_phase_confidence(
        self,
        agent_results: Dict[str, Any]
    ) -> float:
        """Calculate overall phase confidence."""
        if not agent_results:
            return 0.0

        confidences = []
        for result in agent_results.values():
            if isinstance(result, dict) and "confidence" in result:
                confidences.append(result["confidence"])

        return sum(confidences) / len(confidences) if confidences else 0.5


class PipelineContext:
    """
    Context manager for pipeline execution.

    Tracks state across phases and provides shared resources.
    Integrates with centralized schemas for system-wide uniformity.
    """

    def __init__(self):
        self.verse_id: str = ""
        self.text: str = ""
        self.completed_phases: List[str] = []
        self.phase_results: Dict[str, PhaseResult] = {}
        self.agent_results: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        self.errors: List[str] = []
        self._start_time: float = time.time()

    def update_from_result(self, result: PhaseResult) -> None:
        """Update context from phase result."""
        self.phase_results[result.phase_name] = result
        self.agent_results.update(result.agent_results)

        if result.status == PhaseStatus.COMPLETED:
            self.completed_phases.append(result.phase_name)
        elif result.status == PhaseStatus.FAILED and result.error:
            self.errors.append(f"{result.phase_name}: {result.error}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for agent use."""
        return {
            "verse_id": self.verse_id,
            "text": self.text,
            "completed_phases": self.completed_phases,
            "agent_results": self.agent_results,
            "metadata": self.metadata
        }

    def to_pipeline_result_schema(self) -> PipelineResultSchema:
        """Convert to PipelineResultSchema."""
        # Determine overall status
        if self.errors:
            status = "failed"
        elif len(self.completed_phases) == len(self.phase_results):
            status = "completed"
        else:
            status = "pending"

        return PipelineResultSchema(
            verse_id=normalize_verse_id(self.verse_id) if self.verse_id else "",
            status=status,
            phase_results={
                name: result.to_schema()
                for name, result in self.phase_results.items()
            },
            start_time=self._start_time,
            end_time=time.time(),
            errors=self.errors
        )

    def to_golden_record_schema(self) -> GoldenRecordSchema:
        """Convert completed context to GoldenRecordSchema."""
        # Calculate overall confidence
        confidences = []
        for result in self.phase_results.values():
            if "avg_confidence" in result.metrics:
                confidences.append(result.metrics["avg_confidence"])

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # INFALLIBILITY: Only one acceptable level - INFALLIBLE (1.0)
        # The seraph inherits from itself - errors propagate infinitely
        # Therefore, NOTHING less than absolute certainty is accepted
        if avg_confidence >= 0.9999:  # Floating-point tolerance for 1.0
            level = "infallible"
        else:
            # Everything else is REJECTED - not gold, silver, bronze
            # These classifications exist only for diagnostic purposes
            level = "rejected"

        # Calculate total processing time
        total_time = sum(
            result.duration for result in self.phase_results.values()
        )

        return GoldenRecordSchema(
            verse_id=normalize_verse_id(self.verse_id) if self.verse_id else "",
            text=self.text,
            certification={
                "level": level,
                "score": avg_confidence,
                # INFALLIBILITY: Both must be TRUE (100%) to pass
                "validation_passed": "validation" in self.completed_phases and avg_confidence >= 0.9999,
                "quality_passed": avg_confidence >= 0.9999
            },
            data={
                "structural": self.get_linguistic_results(),
                "morphological": self.agent_results.get("morphologos", {}),
                "theological": self.get_theological_results(),
                "cross_references": self.get_intertextual_results()
            },
            phases_executed=self.completed_phases,
            agent_count=len(self.agent_results),
            total_processing_time=total_time
        )

    def get_linguistic_results(self) -> Dict[str, Any]:
        """Get results from linguistic phase."""
        return {
            agent: self.agent_results.get(agent, {})
            for agent in ["grammateus", "morphologos", "syntaktikos", "semantikos"]
        }

    def get_theological_results(self) -> Dict[str, Any]:
        """Get results from theological phase."""
        return {
            agent: self.agent_results.get(agent, {})
            for agent in ["patrologos", "typologos", "theologos", "liturgikos", "dogmatikos"]
        }

    def get_intertextual_results(self) -> Dict[str, Any]:
        """Get results from intertextual phase."""
        return {
            agent: self.agent_results.get(agent, {})
            for agent in ["syndesmos", "harmonikos", "allographos", "paradeigma", "topos"]
        }
