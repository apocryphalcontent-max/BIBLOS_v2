"""
Base Phase Class

Abstract base class for all pipeline phases with dependency management.
"""
import asyncio
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING
from enum import Enum
from dataclasses import dataclass

from pipeline.context import ProcessingContext, PhaseState

if TYPE_CHECKING:
    from pipeline.unified_orchestrator import UnifiedOrchestrator


class PhasePriority(Enum):
    """Execution priority for phases."""
    CRITICAL = 1      # Must complete, blocks pipeline
    HIGH = 2          # Important but gracefully degradable
    NORMAL = 3        # Standard processing
    OPTIONAL = 4      # Can be skipped without impact

    @property
    def timeout_multiplier(self) -> float:
        """Timeout scaling based on priority."""
        return {
            PhasePriority.CRITICAL: 2.0,
            PhasePriority.HIGH: 1.5,
            PhasePriority.NORMAL: 1.0,
            PhasePriority.OPTIONAL: 0.5,
        }[self]


class PhaseCategory(Enum):
    """Category classification for phases."""
    LINGUISTIC = "linguistic"
    THEOLOGICAL = "theological"
    INTERTEXTUAL = "intertextual"
    CROSS_REFERENCE = "cross_reference"
    VALIDATION = "validation"

    @property
    def typical_oracles(self) -> List[str]:
        """Which oracles typically run in this category."""
        return {
            PhaseCategory.LINGUISTIC: ["omni_resolver"],
            PhaseCategory.THEOLOGICAL: ["lxx_extractor"],
            PhaseCategory.INTERTEXTUAL: ["typology_engine", "necessity_calculator", "prophetic_prover"],
            PhaseCategory.CROSS_REFERENCE: ["gnn_model", "vector_store"],
            PhaseCategory.VALIDATION: ["theological_validator"],
        }[self]


@dataclass
class PhaseDependency:
    """Declares a dependency between phases."""
    phase_name: str
    required_outputs: List[str]  # Fields required from that phase
    is_hard: bool = True  # Hard dependency blocks; soft allows degradation


class Phase(ABC):
    """
    Abstract base class for pipeline phases.
    Provides dependency tracking and execution lifecycle.
    """
    name: str
    category: PhaseCategory
    priority: PhasePriority = PhasePriority.NORMAL
    is_critical: bool = False
    base_timeout_seconds: float = 30.0

    def __init__(self, orchestrator: "UnifiedOrchestrator"):
        self.orchestrator = orchestrator

    @property
    def effective_timeout(self) -> float:
        """Calculate timeout based on priority."""
        return self.base_timeout_seconds * self.priority.timeout_multiplier

    @property
    @abstractmethod
    def dependencies(self) -> List[PhaseDependency]:
        """Declare phase dependencies."""
        pass

    @property
    @abstractmethod
    def outputs(self) -> List[str]:
        """Declare what this phase produces in context."""
        pass

    def check_dependencies_satisfied(self, context: ProcessingContext) -> bool:
        """Check if all hard dependencies are satisfied."""
        for dep in self.dependencies:
            if not dep.is_hard:
                continue
            phase_state = context.phase_states.get(dep.phase_name)
            if phase_state != PhaseState.COMPLETED:
                return False
            # Check required outputs exist
            for output in dep.required_outputs:
                if not hasattr(context, output) or getattr(context, output) is None:
                    return False
        return True

    @abstractmethod
    async def execute(self, context: ProcessingContext) -> ProcessingContext:
        """Execute the phase logic."""
        pass

    async def execute_with_timeout(self, context: ProcessingContext) -> ProcessingContext:
        """Execute with timeout protection."""
        return await asyncio.wait_for(
            self.execute(context),
            timeout=self.effective_timeout
        )
