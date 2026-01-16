"""
BIBLOS v2 - Base Pipeline Components

Base classes for pipeline phases and orchestration.
Uses centralized schemas for system-wide uniformity.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
import logging
import time

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
    min_confidence: float = 0.6
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

        # Determine certification level
        if avg_confidence >= 0.9:
            level = "gold"
        elif avg_confidence >= 0.75:
            level = "silver"
        elif avg_confidence >= 0.5:
            level = "bronze"
        else:
            level = "provisional"

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
                "validation_passed": "validation" in self.completed_phases,
                "quality_passed": avg_confidence >= 0.7
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
