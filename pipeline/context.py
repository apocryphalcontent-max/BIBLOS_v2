"""
Processing Context for Unified Pipeline

Carries state through all processing phases with health monitoring.
"""
import traceback
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict
import numpy as np


class ContextCompleteness(Enum):
    """Level of context completeness."""
    FULL = "full"           # All phases completed successfully
    PARTIAL = "partial"     # Some phases skipped or degraded
    MINIMAL = "minimal"     # Only critical phases completed
    FAILED = "failed"       # Critical phase failed

    @property
    def is_usable(self) -> bool:
        """Whether this context can produce a golden record."""
        return self in {ContextCompleteness.FULL, ContextCompleteness.PARTIAL, ContextCompleteness.MINIMAL}


class PhaseState(Enum):
    """State machine for phase execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"

    @property
    def is_terminal(self) -> bool:
        """Check if state is terminal (no further transitions)."""
        return self in {PhaseState.COMPLETED, PhaseState.FAILED, PhaseState.SKIPPED}

    @property
    def allows_retry(self) -> bool:
        """Check if state allows retry."""
        return self == PhaseState.FAILED

    def can_transition_to(self, target: "PhaseState") -> bool:
        """Validate state transition."""
        valid_transitions = {
            PhaseState.PENDING: {PhaseState.RUNNING, PhaseState.SKIPPED},
            PhaseState.RUNNING: {PhaseState.COMPLETED, PhaseState.FAILED},
            PhaseState.FAILED: {PhaseState.RETRYING, PhaseState.SKIPPED},
            PhaseState.RETRYING: {PhaseState.RUNNING},
            PhaseState.COMPLETED: set(),  # Terminal
            PhaseState.SKIPPED: set(),    # Terminal
        }
        return target in valid_transitions.get(self, set())


@dataclass
class ProcessingContext:
    """
    Carries state through all processing phases.
    Provides computed properties for context health monitoring.
    """
    # Identification
    verse_id: str
    correlation_id: str
    testament: Optional[str] = None

    # Phase state tracking
    phase_states: Dict[str, PhaseState] = field(default_factory=dict)

    # Phase results
    linguistic_analysis: Dict[str, Any] = field(default_factory=dict)
    theological_analysis: Dict[str, Any] = field(default_factory=dict)
    lxx_analysis: Optional[Any] = None
    patristic_witness: List[Any] = field(default_factory=list)
    typological_connections: List[Any] = field(default_factory=list)
    prophetic_analysis: Optional[Any] = None
    cross_references: List[Any] = field(default_factory=list)
    validated_cross_references: List[Any] = field(default_factory=list)

    # Embeddings
    embeddings: Dict[str, np.ndarray] = field(default_factory=dict)

    # Metrics
    phase_durations: Dict[str, float] = field(default_factory=dict)
    total_duration_ms: float = 0

    # Errors and warnings
    errors: List[Dict] = field(default_factory=list)
    warnings: List[Dict] = field(default_factory=list)

    # Computed properties
    @property
    def completeness(self) -> ContextCompleteness:
        """Determine overall context completeness."""
        if not self.phase_states:
            return ContextCompleteness.FAILED

        failed = sum(1 for s in self.phase_states.values() if s == PhaseState.FAILED)
        completed = sum(1 for s in self.phase_states.values() if s == PhaseState.COMPLETED)
        skipped = sum(1 for s in self.phase_states.values() if s == PhaseState.SKIPPED)

        if failed > 0:
            # Check if any critical phase failed
            if self.phase_states.get("linguistic") == PhaseState.FAILED:
                return ContextCompleteness.FAILED
            return ContextCompleteness.MINIMAL

        if skipped > 0:
            return ContextCompleteness.PARTIAL

        if completed == len(self.phase_states):
            return ContextCompleteness.FULL

        return ContextCompleteness.PARTIAL

    @property
    def has_rich_theological_data(self) -> bool:
        """Check if context has rich theological analysis."""
        return (
            self.lxx_analysis is not None or
            len(self.patristic_witness) > 0 or
            len(self.typological_connections) > 0
        )

    @property
    def oracle_coverage(self) -> Dict[str, bool]:
        """Which oracles contributed to this context."""
        return {
            "omni_resolver": bool(self.linguistic_analysis.get("resolved_meanings")),
            "lxx_extractor": self.lxx_analysis is not None,
            "typology_engine": len(self.typological_connections) > 0,
            "necessity_calculator": any(
                hasattr(t, "necessity_score") and t.necessity_score > 0
                for t in self.typological_connections
            ),
            "prophetic_prover": self.prophetic_analysis is not None,
        }

    @property
    def oracle_coverage_ratio(self) -> float:
        """Ratio of oracles that contributed."""
        coverage = self.oracle_coverage
        return sum(coverage.values()) / len(coverage) if coverage else 0.0

    @property
    def embedding_domains(self) -> List[str]:
        """List of embedding domains present."""
        return list(self.embeddings.keys())

    @property
    def cross_ref_stats(self) -> Dict[str, Any]:
        """Statistics about cross-references."""
        if not self.validated_cross_references:
            return {"count": 0, "avg_confidence": 0.0, "types": []}

        types = set(getattr(r, 'connection_type', None) for r in self.validated_cross_references)
        types.discard(None)

        confidences = [getattr(r, 'final_confidence', 0.0) for r in self.validated_cross_references]

        return {
            "count": len(self.validated_cross_references),
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
            "types": list(types),
            "by_verdict": self._count_by_verdict(),
        }

    def _count_by_verdict(self) -> Dict[str, int]:
        """Count cross-refs by verdict."""
        counts = defaultdict(int)
        for ref in self.validated_cross_references:
            if hasattr(ref, "verdict"):
                verdict_value = ref.verdict.value if hasattr(ref.verdict, 'value') else str(ref.verdict)
                counts[verdict_value] += 1
        return dict(counts)

    @property
    def total_processing_time_ms(self) -> float:
        """Total time across all phases."""
        return sum(self.phase_durations.values())

    @property
    def slowest_phase(self) -> Optional[Tuple[str, float]]:
        """Identify the slowest phase."""
        if not self.phase_durations:
            return None
        slowest = max(self.phase_durations.items(), key=lambda x: x[1])
        return slowest

    def add_error(self, phase: str, error: Exception):
        """Record an error from a phase."""
        self.errors.append({
            "phase": phase,
            "type": type(error).__name__,
            "message": str(error),
            "timestamp": datetime.utcnow().isoformat(),
            "traceback": traceback.format_exc()
        })

    def add_warning(self, phase: str, message: str):
        """Record a warning from a phase."""
        self.warnings.append({
            "phase": phase,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        })

    def to_summary(self) -> Dict[str, Any]:
        """Generate a summary of this context for logging."""
        return {
            "verse_id": self.verse_id,
            "correlation_id": self.correlation_id,
            "completeness": self.completeness.value,
            "oracle_coverage": self.oracle_coverage_ratio,
            "cross_refs": self.cross_ref_stats,
            "total_time_ms": self.total_processing_time_ms,
            "slowest_phase": self.slowest_phase,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
        }
