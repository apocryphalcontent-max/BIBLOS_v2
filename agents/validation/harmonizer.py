"""
BIBLOS v2 - HARMONIZER Agent

Result harmonization and conflict resolution agent.
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

from agents.base import (
    BaseExtractionAgent,
    AgentConfig,
    ExtractionResult,
    ExtractionType,
    ProcessingStatus
)


class ResolutionStrategy(Enum):
    """Strategies for resolving conflicts."""
    HIGHEST_CONFIDENCE = "highest_confidence"  # Use highest confidence
    MAJORITY_VOTE = "majority_vote"  # Use majority opinion
    WEIGHTED_AVERAGE = "weighted_average"  # Weighted combination
    EXPERT_PRIORITY = "expert_priority"  # Prioritize specialist agent
    MERGE = "merge"  # Merge all information
    MANUAL_REVIEW = "manual_review"  # Flag for human review


class HarmonizationStatus(Enum):
    """Status of harmonization attempt."""
    SUCCESS = "success"  # Harmonized successfully
    PARTIAL = "partial"  # Partially harmonized
    CONFLICT = "conflict"  # Unresolved conflicts remain
    DEFERRED = "deferred"  # Deferred to manual review


@dataclass
class HarmonizedResult:
    """A harmonized result from multiple agents."""
    field: str
    harmonized_value: Any
    sources: List[str]
    strategy_used: ResolutionStrategy
    confidence: float
    conflicts_resolved: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "harmonized_value": self.harmonized_value,
            "sources": self.sources,
            "strategy_used": self.strategy_used.value,
            "confidence": self.confidence,
            "conflicts_resolved": self.conflicts_resolved
        }


class HarmonizerAgent(BaseExtractionAgent):
    """
    HARMONIZER - Result harmonization agent.

    Performs:
    - Multi-agent result merging
    - Conflict resolution
    - Golden record creation
    - Consensus building
    - Uncertainty quantification
    """

    # Priority order for expert agents
    EXPERT_PRIORITY = {
        "morphological": ["morphologos"],
        "syntactic": ["syntaktikos"],
        "semantic": ["semantikos"],
        "theological": ["theologos", "patrologos"],
        "typological": ["typologos"],
        "intertextual": ["syndesmos", "allographos"],
        "structural": ["grammateus"]
    }

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="harmonizer",
                extraction_type=ExtractionType.VALIDATION,
                batch_size=100,
                min_confidence=0.7
            )
        super().__init__(config)
        self.logger = logging.getLogger("biblos.agents.harmonizer")

    async def extract(
        self,
        verse_id: str,
        text: str,
        context: Dict[str, Any]
    ) -> ExtractionResult:
        """Harmonize results from multiple agents."""
        agent_results = context.get("agent_results", {})
        consistency_report = context.get("consistency_report", {})

        # Identify conflicts to resolve
        conflicts = self._identify_conflicts(consistency_report)

        # Harmonize results
        harmonized = self._harmonize_results(agent_results, conflicts)

        # Build golden record
        golden_record = self._build_golden_record(harmonized, agent_results)

        # Calculate harmonization metrics
        metrics = self._calculate_metrics(harmonized, conflicts)

        # Identify remaining issues
        remaining_issues = self._identify_remaining_issues(harmonized)

        status = self._determine_status(remaining_issues)

        data = {
            "verse_id": verse_id,
            "harmonized_results": [h.to_dict() for h in harmonized],
            "golden_record": golden_record,
            "conflicts_resolved": len(conflicts) - len(remaining_issues),
            "remaining_issues": remaining_issues,
            "metrics": metrics,
            "status": status.value,
            "source_agents": list(agent_results.keys())
        }

        confidence = self._calculate_confidence(harmonized, remaining_issues)

        return ExtractionResult(
            agent_name=self.config.name,
            extraction_type=self.config.extraction_type,
            verse_id=verse_id,
            status=ProcessingStatus.COMPLETED,
            data=data,
            confidence=confidence
        )

    def _identify_conflicts(
        self,
        consistency_report: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify conflicts from consistency report."""
        conflicts = consistency_report.get("conflicts", [])
        return conflicts

    def _harmonize_results(
        self,
        agent_results: Dict[str, Any],
        conflicts: List[Dict[str, Any]]
    ) -> List[HarmonizedResult]:
        """Harmonize results across agents."""
        harmonized = []

        # Collect all fields across agents
        all_fields = {}
        for agent_name, result in agent_results.items():
            data = result.get("data", {}) if isinstance(result, dict) else {}
            confidence = result.get("confidence", 0.5) if isinstance(result, dict) else 0.5

            for field, value in data.items():
                if field not in all_fields:
                    all_fields[field] = []
                all_fields[field].append({
                    "agent": agent_name,
                    "value": value,
                    "confidence": confidence
                })

        # Harmonize each field
        for field, sources in all_fields.items():
            harmonized_result = self._harmonize_field(field, sources, conflicts)
            if harmonized_result:
                harmonized.append(harmonized_result)

        return harmonized

    def _harmonize_field(
        self,
        field: str,
        sources: List[Dict[str, Any]],
        conflicts: List[Dict[str, Any]]
    ) -> Optional[HarmonizedResult]:
        """Harmonize a single field across sources."""
        if not sources:
            return None

        # Check if field is in conflict
        field_conflicts = [
            c for c in conflicts
            if any(field in str(claim.get("field", "")) for claim in c.get("claims", []))
        ]

        if field_conflicts:
            # Use resolution strategy
            return self._resolve_conflict(field, sources)
        else:
            # No conflict - merge or use highest confidence
            return self._merge_field(field, sources)

    def _resolve_conflict(
        self,
        field: str,
        sources: List[Dict[str, Any]]
    ) -> HarmonizedResult:
        """Resolve a conflict for a field."""
        # Try highest confidence first
        sorted_sources = sorted(
            sources,
            key=lambda x: x["confidence"],
            reverse=True
        )

        best_source = sorted_sources[0]
        strategy = ResolutionStrategy.HIGHEST_CONFIDENCE

        # Check if values are compatible for merging
        values = [s["value"] for s in sources]
        if self._values_mergeable(values):
            merged_value = self._merge_values(values)
            strategy = ResolutionStrategy.MERGE
            return HarmonizedResult(
                field=field,
                harmonized_value=merged_value,
                sources=[s["agent"] for s in sources],
                strategy_used=strategy,
                confidence=sum(s["confidence"] for s in sources) / max(1, len(sources)),
                conflicts_resolved=1
            )

        return HarmonizedResult(
            field=field,
            harmonized_value=best_source["value"],
            sources=[best_source["agent"]],
            strategy_used=strategy,
            confidence=best_source["confidence"],
            conflicts_resolved=1
        )

    def _merge_field(
        self,
        field: str,
        sources: List[Dict[str, Any]]
    ) -> HarmonizedResult:
        """Merge field values when no conflict."""
        if len(sources) == 1:
            return HarmonizedResult(
                field=field,
                harmonized_value=sources[0]["value"],
                sources=[sources[0]["agent"]],
                strategy_used=ResolutionStrategy.HIGHEST_CONFIDENCE,
                confidence=sources[0]["confidence"],
                conflicts_resolved=0
            )

        values = [s["value"] for s in sources]

        if self._values_mergeable(values):
            merged = self._merge_values(values)
            strategy = ResolutionStrategy.MERGE
        else:
            # Use highest confidence
            sorted_sources = sorted(sources, key=lambda x: x["confidence"], reverse=True)
            merged = sorted_sources[0]["value"]
            strategy = ResolutionStrategy.HIGHEST_CONFIDENCE

        return HarmonizedResult(
            field=field,
            harmonized_value=merged,
            sources=[s["agent"] for s in sources],
            strategy_used=strategy,
            confidence=sum(s["confidence"] for s in sources) / len(sources),
            conflicts_resolved=0
        )

    def _values_mergeable(self, values: List[Any]) -> bool:
        """Check if values can be merged."""
        if not values:
            return False

        # Lists can be merged
        if all(isinstance(v, list) for v in values):
            return True

        # Dicts can be merged
        if all(isinstance(v, dict) for v in values):
            return True

        # Same values are trivially mergeable
        if len(set(str(v) for v in values)) == 1:
            return True

        return False

    def _merge_values(self, values: List[Any]) -> Any:
        """Merge multiple values."""
        if not values:
            return None

        # Lists: union
        if all(isinstance(v, list) for v in values):
            merged = []
            seen = set()
            for v in values:
                for item in v:
                    key = str(item)
                    if key not in seen:
                        merged.append(item)
                        seen.add(key)
            return merged

        # Dicts: deep merge
        if all(isinstance(v, dict) for v in values):
            merged = {}
            for v in values:
                for key, val in v.items():
                    if key not in merged:
                        merged[key] = val
            return merged

        # Same values: return first
        return values[0]

    def _build_golden_record(
        self,
        harmonized: List[HarmonizedResult],
        agent_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build the golden record from harmonized results."""
        golden = {
            "verse_data": {},
            "metadata": {
                "source_agents": list(agent_results.keys()),
                "harmonization_count": len(harmonized),
                "strategies_used": list(set(h.strategy_used.value for h in harmonized))
            }
        }

        for h in harmonized:
            golden["verse_data"][h.field] = h.harmonized_value

        return golden

    def _calculate_metrics(
        self,
        harmonized: List[HarmonizedResult],
        conflicts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate harmonization metrics."""
        if not harmonized:
            return {
                "harmonization_rate": 0.0,
                "avg_confidence": 0.0,
                "conflict_resolution_rate": 0.0
            }

        resolved = sum(h.conflicts_resolved for h in harmonized)
        total_conflicts = len(conflicts)

        return {
            "harmonization_rate": len(harmonized) / max(1, len(harmonized)),
            "avg_confidence": sum(h.confidence for h in harmonized) / max(1, len(harmonized)),
            "conflict_resolution_rate": resolved / max(1, total_conflicts),
            "fields_harmonized": len(harmonized),
            "total_sources": sum(len(h.sources) for h in harmonized)
        }

    def _identify_remaining_issues(
        self,
        harmonized: List[HarmonizedResult]
    ) -> List[Dict[str, Any]]:
        """Identify remaining unresolved issues."""
        issues = []

        for h in harmonized:
            if h.confidence < 0.5:
                issues.append({
                    "field": h.field,
                    "issue": "low_confidence",
                    "confidence": h.confidence
                })
            if h.strategy_used == ResolutionStrategy.MANUAL_REVIEW:
                issues.append({
                    "field": h.field,
                    "issue": "needs_review"
                })

        return issues

    def _determine_status(
        self,
        remaining_issues: List[Dict[str, Any]]
    ) -> HarmonizationStatus:
        """Determine harmonization status."""
        if not remaining_issues:
            return HarmonizationStatus.SUCCESS
        elif len(remaining_issues) <= 2:
            return HarmonizationStatus.PARTIAL
        elif any(i["issue"] == "needs_review" for i in remaining_issues):
            return HarmonizationStatus.DEFERRED
        else:
            return HarmonizationStatus.CONFLICT

    def _calculate_confidence(
        self,
        harmonized: List[HarmonizedResult],
        remaining_issues: List[Dict[str, Any]]
    ) -> float:
        """Calculate overall confidence."""
        if not harmonized:
            return 0.5

        base = sum(h.confidence for h in harmonized) / len(harmonized)
        penalty = len(remaining_issues) * 0.05

        return max(0.3, base - penalty)

    async def validate(self, result: ExtractionResult) -> bool:
        """Validate extraction result."""
        data = result.data
        return "golden_record" in data and "status" in data

    def get_dependencies(self) -> List[str]:
        """Get agent dependencies."""
        return ["elenktikos"]
