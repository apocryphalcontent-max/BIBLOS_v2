"""
BIBLOS v2 - ELENKTIKOS Agent

Cross-agent consistency validation for extraction results.
"""
from typing import Dict, List, Any, Optional, Set
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


class ConsistencyLevel(Enum):
    """Levels of consistency between agents."""
    FULL = "full"  # Complete agreement
    PARTIAL = "partial"  # Some agreement
    CONFLICT = "conflict"  # Disagreement
    INDEPENDENT = "independent"  # No overlap


class ValidationIssue(Enum):
    """Types of validation issues."""
    CONTRADICTION = "contradiction"  # Contradictory claims
    MISSING_DATA = "missing_data"  # Expected data missing
    INCONSISTENT_REFERENCE = "inconsistent_reference"  # Reference mismatch
    CONFIDENCE_MISMATCH = "confidence_mismatch"  # Confidence disagreement
    SCHEMA_VIOLATION = "schema_violation"  # Schema not followed
    DEPENDENCY_ERROR = "dependency_error"  # Dependency not met


@dataclass
class ConsistencyReport:
    """Report on consistency between agent results."""
    agent_pair: tuple
    consistency_level: ConsistencyLevel
    issues: List[ValidationIssue]
    shared_claims: List[str]
    conflicting_claims: List[Dict[str, Any]]
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_pair": list(self.agent_pair),
            "consistency_level": self.consistency_level.value,
            "issues": [i.value for i in self.issues],
            "shared_claims": self.shared_claims,
            "conflicting_claims": self.conflicting_claims,
            "score": self.score
        }


class ElenktikosAgent(BaseExtractionAgent):
    """
    ELENKTIKOS - Consistency validation agent.

    Performs:
    - Cross-agent result comparison
    - Consistency scoring
    - Conflict detection
    - Dependency validation
    - Schema compliance checking
    """

    # Expected fields by extraction type
    REQUIRED_FIELDS = {
        ExtractionType.STRUCTURAL: ["tokens", "text_type", "word_count"],
        ExtractionType.MORPHOLOGICAL: ["analyses", "language"],
        ExtractionType.SYNTACTIC: ["dependencies", "clauses"],
        ExtractionType.SEMANTIC: ["frames", "domains"],
        ExtractionType.THEOLOGICAL: ["categories", "assertions"],
        ExtractionType.TYPOLOGICAL: ["connections", "is_ot"],
        ExtractionType.INTERTEXTUAL: ["cross_references"],
        ExtractionType.LITURGICAL: ["contexts", "usages"]
    }

    # Agent dependencies
    AGENT_DEPENDENCIES = {
        "morphologos": ["grammateus"],
        "syntaktikos": ["grammateus", "morphologos"],
        "semantikos": ["grammateus", "morphologos", "syntaktikos"],
        "patrologos": ["grammateus", "semantikos"],
        "typologos": ["grammateus", "semantikos", "patrologos"],
        "theologos": ["grammateus", "semantikos"],
        "syndesmos": ["grammateus", "semantikos", "typologos"],
        "harmonikos": ["grammateus", "syndesmos"],
        "allographos": ["grammateus", "syndesmos"]
    }

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="elenktikos",
                extraction_type=ExtractionType.VALIDATION,
                batch_size=100,
                min_confidence=0.7
            )
        super().__init__(config)
        self.logger = logging.getLogger("biblos.agents.elenktikos")

    async def extract(
        self,
        verse_id: str,
        text: str,
        context: Dict[str, Any]
    ) -> ExtractionResult:
        """Validate consistency across agent results."""
        # Get all agent results from context
        agent_results = context.get("agent_results", {})

        # Check dependencies
        dependency_issues = self._check_dependencies(agent_results)

        # Validate schema compliance
        schema_issues = self._validate_schemas(agent_results)

        # Compare agent pairs
        consistency_reports = self._compare_agents(agent_results)

        # Calculate overall consistency
        overall = self._calculate_overall_consistency(consistency_reports)

        # Identify conflicts
        conflicts = self._identify_conflicts(consistency_reports)

        data = {
            "verse_id": verse_id,
            "agents_validated": list(agent_results.keys()),
            "dependency_issues": dependency_issues,
            "schema_issues": schema_issues,
            "consistency_reports": [r.to_dict() for r in consistency_reports],
            "overall_consistency": overall,
            "conflicts": conflicts,
            "validation_passed": len(conflicts) == 0 and len(schema_issues) == 0
        }

        confidence = self._calculate_confidence(
            dependency_issues, schema_issues, consistency_reports
        )

        return ExtractionResult(
            agent_name=self.config.name,
            extraction_type=self.config.extraction_type,
            verse_id=verse_id,
            status=ProcessingStatus.COMPLETED,
            data=data,
            confidence=confidence
        )

    def _check_dependencies(
        self,
        agent_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check that agent dependencies are satisfied."""
        issues = []

        for agent, deps in self.AGENT_DEPENDENCIES.items():
            if agent in agent_results:
                for dep in deps:
                    if dep not in agent_results:
                        issues.append({
                            "agent": agent,
                            "missing_dependency": dep,
                            "issue_type": ValidationIssue.DEPENDENCY_ERROR.value
                        })

        return issues

    def _validate_schemas(
        self,
        agent_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Validate that results follow expected schemas."""
        issues = []

        for agent_name, result in agent_results.items():
            if not isinstance(result, dict):
                continue

            data = result.get("data", {})
            extraction_type = result.get("extraction_type")

            # Check required fields
            if extraction_type in self.REQUIRED_FIELDS:
                required = self.REQUIRED_FIELDS[extraction_type]
                missing = [f for f in required if f not in data]
                if missing:
                    issues.append({
                        "agent": agent_name,
                        "missing_fields": missing,
                        "issue_type": ValidationIssue.SCHEMA_VIOLATION.value
                    })

        return issues

    def _compare_agents(
        self,
        agent_results: Dict[str, Any]
    ) -> List[ConsistencyReport]:
        """Compare pairs of agent results for consistency."""
        reports = []
        agents = list(agent_results.keys())

        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                agent1, agent2 = agents[i], agents[j]
                result1 = agent_results[agent1]
                result2 = agent_results[agent2]

                report = self._compare_pair(
                    agent1, result1, agent2, result2
                )
                reports.append(report)

        return reports

    def _compare_pair(
        self,
        agent1: str,
        result1: Dict[str, Any],
        agent2: str,
        result2: Dict[str, Any]
    ) -> ConsistencyReport:
        """Compare a pair of agent results."""
        data1 = result1.get("data", {}) if isinstance(result1, dict) else {}
        data2 = result2.get("data", {}) if isinstance(result2, dict) else {}

        # Find shared keys
        keys1 = set(data1.keys())
        keys2 = set(data2.keys())
        shared_keys = keys1 & keys2

        issues = []
        shared_claims = []
        conflicting_claims = []

        for key in shared_keys:
            val1 = data1[key]
            val2 = data2[key]

            if val1 == val2:
                shared_claims.append(f"{key}: {val1}")
            elif self._is_conflict(val1, val2):
                conflicting_claims.append({
                    "field": key,
                    "agent1_value": str(val1)[:100],
                    "agent2_value": str(val2)[:100]
                })
                issues.append(ValidationIssue.CONTRADICTION)

        # Check confidence consistency
        conf1 = result1.get("confidence", 0) if isinstance(result1, dict) else 0
        conf2 = result2.get("confidence", 0) if isinstance(result2, dict) else 0
        if abs(conf1 - conf2) > 0.4:
            issues.append(ValidationIssue.CONFIDENCE_MISMATCH)

        # Determine consistency level
        if not conflicting_claims and shared_claims:
            level = ConsistencyLevel.FULL
        elif conflicting_claims and shared_claims:
            level = ConsistencyLevel.PARTIAL
        elif conflicting_claims:
            level = ConsistencyLevel.CONFLICT
        else:
            level = ConsistencyLevel.INDEPENDENT

        # Calculate score
        if not shared_keys:
            score = 1.0  # Independent, no comparison possible
        elif not shared_claims and not conflicting_claims:
            score = 1.0  # No claims to compare, assume consistent
        else:
            score = len(shared_claims) / max(1, len(shared_claims) + len(conflicting_claims))

        return ConsistencyReport(
            agent_pair=(agent1, agent2),
            consistency_level=level,
            issues=list(set(issues)),
            shared_claims=shared_claims[:10],
            conflicting_claims=conflicting_claims[:5],
            score=score
        )

    def _is_conflict(self, val1: Any, val2: Any) -> bool:
        """Determine if two values are in conflict."""
        # Simple conflict detection
        if val1 is None or val2 is None:
            return False

        if isinstance(val1, (list, set)) and isinstance(val2, (list, set)):
            # Lists conflict if they have no overlap
            set1, set2 = set(str(v) for v in val1), set(str(v) for v in val2)
            if set1 and set2 and not (set1 & set2):
                return True
            return False

        if isinstance(val1, dict) and isinstance(val2, dict):
            # Dicts don't conflict for this simple check
            return False

        # Scalar values conflict if different
        return str(val1) != str(val2)

    def _calculate_overall_consistency(
        self,
        reports: List[ConsistencyReport]
    ) -> Dict[str, Any]:
        """Calculate overall consistency metrics."""
        if not reports:
            return {
                "score": 1.0,
                "level": "full",
                "conflict_count": 0
            }

        scores = [r.score for r in reports]
        avg_score = sum(scores) / len(scores)

        conflict_count = sum(
            1 for r in reports
            if r.consistency_level == ConsistencyLevel.CONFLICT
        )

        if avg_score > 0.9:
            level = "full"
        elif avg_score > 0.6:
            level = "partial"
        else:
            level = "conflict"

        return {
            "score": avg_score,
            "level": level,
            "conflict_count": conflict_count,
            "report_count": len(reports)
        }

    def _identify_conflicts(
        self,
        reports: List[ConsistencyReport]
    ) -> List[Dict[str, Any]]:
        """Identify and list all conflicts."""
        conflicts = []

        for report in reports:
            if report.conflicting_claims:
                conflicts.append({
                    "agents": list(report.agent_pair),
                    "claims": report.conflicting_claims
                })

        return conflicts

    def _calculate_confidence(
        self,
        dependency_issues: List[Dict[str, Any]],
        schema_issues: List[Dict[str, Any]],
        reports: List[ConsistencyReport]
    ) -> float:
        """Calculate validation confidence."""
        confidence = 0.9  # Start high

        # Reduce for issues
        confidence -= len(dependency_issues) * 0.1
        confidence -= len(schema_issues) * 0.1

        # Reduce for conflicts
        for report in reports:
            if report.consistency_level == ConsistencyLevel.CONFLICT:
                confidence -= 0.1

        return max(0.3, confidence)

    async def validate(self, result: ExtractionResult) -> bool:
        """Validate extraction result."""
        data = result.data
        return "agents_validated" in data and "validation_passed" in data

    def get_dependencies(self) -> List[str]:
        """Get agent dependencies."""
        return []  # Validation agent has no extraction dependencies
