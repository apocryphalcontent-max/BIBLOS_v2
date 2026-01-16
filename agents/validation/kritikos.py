"""
BIBLOS v2 - KRITIKOS Agent

Quality scoring and critique agent for extraction results.
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


class QualityDimension(Enum):
    """Dimensions of quality assessment."""
    COMPLETENESS = "completeness"  # All expected data present
    ACCURACY = "accuracy"  # Data is correct
    CONSISTENCY = "consistency"  # Internal consistency
    DEPTH = "depth"  # Level of analysis
    RELEVANCE = "relevance"  # Theological relevance
    CONFIDENCE = "confidence"  # Confidence appropriateness


class QualityLevel(Enum):
    """Quality levels."""
    EXCELLENT = "excellent"  # 0.9+
    GOOD = "good"  # 0.7-0.9
    ACCEPTABLE = "acceptable"  # 0.5-0.7
    POOR = "poor"  # 0.3-0.5
    UNACCEPTABLE = "unacceptable"  # <0.3


@dataclass
class QualityScore:
    """Quality score for a dimension."""
    dimension: QualityDimension
    score: float
    level: QualityLevel
    feedback: str
    suggestions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "score": self.score,
            "level": self.level.value,
            "feedback": self.feedback,
            "suggestions": self.suggestions
        }


class KritikosAgent(BaseExtractionAgent):
    """
    KRITIKOS - Quality assessment agent.

    Performs:
    - Quality scoring across dimensions
    - Critique generation
    - Improvement suggestions
    - Overall rating
    - Feedback synthesis
    """

    # Minimum expected fields by agent type
    COMPLETENESS_CRITERIA = {
        "grammateus": {
            "required": ["tokens", "text_type", "word_count"],
            "optional": ["rhetorical_devices", "clauses"]
        },
        "morphologos": {
            "required": ["analyses", "language"],
            "optional": ["summary"]
        },
        "syntaktikos": {
            "required": ["dependencies", "clauses"],
            "optional": ["word_order", "patterns"]
        },
        "semantikos": {
            "required": ["frames", "domains"],
            "optional": ["relationships", "key_concepts"]
        },
        "patrologos": {
            "required": ["themes", "patristic_references"],
            "optional": ["consensus", "liturgical_usage"]
        },
        "typologos": {
            "required": ["connections", "is_ot"],
            "optional": ["patterns", "christological_focus"]
        },
        "syndesmos": {
            "required": ["cross_references"],
            "optional": ["network_metrics", "clusters"]
        }
    }

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="kritikos",
                extraction_type=ExtractionType.VALIDATION,
                batch_size=100,
                min_confidence=0.7
            )
        super().__init__(config)
        self.logger = logging.getLogger("biblos.agents.kritikos")

    async def extract(
        self,
        verse_id: str,
        text: str,
        context: Dict[str, Any]
    ) -> ExtractionResult:
        """Generate quality assessment for extractions."""
        agent_results = context.get("agent_results", {})

        # Score each dimension for each agent
        agent_scores = {}
        for agent_name, result in agent_results.items():
            agent_scores[agent_name] = self._score_agent(agent_name, result)

        # Calculate aggregate scores
        aggregate = self._calculate_aggregate(agent_scores)

        # Generate critique
        critique = self._generate_critique(agent_scores)

        # Generate suggestions
        suggestions = self._generate_suggestions(agent_scores)

        # Calculate overall rating
        overall = self._calculate_overall(aggregate)

        data = {
            "verse_id": verse_id,
            "agent_scores": {
                name: [s.to_dict() for s in scores]
                for name, scores in agent_scores.items()
            },
            "aggregate_scores": aggregate,
            "critique": critique,
            "suggestions": suggestions,
            "overall_rating": overall,
            "quality_passed": overall["level"] in ["excellent", "good", "acceptable"]
        }

        confidence = overall["score"]

        return ExtractionResult(
            agent_name=self.config.name,
            extraction_type=self.config.extraction_type,
            verse_id=verse_id,
            status=ProcessingStatus.COMPLETED,
            data=data,
            confidence=confidence
        )

    def _score_agent(
        self,
        agent_name: str,
        result: Dict[str, Any]
    ) -> List[QualityScore]:
        """Score all dimensions for an agent."""
        scores = []
        data = result.get("data", {}) if isinstance(result, dict) else {}

        # Completeness
        scores.append(self._score_completeness(agent_name, data))

        # Accuracy (heuristic-based)
        scores.append(self._score_accuracy(data))

        # Consistency
        scores.append(self._score_consistency(data))

        # Depth
        scores.append(self._score_depth(data))

        # Confidence appropriateness
        conf = result.get("confidence", 0) if isinstance(result, dict) else 0
        scores.append(self._score_confidence(conf, data))

        return scores

    def _score_completeness(
        self,
        agent_name: str,
        data: Dict[str, Any]
    ) -> QualityScore:
        """Score completeness of data."""
        criteria = self.COMPLETENESS_CRITERIA.get(agent_name, {})
        required = criteria.get("required", [])
        optional = criteria.get("optional", [])

        if not required:
            return QualityScore(
                dimension=QualityDimension.COMPLETENESS,
                score=0.7,
                level=QualityLevel.GOOD,
                feedback="No specific completeness criteria defined",
                suggestions=[]
            )

        # Check required fields
        required_present = sum(1 for f in required if f in data)
        optional_present = sum(1 for f in optional if f in data)

        req_score = required_present / len(required) if required else 1.0
        opt_bonus = (optional_present / len(optional) * 0.2) if optional else 0

        score = min(1.0, req_score + opt_bonus)
        level = self._score_to_level(score)

        missing = [f for f in required if f not in data]
        suggestions = [f"Add missing field: {f}" for f in missing]

        return QualityScore(
            dimension=QualityDimension.COMPLETENESS,
            score=score,
            level=level,
            feedback=f"Present: {required_present}/{len(required)} required fields",
            suggestions=suggestions
        )

    def _score_accuracy(self, data: Dict[str, Any]) -> QualityScore:
        """Score accuracy of data (heuristic)."""
        # Check for common accuracy indicators
        issues = []

        # Check for empty values
        empty_count = sum(1 for v in data.values() if not v)
        if empty_count > 0:
            issues.append(f"{empty_count} empty values")

        # Check for None values
        none_count = sum(1 for v in data.values() if v is None)
        if none_count > 0:
            issues.append(f"{none_count} None values")

        # Score based on issues
        score = 1.0 - (len(issues) * 0.15)
        score = max(0.3, score)
        level = self._score_to_level(score)

        return QualityScore(
            dimension=QualityDimension.ACCURACY,
            score=score,
            level=level,
            feedback=f"Issues found: {len(issues)}" if issues else "No accuracy issues detected",
            suggestions=[f"Fix: {issue}" for issue in issues]
        )

    def _score_consistency(self, data: Dict[str, Any]) -> QualityScore:
        """Score internal consistency."""
        issues = []

        # Check for count mismatches
        if "word_count" in data and "tokens" in data:
            if isinstance(data["tokens"], list):
                if data["word_count"] != len(data["tokens"]):
                    issues.append("word_count doesn't match tokens length")

        # Check for reference format consistency
        for key, value in data.items():
            if "ref" in key.lower() and isinstance(value, str):
                if not self._is_valid_ref(value):
                    issues.append(f"Invalid reference format in {key}")

        score = 1.0 - (len(issues) * 0.2)
        score = max(0.3, score)
        level = self._score_to_level(score)

        return QualityScore(
            dimension=QualityDimension.CONSISTENCY,
            score=score,
            level=level,
            feedback=f"Consistency issues: {len(issues)}" if issues else "Data is internally consistent",
            suggestions=[f"Fix: {issue}" for issue in issues]
        )

    def _score_depth(self, data: Dict[str, Any]) -> QualityScore:
        """Score depth of analysis."""
        # Count non-trivial fields
        depth_indicators = 0

        for key, value in data.items():
            if isinstance(value, list) and len(value) > 0:
                depth_indicators += 1
            elif isinstance(value, dict) and len(value) > 0:
                depth_indicators += 1
            elif isinstance(value, str) and len(value) > 50:
                depth_indicators += 1

        # Score based on depth indicators
        score = min(1.0, depth_indicators * 0.15 + 0.4)
        level = self._score_to_level(score)

        return QualityScore(
            dimension=QualityDimension.DEPTH,
            score=score,
            level=level,
            feedback=f"Analysis depth indicators: {depth_indicators}",
            suggestions=["Consider adding more detailed analysis"] if score < 0.6 else []
        )

    def _score_confidence(
        self,
        confidence: float,
        data: Dict[str, Any]
    ) -> QualityScore:
        """Score appropriateness of confidence level."""
        # Check if confidence matches data richness
        data_richness = len([v for v in data.values() if v])

        expected_conf = min(1.0, data_richness * 0.1 + 0.4)
        diff = abs(confidence - expected_conf)

        if diff < 0.1:
            score = 1.0
            feedback = "Confidence well-calibrated"
        elif diff < 0.2:
            score = 0.8
            feedback = "Confidence slightly miscalibrated"
        else:
            score = 0.6
            feedback = "Confidence may be miscalibrated"

        level = self._score_to_level(score)
        suggestions = []
        if confidence > expected_conf + 0.2:
            suggestions.append("Consider lowering confidence")
        elif confidence < expected_conf - 0.2:
            suggestions.append("Consider raising confidence")

        return QualityScore(
            dimension=QualityDimension.CONFIDENCE,
            score=score,
            level=level,
            feedback=feedback,
            suggestions=suggestions
        )

    def _score_to_level(self, score: float) -> QualityLevel:
        """Convert score to quality level."""
        if score >= 0.9:
            return QualityLevel.EXCELLENT
        elif score >= 0.7:
            return QualityLevel.GOOD
        elif score >= 0.5:
            return QualityLevel.ACCEPTABLE
        elif score >= 0.3:
            return QualityLevel.POOR
        else:
            return QualityLevel.UNACCEPTABLE

    def _is_valid_ref(self, ref: str) -> bool:
        """Check if reference format is valid."""
        # Simple check for book.chapter.verse format
        parts = ref.split(".")
        return len(parts) >= 2

    def _calculate_aggregate(
        self,
        agent_scores: Dict[str, List[QualityScore]]
    ) -> Dict[str, float]:
        """Calculate aggregate scores across agents."""
        dimensions = {}

        for scores in agent_scores.values():
            for score in scores:
                dim = score.dimension.value
                if dim not in dimensions:
                    dimensions[dim] = []
                dimensions[dim].append(score.score)

        return {
            dim: sum(scores) / max(1, len(scores))
            for dim, scores in dimensions.items()
        }

    def _generate_critique(
        self,
        agent_scores: Dict[str, List[QualityScore]]
    ) -> List[Dict[str, Any]]:
        """Generate critique for each agent."""
        critiques = []

        for agent_name, scores in agent_scores.items():
            poor_scores = [s for s in scores if s.level in [QualityLevel.POOR, QualityLevel.UNACCEPTABLE]]

            if poor_scores:
                critiques.append({
                    "agent": agent_name,
                    "issues": [
                        {
                            "dimension": s.dimension.value,
                            "feedback": s.feedback
                        }
                        for s in poor_scores
                    ]
                })

        return critiques

    def _generate_suggestions(
        self,
        agent_scores: Dict[str, List[QualityScore]]
    ) -> List[Dict[str, Any]]:
        """Generate improvement suggestions."""
        all_suggestions = []

        for agent_name, scores in agent_scores.items():
            agent_suggestions = []
            for score in scores:
                agent_suggestions.extend(score.suggestions)

            if agent_suggestions:
                all_suggestions.append({
                    "agent": agent_name,
                    "suggestions": agent_suggestions[:5]  # Limit per agent
                })

        return all_suggestions

    def _calculate_overall(
        self,
        aggregate: Dict[str, float]
    ) -> Dict[str, Any]:
        """Calculate overall quality rating."""
        if not aggregate:
            return {"score": 0.5, "level": "acceptable"}

        scores = list(aggregate.values())
        avg_score = sum(scores) / max(1, len(scores))
        level = self._score_to_level(avg_score)

        return {
            "score": avg_score,
            "level": level.value,
            "dimensions_assessed": len(aggregate)
        }

    async def validate(self, result: ExtractionResult) -> bool:
        """Validate extraction result."""
        data = result.data
        return "overall_rating" in data and "quality_passed" in data

    def get_dependencies(self) -> List[str]:
        """Get agent dependencies."""
        return []
