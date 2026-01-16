"""
BIBLOS v2 - WITNESS Agent

Defense agent for responding to challenges.
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


class DefenseType(Enum):
    """Types of defense responses."""
    ACCEPT = "accept"  # Accept the challenge
    REFUTE = "refute"  # Refute the challenge
    CLARIFY = "clarify"  # Clarify the position
    MODIFY = "modify"  # Modify the claim
    DEFER = "defer"  # Defer to expert review


class EvidenceType(Enum):
    """Types of supporting evidence."""
    TEXTUAL = "textual"  # From biblical text
    PATRISTIC = "patristic"  # From Church Fathers
    LINGUISTIC = "linguistic"  # From language analysis
    CONTEXTUAL = "contextual"  # From context
    SCHOLARLY = "scholarly"  # From scholarship
    METHODOLOGICAL = "methodological"  # From methodology


@dataclass
class Defense:
    """A defense response to a challenge."""
    challenge_id: str
    defense_type: DefenseType
    response: str
    supporting_evidence: List[Dict[str, Any]]
    confidence_adjustment: float
    resolution: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "challenge_id": self.challenge_id,
            "defense_type": self.defense_type.value,
            "response": self.response,
            "supporting_evidence": self.supporting_evidence,
            "confidence_adjustment": self.confidence_adjustment,
            "resolution": self.resolution
        }


class WitnessAgent(BaseExtractionAgent):
    """
    WITNESS - Defense agent.

    Performs:
    - Challenge response generation
    - Evidence gathering
    - Defense construction
    - Resolution proposal
    - Confidence recalibration
    """

    # Defense strategies by challenge type
    DEFENSE_STRATEGIES = {
        "evidence": ["cite_text", "reference_sources", "show_methodology"],
        "methodology": ["explain_approach", "cite_standards", "show_validation"],
        "interpretation": ["cite_fathers", "show_tradition", "explain_reasoning"],
        "completeness": ["acknowledge_limitation", "explain_scope", "suggest_future"],
        "confidence": ["justify_level", "show_basis", "adjust_if_warranted"],
        "consistency": ["explain_difference", "reconcile_views", "accept_if_valid"]
    }

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="witness",
                extraction_type=ExtractionType.VALIDATION,
                batch_size=100,
                min_confidence=0.7
            )
        super().__init__(config)
        self.logger = logging.getLogger("biblos.agents.witness")

    async def extract(
        self,
        verse_id: str,
        text: str,
        context: Dict[str, Any]
    ) -> ExtractionResult:
        """Generate defense responses to challenges."""
        challenges = context.get("challenges", [])
        agent_results = context.get("agent_results", {})

        # Generate defenses for each challenge
        defenses = []
        for i, challenge in enumerate(challenges):
            defense = self._generate_defense(
                f"challenge_{i}",
                challenge,
                agent_results,
                text
            )
            defenses.append(defense)

        # Calculate resolution statistics
        stats = self._calculate_stats(defenses)

        # Generate recommended actions
        actions = self._generate_actions(defenses, challenges)

        # Calculate final confidence adjustments
        adjustments = self._calculate_adjustments(defenses)

        data = {
            "verse_id": verse_id,
            "defenses": [d.to_dict() for d in defenses],
            "defense_count": len(defenses),
            "accepted_count": sum(1 for d in defenses if d.defense_type == DefenseType.ACCEPT),
            "refuted_count": sum(1 for d in defenses if d.defense_type == DefenseType.REFUTE),
            "statistics": stats,
            "recommended_actions": actions,
            "confidence_adjustments": adjustments
        }

        confidence = self._calculate_confidence(defenses)

        return ExtractionResult(
            agent_name=self.config.name,
            extraction_type=self.config.extraction_type,
            verse_id=verse_id,
            status=ProcessingStatus.COMPLETED,
            data=data,
            confidence=confidence
        )

    def _generate_defense(
        self,
        challenge_id: str,
        challenge: Dict[str, Any],
        agent_results: Dict[str, Any],
        text: str
    ) -> Defense:
        """Generate a defense for a single challenge."""
        challenge_type = challenge.get("challenge_type", "evidence")
        target_agent = challenge.get("target_agent", "")
        target_field = challenge.get("target_field", "")
        severity = challenge.get("severity", "minor")

        # Get the challenged agent's result
        agent_result = agent_results.get(target_agent, {})
        agent_data = agent_result.get("data", {}) if isinstance(agent_result, dict) else {}

        # Determine defense strategy
        defense_type, response, evidence = self._construct_defense(
            challenge_type,
            target_field,
            agent_data,
            text,
            severity
        )

        # Calculate confidence adjustment
        conf_adjust = self._calculate_conf_adjustment(defense_type, severity)

        # Determine resolution
        resolution = self._determine_resolution(defense_type, severity)

        return Defense(
            challenge_id=challenge_id,
            defense_type=defense_type,
            response=response,
            supporting_evidence=evidence,
            confidence_adjustment=conf_adjust,
            resolution=resolution
        )

    def _construct_defense(
        self,
        challenge_type: str,
        target_field: str,
        agent_data: Dict[str, Any],
        text: str,
        severity: str
    ) -> tuple:
        """Construct defense response."""
        evidence = []

        # Try to find supporting evidence
        if challenge_type == "evidence":
            # Cite textual evidence
            if text:
                evidence.append({
                    "type": EvidenceType.TEXTUAL.value,
                    "content": f"Based on text: '{text[:100]}...'"
                })
                return (
                    DefenseType.CLARIFY,
                    f"The {target_field} is derived from textual analysis",
                    evidence
                )

        elif challenge_type == "methodology":
            evidence.append({
                "type": EvidenceType.METHODOLOGICAL.value,
                "content": "Standard linguistic analysis methodology applied"
            })
            return (
                DefenseType.CLARIFY,
                "Methodology follows established biblical studies practices",
                evidence
            )

        elif challenge_type == "interpretation":
            # Try to find patristic support
            if "patristic" in str(agent_data).lower():
                evidence.append({
                    "type": EvidenceType.PATRISTIC.value,
                    "content": "Supported by patristic tradition"
                })
                return (
                    DefenseType.REFUTE,
                    "Interpretation has patristic support",
                    evidence
                )
            else:
                return (
                    DefenseType.ACCEPT,
                    "Acknowledged - interpretation could benefit from more patristic grounding",
                    evidence
                )

        elif challenge_type == "completeness":
            # Accept completeness challenges as observations
            return (
                DefenseType.ACCEPT,
                f"Valid observation - {target_field} analysis could be expanded",
                evidence
            )

        elif challenge_type == "confidence":
            # Check if confidence is justified
            data_richness = len([v for v in agent_data.values() if v])
            if data_richness > 3:
                evidence.append({
                    "type": EvidenceType.CONTEXTUAL.value,
                    "content": f"Confidence based on {data_richness} data points"
                })
                return (
                    DefenseType.REFUTE,
                    "Confidence level is justified by data richness",
                    evidence
                )
            else:
                return (
                    DefenseType.MODIFY,
                    "Confidence level will be recalibrated",
                    evidence
                )

        elif challenge_type == "consistency":
            # Consistency issues often need acceptance
            if severity == "critical":
                return (
                    DefenseType.ACCEPT,
                    "Critical consistency issue acknowledged - requires resolution",
                    evidence
                )
            else:
                return (
                    DefenseType.CLARIFY,
                    "Apparent inconsistency may reflect different analytical perspectives",
                    evidence
                )

        # Default response
        return (
            DefenseType.DEFER,
            "Challenge requires expert review",
            evidence
        )

    def _calculate_conf_adjustment(
        self,
        defense_type: DefenseType,
        severity: str
    ) -> float:
        """Calculate confidence adjustment based on defense."""
        adjustments = {
            DefenseType.ACCEPT: -0.1 if severity == "critical" else -0.05,
            DefenseType.REFUTE: 0.05,
            DefenseType.CLARIFY: 0.0,
            DefenseType.MODIFY: -0.05,
            DefenseType.DEFER: -0.1
        }
        return adjustments.get(defense_type, 0.0)

    def _determine_resolution(
        self,
        defense_type: DefenseType,
        severity: str
    ) -> str:
        """Determine resolution status."""
        if defense_type == DefenseType.REFUTE:
            return "resolved_defended"
        elif defense_type == DefenseType.ACCEPT:
            return "resolved_accepted"
        elif defense_type == DefenseType.MODIFY:
            return "resolved_modified"
        elif defense_type == DefenseType.CLARIFY:
            return "resolved_clarified"
        else:
            return "pending_review"

    def _calculate_stats(
        self,
        defenses: List[Defense]
    ) -> Dict[str, Any]:
        """Calculate defense statistics."""
        if not defenses:
            return {
                "total": 0,
                "resolution_rate": 0.0
            }

        by_type = {}
        for d in defenses:
            t = d.defense_type.value
            by_type[t] = by_type.get(t, 0) + 1

        resolved = sum(
            1 for d in defenses
            if d.resolution.startswith("resolved")
        )

        return {
            "total": len(defenses),
            "by_type": by_type,
            "resolution_rate": resolved / max(1, len(defenses)),
            "avg_confidence_adjustment": sum(d.confidence_adjustment for d in defenses) / max(1, len(defenses))
        }

    def _generate_actions(
        self,
        defenses: List[Defense],
        challenges: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Generate recommended actions."""
        actions = []

        for defense in defenses:
            if defense.defense_type == DefenseType.ACCEPT:
                actions.append({
                    "action": "revise",
                    "target": defense.challenge_id,
                    "description": f"Revise based on accepted challenge"
                })
            elif defense.defense_type == DefenseType.MODIFY:
                actions.append({
                    "action": "modify",
                    "target": defense.challenge_id,
                    "description": "Modify claim based on challenge feedback"
                })
            elif defense.defense_type == DefenseType.DEFER:
                actions.append({
                    "action": "review",
                    "target": defense.challenge_id,
                    "description": "Schedule for expert review"
                })

        return actions

    def _calculate_adjustments(
        self,
        defenses: List[Defense]
    ) -> Dict[str, float]:
        """Calculate final confidence adjustments by agent."""
        adjustments = {}

        for defense in defenses:
            # Extract agent from challenge_id (simplified)
            agent = "general"  # Would extract from actual challenge data

            if agent not in adjustments:
                adjustments[agent] = 0.0
            adjustments[agent] += defense.confidence_adjustment

        return adjustments

    def _calculate_confidence(
        self,
        defenses: List[Defense]
    ) -> float:
        """Calculate witness agent confidence."""
        if not defenses:
            return 0.7

        # Higher confidence if more defenses were successful
        successful = sum(
            1 for d in defenses
            if d.defense_type in [DefenseType.REFUTE, DefenseType.CLARIFY]
        )

        success_rate = successful / max(1, len(defenses))
        return min(1.0, 0.5 + success_rate * 0.4)

    async def validate(self, result: ExtractionResult) -> bool:
        """Validate extraction result."""
        data = result.data
        return "defenses" in data and "statistics" in data

    def get_dependencies(self) -> List[str]:
        """Get agent dependencies."""
        return ["prosecutor"]
