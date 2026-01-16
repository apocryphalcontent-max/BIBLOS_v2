"""
BIBLOS v2 - PROSECUTOR Agent

Challenge agent for testing extraction robustness.
Integrates theological constraint validation for patristic challenges.
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

from ml.validators import (
    ConstraintType,
    ConstraintViolationSeverity,
)


class ChallengeType(Enum):
    """Types of challenges to pose."""
    # Original challenge types
    EVIDENCE = "evidence"  # Challenge evidence basis
    METHODOLOGY = "methodology"  # Challenge approach
    INTERPRETATION = "interpretation"  # Challenge interpretation
    COMPLETENESS = "completeness"  # Challenge coverage
    CONFIDENCE = "confidence"  # Challenge confidence level
    CONSISTENCY = "consistency"  # Challenge internal consistency

    # Theological constraint-based challenges
    CHRONOLOGICAL_PRIORITY = "chronological_priority"  # Type must precede antitype
    TYPOLOGICAL_ESCALATION = "typological_escalation"  # Antitype must exceed type
    PROPHETIC_COHERENCE = "prophetic_coherence"  # Fulfillment extends promise
    CHRISTOLOGICAL_WARRANT = "christological_warrant"  # Requires patristic support
    LITURGICAL_AMPLIFICATION = "liturgical_amplification"  # Liturgical usage issues
    FOURFOLD_FOUNDATION = "fourfold_foundation"  # Allegorical needs literal base


class ChallengeSeverity(Enum):
    """Severity of challenge."""
    CRITICAL = "critical"  # Must be addressed
    MAJOR = "major"  # Should be addressed
    MINOR = "minor"  # Could be improved
    OBSERVATION = "observation"  # For consideration


@dataclass
class Challenge:
    """A challenge to an extraction result."""
    challenge_type: ChallengeType
    severity: ChallengeSeverity
    target_agent: str
    target_field: str
    challenge_text: str
    evidence: List[str]
    suggested_action: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "challenge_type": self.challenge_type.value,
            "severity": self.severity.value,
            "target_agent": self.target_agent,
            "target_field": self.target_field,
            "challenge_text": self.challenge_text,
            "evidence": self.evidence,
            "suggested_action": self.suggested_action
        }


class ProsecutorAgent(BaseExtractionAgent):
    """
    PROSECUTOR - Challenge agent.

    Performs:
    - Evidence challenging
    - Methodology critique
    - Interpretation questioning
    - Confidence examination
    - Robustness testing
    """

    # Challenge templates by type
    CHALLENGE_TEMPLATES = {
        ChallengeType.EVIDENCE: [
            "What is the textual basis for {claim}?",
            "How does {data} support this conclusion?",
            "Are there alternative readings that contradict {value}?"
        ],
        ChallengeType.METHODOLOGY: [
            "Why was {method} chosen over alternatives?",
            "Is the approach to {field} standard in the field?",
            "Could a different methodology yield different results?"
        ],
        ChallengeType.INTERPRETATION: [
            "Are there patristic interpretations that disagree with {interpretation}?",
            "How certain is the theological reading of {concept}?",
            "Could this be interpreted differently in context?"
        ],
        ChallengeType.COMPLETENESS: [
            "Are there missing elements in the {category} analysis?",
            "Why were only {count} items identified?",
            "Should additional {type} connections be explored?"
        ],
        ChallengeType.CONFIDENCE: [
            "Is {confidence}% confidence justified given the data?",
            "What would increase confidence in {claim}?",
            "Why is confidence lower than expected for {type}?"
        ],
        ChallengeType.CONSISTENCY: [
            "How does {value1} reconcile with {value2}?",
            "Why do {agent1} and {agent2} disagree on {field}?",
            "Is there an explanation for the discrepancy in {area}?"
        ],
        # Theological constraint challenge templates
        ChallengeType.CHRONOLOGICAL_PRIORITY: [
            "Does {type_ref} historically precede {antitype_ref}?",
            "Is the claimed type-antitype ordering chronologically valid?",
            "How can {target} be antitype to {source} if it predates it?"
        ],
        ChallengeType.TYPOLOGICAL_ESCALATION: [
            "Does {antitype} exceed {type} in scope and magnitude?",
            "Is the escalation from {scope1} to {scope2} sufficient?",
            "Why is antitype not greater than type in this connection?"
        ],
        ChallengeType.PROPHETIC_COHERENCE: [
            "Does the fulfillment extend the promise without contradiction?",
            "How does {fulfillment} coherently fulfill {prophecy}?",
            "Is there semantic drift between promise and fulfillment?"
        ],
        ChallengeType.CHRISTOLOGICAL_WARRANT: [
            "What patristic witness supports this Christological reading?",
            "Is there apostolic precedent for this connection?",
            "Does this interpretation have consensus among the Fathers?"
        ],
        ChallengeType.LITURGICAL_AMPLIFICATION: [
            "Is this connection used liturgically?",
            "How does liturgical practice validate this reading?",
            "Should liturgical absence weaken confidence?"
        ],
        ChallengeType.FOURFOLD_FOUNDATION: [
            "Is there a literal foundation for this allegorical reading?",
            "What historical event grounds this typological claim?",
            "Does the spiritual sense rest on literal meaning?"
        ]
    }

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="prosecutor",
                extraction_type=ExtractionType.VALIDATION,
                batch_size=100,
                min_confidence=0.7
            )
        super().__init__(config)
        self.logger = logging.getLogger("biblos.agents.prosecutor")

    async def extract(
        self,
        verse_id: str,
        text: str,
        context: Dict[str, Any]
    ) -> ExtractionResult:
        """Generate challenges for extraction results."""
        agent_results = context.get("agent_results", {})
        consistency_report = context.get("consistency_report", {})

        # Generate challenges
        challenges = []

        # Challenge evidence
        challenges.extend(self._challenge_evidence(agent_results))

        # Challenge methodology
        challenges.extend(self._challenge_methodology(agent_results))

        # Challenge interpretations
        challenges.extend(self._challenge_interpretations(agent_results))

        # Challenge completeness
        challenges.extend(self._challenge_completeness(agent_results))

        # Challenge confidence
        challenges.extend(self._challenge_confidence(agent_results))

        # Challenge consistency
        challenges.extend(self._challenge_consistency(consistency_report))

        # Challenge theological constraints
        challenges.extend(self._challenge_theological_constraints(context))

        # Prioritize challenges
        prioritized = self._prioritize_challenges(challenges)

        # Generate summary
        summary = self._generate_summary(prioritized)

        data = {
            "verse_id": verse_id,
            "challenges": [c.to_dict() for c in prioritized],
            "challenge_count": len(prioritized),
            "critical_count": sum(1 for c in prioritized if c.severity == ChallengeSeverity.CRITICAL),
            "major_count": sum(1 for c in prioritized if c.severity == ChallengeSeverity.MAJOR),
            "summary": summary,
            "agents_challenged": list(set(c.target_agent for c in prioritized))
        }

        confidence = self._calculate_confidence(prioritized)

        return ExtractionResult(
            agent_name=self.config.name,
            extraction_type=self.config.extraction_type,
            verse_id=verse_id,
            status=ProcessingStatus.COMPLETED,
            data=data,
            confidence=confidence
        )

    def _challenge_evidence(
        self,
        agent_results: Dict[str, Any]
    ) -> List[Challenge]:
        """Challenge evidence basis of results."""
        challenges = []

        for agent_name, result in agent_results.items():
            data = result.get("data", {}) if isinstance(result, dict) else {}

            # Check for claims without clear evidence
            for field, value in data.items():
                if self._lacks_evidence(field, value):
                    challenges.append(Challenge(
                        challenge_type=ChallengeType.EVIDENCE,
                        severity=ChallengeSeverity.MAJOR,
                        target_agent=agent_name,
                        target_field=field,
                        challenge_text=f"What is the textual basis for '{field}'?",
                        evidence=[],
                        suggested_action="Provide textual citations supporting this claim"
                    ))

        return challenges

    def _challenge_methodology(
        self,
        agent_results: Dict[str, Any]
    ) -> List[Challenge]:
        """Challenge methodology of analysis."""
        challenges = []

        for agent_name, result in agent_results.items():
            data = result.get("data", {}) if isinstance(result, dict) else {}

            # Check for potentially problematic methodologies
            if "heuristic" in str(data).lower():
                challenges.append(Challenge(
                    challenge_type=ChallengeType.METHODOLOGY,
                    severity=ChallengeSeverity.MINOR,
                    target_agent=agent_name,
                    target_field="methodology",
                    challenge_text="Heuristic methods may miss nuances",
                    evidence=["Heuristic detection used"],
                    suggested_action="Consider validation with lexical resources"
                ))

        return challenges

    def _challenge_interpretations(
        self,
        agent_results: Dict[str, Any]
    ) -> List[Challenge]:
        """Challenge theological interpretations."""
        challenges = []

        theological_agents = ["patrologos", "theologos", "typologos"]

        for agent_name in theological_agents:
            if agent_name in agent_results:
                result = agent_results[agent_name]
                data = result.get("data", {}) if isinstance(result, dict) else {}

                # Challenge single-interpretation claims
                themes = data.get("themes", [])
                if len(themes) == 1:
                    challenges.append(Challenge(
                        challenge_type=ChallengeType.INTERPRETATION,
                        severity=ChallengeSeverity.OBSERVATION,
                        target_agent=agent_name,
                        target_field="themes",
                        challenge_text="Only one theme identified - could there be others?",
                        evidence=[f"Single theme: {themes[0]}"],
                        suggested_action="Explore additional thematic connections"
                    ))

                # Challenge assertions without patristic support
                assertions = data.get("assertions", [])
                for assertion in assertions:
                    if isinstance(assertion, dict) and not assertion.get("patristic_witness"):
                        challenges.append(Challenge(
                            challenge_type=ChallengeType.INTERPRETATION,
                            severity=ChallengeSeverity.MINOR,
                            target_agent=agent_name,
                            target_field="assertions",
                            challenge_text="Theological assertion lacks patristic witness",
                            evidence=[f"Assertion without witness: {assertion.get('doctrine', '')}"],
                            suggested_action="Identify patristic support for this interpretation"
                        ))

        return challenges

    def _challenge_completeness(
        self,
        agent_results: Dict[str, Any]
    ) -> List[Challenge]:
        """Challenge completeness of analysis."""
        challenges = []

        for agent_name, result in agent_results.items():
            data = result.get("data", {}) if isinstance(result, dict) else {}

            # Check for suspiciously empty results
            for field, value in data.items():
                if isinstance(value, list) and len(value) == 0:
                    challenges.append(Challenge(
                        challenge_type=ChallengeType.COMPLETENESS,
                        severity=ChallengeSeverity.MAJOR,
                        target_agent=agent_name,
                        target_field=field,
                        challenge_text=f"Field '{field}' is empty - is this expected?",
                        evidence=[f"Empty list for {field}"],
                        suggested_action="Verify no items should be present or investigate further"
                    ))

        return challenges

    def _challenge_confidence(
        self,
        agent_results: Dict[str, Any]
    ) -> List[Challenge]:
        """Challenge confidence levels."""
        challenges = []

        for agent_name, result in agent_results.items():
            confidence = result.get("confidence", 0) if isinstance(result, dict) else 0
            data = result.get("data", {}) if isinstance(result, dict) else {}

            # Challenge high confidence with sparse data
            data_richness = len([v for v in data.values() if v])
            if confidence > 0.85 and data_richness < 3:
                challenges.append(Challenge(
                    challenge_type=ChallengeType.CONFIDENCE,
                    severity=ChallengeSeverity.MINOR,
                    target_agent=agent_name,
                    target_field="confidence",
                    challenge_text=f"High confidence ({confidence:.0%}) with limited data",
                    evidence=[f"Data fields: {data_richness}"],
                    suggested_action="Verify confidence is warranted"
                ))

            # Challenge very low confidence
            if confidence < 0.4:
                challenges.append(Challenge(
                    challenge_type=ChallengeType.CONFIDENCE,
                    severity=ChallengeSeverity.OBSERVATION,
                    target_agent=agent_name,
                    target_field="confidence",
                    challenge_text=f"Low confidence ({confidence:.0%}) - what could improve it?",
                    evidence=[],
                    suggested_action="Identify factors limiting confidence"
                ))

        return challenges

    def _challenge_consistency(
        self,
        consistency_report: Dict[str, Any]
    ) -> List[Challenge]:
        """Challenge consistency issues."""
        challenges = []
        conflicts = consistency_report.get("conflicts", [])

        for conflict in conflicts:
            agents = conflict.get("agents", [])
            claims = conflict.get("claims", [])

            for claim in claims:
                challenges.append(Challenge(
                    challenge_type=ChallengeType.CONSISTENCY,
                    severity=ChallengeSeverity.CRITICAL,
                    target_agent=", ".join(agents),
                    target_field=claim.get("field", ""),
                    challenge_text=f"Conflict between agents on '{claim.get('field', '')}'",
                    evidence=[
                        f"Value 1: {claim.get('agent1_value', '')}",
                        f"Value 2: {claim.get('agent2_value', '')}"
                    ],
                    suggested_action="Resolve conflict through expert review"
                ))

        return challenges

    def _challenge_theological_constraints(
        self,
        context: Dict[str, Any]
    ) -> List[Challenge]:
        """
        Challenge based on theological constraint validation results.

        Examines constraint_results from postprocessor and generates
        challenges for any violations or warnings.
        """
        challenges = []

        # Get inference results that contain constraint validations
        inference_results = context.get("inference_results", [])

        for result in inference_results:
            constraint_results = result.get("constraint_results", [])
            source = result.get("source_verse", "unknown")
            target = result.get("target_verse", "unknown")

            for cr in constraint_results:
                if not cr.get("passed", True):
                    # Map constraint type to challenge type
                    constraint_type = cr.get("constraint_type", "")
                    challenge_type = self._map_constraint_to_challenge(constraint_type)

                    if challenge_type:
                        # Determine severity based on violation severity
                        severity = self._map_constraint_severity(cr.get("severity"))

                        challenges.append(Challenge(
                            challenge_type=challenge_type,
                            severity=severity,
                            target_agent="inference_pipeline",
                            target_field=f"{source}->{target}",
                            challenge_text=cr.get("message", f"Constraint {constraint_type} violated"),
                            evidence=[
                                f"Source: {source}",
                                f"Target: {target}",
                                f"Confidence modifier: {cr.get('confidence_modifier', 1.0)}"
                            ],
                            suggested_action=self._get_constraint_action(constraint_type)
                        ))

        return challenges

    def _map_constraint_to_challenge(self, constraint_type: str) -> Optional[ChallengeType]:
        """Map constraint type string to ChallengeType enum."""
        mapping = {
            "CHRONOLOGICAL_PRIORITY": ChallengeType.CHRONOLOGICAL_PRIORITY,
            "TYPOLOGICAL_ESCALATION": ChallengeType.TYPOLOGICAL_ESCALATION,
            "PROPHETIC_COHERENCE": ChallengeType.PROPHETIC_COHERENCE,
            "CHRISTOLOGICAL_WARRANT": ChallengeType.CHRISTOLOGICAL_WARRANT,
            "LITURGICAL_AMPLIFICATION": ChallengeType.LITURGICAL_AMPLIFICATION,
            "FOURFOLD_FOUNDATION": ChallengeType.FOURFOLD_FOUNDATION,
        }
        return mapping.get(constraint_type)

    def _map_constraint_severity(self, severity: Optional[str]) -> ChallengeSeverity:
        """Map constraint violation severity to challenge severity."""
        if severity == "IMPOSSIBLE":
            return ChallengeSeverity.CRITICAL
        elif severity == "CRITICAL":
            return ChallengeSeverity.CRITICAL
        elif severity == "SOFT":
            return ChallengeSeverity.MAJOR
        elif severity == "WARNING":
            return ChallengeSeverity.MINOR
        else:
            return ChallengeSeverity.OBSERVATION

    def _get_constraint_action(self, constraint_type: str) -> str:
        """Get suggested action for a constraint violation."""
        actions = {
            "CHRONOLOGICAL_PRIORITY": "Verify historical ordering of type and antitype",
            "TYPOLOGICAL_ESCALATION": "Assess scope/magnitude relationship between type and antitype",
            "PROPHETIC_COHERENCE": "Check semantic alignment between prophecy and fulfillment",
            "CHRISTOLOGICAL_WARRANT": "Locate patristic witness supporting this interpretation",
            "LITURGICAL_AMPLIFICATION": "Research liturgical usage of this connection",
            "FOURFOLD_FOUNDATION": "Identify literal/historical foundation for allegorical reading",
        }
        return actions.get(constraint_type, "Review theological constraint validation")

    def _lacks_evidence(self, field: str, value: Any) -> bool:
        """Check if a field value lacks evidence."""
        # Simple heuristic checks
        if value is None:
            return False  # None isn't a claim

        if isinstance(value, list) and len(value) == 0:
            return False  # Empty isn't a claim

        # Check for high-confidence claims without backing
        evidence_fields = ["sources", "references", "citations", "witness"]
        if field.lower() in ["interpretation", "assertion", "claim"]:
            return True

        return False

    def _prioritize_challenges(
        self,
        challenges: List[Challenge]
    ) -> List[Challenge]:
        """Prioritize challenges by severity."""
        severity_order = {
            ChallengeSeverity.CRITICAL: 0,
            ChallengeSeverity.MAJOR: 1,
            ChallengeSeverity.MINOR: 2,
            ChallengeSeverity.OBSERVATION: 3
        }

        return sorted(challenges, key=lambda c: severity_order[c.severity])

    def _generate_summary(
        self,
        challenges: List[Challenge]
    ) -> Dict[str, Any]:
        """Generate challenge summary."""
        by_type = {}
        by_severity = {}

        for c in challenges:
            t = c.challenge_type.value
            s = c.severity.value
            by_type[t] = by_type.get(t, 0) + 1
            by_severity[s] = by_severity.get(s, 0) + 1

        return {
            "total": len(challenges),
            "by_type": by_type,
            "by_severity": by_severity,
            "most_challenged_agent": self._most_common_agent(challenges)
        }

    def _most_common_agent(self, challenges: List[Challenge]) -> Optional[str]:
        """Find most challenged agent."""
        if not challenges:
            return None

        agents = [c.target_agent for c in challenges]
        agent_counts = {}
        for a in agents:
            agent_counts[a] = agent_counts.get(a, 0) + 1

        return max(agent_counts, key=lambda k: agent_counts[k]) if agent_counts else None

    def _calculate_confidence(
        self,
        challenges: List[Challenge]
    ) -> float:
        """Calculate confidence in challenge assessment."""
        # More challenges found = more confident in assessment
        base = 0.7
        challenge_bonus = min(0.2, len(challenges) * 0.02)

        return min(1.0, base + challenge_bonus)

    async def validate(self, result: ExtractionResult) -> bool:
        """Validate extraction result."""
        data = result.data
        return "challenges" in data and "summary" in data

    def get_dependencies(self) -> List[str]:
        """Get agent dependencies."""
        return ["elenktikos"]
