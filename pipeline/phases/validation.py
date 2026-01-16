"""
Validation Phase

Phase 5: Validation with theological constraints and prosecutor/witness pattern.
Implements adversarial validation where challenges must be defended.
"""
import time
import logging
from typing import List, Tuple, Optional, Any
from enum import Enum
from dataclasses import dataclass

from pipeline.phases.base import Phase, PhasePriority, PhaseCategory, PhaseDependency
from pipeline.context import ProcessingContext


logger = logging.getLogger(__name__)


class ValidationVerdict(Enum):
    """Possible validation verdicts."""
    APPROVED = "approved"
    REJECTED = "rejected"
    DEFERRED = "deferred"  # Needs human review
    CHALLENGED = "challenged"  # Partially valid

    @property
    def allows_inclusion(self) -> bool:
        """Whether this verdict allows inclusion in golden record."""
        return self in {ValidationVerdict.APPROVED, ValidationVerdict.CHALLENGED}


@dataclass
class ChallengeResult:
    """Result from prosecutor challenge."""
    challenge_type: str
    severity: float  # 0-1, how serious the challenge is
    description: str
    supporting_evidence: List[str]

    @property
    def is_blocking(self) -> bool:
        """Challenge severe enough to block acceptance."""
        return self.severity > 0.7


@dataclass
class DefenseResult:
    """Result from witness defense."""
    defense_type: str
    strength: float  # 0-1, how strong the defense is
    description: str
    supporting_sources: List[str]

    @property
    def overcomes_challenge(self) -> bool:
        """Defense strong enough to overcome challenge."""
        return self.strength > 0.6


class ValidationPhase(Phase):
    """
    Phase 5: Validation with theological constraints and prosecutor/witness pattern.
    Implements adversarial validation where challenges must be defended.
    """
    name = "validation"
    category = PhaseCategory.VALIDATION
    priority = PhasePriority.NORMAL
    is_critical = False
    base_timeout_seconds = 45.0

    # Scoring weights for final judgment
    CONSTRAINT_WEIGHT = 0.4
    PROSECUTOR_PENALTY_WEIGHT = 0.3
    WITNESS_BONUS_WEIGHT = 0.3

    @property
    def dependencies(self) -> List[PhaseDependency]:
        return [
            PhaseDependency(
                phase_name="cross_reference",
                required_outputs=["cross_references"],
                is_hard=True
            ),
            PhaseDependency(
                phase_name="theological",
                required_outputs=["patristic_witness"],
                is_hard=False  # Patristic data enhances but isn't required
            )
        ]

    @property
    def outputs(self) -> List[str]:
        return ["validated_cross_references"]

    async def execute(self, context: ProcessingContext) -> ProcessingContext:
        """
        Execute validation phase.

        Steps:
        1. Apply theological constraints to each cross-reference
        2. Run prosecutor challenges
        3. Run witness defenses (if challenged)
        4. Calculate final judgment
        5. Filter by verdict
        """
        start_time = time.time()

        validated_refs = []
        rejection_count = 0

        for ref in context.cross_references:
            # Apply theological constraints first
            constraint_result = await self._validate_constraints(ref, context)

            # Store constraint validations
            ref.constraint_validations = getattr(constraint_result, "validations", [])
            ref.theological_confidence = getattr(constraint_result, "overall_score", 0.5)

            # Skip adversarial check if constraints already reject
            overall_score = getattr(constraint_result, "overall_score", 0.5)
            if overall_score < 0.3:
                ref.final_confidence = overall_score
                ref.verdict = ValidationVerdict.REJECTED
                rejection_count += 1
                continue

            # Prosecutor challenges
            challenges = await self._run_prosecutor(ref, context)
            ref.challenges = challenges

            # Witness defense (if challenged)
            defenses = []
            if challenges:
                defenses = await self._run_witness(ref, challenges, context)
                ref.defenses = defenses

            # Final judgment using weighted scoring
            final_score, verdict = self._calculate_final_judgment(
                ref, constraint_result, challenges, defenses
            )
            ref.final_confidence = final_score
            ref.verdict = verdict

            if verdict.allows_inclusion:
                validated_refs.append(ref)

                # Emit validation success event
                if hasattr(self.orchestrator, 'event_publisher'):
                    from db.events import CrossReferenceValidated
                    target_ref = getattr(ref, "target_ref", getattr(ref, "verse_id", ""))
                    connection_type = getattr(ref, "connection_type", "unknown")

                    await self.orchestrator.event_publisher.publish(
                        CrossReferenceValidated(
                            aggregate_id=context.verse_id,
                            correlation_id=context.correlation_id,
                            source_ref=context.verse_id,
                            target_ref=target_ref,
                            connection_type=connection_type,
                            final_confidence=final_score,
                            theological_score=overall_score,
                            validators=[self.name]
                        )
                    )
            else:
                rejection_count += 1

        context.validated_cross_references = validated_refs

        # Update metrics
        if hasattr(self.orchestrator, '_metrics'):
            from pipeline.unified_orchestrator import OrchestratorMetric
            self.orchestrator._metrics[OrchestratorMetric.VALIDATION_REJECTIONS] += rejection_count

        duration_ms = (time.time() - start_time) * 1000
        context.phase_durations[self.name] = duration_ms

        logger.info(f"Validation completed for {context.verse_id}: {len(validated_refs)} approved, {rejection_count} rejected in {duration_ms:.0f}ms")
        return context

    async def _validate_constraints(self, ref, context: ProcessingContext) -> Any:
        """Apply theological constraints."""
        try:
            if hasattr(self.orchestrator, 'theological_validator'):
                target_ref = getattr(ref, "target_ref", getattr(ref, "verse_id", ""))
                connection_type = getattr(ref, "connection_type", "unknown")
                confidence = getattr(ref, "confidence", getattr(ref, "score", 0.5))

                return await self.orchestrator.theological_validator.validate(
                    source_verse=context.verse_id,
                    target_verse=target_ref,
                    connection_type=connection_type,
                    confidence=confidence
                )
        except Exception as e:
            logger.warning(f"Theological validation failed: {e}")

        # Fallback result
        class ConstraintResult:
            def __init__(self):
                self.validations = []
                self.overall_score = 0.5
                self.failed_constraints = []

        return ConstraintResult()

    async def _run_prosecutor(
        self,
        ref,
        context: ProcessingContext
    ) -> List[ChallengeResult]:
        """Run prosecutor to challenge the cross-reference."""
        challenges = []

        # Challenge 1: Chronological implausibility
        if await self._check_chronological_issue(ref):
            challenges.append(ChallengeResult(
                challenge_type="chronological",
                severity=0.8,
                description="Source text post-dates target, making direct reference impossible",
                supporting_evidence=["date analysis"]
            ))

        # Challenge 2: Genre mismatch
        if await self._check_genre_mismatch(ref, context):
            challenges.append(ChallengeResult(
                challenge_type="genre_mismatch",
                severity=0.5,
                description="Source and target genres suggest coincidental similarity",
                supporting_evidence=["genre analysis"]
            ))

        # Challenge 3: Low mutual transformation
        mutual_influence = getattr(ref, "mutual_influence", 0.5)
        if mutual_influence < 0.4:
            challenges.append(ChallengeResult(
                challenge_type="weak_mutual_influence",
                severity=0.6,
                description="Texts do not significantly illuminate each other",
                supporting_evidence=[f"MI score: {mutual_influence:.2f}"]
            ))

        # Challenge 4: No patristic support
        if not await self._has_patristic_support(ref, context):
            challenges.append(ChallengeResult(
                challenge_type="no_tradition",
                severity=0.4,
                description="No Church Father connects these passages",
                supporting_evidence=["patristic survey"]
            ))

        return challenges

    async def _run_witness(
        self,
        ref,
        challenges: List[ChallengeResult],
        context: ProcessingContext
    ) -> List[DefenseResult]:
        """Run witness to defend against challenges."""
        defenses = []

        for challenge in challenges:
            defense = await self._generate_defense(challenge, ref, context)
            if defense:
                defenses.append(defense)

        return defenses

    async def _generate_defense(
        self,
        challenge: ChallengeResult,
        ref,
        context: ProcessingContext
    ) -> Optional[DefenseResult]:
        """Generate defense for a specific challenge."""
        if challenge.challenge_type == "chronological":
            # Defense: typological connections transcend chronology
            if context.typological_connections:
                return DefenseResult(
                    defense_type="typological_transcendence",
                    strength=0.7,
                    description="Typological patterns operate outside strict chronology",
                    supporting_sources=["patristic hermeneutics"]
                )

        elif challenge.challenge_type == "no_tradition":
            # Defense: novel discovery with strong semantic/structural support
            confidence = getattr(ref, "confidence", getattr(ref, "score", 0.0))
            if confidence > 0.8:
                return DefenseResult(
                    defense_type="strong_structural_evidence",
                    strength=0.6,
                    description="Linguistic and structural evidence warrants new discovery",
                    supporting_sources=["semantic analysis", "syntactic parallels"]
                )

        elif challenge.challenge_type == "weak_mutual_influence":
            # Defense: unidirectional influence is still valid
            return DefenseResult(
                defense_type="unidirectional_validity",
                strength=0.5,
                description="One-way illumination is theologically meaningful",
                supporting_sources=["hermeneutical theory"]
            )

        return None

    def _calculate_final_judgment(
        self,
        ref,
        constraint_result,
        challenges: List[ChallengeResult],
        defenses: List[DefenseResult]
    ) -> Tuple[float, ValidationVerdict]:
        """Calculate final confidence score and verdict."""
        # Base from constraints
        overall_score = getattr(constraint_result, "overall_score", 0.5)
        base_score = overall_score * self.CONSTRAINT_WEIGHT

        # Prosecutor penalty
        total_penalty = sum(c.severity for c in challenges) if challenges else 0.0
        max_penalty = len(challenges) if challenges else 1
        penalty_ratio = total_penalty / max_penalty
        penalty = penalty_ratio * self.PROSECUTOR_PENALTY_WEIGHT

        # Witness bonus (only for defended challenges)
        total_defense = sum(d.strength for d in defenses) if defenses else 0.0
        max_defense = len(challenges) if challenges else 1
        defense_ratio = total_defense / max_defense if challenges else 0
        bonus = defense_ratio * self.WITNESS_BONUS_WEIGHT

        # Final score
        ref_confidence = getattr(ref, "confidence", getattr(ref, "score", 0.5))
        final_score = base_score - penalty + bonus + (ref_confidence * 0.2)
        final_score = max(0.0, min(1.0, final_score))

        # Determine verdict
        if final_score >= 0.7:
            verdict = ValidationVerdict.APPROVED
        elif final_score >= 0.5:
            verdict = ValidationVerdict.CHALLENGED
        elif final_score >= 0.3:
            verdict = ValidationVerdict.DEFERRED
        else:
            verdict = ValidationVerdict.REJECTED

        return final_score, verdict

    async def _check_chronological_issue(self, ref) -> bool:
        """Check for chronological implausibility."""
        # Implementation would check authorship dates
        # Placeholder
        return False

    async def _check_genre_mismatch(self, ref, context) -> bool:
        """Check for problematic genre mismatch."""
        # Implementation would compare genres
        # Placeholder
        return False

    async def _has_patristic_support(self, ref, context: ProcessingContext) -> bool:
        """Check if any Father connects these passages."""
        if not context.patristic_witness:
            return False

        # Check if any interpretation mentions the target
        target_ref = getattr(ref, "target_ref", getattr(ref, "verse_id", ""))
        for interp in context.patristic_witness:
            if target_ref in str(interp):
                return True

        return False
