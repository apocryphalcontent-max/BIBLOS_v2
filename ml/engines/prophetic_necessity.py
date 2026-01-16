"""
Prophetic Necessity Prover - Fifth Impossible Oracle

Uses Bayesian inference to calculate the probability that messianic prophecies
were fulfilled by chance vs. divine design.

Mathematical Foundation:
P(Supernatural | Evidence) = P(Evidence | Supernatural) * P(Supernatural) / P(Evidence)

Where:
- P(Supernatural): Prior probability of supernatural intervention
- P(Evidence | Supernatural): Likelihood of evidence given supernatural
- P(Evidence): Total probability of evidence (natural + supernatural)
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math

logger = logging.getLogger(__name__)


class ProphecyCategory(Enum):
    """Categories of messianic prophecies."""
    LINEAGE = "lineage"                    # Genealogical requirements
    BIRTHPLACE = "birthplace"              # Geographic specificity
    MANNER_OF_BIRTH = "manner_of_birth"    # Virgin birth, etc.
    LIFE_EVENTS = "life_events"            # Ministry events
    DEATH_MANNER = "death_manner"          # Crucifixion details
    RESURRECTION = "resurrection"          # Rising from death
    TIMING = "timing"                      # Chronological precision
    SYMBOLIC = "symbolic"                  # Typological fulfillment


class IndependenceLevel(Enum):
    """Level of independence between prophecies."""
    FULLY_INDEPENDENT = "fully_independent"    # No correlation
    PARTIALLY_INDEPENDENT = "partially_independent"  # Some correlation
    DEPENDENT = "dependent"                    # Causally related


@dataclass
class ProphecyDefinition:
    """
    Definition of a single messianic prophecy.

    Attributes:
        id: Unique identifier
        ot_reference: Old Testament verse(s)
        nt_fulfillment: New Testament verse(s)
        category: Type of prophecy
        natural_probability: Probability of chance fulfillment
        independence_level: Independence from other prophecies
        evidence_strength: How certain the fulfillment is (0-1)
        historical_window_years: Time window for potential fulfillment
        population_size: Population that could fulfill
        description: Human-readable description
    """
    id: str
    ot_reference: str
    nt_fulfillment: List[str]
    category: ProphecyCategory
    natural_probability: float
    independence_level: IndependenceLevel = IndependenceLevel.FULLY_INDEPENDENT
    evidence_strength: float = 1.0
    historical_window_years: Optional[int] = None
    population_size: Optional[int] = None
    description: str = ""
    patristic_witnesses: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate probability bounds."""
        if not (0.0 <= self.natural_probability <= 1.0):
            raise ValueError(f"natural_probability must be in [0, 1], got {self.natural_probability}")
        if not (0.0 <= self.evidence_strength <= 1.0):
            raise ValueError(f"evidence_strength must be in [0, 1], got {self.evidence_strength}")


@dataclass
class BayesianResult:
    """Result of Bayesian calculation."""
    prior_supernatural: float
    likelihood_natural: float
    likelihood_supernatural: float
    posterior_supernatural: float
    bayes_factor: float
    evidence_strength: str  # weak, moderate, strong, very_strong, decisive


@dataclass
class PropheticProofResult:
    """
    Complete result of prophetic necessity calculation.

    Attributes:
        prophecies: List of prophecies analyzed
        compound_natural_probability: P(all by chance)
        bayesian_result: Bayesian inference result
        independent_count: Number of truly independent prophecies
        dependent_groups: Groups of dependent prophecies
        theological_significance: Orthodox theological assessment
    """
    prophecies: List[ProphecyDefinition]
    compound_natural_probability: float
    bayesian_result: BayesianResult
    independent_count: int
    dependent_groups: List[List[str]] = field(default_factory=list)
    theological_significance: str = ""
    calculation_notes: List[str] = field(default_factory=list)


class PropheticNecessityProver:
    """
    Fifth Impossible Oracle: Proves prophetic necessity via Bayesian inference.

    Calculates the probability that messianic prophecies were fulfilled
    by chance vs. supernatural design.
    """

    # Canonical messianic prophecies with probability estimates
    CANONICAL_PROPHECIES = {
        "virgin_birth": ProphecyDefinition(
            id="virgin_birth",
            ot_reference="ISA.7.14",
            nt_fulfillment=["MAT.1.23", "LUK.1.27"],
            category=ProphecyCategory.MANNER_OF_BIRTH,
            natural_probability=1e-8,  # Extremely rare
            evidence_strength=0.98,
            description="Virgin birth (παρθένος in LXX)",
            patristic_witnesses=["Justin Martyr", "Irenaeus", "Tertullian"]
        ),
        "bethlehem_birth": ProphecyDefinition(
            id="bethlehem_birth",
            ot_reference="MIC.5.2",
            nt_fulfillment=["MAT.2.1", "LUK.2.4"],
            category=ProphecyCategory.BIRTHPLACE,
            natural_probability=1/300,  # ~300 towns in Judea
            evidence_strength=0.99,
            historical_window_years=500,
            description="Born in Bethlehem",
            patristic_witnesses=["Justin Martyr", "Origen"]
        ),
        "davidic_lineage": ProphecyDefinition(
            id="davidic_lineage",
            ot_reference="JER.23.5",
            nt_fulfillment=["MAT.1.1", "LUK.3.31", "ROM.1.3"],
            category=ProphecyCategory.LINEAGE,
            natural_probability=1/1000,  # Many claimed Davidic descent
            evidence_strength=0.95,
            description="Descendant of David",
            patristic_witnesses=["Irenaeus", "Eusebius"]
        ),
        "suffering_servant": ProphecyDefinition(
            id="suffering_servant",
            ot_reference="ISA.53.3-12",
            nt_fulfillment=["1PE.2.24", "MAT.27.30", "LUK.23.34"],
            category=ProphecyCategory.DEATH_MANNER,
            natural_probability=1e-4,  # Specific suffering details
            evidence_strength=0.97,
            description="Suffering Servant who bears sins",
            patristic_witnesses=["Irenaeus", "Origen", "Athanasius"]
        ),
        "pierced_hands_feet": ProphecyDefinition(
            id="pierced_hands_feet",
            ot_reference="PSA.22.16",
            nt_fulfillment=["JHN.20.25", "JHN.20.27"],
            category=ProphecyCategory.DEATH_MANNER,
            natural_probability=1/50,  # Crucifixion was used but uncommon
            evidence_strength=0.90,
            description="Hands and feet pierced (ὤρυξαν in LXX)",
            patristic_witnesses=["Justin Martyr", "Tertullian"]
        ),
        "bones_not_broken": ProphecyDefinition(
            id="bones_not_broken",
            ot_reference="PSA.34.20",
            nt_fulfillment=["JHN.19.33", "JHN.19.36"],
            category=ProphecyCategory.DEATH_MANNER,
            natural_probability=1/10,  # Crurifragium was standard
            evidence_strength=0.98,
            independence_level=IndependenceLevel.PARTIALLY_INDEPENDENT,  # Related to crucifixion
            description="Bones not broken at crucifixion"
        ),
        "gall_vinegar": ProphecyDefinition(
            id="gall_vinegar",
            ot_reference="PSA.69.21",
            nt_fulfillment=["MAT.27.34", "JHN.19.29"],
            category=ProphecyCategory.DEATH_MANNER,
            natural_probability=1/20,
            evidence_strength=0.92,
            independence_level=IndependenceLevel.PARTIALLY_INDEPENDENT,
            description="Given gall and vinegar"
        ),
        "garments_divided": ProphecyDefinition(
            id="garments_divided",
            ot_reference="PSA.22.18",
            nt_fulfillment=["MAT.27.35", "JHN.19.24"],
            category=ProphecyCategory.DEATH_MANNER,
            natural_probability=1/5,  # Common with crucifixion
            evidence_strength=0.95,
            independence_level=IndependenceLevel.PARTIALLY_INDEPENDENT,
            description="Garments divided by lot"
        ),
        "resurrection": ProphecyDefinition(
            id="resurrection",
            ot_reference="PSA.16.10",
            nt_fulfillment=["ACT.2.27", "ACT.13.35"],
            category=ProphecyCategory.RESURRECTION,
            natural_probability=1e-9,  # Resurrection from death
            evidence_strength=0.85,  # Historical debate
            description="Would not see decay, resurrection",
            patristic_witnesses=["Peter", "Paul", "Irenaeus"]
        ),
        "daniels_timeline": ProphecyDefinition(
            id="daniels_timeline",
            ot_reference="DAN.9.24-26",
            nt_fulfillment=["LUK.3.1"],
            category=ProphecyCategory.TIMING,
            natural_probability=1/100,  # 70 weeks prophecy
            evidence_strength=0.75,  # Chronological interpretation varies
            description="Timing of Messiah's appearance (70 weeks)"
        ),
        "triumphal_entry_donkey": ProphecyDefinition(
            id="triumphal_entry_donkey",
            ot_reference="ZEC.9.9",
            nt_fulfillment=["MAT.21.5", "JHN.12.15"],
            category=ProphecyCategory.LIFE_EVENTS,
            natural_probability=1/50,  # Specific entry manner
            evidence_strength=0.98,
            description="Entering Jerusalem on a donkey"
        ),
        "betrayed_thirty_silver": ProphecyDefinition(
            id="betrayed_thirty_silver",
            ot_reference="ZEC.11.12",
            nt_fulfillment=["MAT.26.15"],
            category=ProphecyCategory.LIFE_EVENTS,
            natural_probability=1/100,  # Specific price
            evidence_strength=0.93,
            description="Betrayed for thirty pieces of silver"
        ),
    }

    def __init__(self):
        """Initialize prophetic necessity prover."""
        self.prophecies = self.CANONICAL_PROPHECIES.copy()
        logger.info(f"Initialized Prophetic Necessity Prover with {len(self.prophecies)} canonical prophecies")

    async def prove_prophetic_necessity(
        self,
        prophecy_ids: List[str],
        prior_supernatural: float = 0.5,
        apply_independence_adjustment: bool = True
    ) -> PropheticProofResult:
        """
        Calculate prophetic necessity using Bayesian inference.

        Args:
            prophecy_ids: List of prophecy IDs to analyze
            prior_supernatural: Prior probability of supernatural (0-1)
            apply_independence_adjustment: Adjust for dependencies

        Returns:
            Complete prophetic proof result

        Raises:
            ValueError: If prophecy IDs invalid or prior out of range
        """
        if not (0.0 <= prior_supernatural <= 1.0):
            raise ValueError(f"prior_supernatural must be in [0, 1], got {prior_supernatural}")

        # Retrieve prophecy definitions
        prophecies = []
        for pid in prophecy_ids:
            if pid not in self.prophecies:
                raise ValueError(f"Unknown prophecy ID: {pid}")
            prophecies.append(self.prophecies[pid])

        # Group by independence
        independent_groups = self._group_by_independence(prophecies)

        # Calculate compound probability
        compound_prob = self._calculate_compound_probability(
            prophecies,
            independent_groups if apply_independence_adjustment else None
        )

        # Bayesian calculation
        bayesian_result = self._bayesian_inference(
            compound_natural_prob=compound_prob,
            prior_supernatural=prior_supernatural,
            prophecies=prophecies
        )

        # Count independent prophecies
        independent_count = sum(
            1 for p in prophecies
            if p.independence_level == IndependenceLevel.FULLY_INDEPENDENT
        )

        # Theological assessment
        theological_sig = self._assess_theological_significance(
            bayesian_result,
            len(prophecies),
            independent_count
        )

        # Calculation notes
        notes = self._generate_calculation_notes(
            prophecies,
            compound_prob,
            bayesian_result,
            independent_groups
        )

        return PropheticProofResult(
            prophecies=prophecies,
            compound_natural_probability=compound_prob,
            bayesian_result=bayesian_result,
            independent_count=independent_count,
            dependent_groups=[[p.id for p in group] for group in independent_groups],
            theological_significance=theological_sig,
            calculation_notes=notes
        )

    def _group_by_independence(
        self,
        prophecies: List[ProphecyDefinition]
    ) -> List[List[ProphecyDefinition]]:
        """
        Group prophecies by independence level.

        Returns:
            List of groups (each group shares dependencies)
        """
        # Simple grouping by category for now
        # In production, would use domain knowledge of dependencies
        groups = []
        by_category = {}

        for p in prophecies:
            if p.independence_level == IndependenceLevel.FULLY_INDEPENDENT:
                groups.append([p])
            else:
                cat = p.category.value
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append(p)

        # Add dependent groups
        groups.extend(by_category.values())

        return groups

    def _calculate_compound_probability(
        self,
        prophecies: List[ProphecyDefinition],
        independence_groups: Optional[List[List[ProphecyDefinition]]] = None
    ) -> float:
        """
        Calculate compound probability of all prophecies by chance.

        If independence_groups provided, adjusts for dependencies.
        Otherwise assumes full independence.

        Args:
            prophecies: List of prophecies
            independence_groups: Optional grouping for dependent prophecies

        Returns:
            Compound probability (product for independent, adjusted for dependent)
        """
        if independence_groups is None:
            # Assume full independence - simple product
            compound = 1.0
            for p in prophecies:
                compound *= p.natural_probability * p.evidence_strength
            return compound

        # Calculate per group, then multiply groups
        compound = 1.0
        for group in independence_groups:
            if len(group) == 1:
                # Independent prophecy
                p = group[0]
                compound *= p.natural_probability * p.evidence_strength
            else:
                # Dependent group - use maximum probability (conservative)
                # In reality, would model correlation structure
                group_prob = max(p.natural_probability for p in group)
                avg_evidence = sum(p.evidence_strength for p in group) / len(group)
                compound *= group_prob * avg_evidence

        return compound

    def _bayesian_inference(
        self,
        compound_natural_prob: float,
        prior_supernatural: float,
        prophecies: List[ProphecyDefinition]
    ) -> BayesianResult:
        """
        Perform Bayesian inference.

        P(S|E) = P(E|S) * P(S) / P(E)

        Where:
        - S = Supernatural intervention
        - E = Evidence (prophecies fulfilled)
        - P(S) = prior_supernatural
        - P(E|S) ≈ 1 (if supernatural, fulfillment certain)
        - P(E|¬S) = compound_natural_prob
        """
        # Likelihood given supernatural (high but not 1 due to historical uncertainty)
        avg_evidence_strength = sum(p.evidence_strength for p in prophecies) / len(prophecies)
        likelihood_supernatural = avg_evidence_strength

        # Likelihood given natural (compound probability)
        likelihood_natural = compound_natural_prob

        # Total probability of evidence
        # P(E) = P(E|S) * P(S) + P(E|¬S) * P(¬S)
        prob_evidence = (
            likelihood_supernatural * prior_supernatural +
            likelihood_natural * (1 - prior_supernatural)
        )

        # Posterior probability
        # P(S|E) = P(E|S) * P(S) / P(E)
        if prob_evidence > 0:
            posterior_supernatural = (
                likelihood_supernatural * prior_supernatural / prob_evidence
            )
        else:
            posterior_supernatural = 1.0  # Evidence so strong, certainty

        # Bayes factor: K = P(E|S) / P(E|¬S)
        if likelihood_natural > 0:
            bayes_factor = likelihood_supernatural / likelihood_natural
        else:
            bayes_factor = float('inf')

        # Interpret Bayes factor (Kass & Raftery scale)
        if bayes_factor < 1:
            evidence_strength = "negative"
        elif bayes_factor < 3:
            evidence_strength = "weak"
        elif bayes_factor < 10:
            evidence_strength = "moderate"
        elif bayes_factor < 30:
            evidence_strength = "strong"
        elif bayes_factor < 100:
            evidence_strength = "very_strong"
        else:
            evidence_strength = "decisive"

        return BayesianResult(
            prior_supernatural=prior_supernatural,
            likelihood_natural=likelihood_natural,
            likelihood_supernatural=likelihood_supernatural,
            posterior_supernatural=posterior_supernatural,
            bayes_factor=bayes_factor,
            evidence_strength=evidence_strength
        )

    def _assess_theological_significance(
        self,
        bayesian_result: BayesianResult,
        total_prophecies: int,
        independent_count: int
    ) -> str:
        """
        Assess theological significance from Orthodox perspective.

        Args:
            bayesian_result: Bayesian inference result
            total_prophecies: Total number of prophecies
            independent_count: Number of independent prophecies

        Returns:
            Theological assessment
        """
        if bayesian_result.posterior_supernatural > 0.999:
            return (
                f"Overwhelming evidence for divine orchestration. "
                f"With {total_prophecies} prophecies ({independent_count} independent), "
                f"the probability of chance fulfillment is {bayesian_result.likelihood_natural:.2e}. "
                f"The Fathers saw this as the 'seal of prophecy' confirming Christ's divinity."
            )
        elif bayesian_result.posterior_supernatural > 0.95:
            return (
                f"Strong evidence for providential design. "
                f"Bayes factor of {bayesian_result.bayes_factor:.1f} indicates "
                f"{bayesian_result.evidence_strength} support for supernatural explanation."
            )
        elif bayesian_result.posterior_supernatural > 0.75:
            return (
                f"Moderate evidence for divine fulfillment. "
                f"Multiple independent witnesses support prophetic authenticity."
            )
        else:
            return (
                f"Insufficient evidence for definitive conclusion. "
                f"Further historical and textual analysis needed."
            )

    def _generate_calculation_notes(
        self,
        prophecies: List[ProphecyDefinition],
        compound_prob: float,
        bayesian_result: BayesianResult,
        independence_groups: List[List[ProphecyDefinition]]
    ) -> List[str]:
        """Generate detailed calculation notes."""
        notes = []

        notes.append(f"Analyzed {len(prophecies)} messianic prophecies")
        notes.append(f"Compound natural probability: {compound_prob:.2e}")
        notes.append(f"Posterior probability of supernatural: {bayesian_result.posterior_supernatural:.4f}")
        notes.append(f"Bayes factor: {bayesian_result.bayes_factor:.2e} ({bayesian_result.evidence_strength})")

        # Independence structure
        fully_independent = sum(1 for g in independence_groups if len(g) == 1)
        dependent_groups_count = sum(1 for g in independence_groups if len(g) > 1)
        notes.append(f"Independence structure: {fully_independent} fully independent, {dependent_groups_count} dependent groups")

        # Highest probability prophecies (easiest to fulfill by chance)
        sorted_proph = sorted(prophecies, key=lambda p: p.natural_probability, reverse=True)
        notes.append(f"Highest natural probability: {sorted_proph[0].id} ({sorted_proph[0].natural_probability:.2e})")

        # Lowest probability (hardest to fulfill by chance)
        notes.append(f"Lowest natural probability: {sorted_proph[-1].id} ({sorted_proph[-1].natural_probability:.2e})")

        return notes

    async def add_custom_prophecy(self, prophecy: ProphecyDefinition) -> None:
        """
        Add a custom prophecy definition.

        Args:
            prophecy: Prophecy to add
        """
        self.prophecies[prophecy.id] = prophecy
        logger.info(f"Added custom prophecy: {prophecy.id}")

    def get_prophecy(self, prophecy_id: str) -> Optional[ProphecyDefinition]:
        """Get prophecy definition by ID."""
        return self.prophecies.get(prophecy_id)

    def list_prophecies(self) -> List[str]:
        """List all available prophecy IDs."""
        return list(self.prophecies.keys())
