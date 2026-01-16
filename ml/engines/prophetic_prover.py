"""
BIBLOS v2 - Prophetic Necessity Prover
Fifth Impossible Oracle

Implements Bayesian probability mathematics to calculate whether biblical
prophecy fulfillment patterns constitute evidence of supernatural origin.

Uses rigorous statistical methodology:
- Specificity-based probability estimation
- Independence analysis for prophecy clustering
- Compound probability with dependency adjustments
- Bayesian posterior calculation with Jeffreys' scale interpretation
- Sensitivity analysis across prior assumptions
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TYPE_CHECKING,
)

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from ml.engines.necessity_calculator import InterVerseNecessityCalculator
    from ml.engines.lxx_extractor import LXXChristologicalExtractor
    from ml.engines.fractal_typology import HyperFractalTypologyEngine

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class FulfillmentType(Enum):
    """Classification of prophecy fulfillment modality."""
    EXPLICIT = "explicit"           # Direct verbal prophecy fulfilled
    TYPOLOGICAL = "typological"     # Type-antitype pattern
    SYMBOLIC = "symbolic"           # Symbolic/allegorical fulfillment
    MULTIPLE = "multiple"           # Multiple fulfillment levels
    CONDITIONAL = "conditional"     # Conditional prophecy

    @property
    def evidential_weight(self) -> float:
        """Evidential weight based on fulfillment type."""
        weights = {
            FulfillmentType.EXPLICIT: 1.0,
            FulfillmentType.TYPOLOGICAL: 0.8,
            FulfillmentType.SYMBOLIC: 0.6,
            FulfillmentType.MULTIPLE: 0.9,
            FulfillmentType.CONDITIONAL: 0.5
        }
        return weights[self]


class IndependenceLevel(Enum):
    """How independent a prophecy is from others."""
    FULLY_INDEPENDENT = "fully_independent"
    PARTIALLY_DEPENDENT = "partially_dependent"
    CAUSALLY_LINKED = "causally_linked"
    CLUSTER = "cluster"  # Same event, multiple predictions

    @property
    def effective_weight(self) -> float:
        """Contribution to compound probability calculation."""
        weights = {
            IndependenceLevel.FULLY_INDEPENDENT: 1.0,
            IndependenceLevel.PARTIALLY_DEPENDENT: 0.7,
            IndependenceLevel.CAUSALLY_LINKED: 0.3,
            IndependenceLevel.CLUSTER: 0.0  # Counted once per cluster
        }
        return weights[self]


class SpecificityFactor(Enum):
    """Factors that increase prophecy specificity and decrease natural probability."""
    PERSON_NAME = "person_name"
    LOCATION_CITY = "location_city"
    LOCATION_REGION = "location_region"
    TIME_PERIOD = "time_period"
    MANNER_OF_DEATH = "manner_of_death"
    BIOLOGICAL_MIRACLE = "biological_miracle"
    EXACT_PRICE = "exact_price"
    SEQUENCE_EVENTS = "sequence_events"
    NUMERICAL_DETAIL = "numerical_detail"
    GENEALOGICAL = "genealogical"
    DAVIDIC_LINEAGE = "davidic_lineage"
    RESURRECTION = "resurrection"
    BETRAYAL_PRICE = "betrayal_price"
    CRUCIFIXION = "crucifixion"
    MESSIANIC_TITLE = "messianic_title"

    @property
    def base_probability(self) -> float:
        """Base probability for this factor occurring by chance."""
        probabilities = {
            SpecificityFactor.PERSON_NAME: 1e-6,
            SpecificityFactor.LOCATION_CITY: 0.001,
            SpecificityFactor.LOCATION_REGION: 0.01,
            SpecificityFactor.TIME_PERIOD: 0.01,
            SpecificityFactor.MANNER_OF_DEATH: 0.01,
            SpecificityFactor.BIOLOGICAL_MIRACLE: 1e-15,  # Essentially impossible
            SpecificityFactor.EXACT_PRICE: 1e-4,
            SpecificityFactor.SEQUENCE_EVENTS: 0.001,
            SpecificityFactor.NUMERICAL_DETAIL: 0.01,
            SpecificityFactor.GENEALOGICAL: 0.02,
            SpecificityFactor.DAVIDIC_LINEAGE: 0.01,  # Post-exile, rare
            SpecificityFactor.RESURRECTION: 1e-15,
            SpecificityFactor.BETRAYAL_PRICE: 1e-5,  # Exact amount
            SpecificityFactor.CRUCIFIXION: 0.001,  # Specific execution method
            SpecificityFactor.MESSIANIC_TITLE: 0.05,
        }
        return probabilities[self]


class EvidenceStrength(Enum):
    """Jeffreys' scale for Bayes factor interpretation."""
    DECISIVE = "decisive"
    VERY_STRONG = "very_strong"
    STRONG = "strong"
    SUBSTANTIAL = "substantial"
    MODERATE = "moderate"
    WEAK = "weak"
    INCONCLUSIVE = "inconclusive"

    @classmethod
    def from_bayes_factor(cls, bf: float) -> "EvidenceStrength":
        """Classify evidence strength from Bayes factor."""
        if bf > 1e10:
            return cls.DECISIVE
        elif bf > 1e6:
            return cls.VERY_STRONG
        elif bf > 1e3:
            return cls.STRONG
        elif bf > 30:
            return cls.SUBSTANTIAL
        elif bf > 10:
            return cls.MODERATE
        elif bf > 3:
            return cls.WEAK
        return cls.INCONCLUSIVE


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ProbabilityEstimation:
    """Probability estimation with uncertainty quantification."""
    point_estimate: float
    confidence_interval: Tuple[float, float]  # 95% CI
    estimation_method: str
    specificity_factors: List[SpecificityFactor]
    historical_evidence: List[str] = field(default_factory=list)
    scholarly_range: Tuple[float, float] = field(default=(0.0, 1.0))

    def as_beta_distribution(self) -> stats.beta:
        """Convert to Beta distribution for Bayesian updating."""
        mean = self.point_estimate
        low, high = self.confidence_interval
        var = ((high - low) / 4) ** 2

        if var > 0 and 0 < mean < 1:
            common = mean * (1 - mean) / var - 1
            alpha = max(0.1, mean * common)
            beta_param = max(0.1, (1 - mean) * common)
            return stats.beta(alpha, beta_param)
        return stats.beta(1, 1)  # Uniform prior fallback


@dataclass
class ProphecyFulfillmentPair:
    """A prophecy-fulfillment pair with probability metadata."""
    prophecy_id: str
    prophecy_verse: str
    fulfillment_verse: str
    prophecy_text: str
    fulfillment_text: str
    fulfillment_type: FulfillmentType
    probability_estimate: ProbabilityEstimation
    independence_level: IndependenceLevel = IndependenceLevel.FULLY_INDEPENDENT
    dependent_on: List[str] = field(default_factory=list)
    cluster_id: Optional[str] = None

    # Enrichment from other oracles
    necessity_score: float = 0.0        # From InterVerseNecessityCalculator
    manuscript_support: float = 1.0     # From LXXChristologicalExtractor
    typological_depth: int = 0          # From HyperFractalTypologyEngine
    patristic_consensus: float = 0.0    # Patristic agreement on interpretation

    @property
    def natural_probability(self) -> float:
        """Get natural probability estimate."""
        return self.probability_estimate.point_estimate

    @property
    def effective_probability(self) -> float:
        """Probability adjusted by independence weight."""
        return self.natural_probability ** self.independence_level.effective_weight

    @property
    def weighted_probability(self) -> float:
        """Probability weighted by fulfillment type evidential weight."""
        return self.natural_probability * self.fulfillment_type.evidential_weight


@dataclass
class IndependenceAnalysis:
    """Analysis of independence relationships for a prophecy."""
    prophecy_id: str
    independence_level: IndependenceLevel
    related_prophecies: List[str]
    shared_factors: List[str]
    effective_contribution: float
    reasoning: str


@dataclass
class BayesianResult:
    """Complete Bayesian analysis result."""
    prior_supernatural: float
    prior_natural: float
    likelihood_given_supernatural: float
    likelihood_given_natural: float
    posterior_supernatural: float
    posterior_natural: float
    bayes_factor: float
    log_bayes_factor: float
    interpretation: str
    credible_interval: Tuple[float, float]

    @property
    def strength_category(self) -> EvidenceStrength:
        """Jeffreys' scale interpretation."""
        return EvidenceStrength.from_bayes_factor(self.bayes_factor)

    @property
    def is_decisive(self) -> bool:
        """Whether evidence is decisive."""
        return self.strength_category == EvidenceStrength.DECISIVE


@dataclass
class PropheticProofResult:
    """Complete prophetic proof analysis result."""
    prophecy_set: List[ProphecyFulfillmentPair]
    independent_count: int
    effective_count: float
    compound_natural_probability: float
    log_probability: float
    independence_analysis: List[IndependenceAnalysis]
    bayesian_result: BayesianResult
    sensitivity_analysis: Dict[str, BayesianResult]
    enrichment_summary: Dict[str, float]
    confidence: float
    summary: str
    methodology_notes: str
    computation_timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "prophecy_count": len(self.prophecy_set),
            "independent_count": self.independent_count,
            "effective_count": self.effective_count,
            "compound_natural_probability": self.compound_natural_probability,
            "log_probability": self.log_probability,
            "posterior_supernatural": self.bayesian_result.posterior_supernatural,
            "bayes_factor": self.bayesian_result.bayes_factor,
            "log_bayes_factor": self.bayesian_result.log_bayes_factor,
            "evidence_strength": self.bayesian_result.strength_category.value,
            "interpretation": self.bayesian_result.interpretation,
            "confidence": self.confidence,
            "summary": self.summary,
            "timestamp": self.computation_timestamp.isoformat(),
        }


@dataclass
class PropheticProverConfig:
    """Configuration for PropheticNecessityProver."""
    default_prior: float = 0.5
    minimum_probability: float = 1e-20
    maximum_log_probability: float = -700
    include_enrichment: bool = True
    run_sensitivity_analysis: bool = True
    sensitivity_priors: Dict[str, float] = field(default_factory=lambda: {
        "very_skeptical": 0.001,
        "skeptical": 0.01,
        "mildly_skeptical": 0.1,
        "neutral": 0.5,
        "mildly_favorable": 0.7,
        "favorable": 0.9
    })


# =============================================================================
# CANONICAL PROPHECY CATALOG
# =============================================================================

CANONICAL_PROPHECY_CATALOG: Dict[str, Dict[str, Any]] = {
    # VIRGIN BIRTH
    "isa_7_14": {
        "prophecy_verse": "ISA.7.14",
        "fulfillment_verse": "MAT.1.23",
        "prophecy_text": "Behold, the virgin shall conceive and bear a son",
        "fulfillment_text": "The virgin shall conceive and bear a son, and they shall call his name Emmanuel",
        "fulfillment_type": FulfillmentType.EXPLICIT,
        "factors": [SpecificityFactor.BIOLOGICAL_MIRACLE],
    },
    # BETHLEHEM BIRTH
    "mic_5_2": {
        "prophecy_verse": "MIC.5.2",
        "fulfillment_verse": "MAT.2.6",
        "prophecy_text": "But you, O Bethlehem Ephrathah, who are too little to be among the clans of Judah, from you shall come forth for me one who is to be ruler in Israel",
        "fulfillment_text": "And you, O Bethlehem, in the land of Judah, are by no means least among the rulers of Judah; for from you shall come a ruler who will shepherd my people Israel",
        "fulfillment_type": FulfillmentType.EXPLICIT,
        "factors": [SpecificityFactor.LOCATION_CITY],
    },
    # DAVIDIC LINEAGE
    "2sa_7_12": {
        "prophecy_verse": "2SA.7.12",
        "fulfillment_verse": "MAT.1.1",
        "prophecy_text": "I will raise up your offspring after you, who shall come from your body, and I will establish his kingdom",
        "fulfillment_text": "The book of the genealogy of Jesus Christ, the son of David",
        "fulfillment_type": FulfillmentType.EXPLICIT,
        "factors": [SpecificityFactor.DAVIDIC_LINEAGE, SpecificityFactor.GENEALOGICAL],
    },
    # TRIUMPHAL ENTRY
    "zec_9_9": {
        "prophecy_verse": "ZEC.9.9",
        "fulfillment_verse": "MAT.21.5",
        "prophecy_text": "Behold, your king is coming to you, humble and mounted on a donkey",
        "fulfillment_text": "Say to the daughter of Zion, Behold, your king is coming to you, humble, and mounted on a donkey",
        "fulfillment_type": FulfillmentType.EXPLICIT,
        "factors": [SpecificityFactor.MESSIANIC_TITLE, SpecificityFactor.SEQUENCE_EVENTS],
    },
    # BETRAYAL FOR 30 PIECES
    "zec_11_12": {
        "prophecy_verse": "ZEC.11.12",
        "fulfillment_verse": "MAT.26.15",
        "prophecy_text": "They weighed out as my wages thirty pieces of silver",
        "fulfillment_text": "They paid him thirty pieces of silver",
        "fulfillment_type": FulfillmentType.EXPLICIT,
        "factors": [SpecificityFactor.EXACT_PRICE, SpecificityFactor.BETRAYAL_PRICE, SpecificityFactor.NUMERICAL_DETAIL],
    },
    # PIERCED HANDS AND FEET
    "psa_22_16": {
        "prophecy_verse": "PSA.22.16",
        "fulfillment_verse": "JHN.20.25",
        "prophecy_text": "They have pierced my hands and my feet",
        "fulfillment_text": "Unless I see in his hands the mark of the nails",
        "fulfillment_type": FulfillmentType.TYPOLOGICAL,
        "factors": [SpecificityFactor.CRUCIFIXION, SpecificityFactor.MANNER_OF_DEATH],
    },
    # LOTS FOR GARMENTS
    "psa_22_18": {
        "prophecy_verse": "PSA.22.18",
        "fulfillment_verse": "JHN.19.24",
        "prophecy_text": "They divide my garments among them, and for my clothing they cast lots",
        "fulfillment_text": "Let us not tear it, but cast lots for it to see whose it shall be",
        "fulfillment_type": FulfillmentType.TYPOLOGICAL,
        "factors": [SpecificityFactor.SEQUENCE_EVENTS],
    },
    # SUFFERING SERVANT
    "isa_53_5": {
        "prophecy_verse": "ISA.53.5",
        "fulfillment_verse": "1PE.2.24",
        "prophecy_text": "He was pierced for our transgressions; he was crushed for our iniquities",
        "fulfillment_text": "He himself bore our sins in his body on the tree",
        "fulfillment_type": FulfillmentType.EXPLICIT,
        "factors": [SpecificityFactor.MANNER_OF_DEATH, SpecificityFactor.CRUCIFIXION],
    },
    # NO BONES BROKEN
    "exo_12_46": {
        "prophecy_verse": "EXO.12.46",
        "fulfillment_verse": "JHN.19.36",
        "prophecy_text": "You shall not break any of its bones",
        "fulfillment_text": "Not one of his bones will be broken",
        "fulfillment_type": FulfillmentType.TYPOLOGICAL,
        "factors": [SpecificityFactor.MANNER_OF_DEATH],
    },
    # RESURRECTION
    "psa_16_10": {
        "prophecy_verse": "PSA.16.10",
        "fulfillment_verse": "ACT.2.31",
        "prophecy_text": "For you will not abandon my soul to Sheol, or let your holy one see corruption",
        "fulfillment_text": "He was not abandoned to Hades, nor did his flesh see corruption",
        "fulfillment_type": FulfillmentType.TYPOLOGICAL,
        "factors": [SpecificityFactor.RESURRECTION],
    },
    # SECOND TEMPLE TIMING (Daniel 9)
    "dan_9_26": {
        "prophecy_verse": "DAN.9.26",
        "fulfillment_verse": "LUK.3.23",
        "prophecy_text": "After the sixty-two weeks, an anointed one shall be cut off",
        "fulfillment_text": "Jesus, when he began his ministry, was about thirty years of age",
        "fulfillment_type": FulfillmentType.EXPLICIT,
        "factors": [SpecificityFactor.TIME_PERIOD, SpecificityFactor.NUMERICAL_DETAIL],
    },
    # PRECEDED BY MESSENGER
    "mal_3_1": {
        "prophecy_verse": "MAL.3.1",
        "fulfillment_verse": "MAT.11.10",
        "prophecy_text": "Behold, I send my messenger, and he will prepare the way before me",
        "fulfillment_text": "This is he of whom it is written, Behold, I send my messenger before your face",
        "fulfillment_type": FulfillmentType.EXPLICIT,
        "factors": [SpecificityFactor.SEQUENCE_EVENTS],
    },
    # BORN OF A WOMAN (Proto-evangelium)
    "gen_3_15": {
        "prophecy_verse": "GEN.3.15",
        "fulfillment_verse": "GAL.4.4",
        "prophecy_text": "He shall bruise your head, and you shall bruise his heel",
        "fulfillment_text": "God sent forth his Son, born of woman",
        "fulfillment_type": FulfillmentType.TYPOLOGICAL,
        "factors": [SpecificityFactor.GENEALOGICAL, SpecificityFactor.MESSIANIC_TITLE],
    },
}


# =============================================================================
# PROPHETIC NECESSITY PROVER
# =============================================================================

class PropheticNecessityProver:
    """
    Bayesian engine for prophetic necessity analysis.
    Fifth Impossible Oracle.

    Implements rigorous statistical methodology:
    1. Specificity-based probability estimation for each prophecy
    2. Independence analysis to detect clustering/causal links
    3. Compound probability calculation with dependency adjustments
    4. Bayesian posterior using Bayes' theorem
    5. Sensitivity analysis across prior assumptions

    Orthodox theological grounding:
    - Virgin birth (παρθένος) as Christologically essential
    - Davidic lineage as covenantally necessary
    - Passion prophecies as typologically interconnected
    - Resurrection as theologically non-negotiable
    """

    # Known clusters: prophecies about same event counted once
    KNOWN_CLUSTERS: Dict[str, FrozenSet[str]] = {
        "passion_cluster": frozenset([
            "psa_22_16", "psa_22_18", "isa_53_5", "exo_12_46"
        ]),
        "birth_cluster": frozenset([
            "isa_7_14", "mic_5_2", "2sa_7_12"
        ]),
    }

    # Causal links: prophecy A causes B to not be independent
    CAUSAL_LINKS: Dict[str, List[str]] = {
        "2sa_7_12": ["mic_5_2"],  # Davidic lineage → Bethlehem (ancestral land)
    }

    def __init__(
        self,
        necessity_calc: Optional["InterVerseNecessityCalculator"] = None,
        lxx_extractor: Optional["LXXChristologicalExtractor"] = None,
        typology_engine: Optional["HyperFractalTypologyEngine"] = None,
        prophecy_catalog: Optional[Dict[str, ProphecyFulfillmentPair]] = None,
        config: Optional[PropheticProverConfig] = None,
    ):
        """
        Initialize PropheticNecessityProver.

        Args:
            necessity_calc: InterVerseNecessityCalculator for enrichment
            lxx_extractor: LXXChristologicalExtractor for manuscript support
            typology_engine: HyperFractalTypologyEngine for typological depth
            prophecy_catalog: Pre-built catalog of prophecy pairs
            config: Configuration options
        """
        self.necessity_calc = necessity_calc
        self.lxx_extractor = lxx_extractor
        self.typology_engine = typology_engine
        self.config = config or PropheticProverConfig()
        self.current_set: List[ProphecyFulfillmentPair] = []

        # Build catalog from canonical data if not provided
        if prophecy_catalog:
            self.prophecy_catalog = prophecy_catalog
        else:
            self.prophecy_catalog = self._build_canonical_catalog()

        logger.info(
            f"PropheticNecessityProver initialized with "
            f"{len(self.prophecy_catalog)} prophecies"
        )

    def _build_canonical_catalog(self) -> Dict[str, ProphecyFulfillmentPair]:
        """Build catalog from canonical prophecy data."""
        catalog = {}
        for pid, data in CANONICAL_PROPHECY_CATALOG.items():
            # Build probability estimation from factors
            factors = data.get("factors", [])
            if factors:
                prob = 1.0
                for factor in factors:
                    prob *= factor.base_probability
            else:
                prob = 0.1  # Generic prediction

            estimation = ProbabilityEstimation(
                point_estimate=prob,
                confidence_interval=(prob * 0.1, min(prob * 10, 1.0)),
                estimation_method="specificity_factor_analysis",
                specificity_factors=factors,
            )

            pair = ProphecyFulfillmentPair(
                prophecy_id=pid,
                prophecy_verse=data["prophecy_verse"],
                fulfillment_verse=data["fulfillment_verse"],
                prophecy_text=data["prophecy_text"],
                fulfillment_text=data["fulfillment_text"],
                fulfillment_type=data["fulfillment_type"],
                probability_estimate=estimation,
            )
            catalog[pid] = pair

        return catalog

    # =========================================================================
    # MAIN API
    # =========================================================================

    async def prove_prophetic_necessity(
        self,
        prophecy_ids: Optional[List[str]] = None,
        prior_supernatural: Optional[float] = None,
    ) -> PropheticProofResult:
        """
        Main entry point: complete prophetic proof analysis.

        Args:
            prophecy_ids: List of prophecy IDs to analyze (all if None)
            prior_supernatural: Prior probability of supernatural origin

        Returns:
            PropheticProofResult with complete analysis
        """
        prior = prior_supernatural or self.config.default_prior

        # Gather prophecy-fulfillment pairs
        if prophecy_ids:
            prophecies = [
                self.prophecy_catalog[pid]
                for pid in prophecy_ids
                if pid in self.prophecy_catalog
            ]
        else:
            prophecies = list(self.prophecy_catalog.values())

        self.current_set = prophecies

        # Enrich with other oracle data if available
        if self.config.include_enrichment:
            enriched = []
            for p in prophecies:
                p = await self._enrich_prophecy(p)
                enriched.append(p)
            prophecies = enriched

        # Analyze independence
        independence_data = []
        for p in prophecies:
            analysis = self._assess_independence(p.prophecy_id, self.prophecy_catalog)
            independence_data.append(analysis)

        # Calculate compound probability
        compound_prob, log_prob = self._calculate_compound_probability(
            prophecies, independence_data
        )

        # Calculate Bayesian posterior
        bayesian = self._calculate_bayesian_posterior(compound_prob, prior)

        # Sensitivity analysis
        sensitivity = {}
        if self.config.run_sensitivity_analysis:
            sensitivity = self._sensitivity_analysis(compound_prob)

        # Build result
        independent_count = sum(
            1 for a in independence_data
            if a.independence_level == IndependenceLevel.FULLY_INDEPENDENT
        )
        effective_count = sum(a.effective_contribution for a in independence_data)

        enrichment_summary = {
            "avg_necessity": np.mean([p.necessity_score for p in prophecies]),
            "avg_manuscript": np.mean([p.manuscript_support for p in prophecies]),
            "avg_typological_depth": np.mean([p.typological_depth for p in prophecies]),
            "avg_patristic": np.mean([p.patristic_consensus for p in prophecies]),
        }

        summary = self._generate_summary(
            len(prophecies), independent_count, bayesian
        )
        methodology = self._generate_methodology_notes(prophecies, independence_data)

        return PropheticProofResult(
            prophecy_set=prophecies,
            independent_count=independent_count,
            effective_count=effective_count,
            compound_natural_probability=compound_prob,
            log_probability=log_prob,
            independence_analysis=independence_data,
            bayesian_result=bayesian,
            sensitivity_analysis=sensitivity,
            enrichment_summary=enrichment_summary,
            confidence=bayesian.posterior_supernatural,
            summary=summary,
            methodology_notes=methodology,
        )

    async def analyze_single_prophecy(
        self,
        prophecy_id: str,
    ) -> Tuple[ProphecyFulfillmentPair, ProbabilityEstimation]:
        """
        Analyze a single prophecy for probability estimation.

        Args:
            prophecy_id: Prophecy identifier

        Returns:
            Tuple of prophecy pair and its probability estimation
        """
        if prophecy_id not in self.prophecy_catalog:
            raise ValueError(f"Unknown prophecy: {prophecy_id}")

        prophecy = self.prophecy_catalog[prophecy_id]
        enriched = await self._enrich_prophecy(prophecy)

        return enriched, enriched.probability_estimate

    # =========================================================================
    # INDEPENDENCE ANALYSIS
    # =========================================================================

    def _assess_independence(
        self,
        prophecy_id: str,
        all_prophecies: Dict[str, ProphecyFulfillmentPair],
    ) -> IndependenceAnalysis:
        """Assess independence level relative to other prophecies."""
        if prophecy_id not in all_prophecies:
            return IndependenceAnalysis(
                prophecy_id=prophecy_id,
                independence_level=IndependenceLevel.FULLY_INDEPENDENT,
                related_prophecies=[],
                shared_factors=[],
                effective_contribution=1.0,
                reasoning="Unknown prophecy, assumed independent",
            )

        # Check if in a known cluster
        for cluster_id, members in self.KNOWN_CLUSTERS.items():
            if prophecy_id in members:
                other_members = [m for m in members if m != prophecy_id]
                return IndependenceAnalysis(
                    prophecy_id=prophecy_id,
                    independence_level=IndependenceLevel.CLUSTER,
                    related_prophecies=other_members,
                    shared_factors=["same_fulfillment_event"],
                    effective_contribution=0.0,
                    reasoning=f"Part of {cluster_id}: counted once",
                )

        # Check if causally linked
        if prophecy_id in self.CAUSAL_LINKS:
            linked = self.CAUSAL_LINKS[prophecy_id]
            return IndependenceAnalysis(
                prophecy_id=prophecy_id,
                independence_level=IndependenceLevel.CAUSALLY_LINKED,
                related_prophecies=linked,
                shared_factors=["causal_relationship"],
                effective_contribution=0.3,
                reasoning=f"Causally linked to {linked}",
            )

        # Check if dependency of another
        for source, targets in self.CAUSAL_LINKS.items():
            if prophecy_id in targets:
                return IndependenceAnalysis(
                    prophecy_id=prophecy_id,
                    independence_level=IndependenceLevel.PARTIALLY_DEPENDENT,
                    related_prophecies=[source],
                    shared_factors=["dependent_on_causal"],
                    effective_contribution=0.7,
                    reasoning=f"Partially dependent on {source}",
                )

        # Fully independent
        return IndependenceAnalysis(
            prophecy_id=prophecy_id,
            independence_level=IndependenceLevel.FULLY_INDEPENDENT,
            related_prophecies=[],
            shared_factors=[],
            effective_contribution=1.0,
            reasoning="No shared causal or clustering factors detected",
        )

    # =========================================================================
    # PROBABILITY CALCULATIONS
    # =========================================================================

    def _calculate_compound_probability(
        self,
        prophecies: List[ProphecyFulfillmentPair],
        independence_data: List[IndependenceAnalysis],
    ) -> Tuple[float, float]:
        """
        Calculate compound probability with dependency adjustments.

        Uses log scale for numerical stability with very small probabilities.

        Returns:
            Tuple of (compound_probability, log_probability)
        """
        independence_map = {a.prophecy_id: a for a in independence_data}

        # Categorize by independence type
        fully_independent: List[ProphecyFulfillmentPair] = []
        clusters: Dict[str, List[ProphecyFulfillmentPair]] = {}
        partial_deps: List[Tuple[ProphecyFulfillmentPair, IndependenceAnalysis]] = []
        seen_clusters: Set[str] = set()

        for p in prophecies:
            analysis = independence_map.get(p.prophecy_id)
            if not analysis:
                fully_independent.append(p)
                continue

            if analysis.independence_level == IndependenceLevel.FULLY_INDEPENDENT:
                fully_independent.append(p)
            elif analysis.independence_level == IndependenceLevel.CLUSTER:
                # Find cluster ID
                cluster_id = None
                for cid, members in self.KNOWN_CLUSTERS.items():
                    if p.prophecy_id in members:
                        cluster_id = cid
                        break
                if cluster_id and cluster_id not in seen_clusters:
                    clusters[cluster_id] = [p]
                    seen_clusters.add(cluster_id)
                elif cluster_id:
                    clusters[cluster_id].append(p)
            elif analysis.independence_level == IndependenceLevel.PARTIALLY_DEPENDENT:
                partial_deps.append((p, analysis))
            # CAUSALLY_LINKED: contributes less

        # Calculate compound (log scale for numerical stability)
        log_compound = 0.0
        min_prob = self.config.minimum_probability

        # Fully independent: multiply directly
        for p in fully_independent:
            prob = max(p.natural_probability, min_prob)
            log_compound += math.log(prob)

        # Clusters: use minimum probability (most specific)
        for cluster_id, cluster_prophecies in clusters.items():
            if cluster_prophecies:
                min_prob_cluster = min(
                    p.natural_probability for p in cluster_prophecies
                )
                log_compound += math.log(max(min_prob_cluster, min_prob))

        # Partial dependencies: apply effective weight
        for p, analysis in partial_deps:
            prob = max(p.natural_probability, min_prob)
            log_compound += math.log(prob) * analysis.effective_contribution

        # Convert back from log scale
        if log_compound > self.config.maximum_log_probability:
            compound = math.exp(log_compound)
        else:
            compound = 0.0

        return compound, log_compound

    def _calculate_bayesian_posterior(
        self,
        compound_natural_prob: float,
        prior_supernatural: float,
    ) -> BayesianResult:
        """
        Apply Bayes' theorem with proper handling of extreme probabilities.

        Bayes' Theorem:
        P(supernatural | evidence) = P(evidence | supernatural) × P(supernatural)
                                     ─────────────────────────────────────────────
                                                    P(evidence)
        """
        prior_natural = 1.0 - prior_supernatural

        # Likelihood under supernatural: prophecies expected to be fulfilled
        likelihood_supernatural = 0.999

        # Likelihood under natural: compound probability
        likelihood_natural = max(compound_natural_prob, 1e-300)

        # Evidence (normalization constant)
        evidence = (
            likelihood_supernatural * prior_supernatural +
            likelihood_natural * prior_natural
        )

        # Posteriors
        if evidence > 0:
            posterior_supernatural = (
                likelihood_supernatural * prior_supernatural
            ) / evidence
            posterior_natural = (
                likelihood_natural * prior_natural
            ) / evidence
        else:
            posterior_supernatural = prior_supernatural
            posterior_natural = prior_natural

        # Bayes factor
        if likelihood_natural > 0:
            bayes_factor = likelihood_supernatural / likelihood_natural
            log_bf = math.log10(max(bayes_factor, 1e-300))
        else:
            bayes_factor = float('inf')
            log_bf = float('inf')

        # Interpretation
        interpretation = self._interpret_bayes_factor(bayes_factor)

        # Credible interval using Beta approximation
        ci = self._calculate_credible_interval(
            posterior_supernatural, len(self.current_set)
        )

        return BayesianResult(
            prior_supernatural=prior_supernatural,
            prior_natural=prior_natural,
            likelihood_given_supernatural=likelihood_supernatural,
            likelihood_given_natural=likelihood_natural,
            posterior_supernatural=posterior_supernatural,
            posterior_natural=posterior_natural,
            bayes_factor=bayes_factor,
            log_bayes_factor=log_bf,
            interpretation=interpretation,
            credible_interval=ci,
        )

    def _interpret_bayes_factor(self, bf: float) -> str:
        """Interpret Bayes factor on Jeffreys' scale."""
        if math.isinf(bf):
            return "Decisive evidence for supernatural origin (infinite Bayes factor)"
        strength = EvidenceStrength.from_bayes_factor(bf)
        messages = {
            EvidenceStrength.DECISIVE: "Decisive evidence for supernatural origin",
            EvidenceStrength.VERY_STRONG: "Very strong evidence for supernatural origin",
            EvidenceStrength.STRONG: "Strong evidence for supernatural origin",
            EvidenceStrength.SUBSTANTIAL: "Substantial evidence for supernatural origin",
            EvidenceStrength.MODERATE: "Moderate evidence for supernatural origin",
            EvidenceStrength.WEAK: "Weak evidence (anecdotal)",
            EvidenceStrength.INCONCLUSIVE: "Inconclusive",
        }
        return messages[strength]

    def _calculate_credible_interval(
        self,
        posterior: float,
        n_prophecies: int,
    ) -> Tuple[float, float]:
        """Calculate 95% credible interval using Beta approximation."""
        # Use Beta distribution with effective sample size
        effective_n = max(n_prophecies, 2)
        alpha = posterior * effective_n + 1
        beta_param = (1 - posterior) * effective_n + 1

        dist = stats.beta(alpha, beta_param)
        ci = (dist.ppf(0.025), dist.ppf(0.975))
        return ci

    def _sensitivity_analysis(
        self,
        compound_prob: float,
    ) -> Dict[str, BayesianResult]:
        """Test conclusions across range of prior assumptions."""
        results = {}
        for name, prior in self.config.sensitivity_priors.items():
            results[name] = self._calculate_bayesian_posterior(compound_prob, prior)
        return results

    # =========================================================================
    # ENRICHMENT
    # =========================================================================

    async def _enrich_prophecy(
        self,
        prophecy: ProphecyFulfillmentPair,
    ) -> ProphecyFulfillmentPair:
        """Enrich prophecy with data from other oracles."""
        # Necessity score from InterVerseNecessityCalculator
        if self.necessity_calc:
            try:
                result = await self.necessity_calc.calculate_necessity(
                    prophecy.prophecy_verse,
                    prophecy.fulfillment_verse,
                )
                prophecy.necessity_score = result.necessity_score
            except Exception as e:
                logger.debug(f"Could not calculate necessity: {e}")

        # Manuscript support from LXXChristologicalExtractor
        if self.lxx_extractor:
            try:
                result = await self.lxx_extractor.extract_christological_content(
                    prophecy.prophecy_verse
                )
                if result.divergences:
                    prophecy.manuscript_support = max(
                        d.manuscript_confidence for d in result.divergences
                    )
            except Exception as e:
                logger.debug(f"Could not extract LXX data: {e}")

        # Typological depth from HyperFractalTypologyEngine
        if self.typology_engine:
            try:
                result = await self.typology_engine.analyze_fractal_typology(
                    prophecy.prophecy_verse,
                    prophecy.fulfillment_verse,
                )
                prophecy.typological_depth = result.fractal_depth
            except Exception as e:
                logger.debug(f"Could not analyze typology: {e}")

        return prophecy

    # =========================================================================
    # SUMMARY GENERATION
    # =========================================================================

    def _generate_summary(
        self,
        total_prophecies: int,
        independent_count: int,
        bayesian: BayesianResult,
    ) -> str:
        """Generate human-readable summary of analysis."""
        bf_str = (
            "infinite" if math.isinf(bayesian.bayes_factor)
            else f"{bayesian.bayes_factor:.2e}"
        )

        return (
            f"Analysis of {total_prophecies} prophecy-fulfillment pairs "
            f"({independent_count} independent). "
            f"Posterior P(supernatural) = {bayesian.posterior_supernatural:.4f}. "
            f"Bayes factor = {bf_str}. "
            f"Conclusion: {bayesian.interpretation}"
        )

    def _generate_methodology_notes(
        self,
        prophecies: List[ProphecyFulfillmentPair],
        independence_data: List[IndependenceAnalysis],
    ) -> str:
        """Generate methodology documentation."""
        clusters = sum(
            1 for a in independence_data
            if a.independence_level == IndependenceLevel.CLUSTER
        )
        partial = sum(
            1 for a in independence_data
            if a.independence_level == IndependenceLevel.PARTIALLY_DEPENDENT
        )

        return (
            f"Methodology: Specificity-factor probability estimation with "
            f"independence adjustment. "
            f"Found {clusters} clustered prophecies (same event), "
            f"{partial} partially dependent. "
            f"Used log-probability arithmetic for numerical stability. "
            f"Bayesian posterior calculated with user-specified prior. "
            f"Sensitivity analysis across 6 prior assumptions."
        )

    # =========================================================================
    # RESOURCE MANAGEMENT
    # =========================================================================

    async def initialize(self) -> None:
        """Initialize connections and resources."""
        logger.info("PropheticNecessityProver initialized")

    async def cleanup(self) -> None:
        """Cleanup resources."""
        self.current_set = []
        logger.info("PropheticNecessityProver cleaned up")
