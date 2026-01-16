# SESSION 07: PROPHETIC NECESSITY PROVER

## Session Overview

**Objective**: Implement the `PropheticNecessityProver` engine that uses Bayesian probability mathematics to calculate whether biblical prophecy fulfillment patterns constitute evidence of supernatural origin. This is the fifth and final of the Five Impossible Oracles.

**Prerequisites**:
- Session 04 complete (Inter-Verse Necessity Calculator)
- Session 05 complete (LXX Christological Extractor)
- Session 06 complete (Fractal Typology Engine)
- Understanding of Bayesian probability and Bayes factors
- Familiarity with prophecy-fulfillment scholarship

---

## Part 1: Understanding the Oracle Concept

### Core Capability

Given a set of prophecy-fulfillment pairs, calculate:
1. **Individual Fulfillment Probability**: P(fulfillment by chance | natural causes only)
2. **Cumulative Independence**: Are these prophecies independent events?
3. **Compound Probability**: P(all fulfillments | natural causes only)
4. **Bayesian Posterior**: P(supernatural origin | observed evidence)

### The Mathematical Foundation

**Bayes' Theorem Applied**:
```
P(supernatural | evidence) = P(evidence | supernatural) × P(supernatural)
                              ─────────────────────────────────────────────
                                            P(evidence)

Where:
- P(supernatural) = Prior probability (user-configurable)
- P(evidence | supernatural) = Likelihood under supernatural hypothesis (~1.0)
- P(evidence | natural) = Compound probability of natural fulfillments
- P(evidence) = Normalization constant
```

**Bayes Factor Interpretation (Jeffreys' Scale)**:
| Bayes Factor | Evidence Strength |
|--------------|------------------|
| > 10^10 | Decisive |
| 10^6 - 10^10 | Very Strong |
| 10^3 - 10^6 | Strong |
| 30 - 10^3 | Substantial |
| 10 - 30 | Moderate |
| 3 - 10 | Weak |
| < 3 | Inconclusive |

### Canonical Example: Christ's Birth Details

**Independent Prophecy Set**:
| Prophecy | Reference | Natural Probability | Factors |
|----------|-----------|-------------------|---------|
| Virgin birth | ISA.7.14 | ~0 (biological impossibility) | BIOLOGICAL_MIRACLE |
| Born in Bethlehem | MIC.5.2 | 1/1000 (specific village) | LOCATION_CITY |
| Of David's line | 2SA.7.12-16 | 1/50 (post-exile genealogy loss) | GENEALOGICAL |
| Second Temple period | DAN.9.24-27 | 1/100 (490-year window) | TIME_PERIOD |
| Preceded by messenger | MAL.3.1 | 1/100 | SEQUENCE_EVENTS |

**Compound Probability (if independent)**:
```
P(all | natural) = 0 × 0.001 × 0.02 × 0.01 × 0.01 ≈ 0
Bayes Factor → ∞ (decisive evidence)
```

---

## Part 2: Technical Specification

### File: `ml/engines/prophetic_prover.py`

### Core Enums

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


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
        return {
            FulfillmentType.EXPLICIT: 1.0,
            FulfillmentType.TYPOLOGICAL: 0.8,
            FulfillmentType.SYMBOLIC: 0.6,
            FulfillmentType.MULTIPLE: 0.9,
            FulfillmentType.CONDITIONAL: 0.5
        }[self]


class IndependenceLevel(Enum):
    """How independent a prophecy is from others."""
    FULLY_INDEPENDENT = "fully_independent"
    PARTIALLY_DEPENDENT = "partially_dependent"
    CAUSALLY_LINKED = "causally_linked"
    CLUSTER = "cluster"  # Same event, multiple predictions

    @property
    def effective_weight(self) -> float:
        """Contribution to compound probability calculation."""
        return {
            IndependenceLevel.FULLY_INDEPENDENT: 1.0,
            IndependenceLevel.PARTIALLY_DEPENDENT: 0.7,
            IndependenceLevel.CAUSALLY_LINKED: 0.3,
            IndependenceLevel.CLUSTER: 0.0  # Counted once per cluster
        }[self]


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

    @property
    def base_probability(self) -> float:
        """Base probability for this factor occurring by chance."""
        return {
            SpecificityFactor.PERSON_NAME: 1e-6,
            SpecificityFactor.LOCATION_CITY: 0.001,
            SpecificityFactor.LOCATION_REGION: 0.01,
            SpecificityFactor.TIME_PERIOD: 0.01,
            SpecificityFactor.MANNER_OF_DEATH: 0.01,
            SpecificityFactor.BIOLOGICAL_MIRACLE: 1e-15,  # Essentially impossible
            SpecificityFactor.EXACT_PRICE: 1e-4,
            SpecificityFactor.SEQUENCE_EVENTS: 0.001,
            SpecificityFactor.NUMERICAL_DETAIL: 0.01,
            SpecificityFactor.GENEALOGICAL: 0.02
        }[self]
```

### Core Data Classes

```python
@dataclass
class ProbabilityEstimation:
    """Probability estimation with uncertainty quantification."""
    point_estimate: float
    confidence_interval: Tuple[float, float]  # 95% CI
    estimation_method: str
    specificity_factors: List[SpecificityFactor]
    historical_evidence: List[str]
    scholarly_range: Tuple[float, float]

    def as_beta_distribution(self) -> stats.beta:
        """Convert to Beta distribution for Bayesian updating."""
        mean = self.point_estimate
        low, high = self.confidence_interval
        var = ((high - low) / 4) ** 2

        if var > 0 and 0 < mean < 1:
            common = mean * (1 - mean) / var - 1
            alpha = max(0.1, mean * common)
            beta = max(0.1, (1 - mean) * common)
            return stats.beta(alpha, beta)
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
    independence_level: IndependenceLevel
    dependent_on: List[str] = field(default_factory=list)
    cluster_id: Optional[str] = None

    # Enrichment from Sessions 04-06
    necessity_score: float = 0.0        # From Session 04
    manuscript_support: float = 1.0     # From Session 05
    typological_depth: int = 0          # From Session 06

    @property
    def natural_probability(self) -> float:
        return self.probability_estimate.point_estimate

    @property
    def effective_probability(self) -> float:
        """Probability adjusted by independence weight."""
        return self.natural_probability ** self.independence_level.effective_weight


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
    def strength_category(self) -> str:
        """Jeffreys' scale interpretation."""
        bf = self.bayes_factor
        if bf > 1e10: return "decisive"
        elif bf > 1e6: return "very_strong"
        elif bf > 1e3: return "strong"
        elif bf > 30: return "substantial"
        elif bf > 10: return "moderate"
        elif bf > 3: return "weak"
        return "inconclusive"


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
```

---

## Part 3: Probability Estimation Algorithms

### Algorithm 1: Specificity-Based Probability Estimation

```python
async def estimate_natural_probability(
    self, prophecy: ProphecyFulfillmentPair
) -> ProbabilityEstimation:
    """Estimate probability based on prophecy specificity factors."""

    # Detect specificity factors
    factors = await self.detect_specificity_factors(prophecy.prophecy_text)

    if not factors:
        base_probability = 0.1  # Generic prediction
    else:
        # Multiply probabilities for independent factors
        base_probability = 1.0
        for factor in factors:
            base_probability *= factor.base_probability

    # Historical adjustment based on similar prophecy patterns
    historical_adj = await self._get_historical_adjustment(
        prophecy.prophecy_verse, factors
    )
    adjusted = base_probability * historical_adj

    # Confidence interval width inversely proportional to factor count
    certainty = min(0.9, 0.5 + 0.1 * len(factors))
    ci_width = adjusted * (1 - certainty)

    ci_low = max(1e-20, adjusted - ci_width)
    ci_high = min(1.0, adjusted + ci_width)

    return ProbabilityEstimation(
        point_estimate=adjusted,
        confidence_interval=(ci_low, ci_high),
        estimation_method="specificity_factor_analysis",
        specificity_factors=factors,
        historical_evidence=await self._gather_historical_evidence(prophecy),
        scholarly_range=await self._get_scholarly_range(prophecy.prophecy_id)
    )


async def detect_specificity_factors(
    self, prophecy_text: str
) -> List[SpecificityFactor]:
    """Detect specificity factors in prophecy text using NLP."""
    factors = []

    # Location detection
    locations = await self.ner.extract_locations(prophecy_text)
    if any(loc['type'] == 'city' for loc in locations):
        factors.append(SpecificityFactor.LOCATION_CITY)
    elif locations:
        factors.append(SpecificityFactor.LOCATION_REGION)

    # Biological impossibility detection
    miracle_patterns = ['virgin', 'conceive without', 'barren shall bear']
    if any(p in prophecy_text.lower() for p in miracle_patterns):
        factors.append(SpecificityFactor.BIOLOGICAL_MIRACLE)

    # Time period detection
    time_markers = await self.temporal_extractor.extract(prophecy_text)
    if time_markers:
        factors.append(SpecificityFactor.TIME_PERIOD)

    # Numerical details
    numbers = await self.numerical_extractor.extract(prophecy_text)
    if numbers:
        factors.append(SpecificityFactor.NUMERICAL_DETAIL)

    # Manner of death
    death_terms = ['pierced', 'crucified', 'bones', 'stripes', 'wounds']
    if any(t in prophecy_text.lower() for t in death_terms):
        factors.append(SpecificityFactor.MANNER_OF_DEATH)

    return factors
```

### Algorithm 2: Independence Assessment

```python
async def assess_independence(
    self, prophecy_id: str, all_prophecies: Dict[str, ProphecyFulfillmentPair]
) -> IndependenceAnalysis:
    """Assess independence level relative to other prophecies."""

    prophecy = all_prophecies[prophecy_id]
    related = []
    shared_factors = []

    for other_id, other in all_prophecies.items():
        if other_id == prophecy_id:
            continue

        # Same fulfillment event = cluster
        if await self._same_fulfillment_event(prophecy, other):
            return IndependenceAnalysis(
                prophecy_id=prophecy_id,
                independence_level=IndependenceLevel.CLUSTER,
                related_prophecies=[other_id],
                shared_factors=["same_fulfillment_event"],
                effective_contribution=0.0,
                reasoning=f"Clustered with {other_id}: same event"
            )

        # Causal link detection
        if await self._prophecy_causes_other(prophecy, other):
            return IndependenceAnalysis(
                prophecy_id=prophecy_id,
                independence_level=IndependenceLevel.CAUSALLY_LINKED,
                related_prophecies=[other_id],
                shared_factors=["causal_relationship"],
                effective_contribution=0.3,
                reasoning=f"Causally linked to {other_id}"
            )

        # Shared historical factors
        shared = await self._find_shared_factors(prophecy, other)
        if shared:
            related.append(other_id)
            shared_factors.extend(shared)

    if shared_factors:
        effective = max(0.5, 1.0 - 0.1 * len(set(shared_factors)))
        return IndependenceAnalysis(
            prophecy_id=prophecy_id,
            independence_level=IndependenceLevel.PARTIALLY_DEPENDENT,
            related_prophecies=related,
            shared_factors=list(set(shared_factors)),
            effective_contribution=effective,
            reasoning=f"Shares {len(set(shared_factors))} factors with other prophecies"
        )

    return IndependenceAnalysis(
        prophecy_id=prophecy_id,
        independence_level=IndependenceLevel.FULLY_INDEPENDENT,
        related_prophecies=[],
        shared_factors=[],
        effective_contribution=1.0,
        reasoning="No shared causal or historical factors detected"
    )
```

### Algorithm 3: Compound Probability Calculation

```python
async def calculate_compound_probability(
    self,
    prophecies: List[ProphecyFulfillmentPair],
    independence_data: List[IndependenceAnalysis]
) -> Tuple[float, float]:
    """Calculate compound probability with dependency adjustments."""

    independence_map = {a.prophecy_id: a for a in independence_data}

    # Categorize by independence type
    fully_independent = []
    clusters: Dict[str, List[ProphecyFulfillmentPair]] = {}
    partial_deps = []

    for p in prophecies:
        analysis = independence_map[p.prophecy_id]

        if analysis.independence_level == IndependenceLevel.FULLY_INDEPENDENT:
            fully_independent.append(p)
        elif analysis.independence_level == IndependenceLevel.CLUSTER:
            cluster_id = p.cluster_id or analysis.related_prophecies[0]
            clusters.setdefault(cluster_id, []).append(p)
        elif analysis.independence_level == IndependenceLevel.PARTIALLY_DEPENDENT:
            partial_deps.append((p, analysis))
        # CAUSALLY_LINKED: skip (counted via cause)

    # Calculate compound (use log scale for numerical stability)
    log_compound = 0.0

    # Fully independent: multiply directly
    for p in fully_independent:
        prob = max(p.natural_probability, 1e-20)
        log_compound += np.log(prob)

    # Clusters: use minimum probability (most specific)
    for cluster_prophets in clusters.values():
        min_prob = min(p.natural_probability for p in cluster_prophets)
        log_compound += np.log(max(min_prob, 1e-20))

    # Partial dependencies: apply effective weight
    for p, analysis in partial_deps:
        prob = max(p.natural_probability, 1e-20)
        log_compound += np.log(prob) * analysis.effective_contribution

    compound = np.exp(log_compound) if log_compound > -700 else 0.0

    return compound, log_compound
```

### Algorithm 4: Bayesian Posterior Calculation

```python
def calculate_bayesian_posterior(
    self,
    compound_natural_prob: float,
    prior_supernatural: float
) -> BayesianResult:
    """Apply Bayes' theorem with proper handling of extreme probabilities."""

    prior_natural = 1.0 - prior_supernatural

    # Likelihood under supernatural: prophecies expected to be fulfilled
    likelihood_supernatural = 0.999

    # Likelihood under natural: compound probability
    likelihood_natural = max(compound_natural_prob, 1e-300)

    # Evidence (normalization)
    evidence = (likelihood_supernatural * prior_supernatural +
                likelihood_natural * prior_natural)

    # Posteriors
    posterior_supernatural = (likelihood_supernatural * prior_supernatural) / evidence
    posterior_natural = (likelihood_natural * prior_natural) / evidence

    # Bayes factor
    if likelihood_natural > 0:
        bayes_factor = likelihood_supernatural / likelihood_natural
        log_bf = np.log10(max(bayes_factor, 1e-300))
    else:
        bayes_factor = float('inf')
        log_bf = float('inf')

    # Interpretation
    interpretation = self._interpret_bayes_factor(bayes_factor)

    # Credible interval using Beta approximation
    ci = self._calculate_credible_interval(posterior_supernatural, len(self.current_set))

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
        credible_interval=ci
    )


def _interpret_bayes_factor(self, bf: float) -> str:
    """Interpret Bayes factor on Jeffreys' scale."""
    if bf > 1e10:
        return "Decisive evidence for supernatural origin"
    elif bf > 1e6:
        return "Very strong evidence for supernatural origin"
    elif bf > 1e3:
        return "Strong evidence for supernatural origin"
    elif bf > 30:
        return "Substantial evidence for supernatural origin"
    elif bf > 10:
        return "Moderate evidence for supernatural origin"
    elif bf > 3:
        return "Weak evidence (anecdotal)"
    return "Inconclusive"


def sensitivity_analysis(self, compound_prob: float) -> Dict[str, BayesianResult]:
    """Test conclusions across range of prior assumptions."""
    priors = {
        "very_skeptical": 0.001,
        "skeptical": 0.01,
        "mildly_skeptical": 0.1,
        "neutral": 0.5,
        "mildly_favorable": 0.7,
        "favorable": 0.9
    }
    return {name: self.calculate_bayesian_posterior(compound_prob, prior)
            for name, prior in priors.items()}
```

---

## Part 4: Main Engine Class

```python
class PropheticNecessityProver:
    """
    Bayesian engine for prophetic necessity analysis.
    Fifth and final of the Five Impossible Oracles.
    """

    def __init__(
        self,
        necessity_calc,         # Session 04
        lxx_extractor,          # Session 05
        typology_engine,        # Session 06
        prophecy_catalog: Dict[str, ProphecyFulfillmentPair],
        config: Optional['PropheticProverConfig'] = None
    ):
        self.necessity_calc = necessity_calc
        self.lxx_extractor = lxx_extractor
        self.typology_engine = typology_engine
        self.prophecy_catalog = prophecy_catalog
        self.config = config or PropheticProverConfig()
        self.current_set: List[ProphecyFulfillmentPair] = []

        logger.info(f"PropheticNecessityProver initialized with {len(prophecy_catalog)} prophecies")

    async def prove_prophetic_necessity(
        self,
        prophecy_ids: List[str],
        prior_supernatural: float = 0.5
    ) -> PropheticProofResult:
        """Main entry point: complete prophetic proof analysis."""

        # 1. Gather prophecy-fulfillment pairs
        prophecies = [self.prophecy_catalog[pid] for pid in prophecy_ids]
        self.current_set = prophecies

        # 2. Enrich with Sessions 04-06 data
        enriched = []
        for p in prophecies:
            p = await self._enrich_with_necessity(p)
            p = await self._enrich_with_lxx(p)
            p = await self._enrich_with_typology(p)
            enriched.append(p)

        # 3. Estimate probabilities if not pre-computed
        for p in enriched:
            if p.probability_estimate.point_estimate == 0:
                p.probability_estimate = await self.estimate_natural_probability(p)

        # 4. Assess independence
        prophecy_dict = {p.prophecy_id: p for p in enriched}
        independence_analyses = [
            await self.assess_independence(p.prophecy_id, prophecy_dict)
            for p in enriched
        ]

        # 5. Calculate compound probability
        compound_prob, log_prob = await self.calculate_compound_probability(
            enriched, independence_analyses
        )

        # 6. Bayesian analysis
        bayesian = self.calculate_bayesian_posterior(compound_prob, prior_supernatural)

        # 7. Sensitivity analysis
        sensitivity = self.sensitivity_analysis(compound_prob)

        # 8. Compile results
        independent_count = sum(
            1 for a in independence_analyses
            if a.independence_level == IndependenceLevel.FULLY_INDEPENDENT
        )
        effective_count = sum(a.effective_contribution for a in independence_analyses)

        enrichment_summary = {
            "avg_necessity": np.mean([p.necessity_score for p in enriched]),
            "avg_manuscript_support": np.mean([p.manuscript_support for p in enriched]),
            "avg_typological_depth": np.mean([p.typological_depth for p in enriched])
        }

        return PropheticProofResult(
            prophecy_set=enriched,
            independent_count=independent_count,
            effective_count=effective_count,
            compound_natural_probability=compound_prob,
            log_probability=log_prob,
            independence_analysis=independence_analyses,
            bayesian_result=bayesian,
            sensitivity_analysis=sensitivity,
            enrichment_summary=enrichment_summary,
            confidence=self._calculate_overall_confidence(enriched, independence_analyses),
            summary=self._generate_summary(len(enriched), independent_count, compound_prob, bayesian),
            methodology_notes=self._generate_methodology_notes()
        )
```

---

## Part 5: Integration Points

### Session 04: Necessity Calculator

```python
async def _enrich_with_necessity(self, pair: ProphecyFulfillmentPair) -> ProphecyFulfillmentPair:
    """Add necessity score from Session 04."""
    necessity = await self.necessity_calc.calculate_necessity(
        verse_a=pair.fulfillment_verse,
        verse_b=pair.prophecy_verse
    )
    pair.necessity_score = necessity.necessity_score
    return pair
```

### Session 05: LXX Extractor

```python
async def _enrich_with_lxx(self, pair: ProphecyFulfillmentPair) -> ProphecyFulfillmentPair:
    """Add manuscript support from Session 05."""
    lxx_result = await self.lxx_extractor.extract_christological_content(pair.prophecy_verse)

    if lxx_result.divergences:
        christological_divs = [d for d in lxx_result.divergences if d.christological_category]
        if christological_divs:
            pair.manuscript_support = max(d.manuscript_confidence for d in christological_divs)

    return pair
```

### Session 06: Typology Engine

```python
async def _enrich_with_typology(self, pair: ProphecyFulfillmentPair) -> ProphecyFulfillmentPair:
    """Add typological depth from Session 06."""
    typology = await self.typology_engine.analyze_fractal_typology(
        type_ref=pair.prophecy_verse,
        antitype_ref=pair.fulfillment_verse
    )
    pair.typological_depth = typology.fractal_depth
    return pair
```

### Neo4j Schema

```cypher
// Prophecy fulfillment data
CREATE (p:Prophecy {
    id: "virgin_birth",
    verse: "ISA.7.14",
    natural_probability: 1e-15,
    independence_level: "fully_independent"
})-[:FULFILLED_IN {
    confidence: 0.99,
    necessity_score: 0.95,
    typological_depth: 3
}]->(f:Fulfillment {verse: "MAT.1.23"})

// Query for analysis
MATCH (p:Prophecy)-[r:FULFILLED_IN]->(f)
WHERE p.independence_level = "fully_independent"
RETURN p.id, p.natural_probability, r.necessity_score
ORDER BY p.natural_probability ASC
```

---

## Part 6: Prophecy Catalog Structure

### `data/messianic_prophecies.json`

```json
{
  "virgin_birth": {
    "prophecy_verse": "ISA.7.14",
    "fulfillment_verse": "MAT.1.23",
    "fulfillment_type": "explicit",
    "natural_probability": 1e-15,
    "specificity_factors": ["biological_miracle"],
    "reasoning": "Biologically impossible without divine intervention",
    "independence_level": "fully_independent",
    "manuscript_sources": ["4QIsaa", "Codex Vaticanus", "LXX παρθένος"]
  },
  "bethlehem_birth": {
    "prophecy_verse": "MIC.5.2",
    "fulfillment_verse": "MAT.2.1",
    "fulfillment_type": "explicit",
    "natural_probability": 0.001,
    "specificity_factors": ["location_city"],
    "reasoning": "Bethlehem population / total population",
    "independence_level": "fully_independent"
  },
  "thirty_pieces": {
    "prophecy_verse": "ZEC.11.12-13",
    "fulfillment_verse": "MAT.27.3-10",
    "fulfillment_type": "explicit",
    "natural_probability": 0.0001,
    "specificity_factors": ["exact_price", "location_city"],
    "independence_level": "partially_dependent",
    "dependent_on": ["betrayal"],
    "cluster_id": "passion_cluster"
  }
}
```

---

## Part 7: Testing Specification

```python
class TestPropheticProver:

    @pytest.mark.asyncio
    async def test_single_prophecy_probability(self, prover):
        """Virgin birth should have effectively zero natural probability."""
        result = await prover.estimate_natural_probability(
            prover.prophecy_catalog["virgin_birth"]
        )
        assert result.point_estimate < 1e-10
        assert SpecificityFactor.BIOLOGICAL_MIRACLE in result.specificity_factors

    @pytest.mark.asyncio
    async def test_independence_detection(self, prover):
        """Virgin birth and Bethlehem should be fully independent."""
        prophecies = {
            "virgin_birth": prover.prophecy_catalog["virgin_birth"],
            "bethlehem_birth": prover.prophecy_catalog["bethlehem_birth"]
        }
        analysis = await prover.assess_independence("virgin_birth", prophecies)
        assert analysis.independence_level == IndependenceLevel.FULLY_INDEPENDENT

    @pytest.mark.asyncio
    async def test_bayesian_with_neutral_prior(self, prover):
        """Neutral prior with very low compound should yield high posterior."""
        result = prover.calculate_bayesian_posterior(
            compound_natural_prob=1e-20,
            prior_supernatural=0.5
        )
        assert result.posterior_supernatural > 0.999
        assert result.strength_category == "decisive"

    @pytest.mark.asyncio
    async def test_sensitivity_analysis(self, prover):
        """Sensitivity should show robustness to prior assumptions."""
        sensitivity = prover.sensitivity_analysis(1e-15)
        assert all(r.bayes_factor > 1e10 for r in sensitivity.values())
```

---

## Part 8: Configuration

```python
@dataclass
class PropheticProverConfig:
    """Configuration for PropheticNecessityProver."""

    # Prior handling
    default_prior_supernatural: float = 0.5
    sensitivity_priors: List[float] = field(
        default_factory=lambda: [0.001, 0.01, 0.1, 0.5, 0.7, 0.9]
    )

    # Probability bounds
    min_probability_floor: float = 1e-20
    cluster_aggregation: str = "minimum"  # "minimum", "average", "geometric"

    # Integration toggles
    include_necessity_enrichment: bool = True
    include_lxx_enrichment: bool = True
    include_typology_enrichment: bool = True

    # Thresholds
    independence_correlation_threshold: float = 0.7
    require_manuscript_support: bool = True
    min_bayes_factor_for_evidence: float = 10.0
```

---

## Part 9: Success Criteria

### Functional Requirements
- [ ] Estimates individual prophecy probabilities from specificity factors
- [ ] Correctly assesses independence relationships
- [ ] Calculates compound probability with dependency adjustments
- [ ] Performs valid Bayesian analysis with Bayes factors
- [ ] Integrates data from Sessions 04-06
- [ ] Runs sensitivity analysis across prior assumptions

### Mathematical Accuracy
- [ ] Log-scale handling for very small probabilities
- [ ] Proper independence adjustments (clusters, partial deps)
- [ ] Bayesian calculations match known test cases

### Theological Validity
- [ ] Virgin birth treated as biologically impossible
- [ ] Location prophecies use realistic population ratios
- [ ] Passion cluster treated as related events

---

## Part 10: Philosophical Notes

### Handling Common Objections

| Objection | Response | System Integration |
|-----------|----------|-------------------|
| Prophecies post-dated | LXX/DSS predate Christianity | Session 05 manuscript dating |
| Fulfillments fabricated | Some fulfilled by enemies (Romans) | Independence analysis |
| Prior should be 0 | Sensitivity shows robustness | Multiple prior analysis |

### Transparency Requirements

The engine must:
1. Show all probability estimates with reasoning
2. Acknowledge uncertainty via confidence intervals
3. Present full sensitivity analysis
4. Allow user-configurable priors
5. Never claim certainty beyond mathematical support

---

## Session Completion Checklist

```markdown
- [ ] `ml/engines/prophetic_prover.py` implemented
- [ ] All enums and dataclasses defined
- [ ] `data/messianic_prophecies.json` created
- [ ] Probability estimation from specificity factors
- [ ] Independence assessment functional
- [ ] Compound probability calculation correct
- [ ] Bayesian analysis with Bayes factors
- [ ] Integration with Sessions 04-06
- [ ] Sensitivity analysis working
- [ ] Configuration in config.py
- [ ] Unit tests passing
```

**Next Session**: SESSION 08: Event Sourcing Migration
