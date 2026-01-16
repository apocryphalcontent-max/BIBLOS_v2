# SESSION 06: HYPER-FRACTAL TYPOLOGY ENGINE

## Session Overview

**Objective**: Implement the `HyperFractalTypologyEngine` that discovers typological connections at multiple fractal layers - from individual words to covenantal arcs spanning testaments. This is the fourth of the Five Impossible Oracles and arguably the most theologically significant, as it operationalizes the patristic understanding that Scripture exhibits self-similar prophetic patterns at every scale of textual analysis.

**Prerequisites**:
- Session 01 complete (Mutual Transformation Metric)
- Session 03 complete (Omni-Contextual Resolver)
- Session 04 complete (Inter-Verse Necessity Calculator)
- Understanding of Orthodox typological hermeneutics
- Familiarity with fractal/recursive data structures
- Knowledge of self-similarity mathematics (Hausdorff dimension concepts)

---

## Part 1: Understanding the Oracle Concept

### Core Capability

Biblical typology operates at multiple, nested "fractal" layers. Each layer reveals type-antitype connections at different scales, and crucially, **the same typological pattern manifests self-similarly across scales** - this is the fractal insight that distinguishes this engine from simple pattern matching:

1. **WORD Layer**: Individual words that are types (e.g., "seed" → Christ)
2. **PHRASE Layer**: Phrases that prefigure (e.g., "lamb of God")
3. **VERSE Layer**: Complete verse correspondences
4. **PERICOPE Layer**: Narrative units as types (3-30 verses)
5. **CHAPTER Layer**: Extended passage parallels
6. **BOOK Layer**: Entire book structures as types
7. **COVENANTAL Layer**: Multi-book covenant arc fulfillments

### Fractal Nature and Self-Similarity Mathematics

Just as mathematical fractals show self-similar patterns at every zoom level, biblical typology exhibits type-antitype patterns at every textual scale. The burning bush is a type at the pericope level, but within it, "fire" is a type at the word level, and "holy ground" is a type at the phrase level.

**Self-Similarity Coefficient**: We quantify this using a modified Hausdorff dimension approach:

```
D_typological = lim(ε→0) [log(N(ε)) / log(1/ε)]

Where:
- N(ε) = number of typological connections found at scale ε
- ε = textual granularity (WORD=1, PHRASE=2, ..., COVENANTAL=7)
```

A **true fractal typology** exhibits D ≈ 1.5-2.0, meaning connections persist across scales. A **shallow typology** has D < 1.0, meaning connections exist only at one or two adjacent scales.

### Canonical Example: Isaac/Christ Typology (Multi-Layer)

**WORD Layer**:
- "only son" (יָחִיד/μονογενής) → JHN.3.16 "only begotten Son"
- "lamb" (שֶׂה) → JHN.1.29 "Lamb of God"
- "wood" (עֵצִים) → Cross (σταυρός)

**PHRASE Layer**:
- "take your son... offer him" → "God gave his Son"
- "the wood of the burnt offering" → "bearing his own cross"
- "on one of the mountains I will tell you" → "Golgotha, place of the skull"

**VERSE Layer**:
- GEN.22.2 → JHN.3.16 (Father offering Son)
- GEN.22.8 → JHN.1.29 (God provides the lamb)
- GEN.22.14 → HEB.11.19 (received back from death)

**PERICOPE Layer**:
- Akedah (GEN.22.1-19) → Passion narrative (JHN.18-19)
- Three-day journey → Three days in tomb
- Moriah → Golgotha (same mountain tradition per Origen, Jerome)

**CHAPTER Layer**:
- Genesis 22 (testing/sacrifice) → Hebrews 11 (faith heroes exemplified)

**BOOK Layer**:
- Genesis (beginnings, promises, seed) → Gospel of John (new beginning, fulfillment, Logos)

**COVENANTAL Layer**:
- Abrahamic covenant seed promise → New Covenant in Christ's blood

**Fractal Dimension Calculation for Isaac/Christ**:
```
Layer counts: WORD=3, PHRASE=3, VERSE=3, PERICOPE=3, CHAPTER=1, BOOK=1, COVENANTAL=1
Total connections: 15
Active layers: 7
D ≈ log(15) / log(7) ≈ 1.39 — a **strong fractal typology**
```

---

## Part 2: Exhaustive Data Structure Specification

### File: `ml/engines/fractal_typology.py`

**Location**: `ml/engines/`

**Dependencies**:
```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
from enum import Enum, IntEnum
from abc import ABC, abstractmethod
import logging
import asyncio
import math
from functools import lru_cache
from collections import defaultdict

import numpy as np
from scipy.stats import beta as beta_dist
import networkx as nx

# Internal imports
from ml.engines.mutual_transformation import MutualTransformationMetric
from ml.engines.necessity_calculator import InterVerseNecessityCalculator
from ml.engines.omnicontext_resolver import OmniContextualResolver
from db.neo4j_client import Neo4jClient
from db.redis_client import RedisClient
from data.schemas import VerseSchema, CrossReferenceSchema

logger = logging.getLogger(__name__)
```

### Enumeration Definitions

#### `TypologyLayer` (IntEnum for Ordering and Arithmetic)

```python
class TypologyLayer(IntEnum):
    """
    Seven fractal layers of typological analysis.

    IntEnum allows mathematical operations on layer values,
    enabling fractal dimension calculations and layer comparisons.

    Layer numbering follows textual granularity from finest (1)
    to coarsest (7), matching standard fractal analysis conventions.
    """
    WORD = 1          # Individual lexeme correspondences
    PHRASE = 2        # Multi-word syntactic unit correspondences
    VERSE = 3         # Complete verse-level parallels
    PERICOPE = 4      # Narrative unit types (3-30 verses)
    CHAPTER = 5       # Extended passage structural parallels
    BOOK = 6          # Entire book thematic/structural types
    COVENANTAL = 7    # Multi-book covenant arc fulfillments

    @property
    def textual_span(self) -> Tuple[int, int]:
        """Return typical word count range for this layer."""
        spans = {
            TypologyLayer.WORD: (1, 1),
            TypologyLayer.PHRASE: (2, 8),
            TypologyLayer.VERSE: (5, 40),
            TypologyLayer.PERICOPE: (50, 500),
            TypologyLayer.CHAPTER: (200, 1500),
            TypologyLayer.BOOK: (1000, 50000),
            TypologyLayer.COVENANTAL: (5000, 500000)
        }
        return spans[self]

    @property
    def granularity(self) -> str:
        """Textual granularity descriptor."""
        return {
            TypologyLayer.WORD: "lemma",
            TypologyLayer.PHRASE: "clause",
            TypologyLayer.VERSE: "verse",
            TypologyLayer.PERICOPE: "discourse_unit",
            TypologyLayer.CHAPTER: "chapter",
            TypologyLayer.BOOK: "book",
            TypologyLayer.COVENANTAL: "covenant_arc"
        }[self]

    @property
    def analysis_complexity(self) -> str:
        """Complexity class for this layer's analysis."""
        if self <= TypologyLayer.VERSE:
            return "O(n)"      # Linear in vocabulary size
        elif self <= TypologyLayer.CHAPTER:
            return "O(n²)"     # Pairwise comparison of units
        else:
            return "O(n² log n)"  # Graph traversal with sorting
```

#### `TypeAntitypeRelation` (Theological Relationship Categories)

```python
class TypeAntitypeRelation(Enum):
    """
    Theological relationship categories between type and antitype.

    Based on patristic hermeneutical categories:
    - Irenaeus's recapitulation theory (ἀνακεφαλαίωσις)
    - Origen's allegorical-typological distinction
    - Chrysostom's historical-typological synthesis
    - Theodore of Mopsuestia's progressive revelation
    """
    PREFIGURATION = "prefiguration"
    # Type anticipates antitype chronologically and ontologically
    # Example: Passover lamb → Christ's sacrifice
    # Patristic basis: Chrysostom, Cyril of Alexandria

    FULFILLMENT = "fulfillment"
    # Antitype completes what type promised/initiated
    # Example: Davidic covenant → Christ's eternal kingship
    # Patristic basis: All Fathers; primary NT hermeneutic

    RECAPITULATION = "recapitulation"
    # Antitype re-enacts type at higher ontological level
    # Example: Israel's exodus → Christ's temptation in wilderness
    # Patristic basis: Irenaeus (κεφαλαίωσις/recapitulatio)

    INTENSIFICATION = "intensification"
    # Antitype exceeds type in magnitude, scope, or permanence
    # Example: Mosaic covenant → New Covenant (HEB.8.6 "better covenant")
    # Patristic basis: Hebrews commentary tradition

    INVERSION = "inversion"
    # Antitype reverses/negates type's effect or meaning
    # Example: Adam's disobedience → Christ's obedience (ROM.5.19)
    # Patristic basis: Irenaeus, Paul's Adam-Christ typology

    PARTICIPATION = "participation"
    # Both type and antitype participate in eternal heavenly reality
    # Example: Tabernacle → heavenly sanctuary (HEB.8.5)
    # Patristic basis: Origen, Platonic-Christian synthesis

    ESCALATION = "escalation"
    # Pattern repeats with increasing stakes/scope across history
    # Example: Individual sin → National sin → Cosmic redemption
    # Patristic basis: Theodore of Mopsuestia's historical typology

    CONTRAST = "contrast"
    # Type serves as negative example; antitype provides correction
    # Example: Saul's kingship → David's kingship
    # Note: Different from INVERSION; contrast is moral, inversion is ontological

    @property
    def directionality(self) -> str:
        """Whether relationship is forward, reverse, or bidirectional."""
        if self == TypeAntitypeRelation.INVERSION:
            return "contrast"
        elif self in (TypeAntitypeRelation.PREFIGURATION, TypeAntitypeRelation.FULFILLMENT):
            return "forward"
        return "bidirectional"

    @classmethod
    def from_patristic_term(cls, term: str) -> 'TypeAntitypeRelation':
        """Map patristic Greek terminology to relation type."""
        mapping = {
            "τύπος": cls.PREFIGURATION,
            "σκιά": cls.PREFIGURATION,          # "shadow"
            "ἀντίτυπος": cls.FULFILLMENT,
            "ἀνακεφαλαίωσις": cls.RECAPITULATION,
            "κρείττων": cls.INTENSIFICATION,    # "better"
            "ἀντίθεσις": cls.INVERSION,
            "μέθεξις": cls.PARTICIPATION,
            "αὔξησις": cls.ESCALATION
        }
        return mapping.get(term, cls.PREFIGURATION)
```

#### `CorrespondenceType` and `PatristicConfidence` (Supporting Enums)

```python
class CorrespondenceType(Enum):
    """How type and antitype correspond structurally."""
    LEXICAL = "lexical"          # Same/related vocabulary (שֶׂה → ἀμνός)
    SEMANTIC = "semantic"        # Same meaning, different words
    STRUCTURAL = "structural"    # Parallel narrative structure
    FUNCTIONAL = "functional"    # Same theological role
    SYMBOLIC = "symbolic"        # Symbolic equivalence (blood, water, fire)
    NUMERICAL = "numerical"      # Numerical patterns (3 days, 12, 40, 7)


class PatristicConfidence(Enum):
    """Confidence levels for patristic attestation of a typological connection."""
    EXPLICIT = "explicit"
    # Father directly states this type-antitype connection
    # Example: Chrysostom on Isaac/Christ in Homilies on Genesis

    IMPLICIT = "implicit"
    # Father discusses both texts typologically without explicit "type" language

    DERIVATIVE = "derivative"
    # Connection derivable from Father's hermeneutical principles

    LITURGICAL = "liturgical"
    # Connection embedded in liturgical practice (readings, hymns)

    MODERN_EXTENSION = "modern_extension"
    # Modern scholarship applying patristic methods to pairs not explicitly discussed


class CovenantPhase(Enum):
    """Phases within a covenant arc structure."""
    INITIATION = "initiation"       # Covenant establishment
    STIPULATION = "stipulation"     # Terms and conditions
    PROMISE = "promise"             # Blessings for obedience
    WARNING = "warning"             # Curses for disobedience
    SIGN = "sign"                   # Covenant sign/seal
    RENEWAL = "renewal"             # Covenant renewal ceremony
    FULFILLMENT = "fulfillment"     # Covenant promises realized
    SUPERSESSION = "supersession"   # Covenant completed in new covenant
```

### Comprehensive Dataclass Definitions

#### `TypePattern` (Reusable Typological Pattern)

```python
@dataclass
class TypePattern:
    """
    A reusable typological pattern that can manifest across Scripture.

    Patterns are the "templates" that the fractal engine matches against.
    Each pattern has characteristic vocabulary, semantic markers, and
    canonical instances that train the detection algorithms.
    """
    type_id: str
    # Unique identifier: snake_case, e.g., "sacrificial_lamb"

    pattern_name: str
    # Human-readable name: "Sacrificial Lamb"

    description: str
    # Extended theological description of the pattern

    primary_layer: TypologyLayer
    # The layer where this pattern most strongly manifests

    active_layers: List[TypologyLayer]
    # All layers where this pattern can be detected

    hebrew_keywords: List[str]
    # Hebrew lexemes that trigger this pattern (Strong's or Unicode)

    greek_keywords: List[str]
    # Greek lexemes (LXX and NT) that trigger this pattern

    semantic_markers: List[str]
    # Abstract semantic concepts (English) that identify pattern
    # e.g., ["sacrifice", "substitution", "innocent", "blood"]

    canonical_type: str
    # Primary OT type instance (verse reference)

    canonical_antitype: str
    # Primary NT antitype instance (verse reference)

    secondary_types: List[str] = field(default_factory=list)
    # Additional OT instances of this type

    secondary_antitypes: List[str] = field(default_factory=list)
    # Additional NT fulfillment instances

    relation_type: TypeAntitypeRelation = TypeAntitypeRelation.PREFIGURATION
    # Default relationship type for this pattern

    correspondence_points: Dict[str, str] = field(default_factory=dict)
    # Mapping of type elements to antitype elements
    # e.g., {"innocent_lamb": "sinless_Christ", "blood_shed": "blood_of_Christ"}

    patristic_attestation: Dict[str, List[str]] = field(default_factory=dict)
    # Father name → list of works attesting this pattern
    # e.g., {"Chrysostom": ["Homilies on Genesis 47", "Homilies on John 19"]}

    liturgical_usage: List[str] = field(default_factory=list)
    # Feast days, hymnographic references where pattern is invoked

    inverse_pattern: Optional[str] = None
    # If this pattern has an INVERSION counterpart, its type_id

    def compute_activation_score(
        self,
        text_lemmas: Set[str],
        semantic_concepts: Set[str]
    ) -> float:
        """
        Calculate how strongly this pattern activates for given text.
        Uses fuzzy set intersection with weighted terms.
        """
        keyword_set = set(self.hebrew_keywords + self.greek_keywords)
        keyword_overlap = len(text_lemmas & keyword_set) / max(len(keyword_set), 1)

        marker_set = set(self.semantic_markers)
        semantic_overlap = len(semantic_concepts & marker_set) / max(len(marker_set), 1)

        # Semantic markers weighted slightly higher
        return 0.4 * keyword_overlap + 0.6 * semantic_overlap

    def get_all_instances(self) -> List[str]:
        """Return all type and antitype instances."""
        return [self.canonical_type] + self.secondary_types + \
               [self.canonical_antitype] + self.secondary_antitypes
```

#### `LayerConnection` (Individual Connection at One Layer)

```python
@dataclass
class LayerConnection:
    """
    A single typological connection discovered at a specific layer.

    This is the atomic unit of typological evidence. Multiple LayerConnections
    aggregate into a FractalTypologyResult.
    """
    connection_id: str
    # Unique ID: "{type_ref}:{antitype_ref}:{layer}:{index}"

    source_reference: str
    # Type reference (OT): "GEN.22.8" or span "GEN.22.1-19"

    target_reference: str
    # Antitype reference (NT): "JHN.1.29" or span

    source_text: str
    # Actual text span in type (original language or translation)

    target_text: str
    # Actual text span in antitype

    source_lemmas: List[str]
    # Lemmatized form of source text

    target_lemmas: List[str]
    # Lemmatized form of target text

    layer: TypologyLayer
    # Which fractal layer this connection operates at

    relation: TypeAntitypeRelation
    # Type-antitype relationship

    correspondence_type: CorrespondenceType
    # How the correspondence manifests

    correspondence_strength: float
    # Raw similarity score [0, 1]

    semantic_similarity: float
    # Embedding-based semantic similarity [0, 1]

    structural_similarity: float
    # Syntactic/structural parallel strength [0, 1]

    mutual_transformation: Optional[float] = None
    # From Session 01 metric - how much texts transform each other

    necessity_score: Optional[float] = None
    # From Session 04 - how necessary is type for understanding antitype

    patristic_attestation: List[Tuple[str, str, PatristicConfidence]] = field(
        default_factory=list
    )
    # (Father, Work, Confidence) tuples

    pattern_matches: List[str] = field(default_factory=list)
    # TypePattern IDs that this connection instantiates

    explanation: str = ""
    # Human-readable theological explanation

    evidence_chain: List[str] = field(default_factory=list)
    # Step-by-step reasoning that discovered this connection

    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    # Bayesian credible interval for correspondence_strength

    def enriched_strength(self) -> float:
        """
        Combine correspondence with transformation and necessity.
        This produces the final composite score for this connection.
        """
        base = self.correspondence_strength

        # Add mutual transformation influence
        if self.mutual_transformation is not None:
            base = 0.6 * base + 0.4 * self.mutual_transformation

        # Bonus for high necessity (type is essential for antitype)
        if self.necessity_score is not None and self.necessity_score > 0.5:
            necessity_bonus = (self.necessity_score - 0.5) * 0.2
            base = min(1.0, base + necessity_bonus)

        # Patristic attestation bonus
        if len(self.patristic_attestation) >= 2:
            base = min(1.0, base + 0.05)

        return base

    def compute_composite_score(self, weights: Dict[str, float] = None) -> float:
        """Compute weighted composite score from all evidence types."""
        weights = weights or {
            "correspondence": 0.25,
            "semantic": 0.25,
            "structural": 0.15,
            "transformation": 0.15,
            "necessity": 0.10,
            "patristic": 0.10
        }

        patristic_score = min(1.0, len(self.patristic_attestation) * 0.25)

        score = (
            weights["correspondence"] * self.correspondence_strength +
            weights["semantic"] * self.semantic_similarity +
            weights["structural"] * self.structural_similarity +
            weights["transformation"] * (self.mutual_transformation or 0) +
            weights["necessity"] * (self.necessity_score or 0) +
            weights["patristic"] * patristic_score
        )

        return min(1.0, score)
```

#### `CovenantArc` (Covenant Structure Spanning Books)

```python
@dataclass
class CovenantArc:
    """
    A covenant arc spanning potentially multiple books.

    Covenants are the macro-structural backbone of biblical theology.
    Each arc has phases that correspond to type-antitype relationships
    at the COVENANTAL layer.
    """
    covenant_id: str
    # Unique ID: "adamic", "noahic", "abrahamic", "mosaic", "davidic", "new"

    covenant_name: str
    # Display name: "Abrahamic Covenant"

    description: str
    # Theological description

    initiation_reference: str
    # Where covenant is established: "GEN.12.1-3"

    key_promises: List[str]
    # Central promises: ["land", "seed", "blessing"]

    covenant_signs: List[Tuple[str, str]]
    # (Sign name, Reference): [("circumcision", "GEN.17.11")]

    phases: Dict[CovenantPhase, List[str]]
    # Phase → list of verse references in that phase

    type_events: List[Tuple[str, str]]
    # (Event description, Reference) that prefigure fulfillment

    fulfillment_references: List[str]
    # NT passages that fulfill this covenant

    arc_span: Tuple[str, str]
    # (Start book, End book) covered by this arc

    superseded_by: Optional[str] = None
    # covenant_id of superseding covenant (if any)

    supersedes: Optional[str] = None
    # covenant_id of covenant this supersedes (if any)

    participant_scope: str = "individual"
    # "individual", "family", "nation", "humanity", "creation"

    conditional: bool = True
    # Whether covenant is conditional on human obedience

    def contains_reference(self, ref: str) -> bool:
        """Check if a reference falls within this covenant arc."""
        all_refs = [self.initiation_reference]
        for phase_refs in self.phases.values():
            all_refs.extend(phase_refs)
        all_refs.extend([e[1] for e in self.type_events])
        all_refs.extend(self.fulfillment_references)
        return ref in all_refs

    def get_promise_categories(self) -> Dict[str, List[str]]:
        """Organize references by promise category."""
        # Implementation would categorize verses by which promise they address
        return {}
```

#### `SelfSimilarityAnalysis` (Fractal Mathematics)

```python
@dataclass
class SelfSimilarityAnalysis:
    """
    Mathematical analysis of self-similarity across typological layers.

    This captures the "fractal" nature of biblical typology - the same
    pattern recurring at multiple scales.
    """
    pattern_id: str
    # Which TypePattern this analysis is for

    layer_connection_counts: Dict[TypologyLayer, int]
    # Number of connections found at each layer

    hausdorff_dimension: float
    # Estimated fractal dimension D
    # D ≈ 1.0: linear (connections at one scale only)
    # D ≈ 1.5: moderate fractal (2-3 adjacent scales)
    # D ≈ 2.0: strong fractal (connections across all scales)

    box_counting_curve: List[Tuple[int, int]]
    # (scale, count) pairs for box-counting analysis

    scale_invariance_coefficient: float
    # How consistent pattern strength is across scales [0, 1]

    dominant_scale: TypologyLayer
    # Scale where pattern is strongest

    scale_distribution_entropy: float
    # Shannon entropy of connection distribution across scales
    # High entropy = even distribution (more fractal)

    lacunarity: float
    # Measure of "gappiness" in the fractal structure

    is_true_fractal: bool = False
    # Whether this passes threshold for "true" fractal typology
    # Requires D > 1.2 and scale_invariance > 0.6

    @classmethod
    def compute(
        cls,
        pattern_id: str,
        layer_counts: Dict[TypologyLayer, int]
    ) -> 'SelfSimilarityAnalysis':
        """Compute self-similarity metrics from layer connection counts."""
        active = {k: v for k, v in layer_counts.items() if v > 0}

        if not active:
            return cls(
                pattern_id=pattern_id,
                layer_connection_counts=layer_counts,
                hausdorff_dimension=0.0,
                box_counting_curve=[],
                scale_invariance_coefficient=0.0,
                dominant_scale=TypologyLayer.VERSE,
                scale_distribution_entropy=0.0,
                lacunarity=1.0,
                is_true_fractal=False
            )

        # Box-counting curve
        box_curve = [(int(layer), count) for layer, count in active.items()]
        box_curve.sort(key=lambda x: x[0])

        # Hausdorff dimension estimation via log-log regression
        if len(box_curve) >= 2:
            log_scales = [math.log(7 - s[0] + 1) for s in box_curve]
            log_counts = [math.log(max(s[1], 1)) for s in box_curve]

            n = len(log_scales)
            sum_x = sum(log_scales)
            sum_y = sum(log_counts)
            sum_xy = sum(x * y for x, y in zip(log_scales, log_counts))
            sum_x2 = sum(x * x for x in log_scales)

            denom = n * sum_x2 - sum_x * sum_x
            hausdorff = (n * sum_xy - sum_x * sum_y) / denom if abs(denom) > 1e-10 else 1.0
        else:
            hausdorff = 1.0

        hausdorff = max(0.5, min(2.5, abs(hausdorff)))

        # Scale invariance via coefficient of variation
        counts = list(active.values())
        mean_count = sum(counts) / len(counts)
        variance = sum((c - mean_count) ** 2 for c in counts) / len(counts)
        cv = math.sqrt(variance) / mean_count if mean_count > 0 else 1.0
        scale_invariance = 1.0 / (1.0 + cv)

        # Dominant scale
        dominant = max(active.keys(), key=lambda k: active[k])

        # Entropy of distribution
        total = sum(counts)
        probs = [c / total for c in counts]
        entropy = -sum(p * math.log2(p) if p > 0 else 0 for p in probs)
        max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        # Lacunarity
        lacunarity = variance / (mean_count ** 2) if mean_count > 0 else 1.0

        # True fractal determination
        is_fractal = hausdorff > 1.2 and scale_invariance > 0.5 and len(active) >= 3

        return cls(
            pattern_id=pattern_id,
            layer_connection_counts=layer_counts,
            hausdorff_dimension=hausdorff,
            box_counting_curve=box_curve,
            scale_invariance_coefficient=scale_invariance,
            dominant_scale=dominant,
            scale_distribution_entropy=normalized_entropy,
            lacunarity=lacunarity,
            is_true_fractal=is_fractal
        )
```

#### `FractalTypologyResult` (Complete Analysis Output)

```python
@dataclass
class FractalTypologyResult:
    """
    Complete result of fractal typology analysis between type and antitype.

    This is the primary output of the HyperFractalTypologyEngine.
    """
    result_id: str
    # Unique ID: "{type_ref}:{antitype_ref}:{timestamp}"

    type_reference: str
    # OT type reference or range

    antitype_reference: str
    # NT antitype reference or range

    layers: Dict[TypologyLayer, List[LayerConnection]]
    # All connections organized by layer

    dominant_layer: TypologyLayer
    # Layer with strongest/most connections

    total_connections: int
    # Sum across all layers

    composite_strength: float
    # Weighted aggregate score [0, 1]

    fractal_depth: int
    # How many layers have connections (1-7)

    self_similarity: SelfSimilarityAnalysis
    # Fractal mathematics analysis

    matched_patterns: List[TypePattern]
    # TypePatterns that this result instantiates

    covenant_context: Optional[CovenantArc]
    # If applicable, the covenant arc context

    patristic_composite_strength: float
    # Aggregate patristic attestation score [0, 1]

    patristic_witnesses: Dict[str, List[str]]
    # Father → Works that attest connections

    reasoning_chain: List[str]
    # Step-by-step typological reasoning

    theological_synthesis: str
    # Generated synthesis of findings

    confidence: float
    # Overall confidence [0, 1]

    confidence_interval: Tuple[float, float]
    # Bayesian credible interval

    processing_time_ms: float
    # Analysis duration

    cache_hit: bool = False
    # Whether from cache

    @property
    def is_strong_typology(self) -> bool:
        """Whether this meets threshold for strong typological connection."""
        return self.composite_strength >= 0.7 and self.fractal_depth >= 2

    def get_strongest_connection(self) -> Optional[LayerConnection]:
        """Return the single strongest connection across all layers."""
        best = None
        best_score = -1.0
        for layer_conns in self.layers.values():
            for conn in layer_conns:
                score = conn.compute_composite_score()
                if score > best_score:
                    best_score = score
                    best = conn
        return best

    def get_layer_summary(self) -> Dict[str, Any]:
        """Return summary statistics per layer."""
        summary = {}
        for layer in TypologyLayer:
            conns = self.layers.get(layer, [])
            if conns:
                scores = [c.compute_composite_score() for c in conns]
                summary[layer.name] = {
                    "count": len(conns),
                    "avg_score": sum(scores) / len(scores),
                    "max_score": max(scores),
                    "relations": list(set(c.relation.value for c in conns))
                }
        return summary

    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Convert to Neo4j relationship properties."""
        return {
            "composite_strength": self.composite_strength,
            "fractal_depth": self.fractal_depth,
            "hausdorff_dimension": self.self_similarity.hausdorff_dimension,
            "is_true_fractal": self.self_similarity.is_true_fractal,
            "dominant_layer": self.dominant_layer.name,
            "active_layers": [l.name for l in self.layers.keys() if self.layers[l]],
            "pattern_ids": [p.type_id for p in self.matched_patterns],
            "confidence": self.confidence,
            "patristic_strength": self.patristic_composite_strength
        }
```

---

## Part 3: Layer Analysis Algorithms (Exhaustive)

### Layer Weight and Configuration

```python
LAYER_CONFIG = {
    TypologyLayer.WORD: {
        "weight": 0.10,
        "min_strength": 0.6,
        "extraction": "lemma_based",
        "description": "Individual lexeme correspondences"
    },
    TypologyLayer.PHRASE: {
        "weight": 0.12,
        "min_strength": 0.5,
        "extraction": "syntactic_clause",
        "description": "Multi-word syntactic unit parallels"
    },
    TypologyLayer.VERSE: {
        "weight": 0.15,
        "min_strength": 0.4,
        "extraction": "verse_boundary",
        "description": "Complete verse-level correspondences"
    },
    TypologyLayer.PERICOPE: {
        "weight": 0.18,
        "min_strength": 0.4,
        "extraction": "discourse_markers",
        "description": "Narrative unit (3-30 verse) parallels"
    },
    TypologyLayer.CHAPTER: {
        "weight": 0.15,
        "min_strength": 0.35,
        "extraction": "chapter_boundary",
        "description": "Extended passage structural parallels"
    },
    TypologyLayer.BOOK: {
        "weight": 0.12,
        "min_strength": 0.3,
        "extraction": "book_structure",
        "description": "Whole book thematic/structural types"
    },
    TypologyLayer.COVENANTAL: {
        "weight": 0.18,
        "min_strength": 0.5,
        "extraction": "covenant_markers",
        "description": "Multi-book covenant arc fulfillments"
    }
}
```

### Algorithm 1: Word Layer Analysis

```python
async def analyze_word_layer(
    self,
    type_ref: str,
    antitype_ref: str
) -> List[LayerConnection]:
    """
    Find word-level typological correspondences.

    Detects:
    - Hebrew/Greek terms in TYPE_VOCABULARY database
    - Shared roots across testaments
    - LXX translation equivalences
    - Semantic field overlaps
    """
    connections = []

    # Get lemmatized content words with semantic metadata
    type_lemmas = await self.corpus.get_lemmas_with_semantics(type_ref)
    antitype_lemmas = await self.corpus.get_lemmas_with_semantics(antitype_ref)

    for type_lemma, type_sem in type_lemmas.items():
        # Check against known typological vocabulary
        if type_lemma in self.TYPE_VOCABULARY:
            pattern = self.TYPE_VOCABULARY[type_lemma]

            for antitype_lemma, anti_sem in antitype_lemmas.items():
                # Direct correspondence check
                if antitype_lemma in pattern.get('antitype_terms', []):
                    strength = self._calculate_word_correspondence(
                        type_sem, anti_sem, pattern
                    )
                    connections.append(LayerConnection(
                        connection_id=f"{type_ref}:{antitype_ref}:WORD:{len(connections)}",
                        source_reference=type_ref,
                        target_reference=antitype_ref,
                        source_text=type_lemma,
                        target_text=antitype_lemma,
                        source_lemmas=[type_lemma],
                        target_lemmas=[antitype_lemma],
                        layer=TypologyLayer.WORD,
                        relation=TypeAntitypeRelation.PREFIGURATION,
                        correspondence_type=CorrespondenceType.LEXICAL,
                        correspondence_strength=strength,
                        semantic_similarity=self._semantic_overlap(type_sem, anti_sem),
                        structural_similarity=0.0,
                        patristic_attestation=self._get_word_attestation(
                            type_lemma, antitype_lemma
                        ),
                        pattern_matches=[pattern['pattern_id']],
                        explanation=f"'{type_lemma}' → '{antitype_lemma}' ({pattern['pattern_name']})"
                    ))
                # Semantic correspondence (different word, same concept)
                elif self._semantic_overlap(type_sem, anti_sem) > 0.7:
                    connections.append(LayerConnection(
                        connection_id=f"{type_ref}:{antitype_ref}:WORD:{len(connections)}",
                        source_reference=type_ref,
                        target_reference=antitype_ref,
                        source_text=type_lemma,
                        target_text=antitype_lemma,
                        source_lemmas=[type_lemma],
                        target_lemmas=[antitype_lemma],
                        layer=TypologyLayer.WORD,
                        relation=TypeAntitypeRelation.PREFIGURATION,
                        correspondence_type=CorrespondenceType.SEMANTIC,
                        correspondence_strength=self._semantic_overlap(type_sem, anti_sem),
                        semantic_similarity=self._semantic_overlap(type_sem, anti_sem),
                        structural_similarity=0.0,
                        explanation=f"Semantic parallel: '{type_lemma}' ~ '{antitype_lemma}'"
                    ))

    return connections

def _calculate_word_correspondence(
    self, type_sem: dict, anti_sem: dict, pattern: dict
) -> float:
    """Calculate correspondence strength between type and antitype words."""
    base = 0.7  # Known pattern match baseline

    # Boost for semantic domain match
    if type_sem.get('domain') == anti_sem.get('domain'):
        base += 0.1

    # Boost for syntactic role match
    if type_sem.get('role') == anti_sem.get('role'):
        base += 0.1

    # Boost for multiple patristic attestation
    if len(pattern.get('patristic_witnesses', [])) >= 3:
        base += 0.1

    return min(1.0, base)
```

### Algorithm 2: Phrase Layer Analysis

```python
async def analyze_phrase_layer(
    self,
    type_ref: str,
    antitype_ref: str
) -> List[LayerConnection]:
    """
    Find phrase-level typological correspondences.

    Detects:
    - Formulaic expressions
    - Title phrases ("Son of Man", "Lamb of God")
    - Action descriptions
    - Theological phrases
    """
    connections = []

    # Extract syntactic phrases
    type_phrases = await self.corpus.get_syntactic_phrases(type_ref)
    antitype_phrases = await self.corpus.get_syntactic_phrases(antitype_ref)

    if not type_phrases or not antitype_phrases:
        return connections

    # Compute embeddings for all phrases
    type_embeddings = await self._compute_phrase_embeddings(type_phrases)
    antitype_embeddings = await self._compute_phrase_embeddings(antitype_phrases)

    # Compute similarity matrix
    sim_matrix = self._compute_similarity_matrix(type_embeddings, antitype_embeddings)

    threshold = self.config.min_phrase_similarity

    for i, type_phrase in enumerate(type_phrases):
        for j, antitype_phrase in enumerate(antitype_phrases):
            semantic_sim = sim_matrix[i, j]

            if semantic_sim < threshold:
                continue

            # Structural similarity
            structural_sim = self._calculate_structural_similarity(
                type_phrase, antitype_phrase
            )

            # Pattern matching
            pattern_matches = self._match_phrase_patterns(
                type_phrase, antitype_phrase
            )

            correspondence = 0.5 * semantic_sim + 0.3 * structural_sim + \
                           0.2 * (1.0 if pattern_matches else 0.0)

            if correspondence >= LAYER_CONFIG[TypologyLayer.PHRASE]["min_strength"]:
                connections.append(LayerConnection(
                    connection_id=f"{type_ref}:{antitype_ref}:PHRASE:{len(connections)}",
                    source_reference=type_ref,
                    target_reference=antitype_ref,
                    source_text=type_phrase['text'],
                    target_text=antitype_phrase['text'],
                    source_lemmas=type_phrase.get('lemmas', []),
                    target_lemmas=antitype_phrase.get('lemmas', []),
                    layer=TypologyLayer.PHRASE,
                    relation=self._infer_phrase_relation(type_phrase, antitype_phrase),
                    correspondence_type=CorrespondenceType.STRUCTURAL,
                    correspondence_strength=correspondence,
                    semantic_similarity=semantic_sim,
                    structural_similarity=structural_sim,
                    pattern_matches=[p.type_id for p in pattern_matches],
                    explanation=f"Phrase parallel: '{type_phrase['text']}' ≈ '{antitype_phrase['text']}'"
                ))

    return connections
```

### Algorithm 3: Pericope Layer Analysis

```python
async def analyze_pericope_layer(
    self,
    type_pericope: str,
    antitype_pericope: str
) -> List[LayerConnection]:
    """
    Find narrative/pericope-level typological correspondences.

    Detects:
    - Narrative element parallels (actors, actions, outcomes)
    - Scene structure parallels
    - Discourse type matches
    """
    connections = []

    # Extract narrative elements
    type_elements = await self.extract_narrative_elements(type_pericope)
    antitype_elements = await self.extract_narrative_elements(antitype_pericope)

    # Element categories with weights
    element_weights = {
        'actors': 0.25,
        'actions': 0.30,
        'objects': 0.15,
        'outcomes': 0.20,
        'locations': 0.10
    }

    total_similarity = 0.0
    similarities = {}

    for category, weight in element_weights.items():
        type_set = set(type_elements.get(category, []))
        antitype_set = set(antitype_elements.get(category, []))

        if type_set and antitype_set:
            # Semantic set overlap
            similarity = await self._semantic_set_overlap(type_set, antitype_set)
            similarities[category] = similarity
            total_similarity += similarity * weight

    if total_similarity < LAYER_CONFIG[TypologyLayer.PERICOPE]["min_strength"]:
        return connections

    # Infer relation type
    relation = self._infer_pericope_relation(type_elements, antitype_elements)

    # Check for structural sequence parallels
    structural_sim = 0.0
    if self._detect_sequence_parallel(type_elements, antitype_elements):
        structural_sim = 0.8

    connections.append(LayerConnection(
        connection_id=f"{type_pericope}:{antitype_pericope}:PERICOPE:0",
        source_reference=type_pericope,
        target_reference=antitype_pericope,
        source_text=f"Pericope: {type_pericope}",
        target_text=f"Pericope: {antitype_pericope}",
        source_lemmas=[],
        target_lemmas=[],
        layer=TypologyLayer.PERICOPE,
        relation=relation,
        correspondence_type=CorrespondenceType.STRUCTURAL,
        correspondence_strength=total_similarity,
        semantic_similarity=total_similarity,
        structural_similarity=structural_sim,
        explanation=f"Narrative parallel: {similarities}"
    ))

    return connections

async def extract_narrative_elements(self, pericope_ref: str) -> Dict[str, List[str]]:
    """Extract narrative elements using discourse analysis."""
    verses = await self.corpus.get_verses_in_range(pericope_ref)

    elements = {
        'actors': [],
        'actions': [],
        'objects': [],
        'outcomes': [],
        'locations': []
    }

    for verse in verses:
        syntax = await self.corpus.get_syntax_tree(verse)

        elements['actors'].extend(
            self._extract_semantic_role(syntax, ['SUBJ', 'AGENT'])
        )
        elements['actions'].extend(
            self._extract_lemmas_by_pos(syntax, ['VERB'])
        )
        elements['objects'].extend(
            self._extract_semantic_role(syntax, ['OBJ', 'THEME'])
        )
        elements['locations'].extend(
            self._extract_semantic_role(syntax, ['LOC'])
        )

    if verses:
        elements['outcomes'] = self._extract_outcome_markers(verses[-1])

    return elements
```

### Algorithm 4: Covenantal Layer Analysis

```python
async def analyze_covenantal_layer(
    self,
    type_ref: str,
    antitype_ref: str
) -> List[LayerConnection]:
    """
    Find covenant-arc typological correspondences.

    Detects:
    - Promise-fulfillment across covenants
    - Covenant sign correspondences
    - Covenant mediator typology
    """
    connections = []

    # Determine covenant contexts
    type_covenant = await self.trace_covenant_arc(type_ref)
    antitype_covenant = await self.trace_covenant_arc(antitype_ref)

    if not type_covenant or not antitype_covenant:
        return connections

    # Covenant progression order
    covenant_order = ['adamic', 'noahic', 'abrahamic', 'mosaic', 'davidic', 'new']

    try:
        type_idx = covenant_order.index(type_covenant.covenant_id)
        anti_idx = covenant_order.index(antitype_covenant.covenant_id)
    except ValueError:
        return connections

    if anti_idx <= type_idx:
        return connections  # Antitype should be later covenant

    # Check for promise overlap
    shared_promises = set(type_covenant.key_promises) & set(antitype_covenant.key_promises)

    for promise in shared_promises:
        # Measure intensification
        type_intensity = await self._measure_promise_intensity(promise, type_ref, type_covenant)
        antitype_intensity = await self._measure_promise_intensity(promise, antitype_ref, antitype_covenant)

        relation = (
            TypeAntitypeRelation.INTENSIFICATION
            if antitype_intensity > type_intensity * 1.2
            else TypeAntitypeRelation.FULFILLMENT
        )

        connections.append(LayerConnection(
            connection_id=f"{type_ref}:{antitype_ref}:COVENANTAL:{len(connections)}",
            source_reference=type_ref,
            target_reference=antitype_ref,
            source_text=f"{type_covenant.covenant_name}: {promise}",
            target_text=f"{antitype_covenant.covenant_name}: {promise}",
            source_lemmas=[],
            target_lemmas=[],
            layer=TypologyLayer.COVENANTAL,
            relation=relation,
            correspondence_type=CorrespondenceType.FUNCTIONAL,
            correspondence_strength=0.85,
            semantic_similarity=0.9,
            structural_similarity=0.8,
            explanation=f"Covenant promise '{promise}' {relation.value}"
        ))

    return connections

async def trace_covenant_arc(self, verse_ref: str) -> Optional[CovenantArc]:
    """Determine which covenant arc a verse belongs to."""
    # Check cache
    cache_key = f"covenant_arc:{verse_ref}"
    cached = await self.cache.get(cache_key)
    if cached:
        return cached

    # Check direct membership
    for covenant in self.COVENANT_ARCS.values():
        if covenant.contains_reference(verse_ref):
            await self.cache.set(cache_key, covenant, ttl=86400)
            return covenant

    # Book-level association
    book = self._parse_book(verse_ref)
    book_covenant_map = {
        'GEN': 'abrahamic',
        'EXO': 'mosaic',
        'LEV': 'mosaic',
        '2SA': 'davidic',
        'PSA': 'davidic',
        'ISA': 'davidic',
        'JER': 'new',
        'MAT': 'new',
        'JHN': 'new',
        'HEB': 'new',
        'GAL': 'abrahamic'
    }

    if book in book_covenant_map:
        covenant = self.COVENANT_ARCS.get(book_covenant_map[book])
        if covenant:
            await self.cache.set(cache_key, covenant, ttl=86400)
            return covenant

    return None
```

### Algorithm 5: Composite Strength Calculation

```python
def calculate_composite_strength(
    self,
    layers: Dict[TypologyLayer, List[LayerConnection]]
) -> Tuple[float, TypologyLayer]:
    """Calculate weighted composite strength and identify dominant layer."""

    if not any(layers.values()):
        return 0.0, TypologyLayer.WORD

    layer_scores = {}
    weighted_sum = 0.0
    total_weight = 0.0

    for layer, connections in layers.items():
        if connections:
            # Use enriched strength
            strengths = [c.enriched_strength() for c in connections]
            avg_strength = np.mean(strengths)
            max_strength = np.max(strengths)

            # Combine average and max
            layer_score = 0.7 * avg_strength + 0.3 * max_strength
            layer_scores[layer] = layer_score

            weight = LAYER_CONFIG[layer]["weight"]
            weighted_sum += layer_score * weight
            total_weight += weight

    if total_weight == 0:
        return 0.0, TypologyLayer.WORD

    base_score = weighted_sum / total_weight

    # Fractal depth bonus
    active_layers = len(layer_scores)
    depth_bonus = min(self.config.max_depth_bonus, active_layers * self.config.depth_bonus_per_layer)

    # Patristic attestation bonus
    total_patristic = sum(
        len(c.patristic_attestation)
        for conns in layers.values()
        for c in conns
    )
    patristic_bonus = min(self.config.max_patristic_bonus, total_patristic * self.config.patristic_bonus_per_witness)

    composite = min(1.0, base_score + depth_bonus + patristic_bonus)
    dominant = max(layer_scores.keys(), key=lambda k: layer_scores[k])

    return composite, dominant
```

---

## Part 4: Main Engine Class

```python
class HyperFractalTypologyEngine:
    """
    Multi-layer fractal typological analysis engine.

    The fourth of the Five Impossible Oracles. Discovers type-antitype
    connections at seven fractal layers, computing self-similarity
    metrics to identify "true fractal" typologies.
    """

    def __init__(
        self,
        corpus_client,
        mutual_metric: MutualTransformationMetric,      # Session 01
        necessity_calc: InterVerseNecessityCalculator,  # Session 04
        neo4j_client: Neo4jClient,
        redis_client: RedisClient,
        config: Optional['FractalTypologyConfig'] = None
    ):
        self.corpus = corpus_client
        self.mutual_metric = mutual_metric
        self.necessity_calc = necessity_calc
        self.neo4j = neo4j_client
        self.cache = redis_client
        self.config = config or FractalTypologyConfig()

        # Load type patterns and covenant arcs
        self.type_patterns = self._load_type_patterns()
        self.COVENANT_ARCS = self._load_covenant_arcs()
        self.TYPE_VOCABULARY = self._build_type_vocabulary()

        logger.info(
            f"HyperFractalTypologyEngine initialized with "
            f"{len(self.type_patterns)} patterns, "
            f"{len(self.COVENANT_ARCS)} covenant arcs"
        )

    async def analyze_fractal_typology(
        self,
        type_ref: str,
        antitype_ref: str
    ) -> FractalTypologyResult:
        """
        Main entry point: complete multi-layer typological analysis.

        Analyzes connections at all 7 layers, enriches with Session 01
        and Session 04 data, computes fractal self-similarity metrics.
        """
        import time
        start_time = time.time()

        # Check cache
        cache_key = f"fractal:{type_ref}:{antitype_ref}"
        cached = await self.cache.get(cache_key)
        if cached:
            cached.cache_hit = True
            return cached

        layers: Dict[TypologyLayer, List[LayerConnection]] = {}

        # Analyze each layer
        layers[TypologyLayer.WORD] = await self.analyze_word_layer(type_ref, antitype_ref)
        layers[TypologyLayer.PHRASE] = await self.analyze_phrase_layer(type_ref, antitype_ref)
        layers[TypologyLayer.VERSE] = await self.analyze_verse_layer(type_ref, antitype_ref)

        # Expand to larger units
        type_pericope = await self._expand_to_pericope(type_ref)
        antitype_pericope = await self._expand_to_pericope(antitype_ref)

        layers[TypologyLayer.PERICOPE] = await self.analyze_pericope_layer(
            type_pericope, antitype_pericope
        )
        layers[TypologyLayer.CHAPTER] = await self.analyze_chapter_layer(type_ref, antitype_ref)
        layers[TypologyLayer.BOOK] = await self.analyze_book_layer(type_ref, antitype_ref)
        layers[TypologyLayer.COVENANTAL] = await self.analyze_covenantal_layer(type_ref, antitype_ref)

        # Enrich with Session 01 and Session 04 data
        if self.config.enable_mutual_transformation:
            layers = await self._enrich_with_transformation(layers)
        if self.config.enable_necessity_enrichment:
            layers = await self._enrich_with_necessity(layers)

        # Calculate composite scores
        composite, dominant = self.calculate_composite_strength(layers)

        # Compute self-similarity metrics
        layer_counts = {layer: len(conns) for layer, conns in layers.items()}
        self_similarity = SelfSimilarityAnalysis.compute("analysis", layer_counts)

        # Match patterns
        matched_patterns = self._identify_matched_patterns(layers)

        # Covenant context
        covenant_ctx = await self.trace_covenant_arc(type_ref)

        # Build reasoning chain
        reasoning = self._build_reasoning_chain(layers, dominant)

        # Patristic strength
        patristic_strength = self._calculate_patristic_strength(layers)

        processing_time = (time.time() - start_time) * 1000

        result = FractalTypologyResult(
            result_id=f"{type_ref}:{antitype_ref}:{int(time.time())}",
            type_reference=type_ref,
            antitype_reference=antitype_ref,
            layers=layers,
            dominant_layer=dominant,
            total_connections=sum(len(c) for c in layers.values()),
            composite_strength=composite,
            fractal_depth=sum(1 for c in layers.values() if c),
            self_similarity=self_similarity,
            matched_patterns=matched_patterns,
            covenant_context=covenant_ctx,
            patristic_composite_strength=patristic_strength,
            patristic_witnesses=self._aggregate_patristic_witnesses(layers),
            reasoning_chain=reasoning,
            theological_synthesis=self._generate_synthesis(layers, matched_patterns),
            confidence=self._calculate_confidence(layers, composite),
            confidence_interval=self._calculate_confidence_interval(layers),
            processing_time_ms=processing_time
        )

        # Cache result
        await self.cache.set(cache_key, result, ttl=self.config.cache_ttl_seconds)

        # Store in Neo4j
        await self._store_in_neo4j(result)

        return result

    async def discover_fractal_patterns(
        self,
        type_ref: str,
        top_k: int = 10
    ) -> List[FractalTypologyResult]:
        """Given a type, discover ALL potential antitypes ranked by strength."""
        candidates = []

        # Strategy 1: Known pattern lookup
        for pattern in self.type_patterns.values():
            if type_ref in pattern.get_all_instances():
                for antitype in [pattern.canonical_antitype] + pattern.secondary_antitypes:
                    if self._is_nt_reference(antitype):
                        result = await self.analyze_fractal_typology(type_ref, antitype)
                        candidates.append(result)

        # Strategy 2: Semantic embedding search
        type_embedding = await self.mutual_metric.get_verse_embedding(type_ref)
        nt_matches = await self.vector_store.similarity_search(
            type_embedding,
            filter={"testament": "NT"},
            top_k=top_k * 2
        )

        for match in nt_matches:
            result = await self.analyze_fractal_typology(type_ref, match.verse_id)
            if result.composite_strength >= self.config.min_composite_strength:
                candidates.append(result)

        # Deduplicate and rank
        seen = set()
        unique_results = []
        for r in sorted(candidates, key=lambda x: x.composite_strength, reverse=True):
            if r.antitype_reference not in seen:
                seen.add(r.antitype_reference)
                unique_results.append(r)

        return unique_results[:top_k]
```

---

## Part 5: Integration Points

### Integration with Session 01: Mutual Transformation

```python
async def _enrich_with_transformation(
    self,
    layers: Dict[TypologyLayer, List[LayerConnection]]
) -> Dict[TypologyLayer, List[LayerConnection]]:
    """Add mutual transformation scores from Session 01."""
    for layer, connections in layers.items():
        for conn in connections:
            try:
                mt_result = await self.mutual_metric.measure_transformation(
                    source_verse=conn.source_reference,
                    target_verse=conn.target_reference
                )
                conn.mutual_transformation = mt_result.mutual_influence
            except Exception as e:
                logger.warning(f"MT enrichment failed: {e}")
                conn.mutual_transformation = None
    return layers
```

### Integration with Session 04: Necessity Calculator

```python
async def _enrich_with_necessity(
    self,
    layers: Dict[TypologyLayer, List[LayerConnection]]
) -> Dict[TypologyLayer, List[LayerConnection]]:
    """Add necessity scores from Session 04."""
    for layer, connections in layers.items():
        for conn in connections:
            try:
                necessity = await self.necessity_calc.calculate_necessity(
                    verse_a=conn.target_reference,  # NT needs OT
                    verse_b=conn.source_reference
                )
                conn.necessity_score = necessity.necessity_score
            except Exception as e:
                logger.warning(f"Necessity enrichment failed: {e}")
                conn.necessity_score = None
    return layers
```

### Neo4j Graph Schema

```cypher
// TYPIFIES relationship with fractal data
CREATE (type:Verse)-[:TYPIFIES {
    layers: ["WORD", "PHRASE", "PERICOPE"],
    layer_strengths: {WORD: 0.85, PHRASE: 0.72, PERICOPE: 0.91},
    composite_strength: 0.87,
    dominant_layer: "PERICOPE",
    fractal_depth: 3,
    hausdorff_dimension: 1.45,
    is_true_fractal: true,
    pattern_name: "Sacrificial Lamb",
    relation_type: "PREFIGURATION",
    covenant_context: "Abrahamic",
    patristic_strength: 0.8
}]->(antitype:Verse)

// Query for strong fractal typologies
MATCH (t:Verse)-[r:TYPIFIES]->(a:Verse)
WHERE r.is_true_fractal = true AND r.composite_strength > 0.8
RETURN t.id, a.id, r.layers, r.hausdorff_dimension
ORDER BY r.composite_strength DESC
```

---

## Part 6: Type Pattern Catalog

### `data/type_patterns.json`

```json
{
  "sacrificial_lamb": {
    "type_id": "sacrificial_lamb",
    "pattern_name": "Sacrificial Lamb",
    "description": "Innocent lamb sacrificed for others' sins",
    "primary_layer": "WORD",
    "active_layers": ["WORD", "PHRASE", "PERICOPE"],
    "hebrew_keywords": ["שֶׂה", "כֶּבֶשׂ", "תָּמִים"],
    "greek_keywords": ["ἀμνός", "ἀρνίον", "πρόβατον"],
    "semantic_markers": ["sacrifice", "substitution", "innocent", "blood"],
    "canonical_type": "GEN.22.8",
    "canonical_antitype": "JHN.1.29",
    "secondary_types": ["EXO.12.3", "ISA.53.7"],
    "secondary_antitypes": ["1PE.1.19", "REV.5.6"],
    "correspondence_points": {
      "innocent": "sinless",
      "unblemished": "perfect",
      "substitutionary": "vicarious"
    },
    "patristic_attestation": {
      "Chrysostom": ["Homilies on Genesis 47"],
      "Cyril of Alexandria": ["Commentary on John"],
      "Melito of Sardis": ["On Pascha"]
    }
  },
  "adam_christ": {
    "type_id": "adam_christ",
    "pattern_name": "Adam-Christ (First/Last Adam)",
    "description": "Recapitulation and inversion of Adam in Christ",
    "primary_layer": "COVENANTAL",
    "active_layers": ["VERSE", "CHAPTER", "COVENANTAL"],
    "relation_type": "INVERSION",
    "hebrew_keywords": ["אָדָם"],
    "greek_keywords": ["Ἀδάμ", "ἄνθρωπος"],
    "semantic_markers": ["humanity", "fall", "obedience", "death", "life"],
    "canonical_type": "GEN.3.6",
    "canonical_antitype": "ROM.5.19",
    "secondary_types": ["GEN.2.7"],
    "secondary_antitypes": ["1CO.15.22", "1CO.15.45"],
    "correspondence_points": {
      "from_dust": "from_heaven",
      "disobedience": "obedience",
      "death_entered": "life_given",
      "curse": "blessing"
    },
    "patristic_attestation": {
      "Irenaeus": ["Against Heresies 3.22"],
      "Athanasius": ["On the Incarnation"],
      "Gregory of Nyssa": ["Catechetical Oration"]
    }
  }
}
```

### Covenant Arc Definitions

```json
{
  "covenant_arcs": {
    "abrahamic": {
      "covenant_id": "abrahamic",
      "covenant_name": "Abrahamic Covenant",
      "description": "Covenant of promise with Abraham and his seed",
      "initiation_reference": "GEN.12.1-3",
      "key_promises": ["land", "seed", "blessing", "nation"],
      "covenant_signs": [["circumcision", "GEN.17.11"]],
      "arc_span": ["GEN", "GAL"],
      "fulfillment_references": ["GAL.3.8", "GAL.3.14", "GAL.3.29"],
      "superseded_by": "new"
    },
    "new": {
      "covenant_id": "new",
      "covenant_name": "New Covenant",
      "description": "Covenant in Christ's blood fulfilling all prior covenants",
      "initiation_reference": "JER.31.31-34",
      "key_promises": ["internal_law", "forgiveness", "knowledge_of_God"],
      "covenant_signs": [["baptism", "MAT.28.19"], ["eucharist", "LUK.22.20"]],
      "arc_span": ["JER", "REV"],
      "fulfillment_references": ["HEB.8.8-12", "LUK.22.20"],
      "supersedes": "mosaic"
    }
  }
}
```

---

## Part 7: Testing Specification

### `tests/ml/engines/test_fractal_typology.py`

```python
import pytest
from ml.engines.fractal_typology import (
    HyperFractalTypologyEngine,
    TypologyLayer,
    TypeAntitypeRelation,
    SelfSimilarityAnalysis
)

class TestFractalTypology:

    @pytest.mark.asyncio
    async def test_isaac_christ_all_layers(self, engine):
        """Canonical test: Isaac/Christ should show multi-layer typology."""
        result = await engine.analyze_fractal_typology("GEN.22.1-19", "JHN.3.16")

        assert result.fractal_depth >= 4
        assert result.composite_strength > 0.8
        assert TypologyLayer.WORD in result.layers
        assert TypologyLayer.PERICOPE in result.layers
        assert result.self_similarity.is_true_fractal

    @pytest.mark.asyncio
    async def test_passover_lamb_word_layer(self, engine):
        """Passover lamb should have strong WORD layer connection."""
        result = await engine.analyze_fractal_typology("EXO.12.3", "1CO.5.7")

        word_conns = result.layers.get(TypologyLayer.WORD, [])
        assert len(word_conns) > 0
        assert any(c.correspondence_strength > 0.8 for c in word_conns)

    @pytest.mark.asyncio
    async def test_adam_christ_inversion(self, engine):
        """Adam/Christ should detect INVERSION relation."""
        result = await engine.analyze_fractal_typology("GEN.3.6", "ROM.5.19")

        all_relations = [
            c.relation for conns in result.layers.values() for c in conns
        ]
        assert TypeAntitypeRelation.INVERSION in all_relations

    @pytest.mark.asyncio
    async def test_covenant_arc_tracing(self, engine):
        """Abrahamic promise should trace to Galatians fulfillment."""
        result = await engine.analyze_fractal_typology("GEN.12.3", "GAL.3.8")

        assert result.covenant_context is not None
        assert result.covenant_context.covenant_id == "abrahamic"
        cov_conns = result.layers.get(TypologyLayer.COVENANTAL, [])
        assert len(cov_conns) > 0

    @pytest.mark.asyncio
    async def test_self_similarity_computation(self):
        """Test fractal dimension calculation."""
        layer_counts = {
            TypologyLayer.WORD: 3,
            TypologyLayer.PHRASE: 2,
            TypologyLayer.VERSE: 3,
            TypologyLayer.PERICOPE: 2,
            TypologyLayer.COVENANTAL: 1
        }

        analysis = SelfSimilarityAnalysis.compute("test", layer_counts)

        assert analysis.hausdorff_dimension > 1.0
        assert analysis.fractal_depth == 5
        assert analysis.is_true_fractal

    @pytest.mark.asyncio
    async def test_discover_patterns(self, engine):
        """Pattern discovery should find NT antitypes."""
        results = await engine.discover_fractal_patterns("GEN.22.2", top_k=5)

        assert len(results) >= 1
        # Passion narrative should be among top results
        refs = [r.antitype_reference for r in results]
        assert any("JHN" in ref or "HEB" in ref for ref in refs)
```

---

## Part 8: Configuration

### `config.py` Addition

```python
@dataclass
class FractalTypologyConfig:
    """Configuration for HyperFractalTypologyEngine."""

    # Strength thresholds
    min_layer_strength: float = 0.3
    min_composite_strength: float = 0.5
    min_phrase_similarity: float = 0.6

    # Layer weights
    weight_word_layer: float = 0.10
    weight_phrase_layer: float = 0.12
    weight_verse_layer: float = 0.15
    weight_pericope_layer: float = 0.18
    weight_chapter_layer: float = 0.15
    weight_book_layer: float = 0.12
    weight_covenantal_layer: float = 0.18

    # Bonuses
    depth_bonus_per_layer: float = 0.025
    max_depth_bonus: float = 0.15
    patristic_bonus_per_witness: float = 0.01
    max_patristic_bonus: float = 0.10

    # Integration toggles
    enable_mutual_transformation: bool = True
    enable_necessity_enrichment: bool = True

    # Caching
    cache_type_patterns: bool = True
    cache_ttl_seconds: int = 86400
```

---

## Part 9: Success Criteria

### Functional Requirements
- [ ] All 7 layer analysis methods implemented
- [ ] Self-similarity (fractal dimension) correctly computed
- [ ] Relation types detected (prefiguration, fulfillment, inversion, etc.)
- [ ] Session 01 mutual transformation integrated
- [ ] Session 04 necessity calculator integrated
- [ ] Composite strength with depth/patristic bonuses
- [ ] Covenant arc tracing functional

### Theological Accuracy
- [ ] Isaac/Christ: Multi-layer, high strength, true fractal
- [ ] Passover/Eucharist: WORD and PERICOPE strong
- [ ] Adam/Christ: INVERSION detected
- [ ] Davidic covenant: COVENANTAL connections to Christ

### Performance
- [ ] Cached analysis: < 100ms
- [ ] Fresh analysis: < 5 seconds
- [ ] Pattern discovery: < 30 seconds

---

## Part 10: Dependencies

### Depends On
- SESSION 01: Mutual Transformation Metric
- SESSION 03: Omni-Contextual Resolver
- SESSION 04: Inter-Verse Necessity Calculator

### Depended On By
- SESSION 07: Prophetic Necessity Prover
- SESSION 11: Pipeline Integration

---

## Session Completion Checklist

```markdown
- [ ] `ml/engines/fractal_typology.py` implemented
- [ ] All 7 layer analysis methods working
- [ ] `SelfSimilarityAnalysis` computing fractal dimension
- [ ] `data/type_patterns.json` created
- [ ] Covenant arc structures defined
- [ ] Composite strength calculation correct
- [ ] Relation type detection working
- [ ] Integration with Mutual Transformation Metric
- [ ] Integration with Necessity Calculator
- [ ] Neo4j TYPIFIES relationship schema
- [ ] Configuration added to config.py
- [ ] Unit tests passing
- [ ] Isaac/Christ canonical test passing (true fractal)
```

**Next Session**: SESSION 07: Prophetic Necessity Prover
