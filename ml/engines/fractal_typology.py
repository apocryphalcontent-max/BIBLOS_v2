"""
BIBLOS v2 - Hyper-Fractal Typology Engine

The Fourth Impossible Oracle: Discovers typological connections at multiple
fractal layers - from individual words to covenantal arcs spanning testaments.

This engine operationalizes the patristic understanding that Scripture exhibits
self-similar prophetic patterns at every scale of textual analysis.

Core Capability:
- Analyzes type-antitype connections across 7 fractal layers
- Computes self-similarity metrics (Hausdorff dimension)
- Integrates mutual transformation and necessity scores
- Provides comprehensive theological reasoning chains

Fractal Layers:
1. WORD - Individual lexeme correspondences
2. PHRASE - Multi-word syntactic unit correspondences
3. VERSE - Complete verse-level parallels
4. PERICOPE - Narrative unit types (3-30 verses)
5. CHAPTER - Extended passage structural parallels
6. BOOK - Entire book thematic/structural types
7. COVENANTAL - Multi-book covenant arc fulfillments
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================


class TypologyLayer(IntEnum):
    """
    Seven fractal layers of typological analysis.

    IntEnum allows mathematical operations on layer values,
    enabling fractal dimension calculations and layer comparisons.

    Layer numbering follows textual granularity from finest (1)
    to coarsest (7), matching standard fractal analysis conventions.
    """

    WORD = 1  # Individual lexeme correspondences
    PHRASE = 2  # Multi-word syntactic unit correspondences
    VERSE = 3  # Complete verse-level parallels
    PERICOPE = 4  # Narrative unit types (3-30 verses)
    CHAPTER = 5  # Extended passage structural parallels
    BOOK = 6  # Entire book thematic/structural types
    COVENANTAL = 7  # Multi-book covenant arc fulfillments

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
            TypologyLayer.COVENANTAL: (5000, 500000),
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
            TypologyLayer.COVENANTAL: "covenant_arc",
        }[self]

    @property
    def analysis_complexity(self) -> str:
        """Complexity class for this layer's analysis."""
        if self <= TypologyLayer.VERSE:
            return "O(n)"  # Linear in vocabulary size
        elif self <= TypologyLayer.CHAPTER:
            return "O(n²)"  # Pairwise comparison of units
        else:
            return "O(n² log n)"  # Graph traversal with sorting


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
        elif self in (
            TypeAntitypeRelation.PREFIGURATION,
            TypeAntitypeRelation.FULFILLMENT,
        ):
            return "forward"
        return "bidirectional"

    @classmethod
    def from_patristic_term(cls, term: str) -> "TypeAntitypeRelation":
        """Map patristic Greek terminology to relation type."""
        mapping = {
            "τύπος": cls.PREFIGURATION,
            "σκιά": cls.PREFIGURATION,  # "shadow"
            "ἀντίτυπος": cls.FULFILLMENT,
            "ἀνακεφαλαίωσις": cls.RECAPITULATION,
            "κρείττων": cls.INTENSIFICATION,  # "better"
            "ἀντίθεσις": cls.INVERSION,
            "μέθεξις": cls.PARTICIPATION,
            "αὔξησις": cls.ESCALATION,
        }
        return mapping.get(term, cls.PREFIGURATION)


class CorrespondenceType(Enum):
    """How type and antitype correspond structurally."""

    LEXICAL = "lexical"  # Same/related vocabulary (שֶׂה → ἀμνός)
    SEMANTIC = "semantic"  # Same meaning, different words
    STRUCTURAL = "structural"  # Parallel narrative structure
    FUNCTIONAL = "functional"  # Same theological role
    SYMBOLIC = "symbolic"  # Symbolic equivalence (blood, water, fire)
    NUMERICAL = "numerical"  # Numerical patterns (3 days, 12, 40, 7)


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

    INITIATION = "initiation"  # Covenant establishment
    STIPULATION = "stipulation"  # Terms and conditions
    PROMISE = "promise"  # Blessings for obedience
    WARNING = "warning"  # Curses for disobedience
    SIGN = "sign"  # Covenant sign/seal
    RENEWAL = "renewal"  # Covenant renewal ceremony
    FULFILLMENT = "fulfillment"  # Covenant promises realized
    SUPERSESSION = "supersession"  # Covenant completed in new covenant


# =============================================================================
# DATACLASSES
# =============================================================================


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
        self, text_lemmas: Set[str], semantic_concepts: Set[str]
    ) -> float:
        """
        Calculate how strongly this pattern activates for given text.
        Uses fuzzy set intersection with weighted terms.
        """
        keyword_set = set(self.hebrew_keywords + self.greek_keywords)
        keyword_overlap = (
            len(text_lemmas & keyword_set) / max(len(keyword_set), 1)
            if keyword_set
            else 0.0
        )

        marker_set = set(self.semantic_markers)
        semantic_overlap = (
            len(semantic_concepts & marker_set) / max(len(marker_set), 1)
            if marker_set
            else 0.0
        )

        # Semantic markers weighted slightly higher
        return 0.4 * keyword_overlap + 0.6 * semantic_overlap

    def get_all_instances(self) -> List[str]:
        """Return all type and antitype instances."""
        return (
            [self.canonical_type]
            + self.secondary_types
            + [self.canonical_antitype]
            + self.secondary_antitypes
        )


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
            "patristic": 0.10,
        }

        patristic_score = min(1.0, len(self.patristic_attestation) * 0.25)

        score = (
            weights["correspondence"] * self.correspondence_strength
            + weights["semantic"] * self.semantic_similarity
            + weights["structural"] * self.structural_similarity
            + weights["transformation"] * (self.mutual_transformation or 0)
            + weights["necessity"] * (self.necessity_score or 0)
            + weights["patristic"] * patristic_score
        )

        return min(1.0, score)


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
        cls, pattern_id: str, layer_counts: Dict[TypologyLayer, int]
    ) -> "SelfSimilarityAnalysis":
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
                is_true_fractal=False,
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
            hausdorff = (
                (n * sum_xy - sum_x * sum_y) / denom if abs(denom) > 1e-10 else 1.0
            )
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
        lacunarity = variance / (mean_count**2) if mean_count > 0 else 1.0

        # True fractal determination
        is_fractal = (
            hausdorff > 1.2 and scale_invariance > 0.5 and len(active) >= 3
        )

        return cls(
            pattern_id=pattern_id,
            layer_connection_counts=layer_counts,
            hausdorff_dimension=hausdorff,
            box_counting_curve=box_curve,
            scale_invariance_coefficient=scale_invariance,
            dominant_scale=dominant,
            scale_distribution_entropy=normalized_entropy,
            lacunarity=lacunarity,
            is_true_fractal=is_fractal,
        )


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
                    "relations": list(set(c.relation.value for c in conns)),
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
            "active_layers": [
                l.name for l in self.layers.keys() if self.layers[l]
            ],
            "pattern_ids": [p.type_id for p in self.matched_patterns],
            "confidence": self.confidence,
            "patristic_strength": self.patristic_composite_strength,
        }

# =============================================================================
# LAYER CONFIGURATION
# =============================================================================

LAYER_CONFIG = {
    TypologyLayer.WORD: {
        "weight": 0.10,
        "min_strength": 0.6,
        "extraction": "lemma_based",
        "description": "Individual lexeme correspondences",
    },
    TypologyLayer.PHRASE: {
        "weight": 0.12,
        "min_strength": 0.5,
        "extraction": "syntactic_clause",
        "description": "Multi-word syntactic unit parallels",
    },
    TypologyLayer.VERSE: {
        "weight": 0.15,
        "min_strength": 0.4,
        "extraction": "verse_boundary",
        "description": "Complete verse-level correspondences",
    },
    TypologyLayer.PERICOPE: {
        "weight": 0.18,
        "min_strength": 0.4,
        "extraction": "discourse_markers",
        "description": "Narrative unit (3-30 verse) parallels",
    },
    TypologyLayer.CHAPTER: {
        "weight": 0.15,
        "min_strength": 0.35,
        "extraction": "chapter_boundary",
        "description": "Extended passage structural parallels",
    },
    TypologyLayer.BOOK: {
        "weight": 0.12,
        "min_strength": 0.3,
        "extraction": "book_structure",
        "description": "Whole book thematic/structural types",
    },
    TypologyLayer.COVENANTAL: {
        "weight": 0.18,
        "min_strength": 0.5,
        "extraction": "covenant_markers",
        "description": "Multi-book covenant arc fulfillments",
    },
}


# =============================================================================
# CONFIGURATION DATACLASS
# =============================================================================


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

    # Pattern and arc paths
    type_patterns_path: str = "data/type_patterns.json"
    covenant_arcs_path: str = "data/covenant_arcs.json"


# =============================================================================
# MAIN ENGINE CLASS
# =============================================================================


class HyperFractalTypologyEngine:
    """
    Multi-layer fractal typological analysis engine.

    The fourth of the Five Impossible Oracles. Discovers type-antitype
    connections at seven fractal layers, computing self-similarity
    metrics to identify "true fractal" typologies.

    Integrates with:
    - Session 01: Mutual Transformation Metric
    - Session 04: Inter-Verse Necessity Calculator
    - Corpus clients for text access
    - Neo4j for graph storage
    - Redis for caching

    Usage:
        async with HyperFractalTypologyEngine(corpus) as engine:
            result = await engine.analyze_typology("GEN.22.1", "HEB.11.17")
    """

    # New Testament book codes for reference classification
    NT_BOOKS: frozenset = frozenset({
        "MAT", "MRK", "LUK", "JHN", "ACT", "ROM", "1CO", "2CO", "GAL",
        "EPH", "PHP", "COL", "1TH", "2TH", "1TI", "2TI", "TIT", "PHM",
        "HEB", "JAS", "1PE", "2PE", "1JN", "2JN", "3JN", "JUD", "REV"
    })

    def __init__(
        self,
        corpus_client: Any,
        neo4j_client: Optional[Any] = None,
        redis_client: Optional[Any] = None,
        mutual_metric: Optional[Any] = None,
        necessity_calc: Optional[Any] = None,
        config: Optional[FractalTypologyConfig] = None,
    ):
        """
        Initialize the Hyper-Fractal Typology Engine.

        Args:
            corpus_client: Text corpus client (Text-Fabric, Macula, etc.)
            neo4j_client: Neo4j client for graph storage
            redis_client: Redis client for caching
            mutual_metric: MutualTransformationMetric from Session 01
            necessity_calc: InterVerseNecessityCalculator from Session 04
            config: Configuration object
        """
        self.corpus = corpus_client
        self.neo4j = neo4j_client
        self.cache = redis_client
        self.mutual_metric = mutual_metric
        self.necessity_calc = necessity_calc
        self.config = config or FractalTypologyConfig()

        # Load type patterns and covenant arcs
        self.type_patterns = self._load_type_patterns()
        self.covenant_arcs = self._load_covenant_arcs()
        self.type_vocabulary = self._build_type_vocabulary()

        # Import implementation methods
        from ml.engines.fractal_typology_impl import HyperFractalTypologyEngineImpl

        impl = HyperFractalTypologyEngineImpl()

        # Bind implementation methods to self
        for method_name in dir(impl):
            if not method_name.startswith('__'):
                method = getattr(impl, method_name)
                if callable(method):
                    setattr(self, method_name, method.__get__(self, type(self)))

        logger.info(
            f"HyperFractalTypologyEngine initialized with "
            f"{len(self.type_patterns)} patterns, "
            f"{len(self.covenant_arcs)} covenant arcs"
        )

    def _load_type_patterns(self) -> Dict[str, TypePattern]:
        """Load typological pattern catalog."""
        import json
        from pathlib import Path

        patterns = {}
        try:
            path = Path(self.config.type_patterns_path)
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for pid, pdata in data.items():
                        patterns[pid] = TypePattern(
                            type_id=pid,
                            pattern_name=pdata["pattern_name"],
                            description=pdata["description"],
                            primary_layer=TypologyLayer[pdata["primary_layer"]],
                            active_layers=[
                                TypologyLayer[l] for l in pdata["active_layers"]
                            ],
                            hebrew_keywords=pdata.get("hebrew_keywords", []),
                            greek_keywords=pdata.get("greek_keywords", []),
                            semantic_markers=pdata.get("semantic_markers", []),
                            canonical_type=pdata["canonical_type"],
                            canonical_antitype=pdata["canonical_antitype"],
                            secondary_types=pdata.get("secondary_types", []),
                            secondary_antitypes=pdata.get("secondary_antitypes", []),
                            relation_type=TypeAntitypeRelation[
                                pdata.get("relation_type", "PREFIGURATION")
                            ],
                            correspondence_points=pdata.get("correspondence_points", {}),
                            patristic_attestation=pdata.get("patristic_attestation", {}),
                            liturgical_usage=pdata.get("liturgical_usage", []),
                            inverse_pattern=pdata.get("inverse_pattern"),
                        )
                logger.info(f"Loaded {len(patterns)} type patterns")
        except Exception as e:
            logger.warning(f"Could not load type patterns: {e}")

        return patterns

    def _load_covenant_arcs(self) -> Dict[str, CovenantArc]:
        """Load covenant arc definitions."""
        import json
        from pathlib import Path

        arcs = {}
        try:
            path = Path(self.config.covenant_arcs_path)
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for cov_id, cov_data in data.get("covenant_arcs", {}).items():
                        arcs[cov_id] = CovenantArc(
                            covenant_id=cov_id,
                            covenant_name=cov_data["covenant_name"],
                            description=cov_data["description"],
                            initiation_reference=cov_data["initiation_reference"],
                            key_promises=cov_data["key_promises"],
                            covenant_signs=[tuple(s) for s in cov_data.get("covenant_signs", [])],
                            phases={
                                CovenantPhase[phase]: refs
                                for phase, refs in cov_data.get("phases", {}).items()
                            },
                            type_events=[tuple(e) for e in cov_data.get("type_events", [])],
                            fulfillment_references=cov_data.get("fulfillment_references", []),
                            arc_span=tuple(cov_data["arc_span"]),
                            superseded_by=cov_data.get("superseded_by"),
                            supersedes=cov_data.get("supersedes"),
                            participant_scope=cov_data.get("participant_scope", "individual"),
                            conditional=cov_data.get("conditional", True),
                        )
                logger.info(f"Loaded {len(arcs)} covenant arcs")
        except Exception as e:
            logger.warning(f"Could not load covenant arcs: {e}")

        return arcs

    def _build_type_vocabulary(self) -> Dict[str, Dict]:
        """Build type vocabulary index from patterns."""
        vocab = {}
        for pattern in self.type_patterns.values():
            for keyword in pattern.hebrew_keywords + pattern.greek_keywords:
                if keyword not in vocab:
                    vocab[keyword] = {
                        "pattern_id": pattern.type_id,
                        "pattern_name": pattern.pattern_name,
                        "antitype_terms": [],
                        "patristic_witnesses": [],
                    }
        return vocab

    def _is_nt_reference(self, ref: str) -> bool:
        """Check if reference is from New Testament.

        Uses class constant NT_BOOKS for O(1) lookup without object creation.
        """
        if not ref:
            return False
        parts = ref.split(".")
        return len(parts) > 0 and parts[0] in self.NT_BOOKS
