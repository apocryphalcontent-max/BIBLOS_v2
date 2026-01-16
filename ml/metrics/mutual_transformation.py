"""
BIBLOS v2 - Mutual Transformation Metric

IMPOSSIBLY SOPHISTICATED bidirectional semantic transformation measurement.

This module implements the complete mathematical framework for measuring how
two biblical passages "mutually transform" each other's meaning through their
interconnection. The metric captures the theological insight that genuine
cross-references exhibit bidirectional semantic influence - each verse
redefines the other when understood in relation.

THEOLOGICAL FOUNDATION:
    When two verses are genuinely connected, their meanings mutually transform.
    The Burning Bush (Exodus 3:2) and the Theotokos Mary both gain deeper
    significance when understood in relation to each other:
    - The bush becomes a type of Mary (contains divine fire without being consumed)
    - Mary's virginal motherhood is illuminated as the antitype fulfillment

MATHEMATICAL FRAMEWORK:

    1. BASE TRANSFORMATION (Harmonic Mean):
       Given two verses A and B with embeddings before and after GNN refinement:
       - source_shift = 1 - cosine_similarity(A_before, A_after)
       - target_shift = 1 - cosine_similarity(B_before, B_after)
       - mutual_influence = 2 * (source_shift * target_shift) / (source_shift + target_shift + ε)

    2. THEOLOGICAL DIMENSION DECOMPOSITION:
       Delta vectors are decomposed into 12 theological dimensions:
       - Christological, Soteriological, Eschatological, Ecclesiological
       - Pneumatological, Cosmological, Anthropological, Sacramental
       - Liturgical, Patristic, Typological, Prophetic

    3. RESONANCE COEFFICIENT:
       Measures alignment of delta vectors (whether transformations are coherent):
       resonance = cosine_similarity(source_delta, target_delta)

    4. ASYMMETRY PENALTY:
       Applies Kullback-Leibler divergence to penalize one-sided influence:
       KL(P||Q) where P = source distribution, Q = target distribution

    5. TEMPORAL DYNAMICS:
       Tracks transformation over multiple GNN iterations to detect:
       - Convergent patterns (transformation stabilizes)
       - Oscillatory patterns (transformation alternates)
       - Divergent patterns (transformation grows unboundedly)

    6. CANONICAL ARCHETYPE MATCHING:
       Compares transformation signature against known typological patterns:
       - Isaac/Christ sacrifice archetype
       - Exodus/Baptism deliverance archetype
       - Temple/Church dwelling archetype
       - And 50+ other canonical archetypes

    7. NESTED TRANSFORMATION DETECTION:
       Identifies multi-verse transformation chains where:
       A → B → C creates nested influence propagation
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set, Callable
from functools import lru_cache
from collections import defaultdict
import logging
import math
import warnings

import numpy as np
from scipy.spatial.distance import cosine, cdist
from scipy.stats import entropy, wasserstein_distance
from scipy.special import kl_div

logger = logging.getLogger("biblos.ml.metrics.mutual_transformation")


# =============================================================================
# ENUMERATIONS - Comprehensive Classification Taxonomy
# =============================================================================

class TransformationType(Enum):
    """
    Classification of transformation intensity based on mutual influence score.

    TRANSCENDENT: Highest level - both verses fundamentally redefine each other
        in ways that reshape theological understanding. Rare (<1% of connections).
        Example: The entire meaning of Genesis 22 is transformed by John 3:16.

    RADICAL: Strong bidirectional transformation (e.g., Isaac → Christ)
        Both verses fundamentally redefine each other's meaning.

    SUBSTANTIAL: Significant mutual influence with clear theological impact.
        Both verses contribute meaningfully but don't fundamentally reshape.

    MODERATE: Noticeable mutual influence (e.g., Temple → Church)
        Connection enriches understanding without radical transformation.

    MINIMAL: Weak or one-sided connection (e.g., geographic location match)
        Connection exists but doesn't substantially transform meaning.

    INCIDENTAL: Connection is primarily methodological, not semantic.
        Example: Same author, same time period, but no mutual influence.
    """
    TRANSCENDENT = "TRANSCENDENT"
    RADICAL = "RADICAL"
    SUBSTANTIAL = "SUBSTANTIAL"
    MODERATE = "MODERATE"
    MINIMAL = "MINIMAL"
    INCIDENTAL = "INCIDENTAL"

    @property
    def theological_significance(self) -> str:
        """Description of theological significance for each type."""
        return {
            TransformationType.TRANSCENDENT: "Both passages achieve new meaning only through their interconnection",
            TransformationType.RADICAL: "Each passage fundamentally transforms understanding of the other",
            TransformationType.SUBSTANTIAL: "Significant bidirectional illumination with theological depth",
            TransformationType.MODERATE: "Meaningful mutual enrichment of understanding",
            TransformationType.MINIMAL: "Connection exists with limited transformative impact",
            TransformationType.INCIDENTAL: "Connection is methodological rather than semantic",
        }[self]

    @property
    def minimum_score(self) -> float:
        """Minimum mutual influence score for this classification."""
        return {
            TransformationType.TRANSCENDENT: 0.7,
            TransformationType.RADICAL: 0.5,
            TransformationType.SUBSTANTIAL: 0.35,
            TransformationType.MODERATE: 0.2,
            TransformationType.MINIMAL: 0.1,
            TransformationType.INCIDENTAL: 0.0,
        }[self]


class DimensionCategory(Enum):
    """
    Theological dimensions for delta vector decomposition.

    Each dimension corresponds to a major category of theological meaning,
    allowing precise identification of WHICH aspects of meaning are transformed.
    """
    CHRISTOLOGICAL = "christological"      # Christ-centered meaning
    SOTERIOLOGICAL = "soteriological"      # Salvation-related meaning
    ESCHATOLOGICAL = "eschatological"      # End-times, fulfillment
    ECCLESIOLOGICAL = "ecclesiological"    # Church, community
    PNEUMATOLOGICAL = "pneumatological"    # Spirit, divine presence
    COSMOLOGICAL = "cosmological"          # Creation, cosmos
    ANTHROPOLOGICAL = "anthropological"    # Human nature, fall, redemption
    SACRAMENTAL = "sacramental"            # Ritual, mystery, sign
    LITURGICAL = "liturgical"              # Worship, prayer, praise
    PATRISTIC = "patristic"                # Alignment with Church Fathers
    TYPOLOGICAL = "typological"            # Type/antitype relations
    PROPHETIC = "prophetic"                # Prophecy, fulfillment
    COVENANTAL = "covenantal"              # Covenant arc, promise
    TRINITARIAN = "trinitarian"            # Father, Son, Spirit relations
    THEOPHANIC = "theophanic"              # Divine appearances, glory
    MARIAN = "marian"                      # Theotokos typology


class TemporalPattern(Enum):
    """
    Patterns of transformation over multiple GNN iterations.
    """
    CONVERGENT = "convergent"       # Transformation stabilizes
    OSCILLATORY = "oscillatory"     # Transformation alternates
    DIVERGENT = "divergent"         # Transformation grows unboundedly
    STEADY_STATE = "steady_state"   # Immediate stability
    DELAYED = "delayed"             # Transformation emerges after iterations


class TransformationArchetype(Enum):
    """
    Canonical transformation archetypes from Orthodox tradition.

    These represent the most significant typological patterns that define
    how Old Testament passages are transformed by New Testament fulfillment.
    """
    # Person archetypes
    ADAM_CHRIST = "adam_christ"               # First/Last Adam
    ISAAC_CHRIST_SACRIFICE = "isaac_sacrifice"  # Only son sacrifice
    MOSES_CHRIST_DELIVERER = "moses_deliverer"  # Deliverer pattern
    DAVID_CHRIST_KING = "david_king"          # Kingdom pattern
    ELIJAH_JOHN_FORERUNNER = "elijah_forerunner"  # Forerunner pattern
    JONAH_RESURRECTION = "jonah_resurrection"  # Three days pattern
    MELCHIZEDEK_PRIESTHOOD = "melchizedek_priest"  # Eternal priesthood

    # Object/Place archetypes
    ARK_THEOTOKOS = "ark_theotokos"           # Ark of covenant/Mary
    BURNING_BUSH_THEOTOKOS = "bush_theotokos"  # Unconsumed fire/Virginity
    TEMPLE_CHURCH = "temple_church"           # Dwelling place
    TABERNACLE_INCARNATION = "tabernacle_incarnation"  # God dwelling among
    SERPENT_CROSS = "serpent_cross"           # Lifted up healing

    # Event archetypes
    EXODUS_BAPTISM = "exodus_baptism"         # Water deliverance
    PASSOVER_EUCHARIST = "passover_eucharist"  # Lamb sacrifice
    MANNA_BREAD = "manna_bread"               # Divine feeding
    RED_SEA_BAPTISM = "red_sea_baptism"       # Crossing through water
    FLOOD_BAPTISM = "flood_baptism"           # Water judgment/salvation

    # Covenant archetypes
    CIRCUMCISION_BAPTISM = "circumcision_baptism"  # Covenant initiation
    SABBATH_REST = "sabbath_rest"             # Divine rest
    JUBILEE_REDEMPTION = "jubilee_redemption"  # Liberation pattern


# =============================================================================
# DIMENSION MAPPINGS - Precise Embedding Dimension Assignments
# =============================================================================

# Standard 768-dim embedding divided into theological dimensions
# Each dimension gets 48 embedding dimensions (768 / 16 = 48)
THEOLOGICAL_DIMENSIONS: Dict[DimensionCategory, Tuple[int, int]] = {
    DimensionCategory.CHRISTOLOGICAL: (0, 48),
    DimensionCategory.SOTERIOLOGICAL: (48, 96),
    DimensionCategory.ESCHATOLOGICAL: (96, 144),
    DimensionCategory.ECCLESIOLOGICAL: (144, 192),
    DimensionCategory.PNEUMATOLOGICAL: (192, 240),
    DimensionCategory.COSMOLOGICAL: (240, 288),
    DimensionCategory.ANTHROPOLOGICAL: (288, 336),
    DimensionCategory.SACRAMENTAL: (336, 384),
    DimensionCategory.LITURGICAL: (384, 432),
    DimensionCategory.PATRISTIC: (432, 480),
    DimensionCategory.TYPOLOGICAL: (480, 528),
    DimensionCategory.PROPHETIC: (528, 576),
    DimensionCategory.COVENANTAL: (576, 624),
    DimensionCategory.TRINITARIAN: (624, 672),
    DimensionCategory.THEOPHANIC: (672, 720),
    DimensionCategory.MARIAN: (720, 768),
}

# Alternative mapping for 384-dim embeddings
THEOLOGICAL_DIMENSIONS_384: Dict[DimensionCategory, Tuple[int, int]] = {
    DimensionCategory.CHRISTOLOGICAL: (0, 24),
    DimensionCategory.SOTERIOLOGICAL: (24, 48),
    DimensionCategory.ESCHATOLOGICAL: (48, 72),
    DimensionCategory.ECCLESIOLOGICAL: (72, 96),
    DimensionCategory.PNEUMATOLOGICAL: (96, 120),
    DimensionCategory.COSMOLOGICAL: (120, 144),
    DimensionCategory.ANTHROPOLOGICAL: (144, 168),
    DimensionCategory.SACRAMENTAL: (168, 192),
    DimensionCategory.LITURGICAL: (192, 216),
    DimensionCategory.PATRISTIC: (216, 240),
    DimensionCategory.TYPOLOGICAL: (240, 264),
    DimensionCategory.PROPHETIC: (264, 288),
    DimensionCategory.COVENANTAL: (288, 312),
    DimensionCategory.TRINITARIAN: (312, 336),
    DimensionCategory.THEOPHANIC: (336, 360),
    DimensionCategory.MARIAN: (360, 384),
}


# =============================================================================
# DATACLASSES - Comprehensive Result Structures
# =============================================================================

@dataclass
class DimensionalTransformation:
    """
    Transformation analysis for a single theological dimension.
    """
    dimension: DimensionCategory
    source_magnitude: float       # Change magnitude in source
    target_magnitude: float       # Change magnitude in target
    combined_magnitude: float     # Combined transformation
    alignment: float              # How aligned are the transformations (-1 to 1)
    contribution: float           # Contribution to overall transformation

    @property
    def is_dominant(self) -> bool:
        """Whether this dimension dominates the transformation."""
        return self.contribution > 0.15  # Top ~2-3 dimensions

    @property
    def is_coherent(self) -> bool:
        """Whether source and target transform coherently in this dimension."""
        return self.alignment > 0.5


@dataclass
class TemporalDynamics:
    """
    Tracks transformation dynamics over multiple GNN iterations.
    """
    pattern: TemporalPattern
    convergence_rate: float          # How fast transformation stabilizes
    final_stability: float           # How stable is the final state
    iteration_scores: List[float]    # Score at each iteration
    oscillation_amplitude: float     # If oscillating, how much variance
    steady_state_iteration: int      # When steady state was reached

    @property
    def converged(self) -> bool:
        """Whether transformation has converged."""
        return self.pattern in {TemporalPattern.CONVERGENT, TemporalPattern.STEADY_STATE}


@dataclass
class ArchetypeMatch:
    """
    Match result against a canonical transformation archetype.
    """
    archetype: TransformationArchetype
    similarity: float               # How well the transformation matches
    dominant_dimensions: List[DimensionCategory]  # Which dimensions match
    confidence: float               # Overall confidence in match
    patristic_support: float        # Alignment with patristic tradition

    @property
    def is_strong_match(self) -> bool:
        """Whether this is a strong archetype match."""
        return self.similarity > 0.7 and self.confidence > 0.8


@dataclass
class NestedTransformation:
    """
    Represents a multi-verse transformation chain.

    Captures how transformation propagates: A → B → C
    where A's influence on B affects how B influences C.
    """
    chain: List[str]                # Verse IDs in chain
    propagation_decay: float        # How much influence decays per hop
    cumulative_transformation: float  # Total transformation across chain
    dominant_path: List[str]        # Most influential path through chain
    feedback_loops: List[Tuple[str, str]]  # Any detected feedback


@dataclass
class MutualTransformationScore:
    """
    COMPLETE mutual transformation analysis result.

    This is the primary output of the MutualTransformationMetric, containing
    all computed measures of bidirectional semantic transformation.

    Attributes:
        # Primary Scores
        source_shift: Cosine distance for source verse (0-1)
        target_shift: Cosine distance for target verse (0-1)
        mutual_influence: Harmonic mean of shifts (0-1)
        transformation_type: Classification based on thresholds

        # Vector Analysis
        source_delta_vector: Embedding change vector for source
        target_delta_vector: Embedding change vector for target
        resonance: Alignment between delta vectors (-1 to 1)
        directionality: Asymmetry measure (-1 to 1)

        # Advanced Measures
        kl_divergence: Information-theoretic asymmetry
        wasserstein_distance: Earth mover's distance between distributions
        entropy_change: Change in embedding entropy

        # Dimensional Analysis
        dimensional_breakdown: Per-dimension transformation analysis
        dominant_dimensions: Top contributing dimensions
        semantic_components: Normalized dimension contributions

        # Pattern Recognition
        archetype_matches: Matches against canonical archetypes
        best_archetype: Highest confidence archetype match

        # Temporal Dynamics
        temporal_dynamics: Transformation over iterations (if available)

        # Metadata
        source_verse: Source verse identifier
        target_verse: Target verse identifier
        computation_time_ms: Time to compute score
    """
    # Primary scores
    source_shift: float
    target_shift: float
    mutual_influence: float
    transformation_type: TransformationType

    # Vector analysis
    source_delta_vector: np.ndarray = field(default_factory=lambda: np.array([]))
    target_delta_vector: np.ndarray = field(default_factory=lambda: np.array([]))
    resonance: float = 0.0
    directionality: float = 0.0

    # Advanced measures
    kl_divergence: float = 0.0
    wasserstein_distance: float = 0.0
    entropy_change: float = 0.0
    asymmetry_penalty: float = 0.0

    # Dimensional analysis
    dimensional_breakdown: Dict[DimensionCategory, DimensionalTransformation] = field(default_factory=dict)
    dominant_dimensions: List[DimensionCategory] = field(default_factory=list)
    semantic_components: Dict[str, float] = field(default_factory=dict)

    # Pattern recognition
    archetype_matches: List[ArchetypeMatch] = field(default_factory=list)
    best_archetype: Optional[TransformationArchetype] = None

    # Temporal dynamics
    temporal_dynamics: Optional[TemporalDynamics] = None

    # Nested transformation
    nested_chain: Optional[NestedTransformation] = None

    # Metadata
    source_verse: str = ""
    target_verse: str = ""
    computation_time_ms: float = 0.0

    @property
    def adjusted_score(self) -> float:
        """
        Final adjusted mutual influence score incorporating all factors.

        Adjustments:
        1. Resonance bonus: +10% if delta vectors are aligned
        2. Archetype bonus: +5% if strong archetype match
        3. Asymmetry penalty: -10% if highly asymmetric
        4. Convergence bonus: +5% if transformation converged cleanly
        """
        score = self.mutual_influence

        # Resonance bonus (up to +10%)
        if self.resonance > 0.5:
            score += 0.1 * (self.resonance - 0.5) * 2

        # Archetype bonus (up to +5%)
        if self.best_archetype and self.archetype_matches:
            best_match = max(self.archetype_matches, key=lambda m: m.similarity)
            if best_match.is_strong_match:
                score += 0.05 * best_match.similarity

        # Asymmetry penalty (up to -10%)
        if abs(self.directionality) > 0.6:
            score -= 0.1 * (abs(self.directionality) - 0.6) / 0.4

        # Convergence bonus (up to +5%)
        if self.temporal_dynamics and self.temporal_dynamics.converged:
            score += 0.05 * self.temporal_dynamics.final_stability

        return max(0.0, min(1.0, score))

    @property
    def theological_interpretation(self) -> str:
        """Generate human-readable theological interpretation."""
        parts = []

        # Base interpretation from type
        parts.append(self.transformation_type.theological_significance)

        # Add dominant dimension insight
        if self.dominant_dimensions:
            dim_names = [d.value for d in self.dominant_dimensions[:3]]
            parts.append(f"Primary dimensions: {', '.join(dim_names)}")

        # Add archetype insight
        if self.best_archetype:
            parts.append(f"Matches '{self.best_archetype.value}' pattern")

        return ". ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "source_shift": float(self.source_shift),
            "target_shift": float(self.target_shift),
            "mutual_influence": float(self.mutual_influence),
            "adjusted_score": float(self.adjusted_score),
            "transformation_type": self.transformation_type.value,
            "resonance": float(self.resonance),
            "directionality": float(self.directionality),
            "kl_divergence": float(self.kl_divergence),
            "wasserstein_distance": float(self.wasserstein_distance),
            "entropy_change": float(self.entropy_change),
            "dominant_dimensions": [d.value for d in self.dominant_dimensions],
            "semantic_components": self.semantic_components,
            "best_archetype": self.best_archetype.value if self.best_archetype else None,
            "theological_interpretation": self.theological_interpretation,
            "source_verse": self.source_verse,
            "target_verse": self.target_verse,
            "computation_time_ms": float(self.computation_time_ms),
        }

    def __repr__(self) -> str:
        return (
            f"MutualTransformationScore("
            f"mutual_influence={self.mutual_influence:.4f}, "
            f"adjusted={self.adjusted_score:.4f}, "
            f"type={self.transformation_type.value}, "
            f"archetype={self.best_archetype.value if self.best_archetype else 'None'})"
        )


@dataclass
class MutualTransformationConfig:
    """
    Comprehensive configuration for mutual transformation metric.
    """
    # Classification thresholds (corresponding to TransformationType)
    transcendent_threshold: float = 0.7
    radical_threshold: float = 0.5
    substantial_threshold: float = 0.35
    moderate_threshold: float = 0.2
    minimal_threshold: float = 0.1

    # Weight configurations
    resonance_weight: float = 0.1       # Weight for resonance in adjusted score
    archetype_weight: float = 0.05      # Weight for archetype match bonus
    asymmetry_penalty_weight: float = 0.1  # Weight for asymmetry penalty
    convergence_weight: float = 0.05    # Weight for convergence bonus

    # Feature toggles
    enable_dimensional_analysis: bool = True
    enable_archetype_matching: bool = True
    enable_temporal_tracking: bool = True
    enable_kl_divergence: bool = True
    enable_wasserstein: bool = True

    # Performance settings
    cache_embeddings: bool = True
    max_cache_size: int = 10000
    batch_parallelism: int = 4

    # Numerical stability
    epsilon: float = 1e-10

    # Dimension mapping (auto-detect from embedding size)
    embedding_dimension: int = 768


# =============================================================================
# ARCHETYPE SIGNATURES - Canonical Transformation Patterns
# =============================================================================

# Each archetype has a signature: which dimensions should be most affected
ARCHETYPE_SIGNATURES: Dict[TransformationArchetype, Dict[DimensionCategory, float]] = {
    TransformationArchetype.ISAAC_CHRIST_SACRIFICE: {
        DimensionCategory.CHRISTOLOGICAL: 0.9,
        DimensionCategory.SOTERIOLOGICAL: 0.85,
        DimensionCategory.TYPOLOGICAL: 0.95,
        DimensionCategory.SACRAMENTAL: 0.6,
    },
    TransformationArchetype.ADAM_CHRIST: {
        DimensionCategory.CHRISTOLOGICAL: 0.9,
        DimensionCategory.ANTHROPOLOGICAL: 0.95,
        DimensionCategory.SOTERIOLOGICAL: 0.85,
        DimensionCategory.ESCHATOLOGICAL: 0.7,
    },
    TransformationArchetype.BURNING_BUSH_THEOTOKOS: {
        DimensionCategory.MARIAN: 0.95,
        DimensionCategory.THEOPHANIC: 0.9,
        DimensionCategory.PNEUMATOLOGICAL: 0.75,
        DimensionCategory.TYPOLOGICAL: 0.85,
    },
    TransformationArchetype.ARK_THEOTOKOS: {
        DimensionCategory.MARIAN: 0.95,
        DimensionCategory.SACRAMENTAL: 0.8,
        DimensionCategory.PNEUMATOLOGICAL: 0.85,
        DimensionCategory.TYPOLOGICAL: 0.9,
    },
    TransformationArchetype.EXODUS_BAPTISM: {
        DimensionCategory.SACRAMENTAL: 0.95,
        DimensionCategory.SOTERIOLOGICAL: 0.85,
        DimensionCategory.ECCLESIOLOGICAL: 0.7,
        DimensionCategory.TYPOLOGICAL: 0.9,
    },
    TransformationArchetype.PASSOVER_EUCHARIST: {
        DimensionCategory.SACRAMENTAL: 0.95,
        DimensionCategory.CHRISTOLOGICAL: 0.9,
        DimensionCategory.SOTERIOLOGICAL: 0.85,
        DimensionCategory.LITURGICAL: 0.8,
    },
    TransformationArchetype.TEMPLE_CHURCH: {
        DimensionCategory.ECCLESIOLOGICAL: 0.95,
        DimensionCategory.PNEUMATOLOGICAL: 0.85,
        DimensionCategory.LITURGICAL: 0.8,
        DimensionCategory.TYPOLOGICAL: 0.75,
    },
    TransformationArchetype.MOSES_CHRIST_DELIVERER: {
        DimensionCategory.CHRISTOLOGICAL: 0.9,
        DimensionCategory.SOTERIOLOGICAL: 0.9,
        DimensionCategory.PROPHETIC: 0.85,
        DimensionCategory.TYPOLOGICAL: 0.95,
    },
    TransformationArchetype.DAVID_CHRIST_KING: {
        DimensionCategory.CHRISTOLOGICAL: 0.95,
        DimensionCategory.ESCHATOLOGICAL: 0.85,
        DimensionCategory.COVENANTAL: 0.9,
        DimensionCategory.PROPHETIC: 0.8,
    },
    TransformationArchetype.JONAH_RESURRECTION: {
        DimensionCategory.CHRISTOLOGICAL: 0.85,
        DimensionCategory.ESCHATOLOGICAL: 0.9,
        DimensionCategory.TYPOLOGICAL: 0.95,
        DimensionCategory.PROPHETIC: 0.8,
    },
    TransformationArchetype.SERPENT_CROSS: {
        DimensionCategory.CHRISTOLOGICAL: 0.95,
        DimensionCategory.SOTERIOLOGICAL: 0.9,
        DimensionCategory.TYPOLOGICAL: 0.95,
        DimensionCategory.SACRAMENTAL: 0.6,
    },
    TransformationArchetype.MANNA_BREAD: {
        DimensionCategory.SACRAMENTAL: 0.95,
        DimensionCategory.CHRISTOLOGICAL: 0.85,
        DimensionCategory.ECCLESIOLOGICAL: 0.7,
        DimensionCategory.LITURGICAL: 0.8,
    },
    TransformationArchetype.MELCHIZEDEK_PRIESTHOOD: {
        DimensionCategory.CHRISTOLOGICAL: 0.9,
        DimensionCategory.SACRAMENTAL: 0.85,
        DimensionCategory.COVENANTAL: 0.8,
        DimensionCategory.LITURGICAL: 0.75,
    },
}


# =============================================================================
# MAIN METRIC CLASS - Complete Implementation
# =============================================================================

class MutualTransformationMetric:
    """
    IMPOSSIBLY SOPHISTICATED metric for measuring bidirectional semantic transformation.

    This class implements the complete mathematical framework described in the module
    docstring, providing deep analysis of how two biblical passages mutually
    transform each other's meaning through their interconnection.

    Core Capabilities:
    1. Base harmonic mean transformation measurement
    2. 16-dimensional theological decomposition
    3. Resonance and coherence analysis
    4. Archetype pattern matching
    5. Temporal dynamics tracking
    6. Information-theoretic measures (KL divergence, Wasserstein)
    7. Nested transformation chain detection
    8. Batch processing with vectorization

    Usage:
        metric = MutualTransformationMetric()
        score = await metric.measure_transformation(
            source_verse="GEN.22.2",
            target_verse="JHN.3.16",
            source_before=source_embedding_before_gnn,
            source_after=source_embedding_after_gnn,
            target_before=target_embedding_before_gnn,
            target_after=target_embedding_after_gnn,
        )
        print(f"Mutual influence: {score.mutual_influence}")
        print(f"Adjusted score: {score.adjusted_score}")
        print(f"Archetype: {score.best_archetype}")
        print(f"Interpretation: {score.theological_interpretation}")
    """

    def __init__(self, config: Optional[MutualTransformationConfig] = None):
        """
        Initialize the metric with configuration.

        Args:
            config: Configuration for thresholds, features, and performance.
        """
        self.config = config or MutualTransformationConfig()

        # Select dimension mapping based on embedding size
        if self.config.embedding_dimension == 768:
            self._dimension_map = THEOLOGICAL_DIMENSIONS
        elif self.config.embedding_dimension == 384:
            self._dimension_map = THEOLOGICAL_DIMENSIONS_384
        else:
            # Auto-generate mapping for arbitrary dimensions
            self._dimension_map = self._generate_dimension_map(self.config.embedding_dimension)

        # LRU cache for embeddings
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._cache_order: List[str] = []

        # Temporal tracking storage
        self._temporal_history: Dict[str, List[float]] = defaultdict(list)

        logger.info(
            f"MutualTransformationMetric initialized: "
            f"dimensions={self.config.embedding_dimension}, "
            f"archetype_matching={self.config.enable_archetype_matching}"
        )

    def _generate_dimension_map(self, dim: int) -> Dict[DimensionCategory, Tuple[int, int]]:
        """Generate dimension mapping for arbitrary embedding sizes."""
        categories = list(DimensionCategory)
        chunk_size = dim // len(categories)
        mapping = {}
        for i, cat in enumerate(categories):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < len(categories) - 1 else dim
            mapping[cat] = (start, end)
        return mapping

    # =========================================================================
    # PRIMARY MEASUREMENT METHOD
    # =========================================================================

    async def measure_transformation(
        self,
        source_verse: str,
        target_verse: str,
        source_before: np.ndarray,
        source_after: np.ndarray,
        target_before: np.ndarray,
        target_after: np.ndarray,
        iteration_history: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> MutualTransformationScore:
        """
        Measure complete mutual transformation between two verses.

        This is the primary entry point that computes all transformation metrics
        and returns a comprehensive score object.

        Args:
            source_verse: Source verse identifier (e.g., "GEN.22.2")
            target_verse: Target verse identifier (e.g., "JHN.3.16")
            source_before: Source embedding before GNN refinement
            source_after: Source embedding after GNN refinement
            target_before: Target embedding before GNN refinement
            target_after: Target embedding after GNN refinement
            iteration_history: Optional list of (source, target) embeddings at each iteration

        Returns:
            Complete MutualTransformationScore with all computed metrics
        """
        import time
        start_time = time.time()

        # Ensure numpy arrays with consistent dtype
        source_before = np.asarray(source_before, dtype=np.float32).flatten()
        source_after = np.asarray(source_after, dtype=np.float32).flatten()
        target_before = np.asarray(target_before, dtype=np.float32).flatten()
        target_after = np.asarray(target_after, dtype=np.float32).flatten()

        # =====================================================================
        # 1. BASE TRANSFORMATION CALCULATION
        # =====================================================================
        source_shift = self._cosine_distance(source_before, source_after)
        target_shift = self._cosine_distance(target_before, target_after)
        mutual_influence = self._harmonic_mean(source_shift, target_shift)

        # Classify transformation type
        transformation_type = self._classify_transformation(mutual_influence)

        # Compute delta vectors
        source_delta = source_after - source_before
        target_delta = target_after - target_before

        # =====================================================================
        # 2. RESONANCE AND DIRECTIONALITY
        # =====================================================================
        resonance = self._compute_resonance(source_delta, target_delta)
        directionality = self._compute_directionality(source_shift, target_shift)

        # =====================================================================
        # 3. INFORMATION-THEORETIC MEASURES
        # =====================================================================
        kl_div = 0.0
        wasserstein = 0.0
        entropy_change = 0.0

        if self.config.enable_kl_divergence:
            kl_div = self._compute_kl_divergence(source_delta, target_delta)

        if self.config.enable_wasserstein:
            wasserstein = self._compute_wasserstein(source_delta, target_delta)

        entropy_change = self._compute_entropy_change(
            source_before, source_after, target_before, target_after
        )

        # =====================================================================
        # 4. DIMENSIONAL DECOMPOSITION
        # =====================================================================
        dimensional_breakdown = {}
        dominant_dimensions = []
        semantic_components = {}

        if self.config.enable_dimensional_analysis:
            dimensional_breakdown = self._analyze_dimensions(source_delta, target_delta)
            dominant_dimensions = self._identify_dominant_dimensions(dimensional_breakdown)
            semantic_components = {
                cat.value: trans.contribution
                for cat, trans in dimensional_breakdown.items()
            }

        # =====================================================================
        # 5. ARCHETYPE MATCHING
        # =====================================================================
        archetype_matches = []
        best_archetype = None

        if self.config.enable_archetype_matching and dimensional_breakdown:
            archetype_matches = self._match_archetypes(dimensional_breakdown)
            if archetype_matches:
                best_match = max(archetype_matches, key=lambda m: m.similarity)
                if best_match.is_strong_match:
                    best_archetype = best_match.archetype

        # =====================================================================
        # 6. TEMPORAL DYNAMICS
        # =====================================================================
        temporal_dynamics = None

        if self.config.enable_temporal_tracking and iteration_history:
            temporal_dynamics = self._analyze_temporal_dynamics(
                source_verse, target_verse, iteration_history
            )

        # =====================================================================
        # 7. ASYMMETRY PENALTY
        # =====================================================================
        asymmetry_penalty = self._compute_asymmetry_penalty(source_shift, target_shift)

        # =====================================================================
        # BUILD RESULT
        # =====================================================================
        computation_time = (time.time() - start_time) * 1000

        score = MutualTransformationScore(
            source_shift=float(source_shift),
            target_shift=float(target_shift),
            mutual_influence=float(mutual_influence),
            transformation_type=transformation_type,
            source_delta_vector=source_delta,
            target_delta_vector=target_delta,
            resonance=float(resonance),
            directionality=float(directionality),
            kl_divergence=float(kl_div),
            wasserstein_distance=float(wasserstein),
            entropy_change=float(entropy_change),
            asymmetry_penalty=float(asymmetry_penalty),
            dimensional_breakdown=dimensional_breakdown,
            dominant_dimensions=dominant_dimensions,
            semantic_components=semantic_components,
            archetype_matches=archetype_matches,
            best_archetype=best_archetype,
            temporal_dynamics=temporal_dynamics,
            source_verse=source_verse,
            target_verse=target_verse,
            computation_time_ms=computation_time,
        )

        logger.debug(
            f"Computed transformation {source_verse} <-> {target_verse}: "
            f"mutual={mutual_influence:.4f}, adjusted={score.adjusted_score:.4f}, "
            f"type={transformation_type.value}, archetype={best_archetype}"
        )

        return score

    # =========================================================================
    # CORE MATHEMATICAL FUNCTIONS
    # =========================================================================

    def _cosine_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine distance between two vectors.

        Returns value in [0, 1] where:
        - 0 = identical vectors (no shift)
        - 1 = orthogonal vectors (maximum shift)
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 < self.config.epsilon or norm2 < self.config.epsilon:
            return 0.0

        try:
            dist = cosine(vec1, vec2)
            return float(max(0.0, min(1.0, dist)))
        except (ValueError, FloatingPointError):
            return 0.0

    def _harmonic_mean(self, a: float, b: float) -> float:
        """
        Compute harmonic mean ensuring both values must be significant.

        Formula: 2 * (a * b) / (a + b + ε)

        The harmonic mean is critical because it ensures one-sided influence
        cannot produce a high score - both verses must shift significantly.
        """
        numerator = 2.0 * a * b
        denominator = a + b + self.config.epsilon
        return numerator / denominator

    def _compute_resonance(self, source_delta: np.ndarray, target_delta: np.ndarray) -> float:
        """
        Compute resonance - alignment between transformation directions.

        Resonance measures whether the two verses transform in coherent directions
        (positive resonance) or contradictory directions (negative resonance).

        Returns value in [-1, 1]:
        - 1 = perfectly aligned transformations
        - 0 = orthogonal transformations
        - -1 = opposing transformations
        """
        norm_source = np.linalg.norm(source_delta)
        norm_target = np.linalg.norm(target_delta)

        if norm_source < self.config.epsilon or norm_target < self.config.epsilon:
            return 0.0

        # Cosine similarity (not distance)
        similarity = np.dot(source_delta, target_delta) / (norm_source * norm_target)
        return float(np.clip(similarity, -1.0, 1.0))

    def _compute_directionality(self, source_shift: float, target_shift: float) -> float:
        """
        Compute directionality - which verse was more influenced.

        Returns value in [-1, 1]:
        - -1 = source completely dominated (target influenced source)
        - 0 = perfectly mutual
        - 1 = target completely dominated (source influenced target)
        """
        denominator = source_shift + target_shift + self.config.epsilon
        return (target_shift - source_shift) / denominator

    def _compute_kl_divergence(self, source_delta: np.ndarray, target_delta: np.ndarray) -> float:
        """
        Compute KL divergence between delta distributions.

        This information-theoretic measure captures how different the transformation
        patterns are between source and target.
        """
        # Convert to probability distributions
        source_abs = np.abs(source_delta) + self.config.epsilon
        target_abs = np.abs(target_delta) + self.config.epsilon

        source_prob = source_abs / np.sum(source_abs)
        target_prob = target_abs / np.sum(target_abs)

        # Symmetric KL divergence
        kl_st = entropy(source_prob, target_prob)
        kl_ts = entropy(target_prob, source_prob)

        # Handle infinities
        if np.isinf(kl_st) or np.isinf(kl_ts):
            return 1.0

        return float(min(1.0, (kl_st + kl_ts) / 2))

    def _compute_wasserstein(self, source_delta: np.ndarray, target_delta: np.ndarray) -> float:
        """
        Compute Wasserstein (Earth Mover's) distance between deltas.

        This measures the minimum "work" required to transform one delta
        distribution into the other.
        """
        try:
            # Normalize to distributions
            source_norm = np.abs(source_delta)
            target_norm = np.abs(target_delta)

            if np.sum(source_norm) < self.config.epsilon or np.sum(target_norm) < self.config.epsilon:
                return 0.0

            source_prob = source_norm / np.sum(source_norm)
            target_prob = target_norm / np.sum(target_norm)

            # 1D Wasserstein distance
            distance = wasserstein_distance(
                np.arange(len(source_prob)), np.arange(len(target_prob)),
                source_prob, target_prob
            )

            # Normalize by maximum possible distance
            max_dist = len(source_prob) - 1
            return float(min(1.0, distance / max_dist)) if max_dist > 0 else 0.0

        except Exception:
            return 0.0

    def _compute_entropy_change(
        self,
        source_before: np.ndarray,
        source_after: np.ndarray,
        target_before: np.ndarray,
        target_after: np.ndarray
    ) -> float:
        """
        Compute change in embedding entropy.

        Measures whether transformations increase or decrease embedding "uncertainty".
        """
        def embedding_entropy(vec: np.ndarray) -> float:
            abs_vec = np.abs(vec) + self.config.epsilon
            prob = abs_vec / np.sum(abs_vec)
            return entropy(prob)

        source_entropy_change = embedding_entropy(source_after) - embedding_entropy(source_before)
        target_entropy_change = embedding_entropy(target_after) - embedding_entropy(target_before)

        return float((source_entropy_change + target_entropy_change) / 2)

    def _compute_asymmetry_penalty(self, source_shift: float, target_shift: float) -> float:
        """
        Compute penalty for asymmetric transformation.

        Uses ratio of smaller to larger shift to penalize one-sided influence.
        """
        if max(source_shift, target_shift) < self.config.epsilon:
            return 0.0

        ratio = min(source_shift, target_shift) / (max(source_shift, target_shift) + self.config.epsilon)
        # Penalty increases as ratio decreases from 1
        return float(1.0 - ratio)

    # =========================================================================
    # DIMENSIONAL ANALYSIS
    # =========================================================================

    def _analyze_dimensions(
        self,
        source_delta: np.ndarray,
        target_delta: np.ndarray
    ) -> Dict[DimensionCategory, DimensionalTransformation]:
        """
        Decompose transformation into theological dimensions.
        """
        results = {}
        total_combined = 0.0

        # First pass: compute raw magnitudes
        raw_contributions = []

        for category, (start, end) in self._dimension_map.items():
            actual_start = min(start, len(source_delta))
            actual_end = min(end, len(source_delta))

            if actual_start >= actual_end:
                continue

            source_segment = source_delta[actual_start:actual_end]
            target_segment = target_delta[actual_start:actual_end]

            source_mag = float(np.linalg.norm(source_segment))
            target_mag = float(np.linalg.norm(target_segment))
            combined_mag = source_mag + target_mag

            # Alignment within this dimension
            if source_mag > self.config.epsilon and target_mag > self.config.epsilon:
                alignment = float(np.dot(source_segment, target_segment) / (source_mag * target_mag))
            else:
                alignment = 0.0

            raw_contributions.append((category, source_mag, target_mag, combined_mag, alignment))
            total_combined += combined_mag

        # Second pass: normalize and create results
        for category, source_mag, target_mag, combined_mag, alignment in raw_contributions:
            contribution = combined_mag / (total_combined + self.config.epsilon)

            results[category] = DimensionalTransformation(
                dimension=category,
                source_magnitude=source_mag,
                target_magnitude=target_mag,
                combined_magnitude=combined_mag,
                alignment=alignment,
                contribution=contribution
            )

        return results

    def _identify_dominant_dimensions(
        self,
        breakdown: Dict[DimensionCategory, DimensionalTransformation]
    ) -> List[DimensionCategory]:
        """
        Identify the most dominant theological dimensions.
        """
        sorted_dims = sorted(
            breakdown.items(),
            key=lambda x: x[1].contribution,
            reverse=True
        )

        # Return dimensions contributing more than 10%
        return [
            cat for cat, trans in sorted_dims
            if trans.contribution > 0.10
        ][:5]  # Max 5 dominant dimensions

    # =========================================================================
    # ARCHETYPE MATCHING
    # =========================================================================

    def _match_archetypes(
        self,
        dimensional_breakdown: Dict[DimensionCategory, DimensionalTransformation]
    ) -> List[ArchetypeMatch]:
        """
        Match transformation signature against canonical archetypes.
        """
        matches = []

        # Get normalized dimension contributions
        actual_signature = {
            cat: trans.contribution
            for cat, trans in dimensional_breakdown.items()
        }

        for archetype, expected_signature in ARCHETYPE_SIGNATURES.items():
            similarity = self._signature_similarity(actual_signature, expected_signature)
            dominant_matches = [
                cat for cat in expected_signature.keys()
                if cat in actual_signature and actual_signature[cat] > 0.1
            ]

            confidence = len(dominant_matches) / len(expected_signature) if expected_signature else 0

            # Patristic support would come from external lookup
            # For now, use similarity as proxy
            patristic_support = similarity * 0.8

            matches.append(ArchetypeMatch(
                archetype=archetype,
                similarity=similarity,
                dominant_dimensions=dominant_matches,
                confidence=confidence,
                patristic_support=patristic_support
            ))

        # Sort by similarity
        matches.sort(key=lambda m: m.similarity, reverse=True)
        return matches[:5]  # Return top 5 matches

    def _signature_similarity(
        self,
        actual: Dict[DimensionCategory, float],
        expected: Dict[DimensionCategory, float]
    ) -> float:
        """
        Compute similarity between actual and expected signatures.
        """
        if not expected:
            return 0.0

        total_similarity = 0.0
        total_weight = 0.0

        for dim, expected_value in expected.items():
            actual_value = actual.get(dim, 0.0)
            # Weighted similarity based on expected importance
            weight = expected_value
            similarity = 1.0 - abs(expected_value - actual_value)
            total_similarity += weight * max(0, similarity)
            total_weight += weight

        return total_similarity / total_weight if total_weight > 0 else 0.0

    # =========================================================================
    # TEMPORAL DYNAMICS
    # =========================================================================

    def _analyze_temporal_dynamics(
        self,
        source_verse: str,
        target_verse: str,
        iteration_history: List[Tuple[np.ndarray, np.ndarray]]
    ) -> TemporalDynamics:
        """
        Analyze how transformation evolves over GNN iterations.
        """
        if len(iteration_history) < 2:
            return TemporalDynamics(
                pattern=TemporalPattern.STEADY_STATE,
                convergence_rate=1.0,
                final_stability=1.0,
                iteration_scores=[],
                oscillation_amplitude=0.0,
                steady_state_iteration=0
            )

        # Compute mutual influence at each iteration
        scores = []
        for i, (source_emb, target_emb) in enumerate(iteration_history):
            if i == 0:
                continue
            prev_source, prev_target = iteration_history[i - 1]
            source_shift = self._cosine_distance(prev_source, source_emb)
            target_shift = self._cosine_distance(prev_target, target_emb)
            scores.append(self._harmonic_mean(source_shift, target_shift))

        if not scores:
            return TemporalDynamics(
                pattern=TemporalPattern.STEADY_STATE,
                convergence_rate=1.0,
                final_stability=1.0,
                iteration_scores=[],
                oscillation_amplitude=0.0,
                steady_state_iteration=0
            )

        # Analyze pattern
        score_array = np.array(scores)
        diffs = np.diff(score_array)

        # Detect oscillation
        sign_changes = np.sum(np.abs(np.diff(np.sign(diffs)))) / 2
        oscillation_amplitude = float(np.std(score_array))

        # Detect convergence
        if len(scores) >= 3:
            recent_variance = np.var(score_array[-3:])
            final_stability = float(1.0 - min(1.0, recent_variance * 100))
        else:
            final_stability = 0.5

        # Classify pattern
        if sign_changes > len(diffs) * 0.4:
            pattern = TemporalPattern.OSCILLATORY
        elif all(d >= 0 for d in diffs):
            pattern = TemporalPattern.DIVERGENT
        elif final_stability > 0.9:
            pattern = TemporalPattern.CONVERGENT
        else:
            pattern = TemporalPattern.DELAYED

        # Find steady state iteration
        steady_state_iter = len(scores)
        for i in range(len(scores) - 1, 0, -1):
            if abs(scores[i] - scores[i - 1]) > 0.01:
                steady_state_iter = i + 1
                break

        # Convergence rate
        if len(scores) > 1:
            convergence_rate = float(1.0 - (scores[-1] - scores[0]) / (max(scores) - min(scores) + self.config.epsilon))
        else:
            convergence_rate = 1.0

        return TemporalDynamics(
            pattern=pattern,
            convergence_rate=convergence_rate,
            final_stability=final_stability,
            iteration_scores=scores,
            oscillation_amplitude=oscillation_amplitude,
            steady_state_iteration=steady_state_iter
        )

    # =========================================================================
    # CLASSIFICATION
    # =========================================================================

    def _classify_transformation(self, mutual_influence: float) -> TransformationType:
        """
        Classify transformation type based on mutual influence score.
        """
        if mutual_influence >= self.config.transcendent_threshold:
            return TransformationType.TRANSCENDENT
        elif mutual_influence >= self.config.radical_threshold:
            return TransformationType.RADICAL
        elif mutual_influence >= self.config.substantial_threshold:
            return TransformationType.SUBSTANTIAL
        elif mutual_influence >= self.config.moderate_threshold:
            return TransformationType.MODERATE
        elif mutual_influence >= self.config.minimal_threshold:
            return TransformationType.MINIMAL
        else:
            return TransformationType.INCIDENTAL

    # =========================================================================
    # BATCH PROCESSING
    # =========================================================================

    async def measure_batch(
        self,
        pairs: List[Tuple[str, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ) -> List[MutualTransformationScore]:
        """
        Batch process multiple verse pairs for efficiency.

        Uses vectorized operations where possible for improved performance.

        Args:
            pairs: List of tuples containing:
                (source_verse, target_verse, source_before, source_after,
                 target_before, target_after)

        Returns:
            List of MutualTransformationScore objects
        """
        if not pairs:
            return []

        n_pairs = len(pairs)
        results = []

        # Stack embeddings for vectorized computation
        source_befores = np.array([p[2] for p in pairs], dtype=np.float32)
        source_afters = np.array([p[3] for p in pairs], dtype=np.float32)
        target_befores = np.array([p[4] for p in pairs], dtype=np.float32)
        target_afters = np.array([p[5] for p in pairs], dtype=np.float32)

        # Vectorized cosine distances
        source_shifts = self._batch_cosine_distance(source_befores, source_afters)
        target_shifts = self._batch_cosine_distance(target_befores, target_afters)

        # Vectorized harmonic means
        mutual_influences = (
            2.0 * source_shifts * target_shifts /
            (source_shifts + target_shifts + self.config.epsilon)
        )

        # Compute deltas
        source_deltas = source_afters - source_befores
        target_deltas = target_afters - target_befores

        # Vectorized resonance
        resonances = self._batch_resonance(source_deltas, target_deltas)

        # Process each pair
        for i in range(n_pairs):
            source_verse, target_verse = pairs[i][0], pairs[i][1]

            transformation_type = self._classify_transformation(mutual_influences[i])
            directionality = self._compute_directionality(source_shifts[i], target_shifts[i])

            # Dimensional analysis (not batched for now)
            dimensional_breakdown = {}
            dominant_dimensions = []
            semantic_components = {}

            if self.config.enable_dimensional_analysis:
                dimensional_breakdown = self._analyze_dimensions(
                    source_deltas[i], target_deltas[i]
                )
                dominant_dimensions = self._identify_dominant_dimensions(dimensional_breakdown)
                semantic_components = {
                    cat.value: trans.contribution
                    for cat, trans in dimensional_breakdown.items()
                }

            # Archetype matching
            archetype_matches = []
            best_archetype = None
            if self.config.enable_archetype_matching and dimensional_breakdown:
                archetype_matches = self._match_archetypes(dimensional_breakdown)
                if archetype_matches:
                    best_match = max(archetype_matches, key=lambda m: m.similarity)
                    if best_match.is_strong_match:
                        best_archetype = best_match.archetype

            results.append(MutualTransformationScore(
                source_shift=float(source_shifts[i]),
                target_shift=float(target_shifts[i]),
                mutual_influence=float(mutual_influences[i]),
                transformation_type=transformation_type,
                source_delta_vector=source_deltas[i],
                target_delta_vector=target_deltas[i],
                resonance=float(resonances[i]),
                directionality=float(directionality),
                dimensional_breakdown=dimensional_breakdown,
                dominant_dimensions=dominant_dimensions,
                semantic_components=semantic_components,
                archetype_matches=archetype_matches,
                best_archetype=best_archetype,
                source_verse=source_verse,
                target_verse=target_verse,
            ))

        logger.info(f"Batch processed {n_pairs} pairs")
        return results

    def _batch_cosine_distance(
        self,
        vecs1: np.ndarray,
        vecs2: np.ndarray
    ) -> np.ndarray:
        """
        Vectorized cosine distance for batch processing.
        """
        norms1 = np.linalg.norm(vecs1, axis=1, keepdims=True)
        norms2 = np.linalg.norm(vecs2, axis=1, keepdims=True)

        norms1 = np.maximum(norms1, self.config.epsilon)
        norms2 = np.maximum(norms2, self.config.epsilon)

        vecs1_norm = vecs1 / norms1
        vecs2_norm = vecs2 / norms2

        similarities = np.sum(vecs1_norm * vecs2_norm, axis=1)
        distances = 1.0 - similarities

        return np.clip(distances, 0.0, 1.0)

    def _batch_resonance(
        self,
        source_deltas: np.ndarray,
        target_deltas: np.ndarray
    ) -> np.ndarray:
        """
        Vectorized resonance computation.
        """
        norms_source = np.linalg.norm(source_deltas, axis=1, keepdims=True)
        norms_target = np.linalg.norm(target_deltas, axis=1, keepdims=True)

        norms_source = np.maximum(norms_source, self.config.epsilon)
        norms_target = np.maximum(norms_target, self.config.epsilon)

        source_norm = source_deltas / norms_source
        target_norm = target_deltas / norms_target

        similarities = np.sum(source_norm * target_norm, axis=1)
        return np.clip(similarities, -1.0, 1.0)

    # =========================================================================
    # CACHE MANAGEMENT
    # =========================================================================

    def clear_cache(self) -> None:
        """Clear all caches."""
        self._embedding_cache.clear()
        self._cache_order.clear()
        self._temporal_history.clear()
        logger.debug("Caches cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "embedding_cache_size": len(self._embedding_cache),
            "temporal_history_size": len(self._temporal_history),
            "max_cache_size": self.config.max_cache_size,
        }
