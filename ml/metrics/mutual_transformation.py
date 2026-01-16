"""
BIBLOS v2 - Mutual Transformation Metric

Measures bidirectional semantic shift between connected verses to determine
how genuinely two verses "redefine each other" when understood in relation.

Theological Principle:
    When two verses are genuinely connected, their meanings mutually transform.
    The Burning Bush (Exodus 3:2) and the Theotokos Mary both gain deeper
    significance when understood in relation to each other.

Mathematical Representation:
    Given two verses A and B with embeddings before and after GNN refinement:
    - source_shift = 1 - cosine_similarity(A_before, A_after)
    - target_shift = 1 - cosine_similarity(B_before, B_after)
    - mutual_influence = 2 * (source_shift * target_shift) / (source_shift + target_shift + ε)

    The harmonic mean ensures BOTH verses must shift for high mutual influence.
    One-sided influence scores low.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import logging

import numpy as np
from scipy.spatial.distance import cosine


logger = logging.getLogger("biblos.ml.metrics.mutual_transformation")


class TransformationType(Enum):
    """
    Classification of transformation intensity based on mutual influence score.

    RADICAL: Strong bidirectional transformation (e.g., Isaac → Christ)
        Both verses fundamentally redefine each other's meaning.

    MODERATE: Significant mutual influence (e.g., Temple → Church)
        Both verses contribute meaningfully to each other's understanding.

    MINIMAL: Weak or one-sided connection (e.g., geographic location match)
        Connection exists but doesn't substantially transform meaning.
    """
    RADICAL = "RADICAL"
    MODERATE = "MODERATE"
    MINIMAL = "MINIMAL"


@dataclass
class MutualTransformationScore:
    """
    Complete score for mutual transformation between two verses.

    Captures both the overall mutual influence and the detailed breakdown
    of how each verse's embedding shifted during GNN refinement.

    Attributes:
        source_shift: Cosine distance for source verse (0-1).
            Higher = more transformation.
        target_shift: Cosine distance for target verse (0-1).
            Higher = more transformation.
        mutual_influence: Harmonic mean of shifts (0-1).
            High only when BOTH verses shift significantly.
        transformation_type: Classification based on thresholds.
        source_delta_vector: Actual embedding change vector for source.
        target_delta_vector: Actual embedding change vector for target.
        directionality: Asymmetry measure (-1 to 1).
            -1 = source dominated, 0 = mutual, 1 = target dominated.
        semantic_components: Decomposition into interpretable dimensions.
    """
    source_shift: float
    target_shift: float
    mutual_influence: float
    transformation_type: TransformationType
    source_delta_vector: np.ndarray = field(default_factory=lambda: np.array([]))
    target_delta_vector: np.ndarray = field(default_factory=lambda: np.array([]))
    directionality: float = 0.0
    semantic_components: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "source_shift": float(self.source_shift),
            "target_shift": float(self.target_shift),
            "mutual_influence": float(self.mutual_influence),
            "transformation_type": self.transformation_type.value,
            "directionality": float(self.directionality),
            "semantic_components": self.semantic_components,
        }

    def __repr__(self) -> str:
        return (
            f"MutualTransformationScore("
            f"mutual_influence={self.mutual_influence:.4f}, "
            f"type={self.transformation_type.value}, "
            f"source_shift={self.source_shift:.4f}, "
            f"target_shift={self.target_shift:.4f})"
        )


@dataclass
class MutualTransformationConfig:
    """
    Configuration for mutual transformation metric computation.

    Attributes:
        radical_threshold: Score above which transformation is RADICAL.
        moderate_threshold: Score above which transformation is MODERATE.
        directionality_weight: Weight for directionality in scoring adjustment.
        enable_semantic_decomposition: Whether to decompose delta vectors.
        cache_embeddings: Whether to cache intermediate embeddings.
        epsilon: Small value to prevent division by zero.
    """
    radical_threshold: float = 0.4
    moderate_threshold: float = 0.2
    directionality_weight: float = 0.1
    enable_semantic_decomposition: bool = True
    cache_embeddings: bool = True
    epsilon: float = 1e-10


# Semantic dimension mappings for delta vector decomposition
THEOLOGICAL_DIMENSIONS = {
    "christological": [0, 64],      # Dimensions 0-63
    "soteriological": [64, 128],    # Dimensions 64-127
    "eschatological": [128, 192],   # Dimensions 128-191
    "ecclesiological": [192, 256],  # Dimensions 192-255
    "pneumatological": [256, 320],  # Dimensions 256-319
    "cosmological": [320, 384],     # Dimensions 320-383
    "anthropological": [384, 448],  # Dimensions 384-447
    "sacramental": [448, 512],      # Dimensions 448-511
    "liturgical": [512, 576],       # Dimensions 512-575
    "patristic": [576, 640],        # Dimensions 576-639
    "typological": [640, 704],      # Dimensions 640-703
    "prophetic": [704, 768],        # Dimensions 704-767
}


class MutualTransformationMetric:
    """
    Metric for measuring bidirectional semantic transformation between verses.

    This metric captures the core insight that in genuine cross-references,
    both verses mutually transform each other's meaning. A connection is
    considered stronger when understanding one verse changes how we
    understand the other, and vice versa.

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
        print(f"Type: {score.transformation_type}")
    """

    # Maximum cache size to prevent unbounded memory growth
    MAX_CACHE_SIZE = 5000

    def __init__(self, config: Optional[MutualTransformationConfig] = None):
        """
        Initialize the metric with optional configuration.

        Args:
            config: Configuration for thresholds and behavior.
        """
        self.config = config or MutualTransformationConfig()
        # Bounded cache with LRU eviction
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._cache_order: List[str] = []  # For LRU tracking
        logger.info(
            f"MutualTransformationMetric initialized with "
            f"radical_threshold={self.config.radical_threshold}, "
            f"moderate_threshold={self.config.moderate_threshold}"
        )

    def _cache_embedding(self, key: str, embedding: np.ndarray) -> None:
        """
        Cache an embedding with LRU eviction.

        Args:
            key: Cache key for the embedding.
            embedding: The embedding to cache.
        """
        if not self.config.cache_embeddings:
            return

        # Evict oldest entries if cache is full
        while len(self._embedding_cache) >= self.MAX_CACHE_SIZE:
            if self._cache_order:
                oldest_key = self._cache_order.pop(0)
                self._embedding_cache.pop(oldest_key, None)

        self._embedding_cache[key] = embedding
        self._cache_order.append(key)

    def _get_cached_embedding(self, key: str) -> Optional[np.ndarray]:
        """
        Get a cached embedding, updating LRU order.

        Args:
            key: Cache key to look up.

        Returns:
            Cached embedding or None if not found.
        """
        embedding = self._embedding_cache.get(key)
        if embedding is not None and key in self._cache_order:
            # Move to end (most recently used)
            self._cache_order.remove(key)
            self._cache_order.append(key)
        return embedding

    async def measure_transformation(
        self,
        source_verse: str,
        target_verse: str,
        source_before: np.ndarray,
        source_after: np.ndarray,
        target_before: np.ndarray,
        target_after: np.ndarray,
    ) -> MutualTransformationScore:
        """
        Measure the mutual transformation between two verses.

        Computes how much each verse's embedding shifted during GNN refinement,
        then combines using harmonic mean to ensure both must shift for
        high mutual influence.

        Args:
            source_verse: Source verse identifier (e.g., "GEN.22.2").
            target_verse: Target verse identifier (e.g., "JHN.3.16").
            source_before: Source verse embedding before GNN refinement.
            source_after: Source verse embedding after GNN refinement.
            target_before: Target verse embedding before GNN refinement.
            target_after: Target verse embedding after GNN refinement.

        Returns:
            MutualTransformationScore with full analysis.
        """
        # Ensure numpy arrays
        source_before = np.asarray(source_before, dtype=np.float32).flatten()
        source_after = np.asarray(source_after, dtype=np.float32).flatten()
        target_before = np.asarray(target_before, dtype=np.float32).flatten()
        target_after = np.asarray(target_after, dtype=np.float32).flatten()

        # Calculate cosine distances (shift amounts)
        source_shift = self._cosine_distance(source_before, source_after)
        target_shift = self._cosine_distance(target_before, target_after)

        # Compute harmonic mean (mutual influence)
        mutual_influence = self._harmonic_mean(source_shift, target_shift)

        # Classify transformation type
        transformation_type = self._classify_transformation(mutual_influence)

        # Calculate directionality
        directionality = self.calculate_directionality(source_shift, target_shift)

        # Compute delta vectors
        source_delta = source_after - source_before
        target_delta = target_after - target_before

        # Extract semantic components if enabled
        semantic_components = {}
        if self.config.enable_semantic_decomposition:
            semantic_components = self.extract_semantic_components(
                source_delta + target_delta,  # Combined delta
                THEOLOGICAL_DIMENSIONS,
            )

        score = MutualTransformationScore(
            source_shift=float(source_shift),
            target_shift=float(target_shift),
            mutual_influence=float(mutual_influence),
            transformation_type=transformation_type,
            source_delta_vector=source_delta,
            target_delta_vector=target_delta,
            directionality=float(directionality),
            semantic_components=semantic_components,
        )

        logger.debug(
            f"Measured transformation {source_verse} <-> {target_verse}: "
            f"mutual_influence={mutual_influence:.4f}, type={transformation_type.value}"
        )

        return score

    def _cosine_distance(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray,
    ) -> float:
        """
        Calculate cosine distance between two vectors.

        Returns value in [0, 1] where:
        - 0 = identical vectors (no shift)
        - 1 = orthogonal vectors (maximum shift)

        Args:
            vec1: First vector.
            vec2: Second vector.

        Returns:
            Cosine distance (1 - cosine_similarity).
        """
        # Handle zero vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 < self.config.epsilon or norm2 < self.config.epsilon:
            return 0.0  # No meaningful shift if either is zero

        # Use scipy's cosine distance (already returns 1 - similarity)
        try:
            dist = cosine(vec1, vec2)
            # Clamp to [0, 1] to handle numerical errors
            return float(max(0.0, min(1.0, dist)))
        except ValueError:
            return 0.0

    def _harmonic_mean(
        self,
        source_shift: float,
        target_shift: float,
    ) -> float:
        """
        Compute harmonic mean of source and target shifts.

        The harmonic mean ensures BOTH verses must shift significantly
        for a high mutual influence score. If one shift is near zero,
        the result is near zero regardless of the other.

        Formula: 2 * (a * b) / (a + b + ε)

        Args:
            source_shift: Source verse shift amount [0, 1].
            target_shift: Target verse shift amount [0, 1].

        Returns:
            Harmonic mean [0, 1].
        """
        numerator = 2.0 * source_shift * target_shift
        denominator = source_shift + target_shift + self.config.epsilon

        return numerator / denominator

    def _classify_transformation(
        self,
        mutual_influence: float,
    ) -> TransformationType:
        """
        Classify transformation type based on mutual influence score.

        Args:
            mutual_influence: Computed mutual influence score [0, 1].

        Returns:
            TransformationType enum value.
        """
        if mutual_influence > self.config.radical_threshold:
            return TransformationType.RADICAL
        elif mutual_influence > self.config.moderate_threshold:
            return TransformationType.MODERATE
        else:
            return TransformationType.MINIMAL

    def calculate_directionality(
        self,
        source_shift: float,
        target_shift: float,
    ) -> float:
        """
        Calculate directionality (asymmetry) of transformation.

        Returns value in [-1, 1] where:
        - -1 = source completely dominated (target transformed source)
        - 0 = perfectly mutual transformation
        - 1 = target completely dominated (source transformed target)

        Args:
            source_shift: Source verse shift amount.
            target_shift: Target verse shift amount.

        Returns:
            Directionality measure.
        """
        denominator = source_shift + target_shift + self.config.epsilon
        return (target_shift - source_shift) / denominator

    def extract_semantic_components(
        self,
        delta_vector: np.ndarray,
        vocabulary_index: Dict[str, List[int]],
    ) -> Dict[str, float]:
        """
        Decompose delta vector into interpretable semantic dimensions.

        Maps embedding dimension ranges to theological categories,
        computing the magnitude of change in each category.

        Args:
            delta_vector: The change in embedding space.
            vocabulary_index: Mapping of category names to dimension ranges.

        Returns:
            Dictionary of category contributions (normalized).
        """
        components = {}
        total_magnitude = np.linalg.norm(delta_vector) + self.config.epsilon

        for category, (start, end) in vocabulary_index.items():
            # Handle vectors shorter than expected dimensions
            actual_end = min(end, len(delta_vector))
            actual_start = min(start, len(delta_vector))

            if actual_start >= actual_end:
                components[category] = 0.0
                continue

            segment = delta_vector[actual_start:actual_end]
            segment_magnitude = np.linalg.norm(segment)

            # Normalize by total magnitude
            components[category] = float(segment_magnitude / total_magnitude)

        return components

    async def measure_batch(
        self,
        pairs: List[Tuple[str, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ) -> List[MutualTransformationScore]:
        """
        Batch processing of multiple verse pairs for efficiency.

        Uses vectorized operations where possible to improve performance.

        Args:
            pairs: List of tuples, each containing:
                (source_verse, target_verse,
                 source_before, source_after,
                 target_before, target_after)

        Returns:
            List of MutualTransformationScore objects in same order.
        """
        if not pairs:
            return []

        # Vectorized computation for better performance
        n_pairs = len(pairs)

        # Stack all embeddings
        source_befores = np.array([p[2] for p in pairs], dtype=np.float32)
        source_afters = np.array([p[3] for p in pairs], dtype=np.float32)
        target_befores = np.array([p[4] for p in pairs], dtype=np.float32)
        target_afters = np.array([p[5] for p in pairs], dtype=np.float32)

        # Vectorized cosine similarity computation
        source_shifts = self._batch_cosine_distance(source_befores, source_afters)
        target_shifts = self._batch_cosine_distance(target_befores, target_afters)

        # Vectorized harmonic mean
        mutual_influences = (
            2.0 * source_shifts * target_shifts /
            (source_shifts + target_shifts + self.config.epsilon)
        )

        # Build result objects
        results = []
        for i in range(n_pairs):
            source_verse, target_verse = pairs[i][0], pairs[i][1]

            transformation_type = self._classify_transformation(mutual_influences[i])
            directionality = self.calculate_directionality(
                source_shifts[i], target_shifts[i]
            )

            source_delta = source_afters[i] - source_befores[i]
            target_delta = target_afters[i] - target_befores[i]

            semantic_components = {}
            if self.config.enable_semantic_decomposition:
                semantic_components = self.extract_semantic_components(
                    source_delta + target_delta,
                    THEOLOGICAL_DIMENSIONS,
                )

            results.append(MutualTransformationScore(
                source_shift=float(source_shifts[i]),
                target_shift=float(target_shifts[i]),
                mutual_influence=float(mutual_influences[i]),
                transformation_type=transformation_type,
                source_delta_vector=source_delta,
                target_delta_vector=target_delta,
                directionality=float(directionality),
                semantic_components=semantic_components,
            ))

        logger.info(f"Batch processed {n_pairs} pairs")
        return results

    def _batch_cosine_distance(
        self,
        vecs1: np.ndarray,
        vecs2: np.ndarray,
    ) -> np.ndarray:
        """
        Vectorized cosine distance computation for batch processing.

        Args:
            vecs1: First set of vectors [n_samples, embedding_dim].
            vecs2: Second set of vectors [n_samples, embedding_dim].

        Returns:
            Array of cosine distances [n_samples].
        """
        # Normalize vectors
        norms1 = np.linalg.norm(vecs1, axis=1, keepdims=True)
        norms2 = np.linalg.norm(vecs2, axis=1, keepdims=True)

        # Avoid division by zero
        norms1 = np.maximum(norms1, self.config.epsilon)
        norms2 = np.maximum(norms2, self.config.epsilon)

        vecs1_norm = vecs1 / norms1
        vecs2_norm = vecs2 / norms2

        # Cosine similarity via dot product of normalized vectors
        similarities = np.sum(vecs1_norm * vecs2_norm, axis=1)

        # Convert to distance and clamp
        distances = 1.0 - similarities
        return np.clip(distances, 0.0, 1.0)

    def clear_cache(self) -> None:
        """Clear embedding cache."""
        self._embedding_cache.clear()
        self._cache_order.clear()
        logger.debug("Embedding cache cleared")
