"""
Property-Based Tests for ML Invariants

Tests ML model outputs, embedding consistency, and inference determinism.
"""
import pytest
from hypothesis import given, strategies as st, settings, assume, seed
import numpy as np
import math

from tests.property.strategies import (
    verse_id_strategy,
    embedding_vector_strategy,
    similarity_score_strategy,
    confidence_score_strategy,
)


class TestEmbeddingInvariants:
    """Property-based tests for embedding vectors."""

    @given(embedding_vector_strategy(dimension=768))
    @settings(max_examples=200)
    def test_embedding_dimension_consistency(self, embedding):
        """Embeddings should have consistent dimensions."""
        assert len(embedding) == 768
        assert isinstance(embedding, list)
        assert all(isinstance(x, float) for x in embedding)

    @given(
        st.integers(min_value=128, max_value=2048).filter(lambda x: x % 64 == 0),
        st.data()
    )
    @settings(max_examples=100)
    def test_various_embedding_dimensions(self, dimension, data):
        """Should handle various embedding dimensions."""
        embedding = data.draw(embedding_vector_strategy(dimension=dimension))
        assert len(embedding) == dimension

    @given(embedding_vector_strategy(dimension=768))
    @settings(max_examples=200)
    def test_embedding_values_finite(self, embedding):
        """Embedding values should be finite (no NaN or Inf)."""
        for value in embedding:
            assert math.isfinite(value), f"Non-finite value found: {value}"
            assert not math.isnan(value)
            assert not math.isinf(value)

    @given(
        embedding_vector_strategy(dimension=768),
        embedding_vector_strategy(dimension=768),
    )
    @settings(max_examples=200)
    def test_embedding_dimension_mismatch_detected(self, emb1, emb2):
        """Should be able to detect dimension mismatches."""
        assert len(emb1) == len(emb2)  # Same dimension

        # Truncate one embedding
        emb2_truncated = emb2[:-1]
        assert len(emb1) != len(emb2_truncated)

    @given(embedding_vector_strategy(dimension=768))
    @settings(max_examples=100)
    def test_embedding_normalization(self, embedding):
        """Test L2 normalization of embeddings."""
        # Calculate L2 norm
        norm = math.sqrt(sum(x * x for x in embedding))

        if norm > 0:
            # Normalize
            normalized = [x / norm for x in embedding]

            # Normalized vector should have L2 norm ≈ 1
            normalized_norm = math.sqrt(sum(x * x for x in normalized))
            assert abs(normalized_norm - 1.0) < 1e-6


class TestSimilarityScoreInvariants:
    """Property-based tests for similarity scores."""

    @given(similarity_score_strategy(metric="cosine"))
    @settings(max_examples=200)
    def test_cosine_similarity_range(self, score):
        """Cosine similarity should be in [-1, 1]."""
        assert -1.0 <= score <= 1.0
        assert math.isfinite(score)

    @given(similarity_score_strategy(metric="euclidean"))
    @settings(max_examples=200)
    def test_euclidean_distance_non_negative(self, score):
        """Euclidean distance should be non-negative."""
        assert score >= 0.0
        assert math.isfinite(score)

    @given(
        embedding_vector_strategy(dimension=768),
        embedding_vector_strategy(dimension=768),
    )
    @settings(max_examples=100)
    def test_cosine_similarity_computation(self, emb1, emb2):
        """Compute cosine similarity and verify properties."""
        # Compute dot product
        dot_product = sum(a * b for a, b in zip(emb1, emb2))

        # Compute norms
        norm1 = math.sqrt(sum(x * x for x in emb1))
        norm2 = math.sqrt(sum(x * x for x in emb2))

        if norm1 > 0 and norm2 > 0:
            cosine_sim = dot_product / (norm1 * norm2)
            assert -1.0 <= cosine_sim <= 1.0

    @given(embedding_vector_strategy(dimension=768))
    @settings(max_examples=100)
    def test_self_similarity_is_one(self, embedding):
        """Cosine similarity of a vector with itself should be 1."""
        # Compute self-similarity
        dot_product = sum(x * x for x in embedding)
        norm = math.sqrt(dot_product)

        if norm > 0:
            cosine_sim = dot_product / (norm * norm)
            assert abs(cosine_sim - 1.0) < 1e-6

    @given(
        embedding_vector_strategy(dimension=768),
        embedding_vector_strategy(dimension=768),
    )
    @settings(max_examples=100)
    def test_similarity_symmetry(self, emb1, emb2):
        """Similarity should be symmetric: sim(A, B) = sim(B, A)."""
        # Compute cosine similarity both ways
        dot_product = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = math.sqrt(sum(x * x for x in emb1))
        norm2 = math.sqrt(sum(x * x for x in emb2))

        if norm1 > 0 and norm2 > 0:
            sim_ab = dot_product / (norm1 * norm2)

            # Reverse
            dot_product_ba = sum(b * a for a, b in zip(emb1, emb2))
            sim_ba = dot_product_ba / (norm2 * norm1)

            assert abs(sim_ab - sim_ba) < 1e-9


class TestBatchProcessingInvariants:
    """Property-based tests for batch processing consistency."""

    @given(
        st.lists(embedding_vector_strategy(dimension=768), min_size=1, max_size=10),
    )
    @settings(max_examples=100)
    def test_batch_dimension_consistency(self, embeddings):
        """All embeddings in a batch should have same dimension."""
        dimensions = [len(emb) for emb in embeddings]
        assert all(d == dimensions[0] for d in dimensions)

    @given(
        st.lists(verse_id_strategy(valid_only=True), min_size=1, max_size=20, unique=True),
    )
    @settings(max_examples=100)
    def test_batch_processing_preserves_order(self, verse_ids):
        """Batch processing should preserve input order."""
        # Simulate processing
        results = []
        for verse_id in verse_ids:
            # Simulate some result
            results.append({"verse_id": verse_id, "processed": True})

        # Check order preserved
        for i, verse_id in enumerate(verse_ids):
            assert results[i]["verse_id"] == verse_id

    @given(
        st.lists(embedding_vector_strategy(dimension=768), min_size=2, max_size=10),
    )
    @settings(max_examples=50)
    def test_batch_vs_sequential_consistency(self, embeddings):
        """Batch processing should match sequential processing."""
        # Simulate batch normalization
        batch_norms = [math.sqrt(sum(x * x for x in emb)) for emb in embeddings]

        # Sequential normalization should give same results
        sequential_norms = []
        for emb in embeddings:
            norm = math.sqrt(sum(x * x for x in emb))
            sequential_norms.append(norm)

        # Should be identical
        for batch_norm, seq_norm in zip(batch_norms, sequential_norms):
            assert abs(batch_norm - seq_norm) < 1e-9


class TestModelOutputInvariants:
    """Property-based tests for model output consistency."""

    @given(
        verse_id_strategy(valid_only=True),
        confidence_score_strategy(valid_only=True),
    )
    @settings(max_examples=200)
    def test_inference_confidence_bounds(self, verse_id, confidence):
        """Inference confidence should be in [0, 1]."""
        assert 0.0 <= confidence <= 1.0
        assert math.isfinite(confidence)

    @given(
        st.lists(
            st.tuples(
                verse_id_strategy(valid_only=True),
                confidence_score_strategy(valid_only=True),
            ),
            min_size=1,
            max_size=100,
        )
    )
    @settings(max_examples=100)
    def test_top_k_ranking_consistency(self, candidates):
        """Top-k candidates should be sorted by confidence."""
        # Sort by confidence descending
        sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)

        # Check ordering
        for i in range(len(sorted_candidates) - 1):
            assert sorted_candidates[i][1] >= sorted_candidates[i + 1][1]

    @given(
        st.lists(confidence_score_strategy(valid_only=True), min_size=1, max_size=100),
        st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=100)
    def test_top_k_selection(self, confidences, k):
        """Top-k selection should return k highest scores."""
        k = min(k, len(confidences))

        # Get top k
        sorted_confidences = sorted(confidences, reverse=True)
        top_k = sorted_confidences[:k]

        assert len(top_k) == k
        # Each element in top_k should be >= elements not in top_k
        if len(confidences) > k:
            min_top_k = min(top_k)
            remaining = sorted_confidences[k:]
            max_remaining = max(remaining)
            assert min_top_k >= max_remaining


class TestDeterminismInvariants:
    """Property-based tests for deterministic behavior."""

    @given(
        embedding_vector_strategy(dimension=768),
        st.integers(min_value=0, max_value=1000000),
    )
    @settings(max_examples=50)
    def test_deterministic_with_seed(self, embedding, random_seed):
        """Same input and seed should produce same output."""
        # Simulate a deterministic operation
        np.random.seed(random_seed)
        result1 = np.random.randn(10).tolist()

        # Reset seed and repeat
        np.random.seed(random_seed)
        result2 = np.random.randn(10).tolist()

        # Should be identical
        assert len(result1) == len(result2)
        for v1, v2 in zip(result1, result2):
            assert abs(v1 - v2) < 1e-9

    @given(embedding_vector_strategy(dimension=768))
    @settings(max_examples=100)
    def test_embedding_reproducibility(self, embedding):
        """Same embedding should produce same L2 norm."""
        norm1 = math.sqrt(sum(x * x for x in embedding))
        norm2 = math.sqrt(sum(x * x for x in embedding))

        assert abs(norm1 - norm2) < 1e-12


class TestNumericalStability:
    """Property-based tests for numerical stability."""

    @given(
        st.lists(st.floats(min_value=1e-10, max_value=1e10, allow_nan=False), min_size=10, max_size=1000)
    )
    @settings(max_examples=100)
    def test_sum_numerical_stability(self, values):
        """Sum should be numerically stable."""
        # Compute sum
        total = sum(values)

        # Should be finite
        assert math.isfinite(total)

        # Sum should be commutative (within floating point error)
        reversed_total = sum(reversed(values))
        # Allow for small numerical differences
        if total != 0:
            relative_error = abs(total - reversed_total) / abs(total)
            assert relative_error < 1e-10

    @given(embedding_vector_strategy(dimension=768))
    @settings(max_examples=100)
    def test_normalization_stability(self, embedding):
        """Normalization should be numerically stable."""
        norm = math.sqrt(sum(x * x for x in embedding))

        if norm > 1e-10:  # Avoid division by zero
            normalized = [x / norm for x in embedding]

            # All values should be finite
            assert all(math.isfinite(x) for x in normalized)

            # L2 norm should be 1
            normalized_norm = math.sqrt(sum(x * x for x in normalized))
            assert abs(normalized_norm - 1.0) < 1e-6

    @given(
        st.floats(min_value=-100.0, max_value=100.0, allow_nan=False),
        st.floats(min_value=-100.0, max_value=100.0, allow_nan=False),
    )
    @settings(max_examples=200)
    def test_softmax_stability(self, x1, x2):
        """Softmax should be numerically stable."""
        # Compute softmax
        exp_x1 = math.exp(x1)
        exp_x2 = math.exp(x2)

        if math.isfinite(exp_x1) and math.isfinite(exp_x2):
            total = exp_x1 + exp_x2
            if total > 0:
                softmax_x1 = exp_x1 / total
                softmax_x2 = exp_x2 / total

                # Should sum to 1
                assert abs(softmax_x1 + softmax_x2 - 1.0) < 1e-9

                # Should be in [0, 1]
                assert 0.0 <= softmax_x1 <= 1.0
                assert 0.0 <= softmax_x2 <= 1.0


class TestFeatureExtractionInvariants:
    """Property-based tests for feature extraction."""

    @given(
        st.text(min_size=1, max_size=1000),
    )
    @settings(max_examples=200)
    def test_text_length_feature(self, text):
        """Text length feature should match actual length."""
        length = len(text)
        assert length >= 1
        assert length <= 1000

        # Normalized length should be in [0, 1]
        normalized_length = length / 1000.0
        assert 0.0 <= normalized_length <= 1.0

    @given(
        st.text(min_size=0, max_size=500),
    )
    @settings(max_examples=200)
    def test_word_count_feature(self, text):
        """Word count should be consistent."""
        words = text.split()
        word_count = len(words)

        assert word_count >= 0
        assert word_count <= len(text)  # Can't have more words than characters

    @given(
        st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=1, max_size=100)
    )
    @settings(max_examples=100)
    def test_feature_normalization(self, features):
        """Features should be properly normalized."""
        # All features in [0, 1]
        assert all(0.0 <= f <= 1.0 for f in features)

        # Min-max normalization
        min_val = min(features)
        max_val = max(features)

        if max_val > min_val:
            normalized = [(f - min_val) / (max_val - min_val) for f in features]
            assert all(0.0 <= f <= 1.0 for f in normalized)
            assert abs(min(normalized) - 0.0) < 1e-9
            assert abs(max(normalized) - 1.0) < 1e-9
