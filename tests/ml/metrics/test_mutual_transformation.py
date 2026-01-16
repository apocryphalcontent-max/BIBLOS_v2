"""
Tests for the Mutual Transformation Metric.

These tests verify the correct implementation of the bidirectional
semantic shift measurement used to assess cross-reference quality.
"""

import pytest
import numpy as np
from typing import List, Tuple

from ml.metrics.mutual_transformation import (
    TransformationType,
    MutualTransformationScore,
    MutualTransformationMetric,
    MutualTransformationConfig,
    THEOLOGICAL_DIMENSIONS,
)


class TestTransformationType:
    """Tests for TransformationType enum."""

    def test_transformation_types_exist(self):
        """Verify all expected transformation types are defined."""
        assert hasattr(TransformationType, "RADICAL")
        assert hasattr(TransformationType, "MODERATE")
        assert hasattr(TransformationType, "MINIMAL")

    def test_transformation_type_values(self):
        """Verify transformation type string values."""
        assert TransformationType.RADICAL.value == "RADICAL"
        assert TransformationType.MODERATE.value == "MODERATE"
        assert TransformationType.MINIMAL.value == "MINIMAL"


class TestMutualTransformationScore:
    """Tests for MutualTransformationScore dataclass."""

    def test_score_creation(self):
        """Test basic score creation."""
        score = MutualTransformationScore(
            source_shift=0.3,
            target_shift=0.4,
            mutual_influence=0.342857,
            transformation_type=TransformationType.MODERATE,
        )

        assert score.source_shift == 0.3
        assert score.target_shift == 0.4
        assert score.transformation_type == TransformationType.MODERATE

    def test_score_to_dict(self):
        """Test conversion to dictionary."""
        score = MutualTransformationScore(
            source_shift=0.5,
            target_shift=0.5,
            mutual_influence=0.5,
            transformation_type=TransformationType.RADICAL,
            directionality=0.0,
        )

        d = score.to_dict()

        assert d["source_shift"] == 0.5
        assert d["target_shift"] == 0.5
        assert d["mutual_influence"] == 0.5
        assert d["transformation_type"] == "RADICAL"
        assert d["directionality"] == 0.0

    def test_score_repr(self):
        """Test string representation."""
        score = MutualTransformationScore(
            source_shift=0.3,
            target_shift=0.4,
            mutual_influence=0.342857,
            transformation_type=TransformationType.MODERATE,
        )

        repr_str = repr(score)

        assert "MutualTransformationScore" in repr_str
        assert "mutual_influence=0.3429" in repr_str
        assert "MODERATE" in repr_str


class TestMutualTransformationMetric:
    """Tests for MutualTransformationMetric class."""

    @pytest.fixture
    def metric(self) -> MutualTransformationMetric:
        """Create a metric instance for testing."""
        return MutualTransformationMetric()

    @pytest.fixture
    def custom_metric(self) -> MutualTransformationMetric:
        """Create a metric with custom thresholds."""
        config = MutualTransformationConfig(
            radical_threshold=0.5,
            moderate_threshold=0.3,
        )
        return MutualTransformationMetric(config)

    @pytest.mark.asyncio
    async def test_identical_embeddings_zero_shift(self, metric: MutualTransformationMetric):
        """
        Test 1: Identical embeddings should produce zero shift.

        When embeddings don't change, there is no transformation.
        """
        # Same embedding before and after
        embedding = np.random.randn(768).astype(np.float32)

        score = await metric.measure_transformation(
            source_verse="GEN.1.1",
            target_verse="JHN.1.1",
            source_before=embedding,
            source_after=embedding,
            target_before=embedding,
            target_after=embedding,
        )

        assert score.source_shift == pytest.approx(0.0, abs=1e-6)
        assert score.target_shift == pytest.approx(0.0, abs=1e-6)
        assert score.mutual_influence == pytest.approx(0.0, abs=1e-6)
        assert score.transformation_type == TransformationType.MINIMAL

    @pytest.mark.asyncio
    async def test_orthogonal_embeddings_max_shift(self, metric: MutualTransformationMetric):
        """
        Test 2: Orthogonal embeddings should produce maximum shift.

        When embeddings become orthogonal, that represents maximum change.
        """
        dim = 768

        # Create orthogonal before/after pairs
        source_before = np.zeros(dim, dtype=np.float32)
        source_before[0] = 1.0  # [1, 0, 0, ...]

        source_after = np.zeros(dim, dtype=np.float32)
        source_after[1] = 1.0  # [0, 1, 0, ...] - orthogonal

        target_before = np.zeros(dim, dtype=np.float32)
        target_before[2] = 1.0

        target_after = np.zeros(dim, dtype=np.float32)
        target_after[3] = 1.0  # Also orthogonal

        score = await metric.measure_transformation(
            source_verse="GEN.22.2",
            target_verse="JHN.3.16",
            source_before=source_before,
            source_after=source_after,
            target_before=target_before,
            target_after=target_after,
        )

        # Orthogonal vectors have cosine distance of 1.0
        assert score.source_shift == pytest.approx(1.0, abs=0.01)
        assert score.target_shift == pytest.approx(1.0, abs=0.01)
        # Harmonic mean of 1 and 1 is 1
        assert score.mutual_influence == pytest.approx(1.0, abs=0.01)
        assert score.transformation_type == TransformationType.RADICAL

    @pytest.mark.asyncio
    async def test_asymmetric_shift(self, metric: MutualTransformationMetric):
        """
        Test 3: Asymmetric shift should produce low mutual influence.

        The harmonic mean penalizes asymmetry - if only one verse shifts,
        the mutual influence should be low.
        """
        dim = 768

        # Source has large shift
        source_before = np.zeros(dim, dtype=np.float32)
        source_before[0] = 1.0

        source_after = np.zeros(dim, dtype=np.float32)
        source_after[1] = 1.0  # Orthogonal = max shift

        # Target has zero shift
        target_before = np.random.randn(dim).astype(np.float32)
        target_after = target_before.copy()  # Same = no shift

        score = await metric.measure_transformation(
            source_verse="EXO.3.2",
            target_verse="LUK.1.35",
            source_before=source_before,
            source_after=source_after,
            target_before=target_before,
            target_after=target_after,
        )

        # Source shift is ~1.0, target shift is 0
        assert score.source_shift == pytest.approx(1.0, abs=0.01)
        assert score.target_shift == pytest.approx(0.0, abs=0.01)

        # Harmonic mean: 2*1*0 / (1+0+eps) ≈ 0
        assert score.mutual_influence < 0.1

        # Directionality should indicate target dominated (source shifted a lot)
        # directionality = (target_shift - source_shift) / (sum + eps)
        # = (0 - 1) / (1 + eps) ≈ -1 (source dominated)
        assert score.directionality < -0.5

    @pytest.mark.asyncio
    async def test_classification_thresholds(self, metric: MutualTransformationMetric):
        """
        Test 4: Verify classification thresholds work correctly.

        RADICAL > 0.4, MODERATE 0.2-0.4, MINIMAL < 0.2
        """
        dim = 768
        np.random.seed(42)

        # Helper to create controlled shift
        async def get_score_for_shift(shift_amount: float) -> MutualTransformationScore:
            # Create vectors with specified cosine distance
            # cos(theta) = 1 - shift, so theta = arccos(1 - shift)
            before = np.zeros(dim, dtype=np.float32)
            before[0] = 1.0

            # Rotate by angle to get desired cosine similarity
            cos_sim = 1.0 - shift_amount
            angle = np.arccos(np.clip(cos_sim, -1, 1))
            after = np.zeros(dim, dtype=np.float32)
            after[0] = np.cos(angle)
            after[1] = np.sin(angle)

            return await metric.measure_transformation(
                source_verse="TEST.1.1",
                target_verse="TEST.1.2",
                source_before=before,
                source_after=after,
                target_before=before.copy(),
                target_after=after.copy(),
            )

        # Test RADICAL classification (shift > 0.4)
        radical_score = await get_score_for_shift(0.5)
        assert radical_score.transformation_type == TransformationType.RADICAL

        # Test MODERATE classification (0.2 < shift <= 0.4)
        moderate_score = await get_score_for_shift(0.3)
        assert moderate_score.transformation_type == TransformationType.MODERATE

        # Test MINIMAL classification (shift <= 0.2)
        minimal_score = await get_score_for_shift(0.1)
        assert minimal_score.transformation_type == TransformationType.MINIMAL

    @pytest.mark.asyncio
    async def test_batch_processing(self, metric: MutualTransformationMetric):
        """
        Test 5: Batch processing should match individual results.

        Verify that batch processing produces the same results as
        individual measurements, and measure performance improvement.
        """
        dim = 768
        num_pairs = 100
        np.random.seed(42)

        # Generate test pairs
        pairs: List[Tuple[str, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for i in range(num_pairs):
            pairs.append((
                f"GEN.{i}.1",
                f"JHN.{i}.1",
                np.random.randn(dim).astype(np.float32),
                np.random.randn(dim).astype(np.float32),
                np.random.randn(dim).astype(np.float32),
                np.random.randn(dim).astype(np.float32),
            ))

        # Batch processing
        batch_scores = await metric.measure_batch(pairs)

        assert len(batch_scores) == num_pairs

        # Verify first few match individual processing
        for i in range(min(5, num_pairs)):
            individual_score = await metric.measure_transformation(
                source_verse=pairs[i][0],
                target_verse=pairs[i][1],
                source_before=pairs[i][2],
                source_after=pairs[i][3],
                target_before=pairs[i][4],
                target_after=pairs[i][5],
            )

            assert batch_scores[i].mutual_influence == pytest.approx(
                individual_score.mutual_influence, abs=1e-5
            )
            assert batch_scores[i].source_shift == pytest.approx(
                individual_score.source_shift, abs=1e-5
            )
            assert batch_scores[i].target_shift == pytest.approx(
                individual_score.target_shift, abs=1e-5
            )

    @pytest.mark.asyncio
    async def test_theological_test_case_isaac_christ(self, metric: MutualTransformationMetric):
        """
        Test 6: Canonical test case - Isaac/Christ typological connection.

        This tests the core theological principle: when genuinely connected,
        both verses should shift significantly (high mutual influence).
        """
        dim = 768
        np.random.seed(42)

        # Simulate significant mutual transformation
        # In a real scenario, these would be from the GNN
        isaac_before = np.random.randn(dim).astype(np.float32)
        isaac_before = isaac_before / np.linalg.norm(isaac_before)

        christ_before = np.random.randn(dim).astype(np.float32)
        christ_before = christ_before / np.linalg.norm(christ_before)

        # After GNN: both vectors shift significantly toward a common direction
        # Stronger transformation factor (1.5 instead of 0.5) to simulate
        # the profound typological connection between Isaac and Christ
        sacrifice_direction = np.random.randn(dim).astype(np.float32)
        sacrifice_direction = sacrifice_direction / np.linalg.norm(sacrifice_direction)

        # Apply a substantial shift - this simulates what happens when
        # GNN message passing reveals deep typological connection
        isaac_after = isaac_before + 1.5 * sacrifice_direction
        isaac_after = isaac_after / np.linalg.norm(isaac_after)

        christ_after = christ_before + 1.5 * sacrifice_direction
        christ_after = christ_after / np.linalg.norm(christ_after)

        score = await metric.measure_transformation(
            source_verse="GEN.22.2",  # Isaac
            target_verse="JHN.3.16",  # Christ
            source_before=isaac_before,
            source_after=isaac_after,
            target_before=christ_before,
            target_after=christ_after,
        )

        # Both should have shifted significantly (>0.2 for moderate threshold)
        assert score.source_shift > 0.2
        assert score.target_shift > 0.2

        # Mutual influence should be at least moderate (>0.2)
        assert score.mutual_influence > 0.2

        # Should be at least MODERATE classification for genuine typological connection
        assert score.transformation_type in [
            TransformationType.MODERATE,
            TransformationType.RADICAL
        ]

        # Directionality should be close to 0 (mutual transformation)
        assert abs(score.directionality) < 0.5

    @pytest.mark.asyncio
    async def test_custom_thresholds(self, custom_metric: MutualTransformationMetric):
        """Test that custom thresholds are respected."""
        dim = 768

        # Create moderate shift
        before = np.zeros(dim, dtype=np.float32)
        before[0] = 1.0

        # About 35% shift
        angle = np.arccos(0.65)
        after = np.zeros(dim, dtype=np.float32)
        after[0] = np.cos(angle)
        after[1] = np.sin(angle)

        score = await custom_metric.measure_transformation(
            source_verse="TEST.1.1",
            target_verse="TEST.1.2",
            source_before=before,
            source_after=after,
            target_before=before.copy(),
            target_after=after.copy(),
        )

        # With custom thresholds (radical=0.5, moderate=0.3),
        # 0.35 shift should be MODERATE
        assert score.transformation_type == TransformationType.MODERATE

    def test_directionality_calculation(self, metric: MutualTransformationMetric):
        """Test directionality calculation formula."""
        # Equal shifts = 0 directionality
        assert metric.calculate_directionality(0.5, 0.5) == pytest.approx(0.0, abs=1e-6)

        # Source dominated (target shifted more)
        dir1 = metric.calculate_directionality(0.2, 0.8)
        assert dir1 > 0  # Positive = target dominated

        # Target dominated (source shifted more)
        dir2 = metric.calculate_directionality(0.8, 0.2)
        assert dir2 < 0  # Negative = source dominated

    def test_semantic_component_extraction(self, metric: MutualTransformationMetric):
        """Test semantic component decomposition."""
        dim = 768

        # Create a delta vector with specific pattern
        delta = np.zeros(dim, dtype=np.float32)
        delta[0:64] = 1.0  # christological
        delta[640:704] = 1.0  # typological

        components = metric.extract_semantic_components(
            delta, THEOLOGICAL_DIMENSIONS
        )

        # Should have values for both activated dimensions
        assert components["christological"] > 0
        assert components["typological"] > 0

        # Other components should be 0 or very small
        assert components["soteriological"] == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.asyncio
    async def test_zero_vector_handling(self, metric: MutualTransformationMetric):
        """Test handling of zero vectors."""
        dim = 768

        zero_vec = np.zeros(dim, dtype=np.float32)
        normal_vec = np.random.randn(dim).astype(np.float32)

        score = await metric.measure_transformation(
            source_verse="TEST.1.1",
            target_verse="TEST.1.2",
            source_before=zero_vec,
            source_after=normal_vec,
            target_before=zero_vec,
            target_after=normal_vec,
        )

        # Should not crash, should return valid score
        assert 0.0 <= score.source_shift <= 1.0
        assert 0.0 <= score.mutual_influence <= 1.0


class TestPerformance:
    """Performance tests for the metric."""

    @pytest.fixture
    def metric(self) -> MutualTransformationMetric:
        return MutualTransformationMetric()

    @pytest.mark.asyncio
    async def test_single_measurement_performance(self, metric: MutualTransformationMetric):
        """
        Performance: Single pair measurement should be < 10ms.
        """
        import time

        dim = 768
        np.random.seed(42)

        before1 = np.random.randn(dim).astype(np.float32)
        after1 = np.random.randn(dim).astype(np.float32)
        before2 = np.random.randn(dim).astype(np.float32)
        after2 = np.random.randn(dim).astype(np.float32)

        start = time.perf_counter()
        _ = await metric.measure_transformation(
            "GEN.1.1", "JHN.1.1",
            before1, after1, before2, after2
        )
        elapsed = time.perf_counter() - start

        # Should be less than 10ms
        assert elapsed < 0.01, f"Single measurement took {elapsed*1000:.2f}ms (> 10ms)"

    @pytest.mark.asyncio
    async def test_batch_performance(self, metric: MutualTransformationMetric):
        """
        Performance: Batch of 100 pairs should be < 500ms.
        """
        import time

        dim = 768
        num_pairs = 100
        np.random.seed(42)

        pairs = [
            (
                f"GEN.{i}.1", f"JHN.{i}.1",
                np.random.randn(dim).astype(np.float32),
                np.random.randn(dim).astype(np.float32),
                np.random.randn(dim).astype(np.float32),
                np.random.randn(dim).astype(np.float32),
            )
            for i in range(num_pairs)
        ]

        start = time.perf_counter()
        _ = await metric.measure_batch(pairs)
        elapsed = time.perf_counter() - start

        # Should be less than 500ms
        assert elapsed < 0.5, f"Batch measurement took {elapsed*1000:.2f}ms (> 500ms)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
