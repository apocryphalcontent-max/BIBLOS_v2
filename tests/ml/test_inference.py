"""
Tests for ML inference pipeline.
"""
import pytest
import numpy as np


class TestInferenceConfig:
    """Tests for InferenceConfig."""

    def test_default_config(self):
        """Test default configuration."""
        from ml.inference.pipeline import InferenceConfig, InferenceMode

        config = InferenceConfig()
        assert config.mode == InferenceMode.BALANCED
        assert config.batch_size == 32
        assert config.max_candidates == 100
        assert config.min_confidence == 0.5
        assert config.use_cache is True

    def test_custom_config(self):
        """Test custom configuration."""
        from ml.inference.pipeline import InferenceConfig, InferenceMode

        config = InferenceConfig(
            mode=InferenceMode.ACCURATE,
            batch_size=64,
            min_confidence=0.7
        )
        assert config.mode == InferenceMode.ACCURATE
        assert config.batch_size == 64
        assert config.min_confidence == 0.7


class TestCrossReferenceCandidate:
    """Tests for CrossReferenceCandidate."""

    def test_candidate_creation(self):
        """Test creating a candidate."""
        from ml.inference.pipeline import CrossReferenceCandidate

        candidate = CrossReferenceCandidate(
            source_verse="GEN.1.1",
            target_verse="JHN.1.1",
            connection_type="typological",
            confidence=0.85,
            embedding_similarity=0.9,
            semantic_similarity=0.8
        )

        assert candidate.source_verse == "GEN.1.1"
        assert candidate.target_verse == "JHN.1.1"
        assert candidate.connection_type == "typological"
        assert candidate.confidence == 0.85


class TestInferenceResult:
    """Tests for InferenceResult."""

    def test_result_creation(self):
        """Test creating a result."""
        from ml.inference.pipeline import InferenceResult

        result = InferenceResult(
            verse_id="GEN.1.1",
            candidates=[],
            embeddings=np.zeros(768),
            processing_time=1.5
        )

        assert result.verse_id == "GEN.1.1"
        assert result.processing_time == 1.5
        assert len(result.candidates) == 0

    def test_result_to_dict(self):
        """Test result serialization."""
        from ml.inference.pipeline import InferenceResult, CrossReferenceCandidate

        candidate = CrossReferenceCandidate(
            source_verse="GEN.1.1",
            target_verse="JHN.1.1",
            connection_type="typological",
            confidence=0.85,
            embedding_similarity=0.9,
            semantic_similarity=0.8
        )

        result = InferenceResult(
            verse_id="GEN.1.1",
            candidates=[candidate],
            embeddings=None,
            processing_time=1.5
        )

        d = result.to_dict()
        assert d["verse_id"] == "GEN.1.1"
        assert d["processing_time"] == 1.5
        assert len(d["candidates"]) == 1


class TestInferencePipeline:
    """Tests for InferencePipeline."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test pipeline initialization."""
        from ml.inference.pipeline import InferencePipeline

        pipeline = InferencePipeline()
        await pipeline.initialize()

        assert pipeline._initialized is True

        await pipeline.cleanup()

    @pytest.mark.asyncio
    async def test_infer_single(self, sample_verse_id, sample_verse_text):
        """Test single verse inference."""
        from ml.inference.pipeline import InferencePipeline

        pipeline = InferencePipeline()
        await pipeline.initialize()

        result = await pipeline.infer(sample_verse_id, sample_verse_text)

        assert result.verse_id == sample_verse_id
        assert result.processing_time > 0
        assert isinstance(result.candidates, list)

        await pipeline.cleanup()

    @pytest.mark.asyncio
    async def test_infer_with_context(self, sample_verse_id, sample_verse_text, sample_theological_context):
        """Test inference with context."""
        from ml.inference.pipeline import InferencePipeline

        pipeline = InferencePipeline()
        await pipeline.initialize()

        result = await pipeline.infer(sample_verse_id, sample_verse_text, sample_theological_context)

        assert result.verse_id == sample_verse_id

        await pipeline.cleanup()

    @pytest.mark.asyncio
    async def test_infer_batch(self):
        """Test batch inference."""
        from ml.inference.pipeline import InferencePipeline

        pipeline = InferencePipeline()
        await pipeline.initialize()

        verses = [
            {"verse_id": "GEN.1.1", "text": "In the beginning"},
            {"verse_id": "GEN.1.2", "text": "And the earth was without form"}
        ]

        results = await pipeline.infer_batch(verses)

        assert len(results) == 2
        assert results[0].verse_id == "GEN.1.1"
        assert results[1].verse_id == "GEN.1.2"

        await pipeline.cleanup()

    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test pipeline cleanup."""
        from ml.inference.pipeline import InferencePipeline

        pipeline = InferencePipeline()
        await pipeline.initialize()
        await pipeline.cleanup()

        assert pipeline._initialized is False


class TestEnsembleInference:
    """Tests for EnsembleInference."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test ensemble initialization."""
        from ml.inference.ensemble import EnsembleInference

        ensemble = EnsembleInference()
        await ensemble.initialize()

        assert ensemble._initialized is True

        await ensemble.cleanup()

    def test_register_model(self):
        """Test model registration."""
        from ml.inference.ensemble import EnsembleInference

        ensemble = EnsembleInference()

        class MockModel:
            async def predict(self, *args):
                return []

        ensemble.register_model("test_model", MockModel(), weight=0.5)

        assert "test_model" in ensemble._models
        assert ensemble._model_weights["test_model"] == 0.5


class TestResultPostprocessor:
    """Tests for ResultPostprocessor."""

    def test_process_empty(self):
        """Test processing empty results."""
        from ml.inference.postprocessor import ResultPostprocessor

        processor = ResultPostprocessor()
        results = processor.process([])

        assert results == []

    def test_process_with_results(self):
        """Test processing results."""
        from ml.inference.postprocessor import ResultPostprocessor

        processor = ResultPostprocessor()

        results = [
            {
                "source_verse": "GEN.1.1",
                "target_verse": "JHN.1.1",
                "connection_type": "typological",
                "confidence": 0.85
            }
        ]

        processed = processor.process(results)

        assert len(processed) == 1
        assert "strength" in processed[0]

    def test_deduplication(self):
        """Test duplicate removal."""
        from ml.inference.postprocessor import ResultPostprocessor

        processor = ResultPostprocessor()

        results = [
            {"source_verse": "GEN.1.1", "target_verse": "JHN.1.1", "confidence": 0.85},
            {"source_verse": "GEN.1.1", "target_verse": "JHN.1.1", "confidence": 0.80}  # Duplicate
        ]

        processed = processor.process(results)

        assert len(processed) == 1

    def test_filter_novel(self):
        """Test filtering to novel cross-references."""
        from ml.inference.postprocessor import ResultPostprocessor

        processor = ResultPostprocessor()
        processor.add_known_crossref("GEN.1.1", "JHN.1.1")

        results = [
            {"source_verse": "GEN.1.1", "target_verse": "JHN.1.1", "confidence": 0.85},
            {"source_verse": "GEN.1.1", "target_verse": "REV.21.1", "confidence": 0.80}
        ]

        novel = processor.filter_novel(results)

        assert len(novel) == 1
        assert novel[0]["target_verse"] == "REV.21.1"

    def test_generate_report(self):
        """Test report generation."""
        from ml.inference.postprocessor import ResultPostprocessor

        processor = ResultPostprocessor()

        results = [
            {"connection_type": "typological", "strength": "strong", "confidence": 0.9},
            {"connection_type": "verbal", "strength": "moderate", "confidence": 0.7}
        ]

        report = processor.generate_report(results)

        assert report["total"] == 2
        assert "by_type" in report
        assert "by_strength" in report
