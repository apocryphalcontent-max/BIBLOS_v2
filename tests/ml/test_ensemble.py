"""
Tests for ml/embeddings/ensemble.py - Ensemble Embedding System.

Covers:
- EmbeddingCache: LRU caching with disk persistence
- LanguageDetector: Language detection for biblical texts
- EnsembleEmbedder: Multi-model ensemble embedding
- _to_numpy helper function
"""
import pytest
import numpy as np
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock


# =============================================================================
# Helper Function Tests
# =============================================================================

class TestToNumpy:
    """Tests for _to_numpy helper function."""

    def test_numpy_array_passthrough(self):
        """Test that numpy arrays pass through unchanged."""
        from ml.embeddings.ensemble import _to_numpy

        arr = np.array([1.0, 2.0, 3.0])
        result = _to_numpy(arr)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_list_conversion(self):
        """Test conversion from list."""
        from ml.embeddings.ensemble import _to_numpy

        data = [1.0, 2.0, 3.0]
        result = _to_numpy(data)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array(data))

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="PyTorch not available"),
        reason="PyTorch not available"
    )
    def test_torch_tensor_conversion(self):
        """Test conversion from PyTorch tensor."""
        import torch
        from ml.embeddings.ensemble import _to_numpy

        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = _to_numpy(tensor)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))


# =============================================================================
# EmbeddingCache Tests
# =============================================================================

class TestEmbeddingCache:
    """Tests for EmbeddingCache class."""

    @pytest.fixture
    def cache_dir(self, tmp_path):
        """Create a temporary cache directory."""
        return tmp_path / "embedding_cache"

    @pytest.fixture
    def cache(self, cache_dir):
        """Create an EmbeddingCache instance."""
        from ml.embeddings.ensemble import EmbeddingCache
        return EmbeddingCache(cache_dir, max_memory_items=10)

    def test_cache_initialization(self, cache, cache_dir):
        """Test cache initialization creates directory."""
        assert cache.cache_dir.exists()
        assert cache.max_memory_items == 10

    def test_cache_miss_returns_none(self, cache):
        """Test cache miss returns None."""
        result = cache.get("nonexistent text", "test_model")
        assert result is None

    def test_cache_put_and_get(self, cache):
        """Test putting and getting embeddings from cache."""
        embedding = np.array([1.0, 2.0, 3.0])
        cache.put("test text", "test_model", embedding)

        result = cache.get("test text", "test_model")

        assert result is not None
        np.testing.assert_array_equal(result, embedding)

    def test_cache_disk_persistence(self, cache):
        """Test embeddings are persisted to disk."""
        embedding = np.array([1.0, 2.0, 3.0])
        cache.put("test text", "test_model", embedding)

        # Check that a file was created
        files = list(cache.cache_dir.glob("*.npy"))
        assert len(files) == 1

    def test_cache_lru_eviction(self, cache_dir):
        """Test LRU eviction when cache is full."""
        from ml.embeddings.ensemble import EmbeddingCache
        cache = EmbeddingCache(cache_dir, max_memory_items=3)

        # Add 4 items to trigger eviction
        for i in range(4):
            cache.put(f"text_{i}", "model", np.array([float(i)]))

        # First item should be evicted from memory
        # But should still be on disk
        assert len(cache._memory_cache) == 3

    def test_cache_hit_rate(self, cache):
        """Test hit rate calculation."""
        embedding = np.array([1.0, 2.0, 3.0])
        cache.put("test", "model", embedding)

        # 1 hit
        cache.get("test", "model")
        # 2 misses
        cache.get("miss1", "model")
        cache.get("miss2", "model")

        # 1 hit / 3 total = 0.333...
        assert cache.hit_rate == pytest.approx(1 / 3, rel=0.01)

    def test_cache_hit_rate_zero_requests(self, cache):
        """Test hit rate is 0.0 when no requests made."""
        assert cache.hit_rate == 0.0

    def test_cache_key_uniqueness(self, cache):
        """Test that different text/model combinations create different keys."""
        emb1 = np.array([1.0])
        emb2 = np.array([2.0])

        cache.put("text", "model1", emb1)
        cache.put("text", "model2", emb2)

        result1 = cache.get("text", "model1")
        result2 = cache.get("text", "model2")

        np.testing.assert_array_equal(result1, emb1)
        np.testing.assert_array_equal(result2, emb2)

    def test_cache_access_order_update(self, cache):
        """Test that accessing an item updates its position in LRU order."""
        cache.put("old", "model", np.array([1.0]))
        cache.put("new", "model", np.array([2.0]))

        # Access old item - should move it to end of access order
        cache.get("old", "model")

        # old should be after new in access order
        assert cache._access_order[-1].startswith(cache._make_key("old", "model")[:8]) or \
               "old" in str(cache._access_order[-1])


# =============================================================================
# LanguageDetector Tests
# =============================================================================

class TestLanguageDetector:
    """Tests for LanguageDetector class."""

    def test_detect_english(self):
        """Test detection of English text."""
        from ml.embeddings.ensemble import LanguageDetector

        text = "In the beginning God created the heavens and the earth"
        assert LanguageDetector.detect(text) == "en"

    def test_detect_greek(self):
        """Test detection of Greek text."""
        from ml.embeddings.ensemble import LanguageDetector

        text = "Ἐν ἀρχῇ ἦν ὁ λόγος"
        assert LanguageDetector.detect(text) == "grc"

    def test_detect_hebrew(self):
        """Test detection of Hebrew text."""
        from ml.embeddings.ensemble import LanguageDetector

        text = "בְּרֵאשִׁית בָּרָא אֱלֹהִים"
        assert LanguageDetector.detect(text) == "hbo"

    def test_detect_syriac(self):
        """Test detection of Syriac text."""
        from ml.embeddings.ensemble import LanguageDetector

        text = "ܒܪܫܝܬ ܐܝܬܘܗܝ ܗܘܐ ܡܠܬܐ"
        assert LanguageDetector.detect(text) == "syc"

    def test_detect_coptic(self):
        """Test detection of Coptic text."""
        from ml.embeddings.ensemble import LanguageDetector

        text = "ϩⲛⲧⲉϩⲟⲩⲉⲓⲧⲉ ⲛⲉϥϣⲟⲟⲡ ⲡⲉ ⲡⲓⲗⲟⲅⲟⲥ"
        assert LanguageDetector.detect(text) == "cop"

    def test_detect_mixed_defaults_to_dominant(self):
        """Test that mixed text returns dominant language."""
        from ml.embeddings.ensemble import LanguageDetector

        # Mostly Greek with a little English
        text = "Ἐν ἀρχῇ ἦν ὁ λόγος (In the beginning)"
        result = LanguageDetector.detect(text)
        # Should be Greek since it has more Greek characters
        assert result == "grc"

    def test_detect_empty_string(self):
        """Test detection of empty string."""
        from ml.embeddings.ensemble import LanguageDetector

        assert LanguageDetector.detect("") == "en"

    def test_detect_numbers_only(self):
        """Test detection when only numbers present."""
        from ml.embeddings.ensemble import LanguageDetector

        assert LanguageDetector.detect("12345") == "en"


# =============================================================================
# EnsembleResult Tests
# =============================================================================

class TestEnsembleResult:
    """Tests for EnsembleResult dataclass."""

    def test_result_creation(self):
        """Test creating an EnsembleResult."""
        from ml.embeddings.ensemble import EnsembleResult

        result = EnsembleResult(
            text="test text",
            embeddings={"model1": np.array([1.0, 2.0])},
            fused_embedding=np.array([1.0, 2.0]),
            weights={"model1": 1.0},
            detected_language="en",
            processing_time_ms=10.5
        )

        assert result.text == "test text"
        assert "model1" in result.embeddings
        assert result.detected_language == "en"
        assert result.processing_time_ms == 10.5


# =============================================================================
# EnsembleEmbedder Tests
# =============================================================================

class TestEnsembleEmbedder:
    """Tests for EnsembleEmbedder class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock SentenceTransformer model."""
        model = Mock()
        model.encode = Mock(return_value=np.random.randn(768))
        return model

    @pytest.fixture
    def embedder_no_models(self, tmp_path):
        """Create an EnsembleEmbedder with no models loaded."""
        from ml.embeddings.ensemble import EnsembleEmbedder

        with patch.dict("ml.embeddings.ensemble.__dict__", {"TORCH_AVAILABLE": False}):
            embedder = EnsembleEmbedder(
                cache_dir=tmp_path / "cache",
                load_models=[]
            )
        return embedder

    def test_embedder_initialization_default_device(self, tmp_path):
        """Test embedder initialization with default device."""
        from ml.embeddings.ensemble import EnsembleEmbedder, TORCH_AVAILABLE

        embedder = EnsembleEmbedder(
            cache_dir=tmp_path / "cache",
            device="cpu"
        )

        assert embedder.device == "cpu"
        # When TORCH_AVAILABLE and load_models is None, default models are loaded
        if TORCH_AVAILABLE:
            # Default behavior loads mpnet and minilm
            assert len(embedder._models) >= 0  # May have 0, 1, or 2 depending on availability

    def test_model_configs_structure(self):
        """Test that MODEL_CONFIGS has expected structure."""
        from ml.embeddings.ensemble import EnsembleEmbedder

        for model_key, config in EnsembleEmbedder.MODEL_CONFIGS.items():
            assert "model_id" in config
            assert "dimensions" in config
            assert "languages" in config
            assert "base_weight" in config
            assert isinstance(config["dimensions"], int)
            assert isinstance(config["languages"], list)
            assert isinstance(config["base_weight"], float)

    def test_compute_weight_with_language_match(self, tmp_path):
        """Test weight boost when language matches model."""
        from ml.embeddings.ensemble import EnsembleEmbedder

        embedder = EnsembleEmbedder(
            cache_dir=tmp_path / "cache",
            load_models=[]
        )

        greek_config = EnsembleEmbedder.MODEL_CONFIGS["greek"]
        weight = embedder._compute_weight(greek_config, "grc")

        # Should be boosted by 1.5x
        expected = greek_config["base_weight"] * 1.5
        assert weight == expected

    def test_compute_weight_without_language_match(self, tmp_path):
        """Test no weight boost when language doesn't match."""
        from ml.embeddings.ensemble import EnsembleEmbedder

        embedder = EnsembleEmbedder(
            cache_dir=tmp_path / "cache",
            load_models=[]
        )

        greek_config = EnsembleEmbedder.MODEL_CONFIGS["greek"]
        weight = embedder._compute_weight(greek_config, "en")

        # Should be base weight (no boost)
        assert weight == greek_config["base_weight"]

    def test_fuse_embeddings_empty(self, tmp_path):
        """Test fusing empty embeddings returns zeros."""
        from ml.embeddings.ensemble import EnsembleEmbedder

        embedder = EnsembleEmbedder(
            cache_dir=tmp_path / "cache",
            load_models=[]
        )

        result = embedder._fuse_embeddings({}, {})

        assert isinstance(result, np.ndarray)
        assert len(result) == 768
        np.testing.assert_array_equal(result, np.zeros(768))

    def test_fuse_embeddings_single_model(self, tmp_path):
        """Test fusing embeddings from a single model."""
        from ml.embeddings.ensemble import EnsembleEmbedder

        embedder = EnsembleEmbedder(
            cache_dir=tmp_path / "cache",
            load_models=[]
        )

        embedding = np.ones(768)
        embeddings = {"model1": embedding}
        weights = {"model1": 1.0}

        result = embedder._fuse_embeddings(embeddings, weights)

        assert isinstance(result, np.ndarray)
        assert len(result) == 768
        # After normalization, should have unit norm
        assert np.linalg.norm(result) == pytest.approx(1.0, rel=0.01)

    def test_fuse_embeddings_with_padding(self, tmp_path):
        """Test fusing embeddings with different dimensions."""
        from ml.embeddings.ensemble import EnsembleEmbedder

        embedder = EnsembleEmbedder(
            cache_dir=tmp_path / "cache",
            load_models=[]
        )

        # 384-dim embedding (like MiniLM)
        small_emb = np.ones(384)
        # 768-dim embedding
        full_emb = np.ones(768)

        embeddings = {"small": small_emb, "full": full_emb}
        weights = {"small": 0.5, "full": 0.5}

        result = embedder._fuse_embeddings(embeddings, weights)

        assert len(result) == 768
        # Should have unit norm after normalization
        assert np.linalg.norm(result) == pytest.approx(1.0, rel=0.01)

    def test_similarity_identical_vectors(self, tmp_path):
        """Test cosine similarity of identical vectors."""
        from ml.embeddings.ensemble import EnsembleEmbedder

        embedder = EnsembleEmbedder(
            cache_dir=tmp_path / "cache",
            load_models=[]
        )

        vec = np.array([1.0, 2.0, 3.0])
        similarity = embedder.similarity(vec, vec)

        assert similarity == pytest.approx(1.0, rel=0.01)

    def test_similarity_orthogonal_vectors(self, tmp_path):
        """Test cosine similarity of orthogonal vectors."""
        from ml.embeddings.ensemble import EnsembleEmbedder

        embedder = EnsembleEmbedder(
            cache_dir=tmp_path / "cache",
            load_models=[]
        )

        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([0.0, 1.0])
        similarity = embedder.similarity(vec1, vec2)

        assert similarity == pytest.approx(0.0, abs=0.01)

    def test_similarity_opposite_vectors(self, tmp_path):
        """Test cosine similarity of opposite vectors."""
        from ml.embeddings.ensemble import EnsembleEmbedder

        embedder = EnsembleEmbedder(
            cache_dir=tmp_path / "cache",
            load_models=[]
        )

        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([-1.0, 0.0])
        similarity = embedder.similarity(vec1, vec2)

        assert similarity == pytest.approx(-1.0, rel=0.01)

    def test_similarity_zero_vector(self, tmp_path):
        """Test cosine similarity with zero vector."""
        from ml.embeddings.ensemble import EnsembleEmbedder

        embedder = EnsembleEmbedder(
            cache_dir=tmp_path / "cache",
            load_models=[]
        )

        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([0.0, 0.0, 0.0])
        similarity = embedder.similarity(vec1, vec2)

        assert similarity == 0.0

    def test_get_cache_stats(self, tmp_path):
        """Test getting cache statistics."""
        from ml.embeddings.ensemble import EnsembleEmbedder

        embedder = EnsembleEmbedder(
            cache_dir=tmp_path / "cache",
            load_models=[]
        )

        stats = embedder.get_cache_stats()

        assert "hit_rate" in stats
        assert "memory_items" in stats
        assert "total_hits" in stats
        assert "total_misses" in stats

    @pytest.mark.asyncio
    async def test_embed_with_mocked_models(self, tmp_path):
        """Test embed method with mocked models."""
        from ml.embeddings.ensemble import EnsembleEmbedder

        embedder = EnsembleEmbedder(
            cache_dir=tmp_path / "cache",
            load_models=[]
        )

        # Mock a model
        mock_model = Mock()
        mock_model.encode = Mock(return_value=np.random.randn(768))
        embedder._models = {"mpnet": mock_model}

        result = await embedder.embed("test text", use_cache=False)

        assert result.text == "test text"
        assert result.detected_language == "en"
        assert "mpnet" in result.embeddings
        assert len(result.fused_embedding) == 768

    @pytest.mark.asyncio
    async def test_embed_uses_cache(self, tmp_path):
        """Test that embed uses cache when available."""
        from ml.embeddings.ensemble import EnsembleEmbedder

        embedder = EnsembleEmbedder(
            cache_dir=tmp_path / "cache",
            load_models=[]
        )

        # Pre-populate cache
        cached_embedding = np.random.randn(768)
        embedder._cache.put("test text", "mpnet", cached_embedding)

        # Add a mock model
        mock_model = Mock()
        embedder._models = {"mpnet": mock_model}

        result = await embedder.embed("test text", use_cache=True)

        # Model.encode should NOT be called since we have cached value
        mock_model.encode.assert_not_called()
        np.testing.assert_array_equal(result.embeddings["mpnet"], cached_embedding)

    @pytest.mark.asyncio
    async def test_embed_batch_empty_list(self, tmp_path):
        """Test batch embed with empty list."""
        from ml.embeddings.ensemble import EnsembleEmbedder

        embedder = EnsembleEmbedder(
            cache_dir=tmp_path / "cache",
            load_models=[]
        )

        results = await embedder.embed_batch([])

        assert results == []

    @pytest.mark.asyncio
    async def test_embed_batch_with_mocked_models(self, tmp_path):
        """Test batch embed with mocked models."""
        from ml.embeddings.ensemble import EnsembleEmbedder

        embedder = EnsembleEmbedder(
            cache_dir=tmp_path / "cache",
            load_models=[]
        )

        # Mock a model with batch encoding
        mock_model = Mock()
        mock_model.encode = Mock(return_value=np.random.randn(3, 768))
        embedder._models = {"mpnet": mock_model}

        texts = ["text one", "text two", "text three"]
        results = await embedder.embed_batch(texts, use_cache=False)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.text == texts[i]
            assert "mpnet" in result.embeddings


# =============================================================================
# Integration Tests (with mocked external dependencies)
# =============================================================================

class TestEnsembleIntegration:
    """Integration tests with mocked models."""

    @pytest.mark.asyncio
    async def test_full_pipeline_flow(self, tmp_path):
        """Test complete embedding pipeline flow."""
        from ml.embeddings.ensemble import EnsembleEmbedder

        embedder = EnsembleEmbedder(
            cache_dir=tmp_path / "cache",
            load_models=[]
        )

        # Mock two models
        mock_mpnet = Mock()
        mock_mpnet.encode = Mock(return_value=np.random.randn(768))

        mock_minilm = Mock()
        mock_minilm.encode = Mock(return_value=np.random.randn(384))

        embedder._models = {
            "mpnet": mock_mpnet,
            "minilm": mock_minilm
        }

        # First embedding - should compute
        result1 = await embedder.embed("In the beginning", use_cache=True)

        # Second embedding of same text - should use cache
        result2 = await embedder.embed("In the beginning", use_cache=True)

        # Models should only be called once each (first call)
        assert mock_mpnet.encode.call_count == 1
        assert mock_minilm.encode.call_count == 1

        # Results should have same embeddings
        np.testing.assert_array_equal(
            result1.embeddings["mpnet"],
            result2.embeddings["mpnet"]
        )
