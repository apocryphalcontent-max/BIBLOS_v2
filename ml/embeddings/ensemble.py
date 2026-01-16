"""
BIBLOS v2 - Ensemble Embedding System

Combines multiple embedding models for robust semantic representation
of biblical texts across Greek, Hebrew, and English.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import asyncio
import hashlib
import logging
import time
import re

try:
    from sentence_transformers import SentenceTransformer
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    SentenceTransformer = None  # type: ignore


def _to_numpy(embedding: Any) -> np.ndarray:
    """Convert embedding to numpy array, handling Tensor or ndarray input."""
    if isinstance(embedding, np.ndarray):
        return embedding
    # Handle torch Tensor - use try/except for runtime safety
    if TORCH_AVAILABLE:
        try:
            # Attempt Tensor conversion (works for torch.Tensor)
            return embedding.cpu().numpy()
        except (AttributeError, TypeError):
            pass
    # Fallback: try to convert via np.asarray
    return np.asarray(embedding)


@dataclass
class EnsembleResult:
    """Combined result from ensemble embedding."""
    text: str
    embeddings: Dict[str, np.ndarray]  # model_name -> embedding
    fused_embedding: np.ndarray
    weights: Dict[str, float]
    detected_language: str
    processing_time_ms: float


class EmbeddingCache:
    """LRU cache for embeddings with disk persistence."""

    def __init__(
        self,
        cache_dir: Path,
        max_memory_items: int = 10000
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_memory_items = max_memory_items
        self._memory_cache: Dict[str, np.ndarray] = {}
        self._access_order: List[str] = []
        self.logger = logging.getLogger("biblos.ml.embeddings.cache")
        self._hits = 0
        self._misses = 0

    def _make_key(self, text: str, model: str) -> str:
        """Create cache key from text and model."""
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def get(self, text: str, model: str) -> Optional[np.ndarray]:
        """Get embedding from cache."""
        key = self._make_key(text, model)

        # Check memory cache
        if key in self._memory_cache:
            self._access_order.remove(key)
            self._access_order.append(key)
            self._hits += 1
            return self._memory_cache[key]

        # Check disk cache
        cache_file = self.cache_dir / f"{key}.npy"
        if cache_file.exists():
            try:
                embedding = np.load(cache_file)
                self._add_to_memory(key, embedding)
                self._hits += 1
                return embedding
            except Exception as e:
                self.logger.warning(f"Failed to load cached embedding: {e}")

        self._misses += 1
        return None

    def put(self, text: str, model: str, embedding: np.ndarray) -> None:
        """Store embedding in cache."""
        key = self._make_key(text, model)

        # Save to disk
        cache_file = self.cache_dir / f"{key}.npy"
        try:
            np.save(cache_file, embedding)
        except Exception as e:
            self.logger.warning(f"Failed to save embedding to disk: {e}")

        # Add to memory
        self._add_to_memory(key, embedding)

    def _add_to_memory(self, key: str, embedding: np.ndarray) -> None:
        """Add to memory cache with LRU eviction."""
        if key in self._memory_cache:
            self._access_order.remove(key)
        elif len(self._memory_cache) >= self.max_memory_items:
            oldest = self._access_order.pop(0)
            del self._memory_cache[oldest]

        self._memory_cache[key] = embedding
        self._access_order.append(key)

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0


class LanguageDetector:
    """Detect language of biblical text."""

    # Unicode ranges
    GREEK_RANGE = re.compile(r'[\u0370-\u03FF\u1F00-\u1FFF]')
    HEBREW_RANGE = re.compile(r'[\u0590-\u05FF]')
    SYRIAC_RANGE = re.compile(r'[\u0700-\u074F]')
    COPTIC_RANGE = re.compile(r'[\u2C80-\u2CFF]')

    @classmethod
    def detect(cls, text: str) -> str:
        """Detect primary language of text."""
        greek_count = len(cls.GREEK_RANGE.findall(text))
        hebrew_count = len(cls.HEBREW_RANGE.findall(text))
        syriac_count = len(cls.SYRIAC_RANGE.findall(text))
        coptic_count = len(cls.COPTIC_RANGE.findall(text))
        latin_count = len(re.findall(r'[a-zA-Z]', text))

        counts = {
            'grc': greek_count,
            'hbo': hebrew_count,
            'syc': syriac_count,
            'cop': coptic_count,
            'en': latin_count
        }

        if max(counts.values()) == 0:
            return 'en'

        return max(counts.keys(), key=lambda k: counts[k])


class EnsembleEmbedder:
    """
    Multi-model ensemble embedding system for biblical texts.

    Models:
    - MPNet: General purpose English (768 dims)
    - MiniLM: Fast fallback (384 dims)
    - Greek: Fine-tuned for ancient Greek (768 dims)
    - Hebrew: Fine-tuned for biblical Hebrew (768 dims)
    """

    MODEL_CONFIGS = {
        "mpnet": {
            "model_id": "sentence-transformers/all-mpnet-base-v2",
            "dimensions": 768,
            "languages": ["en", "mixed"],
            "base_weight": 0.35
        },
        "minilm": {
            "model_id": "sentence-transformers/all-MiniLM-L6-v2",
            "dimensions": 384,
            "languages": ["en"],
            "base_weight": 0.15
        },
        "greek": {
            "model_id": "nlpaueb/bert-base-greek-uncased-v1",
            "dimensions": 768,
            "languages": ["grc", "el"],
            "base_weight": 0.25
        },
        "hebrew": {
            "model_id": "imvladikon/sentence-transformers-alephbert",
            "dimensions": 768,
            "languages": ["he", "hbo"],
            "base_weight": 0.25
        }
    }

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        device: str = "cpu",
        load_models: Optional[List[str]] = None
    ):
        self.device = device
        self.logger = logging.getLogger("biblos.ml.embeddings.ensemble")
        self._models: Dict[str, Any] = {}  # SentenceTransformer instances
        self._cache = EmbeddingCache(
            cache_dir or Path("data/embeddings_cache"),
            max_memory_items=50000
        )

        if TORCH_AVAILABLE:
            models_to_load = load_models or ["mpnet", "minilm"]
            for model_key in models_to_load:
                self._load_model(model_key)

    def _load_model(self, model_key: str) -> None:
        """Load a model by key."""
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch not available, skipping model load")
            return

        if model_key in self._models:
            return

        config = self.MODEL_CONFIGS.get(model_key)
        if not config:
            self.logger.error(f"Unknown model: {model_key}")
            return

        try:
            self.logger.info(f"Loading model: {model_key}")
            if SentenceTransformer is None:
                self.logger.error("SentenceTransformer not available")
                return
            model = SentenceTransformer(config["model_id"], device=self.device)
            self._models[model_key] = model
            self.logger.info(f"Loaded {model_key} ({config['dimensions']} dims)")
        except Exception as e:
            self.logger.error(f"Failed to load {model_key}: {e}")

    async def embed(self, text: str, use_cache: bool = True) -> EnsembleResult:
        """
        Generate ensemble embedding for text.

        Args:
            text: Input text
            use_cache: Whether to use/update cache

        Returns:
            EnsembleResult with embeddings from all models
        """
        start_time = time.time()

        # Detect language
        language = LanguageDetector.detect(text)

        # Compute embeddings from each model
        embeddings: Dict[str, np.ndarray] = {}
        weights: Dict[str, float] = {}

        # Separate cached vs uncached models
        models_to_compute: List[str] = []
        for model_key in self._models.keys():
            config = self.MODEL_CONFIGS[model_key]
            if use_cache:
                cached = self._cache.get(text, model_key)
                if cached is not None:
                    embeddings[model_key] = cached
                    weights[model_key] = self._compute_weight(config, language)
                    continue
            models_to_compute.append(model_key)

        # Compute uncached embeddings in parallel using asyncio.to_thread
        if models_to_compute:
            async def compute_single(model_key: str) -> Tuple[str, Optional[np.ndarray]]:
                model = self._models[model_key]
                try:
                    # Run blocking encode() in thread pool
                    result = await asyncio.to_thread(
                        model.encode, text, convert_to_numpy=True
                    )
                    # Convert to numpy array (handles Tensor or ndarray)
                    embedding = _to_numpy(result)
                    return (model_key, embedding)
                except Exception as e:
                    self.logger.error(f"Embedding failed for {model_key}: {e}")
                    return (model_key, None)

            # Execute all model encodings concurrently
            tasks = [compute_single(mk) for mk in models_to_compute]
            results = await asyncio.gather(*tasks)

            for model_key, embedding in results:
                if embedding is not None:
                    config = self.MODEL_CONFIGS[model_key]
                    embeddings[model_key] = embedding
                    weights[model_key] = self._compute_weight(config, language)
                    if use_cache:
                        self._cache.put(text, model_key, embedding)

        # Fuse embeddings
        fused = self._fuse_embeddings(embeddings, weights)

        return EnsembleResult(
            text=text,
            embeddings=embeddings,
            fused_embedding=fused,
            weights=weights,
            detected_language=language,
            processing_time_ms=(time.time() - start_time) * 1000
        )

    def _compute_weight(self, config: Dict, language: str) -> float:
        """Compute model weight based on detected language."""
        base_weight = config["base_weight"]

        # Boost weight if model supports detected language
        if language in config["languages"]:
            return base_weight * 1.5

        return base_weight

    def _fuse_embeddings(
        self,
        embeddings: Dict[str, np.ndarray],
        weights: Dict[str, float]
    ) -> np.ndarray:
        """Fuse embeddings using weighted average after normalization."""
        if not embeddings:
            return np.zeros(768)

        # Normalize weights
        total_weight = sum(weights.values())
        norm_weights = {k: v / total_weight for k, v in weights.items()}

        # Project all to same dimension (768) and fuse
        target_dim = 768
        fused = np.zeros(target_dim)

        for model_key, embedding in embeddings.items():
            weight = norm_weights.get(model_key, 0)
            if len(embedding) == target_dim:
                fused += weight * embedding
            else:
                # Simple zero-padding for smaller embeddings
                padded = np.zeros(target_dim)
                padded[:len(embedding)] = embedding
                fused += weight * padded

        # Normalize final embedding
        norm = np.linalg.norm(fused)
        if norm > 0:
            fused = fused / norm

        return fused

    async def embed_batch(
        self,
        texts: List[str],
        use_cache: bool = True,
        batch_size: int = 32
    ) -> List[EnsembleResult]:
        """
        Embed a batch of texts with parallel processing.

        Uses native batch encoding for each model when possible,
        falling back to concurrent individual embeddings.

        Args:
            texts: List of texts to embed
            use_cache: Whether to use/update cache
            batch_size: Sub-batch size for processing (default 32)

        Returns:
            List of EnsembleResults in same order as input
        """
        if not texts:
            return []

        start_time = time.time()

        # Detect languages for all texts (fast operation)
        languages = [LanguageDetector.detect(text) for text in texts]

        # Separate texts into cached vs uncached per model
        all_embeddings: List[Dict[str, np.ndarray]] = [{} for _ in texts]
        all_weights: List[Dict[str, float]] = [{} for _ in texts]

        for model_key, model in self._models.items():
            config = self.MODEL_CONFIGS[model_key]

            # Check cache for all texts
            uncached_indices: List[int] = []
            uncached_texts: List[str] = []

            for i, text in enumerate(texts):
                if use_cache:
                    cached = self._cache.get(text, model_key)
                    if cached is not None:
                        all_embeddings[i][model_key] = cached
                        all_weights[i][model_key] = self._compute_weight(
                            config, languages[i]
                        )
                        continue
                uncached_indices.append(i)
                uncached_texts.append(text)

            # Batch encode uncached texts using model's native batch support
            if uncached_texts:
                try:
                    # Process in sub-batches to manage memory
                    for batch_start in range(0, len(uncached_texts), batch_size):
                        batch_end = min(batch_start + batch_size, len(uncached_texts))
                        batch_texts = uncached_texts[batch_start:batch_end]
                        batch_indices = uncached_indices[batch_start:batch_end]

                        # Run blocking batch encode in thread pool
                        batch_embeddings = await asyncio.to_thread(
                            model.encode,
                            batch_texts,
                            convert_to_numpy=True,
                            batch_size=min(batch_size, len(batch_texts))
                        )

                        # Store results with proper numpy conversion
                        for j, idx in enumerate(batch_indices):
                            embedding = _to_numpy(batch_embeddings[j])
                            all_embeddings[idx][model_key] = embedding
                            all_weights[idx][model_key] = self._compute_weight(
                                config, languages[idx]
                            )
                            if use_cache:
                                self._cache.put(texts[idx], model_key, embedding)

                except Exception as e:
                    self.logger.error(f"Batch embedding failed for {model_key}: {e}")
                    # Fall back to individual processing for failed batch
                    for idx, text in zip(uncached_indices, uncached_texts):
                        if model_key not in all_embeddings[idx]:
                            try:
                                result = await asyncio.to_thread(
                                    model.encode, text, convert_to_numpy=True
                                )
                                embedding = _to_numpy(result)
                                all_embeddings[idx][model_key] = embedding
                                all_weights[idx][model_key] = self._compute_weight(
                                    config, languages[idx]
                                )
                                if use_cache:
                                    self._cache.put(text, model_key, embedding)
                            except Exception as inner_e:
                                self.logger.error(
                                    f"Individual fallback failed for {model_key}: {inner_e}"
                                )

        # Build results
        total_time_ms = (time.time() - start_time) * 1000
        per_text_time = total_time_ms / len(texts) if texts else 0

        results = []
        for i, text in enumerate(texts):
            fused = self._fuse_embeddings(all_embeddings[i], all_weights[i])
            results.append(EnsembleResult(
                text=text,
                embeddings=all_embeddings[i],
                fused_embedding=fused,
                weights=all_weights[i],
                detected_language=languages[i],
                processing_time_ms=per_text_time
            ))

        return results

    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings."""
        dot = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "hit_rate": self._cache.hit_rate,
            "memory_items": len(self._cache._memory_cache),
            "total_hits": self._cache._hits,
            "total_misses": self._cache._misses
        }
