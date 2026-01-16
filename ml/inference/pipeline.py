"""
BIBLOS v2 - Inference Pipeline

Main inference pipeline for cross-reference discovery and verse analysis.
Uses centralized schemas for system-wide uniformity and comprehensive
OpenTelemetry distributed tracing for flame graph visualization.
"""
import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
from enum import Enum

import torch
import numpy as np
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

# Import centralized schemas
from data.schemas import (
    ConnectionType,
    ConnectionStrength,
    InferenceCandidateSchema,
    InferenceResultSchema,
    CrossReferenceSchema,
    validate_verse_id,
    normalize_verse_id,
    validate_connection_type
)

# Import mutual transformation metric
from ml.metrics.mutual_transformation import (
    MutualTransformationMetric,
    MutualTransformationScore,
    TransformationType,
    MutualTransformationConfig,
)

# Import core error types for specific exception handling
from core.errors import (
    BiblosError,
    BiblosMLError,
    BiblosValidationError,
    BiblosResourceError,
    BiblosTimeoutError,
)

# Import observability
from observability import get_tracer, get_logger
from observability.metrics import (
    record_ml_inference_duration,
    record_crossref_discovered,
    record_cache_access,
    timed_ml_inference,
)
from observability.logging import MLLogger

# Get module-level tracer and logger
tracer = get_tracer(__name__)
logger = get_logger(__name__)
ml_logger = MLLogger()


class InferenceMode(Enum):
    """Inference execution modes."""
    FAST = "fast"           # Quick inference, lower accuracy
    BALANCED = "balanced"   # Balance speed and accuracy
    ACCURATE = "accurate"   # Maximum accuracy, slower


@dataclass
class InferenceConfig:
    """Configuration for inference pipeline."""
    mode: InferenceMode = InferenceMode.BALANCED
    batch_size: int = 32
    max_candidates: int = 100
    # INFALLIBILITY: The seraph accepts ONLY absolute certainty (1.0)
    # Uncertainty cannot propagate - the seraph inherits from itself
    min_confidence: float = 1.0
    use_cache: bool = True
    use_ensemble: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    timeout_seconds: int = 60
    enable_tracing: bool = True
    # Cache size limits to prevent memory exhaustion
    max_embedding_cache_size: int = 10000
    max_text_cache_size: int = 5000


@dataclass
class CrossReferenceCandidate:
    """
    A potential cross-reference candidate.

    Aligned with InferenceCandidateSchema for system-wide uniformity.
    """
    source_verse: str
    target_verse: str
    connection_type: str
    confidence: float
    embedding_similarity: float
    semantic_similarity: float
    features: Dict[str, Any] = field(default_factory=dict)  # Changed to Any for mixed value types
    evidence: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate and normalize verse references."""
        if self.source_verse:
            self.source_verse = normalize_verse_id(self.source_verse)
        if self.target_verse:
            self.target_verse = normalize_verse_id(self.target_verse)
        # Validate connection type
        if not validate_connection_type(self.connection_type):
            self.connection_type = "thematic"  # Default fallback

    def to_schema(self) -> InferenceCandidateSchema:
        """Convert to InferenceCandidateSchema."""
        return InferenceCandidateSchema(
            source_verse=self.source_verse,
            target_verse=self.target_verse,
            connection_type=self.connection_type,
            confidence=self.confidence,
            embedding_similarity=self.embedding_similarity,
            semantic_similarity=self.semantic_similarity,
            features=self.features,
            evidence=self.evidence
        )

    def to_crossref_schema(self) -> CrossReferenceSchema:
        """Convert to CrossReferenceSchema for storage."""
        # INFALLIBILITY: Only one acceptable strength - ABSOLUTE (1.0)
        # Everything else is rejected, not "moderate" or "weak"
        if self.confidence >= 0.9999:
            strength = "absolute"
        else:
            # Rejected - but still track for diagnostics
            strength = "rejected"

        return CrossReferenceSchema(
            source_ref=self.source_verse,
            target_ref=self.target_verse,
            connection_type=self.connection_type,
            strength=strength,
            confidence=self.confidence,
            notes=self.evidence,
            sources=["ml_inference"],
            verified=False,
            patristic_support=any("patristic" in e.lower() for e in self.evidence)
        )


@dataclass
class InferenceResult:
    """
    Result of inference pipeline execution.

    Aligned with InferenceResultSchema for system-wide uniformity.
    """
    verse_id: str
    candidates: List[CrossReferenceCandidate]
    embeddings: Optional[np.ndarray]
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None

    def __post_init__(self):
        """Validate and normalize verse_id."""
        if self.verse_id:
            self.verse_id = normalize_verse_id(self.verse_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verse_id": self.verse_id,
            "candidates": [
                {
                    "source": c.source_verse,
                    "target": c.target_verse,
                    "type": c.connection_type,
                    "confidence": c.confidence,
                    "features": c.features
                }
                for c in self.candidates
            ],
            "processing_time": self.processing_time,
            "metadata": self.metadata,
            "trace_id": self.trace_id
        }

    def to_schema(self) -> InferenceResultSchema:
        """Convert to InferenceResultSchema."""
        return InferenceResultSchema(
            verse_id=self.verse_id,
            candidates=[c.to_schema() for c in self.candidates],
            processing_time=self.processing_time,
            metadata=self.metadata
        )

    def get_crossref_schemas(self) -> List[CrossReferenceSchema]:
        """Get list of CrossReferenceSchemas for all candidates."""
        return [c.to_crossref_schema() for c in self.candidates]


def get_current_trace_id() -> Optional[str]:
    """Get current trace ID as hex string."""
    span = trace.get_current_span()
    if span and span.is_recording():
        ctx = span.get_span_context()
        if ctx.is_valid:
            return format(ctx.trace_id, "032x")
    return None


class InferencePipeline:
    """
    Main inference pipeline for cross-reference discovery.

    Combines embedding-based similarity with GNN-based classification
    to identify and rank cross-reference candidates.

    Includes comprehensive OpenTelemetry tracing for:
    - Embedding generation timing
    - Similarity search performance
    - Classification duration
    - GNN refinement overhead
    """

    CONNECTION_TYPES = [
        "thematic",
        "verbal",
        "conceptual",
        "historical",
        "typological",
        "prophetic",
        "liturgical",
        "narrative",
        "genealogical",
        "geographical"
    ]

    def __init__(self, config: Optional[InferenceConfig] = None):
        self.config = config or InferenceConfig()
        self._embedding_model = None
        self._classifier_model = None
        self._gnn_model = None
        self._verse_index = None
        # Use OrderedDict for LRU eviction with bounded size
        self._embedding_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._text_cache: OrderedDict[str, np.ndarray] = OrderedDict()  # For semantic similarity
        self._initialized = False

        # Mutual transformation metric for measuring bidirectional semantic shift
        self._mutual_transformation_metric = MutualTransformationMetric()
        # Cache for embeddings before GNN refinement
        self._pre_gnn_embeddings: Dict[str, np.ndarray] = {}

    def _cache_embedding(self, key: str, embedding: np.ndarray) -> None:
        """Add embedding to cache with LRU eviction."""
        if key in self._embedding_cache:
            # Move to end (most recently used)
            self._embedding_cache.move_to_end(key)
            return

        # Evict oldest entries if at capacity
        while len(self._embedding_cache) >= self.config.max_embedding_cache_size:
            self._embedding_cache.popitem(last=False)

        self._embedding_cache[key] = embedding

    def _get_cached_embedding(self, key: str) -> Optional[np.ndarray]:
        """Get embedding from cache, updating LRU order."""
        if key in self._embedding_cache:
            self._embedding_cache.move_to_end(key)
            return self._embedding_cache[key]
        return None

    def _cache_text_embedding(self, text: str, embedding: np.ndarray) -> None:
        """Add text embedding to semantic similarity cache with LRU eviction."""
        # Use hash for text key to handle long texts
        import hashlib
        key = hashlib.sha256(text.encode()).hexdigest()[:32]

        if key in self._text_cache:
            self._text_cache.move_to_end(key)
            return

        while len(self._text_cache) >= self.config.max_text_cache_size:
            self._text_cache.popitem(last=False)

        self._text_cache[key] = embedding

    def _get_cached_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get text embedding from semantic cache."""
        import hashlib
        key = hashlib.sha256(text.encode()).hexdigest()[:32]

        if key in self._text_cache:
            self._text_cache.move_to_end(key)
            return self._text_cache[key]
        return None

    async def initialize(self) -> None:
        """Initialize inference components with tracing."""
        with tracer.start_as_current_span(
            "ml.inference.initialize",
            kind=SpanKind.INTERNAL,
        ) as span:
            logger.info("Initializing inference pipeline")
            span.set_attribute("config.mode", self.config.mode.value)
            span.set_attribute("config.device", self.config.device)

            try:
                # Load embedding model
                with tracer.start_as_current_span("ml.inference.load_embedding"):
                    await self._load_embedding_model()

                # Load classifier
                with tracer.start_as_current_span("ml.inference.load_classifier"):
                    await self._load_classifier()

                # Load GNN model
                with tracer.start_as_current_span("ml.inference.load_gnn"):
                    await self._load_gnn_model()

                # Build verse index
                with tracer.start_as_current_span("ml.inference.build_index"):
                    await self._build_verse_index()

                self._initialized = True
                span.set_attribute("status", "success")
                logger.info("Inference pipeline initialized")

            except asyncio.TimeoutError as e:
                span.set_status(Status(StatusCode.ERROR, "Initialization timeout"))
                span.set_attribute("error.type", "timeout")
                logger.error("Inference pipeline initialization timed out")
                raise BiblosTimeoutError("Inference pipeline initialization timed out")
            except (MemoryError, BiblosResourceError) as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                span.set_attribute("error.type", "resource")
                logger.critical(f"Resource exhaustion during initialization: {e}")
                raise
            except BiblosMLError as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                span.set_attribute("error.type", "ml")
                logger.error(f"ML error during initialization: {e}")
                raise
            except BiblosError as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                span.set_attribute("error.type", "biblos")
                logger.error(f"BIBLOS error during initialization: {e}")
                raise
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                span.set_attribute("error.type", "unexpected")
                logger.error(f"Unexpected error initializing inference pipeline: {e} ({type(e).__name__})")
                raise BiblosMLError(f"Failed to initialize inference pipeline: {e}")

    async def _load_embedding_model(self) -> None:
        """Load the embedding model."""
        try:
            from ml.embeddings.ensemble import EmbeddingEnsemble
            start_time = datetime.now(timezone.utc).timestamp()
            self._embedding_model = EmbeddingEnsemble()
            await self._embedding_model.initialize()
            load_time = datetime.now(timezone.utc).timestamp() - start_time
            ml_logger.model_loaded("embedding_ensemble", load_time)
            logger.info("Embedding model loaded", load_time_seconds=load_time)
        except (MemoryError, BiblosResourceError) as e:
            logger.critical(f"Resource exhaustion loading embedding model: {e}")
            raise
        except BiblosMLError as e:
            logger.warning(f"ML error loading embedding model: {e}")
        except ImportError as e:
            logger.warning(f"Failed to import embedding model: {e}")
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e} ({type(e).__name__})")

    async def _load_classifier(self) -> None:
        """Load the cross-reference classifier."""
        try:
            from ml.models.classifier import CrossReferenceClassifier
            start_time = datetime.now(timezone.utc).timestamp()
            self._classifier_model = CrossReferenceClassifier()
            load_time = datetime.now(timezone.utc).timestamp() - start_time
            ml_logger.model_loaded("classifier", load_time)
            logger.info("Classifier loaded", load_time_seconds=load_time)
        except (MemoryError, BiblosResourceError) as e:
            logger.critical(f"Resource exhaustion loading classifier: {e}")
            raise
        except BiblosMLError as e:
            logger.warning(f"ML error loading classifier: {e}")
        except ImportError as e:
            logger.warning(f"Failed to import classifier: {e}")
        except Exception as e:
            logger.warning(f"Failed to load classifier: {e} ({type(e).__name__})")

    async def _load_gnn_model(self) -> None:
        """Load the GNN discovery model."""
        try:
            from ml.models.gnn_discovery import CrossReferenceGNN
            start_time = datetime.now(timezone.utc).timestamp()
            self._gnn_model = CrossReferenceGNN()
            load_time = datetime.now(timezone.utc).timestamp() - start_time
            ml_logger.model_loaded("gnn", load_time)
            logger.info("GNN model loaded", load_time_seconds=load_time)
        except (MemoryError, BiblosResourceError) as e:
            logger.critical(f"Resource exhaustion loading GNN model: {e}")
            raise
        except BiblosMLError as e:
            logger.warning(f"ML error loading GNN model: {e}")
        except ImportError as e:
            logger.warning(f"Failed to import GNN model: {e}")
        except Exception as e:
            logger.warning(f"Failed to load GNN model: {e} ({type(e).__name__})")

    async def _build_verse_index(self) -> None:
        """Build index for fast verse lookup."""
        # Initialize empty index - populated during inference
        self._verse_index = {}
        logger.info("Verse index initialized")

    async def infer(
        self,
        verse_id: str,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> InferenceResult:
        """
        Run inference for a single verse with comprehensive tracing.

        Creates a parent span for the entire inference operation, with
        child spans for each stage:
        - Embedding generation
        - Candidate finding
        - Classification
        - GNN refinement
        - Filtering and ranking

        Args:
            verse_id: Verse identifier (e.g., "GEN.1.1")
            text: Verse text
            context: Optional context from previous pipeline stages

        Returns:
            InferenceResult with cross-reference candidates
        """
        if not self._initialized:
            await self.initialize()

        # Create parent span for entire inference
        with tracer.start_as_current_span(
            "ml.inference.infer",
            kind=SpanKind.INTERNAL,
        ) as inference_span:
            inference_span.set_attribute("verse.id", verse_id)
            inference_span.set_attribute("verse.text_length", len(text))
            inference_span.set_attribute("config.mode", self.config.mode.value)
            inference_span.set_attribute("config.max_candidates", self.config.max_candidates)

            start_time = datetime.now(timezone.utc).timestamp()
            ml_logger.start_inference("cross_reference_discovery")

            try:
                # Get verse embedding
                with tracer.start_as_current_span(
                    "ml.inference.embed",
                    kind=SpanKind.INTERNAL,
                ) as embed_span:
                    embed_start = datetime.now(timezone.utc).timestamp()
                    embedding = await self._get_embedding(verse_id, text)
                    embed_duration = datetime.now(timezone.utc).timestamp() - embed_start

                    embed_span.set_attribute("embedding.shape", str(embedding.shape))
                    embed_span.set_attribute("embedding.cached", verse_id in self._embedding_cache)
                    embed_span.set_attribute("duration_seconds", embed_duration)

                    record_ml_inference_duration("embedding", embed_duration, "ensemble")

                # Find candidate verses
                with tracer.start_as_current_span(
                    "ml.inference.find_candidates",
                    kind=SpanKind.INTERNAL,
                ) as find_span:
                    candidates = await self._find_candidates(
                        verse_id,
                        embedding,
                        context
                    )
                    find_span.set_attribute("candidates.found", len(candidates))

                # Classify candidates
                with tracer.start_as_current_span(
                    "ml.inference.classify",
                    kind=SpanKind.INTERNAL,
                ) as classify_span:
                    classify_start = datetime.now(timezone.utc).timestamp()
                    classified = await self._classify_candidates(
                        verse_id,
                        text,
                        candidates,
                        context
                    )
                    classify_duration = datetime.now(timezone.utc).timestamp() - classify_start

                    classify_span.set_attribute("classified.count", len(classified))
                    classify_span.set_attribute("duration_seconds", classify_duration)

                    record_ml_inference_duration("classification", classify_duration, "classifier")

                # Apply GNN refinement
                if self.config.use_ensemble and self._gnn_model:
                    with tracer.start_as_current_span(
                        "ml.inference.gnn_refine",
                        kind=SpanKind.INTERNAL,
                    ) as gnn_span:
                        gnn_start = datetime.now(timezone.utc).timestamp()
                        classified = await self._refine_with_gnn(
                            verse_id,
                            classified,
                            context
                        )
                        gnn_duration = datetime.now(timezone.utc).timestamp() - gnn_start

                        gnn_span.set_attribute("gnn.refined_count", len(classified))
                        gnn_span.set_attribute("duration_seconds", gnn_duration)

                        record_ml_inference_duration("gnn_refinement", gnn_duration, "gnn")

                # Filter and sort
                with tracer.start_as_current_span(
                    "ml.inference.filter_sort",
                    kind=SpanKind.INTERNAL,
                ) as filter_span:
                    # Filter by confidence threshold
                    filtered = [
                        c for c in classified
                        if c.confidence >= self.config.min_confidence
                    ]

                    # Sort by confidence
                    filtered.sort(key=lambda x: x.confidence, reverse=True)

                    # Limit results
                    filtered = filtered[:self.config.max_candidates]

                    filter_span.set_attribute("filtered.count", len(filtered))
                    filter_span.set_attribute("threshold", self.config.min_confidence)

                # Record cross-reference discoveries
                for candidate in filtered:
                    record_crossref_discovered(
                        candidate.connection_type,
                        candidate.confidence,
                        candidate.source_verse,
                        candidate.target_verse
                    )

                processing_time = datetime.now(timezone.utc).timestamp() - start_time

                # Set final span attributes
                inference_span.set_attribute("result.candidate_count", len(filtered))
                inference_span.set_attribute("processing_time_seconds", processing_time)
                inference_span.set_status(Status(StatusCode.OK))

                ml_logger.end_inference("cross_reference_discovery", processing_time)

                return InferenceResult(
                    verse_id=verse_id,
                    candidates=filtered,
                    embeddings=embedding,
                    processing_time=processing_time,
                    metadata={
                        "mode": self.config.mode.value,
                        "total_candidates_found": len(candidates),
                        "candidates_after_filter": len(filtered)
                    },
                    trace_id=get_current_trace_id()
                )

            except Exception as e:
                processing_time = datetime.now(timezone.utc).timestamp() - start_time

                inference_span.set_status(Status(StatusCode.ERROR, str(e)))
                inference_span.record_exception(e)
                inference_span.set_attribute("processing_time_seconds", processing_time)

                logger.error(f"Inference failed for {verse_id}: {e}")

                return InferenceResult(
                    verse_id=verse_id,
                    candidates=[],
                    embeddings=None,
                    processing_time=processing_time,
                    metadata={"error": str(e)},
                    trace_id=get_current_trace_id()
                )

    async def _get_embedding(
        self,
        verse_id: str,
        text: str
    ) -> np.ndarray:
        """Get embedding for a verse, using LRU cache if available."""
        if self.config.use_cache:
            cached = self._get_cached_embedding(verse_id)
            if cached is not None:
                record_cache_access(hit=True, cache_type="embedding")
                return cached

        record_cache_access(hit=False, cache_type="embedding")

        if self._embedding_model:
            embedding = await self._embedding_model.embed(text)
            if self.config.use_cache:
                self._cache_embedding(verse_id, embedding)
            return embedding

        # Fallback to random embedding for testing
        embedding = np.random.randn(768).astype(np.float32)
        return embedding

    async def _find_candidates(
        self,
        verse_id: str,
        embedding: np.ndarray,
        context: Optional[Dict[str, Any]]
    ) -> List[Tuple[str, float]]:
        """Find candidate verses based on embedding similarity."""
        candidates = []

        # Use context-provided cross-references if available
        if context:
            agent_results = context.get("agent_results", {})
            syndesmos = agent_results.get("syndesmos", {})
            cross_refs = syndesmos.get("data", {}).get("cross_references", [])

            for ref in cross_refs:
                target = ref.get("target_ref")
                if target:
                    candidates.append((
                        target,
                        ref.get("confidence", 0.5)
                    ))

        # Add similarity-based candidates from index
        if self._verse_index and len(self._embedding_cache) > 0:
            for other_id, other_embedding in self._embedding_cache.items():
                if other_id == verse_id:
                    continue

                # Compute cosine similarity
                similarity = float(np.dot(embedding, other_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(other_embedding) + 1e-8
                ))

                if similarity > 0.3:  # Threshold
                    candidates.append((other_id, similarity))

        # Deduplicate and sort
        seen = set()
        unique_candidates = []
        for target, sim in candidates:
            if target not in seen:
                seen.add(target)
                unique_candidates.append((target, sim))

        unique_candidates.sort(key=lambda x: x[1], reverse=True)
        return unique_candidates[:self.config.max_candidates * 2]

    async def _classify_candidates(
        self,
        source_id: str,
        source_text: str,
        candidates: List[Tuple[str, float]],
        context: Optional[Dict[str, Any]]
    ) -> List[CrossReferenceCandidate]:
        """
        Classify candidate pairs into connection types.

        Optimized to pre-compute all embeddings in batch before classification
        to avoid redundant embedding calls during semantic similarity computation.
        """
        if not candidates:
            return []

        results = []

        # Phase 1: Collect all unique texts needing embeddings
        texts_to_embed: List[str] = []
        text_to_index: Dict[str, int] = {}

        # Add source text
        if source_text not in text_to_index:
            cached = self._get_cached_text_embedding(source_text)
            if cached is None:
                text_to_index[source_text] = len(texts_to_embed)
                texts_to_embed.append(source_text)

        # Collect target texts
        target_texts: List[str] = []
        for target_id, _ in candidates:
            target_text = self._get_verse_text(target_id, context)
            target_texts.append(target_text)

            if target_text not in text_to_index:
                cached = self._get_cached_text_embedding(target_text)
                if cached is None:
                    text_to_index[target_text] = len(texts_to_embed)
                    texts_to_embed.append(target_text)

        # Phase 2: Batch compute embeddings for all uncached texts
        precomputed_embeddings: Dict[str, np.ndarray] = {}

        if texts_to_embed and self._embedding_model:
            try:
                # Use batch embedding for efficiency
                batch_results = await self._embedding_model.embed_batch(texts_to_embed)

                for text, result in zip(texts_to_embed, batch_results):
                    # Handle both raw embeddings and EnsembleResult objects
                    if hasattr(result, 'fused_embedding'):
                        embedding = result.fused_embedding
                    else:
                        embedding = result

                    precomputed_embeddings[text] = embedding
                    self._cache_text_embedding(text, embedding)

            except Exception as e:
                logger.warning(f"Batch embedding failed, falling back to individual: {e}")
                # Fallback to individual embedding on batch failure
                for text in texts_to_embed:
                    try:
                        result = await self._embedding_model.embed(text)
                        if hasattr(result, 'fused_embedding'):
                            embedding = result.fused_embedding
                        else:
                            embedding = result
                        precomputed_embeddings[text] = embedding
                        self._cache_text_embedding(text, embedding)
                    except Exception as inner_e:
                        logger.warning(f"Individual embedding failed: {inner_e}")

        # Helper to get embedding (from precomputed, cache, or fallback)
        def get_text_embedding(text: str) -> Optional[np.ndarray]:
            if text in precomputed_embeddings:
                return precomputed_embeddings[text]
            cached = self._get_cached_text_embedding(text)
            if cached is not None:
                return cached
            return None

        # Phase 3: Get source embedding
        source_embedding = get_text_embedding(source_text)

        # Phase 4: Process candidates with precomputed embeddings
        for i, (target_id, similarity) in enumerate(candidates):
            target_text = target_texts[i]

            # Classify connection type (heuristic, fast)
            connection_type, type_confidence = await self._classify_type(
                source_text,
                target_text,
                context
            )

            # Compute semantic similarity using precomputed embeddings
            target_embedding = get_text_embedding(target_text)

            if source_embedding is not None and target_embedding is not None:
                semantic_sim = float(np.dot(source_embedding, target_embedding) / (
                    np.linalg.norm(source_embedding) * np.linalg.norm(target_embedding) + 1e-8
                ))
                semantic_sim = max(0.0, min(1.0, semantic_sim))
            else:
                semantic_sim = 0.5  # Default fallback

            # Combine scores
            confidence = (similarity * 0.4 + semantic_sim * 0.3 + type_confidence * 0.3)

            # Gather evidence
            evidence = self._gather_evidence(
                source_id,
                target_id,
                connection_type,
                context
            )

            results.append(CrossReferenceCandidate(
                source_verse=source_id,
                target_verse=target_id,
                connection_type=connection_type,
                confidence=confidence,
                embedding_similarity=similarity,
                semantic_similarity=semantic_sim,
                features={
                    "type_confidence": type_confidence,
                    "embedding_sim": similarity,
                    "semantic_sim": semantic_sim
                },
                evidence=evidence
            ))

        return results

    def _get_verse_text(
        self,
        verse_id: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Get verse text from context or index."""
        if context:
            texts = context.get("verse_texts", {})
            if verse_id in texts:
                return texts[verse_id]
        return f"[{verse_id}]"  # Placeholder

    async def _classify_type(
        self,
        source_text: str,
        target_text: str,
        context: Optional[Dict[str, Any]]
    ) -> Tuple[str, float]:
        """Classify the connection type between two verses."""
        if self._classifier_model:
            try:
                result = await self._classifier_model.classify(
                    source_text,
                    target_text
                )
                return result.get("type", "thematic"), result.get("confidence", 0.5)
            except Exception:
                pass

        # Fallback heuristics
        source_lower = source_text.lower()
        target_lower = target_text.lower()

        # Check for verbal connections (shared words)
        source_words = set(source_lower.split())
        target_words = set(target_lower.split())
        shared = source_words & target_words

        if len(shared) > 3:
            return "verbal", 0.7

        # Check for typological indicators
        type_words = {"type", "antitype", "shadow", "fulfillment", "foreshadow"}
        if any(w in source_lower or w in target_lower for w in type_words):
            return "typological", 0.6

        # Check for prophetic indicators
        prophetic_words = {"prophecy", "fulfilled", "foretold", "prediction"}
        if any(w in source_lower or w in target_lower for w in prophetic_words):
            return "prophetic", 0.6

        # Default to thematic
        return "thematic", 0.5

    async def _compute_semantic_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        Compute semantic similarity between two texts.

        Uses LRU text cache to avoid redundant embedding computation.
        """
        # Check cache first
        emb1 = self._get_cached_text_embedding(text1)
        emb2 = self._get_cached_text_embedding(text2)

        # Compute missing embeddings
        if self._embedding_model:
            try:
                texts_to_embed = []
                if emb1 is None:
                    texts_to_embed.append(text1)
                if emb2 is None:
                    texts_to_embed.append(text2)

                if texts_to_embed:
                    # Batch embed any missing texts
                    results = await self._embedding_model.embed_batch(texts_to_embed)

                    for text, result in zip(texts_to_embed, results):
                        if hasattr(result, 'fused_embedding'):
                            embedding = result.fused_embedding
                        else:
                            embedding = result
                        self._cache_text_embedding(text, embedding)

                        if text == text1:
                            emb1 = embedding
                        if text == text2:
                            emb2 = embedding

                if emb1 is not None and emb2 is not None:
                    similarity = float(np.dot(emb1, emb2) / (
                        np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8
                    ))
                    return max(0.0, min(1.0, similarity))

            except Exception as e:
                logger.warning(f"Semantic similarity computation failed: {e}")

        return 0.5  # Default

    def _gather_evidence(
        self,
        source_id: str,
        target_id: str,
        connection_type: str,
        context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Gather supporting evidence for a connection."""
        evidence = []

        if context:
            # Check patristic citations
            patrologos = context.get("agent_results", {}).get("patrologos", {})
            citations = patrologos.get("data", {}).get("citations", [])
            for citation in citations:
                if target_id in citation.get("references", []):
                    evidence.append(f"Patristic: {citation.get('father', 'Church Father')}")

            # Check typological connections
            typologos = context.get("agent_results", {}).get("typologos", {})
            connections = typologos.get("data", {}).get("connections", [])
            for conn in connections:
                if conn.get("antitype") == target_id:
                    evidence.append(f"Typological: {conn.get('description', '')}")

        return evidence

    async def _refine_with_gnn(
        self,
        verse_id: str,
        candidates: List[CrossReferenceCandidate],
        context: Optional[Dict[str, Any]]
    ) -> List[CrossReferenceCandidate]:
        """
        Refine candidate scores using GNN model and compute mutual transformation.

        This method now captures embeddings before and after GNN refinement
        to measure the mutual transformation between connected verses.
        """
        if not self._gnn_model or not candidates:
            return candidates

        try:
            # Build local graph for GNN
            nodes = [verse_id] + [c.target_verse for c in candidates]
            edges = [(0, i + 1) for i in range(len(candidates))]

            # Step 1: Capture embeddings BEFORE GNN refinement
            pre_gnn_embeddings: Dict[str, np.ndarray] = {}
            source_embedding = self._get_cached_embedding(verse_id)
            if source_embedding is not None:
                pre_gnn_embeddings[verse_id] = source_embedding.copy()

            for candidate in candidates:
                target_emb = self._get_cached_embedding(candidate.target_verse)
                if target_emb is not None:
                    pre_gnn_embeddings[candidate.target_verse] = target_emb.copy()

            # Step 2: Get GNN predictions (this refines the embeddings)
            gnn_scores = await self._gnn_model.predict(nodes, edges)

            # Step 3: Capture embeddings AFTER GNN refinement
            post_gnn_embeddings: Dict[str, np.ndarray] = {}
            if hasattr(self._gnn_model, 'get_all_embeddings'):
                post_gnn_embeddings = self._gnn_model.get_all_embeddings()

            # Step 4: Update candidate scores and compute mutual transformation
            for i, candidate in enumerate(candidates):
                if i < len(gnn_scores):
                    # Blend GNN score with original confidence
                    gnn_score = gnn_scores[i]
                    candidate.confidence = (
                        candidate.confidence * 0.6 + gnn_score * 0.4
                    )
                    candidate.features["gnn_score"] = float(gnn_score)

                # Compute mutual transformation if we have both before/after embeddings
                target_id = candidate.target_verse
                if (verse_id in pre_gnn_embeddings and
                    target_id in pre_gnn_embeddings and
                    verse_id in post_gnn_embeddings and
                    target_id in post_gnn_embeddings):

                    try:
                        mt_score = await self._mutual_transformation_metric.measure_transformation(
                            source_verse=verse_id,
                            target_verse=target_id,
                            source_before=pre_gnn_embeddings[verse_id],
                            source_after=post_gnn_embeddings[verse_id],
                            target_before=pre_gnn_embeddings[target_id],
                            target_after=post_gnn_embeddings[target_id],
                        )

                        # Store mutual transformation results in features
                        candidate.features["mutual_influence"] = mt_score.mutual_influence
                        candidate.features["source_shift"] = mt_score.source_shift
                        candidate.features["target_shift"] = mt_score.target_shift
                        candidate.features["transformation_type"] = mt_score.transformation_type.value
                        candidate.features["directionality"] = mt_score.directionality

                        # Boost confidence for high mutual influence (RADICAL connections)
                        if mt_score.transformation_type == TransformationType.RADICAL:
                            candidate.confidence = min(1.0, candidate.confidence * 1.15)
                        elif mt_score.transformation_type == TransformationType.MODERATE:
                            candidate.confidence = min(1.0, candidate.confidence * 1.05)

                        logger.debug(
                            f"Mutual transformation {verse_id} <-> {target_id}: "
                            f"influence={mt_score.mutual_influence:.4f}, "
                            f"type={mt_score.transformation_type.value}"
                        )

                    except Exception as mt_error:
                        logger.warning(f"Mutual transformation computation failed: {mt_error}")
                        # Set default values on failure
                        candidate.features["mutual_influence"] = 0.0
                        candidate.features["transformation_type"] = "MINIMAL"

        except Exception as e:
            logger.warning(f"GNN refinement failed: {e}")

        return candidates

    async def infer_batch(
        self,
        verses: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None
    ) -> List[InferenceResult]:
        """Run inference on a batch of verses with tracing."""
        with tracer.start_as_current_span(
            "ml.inference.infer_batch",
            kind=SpanKind.INTERNAL,
        ) as batch_span:
            batch_span.set_attribute("batch.size", len(verses))
            batch_span.set_attribute("batch.batch_size", self.config.batch_size)

            results = []

            for i in range(0, len(verses), self.config.batch_size):
                batch = verses[i:i + self.config.batch_size]

                with tracer.start_as_current_span(
                    f"ml.inference.batch_chunk_{i // self.config.batch_size}"
                ) as chunk_span:
                    chunk_span.set_attribute("chunk.size", len(batch))

                    # Process batch in parallel
                    tasks = [
                        self.infer(v["verse_id"], v["text"], context)
                        for v in batch
                    ]
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                    for j, result in enumerate(batch_results):
                        if isinstance(result, Exception):
                            logger.error(f"Batch inference failed: {result}")
                            results.append(InferenceResult(
                                verse_id=batch[j]["verse_id"],
                                candidates=[],
                                embeddings=None,
                                processing_time=0,
                                metadata={"error": str(result)},
                                trace_id=get_current_trace_id()
                            ))
                        else:
                            results.append(result)

            # Set batch completion attributes
            successful = sum(1 for r in results if r.candidates)
            batch_span.set_attribute("batch.successful", successful)
            batch_span.set_attribute("batch.failed", len(results) - successful)

            return results

    async def cleanup(self) -> None:
        """Cleanup inference resources."""
        with tracer.start_as_current_span(
            "ml.inference.cleanup",
            kind=SpanKind.INTERNAL,
        ):
            self._embedding_cache.clear()
            self._text_cache.clear()
            self._verse_index = None
            self._initialized = False
            logger.info("Inference pipeline cleaned up")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        return {
            "embedding_cache_size": len(self._embedding_cache),
            "embedding_cache_max": self.config.max_embedding_cache_size,
            "text_cache_size": len(self._text_cache),
            "text_cache_max": self.config.max_text_cache_size,
        }
