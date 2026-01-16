"""
Multi-Vector Store for BIBLOS v2

Manages multiple domain-specific embedding spaces:
- Semantic: General meaning and context (768-dim, all-mpnet-base-v2)
- Typological: Type/antitype relationships (384-dim, all-MiniLM-L6-v2)
- Prophetic: Prophecy fulfillment patterns (384-dim, all-MiniLM-L6-v2)
- Patristic: Church Father interpretations (768-dim, all-mpnet-base-v2)
- Liturgical: Worship and liturgical usage (384-dim, all-MiniLM-L6-v2)
- Covenantal: Covenant structure and progression (256-dim, fine-tuned)

Each domain has its own embedding model and vector collection.
Supports weighted hybrid search with multiple aggregation strategies:
- Weighted Sum: Linear combination of domain scores
- Max Aggregation: Maximum score across domains
- RRF (Reciprocal Rank Fusion): Rank-based combination resistant to score normalization issues
"""
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
from datetime import datetime, timezone

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
        Range,
        HasIdCondition,
        IsEmptyCondition,
        PayloadField,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


logger = logging.getLogger(__name__)


class EmbeddingDomain(Enum):
    """
    Domain-specific embedding spaces for biblical scholarship.

    Each domain captures a distinct semantic dimension:
    - SEMANTIC: General meaning and contextual similarity
    - TYPOLOGICAL: OT type / NT antitype structural patterns
    - PROPHETIC: Prophecy-fulfillment connections
    - PATRISTIC: Church Father interpretative traditions
    - LITURGICAL: Worship and feast day associations
    - COVENANTAL: Covenant structure, roles, and progression
    """
    SEMANTIC = "semantic"
    TYPOLOGICAL = "typological"
    PROPHETIC = "prophetic"
    PATRISTIC = "patristic"
    LITURGICAL = "liturgical"
    COVENANTAL = "covenantal"

    @property
    def dimension(self) -> int:
        """Embedding dimension for this domain."""
        return {
            EmbeddingDomain.SEMANTIC: 768,      # all-mpnet-base-v2
            EmbeddingDomain.TYPOLOGICAL: 384,   # all-MiniLM-L6-v2
            EmbeddingDomain.PROPHETIC: 384,     # all-MiniLM-L6-v2
            EmbeddingDomain.PATRISTIC: 768,     # all-mpnet-base-v2
            EmbeddingDomain.LITURGICAL: 384,    # all-MiniLM-L6-v2
            EmbeddingDomain.COVENANTAL: 256,    # Custom fine-tuned model
        }[self]

    @property
    def model_name(self) -> str:
        """SentenceTransformer model for this domain."""
        return {
            EmbeddingDomain.SEMANTIC: "sentence-transformers/all-mpnet-base-v2",
            EmbeddingDomain.TYPOLOGICAL: "sentence-transformers/all-MiniLM-L6-v2",
            EmbeddingDomain.PROPHETIC: "sentence-transformers/all-MiniLM-L6-v2",
            EmbeddingDomain.PATRISTIC: "sentence-transformers/all-mpnet-base-v2",
            EmbeddingDomain.LITURGICAL: "sentence-transformers/all-MiniLM-L6-v2",
            EmbeddingDomain.COVENANTAL: "sentence-transformers/all-MiniLM-L6-v2",
        }[self]

    @property
    def collection_name(self) -> str:
        """Qdrant collection name."""
        return f"biblos_{self.value}"

    @property
    def description(self) -> str:
        """Human-readable description of this domain."""
        return {
            EmbeddingDomain.SEMANTIC: "General semantic meaning and context",
            EmbeddingDomain.TYPOLOGICAL: "Type/antitype structural patterns between testaments",
            EmbeddingDomain.PROPHETIC: "Prophecy and fulfillment connections",
            EmbeddingDomain.PATRISTIC: "Church Father interpretative traditions",
            EmbeddingDomain.LITURGICAL: "Liturgical usage and feast associations",
            EmbeddingDomain.COVENANTAL: "Covenant structure, roles, and progression",
        }[self]


class AggregationMethod(Enum):
    """Methods for combining scores across domains in hybrid search."""
    WEIGHTED_SUM = "weighted_sum"
    MAX = "max"
    RRF = "rrf"  # Reciprocal Rank Fusion


@dataclass
class DomainWeight:
    """Weight configuration for domain in hybrid search."""
    domain: EmbeddingDomain
    weight: float = 1.0
    min_score: float = 0.0
    boost_factor: float = 1.0  # Additional boost for specific domains


@dataclass
class SearchResult:
    """Result from vector search with multi-domain scoring."""
    verse_id: str
    score: float
    domain_scores: Dict[str, float] = field(default_factory=dict)
    domain_ranks: Dict[str, int] = field(default_factory=dict)
    payload: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""  # Human-readable explanation of why this result matched


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search behavior."""
    aggregation_method: AggregationMethod = AggregationMethod.WEIGHTED_SUM
    rrf_k: int = 60  # RRF constant (typically 60)
    min_domains_matched: int = 1  # Require results to appear in at least N domains
    score_normalization: bool = True  # Normalize scores to [0,1] before combining
    diversity_penalty: float = 0.0  # Penalize similar results (MMR-style)
    explain: bool = False  # Generate explanation strings


@dataclass
class CovenantMetadata:
    """Metadata for covenantal domain embeddings."""
    covenant_name: Optional[str] = None  # "Adamic", "Noahic", "Abrahamic", "Mosaic", "Davidic", "New"
    covenant_role: Optional[str] = None  # "initiation", "promise", "condition", "sign", "fulfillment", "renewal"
    covenant_parties: Optional[List[str]] = None
    covenant_signs: Optional[List[str]] = None
    progression_index: Optional[int] = None  # Order in covenant progression


class MultiVectorStore:
    """
    Multi-domain vector store using Qdrant.

    Manages separate collections for each embedding domain,
    enabling domain-specific and sophisticated hybrid searches.

    Features:
    - Six embedding domains optimized for biblical scholarship
    - Multiple aggregation methods (weighted sum, max, RRF)
    - Score normalization and diversity ranking
    - Event-driven updates via VectorStoreProjection
    - Batch operations for efficient indexing
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        prefer_grpc: bool = True,
        default_config: Optional[HybridSearchConfig] = None
    ):
        """
        Initialize multi-vector store.

        Args:
            host: Qdrant host
            port: Qdrant port
            api_key: Optional API key for cloud
            prefer_grpc: Use gRPC for better performance
            default_config: Default hybrid search configuration
        """
        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant-client not installed. Install with: pip install qdrant-client")

        self.client = QdrantClient(
            host=host,
            port=port,
            api_key=api_key,
            prefer_grpc=prefer_grpc
        )
        self._embedding_models: Dict[EmbeddingDomain, Any] = {}
        self.default_config = default_config or HybridSearchConfig()
        logger.info(f"Initialized MultiVectorStore at {host}:{port} with {len(EmbeddingDomain)} domains")

    async def create_collections(self) -> None:
        """
        Create all domain-specific collections.

        Each collection is optimized for its embedding dimension
        and uses cosine distance for similarity.
        """
        for domain in EmbeddingDomain:
            try:
                self.client.create_collection(
                    collection_name=domain.collection_name,
                    vectors_config=VectorParams(
                        size=domain.dimension,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {domain.collection_name} ({domain.dimension}-dim)")
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.info(f"Collection {domain.collection_name} already exists")
                else:
                    raise

    async def upsert_verse(
        self,
        verse_id: str,
        embeddings: Dict[EmbeddingDomain, np.ndarray],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Insert or update verse embeddings across all domains.

        Args:
            verse_id: Verse identifier (e.g., "GEN.1.1")
            embeddings: Dict mapping domains to embedding vectors
            metadata: Additional metadata to store
        """
        metadata = metadata or {}
        metadata['verse_id'] = verse_id
        metadata['updated_at'] = datetime.now(timezone.utc).isoformat()

        for domain, embedding in embeddings.items():
            if embedding is None:
                continue

            point = PointStruct(
                id=self._verse_id_to_int(verse_id),
                vector=embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                payload=metadata
            )

            self.client.upsert(
                collection_name=domain.collection_name,
                points=[point]
            )

        logger.debug(f"Upserted embeddings for {verse_id} across {len(embeddings)} domains")

    async def search(
        self,
        query_vector: np.ndarray,
        domain: EmbeddingDomain,
        top_k: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.0
    ) -> List[SearchResult]:
        """
        Search in a single domain.

        Args:
            query_vector: Query embedding vector
            domain: Domain to search
            top_k: Number of results
            filter_conditions: Optional metadata filters
            score_threshold: Minimum score threshold

        Returns:
            List of search results
        """
        query_filter = None
        if filter_conditions:
            query_filter = self._build_filter(filter_conditions)

        results = self.client.search(
            collection_name=domain.collection_name,
            query_vector=query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector,
            limit=top_k,
            query_filter=query_filter,
            score_threshold=score_threshold
        )

        return [
            SearchResult(
                verse_id=result.payload.get('verse_id', ''),
                score=result.score,
                domain_scores={domain.value: result.score},
                domain_ranks={domain.value: rank + 1},
                payload=result.payload
            )
            for rank, result in enumerate(results)
        ]

    async def hybrid_search(
        self,
        query_vectors: Dict[EmbeddingDomain, np.ndarray],
        weights: Optional[Dict[EmbeddingDomain, float]] = None,
        top_k: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
        config: Optional[HybridSearchConfig] = None
    ) -> List[SearchResult]:
        """
        Perform sophisticated hybrid search across multiple domains.

        Supports multiple aggregation methods:
        - WEIGHTED_SUM: Linear combination of normalized scores
        - MAX: Maximum score across all domains
        - RRF: Reciprocal Rank Fusion (rank-based, score-agnostic)

        Args:
            query_vectors: Query vectors for each domain
            weights: Optional weights for each domain (default: equal)
            top_k: Number of results
            filter_conditions: Optional metadata filters
            config: Search configuration (overrides defaults)

        Returns:
            Ranked list of search results
        """
        config = config or self.default_config

        if weights is None:
            weights = {domain: 1.0 for domain in query_vectors.keys()}

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        # Search each domain and collect results
        domain_results: Dict[str, Dict[EmbeddingDomain, float]] = {}
        domain_ranks: Dict[str, Dict[EmbeddingDomain, int]] = {}
        all_payloads: Dict[str, Dict[str, Any]] = {}

        for domain, query_vector in query_vectors.items():
            if query_vector is None:
                continue

            weight = weights.get(domain, 0.0)
            if weight == 0.0:
                continue

            # Get more candidates for re-ranking
            results = await self.search(
                query_vector=query_vector,
                domain=domain,
                top_k=top_k * 5,
                filter_conditions=filter_conditions
            )

            for rank, result in enumerate(results):
                verse_id = result.verse_id
                if verse_id not in domain_results:
                    domain_results[verse_id] = {}
                    domain_ranks[verse_id] = {}
                    all_payloads[verse_id] = result.payload

                domain_results[verse_id][domain] = result.score
                domain_ranks[verse_id][domain] = rank + 1

        # Apply minimum domains filter
        if config.min_domains_matched > 1:
            domain_results = {
                vid: scores for vid, scores in domain_results.items()
                if len(scores) >= config.min_domains_matched
            }

        # Compute hybrid scores based on aggregation method
        if config.aggregation_method == AggregationMethod.WEIGHTED_SUM:
            hybrid_results = self._aggregate_weighted_sum(
                domain_results, domain_ranks, all_payloads,
                weights, query_vectors.keys(), config
            )
        elif config.aggregation_method == AggregationMethod.MAX:
            hybrid_results = self._aggregate_max(
                domain_results, domain_ranks, all_payloads,
                weights, query_vectors.keys(), config
            )
        elif config.aggregation_method == AggregationMethod.RRF:
            hybrid_results = self._aggregate_rrf(
                domain_results, domain_ranks, all_payloads,
                weights, query_vectors.keys(), config
            )
        else:
            raise ValueError(f"Unknown aggregation method: {config.aggregation_method}")

        # Sort by hybrid score
        hybrid_results.sort(key=lambda x: x.score, reverse=True)

        # Apply diversity penalty if configured
        if config.diversity_penalty > 0:
            hybrid_results = self._apply_diversity_penalty(
                hybrid_results, config.diversity_penalty
            )

        return hybrid_results[:top_k]

    def _aggregate_weighted_sum(
        self,
        domain_results: Dict[str, Dict[EmbeddingDomain, float]],
        domain_ranks: Dict[str, Dict[EmbeddingDomain, int]],
        all_payloads: Dict[str, Dict[str, Any]],
        weights: Dict[EmbeddingDomain, float],
        domains: Any,
        config: HybridSearchConfig
    ) -> List[SearchResult]:
        """Aggregate using weighted sum of scores."""
        results = []

        # Optionally normalize scores per domain
        if config.score_normalization:
            normalized = self._normalize_scores(domain_results)
        else:
            normalized = domain_results

        for verse_id, scores in normalized.items():
            hybrid_score = sum(
                scores.get(domain, 0.0) * weights.get(domain, 0.0)
                for domain in domains
            )

            explanation = ""
            if config.explain:
                score_parts = [
                    f"{d.value}:{scores.get(d, 0):.3f}*{weights.get(d, 0):.2f}"
                    for d in domains if d in scores
                ]
                explanation = f"Weighted sum: {' + '.join(score_parts)} = {hybrid_score:.3f}"

            results.append(
                SearchResult(
                    verse_id=verse_id,
                    score=hybrid_score,
                    domain_scores={d.value: s for d, s in domain_results[verse_id].items()},
                    domain_ranks={d.value: r for d, r in domain_ranks.get(verse_id, {}).items()},
                    payload=all_payloads[verse_id],
                    explanation=explanation
                )
            )

        return results

    def _aggregate_max(
        self,
        domain_results: Dict[str, Dict[EmbeddingDomain, float]],
        domain_ranks: Dict[str, Dict[EmbeddingDomain, int]],
        all_payloads: Dict[str, Dict[str, Any]],
        weights: Dict[EmbeddingDomain, float],
        domains: Any,
        config: HybridSearchConfig
    ) -> List[SearchResult]:
        """Aggregate using maximum score across domains (with weight boost)."""
        results = []

        for verse_id, scores in domain_results.items():
            # Apply weight as a boost factor
            weighted_scores = [
                scores.get(domain, 0.0) * (1.0 + weights.get(domain, 0.0))
                for domain in domains
            ]
            hybrid_score = max(weighted_scores) if weighted_scores else 0.0

            explanation = ""
            if config.explain:
                best_domain = max(scores.keys(), key=lambda d: scores[d])
                explanation = f"Max from {best_domain.value}: {scores[best_domain]:.3f}"

            results.append(
                SearchResult(
                    verse_id=verse_id,
                    score=hybrid_score,
                    domain_scores={d.value: s for d, s in scores.items()},
                    domain_ranks={d.value: r for d, r in domain_ranks.get(verse_id, {}).items()},
                    payload=all_payloads[verse_id],
                    explanation=explanation
                )
            )

        return results

    def _aggregate_rrf(
        self,
        domain_results: Dict[str, Dict[EmbeddingDomain, float]],
        domain_ranks: Dict[str, Dict[EmbeddingDomain, int]],
        all_payloads: Dict[str, Dict[str, Any]],
        weights: Dict[EmbeddingDomain, float],
        domains: Any,
        config: HybridSearchConfig
    ) -> List[SearchResult]:
        """
        Aggregate using Reciprocal Rank Fusion (RRF).

        RRF Score = Σ (weight_i / (k + rank_i))

        Where k is typically 60. This method is more robust to
        score normalization issues and focuses on rank order.
        """
        results = []
        k = config.rrf_k

        for verse_id, ranks in domain_ranks.items():
            rrf_score = 0.0
            contributing_domains = []

            for domain in domains:
                if domain in ranks:
                    rank = ranks[domain]
                    weight = weights.get(domain, 1.0)
                    contribution = weight / (k + rank)
                    rrf_score += contribution
                    contributing_domains.append(f"{domain.value}:rank{rank}")

            explanation = ""
            if config.explain:
                explanation = f"RRF(k={k}): {' + '.join(contributing_domains)} = {rrf_score:.4f}"

            results.append(
                SearchResult(
                    verse_id=verse_id,
                    score=rrf_score,
                    domain_scores={d.value: s for d, s in domain_results[verse_id].items()},
                    domain_ranks={d.value: r for d, r in ranks.items()},
                    payload=all_payloads[verse_id],
                    explanation=explanation
                )
            )

        return results

    def _normalize_scores(
        self,
        domain_results: Dict[str, Dict[EmbeddingDomain, float]]
    ) -> Dict[str, Dict[EmbeddingDomain, float]]:
        """Normalize scores to [0, 1] per domain using min-max scaling."""
        # Find min/max per domain
        domain_mins: Dict[EmbeddingDomain, float] = {}
        domain_maxs: Dict[EmbeddingDomain, float] = {}

        for scores in domain_results.values():
            for domain, score in scores.items():
                if domain not in domain_mins:
                    domain_mins[domain] = score
                    domain_maxs[domain] = score
                else:
                    domain_mins[domain] = min(domain_mins[domain], score)
                    domain_maxs[domain] = max(domain_maxs[domain], score)

        # Normalize
        normalized = {}
        for verse_id, scores in domain_results.items():
            normalized[verse_id] = {}
            for domain, score in scores.items():
                min_score = domain_mins[domain]
                max_score = domain_maxs[domain]
                if max_score > min_score:
                    normalized[verse_id][domain] = (score - min_score) / (max_score - min_score)
                else:
                    normalized[verse_id][domain] = 1.0

        return normalized

    def _apply_diversity_penalty(
        self,
        results: List[SearchResult],
        penalty_factor: float
    ) -> List[SearchResult]:
        """
        Apply MMR-style diversity penalty to reduce redundancy.

        Results similar to already-selected results get penalized.
        """
        if len(results) <= 1:
            return results

        # Simple book-based diversity: penalize results from same book
        selected = []
        remaining = list(results)

        while remaining and len(selected) < len(results):
            if not selected:
                selected.append(remaining.pop(0))
                continue

            # Find result with best diversity-adjusted score
            best_idx = 0
            best_adjusted_score = -float('inf')

            for i, result in enumerate(remaining):
                # Check similarity to already selected
                result_book = result.verse_id.split('.')[0] if '.' in result.verse_id else result.verse_id
                similarity = 0.0

                for sel in selected:
                    sel_book = sel.verse_id.split('.')[0] if '.' in sel.verse_id else sel.verse_id
                    if result_book == sel_book:
                        similarity += 1.0 / len(selected)

                adjusted_score = result.score - (penalty_factor * similarity * result.score)

                if adjusted_score > best_adjusted_score:
                    best_adjusted_score = adjusted_score
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return selected

    async def get_embedding(
        self,
        domain: EmbeddingDomain,
        verse_id: str
    ) -> Optional[np.ndarray]:
        """
        Retrieve stored embedding for a verse.

        Args:
            domain: Embedding domain
            verse_id: Verse identifier

        Returns:
            Embedding vector or None if not found
        """
        try:
            point = self.client.retrieve(
                collection_name=domain.collection_name,
                ids=[self._verse_id_to_int(verse_id)]
            )

            if point:
                return np.array(point[0].vector)
            return None
        except Exception as e:
            logger.error(f"Error retrieving embedding for {verse_id}: {e}")
            return None

    async def get_similar_verses(
        self,
        verse_id: str,
        domain: EmbeddingDomain,
        top_k: int = 10,
        exclude_self: bool = True
    ) -> List[SearchResult]:
        """
        Find verses similar to a given verse in a domain.

        Args:
            verse_id: Source verse
            domain: Domain to search
            top_k: Number of results
            exclude_self: Whether to exclude the source verse

        Returns:
            Similar verses
        """
        # Get embedding for source verse
        embedding = await self.get_embedding(domain, verse_id)
        if embedding is None:
            logger.warning(f"No embedding found for {verse_id} in {domain.value}")
            return []

        # Search
        results = await self.search(
            query_vector=embedding,
            domain=domain,
            top_k=top_k + (1 if exclude_self else 0)
        )

        # Remove self if requested
        if exclude_self:
            results = [r for r in results if r.verse_id != verse_id]

        return results[:top_k]

    async def find_cross_domain_similar(
        self,
        verse_id: str,
        source_domain: EmbeddingDomain,
        target_domain: EmbeddingDomain,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Find verses in target domain that are similar in source domain.

        Useful for finding, e.g., typologically similar verses that
        share prophetic patterns.

        Args:
            verse_id: Source verse
            source_domain: Domain to get embedding from
            target_domain: Domain to search in
            top_k: Number of results

        Returns:
            Similar verses from target domain perspective
        """
        source_embedding = await self.get_embedding(source_domain, verse_id)
        if source_embedding is None:
            return []

        # Get verses similar in source domain
        source_similar = await self.search(
            query_vector=source_embedding,
            domain=source_domain,
            top_k=top_k * 3
        )

        if not source_similar:
            return []

        # Get target domain embeddings for these verses and re-rank
        target_results = []
        for result in source_similar:
            target_embedding = await self.get_embedding(target_domain, result.verse_id)
            if target_embedding is not None:
                verse_embedding = await self.get_embedding(target_domain, verse_id)
                if verse_embedding is not None:
                    similarity = float(np.dot(target_embedding, verse_embedding) /
                                      (np.linalg.norm(target_embedding) * np.linalg.norm(verse_embedding)))
                    target_results.append(
                        SearchResult(
                            verse_id=result.verse_id,
                            score=similarity,
                            domain_scores={
                                source_domain.value: result.score,
                                target_domain.value: similarity
                            },
                            payload=result.payload
                        )
                    )

        target_results.sort(key=lambda x: x.score, reverse=True)
        return target_results[:top_k]

    async def batch_upsert(
        self,
        verses: List[Tuple[str, Dict[EmbeddingDomain, np.ndarray], Dict[str, Any]]],
        batch_size: int = 100
    ) -> None:
        """
        Batch upsert multiple verses efficiently.

        Args:
            verses: List of (verse_id, embeddings, metadata) tuples
            batch_size: Number of verses per batch
        """
        # Group by domain
        domain_batches: Dict[EmbeddingDomain, List[PointStruct]] = {
            domain: [] for domain in EmbeddingDomain
        }

        for verse_id, embeddings, metadata in verses:
            metadata = metadata or {}
            metadata['verse_id'] = verse_id
            metadata['updated_at'] = datetime.now(timezone.utc).isoformat()

            for domain, embedding in embeddings.items():
                if embedding is None:
                    continue

                point = PointStruct(
                    id=self._verse_id_to_int(verse_id),
                    vector=embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                    payload=metadata
                )
                domain_batches[domain].append(point)

        # Upsert in batches per domain
        for domain, points in domain_batches.items():
            if not points:
                continue

            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=domain.collection_name,
                    points=batch
                )

            logger.info(f"Batch upserted {len(points)} points to {domain.collection_name}")

    def _verse_id_to_int(self, verse_id: str) -> int:
        """
        Convert verse ID to integer for Qdrant point ID.

        GEN.1.1 -> hash to positive integer
        Uses a stable hash to ensure consistency across sessions.
        """
        # Use a more stable hashing approach
        import hashlib
        hash_bytes = hashlib.sha256(verse_id.encode()).digest()
        return int.from_bytes(hash_bytes[:8], 'big') % (2**63)

    def _build_filter(self, conditions: Dict[str, Any]) -> Filter:
        """Build Qdrant filter from conditions."""
        field_conditions = []

        for key, value in conditions.items():
            if isinstance(value, dict):
                # Range filter
                if 'gte' in value or 'lte' in value or 'gt' in value or 'lt' in value:
                    field_conditions.append(
                        FieldCondition(
                            key=key,
                            range=Range(
                                gte=value.get('gte'),
                                lte=value.get('lte'),
                                gt=value.get('gt'),
                                lt=value.get('lt')
                            )
                        )
                    )
            elif isinstance(value, list):
                # Match any of the values
                for v in value:
                    field_conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=v)
                        )
                    )
            else:
                field_conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )

        return Filter(must=field_conditions)

    async def cleanup(self) -> None:
        """Cleanup resources."""
        # Qdrant client doesn't need explicit cleanup
        pass

    async def get_collection_info(self, domain: EmbeddingDomain) -> Dict[str, Any]:
        """
        Get information about a collection.

        Args:
            domain: Domain to query

        Returns:
            Collection info including count, dimension, etc.
        """
        info = self.client.get_collection(collection_name=domain.collection_name)
        return {
            "name": domain.collection_name,
            "domain": domain.value,
            "dimension": domain.dimension,
            "model": domain.model_name,
            "description": domain.description,
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "points_count": info.points_count,
            "status": str(info.status),
        }

    async def get_all_collection_stats(self) -> Dict[str, Any]:
        """Get statistics for all collections."""
        stats = {}
        total_vectors = 0

        for domain in EmbeddingDomain:
            try:
                info = await self.get_collection_info(domain)
                stats[domain.value] = info
                total_vectors += info.get('vectors_count', 0)
            except Exception as e:
                stats[domain.value] = {"error": str(e)}

        stats['_total'] = {
            "domains": len(EmbeddingDomain),
            "total_vectors": total_vectors
        }

        return stats

    async def delete_verse(self, verse_id: str) -> None:
        """
        Delete a verse from all domain collections.

        Args:
            verse_id: Verse identifier to delete
        """
        point_id = self._verse_id_to_int(verse_id)

        for domain in EmbeddingDomain:
            try:
                self.client.delete(
                    collection_name=domain.collection_name,
                    points_selector=[point_id]
                )
            except Exception as e:
                logger.warning(f"Error deleting {verse_id} from {domain.value}: {e}")

        logger.info(f"Deleted verse {verse_id} from all collections")
