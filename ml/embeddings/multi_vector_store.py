"""
Multi-Vector Store for BIBLOS v2

Manages multiple domain-specific embedding spaces:
- Semantic: General meaning and context
- Typological: Type/antitype relationships
- Prophetic: Prophecy fulfillment patterns
- Patristic: Church Father interpretations
- Liturgical: Worship and liturgical usage

Each domain has its own embedding model and vector collection.
"""
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
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
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


logger = logging.getLogger(__name__)


class EmbeddingDomain(Enum):
    """Domain-specific embedding spaces."""
    SEMANTIC = "semantic"
    TYPOLOGICAL = "typological"
    PROPHETIC = "prophetic"
    PATRISTIC = "patristic"
    LITURGICAL = "liturgical"

    @property
    def dimension(self) -> int:
        """Embedding dimension for this domain."""
        return {
            EmbeddingDomain.SEMANTIC: 768,      # all-mpnet-base-v2
            EmbeddingDomain.TYPOLOGICAL: 384,   # all-MiniLM-L6-v2
            EmbeddingDomain.PROPHETIC: 384,
            EmbeddingDomain.PATRISTIC: 768,
            EmbeddingDomain.LITURGICAL: 384,
        }[self]

    @property
    def model_name(self) -> str:
        """Model for this domain."""
        return {
            EmbeddingDomain.SEMANTIC: "sentence-transformers/all-mpnet-base-v2",
            EmbeddingDomain.TYPOLOGICAL: "sentence-transformers/all-MiniLM-L6-v2",
            EmbeddingDomain.PROPHETIC: "sentence-transformers/all-MiniLM-L6-v2",
            EmbeddingDomain.PATRISTIC: "sentence-transformers/all-mpnet-base-v2",
            EmbeddingDomain.LITURGICAL: "sentence-transformers/all-MiniLM-L6-v2",
        }[self]

    @property
    def collection_name(self) -> str:
        """Qdrant collection name."""
        return f"biblos_{self.value}"


@dataclass
class DomainWeight:
    """Weight configuration for domain in hybrid search."""
    domain: EmbeddingDomain
    weight: float = 1.0
    min_score: float = 0.0


@dataclass
class SearchResult:
    """Result from vector search."""
    verse_id: str
    score: float
    domain_scores: Dict[str, float] = field(default_factory=dict)
    payload: Dict[str, Any] = field(default_factory=dict)


class MultiVectorStore:
    """
    Multi-domain vector store using Qdrant.

    Manages separate collections for each embedding domain,
    enabling domain-specific and hybrid searches.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        prefer_grpc: bool = True
    ):
        """
        Initialize multi-vector store.

        Args:
            host: Qdrant host
            port: Qdrant port
            api_key: Optional API key for cloud
            prefer_grpc: Use gRPC for better performance
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
        logger.info(f"Initialized MultiVectorStore at {host}:{port}")

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
                logger.info(f"Created collection: {domain.collection_name}")
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
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search in a single domain.

        Args:
            query_vector: Query embedding vector
            domain: Domain to search
            top_k: Number of results
            filter_conditions: Optional metadata filters

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
            query_filter=query_filter
        )

        return [
            SearchResult(
                verse_id=result.payload.get('verse_id', ''),
                score=result.score,
                domain_scores={domain.value: result.score},
                payload=result.payload
            )
            for result in results
        ]

    async def hybrid_search(
        self,
        query_vectors: Dict[EmbeddingDomain, np.ndarray],
        weights: Optional[Dict[EmbeddingDomain, float]] = None,
        top_k: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Perform hybrid search across multiple domains.

        Combines scores from multiple domains using weighted sum.

        Args:
            query_vectors: Query vectors for each domain
            weights: Optional weights for each domain (default: equal)
            top_k: Number of results
            filter_conditions: Optional metadata filters

        Returns:
            Ranked list of search results
        """
        if weights is None:
            weights = {domain: 1.0 for domain in query_vectors.keys()}

        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}

        # Search each domain
        domain_results: Dict[str, Dict[EmbeddingDomain, float]] = {}
        all_payloads: Dict[str, Dict[str, Any]] = {}

        for domain, query_vector in query_vectors.items():
            if query_vector is None:
                continue

            weight = weights.get(domain, 0.0)
            if weight == 0.0:
                continue

            results = await self.search(
                query_vector=query_vector,
                domain=domain,
                top_k=top_k * 3,  # Get more candidates for re-ranking
                filter_conditions=filter_conditions
            )

            for result in results:
                verse_id = result.verse_id
                if verse_id not in domain_results:
                    domain_results[verse_id] = {}
                    all_payloads[verse_id] = result.payload

                domain_results[verse_id][domain] = result.score

        # Compute hybrid scores
        hybrid_results = []
        for verse_id, scores in domain_results.items():
            # Weighted sum
            hybrid_score = sum(
                scores.get(domain, 0.0) * weights.get(domain, 0.0)
                for domain in query_vectors.keys()
            )

            hybrid_results.append(
                SearchResult(
                    verse_id=verse_id,
                    score=hybrid_score,
                    domain_scores={d.value: s for d, s in scores.items()},
                    payload=all_payloads[verse_id]
                )
            )

        # Sort by hybrid score
        hybrid_results.sort(key=lambda x: x.score, reverse=True)

        return hybrid_results[:top_k]

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
        """
        # Simple hash - in production might use a mapping table
        return abs(hash(verse_id)) % (2**63)

    def _build_filter(self, conditions: Dict[str, Any]) -> Filter:
        """Build Qdrant filter from conditions."""
        field_conditions = []

        for key, value in conditions.items():
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
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "points_count": info.points_count,
            "status": info.status,
        }
