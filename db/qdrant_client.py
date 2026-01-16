"""
BIBLOS v2 - Qdrant Vector Database Client

High-performance vector similarity search for biblical text embeddings.
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import os
import uuid

try:
    from qdrant_client import QdrantClient, AsyncQdrantClient
    from qdrant_client.http import models as qmodels
    from qdrant_client.http.exceptions import UnexpectedResponse
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    UnexpectedResponse = Exception  # Fallback for type checking

# Import core error types for specific exception handling
from core.errors import (
    BiblosError,
    BiblosDatabaseError,
    BiblosResourceError,
)

logger = logging.getLogger("biblos.db.qdrant")


@dataclass
class SearchResult:
    """Vector search result."""
    id: str
    score: float
    payload: Dict[str, Any]


class QdrantVectorStore:
    """
    Qdrant vector store for biblical text embeddings.

    Collections:
    - verses: Full verse embeddings (768 dims)
    - greek_verses: Greek-specific embeddings
    - hebrew_verses: Hebrew-specific embeddings
    - patristic: Patristic text embeddings
    - cross_references: Cross-reference pair embeddings
    """

    COLLECTIONS = {
        "verses": {"dimension": 768, "distance": "Cosine"},
        "greek_verses": {"dimension": 768, "distance": "Cosine"},
        "hebrew_verses": {"dimension": 768, "distance": "Cosine"},
        "patristic": {"dimension": 768, "distance": "Cosine"},
        "cross_references": {"dimension": 1536, "distance": "Cosine"}  # Concatenated pairs
    }

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        prefer_grpc: bool = True
    ):
        self.host = host or os.getenv("QDRANT_HOST", "localhost")
        self.port = port or int(os.getenv("QDRANT_PORT", "6333"))
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.prefer_grpc = prefer_grpc

        self._client: Optional['AsyncQdrantClient'] = None
        self._sync_client: Optional['QdrantClient'] = None

    async def connect(self) -> None:
        """Establish connection to Qdrant."""
        if not QDRANT_AVAILABLE:
            logger.warning("Qdrant client not available")
            return

        self._client = AsyncQdrantClient(
            host=self.host,
            port=self.port,
            api_key=self.api_key,
            prefer_grpc=self.prefer_grpc
        )
        logger.info(f"Connected to Qdrant at {self.host}:{self.port}")

    def get_sync_client(self) -> Optional['QdrantClient']:
        """Get synchronous client for non-async operations."""
        if not QDRANT_AVAILABLE:
            return None

        if not self._sync_client:
            self._sync_client = QdrantClient(
                host=self.host,
                port=self.port,
                api_key=self.api_key,
                prefer_grpc=self.prefer_grpc
            )
        return self._sync_client

    async def close(self) -> None:
        """Close Qdrant connection."""
        if self._client:
            await self._client.close()
            logger.info("Qdrant connection closed")

    async def create_collections(self) -> None:
        """Create all required collections."""
        if not self._client:
            return

        for name, config in self.COLLECTIONS.items():
            try:
                collections = await self._client.get_collections()
                existing = [c.name for c in collections.collections]

                if name not in existing:
                    await self._client.create_collection(
                        collection_name=name,
                        vectors_config=qmodels.VectorParams(
                            size=config["dimension"],
                            distance=getattr(qmodels.Distance, config["distance"].upper())
                        )
                    )
                    logger.info(f"Created collection: {name}")
                else:
                    logger.info(f"Collection exists: {name}")

            except Exception as e:
                logger.error(f"Failed to create collection {name}: {e}")

    # Upsert operations
    async def upsert_verse(
        self,
        verse_ref: str,
        embedding: List[float],
        payload: Optional[Dict[str, Any]] = None,
        collection: str = "verses"
    ) -> bool:
        """Upsert a verse embedding."""
        if not self._client:
            return False

        point_id = self._ref_to_uuid(verse_ref)
        full_payload = {"reference": verse_ref, **(payload or {})}

        try:
            await self._client.upsert(
                collection_name=collection,
                points=[
                    qmodels.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=full_payload
                    )
                ]
            )
            return True
        except UnexpectedResponse as e:
            logger.error(f"Qdrant API error upserting verse {verse_ref}: {e}")
            return False
        except (MemoryError, BiblosResourceError) as e:
            logger.critical(f"Resource exhaustion upserting verse {verse_ref}: {e}")
            raise
        except ConnectionError as e:
            logger.error(f"Connection error upserting verse {verse_ref}: {e}")
            return False
        except BiblosDatabaseError as e:
            logger.error(f"Database error upserting verse {verse_ref}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error upserting verse {verse_ref}: {e} ({type(e).__name__})")
            return False

    async def batch_upsert(
        self,
        points: List[Tuple[str, List[float], Dict[str, Any]]],
        collection: str = "verses",
        batch_size: int = 100
    ) -> int:
        """Batch upsert embeddings."""
        if not self._client:
            return 0

        count = 0
        batch = []

        for ref, embedding, payload in points:
            point_id = self._ref_to_uuid(ref)
            full_payload = {"reference": ref, **payload}

            batch.append(qmodels.PointStruct(
                id=point_id,
                vector=embedding,
                payload=full_payload
            ))

            if len(batch) >= batch_size:
                try:
                    await self._client.upsert(
                        collection_name=collection,
                        points=batch
                    )
                    count += len(batch)
                    batch = []
                except Exception as e:
                    logger.error(f"Batch upsert failed: {e}")

        # Final batch
        if batch:
            try:
                await self._client.upsert(
                    collection_name=collection,
                    points=batch
                )
                count += len(batch)
            except Exception as e:
                logger.error(f"Final batch upsert failed: {e}")

        return count

    # Search operations
    async def search_similar(
        self,
        query_embedding: List[float],
        collection: str = "verses",
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar vectors."""
        if not self._client:
            return []

        try:
            # Build filter
            query_filter = None
            if filter_conditions:
                must_conditions = []
                for key, value in filter_conditions.items():
                    if isinstance(value, list):
                        must_conditions.append(
                            qmodels.FieldCondition(
                                key=key,
                                match=qmodels.MatchAny(any=value)
                            )
                        )
                    else:
                        must_conditions.append(
                            qmodels.FieldCondition(
                                key=key,
                                match=qmodels.MatchValue(value=value)
                            )
                        )
                query_filter = qmodels.Filter(must=must_conditions)

            results = await self._client.search(
                collection_name=collection,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter
            )

            return [
                SearchResult(
                    id=str(r.id),
                    score=r.score,
                    payload=r.payload or {}
                )
                for r in results
            ]

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    async def search_by_reference(
        self,
        verse_ref: str,
        collection: str = "verses",
        limit: int = 10
    ) -> List[SearchResult]:
        """Search for similar verses given a reference."""
        if not self._client:
            return []

        point_id = self._ref_to_uuid(verse_ref)

        try:
            # Get the point first
            points = await self._client.retrieve(
                collection_name=collection,
                ids=[point_id],
                with_vectors=True
            )

            if not points:
                logger.warning(f"Verse not found: {verse_ref}")
                return []

            # Search with its vector
            return await self.search_similar(
                query_embedding=points[0].vector,
                collection=collection,
                limit=limit + 1  # +1 because it will find itself
            )

        except Exception as e:
            logger.error(f"Search by reference failed: {e}")
            return []

    # Cross-reference discovery
    async def discover_cross_references(
        self,
        verse_ref: str,
        top_k: int = 20,
        min_score: float = 0.7,
        exclude_same_book: bool = False
    ) -> List[Dict[str, Any]]:
        """Discover potential cross-references for a verse."""
        results = await self.search_by_reference(
            verse_ref=verse_ref,
            collection="verses",
            limit=top_k + 10
        )

        # Filter and format
        discoveries = []
        source_book = verse_ref.split(".")[0]

        for result in results:
            if result.payload.get("reference") == verse_ref:
                continue  # Skip self

            if result.score < min_score:
                continue

            target_ref = result.payload.get("reference", "")
            target_book = target_ref.split(".")[0] if target_ref else ""

            if exclude_same_book and target_book == source_book:
                continue

            discoveries.append({
                "source": verse_ref,
                "target": target_ref,
                "similarity": result.score,
                "payload": result.payload
            })

            if len(discoveries) >= top_k:
                break

        return discoveries

    # Collection management
    async def delete_collection(self, collection: str) -> bool:
        """Delete a collection."""
        if not self._client:
            return False

        try:
            await self._client.delete_collection(collection_name=collection)
            logger.info(f"Deleted collection: {collection}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection {collection}: {e}")
            return False

    async def get_collection_info(self, collection: str) -> Optional[Dict[str, Any]]:
        """Get collection information."""
        if not self._client:
            return None

        try:
            info = await self._client.get_collection(collection_name=collection)
            return {
                "name": collection,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return None

    async def get_all_collections_info(self) -> List[Dict[str, Any]]:
        """Get info for all collections."""
        if not self._client:
            return []

        infos = []
        for collection in self.COLLECTIONS.keys():
            info = await self.get_collection_info(collection)
            if info:
                infos.append(info)
        return infos

    # Utility methods
    def _ref_to_uuid(self, reference: str) -> str:
        """Convert verse reference to deterministic UUID."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"biblos.verse.{reference}"))
