"""
BIBLOS v2 - Database Tools

LangChain-compatible tools for Neo4j, Qdrant, and PostgreSQL operations.
Provides structured interfaces for agents to query and store data.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Type, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, ConfigDict

from db.neo4j_client import Neo4jClient
from db.qdrant_client import QdrantVectorStore
from data.schemas import ConnectionType, ConnectionStrength, normalize_verse_id


logger = logging.getLogger("biblos.tools.database")


# =============================================================================
# TOOL INPUT/OUTPUT SCHEMAS
# =============================================================================


class Neo4jCrossReferenceInput(BaseModel):
    """Input schema for Neo4j cross-reference queries."""

    model_config = ConfigDict(extra="forbid")

    verse_ref: str = Field(
        ...,
        description="Verse reference to query (e.g., GEN.1.1)",
        pattern=r"^[A-Z0-9]{3}\.\d+\.\d+$",
    )
    direction: str = Field(
        default="both",
        description="Direction of relationships: 'outgoing', 'incoming', or 'both'",
        pattern="^(outgoing|incoming|both)$",
    )
    relationship_types: Optional[List[str]] = Field(
        default=None,
        description="Filter by relationship types (e.g., ['TYPOLOGICALLY_FULFILLS', 'QUOTES'])",
    )
    max_depth: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Maximum traversal depth for graph queries",
    )


class Neo4jCrossReferenceOutput(BaseModel):
    """Output schema for Neo4j cross-reference queries."""

    model_config = ConfigDict(extra="forbid")

    verse_ref: str
    cross_references: List[Dict[str, Any]]
    total_count: int
    relationship_types_found: List[str]


class Neo4jCreateCrossReferenceInput(BaseModel):
    """Input schema for creating cross-references in Neo4j."""

    model_config = ConfigDict(extra="forbid")

    source_ref: str = Field(
        ...,
        description="Source verse reference",
        pattern=r"^[A-Z0-9]{3}\.\d+\.\d+$",
    )
    target_ref: str = Field(
        ...,
        description="Target verse reference",
        pattern=r"^[A-Z0-9]{3}\.\d+\.\d+$",
    )
    relationship_type: str = Field(
        ...,
        description="Type of relationship (e.g., TYPOLOGICALLY_FULFILLS, QUOTES)",
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence score for the connection",
    )
    properties: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional properties for the relationship",
    )


class QdrantSimilarityInput(BaseModel):
    """Input schema for Qdrant similarity search."""

    model_config = ConfigDict(extra="forbid")

    verse_ref: Optional[str] = Field(
        default=None,
        description="Verse reference to find similar verses for",
        pattern=r"^[A-Z0-9]{3}\.\d+\.\d+$",
    )
    query_embedding: Optional[List[float]] = Field(
        default=None,
        description="Direct embedding vector for search (768 dimensions)",
    )
    collection: str = Field(
        default="verses",
        description="Collection to search: verses, greek_verses, hebrew_verses, patristic",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results to return",
    )
    min_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold",
    )
    exclude_same_book: bool = Field(
        default=False,
        description="Exclude verses from the same book",
    )
    filter_conditions: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional filter conditions for the search",
    )


class QdrantSimilarityOutput(BaseModel):
    """Output schema for Qdrant similarity search."""

    model_config = ConfigDict(extra="forbid")

    query_verse: Optional[str]
    similar_verses: List[Dict[str, Any]]
    total_found: int


class PostgresVerseLookupInput(BaseModel):
    """Input schema for PostgreSQL verse lookup."""

    model_config = ConfigDict(extra="forbid")

    verse_refs: List[str] = Field(
        ...,
        description="List of verse references to look up",
        min_length=1,
        max_length=100,
    )
    include_morphology: bool = Field(
        default=False,
        description="Include word-level morphological data",
    )
    include_parallel_versions: bool = Field(
        default=False,
        description="Include parallel text versions (LXX, MT, etc.)",
    )


class PostgresVerseLookupOutput(BaseModel):
    """Output schema for PostgreSQL verse lookup."""

    model_config = ConfigDict(extra="forbid")

    verses: List[Dict[str, Any]]
    found_count: int
    not_found: List[str]


# =============================================================================
# NEO4J CROSS-REFERENCE TOOL
# =============================================================================


class Neo4jCrossReferenceTool(BaseTool):
    """
    Tool for querying cross-references from Neo4j graph database.

    Supports:
    - Finding cross-references for a verse
    - Filtering by relationship type and direction
    - Multi-hop traversal for discovering connection chains
    """

    name: str = "neo4j_cross_reference"
    description: str = (
        "Query the biblical cross-reference graph database to find connections "
        "between verses. Returns typological, prophetic, verbal, thematic, and "
        "other types of cross-references. Use this to discover how verses relate "
        "to each other across the Bible."
    )
    args_schema: Type[BaseModel] = Neo4jCrossReferenceInput
    return_direct: bool = False

    # Client instance
    _client: Optional[Neo4jClient] = None

    def __init__(self, client: Optional[Neo4jClient] = None, **kwargs):
        """Initialize with optional client."""
        super().__init__(**kwargs)
        self._client = client

    async def _ensure_client(self) -> Neo4jClient:
        """Ensure client is connected."""
        if self._client is None:
            self._client = Neo4jClient()
            await self._client.connect()
        return self._client

    def _run(
        self,
        verse_ref: str,
        direction: str = "both",
        relationship_types: Optional[List[str]] = None,
        max_depth: int = 1,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Synchronous execution."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is None:
            result = asyncio.run(
                self._arun(verse_ref, direction, relationship_types, max_depth)
            )
        else:
            import nest_asyncio
            nest_asyncio.apply()
            result = loop.run_until_complete(
                self._arun(verse_ref, direction, relationship_types, max_depth)
            )

        return result

    async def _arun(
        self,
        verse_ref: str,
        direction: str = "both",
        relationship_types: Optional[List[str]] = None,
        max_depth: int = 1,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Asynchronous execution."""
        try:
            client = await self._ensure_client()
            verse_ref = normalize_verse_id(verse_ref)

            # Get cross-references
            results = await client.get_cross_references(
                verse_ref=verse_ref,
                direction=direction,
                rel_types=relationship_types,
            )

            # Format output
            relationship_types_found = list(set(r.get("rel_type", "") for r in results))

            output = Neo4jCrossReferenceOutput(
                verse_ref=verse_ref,
                cross_references=results,
                total_count=len(results),
                relationship_types_found=relationship_types_found,
            )

            return output.model_dump_json(indent=2)

        except Exception as e:
            logger.error(f"Neo4j query failed: {e}")
            return f'{{"error": "{str(e)}"}}'


class Neo4jCreateCrossReferenceTool(BaseTool):
    """
    Tool for creating cross-references in Neo4j graph database.

    Creates verse nodes if they don't exist and establishes relationships.
    """

    name: str = "neo4j_create_cross_reference"
    description: str = (
        "Create a new cross-reference relationship between two verses in the "
        "graph database. Use this when you've discovered a new connection "
        "between biblical passages that should be recorded."
    )
    args_schema: Type[BaseModel] = Neo4jCreateCrossReferenceInput
    return_direct: bool = False

    _client: Optional[Neo4jClient] = None

    def __init__(self, client: Optional[Neo4jClient] = None, **kwargs):
        super().__init__(**kwargs)
        self._client = client

    async def _ensure_client(self) -> Neo4jClient:
        if self._client is None:
            self._client = Neo4jClient()
            await self._client.connect()
        return self._client

    def _run(
        self,
        source_ref: str,
        target_ref: str,
        relationship_type: str,
        confidence: float = 0.8,
        properties: Optional[Dict[str, Any]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Synchronous execution."""
        return asyncio.run(
            self._arun(source_ref, target_ref, relationship_type, confidence, properties)
        )

    async def _arun(
        self,
        source_ref: str,
        target_ref: str,
        relationship_type: str,
        confidence: float = 0.8,
        properties: Optional[Dict[str, Any]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Asynchronous execution."""
        try:
            client = await self._ensure_client()

            source_ref = normalize_verse_id(source_ref)
            target_ref = normalize_verse_id(target_ref)

            # Ensure verse nodes exist
            await client.create_verse_node(source_ref, {"reference": source_ref})
            await client.create_verse_node(target_ref, {"reference": target_ref})

            # Create relationship
            props = properties or {}
            props["confidence"] = confidence
            props["created_by"] = "agent"

            rel_id = await client.create_cross_reference(
                source_ref=source_ref,
                target_ref=target_ref,
                rel_type=relationship_type,
                properties=props,
            )

            return f'{{"success": true, "relationship_id": "{rel_id}", "source": "{source_ref}", "target": "{target_ref}"}}'

        except Exception as e:
            logger.error(f"Failed to create cross-reference: {e}")
            return f'{{"success": false, "error": "{str(e)}"}}'


# =============================================================================
# QDRANT SIMILARITY TOOL
# =============================================================================


class QdrantSimilarityTool(BaseTool):
    """
    Tool for vector similarity search using Qdrant.

    Supports:
    - Finding semantically similar verses
    - Filtering by collection (Greek, Hebrew, Patristic)
    - Cross-reference discovery based on embedding similarity
    """

    name: str = "qdrant_similarity_search"
    description: str = (
        "Search for semantically similar biblical verses using vector embeddings. "
        "Can find verses with similar meaning, themes, or vocabulary patterns. "
        "Useful for discovering thematic and conceptual cross-references."
    )
    args_schema: Type[BaseModel] = QdrantSimilarityInput
    return_direct: bool = False

    _client: Optional[QdrantVectorStore] = None

    def __init__(self, client: Optional[QdrantVectorStore] = None, **kwargs):
        super().__init__(**kwargs)
        self._client = client

    async def _ensure_client(self) -> QdrantVectorStore:
        if self._client is None:
            self._client = QdrantVectorStore()
            await self._client.connect()
        return self._client

    def _run(
        self,
        verse_ref: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        collection: str = "verses",
        top_k: int = 10,
        min_score: float = 0.5,
        exclude_same_book: bool = False,
        filter_conditions: Optional[Dict[str, Any]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Synchronous execution."""
        return asyncio.run(
            self._arun(
                verse_ref, query_embedding, collection, top_k,
                min_score, exclude_same_book, filter_conditions
            )
        )

    async def _arun(
        self,
        verse_ref: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        collection: str = "verses",
        top_k: int = 10,
        min_score: float = 0.5,
        exclude_same_book: bool = False,
        filter_conditions: Optional[Dict[str, Any]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Asynchronous execution."""
        try:
            client = await self._ensure_client()

            if verse_ref:
                verse_ref = normalize_verse_id(verse_ref)
                # Use discover_cross_references for verse-based search
                results = await client.discover_cross_references(
                    verse_ref=verse_ref,
                    top_k=top_k,
                    min_score=min_score,
                    exclude_same_book=exclude_same_book,
                )

                output = QdrantSimilarityOutput(
                    query_verse=verse_ref,
                    similar_verses=results,
                    total_found=len(results),
                )

            elif query_embedding:
                # Direct embedding search
                results = await client.search_similar(
                    query_embedding=query_embedding,
                    collection=collection,
                    limit=top_k,
                    score_threshold=min_score,
                    filter_conditions=filter_conditions,
                )

                output = QdrantSimilarityOutput(
                    query_verse=None,
                    similar_verses=[
                        {
                            "reference": r.payload.get("reference"),
                            "similarity": r.score,
                            "payload": r.payload,
                        }
                        for r in results
                    ],
                    total_found=len(results),
                )

            else:
                return '{"error": "Either verse_ref or query_embedding must be provided"}'

            return output.model_dump_json(indent=2)

        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return f'{{"error": "{str(e)}"}}'


# =============================================================================
# POSTGRESQL VERSE LOOKUP TOOL
# =============================================================================


class PostgresVerseLookupTool(BaseTool):
    """
    Tool for looking up verse text and metadata from PostgreSQL.

    Supports:
    - Bulk verse lookup
    - Including morphological data
    - Fetching parallel text versions
    """

    name: str = "postgres_verse_lookup"
    description: str = (
        "Look up verse text and metadata from the PostgreSQL database. "
        "Can retrieve verse text, morphological analysis, and parallel versions "
        "(LXX, Masoretic, etc.). Use this to get the actual text of verses."
    )
    args_schema: Type[BaseModel] = PostgresVerseLookupInput
    return_direct: bool = False

    _pool: Any = None  # Database connection pool

    def __init__(self, pool: Any = None, **kwargs):
        super().__init__(**kwargs)
        self._pool = pool

    async def _ensure_pool(self) -> Any:
        """Ensure database pool is connected."""
        if self._pool is None:
            import asyncpg
            from config import get_config

            config = get_config()
            self._pool = await asyncpg.create_pool(
                host=config.database.postgres_host,
                port=config.database.postgres_port,
                user=config.database.postgres_user,
                password=config.database.postgres_password,
                database=config.database.postgres_database,
            )
        return self._pool

    def _run(
        self,
        verse_refs: List[str],
        include_morphology: bool = False,
        include_parallel_versions: bool = False,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Synchronous execution."""
        return asyncio.run(
            self._arun(verse_refs, include_morphology, include_parallel_versions)
        )

    async def _arun(
        self,
        verse_refs: List[str],
        include_morphology: bool = False,
        include_parallel_versions: bool = False,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Asynchronous execution."""
        try:
            pool = await self._ensure_pool()

            # Normalize verse references
            normalized_refs = [normalize_verse_id(ref) for ref in verse_refs]

            verses = []
            not_found = []

            async with pool.acquire() as conn:
                for ref in normalized_refs:
                    # Parse verse reference
                    parts = ref.split(".")
                    if len(parts) < 3:
                        not_found.append(ref)
                        continue

                    book, chapter, verse = parts[0], int(parts[1]), int(parts[2])

                    # Query verse
                    row = await conn.fetchrow(
                        """
                        SELECT book, chapter, verse, text, original_text,
                               language, testament
                        FROM verses
                        WHERE book = $1 AND chapter = $2 AND verse = $3
                        """,
                        book, chapter, verse
                    )

                    if row:
                        verse_data = dict(row)
                        verse_data["reference"] = ref

                        # Optionally fetch morphology
                        if include_morphology:
                            morphology = await conn.fetch(
                                """
                                SELECT position, surface_form, lemma,
                                       part_of_speech, morphology_code
                                FROM words
                                WHERE verse_ref = $1
                                ORDER BY position
                                """,
                                ref
                            )
                            verse_data["morphology"] = [dict(m) for m in morphology]

                        # Optionally fetch parallel versions
                        if include_parallel_versions:
                            parallels = await conn.fetch(
                                """
                                SELECT version, text
                                FROM verse_versions
                                WHERE verse_ref = $1
                                """,
                                ref
                            )
                            verse_data["parallel_versions"] = {
                                p["version"]: p["text"] for p in parallels
                            }

                        verses.append(verse_data)
                    else:
                        not_found.append(ref)

            output = PostgresVerseLookupOutput(
                verses=verses,
                found_count=len(verses),
                not_found=not_found,
            )

            return output.model_dump_json(indent=2)

        except Exception as e:
            logger.error(f"PostgreSQL lookup failed: {e}")
            return f'{{"error": "{str(e)}"}}'


# =============================================================================
# TOOL FACTORY
# =============================================================================


def create_database_tools(
    neo4j_client: Optional[Neo4jClient] = None,
    qdrant_client: Optional[QdrantVectorStore] = None,
    postgres_pool: Any = None,
) -> List[BaseTool]:
    """
    Factory function to create all database tools.

    Args:
        neo4j_client: Optional Neo4j client instance
        qdrant_client: Optional Qdrant client instance
        postgres_pool: Optional PostgreSQL connection pool

    Returns:
        List of configured database tools
    """
    return [
        Neo4jCrossReferenceTool(client=neo4j_client),
        Neo4jCreateCrossReferenceTool(client=neo4j_client),
        QdrantSimilarityTool(client=qdrant_client),
        PostgresVerseLookupTool(pool=postgres_pool),
    ]


# =============================================================================
# TOOL DESCRIPTIONS FOR AGENT PROMPTS
# =============================================================================


TOOL_DESCRIPTIONS = """
Available Database Tools:

1. neo4j_cross_reference: Query the biblical cross-reference graph database
   - Input: verse_ref (required), direction, relationship_types, max_depth
   - Returns: List of cross-references with relationship types

2. neo4j_create_cross_reference: Create new cross-reference in graph database
   - Input: source_ref, target_ref, relationship_type, confidence
   - Returns: Success status and relationship ID

3. qdrant_similarity_search: Find semantically similar verses using embeddings
   - Input: verse_ref or query_embedding, collection, top_k, min_score
   - Returns: List of similar verses with similarity scores

4. postgres_verse_lookup: Look up verse text and metadata
   - Input: verse_refs (list), include_morphology, include_parallel_versions
   - Returns: Verse text, morphology, and parallel versions

Relationship Types for Neo4j:
- TYPOLOGICALLY_FULFILLS: Type/antitype connections (OT -> NT)
- PROPHETICALLY_FULFILLS: Prophecy fulfillment
- QUOTES: Direct quotation
- ALLUDES_TO: Allusion
- VERBAL_PARALLEL: Shared vocabulary
- THEMATICALLY_CONNECTED: Thematic links
- LITURGICALLY_USED: Liturgical connections
"""
