"""
BIBLOS v2 - PostgreSQL Client with Async Support

Provides async database operations using SQLAlchemy 2.0 and asyncpg.
"""
from typing import AsyncGenerator, Optional, List, Dict, Any
from contextlib import asynccontextmanager
import logging
import os

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker
)
from sqlalchemy import select, update, delete, func, text
from sqlalchemy.orm import selectinload

from db.models import Base, Book, Verse, CrossReference, PatristicCitation, ExtractionResult


logger = logging.getLogger("biblos.db.postgres")


class PostgresClient:
    """
    Async PostgreSQL client for BIBLOS database operations.

    Features:
    - Connection pooling with asyncpg
    - Automatic session management
    - Batch operations for efficiency
    - pgvector integration for embeddings
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        pool_size: int = 10,
        max_overflow: int = 20,
        echo: bool = False
    ):
        self.database_url = database_url or os.getenv(
            "DATABASE_URL",
            "postgresql+asyncpg://biblos:biblos@localhost:5432/biblos_v2"
        )
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.echo = echo

        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker] = None

    async def initialize(self) -> None:
        """
        Initialize database engine and session factory.

        Connection pool optimizations:
        - pool_size: Base number of persistent connections
        - max_overflow: Additional connections under high load
        - pool_pre_ping: Verify connections before use (handles stale connections)
        - pool_recycle: Recycle connections after 1800s to prevent timeouts
        - pool_timeout: Max wait time for connection from pool
        - connect_args: asyncpg-specific optimizations for prepared statements
        """
        self._engine = create_async_engine(
            self.database_url,
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
            echo=self.echo,
            pool_pre_ping=True,
            pool_recycle=1800,  # Recycle connections every 30 minutes
            pool_timeout=30,     # Wait max 30s for connection from pool
            connect_args={
                # asyncpg-specific: cache prepared statements for repeated queries
                "prepared_statement_cache_size": 100,
                # Command timeout prevents runaway queries
                "command_timeout": 60,
            }
        )

        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

        logger.info(
            f"PostgreSQL client initialized (pool_size={self.pool_size}, "
            f"max_overflow={self.max_overflow})"
        )

    async def close(self) -> None:
        """Close database connections."""
        if self._engine:
            await self._engine.dispose()
            logger.info("PostgreSQL connections closed")

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session."""
        if not self._session_factory:
            await self.initialize()

        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def create_tables(self) -> None:
        """Create all database tables."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created")

    # Book operations
    async def get_book(self, code: str) -> Optional[Book]:
        """Get book by code."""
        async with self.session() as session:
            result = await session.execute(
                select(Book).where(Book.code == code)
            )
            return result.scalar_one_or_none()

    async def get_all_books(self) -> List[Book]:
        """Get all books."""
        async with self.session() as session:
            result = await session.execute(
                select(Book).order_by(Book.order_num)
            )
            return list(result.scalars().all())

    async def upsert_book(self, book_data: Dict[str, Any]) -> Book:
        """Insert or update book."""
        async with self.session() as session:
            existing = await session.execute(
                select(Book).where(Book.code == book_data["code"])
            )
            book = existing.scalar_one_or_none()

            if book:
                for key, value in book_data.items():
                    setattr(book, key, value)
            else:
                book = Book(**book_data)
                session.add(book)

            await session.flush()
            return book

    # Verse operations
    async def get_verse(self, reference: str) -> Optional[Verse]:
        """Get verse by reference."""
        async with self.session() as session:
            result = await session.execute(
                select(Verse)
                .where(Verse.reference == reference)
                .options(selectinload(Verse.book))
            )
            return result.scalar_one_or_none()

    async def get_verses_by_book(self, book_code: str) -> List[Verse]:
        """Get all verses for a book."""
        async with self.session() as session:
            result = await session.execute(
                select(Verse)
                .join(Book)
                .where(Book.code == book_code)
                .order_by(Verse.chapter, Verse.verse_num)
            )
            return list(result.scalars().all())

    async def upsert_verse(self, verse_data: Dict[str, Any]) -> Verse:
        """Insert or update verse."""
        async with self.session() as session:
            existing = await session.execute(
                select(Verse).where(Verse.reference == verse_data["reference"])
            )
            verse = existing.scalar_one_or_none()

            if verse:
                for key, value in verse_data.items():
                    if key != "reference":
                        setattr(verse, key, value)
            else:
                verse = Verse(**verse_data)
                session.add(verse)

            await session.flush()
            return verse

    async def batch_upsert_verses(self, verses: List[Dict[str, Any]]) -> int:
        """
        Batch upsert verses for efficiency.

        Optimized to use bulk SELECT with IN clause instead of N+1 individual queries.
        Processes in batches of 500 to balance memory usage and query efficiency.
        """
        if not verses:
            return 0

        count = 0
        batch_size = 500  # Optimal batch size for PostgreSQL IN clause

        async with self.session() as session:
            # Process in batches to avoid overly large IN clauses
            for batch_start in range(0, len(verses), batch_size):
                batch = verses[batch_start:batch_start + batch_size]

                # Extract all references for this batch
                references = [v["reference"] for v in batch]

                # Single bulk SELECT for all verses in batch
                existing_result = await session.execute(
                    select(Verse).where(Verse.reference.in_(references))
                )
                existing_verses = {v.reference: v for v in existing_result.scalars().all()}

                # Process each verse - update existing or insert new
                for verse_data in batch:
                    ref = verse_data["reference"]
                    verse = existing_verses.get(ref)

                    if verse:
                        # Update existing verse
                        for key, value in verse_data.items():
                            if key != "reference":
                                setattr(verse, key, value)
                    else:
                        # Insert new verse
                        verse = Verse(**verse_data)
                        session.add(verse)

                    count += 1

                # Flush after each batch to manage memory
                await session.flush()
                logger.info(f"Processed {count} verses")

        return count

    # Cross-reference operations
    async def get_crossrefs_for_verse(self, verse_ref: str) -> List[CrossReference]:
        """Get all cross-references where verse is source or target."""
        async with self.session() as session:
            verse = await session.execute(
                select(Verse).where(Verse.reference == verse_ref)
            )
            verse = verse.scalar_one_or_none()
            if not verse:
                return []

            result = await session.execute(
                select(CrossReference)
                .where(
                    (CrossReference.source_id == verse.id) |
                    (CrossReference.target_id == verse.id)
                )
                .options(
                    selectinload(CrossReference.source_verse),
                    selectinload(CrossReference.target_verse)
                )
            )
            return list(result.scalars().all())

    async def add_crossref(
        self,
        source_ref: str,
        target_ref: str,
        connection_type: str,
        confidence: float = 0.0,
        **kwargs
    ) -> Optional[CrossReference]:
        """
        Add a cross-reference between verses.

        Optimized to fetch both source and target verses in a single query.
        """
        async with self.session() as session:
            # Fetch both verses in a single query using IN clause
            result = await session.execute(
                select(Verse).where(Verse.reference.in_([source_ref, target_ref]))
            )
            verses_by_ref = {v.reference: v for v in result.scalars().all()}

            source = verses_by_ref.get(source_ref)
            target = verses_by_ref.get(target_ref)

            if not source or not target:
                missing = []
                if not source:
                    missing.append(source_ref)
                if not target:
                    missing.append(target_ref)
                logger.warning(f"Cannot create crossref: missing verse(s): {missing}")
                return None

            crossref = CrossReference(
                source_id=source.id,
                target_id=target.id,
                connection_type=connection_type,
                confidence=confidence,
                **kwargs
            )
            session.add(crossref)
            await session.flush()
            return crossref

    async def batch_add_crossrefs(
        self,
        crossrefs: List[Dict[str, Any]]
    ) -> int:
        """
        Batch add cross-references efficiently.

        Each dict should contain: source_ref, target_ref, connection_type,
        and optionally: confidence, strength, evidence, notes.

        Optimized to fetch all referenced verses in a single bulk query,
        eliminating N+1 pattern for large batch insertions.
        """
        if not crossrefs:
            return 0

        count = 0
        batch_size = 500

        async with self.session() as session:
            for batch_start in range(0, len(crossrefs), batch_size):
                batch = crossrefs[batch_start:batch_start + batch_size]

                # Collect all unique verse references from this batch
                all_refs = set()
                for cr in batch:
                    all_refs.add(cr["source_ref"])
                    all_refs.add(cr["target_ref"])

                # Single bulk query for all verses
                result = await session.execute(
                    select(Verse).where(Verse.reference.in_(list(all_refs)))
                )
                verses_by_ref = {v.reference: v for v in result.scalars().all()}

                # Create cross-references
                for cr_data in batch:
                    source = verses_by_ref.get(cr_data["source_ref"])
                    target = verses_by_ref.get(cr_data["target_ref"])

                    if not source or not target:
                        logger.warning(
                            f"Skipping crossref {cr_data['source_ref']} -> "
                            f"{cr_data['target_ref']}: missing verse(s)"
                        )
                        continue

                    crossref = CrossReference(
                        source_id=source.id,
                        target_id=target.id,
                        connection_type=cr_data["connection_type"],
                        confidence=cr_data.get("confidence", 0.0),
                        strength=cr_data.get("strength"),
                        evidence=cr_data.get("evidence"),
                        notes=cr_data.get("notes")
                    )
                    session.add(crossref)
                    count += 1

                await session.flush()
                logger.info(f"Added {count} cross-references")

        return count

    async def get_verses_by_refs(self, references: List[str]) -> Dict[str, Verse]:
        """
        Bulk fetch verses by references.

        Utility method for efficient batch lookups.
        Returns a dict mapping reference -> Verse object.
        """
        if not references:
            return {}

        async with self.session() as session:
            result = await session.execute(
                select(Verse)
                .where(Verse.reference.in_(references))
                .options(selectinload(Verse.book))
            )
            return {v.reference: v for v in result.scalars().all()}

    # Extraction results operations
    async def save_extraction_result(
        self,
        verse_ref: str,
        agent_name: str,
        extraction_type: str,
        data: Dict[str, Any],
        confidence: float,
        status: str = "completed"
    ) -> ExtractionResult:
        """Save extraction result for a verse."""
        async with self.session() as session:
            verse = await session.execute(
                select(Verse).where(Verse.reference == verse_ref)
            )
            verse = verse.scalar_one_or_none()
            if not verse:
                raise ValueError(f"Verse not found: {verse_ref}")

            result = ExtractionResult(
                verse_id=verse.id,
                agent_name=agent_name,
                extraction_type=extraction_type,
                data=data,
                confidence=confidence,
                status=status
            )
            session.add(result)
            await session.flush()
            return result

    # Vector search with pgvector
    async def find_similar_verses(
        self,
        embedding: List[float],
        limit: int = 10,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Find similar verses using cosine similarity."""
        async with self.session() as session:
            # Using pgvector cosine distance operator <=>
            query = text("""
                SELECT reference, text_english,
                       1 - (embedding <=> :embedding::vector) as similarity
                FROM verses
                WHERE embedding IS NOT NULL
                  AND 1 - (embedding <=> :embedding::vector) > :threshold
                ORDER BY embedding <=> :embedding::vector
                LIMIT :limit
            """)

            result = await session.execute(
                query,
                {"embedding": embedding, "threshold": threshold, "limit": limit}
            )

            return [
                {"reference": row[0], "text": row[1], "similarity": row[2]}
                for row in result.fetchall()
            ]

    # Statistics
    async def get_statistics(self) -> Dict[str, int]:
        """
        Get database statistics.

        Optimized to run a single query with UNION ALL instead of 5 separate queries.
        """
        async with self.session() as session:
            # Single query with UNION ALL for all counts
            query = text("""
                SELECT 'books' as table_name, COUNT(*) as cnt FROM books
                UNION ALL
                SELECT 'verses', COUNT(*) FROM verses
                UNION ALL
                SELECT 'cross_references', COUNT(*) FROM cross_references
                UNION ALL
                SELECT 'patristic_citations', COUNT(*) FROM patristic_citations
                UNION ALL
                SELECT 'extraction_results', COUNT(*) FROM extraction_results
            """)

            result = await session.execute(query)
            return {row[0]: row[1] for row in result.fetchall()}


# Global client instance
_client: Optional[PostgresClient] = None


async def get_db_client() -> PostgresClient:
    """Get or create global database client."""
    global _client
    if _client is None:
        _client = PostgresClient()
        await _client.initialize()
    return _client


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for FastAPI - get database session."""
    client = await get_db_client()
    async with client.session() as session:
        yield session
