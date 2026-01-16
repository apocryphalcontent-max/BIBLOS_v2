"""
Vector Store Projection from Event Store

Builds multi-domain vector embeddings as a projection of events.
Listens to verse processing events and generates embeddings.
"""
import logging
from typing import Optional, List
import asyncio

from db.events import (
    BaseEvent,
    EventType,
    VerseProcessingCompleted,
    WordAnalyzed,
    PatristicWitnessAdded,
)
from db.projections import ProjectionBase
from db.event_store import EventStore
from ml.embeddings.multi_vector_store import MultiVectorStore, EmbeddingDomain
from ml.embeddings.domain_embedders import MultiDomainEmbedder, VerseContext


logger = logging.getLogger(__name__)


class VectorProjection(ProjectionBase):
    """
    Event-driven projection that builds vector embeddings.

    Listens to:
    - VerseProcessingCompleted: Generate all domain embeddings
    - WordAnalyzed: Update semantic embeddings
    - PatristicWitnessAdded: Update patristic embeddings
    """

    def __init__(
        self,
        vector_store: MultiVectorStore,
        event_store: EventStore,
        embedder: Optional[MultiDomainEmbedder] = None
    ):
        """
        Initialize vector projection.

        Args:
            vector_store: Multi-vector store
            event_store: Event store
            embedder: Multi-domain embedder (creates if None)
        """
        self.vector_store = vector_store
        self.event_store = event_store
        self.embedder = embedder or MultiDomainEmbedder()
        self.projection_name = "VectorProjection"
        self._last_position = 0
        self._is_running = False

        # Cache verse contexts during rebuild
        self._verse_contexts: dict = {}

    async def initialize(self) -> None:
        """Initialize vector collections."""
        await self.vector_store.create_collections()
        logger.info("Vector collections initialized")

    async def handle_event(self, event: BaseEvent) -> None:
        """
        Handle event and update vectors.

        Args:
            event: Event to process
        """
        if isinstance(event, VerseProcessingCompleted):
            await self._handle_verse_completed(event)
        elif isinstance(event, WordAnalyzed):
            await self._handle_word_analyzed(event)
        elif isinstance(event, PatristicWitnessAdded):
            await self._handle_patristic_added(event)

    async def _handle_verse_completed(
        self,
        event: VerseProcessingCompleted
    ) -> None:
        """
        Generate embeddings when verse processing completes.

        Args:
            event: Verse processing completed event
        """
        # Build verse context from event
        context = self._build_context(event)

        # Generate embeddings across all domains
        embeddings = self.embedder.embed_verse(context)

        # Store in vector database
        metadata = {
            'verse_id': event.verse_id,
            'quality_tier': event.quality_tier,
            'phases_completed': event.phases_completed,
        }

        await self.vector_store.upsert_verse(
            verse_id=event.verse_id,
            embeddings=embeddings,
            metadata=metadata
        )

        logger.debug(f"Generated embeddings for {event.verse_id}")

    async def _handle_word_analyzed(self, event: WordAnalyzed) -> None:
        """
        Update semantic embedding when words are analyzed.

        Args:
            event: Word analyzed event
        """
        # Get cached context or build new one
        context = self._verse_contexts.get(event.verse_id)
        if context is None:
            # Would need to query for verse text - skip for now
            return

        # Update words in context
        if context.words is None:
            context.words = []
        context.words.append({
            'lemma': event.lemma,
            'part_of_speech': event.part_of_speech,
            'gloss': event.gloss
        })

        # Regenerate semantic embedding with word context
        semantic_embedding = self.embedder.embedders[EmbeddingDomain.SEMANTIC].embed(context)

        # Update only semantic domain
        await self.vector_store.upsert_verse(
            verse_id=event.verse_id,
            embeddings={EmbeddingDomain.SEMANTIC: semantic_embedding},
            metadata={'verse_id': event.verse_id}
        )

    async def _handle_patristic_added(self, event: PatristicWitnessAdded) -> None:
        """
        Update patristic embedding when witness is added.

        Args:
            event: Patristic witness added event
        """
        # Get cached context or build new one
        context = self._verse_contexts.get(event.verse_id)
        if context is None:
            # Would need to query for verse text - skip for now
            return

        # Update patristic witnesses in context
        if context.patristic_witnesses is None:
            context.patristic_witnesses = []

        context.patristic_witnesses.append({
            'father_name': event.father_name,
            'authority_level': event.authority_level,
            'interpretation': event.interpretation
        })

        # Regenerate patristic embedding
        patristic_embedding = self.embedder.embedders[EmbeddingDomain.PATRISTIC].embed(context)

        # Update only patristic domain
        await self.vector_store.upsert_verse(
            verse_id=event.verse_id,
            embeddings={EmbeddingDomain.PATRISTIC: patristic_embedding},
            metadata={'verse_id': event.verse_id}
        )

    def _build_context(self, event: VerseProcessingCompleted) -> VerseContext:
        """
        Build verse context from completed event.

        In production, would query for full verse data.
        For now, creates minimal context.
        """
        # Parse verse ID
        parts = event.verse_id.split('.')
        book = parts[0]
        chapter = int(parts[1])
        verse = int(parts[2])

        # Determine testament
        OT_BOOKS = {
            "GEN", "EXO", "LEV", "NUM", "DEU", "JOS", "JDG", "RUT",
            "1SA", "2SA", "1KI", "2KI", "1CH", "2CH", "EZR", "NEH",
            "EST", "JOB", "PSA", "PRO", "ECC", "SNG", "ISA", "JER",
            "LAM", "EZK", "DAN", "HOS", "JOL", "AMO", "OBA", "JON",
            "MIC", "NAH", "HAB", "ZEP", "HAG", "ZEC", "MAL"
        }
        testament = "old" if book in OT_BOOKS else "new"

        context = VerseContext(
            verse_id=event.verse_id,
            text="",  # Would be populated from database
            testament=testament,
            book=book,
            chapter=chapter,
            verse=verse
        )

        # Cache for incremental updates
        self._verse_contexts[event.verse_id] = context

        return context

    async def rebuild_batch(
        self,
        verse_ids: List[str],
        batch_size: int = 50
    ) -> None:
        """
        Rebuild embeddings for a batch of verses.

        Args:
            verse_ids: List of verse IDs to rebuild
            batch_size: Batch size for embedding generation
        """
        logger.info(f"Rebuilding embeddings for {len(verse_ids)} verses")

        for i in range(0, len(verse_ids), batch_size):
            batch = verse_ids[i:i + batch_size]

            # Build contexts (would query database in production)
            contexts = [self._build_minimal_context(vid) for vid in batch]

            # Generate embeddings in batch
            batch_embeddings = self.embedder.embed_batch(contexts)

            # Prepare for batch upsert
            upsert_data = [
                (
                    ctx.verse_id,
                    embeddings,
                    {'verse_id': ctx.verse_id}
                )
                for ctx, embeddings in zip(contexts, batch_embeddings)
            ]

            # Batch upsert
            await self.vector_store.batch_upsert(upsert_data)

            logger.info(f"Rebuilt embeddings for batch {i // batch_size + 1}")

    def _build_minimal_context(self, verse_id: str) -> VerseContext:
        """Build minimal context for verse ID."""
        parts = verse_id.split('.')
        book = parts[0]
        chapter = int(parts[1])
        verse = int(parts[2])

        OT_BOOKS = {
            "GEN", "EXO", "LEV", "NUM", "DEU", "JOS", "JDG", "RUT",
            "1SA", "2SA", "1KI", "2KI", "1CH", "2CH", "EZR", "NEH",
            "EST", "JOB", "PSA", "PRO", "ECC", "SNG", "ISA", "JER",
            "LAM", "EZK", "DAN", "HOS", "JOL", "AMO", "OBA", "JON",
            "MIC", "NAH", "HAB", "ZEP", "HAG", "ZEC", "MAL"
        }
        testament = "old" if book in OT_BOOKS else "new"

        return VerseContext(
            verse_id=verse_id,
            text=f"Verse text for {verse_id}",  # Placeholder
            testament=testament,
            book=book,
            chapter=chapter,
            verse=verse
        )

    async def _clear_projection(self) -> None:
        """Clear all vector data."""
        # Delete all collections and recreate
        for domain in EmbeddingDomain:
            try:
                self.vector_store.client.delete_collection(
                    collection_name=domain.collection_name
                )
            except:
                pass

        await self.vector_store.create_collections()
        logger.info("Vector data cleared and collections recreated")
