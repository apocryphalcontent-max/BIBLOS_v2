"""
Vector Store Projection - Event-Driven Embedding Updates

Builds multi-domain vector embeddings as a projection of events.
Listens to verse processing events and generates embeddings across
all six domains in real-time.

Events handled:
- VerseProcessingCompleted: Generate and store embeddings
- CrossReferenceValidated: Update related embeddings with relationship info
- PatristicWitnessAdded: Update patristic domain embeddings
- OmniResolutionComputed: Update semantic embeddings with word analysis
- TypologyDiscovered: Update typological domain embeddings

Integration points:
- Event Store subscription
- MultiVectorStore upsert
- MultiDomainEmbedder for embedding generation
"""
import logging
import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from db.events import (
    BaseEvent,
    EventType,
    VerseProcessingCompleted,
    CrossReferenceValidated,
    OmniResolutionComputed,
    PatristicWitnessAdded,
    TypologyDiscovered,
)
from db.projections import ProjectionBase
from db.event_store import EventStore
from ml.embeddings.multi_vector_store import MultiVectorStore, EmbeddingDomain
from ml.embeddings.domain_embedders import MultiDomainEmbedder, VerseContext


logger = logging.getLogger(__name__)


@dataclass
class VectorProjectionConfig:
    """Configuration for vector store projection."""
    batch_size: int = 50
    parallel_domains: bool = True
    update_on_crossref: bool = True
    update_on_patristic: bool = True
    update_on_typology: bool = True
    re_embed_threshold: int = 5  # Re-embed after N updates
    lazy_load_embedder: bool = True  # Lazy load embedding models


class VectorProjection(ProjectionBase):
    """
    Event-driven projection that builds vector embeddings.

    Subscribes to verse processing events and generates/updates embeddings
    across all six domains in real-time.

    Features:
    - Automatic embedding generation on verse completion
    - Incremental updates for patristic witnesses
    - Batch processing for efficiency
    - Cross-domain consistency
    - Covenantal domain support

    Listens to:
    - VerseProcessingCompleted: Generate all domain embeddings
    - OmniResolutionComputed: Update semantic embeddings
    - PatristicWitnessAdded: Update patristic embeddings
    - CrossReferenceValidated: Track relationship updates
    - TypologyDiscovered: Update typological embeddings
    """

    # Old Testament book codes
    OT_BOOKS = {
        "GEN", "EXO", "LEV", "NUM", "DEU", "JOS", "JDG", "RUT",
        "1SA", "2SA", "1KI", "2KI", "1CH", "2CH", "EZR", "NEH",
        "EST", "JOB", "PSA", "PRO", "ECC", "SNG", "ISA", "JER",
        "LAM", "EZK", "DAN", "HOS", "JOL", "AMO", "OBA", "JON",
        "MIC", "NAH", "HAB", "ZEP", "HAG", "ZEC", "MAL"
    }

    def __init__(
        self,
        vector_store: MultiVectorStore,
        event_store: EventStore,
        embedder: Optional[MultiDomainEmbedder] = None,
        config: Optional[VectorProjectionConfig] = None
    ):
        """
        Initialize vector projection.

        Args:
            vector_store: Multi-vector store
            event_store: Event store
            embedder: Multi-domain embedder (creates if None)
            config: Projection configuration
        """
        self.vector_store = vector_store
        self.event_store = event_store
        self.config = config or VectorProjectionConfig()
        self.embedder = embedder or MultiDomainEmbedder(
            lazy_load=self.config.lazy_load_embedder
        )
        self.projection_name = "VectorProjection"
        self._last_position = 0
        self._is_running = False

        # Cache verse contexts during rebuild
        self._verse_contexts: Dict[str, VerseContext] = {}

        # Track pending updates for batching
        self._pending_updates: Dict[str, Dict[str, Any]] = {}
        self._update_counts: Dict[str, int] = {}

        logger.info("Initialized VectorProjection with 6 domain support")

    async def initialize(self) -> None:
        """Initialize vector collections."""
        await self.vector_store.create_collections()
        logger.info(f"Vector collections initialized for {len(EmbeddingDomain)} domains")

    async def handle_event(self, event: BaseEvent) -> None:
        """
        Handle event and update vectors.

        Routes events to appropriate handlers based on type.
        """
        handlers = {
            EventType.VERSE_PROCESSING_COMPLETED.value: self._handle_verse_completed,
            EventType.OMNI_RESOLUTION_COMPUTED.value: self._handle_word_analyzed,
            EventType.PATRISTIC_WITNESS_ADDED.value: self._handle_patristic_added,
            EventType.CROSS_REFERENCE_VALIDATED.value: self._handle_cross_reference,
            EventType.TYPOLOGY_DISCOVERED.value: self._handle_typology,
        }

        handler = handlers.get(event.event_type)
        if handler:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Error handling {event.event_type}: {e}", exc_info=True)

    async def _handle_verse_completed(
        self,
        event: VerseProcessingCompleted
    ) -> None:
        """
        Generate embeddings when verse processing completes.

        This is the primary embedding generation event - creates embeddings
        across all six domains.
        """
        # Build verse context from event
        context = self._build_context(event)

        # Merge in any pending updates (words, patristic, etc.)
        pending = self._pending_updates.get(event.verse_id, {})
        if pending.get('words'):
            context.words = pending['words']
        if pending.get('patristic_witnesses'):
            context.patristic_witnesses = pending['patristic_witnesses']
        if pending.get('covenant_name'):
            context.covenant_name = pending['covenant_name']
        if pending.get('covenant_role'):
            context.covenant_role = pending['covenant_role']

        # Generate embeddings across all domains
        embeddings = self.embedder.embed_verse(context)

        # Store in vector database with rich metadata
        metadata = {
            'verse_id': event.verse_id,
            'quality_tier': event.quality_tier,
            'phases_completed': event.phases_completed,
            'cross_reference_count': event.cross_reference_count,
            'testament': context.testament,
            'book': context.book,
            'chapter': context.chapter,
            'verse_num': context.verse,
        }

        await self.vector_store.upsert_verse(
            verse_id=event.verse_id,
            embeddings=embeddings,
            metadata=metadata
        )

        # Clear pending updates for this verse
        self._pending_updates.pop(event.verse_id, None)
        self._update_counts[event.verse_id] = 0

        logger.debug(f"Generated embeddings for {event.verse_id} across {len(embeddings)} domains")

    async def _handle_word_analyzed(self, event: OmniResolutionComputed) -> None:
        """
        Update semantic embedding when words are analyzed.

        Accumulates word data and may trigger selective re-embedding.
        """
        verse_id = event.verse_id

        # Store word data for next embedding generation
        if verse_id not in self._pending_updates:
            self._pending_updates[verse_id] = {}

        words = self._pending_updates[verse_id].get('words', [])
        words.append({
            'word': event.word,
            'lemma': event.word,  # Would be actual lemma
            'language': event.language,
            'primary_meaning': event.primary_meaning,
            'semantic_field_map': event.semantic_field_map,
        })
        self._pending_updates[verse_id]['words'] = words

        # Get cached context or skip
        context = self._verse_contexts.get(verse_id)
        if context is None:
            return

        # Update words in context
        context.words = words

        # Track update count
        self._update_counts[verse_id] = self._update_counts.get(verse_id, 0) + 1

        # Check if re-embedding threshold reached
        if self._update_counts[verse_id] >= self.config.re_embed_threshold:
            # Regenerate semantic embedding with word context
            embedder = self.embedder._get_embedder(EmbeddingDomain.SEMANTIC)
            semantic_embedding = embedder.embed(context)

            await self.vector_store.upsert_verse(
                verse_id=verse_id,
                embeddings={EmbeddingDomain.SEMANTIC: semantic_embedding},
                metadata={'verse_id': verse_id}
            )
            self._update_counts[verse_id] = 0
            logger.debug(f"Re-embedded semantic for {verse_id} after word analysis")

    async def _handle_patristic_added(self, event: PatristicWitnessAdded) -> None:
        """
        Update patristic embedding when witness is added.

        Accumulates patristic data and updates patristic domain embedding.
        """
        if not self.config.update_on_patristic:
            return

        verse_id = event.verse_id

        # Store patristic data
        if verse_id not in self._pending_updates:
            self._pending_updates[verse_id] = {}

        witnesses = self._pending_updates[verse_id].get('patristic_witnesses', [])
        witnesses.append({
            'father_name': event.father_name,
            'authority_level': event.authority_level,
            'interpretation': event.interpretation,
            'source_reference': event.source_reference,
        })
        self._pending_updates[verse_id]['patristic_witnesses'] = witnesses

        # Get cached context
        context = self._verse_contexts.get(verse_id)
        if context is None:
            return

        # Update patristic witnesses in context
        context.patristic_witnesses = witnesses

        # Regenerate patristic embedding
        embedder = self.embedder._get_embedder(EmbeddingDomain.PATRISTIC)
        patristic_embedding = embedder.embed(context)

        await self.vector_store.upsert_verse(
            verse_id=verse_id,
            embeddings={EmbeddingDomain.PATRISTIC: patristic_embedding},
            metadata={'verse_id': verse_id}
        )
        logger.debug(f"Updated patristic embedding for {verse_id}")

    async def _handle_cross_reference(
        self,
        event: CrossReferenceValidated
    ) -> None:
        """
        Handle cross-reference validation.

        Tracks updates for both source and target verses.
        """
        if not self.config.update_on_crossref:
            return

        # Track that both verses have new relationships
        for verse_id in [event.source_ref, event.target_ref]:
            self._update_counts[verse_id] = self._update_counts.get(verse_id, 0) + 1

            # Store cross-reference data
            if verse_id not in self._pending_updates:
                self._pending_updates[verse_id] = {}
            refs = self._pending_updates[verse_id].get('cross_references', [])
            other_ref = event.target_ref if verse_id == event.source_ref else event.source_ref
            refs.append({
                'reference': other_ref,
                'type': event.connection_type,
                'confidence': event.final_confidence,
            })
            self._pending_updates[verse_id]['cross_references'] = refs

    async def _handle_typology(self, event: TypologyDiscovered) -> None:
        """
        Handle typology discovery.

        Updates typological domain embeddings for both type and antitype.
        """
        if not self.config.update_on_typology:
            return

        # Update both type and antitype
        for verse_id in [event.type_ref, event.antitype_ref]:
            context = self._verse_contexts.get(verse_id)
            if context is None:
                continue

            # Regenerate typological embedding
            embedder = self.embedder._get_embedder(EmbeddingDomain.TYPOLOGICAL)
            typological_embedding = embedder.embed(context)

            await self.vector_store.upsert_verse(
                verse_id=verse_id,
                embeddings={EmbeddingDomain.TYPOLOGICAL: typological_embedding},
                metadata={'verse_id': verse_id}
            )
            logger.debug(f"Updated typological embedding for {verse_id}")

    def _build_context(self, event: VerseProcessingCompleted) -> VerseContext:
        """
        Build verse context from completed event.

        In production, would query for full verse data.
        """
        # Parse verse ID
        parts = event.verse_id.split('.')
        book = parts[0] if parts else "GEN"
        chapter = int(parts[1]) if len(parts) > 1 else 1
        verse = int(parts[2]) if len(parts) > 2 else 1

        # Determine testament
        testament = "old" if book in self.OT_BOOKS else "new"

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

    def _build_minimal_context(self, verse_id: str) -> VerseContext:
        """Build minimal context for verse ID."""
        parts = verse_id.split('.')
        book = parts[0] if parts else "GEN"
        chapter = int(parts[1]) if len(parts) > 1 else 1
        verse = int(parts[2]) if len(parts) > 2 else 1

        testament = "old" if book in self.OT_BOOKS else "new"

        return VerseContext(
            verse_id=verse_id,
            text=f"Verse text for {verse_id}",  # Placeholder
            testament=testament,
            book=book,
            chapter=chapter,
            verse=verse
        )

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
                    {
                        'verse_id': ctx.verse_id,
                        'testament': ctx.testament,
                        'book': ctx.book,
                        'chapter': ctx.chapter,
                        'verse_num': ctx.verse,
                    }
                )
                for ctx, embeddings in zip(contexts, batch_embeddings)
            ]

            # Batch upsert
            await self.vector_store.batch_upsert(upsert_data)

            logger.info(f"Rebuilt embeddings for batch {i // batch_size + 1}")

    async def _clear_projection(self) -> None:
        """Clear all vector data."""
        for domain in EmbeddingDomain:
            try:
                self.vector_store.client.delete_collection(
                    collection_name=domain.collection_name
                )
            except Exception as e:
                logger.warning(f"Error deleting {domain.collection_name}: {e}")

        await self.vector_store.create_collections()
        self._verse_contexts.clear()
        self._pending_updates.clear()
        self._update_counts.clear()
        logger.info("Vector data cleared and collections recreated")

    async def rebuild(self, from_position: int = 0) -> None:
        """
        Rebuild vector store from event stream.

        Replays all events to regenerate embeddings.
        """
        logger.info(f"Rebuilding VectorProjection from position {from_position}")

        await self._clear_projection()

        async for event in self.event_store.stream_events(from_position):
            try:
                await self.handle_event(event)
                self._last_position += 1

                if self._last_position % 100 == 0:
                    logger.info(f"VectorProjection: processed {self._last_position} events")
            except Exception as e:
                logger.error(f"Error rebuilding at position {self._last_position}: {e}")

        logger.info(f"VectorProjection rebuild complete, processed {self._last_position} events")

    async def get_stats(self) -> Dict[str, Any]:
        """Get projection statistics."""
        stats = await self.vector_store.get_all_collection_stats()
        stats['projection'] = {
            'cached_contexts': len(self._verse_contexts),
            'pending_updates': len(self._pending_updates),
            'update_counts': len(self._update_counts),
            'last_position': self._last_position,
            'is_running': self._is_running,
        }
        return stats


class BatchEmbeddingProcessor:
    """
    Batch processor for efficient embedding generation.

    Accumulates verses and processes in batches for better
    GPU utilization and reduced API calls.
    """

    def __init__(
        self,
        vector_store: MultiVectorStore,
        embedder: MultiDomainEmbedder,
        batch_size: int = 50,
        flush_interval: float = 5.0
    ):
        """
        Initialize batch processor.

        Args:
            vector_store: Vector store for upserts
            embedder: Multi-domain embedder
            batch_size: Process when batch reaches this size
            flush_interval: Process after this many seconds even if batch not full
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        self._pending: List[tuple] = []  # (verse_id, context, metadata)
        self._last_flush = 0.0
        self._lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start background flush task."""
        self._flush_task = asyncio.create_task(self._periodic_flush())

    async def stop(self) -> None:
        """Stop and flush remaining."""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        await self.flush()

    async def _periodic_flush(self) -> None:
        """Periodically flush pending items."""
        while True:
            await asyncio.sleep(self.flush_interval)
            await self.flush()

    async def add(
        self,
        verse_id: str,
        context: VerseContext,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Add verse to processing queue.

        May trigger batch processing if batch is full.
        """
        async with self._lock:
            self._pending.append((verse_id, context, metadata))

            if len(self._pending) >= self.batch_size:
                await self._flush()

    async def flush(self) -> None:
        """Force flush pending verses."""
        async with self._lock:
            await self._flush()

    async def _flush(self) -> None:
        """Process pending verses."""
        if not self._pending:
            return

        batch = self._pending
        self._pending = []
        self._last_flush = asyncio.get_event_loop().time()

        logger.info(f"Processing batch of {len(batch)} verses")

        # Generate embeddings
        contexts = [item[1] for item in batch]
        embeddings_list = self.embedder.embed_batch(contexts)

        # Prepare for batch upsert
        verses = [
            (item[0], embeddings, item[2])
            for item, embeddings in zip(batch, embeddings_list)
        ]

        # Batch upsert to vector store
        await self.vector_store.batch_upsert(verses)

        logger.info(f"Batch upserted {len(verses)} verses to vector store")

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
