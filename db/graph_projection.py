"""
Neo4j Graph Projection from Event Store

Builds the SPIDERWEB graph as a projection of the event stream.
Listens to events and updates the graph in real-time.
"""
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from db.events import (
    BaseEvent,
    EventType,
    VerseProcessingCompleted,
    CrossReferenceValidated,
    OmniResolutionComputed,
    NecessityCalculated,
    LXXDivergenceDetected,
    TypologyDiscovered,
    PatristicWitnessAdded,
)
from db.projections import ProjectionBase
from db.event_store import EventStore


logger = logging.getLogger(__name__)


@dataclass
class GraphProjectionConfig:
    """Configuration for graph projection."""
    enable_centrality_updates: bool = True
    enable_similarity_edges: bool = True
    batch_size: int = 100
    min_confidence: float = 0.7


class GraphProjection(ProjectionBase):
    """
    Event-driven projection that builds SPIDERWEB graph in Neo4j.

    Node Types:
    - Verse: Biblical verses with text, testament, book, chapter
    - Word: Lexical entries with lemma, morph, semantic fields
    - CrossReference: Connection metadata
    - PatristicWitness: Church Father interpretations
    - Theme: Thematic categories

    Relationship Types:
    - CONTAINS_WORD: Verse->Word
    - REFERENCES: Verse->Verse (cross-reference)
    - TYPOLOGICALLY_FULFILLS: Type->Antitype
    - QUOTED_BY: OT->NT quotation
    - WITNESSED_BY: Verse->PatristicWitness
    - BELONGS_TO_THEME: Verse->Theme
    """

    def __init__(
        self,
        db_pool,  # Neo4j driver
        event_store: EventStore,
        config: Optional[GraphProjectionConfig] = None
    ):
        """
        Initialize graph projection.

        Args:
            db_pool: Neo4j AsyncDriver
            event_store: Event store
            config: Configuration options
        """
        self.neo4j_driver = db_pool
        self.event_store = event_store
        self.config = config or GraphProjectionConfig()
        self.projection_name = "GraphProjection"
        self._last_position = 0
        self._is_running = False

    async def initialize(self) -> None:
        """
        Initialize graph schema with constraints and indexes.

        Creates:
        - Uniqueness constraints
        - Property indexes
        - Full-text search indexes
        """
        async with self.neo4j_driver.session() as session:
            # Verse constraints and indexes
            await session.run("""
                CREATE CONSTRAINT verse_id_unique IF NOT EXISTS
                FOR (v:Verse) REQUIRE v.id IS UNIQUE
            """)
            await session.run("""
                CREATE INDEX verse_testament IF NOT EXISTS
                FOR (v:Verse) ON (v.testament)
            """)
            await session.run("""
                CREATE INDEX verse_book IF NOT EXISTS
                FOR (v:Verse) ON (v.book)
            """)

            # Word constraints and indexes
            await session.run("""
                CREATE CONSTRAINT word_id_unique IF NOT EXISTS
                FOR (w:Word) REQUIRE w.id IS UNIQUE
            """)
            await session.run("""
                CREATE INDEX word_lemma IF NOT EXISTS
                FOR (w:Word) ON (w.lemma)
            """)

            # CrossReference indexes
            await session.run("""
                CREATE INDEX crossref_type IF NOT EXISTS
                FOR ()-[r:REFERENCES]-() ON (r.connection_type)
            """)
            await session.run("""
                CREATE INDEX crossref_confidence IF NOT EXISTS
                FOR ()-[r:REFERENCES]-() ON (r.confidence)
            """)

            # PatristicWitness constraints
            await session.run("""
                CREATE CONSTRAINT patristic_id_unique IF NOT EXISTS
                FOR (p:PatristicWitness) REQUIRE p.id IS UNIQUE
            """)

            # Full-text search indexes
            await session.run("""
                CREATE FULLTEXT INDEX verse_text_search IF NOT EXISTS
                FOR (v:Verse) ON EACH [v.text_hebrew, v.text_greek, v.text_english]
            """)

            logger.info("Graph schema initialized with constraints and indexes")

    async def handle_event(self, event: BaseEvent) -> None:
        """
        Handle event and update graph.

        Routes events to appropriate handlers based on type.
        """
        handlers = {
            EventType.VERSE_PROCESSING_COMPLETED.value: self._handle_verse_completed,
            EventType.CROSS_REFERENCE_VALIDATED.value: self._handle_cross_reference,
            EventType.OMNI_RESOLUTION_COMPUTED.value: self._handle_word_analysis,
            EventType.TYPOLOGY_DISCOVERED.value: self._handle_typology,
            EventType.PATRISTIC_WITNESS_ADDED.value: self._handle_patristic,
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
        Create or update Verse node when processing completes.

        Args:
            event: Verse processing completed event
        """
        async with self.neo4j_driver.session() as session:
            await session.run("""
                MERGE (v:Verse {id: $verse_id})
                SET v.status = 'completed',
                    v.quality_tier = $quality_tier,
                    v.cross_reference_count = $cross_reference_count,
                    v.updated_at = datetime($timestamp)
            """, {
                "verse_id": event.verse_id,
                "quality_tier": event.quality_tier,
                "cross_reference_count": event.cross_reference_count,
                "timestamp": event.timestamp.isoformat()
            })

    async def _handle_cross_reference(
        self,
        event: CrossReferenceValidated
    ) -> None:
        """
        Create REFERENCES relationship between verses.

        Args:
            event: Cross-reference validated event
        """
        if event.final_confidence < self.config.min_confidence:
            return

        async with self.neo4j_driver.session() as session:
            await session.run("""
                MERGE (source:Verse {id: $source_ref})
                MERGE (target:Verse {id: $target_ref})
                MERGE (source)-[r:REFERENCES {type: $connection_type}]->(target)
                SET r.confidence = $confidence,
                    r.theological_score = $theological_score,
                    r.validators = $validators,
                    r.created_at = datetime($timestamp)
            """, {
                "source_ref": event.source_ref,
                "target_ref": event.target_ref,
                "connection_type": event.connection_type,
                "confidence": event.final_confidence,
                "theological_score": event.theological_score,
                "validators": event.validators,
                "timestamp": event.timestamp.isoformat()
            })

    async def _handle_word_analysis(
        self,
        event: OmniResolutionComputed
    ) -> None:
        """
        Create Word node and CONTAINS_WORD relationship.

        Args:
            event: Omni-resolution computed event
        """
        async with self.neo4j_driver.session() as session:
            # Create word node
            word_id = f"{event.verse_id}:{event.word}"
            await session.run("""
                MERGE (w:Word {id: $word_id})
                SET w.lemma = $word,
                    w.language = $language,
                    w.primary_meaning = $primary_meaning,
                    w.total_occurrences = $total_occurrences,
                    w.confidence = $confidence
            """, {
                "word_id": word_id,
                "word": event.word,
                "language": event.language,
                "primary_meaning": event.primary_meaning,
                "total_occurrences": event.total_occurrences,
                "confidence": event.confidence
            })

            # Link verse to word
            await session.run("""
                MATCH (v:Verse {id: $verse_id})
                MATCH (w:Word {id: $word_id})
                MERGE (v)-[r:CONTAINS_WORD]->(w)
                SET r.semantic_field_map = $semantic_field_map
            """, {
                "verse_id": event.verse_id,
                "word_id": word_id,
                "semantic_field_map": event.semantic_field_map
            })

    async def _handle_typology(self, event: TypologyDiscovered) -> None:
        """
        Create TYPOLOGICALLY_FULFILLS relationship.

        Args:
            event: Typology discovered event
        """
        async with self.neo4j_driver.session() as session:
            await session.run("""
                MERGE (type:Verse {id: $type_ref})
                MERGE (antitype:Verse {id: $antitype_ref})
                MERGE (type)-[r:TYPOLOGICALLY_FULFILLS]->(antitype)
                SET r.composite_strength = $composite_strength,
                    r.layers = $layers,
                    r.pattern_type = $pattern_type,
                    r.created_at = datetime($timestamp)
            """, {
                "type_ref": event.type_ref,
                "antitype_ref": event.antitype_ref,
                "composite_strength": event.composite_strength,
                "layers": event.typology_layers,
                "pattern_type": event.pattern_type,
                "timestamp": event.timestamp.isoformat()
            })

    async def _handle_patristic(self, event: PatristicWitnessAdded) -> None:
        """
        Create PatristicWitness node and WITNESSED_BY relationship.

        Args:
            event: Patristic witness added event
        """
        async with self.neo4j_driver.session() as session:
            witness_id = f"{event.verse_id}:{event.father_name}"
            await session.run("""
                MERGE (p:PatristicWitness {id: $witness_id})
                SET p.father_name = $father_name,
                    p.authority_level = $authority_level,
                    p.interpretation = $interpretation,
                    p.source_reference = $source_reference

                WITH p
                MATCH (v:Verse {id: $verse_id})
                MERGE (v)-[r:WITNESSED_BY]->(p)
                SET r.created_at = datetime($timestamp)
            """, {
                "witness_id": witness_id,
                "father_name": event.father_name,
                "authority_level": event.authority_level,
                "interpretation": event.interpretation,
                "source_reference": event.source_reference,
                "verse_id": event.verse_id,
                "timestamp": event.timestamp.isoformat()
            })

    async def calculate_verse_centrality(self) -> None:
        """
        Calculate betweenness centrality for all verses.

        Uses Neo4j Graph Data Science library algorithms.
        """
        if not self.config.enable_centrality_updates:
            return

        async with self.neo4j_driver.session() as session:
            # Create in-memory graph projection
            await session.run("""
                CALL gds.graph.project.cypher(
                    'verse-network',
                    'MATCH (v:Verse) RETURN id(v) AS id',
                    'MATCH (v1:Verse)-[r:REFERENCES]->(v2:Verse)
                     RETURN id(v1) AS source, id(v2) AS target, r.confidence AS weight'
                )
            """)

            # Calculate betweenness centrality
            await session.run("""
                CALL gds.betweenness.write('verse-network', {
                    writeProperty: 'centrality_score'
                })
            """)

            # Drop projection
            await session.run("""
                CALL gds.graph.drop('verse-network')
            """)

            logger.info("Verse centrality scores updated")

    async def _clear_projection(self) -> None:
        """Clear all graph data."""
        async with self.neo4j_driver.session() as session:
            await session.run("MATCH (n) DETACH DELETE n")
            logger.info("Graph data cleared")
