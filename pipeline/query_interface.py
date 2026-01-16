"""
Query Interface

High-level query interface for processed data.
"""
import logging
from typing import List, Dict, Optional, Any

from pipeline.unified_orchestrator import UnifiedOrchestrator
from pipeline.golden_record import GoldenRecord


logger = logging.getLogger(__name__)


class QueryInterface:
    """
    High-level query interface for processed data.
    """

    def __init__(self, orchestrator: UnifiedOrchestrator):
        self.orchestrator = orchestrator

    async def get_verse_analysis(self, verse_id: str) -> Optional[GoldenRecord]:
        """
        Get complete analysis for a verse.

        Args:
            verse_id: Verse identifier

        Returns:
            GoldenRecord if found, None otherwise
        """
        # Check cache first
        if self.orchestrator.redis:
            try:
                cached = await self.orchestrator.redis.get(f"golden:{verse_id}")
                if cached:
                    # Would deserialize from cache
                    logger.debug(f"Cache hit for {verse_id}")
                    pass
            except Exception as e:
                logger.debug(f"Cache check failed: {e}")

        # Load from database or process
        return await self._load_or_process_verse(verse_id)

    async def find_cross_references(
        self,
        verse_id: str,
        min_confidence: float = 0.5,
        connection_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find cross-references for a verse.

        Args:
            verse_id: Source verse
            min_confidence: Minimum confidence threshold
            connection_types: Filter by connection types

        Returns:
            List of cross-reference dicts
        """
        if not self.orchestrator.neo4j:
            logger.warning("Neo4j client not available")
            return []

        try:
            return await self.orchestrator.neo4j.execute("""
                MATCH (v:Verse {id: $verse_id})-[r:REFERENCES]->(t:Verse)
                WHERE r.confidence >= $min_conf
                AND ($types IS NULL OR r.connection_type IN $types)
                RETURN t.id AS target,
                       r.connection_type AS type,
                       r.confidence AS confidence,
                       r.mutual_influence AS mutual_influence
                ORDER BY r.confidence DESC
            """, verse_id=verse_id, min_conf=min_confidence, types=connection_types)
        except Exception as e:
            logger.error(f"Failed to query cross-references: {e}")
            return []

    async def find_typological_chain(
        self,
        verse_id: str,
        pattern: Optional[str] = None
    ) -> List[Any]:
        """
        Find typological chain for a verse.

        Args:
            verse_id: Source verse
            pattern: Optional pattern filter

        Returns:
            List of typological connections
        """
        if not self.orchestrator.typology_engine:
            logger.warning("Typology engine not available")
            return []

        try:
            return await self.orchestrator.typology_engine.discover_fractal_patterns(
                verse_id
            )
        except Exception as e:
            logger.error(f"Failed to discover typological patterns: {e}")
            return []

    async def get_patristic_consensus(
        self,
        verse_id: str,
        traditions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get patristic consensus on a verse.

        Args:
            verse_id: Verse identifier
            traditions: Optional filter by traditions

        Returns:
            Consensus information dict
        """
        if not hasattr(self.orchestrator, 'patristic_db'):
            logger.warning("Patristic DB not available")
            return {
                "verse_id": verse_id,
                "interpretations": [],
                "consensus_score": 0.0,
                "dominant_sense": None
            }

        try:
            interpretations = await self.orchestrator.patristic_db.get_interpretations(
                verse_id,
                traditions=traditions
            )

            return {
                "verse_id": verse_id,
                "interpretations": interpretations,
                "consensus_score": self._calculate_consensus(interpretations),
                "dominant_sense": self._determine_dominant_sense(interpretations)
            }
        except Exception as e:
            logger.error(f"Failed to get patristic consensus: {e}")
            return {
                "verse_id": verse_id,
                "interpretations": [],
                "consensus_score": 0.0,
                "dominant_sense": None
            }

    async def prove_prophecy(
        self,
        prophecy_verses: List[str],
        prior: float = 0.5
    ) -> Optional[Any]:
        """
        Run prophetic proof calculation.

        Args:
            prophecy_verses: List of prophecy verse IDs
            prior: Prior probability of supernatural explanation

        Returns:
            PropheticProofResult if successful
        """
        if not self.orchestrator.prophetic_prover:
            logger.warning("Prophetic prover not available")
            return None

        try:
            return await self.orchestrator.prophetic_prover.prove_prophetic_necessity(
                prophecy_verses,
                prior_supernatural=prior
            )
        except Exception as e:
            logger.error(f"Failed to prove prophecy: {e}")
            return None

    async def semantic_search(
        self,
        query: str,
        strategy: str = "theological",
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Semantic search across the corpus.

        Args:
            query: Search query text
            strategy: Search strategy
            top_k: Number of results

        Returns:
            List of search results
        """
        if not self.orchestrator.vector_store:
            logger.warning("Vector store not available")
            return []

        try:
            # Embed query
            query_vector = await self._embed_query(query)

            # Get strategy weights
            weights = self._get_strategy_weights(strategy)

            # Hybrid search
            results = await self.orchestrator.vector_store.hybrid_search(
                query_vectors={"semantic": query_vector},
                weights=weights,
                top_k=top_k
            )

            return [
                {
                    "verse_id": r.verse_id,
                    "score": r.score,
                    "domain_scores": r.domain_scores,
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    async def _load_or_process_verse(self, verse_id: str) -> Optional[GoldenRecord]:
        """Load verse from database or process if not found."""
        # Would query database for existing record
        # For now, process fresh
        try:
            return await self.orchestrator.process_verse(verse_id)
        except Exception as e:
            logger.error(f"Failed to load/process verse {verse_id}: {e}")
            return None

    def _calculate_consensus(self, interpretations: List) -> float:
        """Calculate consensus score from interpretations."""
        if not interpretations:
            return 0.0
        # Simple consensus: ratio of agreement
        return min(1.0, len(interpretations) / 5.0)

    def _determine_dominant_sense(self, interpretations: List) -> Optional[str]:
        """Determine dominant interpretative sense."""
        if not interpretations:
            return None
        # Would analyze interpretations for dominant theme
        return "literal"  # Placeholder

    async def _embed_query(self, query: str):
        """Embed query text."""
        # Would use embedder
        # Placeholder
        import numpy as np
        return np.random.rand(768)

    def _get_strategy_weights(self, strategy: str) -> Dict[str, float]:
        """Get weights for search strategy."""
        strategies = {
            "theological": {"semantic": 0.4, "patristic": 0.3, "typological": 0.3},
            "semantic": {"semantic": 1.0},
            "typological": {"typological": 0.6, "semantic": 0.4},
            "liturgical": {"liturgical": 0.6, "patristic": 0.4},
        }
        return strategies.get(strategy, {"semantic": 1.0})
