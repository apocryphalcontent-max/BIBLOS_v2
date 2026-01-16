"""
Cross-Reference Phase

Phase 4: Cross-reference discovery with multi-vector search and GNN refinement.
Integrates vector similarity, graph structure, and mutual transformation analysis.
"""
import asyncio
import time
import logging
from typing import List, Dict, Any
from enum import Enum
import numpy as np

from pipeline.phases.base import Phase, PhasePriority, PhaseCategory, PhaseDependency
from pipeline.context import ProcessingContext


logger = logging.getLogger(__name__)


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class CrossRefStrategy(Enum):
    """Strategy for cross-reference discovery."""
    SEMANTIC = "semantic"
    TYPOLOGICAL = "typological"
    PROPHETIC = "prophetic"
    LITURGICAL = "liturgical"
    BALANCED = "balanced"

    @property
    def domain_weights(self) -> Dict[str, float]:
        """Get embedding domain weights for this strategy."""
        return {
            CrossRefStrategy.SEMANTIC: {"semantic": 0.8, "patristic": 0.2},
            CrossRefStrategy.TYPOLOGICAL: {"typological": 0.5, "semantic": 0.3, "patristic": 0.2},
            CrossRefStrategy.PROPHETIC: {"prophetic": 0.5, "semantic": 0.3, "typological": 0.2},
            CrossRefStrategy.LITURGICAL: {"liturgical": 0.6, "semantic": 0.3, "patristic": 0.1},
            CrossRefStrategy.BALANCED: {"semantic": 0.25, "patristic": 0.25, "typological": 0.25, "prophetic": 0.25},
        }[self]


class CrossReferencePhase(Phase):
    """
    Phase 4: Cross-reference discovery with multi-vector search and GNN refinement.
    Integrates vector similarity, graph structure, and mutual transformation analysis.
    """
    name = "cross_reference"
    category = PhaseCategory.CROSS_REFERENCE
    priority = PhasePriority.NORMAL
    is_critical = False
    base_timeout_seconds = 60.0

    # Configuration
    INITIAL_CANDIDATES = 50
    GNN_REFINED_TOP_K = 20
    MIN_MUTUAL_INFLUENCE = 0.3

    @property
    def dependencies(self) -> List[PhaseDependency]:
        return [
            PhaseDependency(
                phase_name="linguistic",
                required_outputs=["linguistic_analysis"],
                is_hard=True
            ),
            PhaseDependency(
                phase_name="theological",
                required_outputs=["embeddings"],
                is_hard=False
            ),
            PhaseDependency(
                phase_name="intertextual",
                required_outputs=["typological_connections"],
                is_hard=False
            )
        ]

    @property
    def outputs(self) -> List[str]:
        return ["cross_references"]

    async def execute(self, context: ProcessingContext) -> ProcessingContext:
        """
        Execute cross-reference discovery phase.

        Steps:
        1. Select query strategy based on context
        2. Build query vectors
        3. Multi-vector hybrid search
        4. GNN refinement
        5. Calculate mutual transformation
        6. Filter by minimum mutual influence
        """
        start_time = time.time()

        # Select strategy based on context
        strategy = self._select_query_strategy(context)
        logger.debug(f"Using cross-reference strategy: {strategy.value}")

        # Build query vectors
        query_vectors = await self._build_query_vectors(context)

        # Multi-vector hybrid search with strategy-specific weights
        candidates = []
        try:
            candidates = await self.orchestrator._execute_with_circuit_breaker(
                "vector_store",
                self.orchestrator.vector_store.hybrid_search(
                    query_vectors=query_vectors,
                    weights=strategy.domain_weights,
                    top_k=self.INITIAL_CANDIDATES
                )
            )
        except CircuitOpenError:
            context.add_warning(self.name, "Vector store circuit open, using graph-only")
            candidates = await self._fallback_graph_candidates(context.verse_id)
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            context.add_warning(self.name, f"Vector search failed: {e}")
            candidates = []

        # Capture pre-GNN embeddings for mutual transformation
        pre_gnn_embeddings = await self._capture_embeddings(context.verse_id, candidates)

        # GNN refinement with circuit breaker
        refined = candidates
        if candidates:
            try:
                refined = await self.orchestrator._execute_with_circuit_breaker(
                    "gnn_model",
                    self.orchestrator.gnn_model.refine_candidates(
                        source_verse=context.verse_id,
                        candidates=candidates,
                        top_k=self.GNN_REFINED_TOP_K
                    )
                )
            except CircuitOpenError:
                context.add_warning(self.name, "GNN circuit open, using unrefined candidates")
                refined = candidates[:self.GNN_REFINED_TOP_K]
            except Exception as e:
                logger.warning(f"GNN refinement failed: {e}")
                context.add_warning(self.name, f"GNN refinement failed: {e}")
                refined = candidates[:self.GNN_REFINED_TOP_K]

        # Capture post-GNN embeddings
        post_gnn_embeddings = await self._capture_embeddings(context.verse_id, refined)

        # Calculate mutual transformation in parallel
        mt_tasks = [
            self._calculate_mutual_transformation(
                context.verse_id,
                candidate,
                pre_gnn_embeddings,
                post_gnn_embeddings
            )
            for candidate in refined
        ]
        if mt_tasks:
            await asyncio.gather(*mt_tasks, return_exceptions=True)

        # Filter by minimum mutual influence
        context.cross_references = [
            c for c in refined
            if getattr(c, "mutual_influence", 0.0) >= self.MIN_MUTUAL_INFLUENCE
        ]

        # Emit discovery events for significant finds
        for candidate in context.cross_references:
            if hasattr(self.orchestrator, 'event_publisher'):
                from db.events import CrossReferenceDiscovered
                await self.orchestrator.event_publisher.publish(
                    CrossReferenceDiscovered(
                        aggregate_id=context.verse_id,
                        correlation_id=context.correlation_id,
                        source_ref=context.verse_id,
                        target_ref=getattr(candidate, "target_ref", getattr(candidate, "verse_id", "")),
                        connection_type=getattr(candidate, "connection_type", "semantic"),
                        confidence=getattr(candidate, "confidence", getattr(candidate, "score", 0.0)),
                        discovery_method=strategy.value
                    )
                )

        # Update metrics
        if hasattr(self.orchestrator, '_metrics'):
            from pipeline.unified_orchestrator import OrchestratorMetric
            self.orchestrator._metrics[OrchestratorMetric.CROSS_REFS_DISCOVERED] += len(context.cross_references)

        duration_ms = (time.time() - start_time) * 1000
        context.phase_durations[self.name] = duration_ms

        logger.info(f"Cross-reference discovery completed for {context.verse_id}: {len(context.cross_references)} refs in {duration_ms:.0f}ms")
        return context

    def _select_query_strategy(self, context: ProcessingContext) -> CrossRefStrategy:
        """Select optimal query strategy based on context."""
        # Prophetic passages use prophetic strategy
        if context.prophetic_analysis:
            return CrossRefStrategy.PROPHETIC

        # Strong typological connections use typological strategy
        if context.typological_connections:
            strong_typo = any(
                getattr(t, "composite_strength", 0.0) > 0.7
                for t in context.typological_connections
            )
            if strong_typo:
                return CrossRefStrategy.TYPOLOGICAL

        # Liturgical embedding present suggests liturgical strategy
        if "liturgical" in context.embeddings:
            return CrossRefStrategy.LITURGICAL

        # Default to balanced
        return CrossRefStrategy.BALANCED

    async def _build_query_vectors(self, context: ProcessingContext) -> Dict[str, np.ndarray]:
        """Build query vectors from available embeddings."""
        vectors = {}

        # Use pre-computed embeddings from context
        for domain, embedding in context.embeddings.items():
            vectors[domain] = embedding

        # Generate semantic embedding if missing
        if "semantic" not in vectors:
            try:
                if hasattr(self.orchestrator, 'vector_store'):
                    # Would get verse text and generate embedding
                    vectors["semantic"] = np.random.rand(768)  # Placeholder
            except Exception as e:
                logger.warning(f"Failed to generate semantic embedding: {e}")

        return vectors

    async def _capture_embeddings(
        self,
        source_id: str,
        candidates: List
    ) -> Dict[str, np.ndarray]:
        """Capture embeddings for source and all candidates."""
        embeddings = {}

        # Get source embedding
        try:
            if hasattr(self.orchestrator, 'vector_store'):
                from ml.embeddings.multi_vector_store import EmbeddingDomain
                embedding = await self.orchestrator.vector_store.get_embedding(
                    EmbeddingDomain.SEMANTIC,
                    source_id
                )
                if embedding is not None:
                    embeddings[source_id] = embedding
        except Exception as e:
            logger.debug(f"Failed to capture source embedding: {e}")

        # Get candidate embeddings
        for candidate in candidates:
            target_id = getattr(candidate, "target_ref", getattr(candidate, "verse_id", ""))
            if target_id:
                try:
                    if hasattr(self.orchestrator, 'vector_store'):
                        from ml.embeddings.multi_vector_store import EmbeddingDomain
                        embedding = await self.orchestrator.vector_store.get_embedding(
                            EmbeddingDomain.SEMANTIC,
                            target_id
                        )
                        if embedding is not None:
                            embeddings[target_id] = embedding
                except Exception as e:
                    logger.debug(f"Failed to capture embedding for {target_id}: {e}")

        return embeddings

    async def _calculate_mutual_transformation(
        self,
        source_verse: str,
        candidate,
        pre_embeddings: Dict,
        post_embeddings: Dict
    ) -> None:
        """Calculate and attach mutual transformation metrics."""
        try:
            target_verse = getattr(candidate, "target_ref", getattr(candidate, "verse_id", ""))

            mt_result = await self.orchestrator.mutual_transformation.measure_transformation(
                source_verse=source_verse,
                target_verse=target_verse,
                source_before=pre_embeddings.get(source_verse),
                source_after=post_embeddings.get(source_verse),
                target_before=pre_embeddings.get(target_verse),
                target_after=post_embeddings.get(target_verse)
            )
            candidate.mutual_influence = getattr(mt_result, "mutual_influence", 0.0)
            candidate.transformation_type = getattr(mt_result, "transformation_type", "unknown")
        except Exception as e:
            logger.debug(f"Mutual transformation calculation failed: {e}")
            candidate.mutual_influence = 0.0
            candidate.transformation_type = "unknown"

    async def _fallback_graph_candidates(self, verse_id: str) -> List:
        """Fallback to graph-based candidates when vector store unavailable."""
        if not hasattr(self.orchestrator, 'neo4j'):
            return []

        try:
            results = await self.orchestrator.neo4j.execute("""
                MATCH (v:Verse {id: $verse_id})-[r:REFERENCES|SHARES_LEMMA|TYPOLOGICALLY_LINKED]-(t:Verse)
                RETURN t.id AS target_ref, type(r) AS connection_type, 0.5 AS confidence
                LIMIT 50
            """, verse_id=verse_id)

            # Convert to candidate objects
            candidates = []
            for result in results:
                class Candidate:
                    def __init__(self, data):
                        self.target_ref = data.get("target_ref", "")
                        self.connection_type = data.get("connection_type", "")
                        self.confidence = data.get("confidence", 0.5)
                        self.mutual_influence = 0.0
                        self.transformation_type = "graph"

                candidates.append(Candidate(result))

            return candidates
        except Exception as e:
            logger.warning(f"Graph fallback failed: {e}")
            return []
