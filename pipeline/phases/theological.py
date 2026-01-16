"""
Theological Phase

Phase 2: Theological analysis including LXX extraction and patristic integration.
"""
import time
import logging
from typing import List

from pipeline.phases.base import Phase, PhasePriority, PhaseCategory, PhaseDependency
from pipeline.context import ProcessingContext


logger = logging.getLogger(__name__)


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class TheologicalPhase(Phase):
    """
    Phase 2: Theological analysis including LXX extraction and patristic integration.
    """
    name = "theological"
    category = PhaseCategory.THEOLOGICAL
    priority = PhasePriority.HIGH
    is_critical = False  # Can degrade gracefully
    base_timeout_seconds = 60.0

    @property
    def dependencies(self) -> List[PhaseDependency]:
        return [
            PhaseDependency(
                phase_name="linguistic",
                required_outputs=["linguistic_analysis"],
                is_hard=True
            )
        ]

    @property
    def outputs(self) -> List[str]:
        return ["lxx_analysis", "patristic_witness", "embeddings"]

    async def execute(self, context: ProcessingContext) -> ProcessingContext:
        """
        Execute theological analysis phase.

        Steps:
        1. Determine testament
        2. Run LXX Christological Analysis (OT only)
        3. Gather patristic interpretations
        4. Generate theological embeddings
        """
        start_time = time.time()

        # Testament already determined in linguistic phase
        testament = context.testament or "NT"

        # LXX Christological Analysis (OT only)
        if testament == "OT":
            try:
                lxx_result = await self.orchestrator._execute_with_circuit_breaker(
                    "lxx_extractor",
                    self.orchestrator.lxx_extractor.extract_christological_content(
                        context.verse_id
                    )
                )
                context.lxx_analysis = lxx_result

                # Emit divergence event if significant
                if hasattr(lxx_result, "christological_divergence_count"):
                    if lxx_result.christological_divergence_count > 0:
                        if hasattr(self.orchestrator, 'event_publisher'):
                            from db.events import LXXDivergenceDetected
                            await self.orchestrator.event_publisher.publish(
                                LXXDivergenceDetected(
                                    aggregate_id=context.verse_id,
                                    correlation_id=context.correlation_id,
                                    verse_id=context.verse_id,
                                    divergence_count=lxx_result.christological_divergence_count,
                                    primary_divergence=getattr(lxx_result, "primary_christological_insight", "")
                                )
                            )
            except CircuitOpenError:
                context.add_warning(self.name, "LXX extractor circuit open, skipping")
            except Exception as e:
                logger.warning(f"LXX extraction failed for {context.verse_id}: {e}")
                context.add_warning(self.name, f"LXX extraction failed: {e}")

        # Gather patristic interpretations with fallback
        try:
            patristic = await self._get_patristic_interpretations(context.verse_id)
            context.patristic_witness = patristic
        except Exception as e:
            context.add_warning(self.name, f"Patristic lookup failed: {e}")
            context.patristic_witness = []

        # Generate theological embeddings (multi-domain)
        # Initialize embeddings dict if not present
        if not hasattr(context, 'embeddings') or context.embeddings is None:
            context.embeddings = {}

        # Generate patristic embedding if we have witnesses
        if context.patristic_witness:
            try:
                patristic_embedding = await self._generate_patristic_embedding(
                    context.verse_id, context.patristic_witness
                )
                context.embeddings["patristic"] = patristic_embedding
            except Exception as e:
                logger.warning(f"Patristic embedding generation failed: {e}")

        # Generate liturgical embedding if relevant
        try:
            liturgical_refs = await self._get_liturgical_references(context.verse_id)
            if liturgical_refs:
                liturgical_embedding = await self._generate_liturgical_embedding(
                    context.verse_id, liturgical_refs
                )
                context.embeddings["liturgical"] = liturgical_embedding
        except Exception as e:
            logger.debug(f"Liturgical embedding generation failed: {e}")

        duration_ms = (time.time() - start_time) * 1000
        context.phase_durations[self.name] = duration_ms

        logger.info(f"Theological analysis completed for {context.verse_id} in {duration_ms:.0f}ms")
        return context

    async def _get_patristic_interpretations(self, verse_id: str) -> List:
        """Get patristic interpretations from database."""
        # Placeholder - would query patristic database
        # Check if orchestrator has patristic_db client
        if hasattr(self.orchestrator, 'patristic_db'):
            try:
                return await self.orchestrator.patristic_db.get_interpretations(verse_id)
            except Exception as e:
                logger.warning(f"Patristic DB query failed: {e}")

        # Fallback to empty list
        return []

    async def _get_liturgical_references(self, verse_id: str) -> List[dict]:
        """Check for liturgical usage of this verse."""
        # Placeholder - would query liturgical database
        if hasattr(self.orchestrator, 'postgres'):
            try:
                return await self.orchestrator.postgres.query(
                    "SELECT * FROM liturgical_readings WHERE verse_id = $1",
                    verse_id
                )
            except Exception as e:
                logger.debug(f"Liturgical query failed: {e}")

        return []

    async def _generate_patristic_embedding(self, verse_id: str, witnesses: List) -> 'np.ndarray':
        """Generate patristic embedding."""
        # Placeholder - would use vector store's patristic embedder
        if hasattr(self.orchestrator, 'vector_store'):
            try:
                # Use domain embedder if available
                from ml.embeddings.domain_embedders import VerseContext
                import numpy as np

                # Build context
                context = VerseContext(
                    verse_id=verse_id,
                    text=f"Verse {verse_id}",  # Would get actual text
                    testament="OT" if verse_id.split(".")[0] in {"GEN", "EXO"} else "NT",
                    book=verse_id.split(".")[0],
                    chapter=int(verse_id.split(".")[1]) if len(verse_id.split(".")) > 1 else 1,
                    verse=int(verse_id.split(".")[2]) if len(verse_id.split(".")) > 2 else 1,
                    patristic_witnesses=[{"interpretation": str(w)} for w in witnesses[:3]]
                )

                # Generate embedding
                return np.random.rand(768)  # Placeholder
            except Exception as e:
                logger.warning(f"Embedding generation failed: {e}")

        import numpy as np
        return np.random.rand(768)

    async def _generate_liturgical_embedding(self, verse_id: str, liturgical_refs: List) -> 'np.ndarray':
        """Generate liturgical embedding."""
        # Placeholder
        import numpy as np
        return np.random.rand(384)
