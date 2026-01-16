"""
Intertextual Phase

Phase 3: Intertextual analysis with typology, necessity, and prophetic proving.
Coordinates the three Impossible Oracles: Typology, Necessity, and Prophetic.
"""
import asyncio
import time
import logging
from typing import List

from pipeline.phases.base import Phase, PhasePriority, PhaseCategory, PhaseDependency
from pipeline.context import ProcessingContext


logger = logging.getLogger(__name__)


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class IntertextualPhase(Phase):
    """
    Phase 3: Intertextual analysis with typology, necessity, and prophetic proving.
    Coordinates the three Impossible Oracles: Typology, Necessity, and Prophetic.
    """
    name = "intertextual"
    category = PhaseCategory.INTERTEXTUAL
    priority = PhasePriority.HIGH
    is_critical = False
    base_timeout_seconds = 90.0  # Longest phase due to multiple oracle invocations

    # Threshold for necessity calculation
    TYPOLOGY_STRENGTH_THRESHOLD = 0.7

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
                required_outputs=["lxx_analysis"],
                is_hard=False  # Soft - can proceed without LXX
            )
        ]

    @property
    def outputs(self) -> List[str]:
        return ["typological_connections", "prophetic_analysis"]

    async def execute(self, context: ProcessingContext) -> ProcessingContext:
        """
        Execute intertextual analysis phase.

        Steps:
        1. Run Fractal Typology Analysis
        2. Calculate Necessity for strong typological connections
        3. Run Prophetic Analysis (for prophetic passages)
        """
        start_time = time.time()

        # Track oracle invocations for metrics
        oracle_invocations = 0

        # Fractal Typology Analysis with circuit breaker
        try:
            typology_results = await self.orchestrator._execute_with_circuit_breaker(
                "typology_engine",
                self.orchestrator.typology_engine.discover_fractal_patterns(
                    context.verse_id
                )
            )
            oracle_invocations += 1
            context.typological_connections = typology_results

            # Emit events for significant discoveries
            for result in typology_results:
                strength = getattr(result, "composite_strength", 0.0)
                if strength > 0.5:  # Only notable connections
                    if hasattr(self.orchestrator, 'event_publisher'):
                        from db.events import TypologyDiscovered
                        await self.orchestrator.event_publisher.publish(
                            TypologyDiscovered(
                                aggregate_id=context.verse_id,
                                correlation_id=context.correlation_id,
                                type_ref=getattr(result, "type_reference", context.verse_id),
                                antitype_ref=getattr(result, "antitype_reference", ""),
                                composite_strength=strength,
                                fractal_depth=getattr(result, "fractal_depth", 0),
                                typology_layers=getattr(result, "layers", []),
                                pattern_type=getattr(result, "pattern_signature", "")
                            )
                        )
        except CircuitOpenError:
            context.add_warning(self.name, "Typology engine circuit open")
            context.typological_connections = []
        except Exception as e:
            logger.warning(f"Typology discovery failed for {context.verse_id}: {e}")
            context.add_warning(self.name, f"Typology discovery failed: {e}")
            context.typological_connections = []

        # Necessity Calculation for strong typological connections
        # Run in parallel for efficiency
        necessity_tasks = []
        for result in context.typological_connections:
            strength = getattr(result, "composite_strength", 0.0)
            if strength > self.TYPOLOGY_STRENGTH_THRESHOLD:
                testament = context.testament or "NT"
                target_verse = (
                    getattr(result, "antitype_reference", "") if testament == "OT"
                    else getattr(result, "type_reference", "")
                )
                if target_verse:
                    necessity_tasks.append(
                        self._calculate_necessity_with_tracking(
                            context.verse_id,
                            target_verse,
                            result
                        )
                    )

        if necessity_tasks:
            await asyncio.gather(*necessity_tasks, return_exceptions=True)
            oracle_invocations += len(necessity_tasks)

        # Prophetic Analysis (for prophetic passages only)
        if await self._is_prophetic_passage(context.verse_id):
            try:
                prophecy_data = await self.orchestrator.prophetic_prover.analyze_prophecy(
                    context.verse_id
                )
                context.prophetic_analysis = prophecy_data
                oracle_invocations += 1

                # Emit event if significant
                posterior = getattr(prophecy_data, "posterior_probability", 0.0)
                if posterior > 0.8:
                    if hasattr(self.orchestrator, 'event_publisher'):
                        from db.events import PropheticFulfillmentIdentified
                        await self.orchestrator.event_publisher.publish(
                            PropheticFulfillmentIdentified(
                                aggregate_id=context.verse_id,
                                correlation_id=context.correlation_id,
                                prophecy_verse=context.verse_id,
                                fulfillment_verses=getattr(prophecy_data, "fulfillment_references", []),
                                posterior_probability=posterior
                            )
                        )
            except Exception as e:
                context.add_warning(self.name, f"Prophetic analysis failed: {e}")

        # Update metrics if available
        if hasattr(self.orchestrator, '_metrics'):
            from pipeline.unified_orchestrator import OrchestratorMetric
            self.orchestrator._metrics[OrchestratorMetric.ORACLE_INVOCATIONS] += oracle_invocations

        duration_ms = (time.time() - start_time) * 1000
        context.phase_durations[self.name] = duration_ms

        logger.info(f"Intertextual analysis completed for {context.verse_id} in {duration_ms:.0f}ms")
        return context

    async def _calculate_necessity_with_tracking(
        self,
        source_verse: str,
        target_verse: str,
        typology_result
    ) -> None:
        """Calculate necessity and attach to typology result."""
        try:
            necessity = await self.orchestrator.necessity_calculator.calculate_necessity(
                source_verse,
                target_verse
            )
            typology_result.necessity_score = getattr(necessity, "necessity_score", 0.0)

            # Emit event
            if hasattr(self.orchestrator, 'event_publisher'):
                from db.events import NecessityCalculated
                await self.orchestrator.event_publisher.publish(
                    NecessityCalculated(
                        aggregate_id=source_verse,
                        correlation_id="",
                        source_verse=source_verse,
                        target_verse=target_verse,
                        necessity_score=getattr(necessity, "necessity_score", 0.0),
                        layer_breakdown=getattr(necessity, "layer_breakdown", {})
                    )
                )
        except Exception as e:
            logger.warning(f"Necessity calculation failed for {source_verse} -> {target_verse}: {e}")
            typology_result.necessity_score = 0.0  # Fallback

    async def _is_prophetic_passage(self, verse_id: str) -> bool:
        """Check if verse is in a prophetic section."""
        book = verse_id.split(".")[0]

        # Prophetic books
        prophetic_books = {
            "ISA", "JER", "EZK", "DAN", "HOS", "JOL", "AMO",
            "OBA", "JON", "MIC", "NAH", "HAB", "ZEP", "HAG", "ZEC", "MAL"
        }
        if book in prophetic_books:
            return True

        # Also check for prophetic passages in non-prophetic books
        if hasattr(self.orchestrator, 'postgres'):
            try:
                prophetic_sections = await self.orchestrator.postgres.query(
                    "SELECT 1 FROM prophetic_sections WHERE $1 BETWEEN start_verse AND end_verse",
                    verse_id
                )
                return bool(prophetic_sections)
            except Exception:
                pass

        return False
