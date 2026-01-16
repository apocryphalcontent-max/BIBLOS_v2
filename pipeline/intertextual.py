"""
BIBLOS v2 - Intertextual Pipeline Phase

Coordinates intertextual analysis agents.
"""
import asyncio
import time
from typing import Dict, Any, Optional

from pipeline.base import (
    BasePipelinePhase,
    PhaseConfig,
    PhaseResult,
    PhaseStatus
)
from agents.intertextual import (
    SyndesmosAgent,
    HarmonikosAgent,
    AllographosAgent,
    ParadeigmaAgent,
    ToposAgent
)


class IntertextualPhase(BasePipelinePhase):
    """
    Intertextual analysis phase.

    Executes agents:
    - SYNDESMOS: Cross-reference connections
    - HARMONIKOS: Parallel passage harmonization
    - ALLOGRAPHOS: Quotation/allusion detection
    - PARADEIGMA: Example/precedent identification
    - TOPOS: Common topic/motif analysis
    """

    def __init__(self, config: Optional[PhaseConfig] = None):
        if config is None:
            config = PhaseConfig(
                name="intertextual",
                agents=["syndesmos", "harmonikos", "allographos", "paradeigma", "topos"],
                parallel=True,
                timeout_seconds=300,
                min_confidence=0.6,
                dependencies=["linguistic", "theological"]
            )
        super().__init__(config)

    async def initialize(self) -> None:
        """Initialize intertextual agents."""
        self.logger.info("Initializing intertextual phase agents")

        self._agents = {
            "syndesmos": SyndesmosAgent(),
            "harmonikos": HarmonikosAgent(),
            "allographos": AllographosAgent(),
            "paradeigma": ParadeigmaAgent(),
            "topos": ToposAgent()
        }

        self.logger.info(f"Initialized {len(self._agents)} agents")

    async def execute(
        self,
        verse_id: str,
        text: str,
        context: Dict[str, Any]
    ) -> PhaseResult:
        """Execute intertextual analysis."""
        start_time = time.time()
        agent_results = {}

        try:
            # Validate dependencies
            if not await self.validate_dependencies(context):
                return self._create_result(
                    PhaseStatus.SKIPPED,
                    {},
                    start_time,
                    error="Dependencies not satisfied - requires linguistic and theological phases"
                )

            # Build intertextual context
            intertextual_context = {
                **context,
                "linguistic_results": context.get("linguistic_results", {}),
                "theological_results": context.get("theological_results", {}),
                "agent_results": context.get("agent_results", {})
            }

            # SYNDESMOS first (foundation for cross-references)
            syndesmos_result = await self._agents["syndesmos"].extract(
                verse_id, text, intertextual_context
            )
            agent_results["syndesmos"] = {
                "data": syndesmos_result.data,
                "confidence": syndesmos_result.confidence
            }
            intertextual_context["agent_results"]["syndesmos"] = agent_results["syndesmos"]

            # ALLOGRAPHOS (quotation detection)
            allographos_result = await self._agents["allographos"].extract(
                verse_id, text, intertextual_context
            )
            agent_results["allographos"] = {
                "data": allographos_result.data,
                "confidence": allographos_result.confidence
            }
            intertextual_context["agent_results"]["allographos"] = agent_results["allographos"]

            # Remaining agents can run in parallel
            harmonikos_task = asyncio.create_task(
                self._agents["harmonikos"].extract(verse_id, text, intertextual_context)
            )
            paradeigma_task = asyncio.create_task(
                self._agents["paradeigma"].extract(verse_id, text, intertextual_context)
            )
            topos_task = asyncio.create_task(
                self._agents["topos"].extract(verse_id, text, intertextual_context)
            )

            harmonikos_result, paradeigma_result, topos_result = await asyncio.gather(
                harmonikos_task, paradeigma_task, topos_task
            )

            agent_results["harmonikos"] = {
                "data": harmonikos_result.data,
                "confidence": harmonikos_result.confidence
            }
            agent_results["paradeigma"] = {
                "data": paradeigma_result.data,
                "confidence": paradeigma_result.confidence
            }
            agent_results["topos"] = {
                "data": topos_result.data,
                "confidence": topos_result.confidence
            }

            # Calculate metrics
            cross_refs = agent_results.get("syndesmos", {}).get("data", {}).get("cross_references", [])
            quotations = agent_results.get("allographos", {}).get("data", {}).get("quotations", [])
            motifs = agent_results.get("topos", {}).get("data", {}).get("motifs", [])

            metrics = {
                "phase_confidence": self.calculate_phase_confidence(agent_results),
                "agents_executed": len(agent_results),
                "cross_reference_count": len(cross_refs),
                "quotation_count": len(quotations),
                "motif_count": len(motifs)
            }

            return self._create_result(
                PhaseStatus.COMPLETED,
                agent_results,
                start_time,
                metrics=metrics
            )

        except asyncio.TimeoutError:
            return self._create_result(
                PhaseStatus.FAILED,
                agent_results,
                start_time,
                error=f"Phase timeout after {self.config.timeout_seconds}s"
            )
        except Exception as e:
            self.logger.error(f"Intertextual phase failed: {e}")
            return self._create_result(
                PhaseStatus.FAILED,
                agent_results,
                start_time,
                error=str(e)
            )

    async def cleanup(self) -> None:
        """Cleanup phase resources."""
        self._agents.clear()
        self.logger.info("Intertextual phase cleaned up")
