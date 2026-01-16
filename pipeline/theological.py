"""
BIBLOS v2 - Theological Pipeline Phase

Coordinates theological analysis agents.
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
from agents.theological import (
    PatrologosAgent,
    TypologosAgent,
    TheologosAgent,
    LiturgikosAgent,
    DogmatikosAgent
)


class TheologicalPhase(BasePipelinePhase):
    """
    Theological analysis phase.

    Executes agents:
    - PATROLOGOS: Patristic interpretation
    - TYPOLOGOS: Typological connections
    - THEOLOGOS: Systematic theology
    - LITURGIKOS: Liturgical usage
    - DOGMATIKOS: Dogmatic analysis
    """

    def __init__(self, config: Optional[PhaseConfig] = None):
        if config is None:
            config = PhaseConfig(
                name="theological",
                agents=["patrologos", "typologos", "theologos", "liturgikos", "dogmatikos"],
                parallel=True,
                timeout_seconds=300,
                min_confidence=0.6,
                dependencies=["linguistic"]
            )
        super().__init__(config)

    async def initialize(self) -> None:
        """Initialize theological agents."""
        self.logger.info("Initializing theological phase agents")

        self._agents = {
            "patrologos": PatrologosAgent(),
            "typologos": TypologosAgent(),
            "theologos": TheologosAgent(),
            "liturgikos": LiturgikosAgent(),
            "dogmatikos": DogmatikosAgent()
        }

        self.logger.info(f"Initialized {len(self._agents)} agents")

    async def execute(
        self,
        verse_id: str,
        text: str,
        context: Dict[str, Any]
    ) -> PhaseResult:
        """Execute theological analysis."""
        start_time = time.time()
        agent_results = {}

        try:
            # Validate dependencies
            if not await self.validate_dependencies(context):
                return self._create_result(
                    PhaseStatus.SKIPPED,
                    {},
                    start_time,
                    error="Dependencies not satisfied - requires linguistic phase"
                )

            # Build theological context
            theological_context = {
                **context,
                "linguistic_results": context.get("linguistic_results", {}),
                "agent_results": context.get("agent_results", {})
            }

            # Execute agents - some can run in parallel
            # THEOLOGOS and PATROLOGOS can run in parallel
            theologos_task = asyncio.create_task(
                self._agents["theologos"].extract(verse_id, text, theological_context)
            )
            patrologos_task = asyncio.create_task(
                self._agents["patrologos"].extract(verse_id, text, theological_context)
            )

            theologos_result, patrologos_result = await asyncio.gather(
                theologos_task, patrologos_task
            )

            agent_results["theologos"] = {
                "data": theologos_result.data,
                "confidence": theologos_result.confidence
            }
            agent_results["patrologos"] = {
                "data": patrologos_result.data,
                "confidence": patrologos_result.confidence
            }

            # Update context
            theological_context["agent_results"].update(agent_results)

            # TYPOLOGOS (benefits from patrologos)
            typologos_result = await self._agents["typologos"].extract(
                verse_id, text, theological_context
            )
            agent_results["typologos"] = {
                "data": typologos_result.data,
                "confidence": typologos_result.confidence
            }
            theological_context["agent_results"]["typologos"] = agent_results["typologos"]

            # LITURGIKOS and DOGMATIKOS can run in parallel
            liturgikos_task = asyncio.create_task(
                self._agents["liturgikos"].extract(verse_id, text, theological_context)
            )
            dogmatikos_task = asyncio.create_task(
                self._agents["dogmatikos"].extract(verse_id, text, theological_context)
            )

            liturgikos_result, dogmatikos_result = await asyncio.gather(
                liturgikos_task, dogmatikos_task
            )

            agent_results["liturgikos"] = {
                "data": liturgikos_result.data,
                "confidence": liturgikos_result.confidence
            }
            agent_results["dogmatikos"] = {
                "data": dogmatikos_result.data,
                "confidence": dogmatikos_result.confidence
            }

            # Calculate metrics
            metrics = {
                "phase_confidence": self.calculate_phase_confidence(agent_results),
                "agents_executed": len(agent_results),
                "themes_identified": len(
                    agent_results.get("patrologos", {}).get("data", {}).get("themes", [])
                ),
                "typological_connections": len(
                    agent_results.get("typologos", {}).get("data", {}).get("connections", [])
                )
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
            self.logger.error(f"Theological phase failed: {e}")
            return self._create_result(
                PhaseStatus.FAILED,
                agent_results,
                start_time,
                error=str(e)
            )

    async def cleanup(self) -> None:
        """Cleanup phase resources."""
        self._agents.clear()
        self.logger.info("Theological phase cleaned up")
