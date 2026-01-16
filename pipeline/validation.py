"""
BIBLOS v2 - Validation Pipeline Phase

Coordinates validation and quality assurance agents.
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
from agents.validation import (
    ElenktikosAgent,
    KritikosAgent,
    HarmonizerAgent,
    ProsecutorAgent,
    WitnessAgent
)


class ValidationPhase(BasePipelinePhase):
    """
    Validation and quality assurance phase.

    Executes agents:
    - ELENKTIKOS: Cross-agent consistency
    - KRITIKOS: Quality scoring
    - HARMONIZER: Result harmonization
    - PROSECUTOR: Challenge generation
    - WITNESS: Defense responses
    """

    def __init__(self, config: Optional[PhaseConfig] = None):
        if config is None:
            config = PhaseConfig(
                name="validation",
                agents=["elenktikos", "kritikos", "harmonizer", "prosecutor", "witness"],
                parallel=False,  # Sequential for dependency reasons
                timeout_seconds=300,
                min_confidence=0.7,
                dependencies=["linguistic", "theological", "intertextual"]
            )
        super().__init__(config)

    async def initialize(self) -> None:
        """Initialize validation agents."""
        self.logger.info("Initializing validation phase agents")

        self._agents = {
            "elenktikos": ElenktikosAgent(),
            "kritikos": KritikosAgent(),
            "harmonizer": HarmonizerAgent(),
            "prosecutor": ProsecutorAgent(),
            "witness": WitnessAgent()
        }

        self.logger.info(f"Initialized {len(self._agents)} agents")

    async def execute(
        self,
        verse_id: str,
        text: str,
        context: Dict[str, Any]
    ) -> PhaseResult:
        """Execute validation analysis."""
        start_time = time.time()
        agent_results = {}

        try:
            # Validate dependencies
            if not await self.validate_dependencies(context):
                return self._create_result(
                    PhaseStatus.SKIPPED,
                    {},
                    start_time,
                    error="Dependencies not satisfied - requires all extraction phases"
                )

            # Build validation context with all prior results
            validation_context = {
                **context,
                "agent_results": context.get("agent_results", {})
            }

            # ELENKTIKOS - Check consistency
            elenktikos_result = await self._agents["elenktikos"].extract(
                verse_id, text, validation_context
            )
            agent_results["elenktikos"] = {
                "data": elenktikos_result.data,
                "confidence": elenktikos_result.confidence
            }
            validation_context["consistency_report"] = elenktikos_result.data

            # KRITIKOS - Score quality
            kritikos_result = await self._agents["kritikos"].extract(
                verse_id, text, validation_context
            )
            agent_results["kritikos"] = {
                "data": kritikos_result.data,
                "confidence": kritikos_result.confidence
            }
            validation_context["quality_report"] = kritikos_result.data

            # HARMONIZER - Harmonize results
            harmonizer_result = await self._agents["harmonizer"].extract(
                verse_id, text, validation_context
            )
            agent_results["harmonizer"] = {
                "data": harmonizer_result.data,
                "confidence": harmonizer_result.confidence
            }
            validation_context["harmonized"] = harmonizer_result.data

            # PROSECUTOR - Generate challenges
            prosecutor_result = await self._agents["prosecutor"].extract(
                verse_id, text, validation_context
            )
            agent_results["prosecutor"] = {
                "data": prosecutor_result.data,
                "confidence": prosecutor_result.confidence
            }
            validation_context["challenges"] = prosecutor_result.data.get("challenges", [])

            # WITNESS - Respond to challenges
            witness_result = await self._agents["witness"].extract(
                verse_id, text, validation_context
            )
            agent_results["witness"] = {
                "data": witness_result.data,
                "confidence": witness_result.confidence
            }

            # Calculate validation metrics
            consistency = agent_results.get("elenktikos", {}).get("data", {})
            quality = agent_results.get("kritikos", {}).get("data", {})

            metrics = {
                "phase_confidence": self.calculate_phase_confidence(agent_results),
                "agents_executed": len(agent_results),
                "validation_passed": consistency.get("validation_passed", False),
                "quality_passed": quality.get("quality_passed", False),
                "conflicts_resolved": agent_results.get("harmonizer", {}).get("data", {}).get("conflicts_resolved", 0),
                "challenges_count": len(validation_context.get("challenges", []))
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
            self.logger.error(f"Validation phase failed: {e}")
            return self._create_result(
                PhaseStatus.FAILED,
                agent_results,
                start_time,
                error=str(e)
            )

    async def cleanup(self) -> None:
        """Cleanup phase resources."""
        self._agents.clear()
        self.logger.info("Validation phase cleaned up")
