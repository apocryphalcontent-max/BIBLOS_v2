"""
BIBLOS v2 - Linguistic Pipeline Phase

Coordinates linguistic analysis agents.
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
from agents.linguistic import (
    GramateusAgent,
    MorphologosAgent,
    SyntaktikosAgent,
    SemantikosAgent
)


class LinguisticPhase(BasePipelinePhase):
    """
    Linguistic analysis phase.

    Executes agents:
    - GRAMMATEUS: Textual analysis coordinator
    - MORPHOLOGOS: Morphological analysis
    - SYNTAKTIKOS: Syntactic parsing
    - SEMANTIKOS: Semantic role labeling
    """

    def __init__(self, config: Optional[PhaseConfig] = None):
        if config is None:
            config = PhaseConfig(
                name="linguistic",
                agents=["grammateus", "morphologos", "syntaktikos", "semantikos"],
                parallel=True,
                timeout_seconds=300,
                min_confidence=0.6,
                dependencies=[]
            )
        super().__init__(config)

    async def initialize(self) -> None:
        """Initialize linguistic agents."""
        self.logger.info("Initializing linguistic phase agents")

        self._agents = {
            "grammateus": GramateusAgent(),
            "morphologos": MorphologosAgent(),
            "syntaktikos": SyntaktikosAgent(),
            "semantikos": SemantikosAgent()
        }

        self.logger.info(f"Initialized {len(self._agents)} agents")

    async def execute(
        self,
        verse_id: str,
        text: str,
        context: Dict[str, Any]
    ) -> PhaseResult:
        """Execute linguistic analysis."""
        start_time = time.time()
        agent_results = {}

        try:
            # Validate dependencies
            if not await self.validate_dependencies(context):
                return self._create_result(
                    PhaseStatus.SKIPPED,
                    {},
                    start_time,
                    error="Dependencies not satisfied"
                )

            # Execute agents in dependency order
            # GRAMMATEUS first (no dependencies)
            grammateus_result = await self._agents["grammateus"].extract(
                verse_id, text, context
            )
            agent_results["grammateus"] = {
                "data": grammateus_result.data,
                "confidence": grammateus_result.confidence
            }

            # Build context for dependent agents
            linguistic_context = {
                **context,
                "linguistic_results": {"grammateus": agent_results["grammateus"]}
            }

            # MORPHOLOGOS (depends on grammateus)
            morphologos_result = await self._agents["morphologos"].extract(
                verse_id, text, linguistic_context
            )
            agent_results["morphologos"] = {
                "data": morphologos_result.data,
                "confidence": morphologos_result.confidence
            }
            linguistic_context["linguistic_results"]["morphologos"] = agent_results["morphologos"]

            # SYNTAKTIKOS (depends on grammateus, morphologos)
            syntaktikos_result = await self._agents["syntaktikos"].extract(
                verse_id, text, linguistic_context
            )
            agent_results["syntaktikos"] = {
                "data": syntaktikos_result.data,
                "confidence": syntaktikos_result.confidence
            }
            linguistic_context["linguistic_results"]["syntaktikos"] = agent_results["syntaktikos"]

            # SEMANTIKOS (depends on all above)
            semantikos_result = await self._agents["semantikos"].extract(
                verse_id, text, linguistic_context
            )
            agent_results["semantikos"] = {
                "data": semantikos_result.data,
                "confidence": semantikos_result.confidence
            }

            # Calculate metrics
            metrics = {
                "phase_confidence": self.calculate_phase_confidence(agent_results),
                "agents_executed": len(agent_results),
                "word_count": agent_results.get("grammateus", {}).get("data", {}).get("word_count", 0),
                "language": agent_results.get("morphologos", {}).get("data", {}).get("language", "unknown")
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
            self.logger.error(f"Linguistic phase failed: {e}")
            return self._create_result(
                PhaseStatus.FAILED,
                agent_results,
                start_time,
                error=str(e)
            )

    async def cleanup(self) -> None:
        """Cleanup phase resources."""
        self._agents.clear()
        self.logger.info("Linguistic phase cleaned up")
