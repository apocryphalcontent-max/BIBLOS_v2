"""
BIBLOS v2 - Stream-Enabled Pipeline Phases

Stream-aware versions of pipeline phases that integrate with the
Redis Streams event bus for decoupled, scalable processing.
"""
import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from pipeline.stream_consumer import (
    BaseStreamConsumer,
    StreamConsumerConfig,
    PhaseStreamConsumer,
    StreamConsumerManager,
)
from pipeline.event_bus import EventBus, get_event_bus
from pipeline.base import (
    BasePipelinePhase,
    PhaseConfig,
    PhaseResult,
    PhaseStatus,
)

# Import existing phases
from pipeline.linguistic import LinguisticPhase
from pipeline.theological import TheologicalPhase
from pipeline.intertextual import IntertextualPhase
from pipeline.validation import ValidationPhase
from pipeline.finalization import FinalizationPhase


logger = logging.getLogger("biblos.pipeline.stream_phases")


# =============================================================================
# STREAM-ENABLED PHASE BASE
# =============================================================================

class StreamEnabledPhase(BaseStreamConsumer):
    """
    Base class for phases that are natively stream-aware.

    Unlike PhaseStreamConsumer which wraps existing phases,
    this provides a template for building new stream-native phases.
    """

    def __init__(
        self,
        phase_config: PhaseConfig,
        consumer_config: Optional[StreamConsumerConfig] = None,
        event_bus: Optional[EventBus] = None,
    ):
        self.phase_config = phase_config
        self._agents: Dict[str, Any] = {}

        super().__init__(
            phase_name=phase_config.name,
            config=consumer_config,
            event_bus=event_bus,
        )

    async def _initialize_phase(self) -> None:
        """Initialize phase agents."""
        await self._initialize_agents()

    async def _cleanup_phase(self) -> None:
        """Cleanup phase agents."""
        self._agents.clear()

    async def _initialize_agents(self) -> None:
        """Override to initialize specific agents."""
        pass

    async def _execute_phase(
        self,
        verse_id: str,
        text: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute phase processing."""
        start_time = time.time()
        agent_results = {}
        metrics = {}

        try:
            # Execute agents
            agent_results = await self._execute_agents(verse_id, text, context)

            # Calculate confidence
            confidence = self._calculate_confidence(agent_results)
            metrics["phase_confidence"] = confidence
            metrics["agents_executed"] = len(agent_results)

            return {
                "status": "completed",
                "agent_results": agent_results,
                "metrics": metrics,
                "duration": time.time() - start_time,
            }

        except Exception as e:
            self.logger.error(f"Phase execution failed: {e}")
            return {
                "status": "failed",
                "agent_results": agent_results,
                "metrics": metrics,
                "duration": time.time() - start_time,
                "error": str(e),
            }

    async def _execute_agents(
        self,
        verse_id: str,
        text: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Override to execute specific agents."""
        return {}

    def _calculate_confidence(
        self,
        agent_results: Dict[str, Any],
    ) -> float:
        """Calculate overall phase confidence."""
        if not agent_results:
            return 0.0

        confidences = []
        for result in agent_results.values():
            if isinstance(result, dict) and "confidence" in result:
                confidences.append(result["confidence"])

        return sum(confidences) / len(confidences) if confidences else 0.5


# =============================================================================
# STREAM-ENABLED LINGUISTIC PHASE
# =============================================================================

class StreamLinguisticPhase(StreamEnabledPhase):
    """
    Stream-native linguistic analysis phase.

    Demonstrates how to build a phase that's designed from the ground
    up to work with Redis Streams.
    """

    def __init__(
        self,
        consumer_config: Optional[StreamConsumerConfig] = None,
        event_bus: Optional[EventBus] = None,
    ):
        phase_config = PhaseConfig(
            name="linguistic",
            agents=["grammateus", "morphologos", "syntaktikos", "semantikos"],
            parallel=True,
            timeout_seconds=300,
            min_confidence=0.6,
            dependencies=[],
        )

        super().__init__(
            phase_config=phase_config,
            consumer_config=consumer_config,
            event_bus=event_bus,
        )

    async def _initialize_agents(self) -> None:
        """Initialize linguistic agents."""
        from agents.linguistic import (
            GramateusAgent,
            MorphologosAgent,
            SyntaktikosAgent,
            SemantikosAgent,
        )

        self._agents = {
            "grammateus": GramateusAgent(),
            "morphologos": MorphologosAgent(),
            "syntaktikos": SyntaktikosAgent(),
            "semantikos": SemantikosAgent(),
        }

        self.logger.info(f"Initialized {len(self._agents)} linguistic agents")

    async def _execute_agents(
        self,
        verse_id: str,
        text: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute linguistic agents in dependency order."""
        agent_results = {}
        linguistic_context = {**context, "linguistic_results": {}}

        # GRAMMATEUS first (no dependencies)
        grammateus_result = await self._agents["grammateus"].extract(
            verse_id, text, linguistic_context
        )
        agent_results["grammateus"] = {
            "data": grammateus_result.data,
            "confidence": grammateus_result.confidence,
        }
        linguistic_context["linguistic_results"]["grammateus"] = agent_results["grammateus"]

        # MORPHOLOGOS (depends on grammateus)
        morphologos_result = await self._agents["morphologos"].extract(
            verse_id, text, linguistic_context
        )
        agent_results["morphologos"] = {
            "data": morphologos_result.data,
            "confidence": morphologos_result.confidence,
        }
        linguistic_context["linguistic_results"]["morphologos"] = agent_results["morphologos"]

        # SYNTAKTIKOS (depends on grammateus, morphologos)
        syntaktikos_result = await self._agents["syntaktikos"].extract(
            verse_id, text, linguistic_context
        )
        agent_results["syntaktikos"] = {
            "data": syntaktikos_result.data,
            "confidence": syntaktikos_result.confidence,
        }
        linguistic_context["linguistic_results"]["syntaktikos"] = agent_results["syntaktikos"]

        # SEMANTIKOS (depends on all above)
        semantikos_result = await self._agents["semantikos"].extract(
            verse_id, text, linguistic_context
        )
        agent_results["semantikos"] = {
            "data": semantikos_result.data,
            "confidence": semantikos_result.confidence,
        }

        return agent_results


# =============================================================================
# PHASE FACTORY
# =============================================================================

class StreamPhaseFactory:
    """
    Factory for creating stream-enabled phase consumers.

    Provides two modes:
    1. Wrap existing phase implementations (PhaseStreamConsumer)
    2. Use stream-native implementations (StreamEnabledPhase)
    """

    # Phase class registry
    EXISTING_PHASES = {
        "linguistic": LinguisticPhase,
        "theological": TheologicalPhase,
        "intertextual": IntertextualPhase,
        "validation": ValidationPhase,
        "finalization": FinalizationPhase,
    }

    STREAM_NATIVE_PHASES = {
        "linguistic": StreamLinguisticPhase,
        # Add more as they're implemented
    }

    @classmethod
    def create_consumer(
        cls,
        phase_name: str,
        use_native: bool = False,
        config: Optional[StreamConsumerConfig] = None,
        event_bus: Optional[EventBus] = None,
    ) -> BaseStreamConsumer:
        """
        Create a stream consumer for a phase.

        Args:
            phase_name: Name of the phase
            use_native: Use stream-native implementation if available
            config: Consumer configuration
            event_bus: Event bus instance

        Returns:
            Stream consumer for the phase
        """
        # Try native implementation first
        if use_native and phase_name in cls.STREAM_NATIVE_PHASES:
            return cls.STREAM_NATIVE_PHASES[phase_name](
                consumer_config=config,
                event_bus=event_bus,
            )

        # Fall back to wrapping existing phase
        if phase_name in cls.EXISTING_PHASES:
            phase = cls.EXISTING_PHASES[phase_name]()
            return PhaseStreamConsumer(
                phase=phase,
                config=config,
                event_bus=event_bus,
            )

        raise ValueError(f"Unknown phase: {phase_name}")

    @classmethod
    async def create_all_consumers(
        cls,
        phases: Optional[List[str]] = None,
        use_native: bool = False,
        config: Optional[StreamConsumerConfig] = None,
    ) -> StreamConsumerManager:
        """
        Create consumers for all specified phases.

        Args:
            phases: List of phase names (None for all)
            use_native: Use stream-native implementations
            config: Consumer configuration

        Returns:
            Consumer manager with all phases registered
        """
        phases = phases or list(cls.EXISTING_PHASES.keys())

        manager = StreamConsumerManager(config=config)

        event_bus = await get_event_bus()

        for phase_name in phases:
            consumer = cls.create_consumer(
                phase_name=phase_name,
                use_native=use_native,
                config=config,
                event_bus=event_bus,
            )
            manager._consumers[phase_name] = consumer

        return manager


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def start_all_phase_consumers(
    phases: Optional[List[str]] = None,
    config: Optional[StreamConsumerConfig] = None,
) -> StreamConsumerManager:
    """
    Start stream consumers for all pipeline phases.

    Args:
        phases: List of phase names (None for all)
        config: Consumer configuration

    Returns:
        Running consumer manager
    """
    manager = await StreamPhaseFactory.create_all_consumers(
        phases=phases,
        config=config,
    )

    await manager.initialize()
    await manager.start()

    return manager


async def start_phase_consumer(
    phase_name: str,
    config: Optional[StreamConsumerConfig] = None,
) -> BaseStreamConsumer:
    """
    Start a stream consumer for a single phase.

    Args:
        phase_name: Name of the phase
        config: Consumer configuration

    Returns:
        Running phase consumer
    """
    consumer = StreamPhaseFactory.create_consumer(
        phase_name=phase_name,
        config=config,
    )

    await consumer.initialize()
    await consumer.start()

    return consumer
