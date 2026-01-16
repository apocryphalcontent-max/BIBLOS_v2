"""
BIBLOS v2 - Agent Orchestrator with LangChain/LangGraph Integration

Multi-agent coordination system using LangGraph for stateful workflows
and parallel agent execution.
"""
from typing import Any, Dict, List, Optional, Callable, TypedDict
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
import time

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from agents.base import (
    BaseExtractionAgent,
    ExtractionResult,
    ProcessingStatus,
    ExtractionType
)
from agents.registry import AgentRegistry, registry

# Import specific error types for proper exception handling
from core.errors import (
    BiblosError,
    BiblosAgentError,
    BiblosValidationError,
    BiblosTimeoutError,
    BiblosResourceError,
)


class WorkflowPhase(Enum):
    """Phases of the extraction workflow."""
    LINGUISTIC = "linguistic"
    THEOLOGICAL = "theological"
    INTERTEXTUAL = "intertextual"
    VALIDATION = "validation"
    FINALIZATION = "finalization"


class WorkflowState(TypedDict):
    """State passed through the LangGraph workflow."""
    verse_id: str
    text: str
    context: Dict[str, Any]
    current_phase: str
    results: Dict[str, Any]
    errors: List[str]
    confidence_scores: Dict[str, float]
    metadata: Dict[str, Any]


@dataclass
class OrchestrationConfig:
    """Configuration for agent orchestration."""
    parallel_phases: bool = True
    max_parallel_agents: int = 8
    phase_timeout_seconds: int = 300
    retry_failed_agents: bool = True
    max_retries: int = 2
    min_phase_confidence: float = 0.6
    enable_validation_phase: bool = True
    checkpointing: bool = True
    checkpoint_dir: str = "./checkpoints"


class AgentOrchestrator:
    """
    Orchestrates multiple extraction agents using LangGraph workflows.

    Workflow Structure:
    1. LINGUISTIC phase: phonology, morphology, syntax, semantics
    2. THEOLOGICAL phase: patristic, typological, liturgical
    3. INTERTEXTUAL phase: cross-references, parallels, allusions
    4. VALIDATION phase: verification, correction, adversarial testing
    5. FINALIZATION: merge results, compute confidence
    """

    # Agent groupings by phase
    PHASE_AGENTS = {
        WorkflowPhase.LINGUISTIC: [
            ExtractionType.PHONOLOGICAL,
            ExtractionType.MORPHOLOGICAL,
            ExtractionType.SYNTACTIC,
            ExtractionType.SEMANTIC,
            ExtractionType.LEXICAL,
            ExtractionType.ETYMOLOGICAL,
            ExtractionType.DISCOURSE,
            ExtractionType.PRAGMATIC
        ],
        WorkflowPhase.THEOLOGICAL: [
            ExtractionType.PATRISTIC,
            ExtractionType.TYPOLOGICAL,
            ExtractionType.LITURGICAL
        ],
        WorkflowPhase.INTERTEXTUAL: [
            ExtractionType.INTERTEXTUAL,
            ExtractionType.STRUCTURAL
        ],
        WorkflowPhase.VALIDATION: [
            ExtractionType.VALIDATION
        ]
    }

    def __init__(
        self,
        config: Optional[OrchestrationConfig] = None,
        agent_registry: Optional[AgentRegistry] = None
    ):
        self.config = config or OrchestrationConfig()
        self.registry = agent_registry or registry
        self.logger = logging.getLogger("biblos.orchestrator")
        self._workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(WorkflowState)

        # Add nodes for each phase
        workflow.add_node("linguistic", self._run_linguistic_phase)
        workflow.add_node("theological", self._run_theological_phase)
        workflow.add_node("intertextual", self._run_intertextual_phase)
        workflow.add_node("validation", self._run_validation_phase)
        workflow.add_node("finalization", self._run_finalization)

        # Define edges (sequential flow)
        workflow.set_entry_point("linguistic")
        workflow.add_edge("linguistic", "theological")
        workflow.add_edge("theological", "intertextual")

        # Conditional validation
        workflow.add_conditional_edges(
            "intertextual",
            self._should_validate,
            {
                True: "validation",
                False: "finalization"
            }
        )
        workflow.add_edge("validation", "finalization")
        workflow.add_edge("finalization", END)

        return workflow.compile()

    def _should_validate(self, state: WorkflowState) -> bool:
        """Determine if validation phase should run."""
        if not self.config.enable_validation_phase:
            return False

        # Check confidence scores
        confidences = state.get("confidence_scores", {})
        if not confidences:
            return True

        avg_confidence = sum(confidences.values()) / len(confidences)
        return avg_confidence < self.config.min_phase_confidence

    async def _run_phase_agents(
        self,
        state: WorkflowState,
        phase: WorkflowPhase
    ) -> Dict[str, ExtractionResult]:
        """Run all agents for a phase, optionally in parallel."""
        results = {}
        extraction_types = self.PHASE_AGENTS[phase]

        # Gather agents for this phase
        phase_agents = []
        for ext_type in extraction_types:
            agents = self.registry.get_by_type(ext_type)
            phase_agents.extend(agents)

        if not phase_agents:
            self.logger.warning(f"No agents registered for phase {phase.name}")
            return results

        self.logger.info(f"Running {len(phase_agents)} agents for {phase.name}")

        if self.config.parallel_phases:
            # Parallel execution with semaphore
            semaphore = asyncio.Semaphore(self.config.max_parallel_agents)

            async def run_with_semaphore(agent: BaseExtractionAgent):
                async with semaphore:
                    return await agent.process(
                        state["verse_id"],
                        state["text"],
                        state["context"]
                    )

            tasks = [run_with_semaphore(agent) for agent in phase_agents]
            phase_results = await asyncio.gather(*tasks, return_exceptions=True)

            for agent, result in zip(phase_agents, phase_results):
                if isinstance(result, Exception):
                    self.logger.error(f"Agent {agent.config.name} failed: {result}")
                    results[agent.config.name] = ExtractionResult(
                        agent_name=agent.config.name,
                        extraction_type=agent.config.extraction_type,
                        verse_id=state["verse_id"],
                        status=ProcessingStatus.FAILED,
                        data={},
                        confidence=0.0,
                        processing_time_ms=0,
                        errors=[str(result)]
                    )
                else:
                    results[agent.config.name] = result
        else:
            # Sequential execution
            for agent in phase_agents:
                try:
                    result = await agent.process(
                        state["verse_id"],
                        state["text"],
                        state["context"]
                    )
                    results[agent.config.name] = result
                except asyncio.TimeoutError as e:
                    # Handle agent timeout
                    self.logger.error(f"Agent {agent.config.name} timed out")
                    results[agent.config.name] = ExtractionResult(
                        agent_name=agent.config.name,
                        extraction_type=agent.config.extraction_type,
                        verse_id=state["verse_id"],
                        status=ProcessingStatus.FAILED,
                        data={},
                        confidence=0.0,
                        processing_time_ms=0,
                        errors=["Agent execution timed out"]
                    )
                except BiblosAgentError as e:
                    # Handle agent-specific errors
                    self.logger.error(f"Agent {agent.config.name} error: {e}")
                    results[agent.config.name] = ExtractionResult(
                        agent_name=agent.config.name,
                        extraction_type=agent.config.extraction_type,
                        verse_id=state["verse_id"],
                        status=ProcessingStatus.FAILED,
                        data={},
                        confidence=0.0,
                        processing_time_ms=0,
                        errors=[str(e)]
                    )
                except BiblosValidationError as e:
                    # Handle validation errors - mark for review
                    self.logger.warning(f"Agent {agent.config.name} validation error: {e}")
                    results[agent.config.name] = ExtractionResult(
                        agent_name=agent.config.name,
                        extraction_type=agent.config.extraction_type,
                        verse_id=state["verse_id"],
                        status=ProcessingStatus.NEEDS_REVIEW,
                        data={},
                        confidence=0.0,
                        processing_time_ms=0,
                        errors=[f"Validation error: {e}"]
                    )
                except (MemoryError, BiblosResourceError) as e:
                    # Handle resource exhaustion - critical
                    self.logger.critical(f"Resource exhaustion in agent {agent.config.name}: {e}")
                    results[agent.config.name] = ExtractionResult(
                        agent_name=agent.config.name,
                        extraction_type=agent.config.extraction_type,
                        verse_id=state["verse_id"],
                        status=ProcessingStatus.FAILED,
                        data={},
                        confidence=0.0,
                        processing_time_ms=0,
                        errors=[f"Resource exhaustion: {e}"]
                    )
                except BiblosError as e:
                    # Handle other BIBLOS errors
                    self.logger.error(f"BIBLOS error in agent {agent.config.name}: {e}")
                    results[agent.config.name] = ExtractionResult(
                        agent_name=agent.config.name,
                        extraction_type=agent.config.extraction_type,
                        verse_id=state["verse_id"],
                        status=ProcessingStatus.FAILED,
                        data={},
                        confidence=0.0,
                        processing_time_ms=0,
                        errors=[str(e)]
                    )
                except Exception as e:
                    # Catch-all for unexpected errors
                    self.logger.error(f"Unexpected error in agent {agent.config.name}: {e} ({type(e).__name__})")
                    results[agent.config.name] = ExtractionResult(
                        agent_name=agent.config.name,
                        extraction_type=agent.config.extraction_type,
                        verse_id=state["verse_id"],
                        status=ProcessingStatus.FAILED,
                        data={},
                        confidence=0.0,
                        processing_time_ms=0,
                        errors=[str(e)]
                    )

        return results

    async def _run_linguistic_phase(self, state: WorkflowState) -> WorkflowState:
        """Execute linguistic analysis phase."""
        state["current_phase"] = WorkflowPhase.LINGUISTIC.value
        results = await self._run_phase_agents(state, WorkflowPhase.LINGUISTIC)

        state["results"]["linguistic"] = {
            name: r.to_dict() for name, r in results.items()
        }
        state["confidence_scores"]["linguistic"] = self._compute_phase_confidence(results)

        return state

    async def _run_theological_phase(self, state: WorkflowState) -> WorkflowState:
        """Execute theological analysis phase."""
        state["current_phase"] = WorkflowPhase.THEOLOGICAL.value

        # Add linguistic results to context
        state["context"]["linguistic_results"] = state["results"].get("linguistic", {})

        results = await self._run_phase_agents(state, WorkflowPhase.THEOLOGICAL)

        state["results"]["theological"] = {
            name: r.to_dict() for name, r in results.items()
        }
        state["confidence_scores"]["theological"] = self._compute_phase_confidence(results)

        return state

    async def _run_intertextual_phase(self, state: WorkflowState) -> WorkflowState:
        """Execute intertextual analysis phase."""
        state["current_phase"] = WorkflowPhase.INTERTEXTUAL.value

        # Add prior results to context
        state["context"]["prior_results"] = {
            "linguistic": state["results"].get("linguistic", {}),
            "theological": state["results"].get("theological", {})
        }

        results = await self._run_phase_agents(state, WorkflowPhase.INTERTEXTUAL)

        state["results"]["intertextual"] = {
            name: r.to_dict() for name, r in results.items()
        }
        state["confidence_scores"]["intertextual"] = self._compute_phase_confidence(results)

        return state

    async def _run_validation_phase(self, state: WorkflowState) -> WorkflowState:
        """Execute validation phase."""
        state["current_phase"] = WorkflowPhase.VALIDATION.value

        # Full context for validation
        state["context"]["all_results"] = state["results"]

        results = await self._run_phase_agents(state, WorkflowPhase.VALIDATION)

        state["results"]["validation"] = {
            name: r.to_dict() for name, r in results.items()
        }

        return state

    async def _run_finalization(self, state: WorkflowState) -> WorkflowState:
        """Finalize results and compute overall metrics."""
        state["current_phase"] = WorkflowPhase.FINALIZATION.value

        # Compute overall confidence
        confidences = state["confidence_scores"]
        if confidences:
            state["metadata"]["overall_confidence"] = sum(confidences.values()) / len(confidences)

        # Count successes/failures
        success_count = 0
        failure_count = 0
        for phase_results in state["results"].values():
            for result in phase_results.values():
                if isinstance(result, dict):
                    if result.get("status") == "completed":
                        success_count += 1
                    elif result.get("status") == "failed":
                        failure_count += 1

        state["metadata"]["success_count"] = success_count
        state["metadata"]["failure_count"] = failure_count
        state["metadata"]["success_rate"] = success_count / max(1, success_count + failure_count)

        return state

    def _compute_phase_confidence(self, results: Dict[str, ExtractionResult]) -> float:
        """Compute average confidence for a phase."""
        if not results:
            return 0.0

        confidences = [r.confidence for r in results.values() if r.confidence > 0]
        return sum(confidences) / len(confidences) if confidences else 0.0

    async def process_verse(
        self,
        verse_id: str,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> WorkflowState:
        """
        Process a single verse through the full extraction workflow.

        Args:
            verse_id: Canonical verse identifier
            text: Verse text in original language
            context: Additional context

        Returns:
            Final workflow state with all extraction results
        """
        initial_state: WorkflowState = {
            "verse_id": verse_id,
            "text": text,
            "context": context or {},
            "current_phase": "",
            "results": {},
            "errors": [],
            "confidence_scores": {},
            "metadata": {"start_time": time.time()}
        }

        self.logger.info(f"Processing verse: {verse_id}")

        final_state = await self._workflow.ainvoke(initial_state)

        final_state["metadata"]["end_time"] = time.time()
        final_state["metadata"]["total_time_seconds"] = (
            final_state["metadata"]["end_time"] - final_state["metadata"]["start_time"]
        )

        self.logger.info(
            f"Completed {verse_id} in {final_state['metadata']['total_time_seconds']:.2f}s "
            f"(confidence: {final_state['metadata'].get('overall_confidence', 0):.2f})"
        )

        return final_state

    async def process_batch(
        self,
        verses: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[WorkflowState]:
        """Process a batch of verses."""
        results = []
        total = len(verses)

        for i, verse in enumerate(verses):
            state = await self.process_verse(
                verse["verse_id"],
                verse["text"],
                verse.get("context")
            )
            results.append(state)

            if progress_callback:
                progress_callback(i + 1, total)

        return results
