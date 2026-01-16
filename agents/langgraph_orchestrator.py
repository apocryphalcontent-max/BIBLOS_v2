"""
BIBLOS v2 - LangGraph Pipeline Orchestrator

Production-grade multi-agent orchestration using LangGraph with:
- Typed state management
- Checkpointing for recovery
- Conditional routing
- Parallel agent execution
- Streaming support
- Comprehensive observability
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    TypedDict,
    Union,
)

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from agents.base_v2 import (
    AgentConfig,
    AgentPhase,
    BaseExtractionAgent,
    ExtractionContext,
    ExtractionResult,
    ExtractionType,
)
from agents.registry import AgentRegistry, registry
from data.schemas import ProcessingStatus


# =============================================================================
# STATE DEFINITIONS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a pipeline phase."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    NEEDS_RETRY = "needs_retry"


class PipelineState(TypedDict):
    """
    State passed through the LangGraph workflow.

    This is the central state object that flows through all nodes.
    Uses TypedDict for LangGraph compatibility.
    """

    # Core Data
    verse_id: str
    text: str
    original_language: str

    # Phase Tracking
    current_phase: str
    completed_phases: List[str]
    pending_phases: List[str]
    phase_statuses: Dict[str, str]

    # Results Storage (organized by phase)
    linguistic_results: Dict[str, Any]
    theological_results: Dict[str, Any]
    intertextual_results: Dict[str, Any]
    validation_results: Dict[str, Any]
    finalization_results: Dict[str, Any]

    # Agent-level results
    agent_results: Dict[str, Dict[str, Any]]

    # Quality Metrics
    phase_confidences: Dict[str, float]
    agent_confidences: Dict[str, float]
    overall_confidence: float

    # Error Handling
    errors: List[Dict[str, Any]]
    warnings: List[str]
    retry_count: int
    max_retries: int

    # Metadata
    start_time: float
    processing_times: Dict[str, float]
    total_agents_run: int

    # Message history for tracing
    messages: Annotated[Sequence[BaseMessage], add_messages]


def create_initial_state(
    verse_id: str,
    text: str,
    original_language: str = "hebrew",
    max_retries: int = 2,
) -> PipelineState:
    """Create initial pipeline state."""
    return PipelineState(
        verse_id=verse_id,
        text=text,
        original_language=original_language,
        current_phase="",
        completed_phases=[],
        pending_phases=["linguistic", "theological", "intertextual", "validation", "finalization"],
        phase_statuses={
            "linguistic": PhaseStatus.PENDING.value,
            "theological": PhaseStatus.PENDING.value,
            "intertextual": PhaseStatus.PENDING.value,
            "validation": PhaseStatus.PENDING.value,
            "finalization": PhaseStatus.PENDING.value,
        },
        linguistic_results={},
        theological_results={},
        intertextual_results={},
        validation_results={},
        finalization_results={},
        agent_results={},
        phase_confidences={},
        agent_confidences={},
        overall_confidence=0.0,
        errors=[],
        warnings=[],
        retry_count=0,
        max_retries=max_retries,
        start_time=time.time(),
        processing_times={},
        total_agents_run=0,
        messages=[],
    )


# =============================================================================
# ORCHESTRATION CONFIG
# =============================================================================


class OrchestrationConfig(BaseModel):
    """Configuration for pipeline orchestration."""

    # Parallel execution
    parallel_phases: bool = Field(default=True)
    max_parallel_agents: int = Field(default=8, ge=1, le=32)

    # Timeouts
    phase_timeout_seconds: int = Field(default=300, ge=1)
    agent_timeout_seconds: int = Field(default=60, ge=1)

    # Retry policy
    max_retries: int = Field(default=2, ge=0, le=5)
    retry_backoff_factor: float = Field(default=1.5, ge=1.0)

    # Quality thresholds
    min_phase_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    min_overall_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    validation_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    # Features
    enable_validation_phase: bool = Field(default=True)
    enable_checkpointing: bool = Field(default=True)
    enable_streaming: bool = Field(default=False)

    # Checkpoint settings
    checkpoint_dir: str = Field(default="./checkpoints")


# =============================================================================
# PHASE CONFIGURATION
# =============================================================================


PHASE_AGENTS: Dict[str, List[ExtractionType]] = {
    "linguistic": [
        ExtractionType.PHONOLOGICAL,
        ExtractionType.MORPHOLOGICAL,
        ExtractionType.SYNTACTIC,
        ExtractionType.SEMANTIC,
        ExtractionType.LEXICAL,
        ExtractionType.ETYMOLOGICAL,
        ExtractionType.DISCOURSE,
        ExtractionType.PRAGMATIC,
    ],
    "theological": [
        ExtractionType.PATRISTIC,
        ExtractionType.TYPOLOGICAL,
        ExtractionType.LITURGICAL,
    ],
    "intertextual": [
        ExtractionType.INTERTEXTUAL,
        ExtractionType.STRUCTURAL,
    ],
    "validation": [
        ExtractionType.VALIDATION,
    ],
}


# =============================================================================
# ROUTING FUNCTIONS
# =============================================================================


def should_proceed_to_theological(state: PipelineState) -> Literal["theological", "retry_linguistic", "error"]:
    """Determine routing after linguistic phase."""
    linguistic_confidence = state["phase_confidences"].get("linguistic", 0.0)
    errors = [e for e in state["errors"] if e.get("phase") == "linguistic"]

    if errors and len(errors) >= 3:
        return "error"
    elif linguistic_confidence < 0.3 and state["retry_count"] < state["max_retries"]:
        return "retry_linguistic"
    else:
        return "theological"


def should_proceed_to_intertextual(state: PipelineState) -> Literal["intertextual", "retry_theological", "error"]:
    """Determine routing after theological phase."""
    theological_confidence = state["phase_confidences"].get("theological", 0.0)
    errors = [e for e in state["errors"] if e.get("phase") == "theological"]

    if errors and len(errors) >= 3:
        return "error"
    elif theological_confidence < 0.3 and state["retry_count"] < state["max_retries"]:
        return "retry_theological"
    else:
        return "intertextual"


def should_run_validation(state: PipelineState) -> Literal["validation", "finalization"]:
    """Determine if validation phase should run."""
    avg_confidence = state["overall_confidence"]
    error_count = len(state["errors"])
    warnings_count = len(state["warnings"])

    # Run validation if:
    # - Average confidence is below threshold
    # - There are errors
    # - There are many warnings
    if avg_confidence < 0.7 or error_count > 0 or warnings_count > 5:
        return "validation"
    else:
        return "finalization"


def determine_next_phase(state: PipelineState) -> str:
    """Determine the next phase based on current state."""
    current = state["current_phase"]

    phase_order = ["linguistic", "theological", "intertextual", "validation", "finalization"]

    if not current:
        return "linguistic"

    try:
        current_idx = phase_order.index(current)
        if current_idx < len(phase_order) - 1:
            return phase_order[current_idx + 1]
    except ValueError:
        pass

    return END


# =============================================================================
# PIPELINE ORCHESTRATOR
# =============================================================================


class LangGraphOrchestrator:
    """
    Orchestrates multi-agent extraction using LangGraph.

    Features:
    - Typed state management with checkpointing
    - Conditional routing based on confidence
    - Parallel agent execution within phases
    - Automatic retry with backoff
    - Comprehensive error handling
    - Streaming support for real-time output
    """

    def __init__(
        self,
        config: Optional[OrchestrationConfig] = None,
        agent_registry: Optional[AgentRegistry] = None,
    ):
        self.config = config or OrchestrationConfig()
        self.registry = agent_registry or registry
        self.logger = logging.getLogger("biblos.orchestrator.langgraph")

        # Initialize checkpointer if enabled
        self._checkpointer = None
        if self.config.enable_checkpointing:
            self._checkpointer = MemorySaver()

        # Build the workflow
        self._workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with all nodes and edges."""
        workflow = StateGraph(PipelineState)

        # Add phase nodes
        workflow.add_node("linguistic", self._run_linguistic_phase)
        workflow.add_node("theological", self._run_theological_phase)
        workflow.add_node("intertextual", self._run_intertextual_phase)
        workflow.add_node("validation", self._run_validation_phase)
        workflow.add_node("finalization", self._run_finalization)

        # Add utility nodes
        workflow.add_node("error_handler", self._handle_error)
        workflow.add_node("retry_handler", self._handle_retry)

        # Define edges
        workflow.set_entry_point("linguistic")

        # Conditional routing after linguistic
        workflow.add_conditional_edges(
            "linguistic",
            should_proceed_to_theological,
            {
                "theological": "theological",
                "retry_linguistic": "retry_handler",
                "error": "error_handler",
            },
        )

        # Conditional routing after theological
        workflow.add_conditional_edges(
            "theological",
            should_proceed_to_intertextual,
            {
                "intertextual": "intertextual",
                "retry_theological": "retry_handler",
                "error": "error_handler",
            },
        )

        # Conditional routing after intertextual
        workflow.add_conditional_edges(
            "intertextual",
            should_run_validation,
            {
                "validation": "validation",
                "finalization": "finalization",
            },
        )

        # Validation always goes to finalization
        workflow.add_edge("validation", "finalization")

        # Finalization ends the workflow
        workflow.add_edge("finalization", END)

        # Error handler ends the workflow
        workflow.add_edge("error_handler", END)

        # Retry handler goes back to appropriate phase
        workflow.add_conditional_edges(
            "retry_handler",
            lambda state: state["current_phase"],
            {
                "linguistic": "linguistic",
                "theological": "theological",
                "intertextual": "intertextual",
            },
        )

        # Compile with checkpointer
        return workflow.compile(checkpointer=self._checkpointer)

    # -------------------------------------------------------------------------
    # Phase Execution
    # -------------------------------------------------------------------------

    async def _run_phase_agents(
        self,
        state: PipelineState,
        phase: str,
    ) -> Dict[str, Any]:
        """Run all agents for a phase with optional parallelization."""
        results: Dict[str, Any] = {}
        extraction_types = PHASE_AGENTS.get(phase, [])

        # Gather agents for this phase
        phase_agents: List[BaseExtractionAgent] = []
        for ext_type in extraction_types:
            agents = self.registry.get_by_type(ext_type)
            phase_agents.extend(agents)

        if not phase_agents:
            self.logger.warning(f"No agents registered for phase {phase}")
            return results

        self.logger.info(f"Running {len(phase_agents)} agents for {phase} phase")

        # Build context from state
        context = self._build_context_from_state(state)

        if self.config.parallel_phases:
            # Parallel execution with semaphore
            semaphore = asyncio.Semaphore(self.config.max_parallel_agents)

            async def run_agent(agent: BaseExtractionAgent) -> tuple[str, ExtractionResult]:
                async with semaphore:
                    try:
                        result = await asyncio.wait_for(
                            agent.process(
                                state["verse_id"],
                                state["text"],
                                context,
                            ),
                            timeout=self.config.agent_timeout_seconds,
                        )
                        return agent.config.name, result
                    except asyncio.TimeoutError:
                        return agent.config.name, ExtractionResult.failure(
                            agent_name=agent.config.name,
                            extraction_type=agent.config.extraction_type,
                            verse_id=state["verse_id"],
                            error=f"Agent timed out after {self.config.agent_timeout_seconds}s",
                        )
                    except Exception as e:
                        return agent.config.name, ExtractionResult.failure(
                            agent_name=agent.config.name,
                            extraction_type=agent.config.extraction_type,
                            verse_id=state["verse_id"],
                            error=str(e),
                        )

            tasks = [run_agent(agent) for agent in phase_agents]
            agent_results = await asyncio.gather(*tasks)

            for agent_name, result in agent_results:
                results[agent_name] = result.model_dump() if hasattr(result, "model_dump") else result.to_dict()

        else:
            # Sequential execution
            for agent in phase_agents:
                try:
                    result = await asyncio.wait_for(
                        agent.process(
                            state["verse_id"],
                            state["text"],
                            context,
                        ),
                        timeout=self.config.agent_timeout_seconds,
                    )
                    results[agent.config.name] = result.model_dump() if hasattr(result, "model_dump") else result.to_dict()
                except Exception as e:
                    self.logger.error(f"Agent {agent.config.name} failed: {e}")
                    results[agent.config.name] = {
                        "status": ProcessingStatus.FAILED.value,
                        "error": str(e),
                    }

        return results

    def _build_context_from_state(self, state: PipelineState) -> ExtractionContext:
        """Build extraction context from pipeline state."""
        # Parse verse_id for book info
        parts = state["verse_id"].split(".")
        book = parts[0] if parts else ""
        chapter = int(parts[1]) if len(parts) > 1 else 0

        # Determine testament
        ot_books = {"GEN", "EXO", "LEV", "NUM", "DEU", "JOS", "JDG", "RUT", "1SA", "2SA",
                    "1KI", "2KI", "1CH", "2CH", "EZR", "NEH", "EST", "JOB", "PSA", "PRO",
                    "ECC", "SNG", "ISA", "JER", "LAM", "EZK", "DAN", "HOS", "JOL", "AMO",
                    "OBA", "JON", "MIC", "NAH", "HAB", "ZEP", "HAG", "ZEC", "MAL"}
        testament = "OT" if book in ot_books else "NT"

        return ExtractionContext(
            book=book,
            chapter=chapter,
            testament=testament,
            language=state["original_language"],
            linguistic_results=state["linguistic_results"],
            theological_results=state["theological_results"],
            intertextual_results=state["intertextual_results"],
            completed_phases=state["completed_phases"],
            completed_agents=list(state["agent_results"].keys()),
        )

    def _compute_phase_confidence(self, results: Dict[str, Any]) -> float:
        """Compute average confidence for a phase."""
        if not results:
            return 0.0

        confidences = []
        for agent_result in results.values():
            if isinstance(agent_result, dict):
                conf = agent_result.get("confidence", 0.0)
                if conf > 0:
                    confidences.append(conf)

        return sum(confidences) / len(confidences) if confidences else 0.0

    def _compute_overall_confidence(self, state: PipelineState) -> float:
        """Compute overall pipeline confidence."""
        confidences = list(state["phase_confidences"].values())
        return sum(confidences) / len(confidences) if confidences else 0.0

    # -------------------------------------------------------------------------
    # Phase Nodes
    # -------------------------------------------------------------------------

    async def _run_linguistic_phase(self, state: PipelineState) -> PipelineState:
        """Execute linguistic analysis phase."""
        phase = "linguistic"
        start_time = time.time()

        # Update state
        state["current_phase"] = phase
        state["phase_statuses"][phase] = PhaseStatus.RUNNING.value
        state["messages"].append(
            SystemMessage(content=f"Starting {phase} phase for {state['verse_id']}")
        )

        # Run agents
        results = await self._run_phase_agents(state, phase)

        # Update state with results
        state["linguistic_results"] = results
        state["agent_results"].update(results)

        # Compute confidence
        confidence = self._compute_phase_confidence(results)
        state["phase_confidences"][phase] = confidence
        state["overall_confidence"] = self._compute_overall_confidence(state)

        # Mark phase complete
        state["phase_statuses"][phase] = PhaseStatus.COMPLETED.value
        state["completed_phases"].append(phase)
        state["pending_phases"].remove(phase)
        state["processing_times"][phase] = time.time() - start_time
        state["total_agents_run"] += len(results)

        state["messages"].append(
            AIMessage(content=f"Completed {phase} phase with confidence {confidence:.2f}")
        )

        return state

    async def _run_theological_phase(self, state: PipelineState) -> PipelineState:
        """Execute theological analysis phase."""
        phase = "theological"
        start_time = time.time()

        state["current_phase"] = phase
        state["phase_statuses"][phase] = PhaseStatus.RUNNING.value
        state["messages"].append(
            SystemMessage(content=f"Starting {phase} phase for {state['verse_id']}")
        )

        results = await self._run_phase_agents(state, phase)

        state["theological_results"] = results
        state["agent_results"].update(results)

        confidence = self._compute_phase_confidence(results)
        state["phase_confidences"][phase] = confidence
        state["overall_confidence"] = self._compute_overall_confidence(state)

        state["phase_statuses"][phase] = PhaseStatus.COMPLETED.value
        state["completed_phases"].append(phase)
        state["pending_phases"].remove(phase)
        state["processing_times"][phase] = time.time() - start_time
        state["total_agents_run"] += len(results)

        state["messages"].append(
            AIMessage(content=f"Completed {phase} phase with confidence {confidence:.2f}")
        )

        return state

    async def _run_intertextual_phase(self, state: PipelineState) -> PipelineState:
        """Execute intertextual analysis phase."""
        phase = "intertextual"
        start_time = time.time()

        state["current_phase"] = phase
        state["phase_statuses"][phase] = PhaseStatus.RUNNING.value
        state["messages"].append(
            SystemMessage(content=f"Starting {phase} phase for {state['verse_id']}")
        )

        results = await self._run_phase_agents(state, phase)

        state["intertextual_results"] = results
        state["agent_results"].update(results)

        confidence = self._compute_phase_confidence(results)
        state["phase_confidences"][phase] = confidence
        state["overall_confidence"] = self._compute_overall_confidence(state)

        state["phase_statuses"][phase] = PhaseStatus.COMPLETED.value
        state["completed_phases"].append(phase)
        state["pending_phases"].remove(phase)
        state["processing_times"][phase] = time.time() - start_time
        state["total_agents_run"] += len(results)

        state["messages"].append(
            AIMessage(content=f"Completed {phase} phase with confidence {confidence:.2f}")
        )

        return state

    async def _run_validation_phase(self, state: PipelineState) -> PipelineState:
        """Execute validation phase."""
        phase = "validation"
        start_time = time.time()

        state["current_phase"] = phase
        state["phase_statuses"][phase] = PhaseStatus.RUNNING.value
        state["messages"].append(
            SystemMessage(content=f"Starting {phase} phase for {state['verse_id']}")
        )

        results = await self._run_phase_agents(state, phase)

        state["validation_results"] = results
        state["agent_results"].update(results)

        confidence = self._compute_phase_confidence(results)
        state["phase_confidences"][phase] = confidence
        state["overall_confidence"] = self._compute_overall_confidence(state)

        state["phase_statuses"][phase] = PhaseStatus.COMPLETED.value
        state["completed_phases"].append(phase)
        state["pending_phases"].remove(phase)
        state["processing_times"][phase] = time.time() - start_time
        state["total_agents_run"] += len(results)

        state["messages"].append(
            AIMessage(content=f"Completed {phase} phase with confidence {confidence:.2f}")
        )

        return state

    async def _run_finalization(self, state: PipelineState) -> PipelineState:
        """Finalize results and compute overall metrics."""
        phase = "finalization"
        start_time = time.time()

        state["current_phase"] = phase
        state["phase_statuses"][phase] = PhaseStatus.RUNNING.value

        # Compute final statistics
        success_count = 0
        failure_count = 0

        for agent_result in state["agent_results"].values():
            if isinstance(agent_result, dict):
                status = agent_result.get("status", "")
                if status == ProcessingStatus.COMPLETED.value:
                    success_count += 1
                elif status == ProcessingStatus.FAILED.value:
                    failure_count += 1

        # Determine certification level
        avg_confidence = state["overall_confidence"]
        if avg_confidence >= 0.9:
            certification = "gold"
        elif avg_confidence >= 0.75:
            certification = "silver"
        elif avg_confidence >= 0.5:
            certification = "bronze"
        else:
            certification = "provisional"

        # Build finalization results
        state["finalization_results"] = {
            "verse_id": state["verse_id"],
            "certification_level": certification,
            "overall_confidence": avg_confidence,
            "success_count": success_count,
            "failure_count": failure_count,
            "success_rate": success_count / max(1, success_count + failure_count),
            "total_processing_time": time.time() - state["start_time"],
            "phases_completed": len(state["completed_phases"]),
            "agents_run": state["total_agents_run"],
            "error_count": len(state["errors"]),
            "warning_count": len(state["warnings"]),
        }

        state["phase_statuses"][phase] = PhaseStatus.COMPLETED.value
        state["completed_phases"].append(phase)
        state["pending_phases"].remove(phase)
        state["processing_times"][phase] = time.time() - start_time

        state["messages"].append(
            AIMessage(
                content=f"Pipeline completed with {certification} certification "
                f"(confidence: {avg_confidence:.2f})"
            )
        )

        return state

    async def _handle_error(self, state: PipelineState) -> PipelineState:
        """Handle pipeline errors."""
        state["messages"].append(
            SystemMessage(content=f"Pipeline error encountered in {state['current_phase']} phase")
        )

        # Mark current phase as failed
        current_phase = state["current_phase"]
        if current_phase:
            state["phase_statuses"][current_phase] = PhaseStatus.FAILED.value

        # Add error summary
        state["finalization_results"] = {
            "verse_id": state["verse_id"],
            "certification_level": "failed",
            "overall_confidence": 0.0,
            "error_count": len(state["errors"]),
            "last_successful_phase": state["completed_phases"][-1] if state["completed_phases"] else None,
        }

        return state

    async def _handle_retry(self, state: PipelineState) -> PipelineState:
        """Handle retry logic."""
        state["retry_count"] += 1
        current_phase = state["current_phase"]

        state["messages"].append(
            SystemMessage(
                content=f"Retrying {current_phase} phase (attempt {state['retry_count']})"
            )
        )

        # Reset phase status
        if current_phase:
            state["phase_statuses"][current_phase] = PhaseStatus.PENDING.value

            # Clear previous results for this phase
            if current_phase == "linguistic":
                state["linguistic_results"] = {}
            elif current_phase == "theological":
                state["theological_results"] = {}
            elif current_phase == "intertextual":
                state["intertextual_results"] = {}

        # Add exponential backoff delay
        backoff = self.config.retry_backoff_factor ** (state["retry_count"] - 1)
        await asyncio.sleep(min(backoff, 30))  # Cap at 30 seconds

        return state

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def process_verse(
        self,
        verse_id: str,
        text: str,
        original_language: str = "hebrew",
        config: Optional[RunnableConfig] = None,
    ) -> PipelineState:
        """
        Process a single verse through the full pipeline.

        Args:
            verse_id: Canonical verse identifier
            text: Verse text in original language
            original_language: Language of the text
            config: Optional LangChain runnable config

        Returns:
            Final pipeline state with all extraction results
        """
        initial_state = create_initial_state(
            verse_id=verse_id,
            text=text,
            original_language=original_language,
            max_retries=self.config.max_retries,
        )

        self.logger.info(f"Processing verse: {verse_id}")

        # Run the workflow
        runnable_config = config or {}
        if self.config.enable_checkpointing:
            runnable_config["configurable"] = {"thread_id": verse_id}

        final_state = await self._workflow.ainvoke(
            initial_state,
            config=runnable_config,
        )

        self.logger.info(
            f"Completed {verse_id} with {final_state.get('finalization_results', {}).get('certification_level', 'unknown')} "
            f"certification (confidence: {final_state.get('overall_confidence', 0):.2f})"
        )

        return final_state

    async def process_batch(
        self,
        verses: List[Dict[str, Any]],
        parallel: int = 4,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[PipelineState]:
        """
        Process a batch of verses.

        Args:
            verses: List of {verse_id, text, language?} dicts
            parallel: Number of parallel verse processing
            progress_callback: Optional callback(current, total)

        Returns:
            List of final pipeline states
        """
        results: List[PipelineState] = []
        total = len(verses)

        # Process in chunks
        for i in range(0, total, parallel):
            chunk = verses[i : i + parallel]

            tasks = [
                self.process_verse(
                    v["verse_id"],
                    v["text"],
                    v.get("language", "hebrew"),
                )
                for v in chunk
            ]

            chunk_results = await asyncio.gather(*tasks, return_exceptions=True)

            for j, result in enumerate(chunk_results):
                if isinstance(result, Exception):
                    self.logger.error(f"Verse processing failed: {result}")
                    # Create error state
                    error_state = create_initial_state(
                        chunk[j]["verse_id"],
                        chunk[j]["text"],
                    )
                    error_state["errors"].append({
                        "phase": "pipeline",
                        "error": str(result),
                    })
                    results.append(error_state)
                else:
                    results.append(result)

            if progress_callback:
                progress_callback(min(i + parallel, total), total)

        return results

    async def stream_verse(
        self,
        verse_id: str,
        text: str,
        original_language: str = "hebrew",
    ):
        """
        Stream verse processing with real-time updates.

        Yields state updates as each phase completes.
        """
        initial_state = create_initial_state(
            verse_id=verse_id,
            text=text,
            original_language=original_language,
            max_retries=self.config.max_retries,
        )

        config = {"configurable": {"thread_id": f"stream_{verse_id}"}}

        async for event in self._workflow.astream(initial_state, config=config):
            yield event

    def get_checkpoint(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Get checkpoint for a thread."""
        if not self._checkpointer:
            return None

        try:
            checkpoint = self._checkpointer.get(
                config={"configurable": {"thread_id": thread_id}}
            )
            return checkpoint
        except Exception:
            return None

    def list_checkpoints(self) -> List[str]:
        """List all available checkpoints."""
        if not self._checkpointer:
            return []

        try:
            # MemorySaver stores checkpoints in memory
            return list(self._checkpointer.storage.keys())
        except Exception:
            return []


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_orchestrator(
    config: Optional[OrchestrationConfig] = None,
    agent_registry: Optional[AgentRegistry] = None,
) -> LangGraphOrchestrator:
    """Factory function to create configured orchestrator."""
    return LangGraphOrchestrator(config=config, agent_registry=agent_registry)
