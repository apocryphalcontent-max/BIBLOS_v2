"""
BIBLOS v2 - Agent Orchestrator (Deprecated Shim)

DEPRECATED: This module has been superseded by agents.langgraph_orchestrator.
Please use agents.langgraph_orchestrator or import from agents directly.

The langgraph_orchestrator provides:
- Typed state management with Pydantic
- Checkpointing for recovery
- Conditional routing
- Parallel agent execution with semaphore
- Streaming support
- Comprehensive observability
"""
import warnings

warnings.warn(
    "agents.orchestrator is deprecated. Use agents.langgraph_orchestrator.LangGraphOrchestrator instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from the newer implementation for backwards compatibility
from agents.langgraph_orchestrator import (
    LangGraphOrchestrator,
    OrchestrationConfig,
    PipelineState,
    PhaseStatus,
)

# Backwards compatibility aliases
AgentOrchestrator = LangGraphOrchestrator
WorkflowState = PipelineState
WorkflowPhase = PhaseStatus

__all__ = [
    # New names
    "LangGraphOrchestrator",
    "OrchestrationConfig",
    "PipelineState",
    "PhaseStatus",
    # Backwards compatibility aliases
    "AgentOrchestrator",
    "WorkflowState",
    "WorkflowPhase",
]
