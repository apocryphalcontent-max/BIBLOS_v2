# BIBLOS v2 LangChain/LangGraph Architecture Review

## Executive Summary

This document provides a comprehensive review of the BIBLOS v2 agent architecture with recommendations for enhancing LangChain/LangGraph integration, improving production reliability, and establishing best practices for the 24-agent SDES pipeline.

**Current State**: The codebase demonstrates solid foundations with good separation of concerns, centralized schemas, and a working LangGraph integration in the orchestrator. However, several areas can be enhanced for production-grade reliability.

**Key Recommendations**:
1. Upgrade to modern LangChain patterns (structured outputs, tool binding)
2. Implement proper LangGraph StateGraph with checkpointing
3. Add comprehensive observability with LangSmith integration
4. Enhance multi-agent coordination with shared memory
5. Implement proper tool abstractions for database operations

---

## 1. Agent Architecture Review (`agents/base.py`)

### Current Strengths
- Clean ABC pattern with required `extract()`, `validate()`, `get_dependencies()` methods
- Integrated metrics collection (`AgentMetrics` class)
- Caching with SHA-256 hash keys
- Proper async support throughout
- LangChain tool wrapper (`as_langchain_tool()`)

### Issues Identified

#### Issue 1: Dataclass vs Pydantic for Structured Outputs
```python
# Current: Using dataclass
@dataclass
class ExtractionResult:
    agent_name: str
    extraction_type: ExtractionType
    ...
```

**Problem**: LangChain structured outputs work best with Pydantic models for validation, serialization, and LLM output parsing.

#### Issue 2: Missing State Management
Agents don't have proper state management for intermediate results, making recovery and debugging difficult.

#### Issue 3: Limited Tool Input Schema
```python
class AgentToolInput(BaseModel):
    verse_id: str = Field(description="Canonical verse ID (e.g., GEN.1.1)")
    text: str = Field(description="Verse text in original language")
    context: Dict[str, Any] = Field(default={}, description="Additional context")
```

**Problem**: Generic `Dict[str, Any]` for context loses type safety and documentation.

#### Issue 4: Synchronous Fallback in Tool
```python
def _run(self, ...):
    result = asyncio.run(self.agent.process(verse_id, text, context))
```

**Problem**: `asyncio.run()` creates new event loops, causing issues in async contexts.

### Recommendations

See Section 6 for the enhanced `agents/base.py` implementation.

---

## 2. LangGraph Integration Review (`agents/orchestrator.py`)

### Current Implementation

```python
class WorkflowState(TypedDict):
    verse_id: str
    text: str
    context: Dict[str, Any]
    current_phase: str
    results: Dict[str, Any]
    errors: List[str]
    confidence_scores: Dict[str, float]
    metadata: Dict[str, Any]
```

### Strengths
- Uses `TypedDict` for state (good)
- Conditional edges for validation phase
- Parallel agent execution with semaphore

### Issues Identified

#### Issue 1: No Checkpointing
The current implementation doesn't use LangGraph's checkpointing, making recovery impossible after failures.

#### Issue 2: Limited Conditional Logic
Only one conditional edge (validation decision). The pipeline could benefit from:
- Early termination on critical failures
- Dynamic phase ordering based on content type
- Retry logic for failed agents

#### Issue 3: State Mutation
```python
async def _run_linguistic_phase(self, state: WorkflowState) -> WorkflowState:
    state["current_phase"] = WorkflowPhase.LINGUISTIC.value
    state["results"]["linguistic"] = {...}
```

**Problem**: Direct state mutation can cause issues. LangGraph prefers returning new state objects.

#### Issue 4: No Message History
The workflow doesn't track decision history, making debugging difficult.

### Recommendations

See Section 7 for the enhanced LangGraph StateGraph design.

---

## 3. Multi-Agent Coordination Review

### Current Coordination Model
- Agents registered in singleton `AgentRegistry`
- Topological sort for dependency ordering
- Phases run agents in parallel with semaphore

### Issues Identified

#### Issue 1: No Shared Memory Between Agents
Agents don't share memory, leading to:
- Duplicate LLM calls for same information
- No context accumulation across phases
- Limited agent collaboration

#### Issue 2: Rigid Dependency Model
Static dependency declaration doesn't support:
- Dynamic dependencies based on content
- Optional enhancements from available agents
- Graceful degradation when agents fail

#### Issue 3: No Agent Communication
Agents can't directly communicate, only through the orchestrator.

### Recommendations

1. **Implement Shared Memory Store**
```python
class SharedAgentMemory:
    """Shared memory for inter-agent communication."""
    semantic_cache: Dict[str, Any]  # LLM responses
    entity_memory: Dict[str, Any]   # Extracted entities
    cross_ref_memory: Dict[str, Any]  # Discovered connections
```

2. **Add Dynamic Dependency Resolution**
```python
async def resolve_dependencies(
    self,
    context: Dict[str, Any]
) -> List[str]:
    """Dynamically resolve dependencies based on context."""
    base_deps = self.get_dependencies()
    # Add optional deps if available and relevant
    if "typological_content" in context:
        base_deps.append("typologos")
    return base_deps
```

3. **Implement Agent Communication Channels**
```python
class AgentChannel:
    """Communication channel between agents."""
    async def request(self, target_agent: str, query: Dict) -> Dict:
        """Request information from another agent."""
    async def broadcast(self, message: Dict) -> None:
        """Broadcast to all agents in phase."""
```

---

## 4. Tool Development Recommendations

### Recommended Tools Architecture

```
tools/
  __init__.py
  base.py              # BaseTool with common functionality
  database/
    __init__.py
    neo4j_tools.py     # Graph database operations
    postgres_tools.py  # Relational database operations
    qdrant_tools.py    # Vector similarity search
  retrieval/
    __init__.py
    crossref_tools.py  # Cross-reference lookup
    patristic_tools.py # Patristic citation retrieval
    verse_tools.py     # Verse text retrieval
  analysis/
    __init__.py
    embedding_tools.py # Embedding generation
    similarity_tools.py # Similarity computation
```

### Critical Tools Needed

1. **Neo4j Cross-Reference Tool**
2. **Qdrant Similarity Search Tool**
3. **Patristic Citation Retrieval Tool**
4. **PostgreSQL Verse Lookup Tool**

See Section 8 for full implementations.

---

## 5. Streaming and Observability

### Current State
- Basic logging with Python `logging`
- No streaming support
- No distributed tracing

### Recommendations

1. **LangSmith Integration**
```python
from langchain.callbacks.tracers import LangSmithTracer

tracer = LangSmithTracer(
    project_name="biblos-v2",
    client=langsmith_client
)
```

2. **OpenTelemetry Tracing**
```python
from opentelemetry import trace
from opentelemetry.instrumentation.langchain import LangChainInstrumentor

LangChainInstrumentor().instrument()
tracer = trace.get_tracer("biblos.agents")
```

3. **Streaming Callbacks**
```python
class AgentStreamingCallback(BaseCallbackHandler):
    """Streaming callback for real-time output."""

    async def on_llm_new_token(self, token: str, **kwargs):
        await self.stream_queue.put(token)
```

---

## 6. Enhanced Base Agent Implementation

### File: `agents/base_v2.py`

Key improvements:
- Pydantic models for structured outputs
- Proper state management with dataclass
- Enhanced tool binding
- Context window management
- Built-in observability hooks

---

## 7. LangGraph State Machine Design

### Pipeline State Schema

```python
class PipelineState(TypedDict):
    # Core Data
    verse_id: str
    text: str
    original_language: str

    # Phase Tracking
    current_phase: str
    completed_phases: List[str]
    pending_phases: List[str]

    # Results Storage
    linguistic_results: Dict[str, Any]
    theological_results: Dict[str, Any]
    intertextual_results: Dict[str, Any]
    validation_results: Dict[str, Any]

    # Quality Metrics
    phase_confidences: Dict[str, float]
    overall_confidence: float

    # Error Handling
    errors: List[Dict[str, str]]
    warnings: List[str]
    retry_count: int

    # Metadata
    start_time: float
    processing_times: Dict[str, float]
    messages: List[BaseMessage]
```

### Conditional Routing Functions

```python
def should_proceed_to_theological(state: PipelineState) -> str:
    """Determine if theological phase should run."""
    linguistic_confidence = state["phase_confidences"].get("linguistic", 0)

    if linguistic_confidence < 0.3:
        return "retry_linguistic"
    elif linguistic_confidence < 0.5:
        return "theological_with_caution"
    else:
        return "theological"

def should_run_validation(state: PipelineState) -> bool:
    """Determine if validation phase is needed."""
    avg_confidence = state["overall_confidence"]
    error_count = len(state["errors"])

    return avg_confidence < 0.7 or error_count > 0
```

---

## 8. Agent Evaluation Framework

### Evaluation Dimensions

1. **Extraction Quality**
   - Precision: Correct extractions / Total extractions
   - Recall: Correct extractions / Expected extractions
   - F1 Score: Harmonic mean of precision and recall

2. **Cross-Reference Quality**
   - Connection accuracy vs ground truth
   - Type classification accuracy
   - Strength rating accuracy

3. **Latency**
   - P50, P95, P99 latency per agent
   - Total pipeline latency

4. **Resource Efficiency**
   - Token usage per extraction
   - Cost per verse processed

### MLflow Integration

```python
from mlflow import MlflowClient

class AgentEvaluator:
    def __init__(self):
        self.mlflow = MlflowClient()

    def evaluate_agent(
        self,
        agent: BaseExtractionAgent,
        test_cases: List[Dict]
    ) -> Dict[str, float]:
        """Evaluate agent on test cases."""
        with mlflow.start_run(run_name=f"eval_{agent.config.name}"):
            results = []
            for case in test_cases:
                result = await agent.process(
                    case["verse_id"],
                    case["text"],
                    case["context"]
                )
                results.append(self._score_result(result, case["expected"]))

            metrics = self._aggregate_metrics(results)
            mlflow.log_metrics(metrics)
            return metrics
```

---

## 9. Context Window Management

### Strategies for Long Biblical Texts

1. **Hierarchical Summarization**
```python
class ContextCompressor:
    """Compress context to fit within token limits."""

    async def compress(
        self,
        context: Dict[str, Any],
        max_tokens: int = 4000
    ) -> Dict[str, Any]:
        # Prioritize most relevant context
        # Summarize verbose sections
        # Remove redundant information
```

2. **Retrieval Augmentation**
```python
class ContextRetriever:
    """Retrieve relevant context based on query."""

    async def retrieve(
        self,
        verse_id: str,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        # Vector search for relevant passages
        # Patristic citation retrieval
        # Cross-reference context
```

3. **Progressive Disclosure**
```python
class ProgressiveContext:
    """Load context progressively as needed."""

    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth
        self._loaded_levels = 0

    async def get_next_level(self) -> Dict[str, Any]:
        """Load next level of context detail."""
```

---

## 10. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
- [ ] Update `agents/base.py` with Pydantic models
- [ ] Implement LangGraph checkpointing
- [ ] Add basic observability hooks

### Phase 2: Tools (Week 3-4)
- [ ] Implement database tools (Neo4j, Qdrant, PostgreSQL)
- [ ] Add retrieval tools (patristic, cross-reference)
- [ ] Create tool binding utilities

### Phase 3: Coordination (Week 5-6)
- [ ] Implement shared memory store
- [ ] Add agent communication channels
- [ ] Enhance dependency resolution

### Phase 4: Observability (Week 7-8)
- [ ] LangSmith integration
- [ ] OpenTelemetry tracing
- [ ] Streaming callbacks

### Phase 5: Evaluation (Week 9-10)
- [ ] Build evaluation framework
- [ ] Create test case repository
- [ ] Implement MLflow tracking

---

## Appendix A: File Locations

| Component | Current | Enhanced |
|-----------|---------|----------|
| Base Agent | `agents/base.py` | `agents/base_v2.py` |
| Orchestrator | `agents/orchestrator.py` | `agents/langgraph_orchestrator.py` |
| Tools | N/A | `agents/tools/` |
| Evaluation | N/A | `evaluation/` |

---

## Appendix B: Dependency Updates

Update `requirements.txt`:
```
# Enhanced LangChain
langchain>=0.2.0
langchain-core>=0.2.0
langchain-community>=0.2.0
langgraph>=0.1.0

# Observability
langsmith>=0.1.0
opentelemetry-api>=1.22.0
opentelemetry-sdk>=1.22.0
opentelemetry-instrumentation>=0.43b0
```
