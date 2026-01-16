"""
BIBLOS v2 - Enhanced Base Extraction Agent

DEPRECATED: This module has been merged into agents.base
Please use agents.base.BaseExtractionAgent instead.

This file is kept for backward compatibility but will be removed in a future version.
All new features are in agents.base which combines the best patterns from both
base.py and base_v2.py.
"""
import warnings

warnings.warn(
    "agents.base_v2 is deprecated. Use agents.base.BaseExtractionAgent instead.",
    DeprecationWarning,
    stacklevel=2,
)
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)
import asyncio
import hashlib
import logging
import time
import traceback
from functools import wraps

from pydantic import BaseModel, Field, ConfigDict, field_validator
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.callbacks import (
    CallbackManagerForToolRun,
    AsyncCallbackManagerForToolRun,
    BaseCallbackHandler,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.runnables import RunnableConfig

# Import centralized schemas
from data.schemas import (
    ProcessingStatus,
    ExtractionResultSchema,
    validate_verse_id,
    normalize_verse_id,
)


# =============================================================================
# ENUMS
# =============================================================================


class ExtractionType(str, Enum):
    """Categories of extraction operations."""

    PHONOLOGICAL = "phonological"
    MORPHOLOGICAL = "morphological"
    SYNTACTIC = "syntactic"
    SEMANTIC = "semantic"
    DISCOURSE = "discourse"
    PRAGMATIC = "pragmatic"
    LEXICAL = "lexical"
    ETYMOLOGICAL = "etymological"
    PATRISTIC = "patristic"
    TYPOLOGICAL = "typological"
    LITURGICAL = "liturgical"
    INTERTEXTUAL = "intertextual"
    STRUCTURAL = "structural"
    VALIDATION = "validation"


class AgentPhase(str, Enum):
    """Pipeline phases."""

    LINGUISTIC = "linguistic"
    THEOLOGICAL = "theological"
    INTERTEXTUAL = "intertextual"
    VALIDATION = "validation"


# =============================================================================
# PYDANTIC MODELS - Structured Outputs
# =============================================================================


class AgentConfig(BaseModel):
    """Configuration for an extraction agent."""

    model_config = ConfigDict(frozen=False, extra="allow")

    name: str = Field(..., description="Unique agent identifier")
    extraction_type: ExtractionType = Field(..., description="Type of extraction")
    phase: AgentPhase = Field(default=AgentPhase.LINGUISTIC, description="Pipeline phase")

    # Processing settings
    batch_size: int = Field(default=1000, ge=1, le=10000)
    max_retries: int = Field(default=3, ge=0, le=10)
    timeout_seconds: int = Field(default=300, ge=1, le=3600)
    checkpoint_interval: int = Field(default=500, ge=1)
    parallel_workers: int = Field(default=4, ge=1, le=16)

    # INFALLIBILITY: The seraph accepts ONLY absolute certainty (1.0)
    # Uncertainty cannot propagate - the seraph inherits from itself
    min_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    min_coverage: float = Field(default=1.0, ge=0.0, le=1.0)

    # Feature flags
    enable_caching: bool = Field(default=True)
    enable_validation: bool = Field(default=True)
    enable_metrics: bool = Field(default=True)
    enable_streaming: bool = Field(default=False)

    # LLM settings
    llm_model: str = Field(default="gpt-4-turbo-preview")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)

    # Custom parameters
    custom_params: Dict[str, Any] = Field(default_factory=dict)


class ExtractionContext(BaseModel):
    """Typed context for extraction operations."""

    model_config = ConfigDict(extra="allow")

    # Book/chapter context
    book: str = Field(default="", description="Book code (e.g., GEN, MAT)")
    book_name: str = Field(default="", description="Full book name")
    chapter: int = Field(default=0, ge=0)
    testament: str = Field(default="OT", pattern="^(OT|NT)$")
    language: str = Field(default="hebrew")

    # Prior agent results
    linguistic_results: Dict[str, Any] = Field(default_factory=dict)
    theological_results: Dict[str, Any] = Field(default_factory=dict)
    intertextual_results: Dict[str, Any] = Field(default_factory=dict)

    # Surrounding context
    preceding_verses: List[str] = Field(default_factory=list)
    following_verses: List[str] = Field(default_factory=list)
    pericope_id: Optional[str] = Field(default=None)

    # External data
    patristic_citations: List[Dict[str, Any]] = Field(default_factory=list)
    cross_references: List[Dict[str, Any]] = Field(default_factory=list)
    lectionary_usage: List[str] = Field(default_factory=list)

    # Processing metadata
    completed_phases: List[str] = Field(default_factory=list)
    completed_agents: List[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    """
    Result from an extraction operation.

    Pydantic model for proper validation and serialization.
    Aligned with ExtractionResultSchema for system-wide uniformity.
    """

    model_config = ConfigDict(extra="forbid")

    # Required fields
    agent_name: str = Field(..., description="Name of the agent that produced this result")
    extraction_type: ExtractionType = Field(..., description="Type of extraction performed")
    verse_id: str = Field(..., description="Canonical verse ID (e.g., GEN.1.1)")
    status: ProcessingStatus = Field(..., description="Processing status")
    data: Dict[str, Any] = Field(default_factory=dict, description="Extracted data")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")

    # Timing
    processing_time_ms: float = Field(default=0.0, ge=0.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Issues
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("verse_id", mode="before")
    @classmethod
    def normalize_verse_id(cls, v: str) -> str:
        """Normalize verse ID on creation."""
        if v and isinstance(v, str):
            return normalize_verse_id(v)
        return v

    def to_schema(self) -> ExtractionResultSchema:
        """Convert to centralized ExtractionResultSchema."""
        return ExtractionResultSchema(
            agent_name=self.agent_name,
            extraction_type=self.extraction_type.value,
            verse_id=self.verse_id,
            status=self.status.value,
            confidence=self.confidence,
            data=self.data,
            error="; ".join(self.errors) if self.errors else None,
            processing_time=self.processing_time_ms / 1000.0,
        )

    @classmethod
    def from_schema(
        cls, schema: ExtractionResultSchema, extraction_type: ExtractionType
    ) -> "ExtractionResult":
        """Create from ExtractionResultSchema."""
        return cls(
            agent_name=schema.agent_name,
            extraction_type=extraction_type,
            verse_id=schema.verse_id,
            status=ProcessingStatus(schema.status),
            data=schema.data,
            confidence=schema.confidence,
            processing_time_ms=schema.processing_time * 1000.0,
            errors=[schema.error] if schema.error else [],
        )

    @classmethod
    def failure(
        cls,
        agent_name: str,
        extraction_type: ExtractionType,
        verse_id: str,
        error: str,
        processing_time_ms: float = 0.0,
    ) -> "ExtractionResult":
        """Create a failure result."""
        return cls(
            agent_name=agent_name,
            extraction_type=extraction_type,
            verse_id=verse_id,
            status=ProcessingStatus.FAILED,
            data={},
            confidence=0.0,
            processing_time_ms=processing_time_ms,
            errors=[error],
        )


# =============================================================================
# AGENT STATE
# =============================================================================


class AgentState(BaseModel):
    """State container for agent execution."""

    model_config = ConfigDict(extra="allow")

    # Processing state
    is_initialized: bool = False
    is_processing: bool = False
    current_verse_id: Optional[str] = None

    # Statistics
    total_processed: int = 0
    successful: int = 0
    failed: int = 0
    total_time_ms: float = 0.0
    avg_confidence: float = 0.0

    # Cache
    cache_hits: int = 0
    cache_misses: int = 0

    # Error tracking
    last_error: Optional[str] = None
    consecutive_errors: int = 0

    def record_success(self, result: ExtractionResult) -> None:
        """Record a successful extraction."""
        self.total_processed += 1
        self.successful += 1
        self.total_time_ms += result.processing_time_ms
        # Rolling average
        self.avg_confidence = (
            (self.avg_confidence * (self.successful - 1) + result.confidence)
            / self.successful
        )
        self.consecutive_errors = 0

    def record_failure(self, error: str) -> None:
        """Record a failed extraction."""
        self.total_processed += 1
        self.failed += 1
        self.last_error = error
        self.consecutive_errors += 1


# =============================================================================
# METRICS
# =============================================================================


class AgentMetrics(BaseModel):
    """Metrics collector for agent performance monitoring."""

    model_config = ConfigDict(extra="forbid")

    agent_name: str
    total_processed: int = 0
    successful: int = 0
    failed: int = 0
    total_time_ms: float = 0.0
    avg_confidence: float = 0.0
    _confidence_sum: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.successful / max(1, self.total_processed)

    @property
    def avg_time_ms(self) -> float:
        """Calculate average processing time."""
        return self.total_time_ms / max(1, self.total_processed)

    def record(self, result: ExtractionResult) -> None:
        """Record metrics from extraction result."""
        self.total_processed += 1
        self.total_time_ms += result.processing_time_ms

        if result.status == ProcessingStatus.COMPLETED:
            self.successful += 1
            self._confidence_sum += result.confidence
            self.avg_confidence = self._confidence_sum / self.successful
        else:
            self.failed += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_name": self.agent_name,
            "total_processed": self.total_processed,
            "successful": self.successful,
            "failed": self.failed,
            "success_rate": self.success_rate,
            "total_time_ms": self.total_time_ms,
            "avg_time_ms": self.avg_time_ms,
            "avg_confidence": self.avg_confidence,
        }


# =============================================================================
# CALLBACK HANDLERS
# =============================================================================


class AgentCallbackHandler(BaseCallbackHandler):
    """Callback handler for agent observability."""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.logger = logging.getLogger(f"biblos.agents.{agent_name}.callbacks")
        self.events: List[Dict[str, Any]] = []

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> None:
        """Log LLM start."""
        self.events.append({
            "event": "llm_start",
            "agent": self.agent_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt_count": len(prompts),
        })
        self.logger.debug(f"LLM started with {len(prompts)} prompts")

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Log LLM end."""
        self.events.append({
            "event": "llm_end",
            "agent": self.agent_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self.logger.debug("LLM completed")

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """Log LLM error."""
        self.events.append({
            "event": "llm_error",
            "agent": self.agent_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(error),
        })
        self.logger.error(f"LLM error: {error}")

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Log tool start."""
        self.events.append({
            "event": "tool_start",
            "agent": self.agent_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": serialized.get("name", "unknown"),
        })

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Log tool end."""
        self.events.append({
            "event": "tool_end",
            "agent": self.agent_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })


# =============================================================================
# BASE EXTRACTION AGENT
# =============================================================================

T = TypeVar("T", bound="BaseExtractionAgent")


class BaseExtractionAgent(ABC):
    """
    Abstract base class for SDES extraction agents.

    Production-grade implementation with:
    - Pydantic models for structured outputs
    - Proper state management
    - Enhanced caching with TTL
    - Built-in observability
    - Async-first design

    All extraction agents must inherit from this class and implement:
    - extract(): Core extraction logic
    - validate(): Result validation
    - get_dependencies(): Agent dependencies
    """

    # Class-level configuration
    DEFAULT_CONFIG: ClassVar[Dict[str, Any]] = {
        "batch_size": 1000,
        "max_retries": 3,
        "timeout_seconds": 300,
        "min_confidence": 0.7,
    }

    def __init__(self, config: AgentConfig):
        """Initialize agent with configuration."""
        self.config = config
        self.logger = logging.getLogger(f"biblos.agents.{config.name}")
        self.metrics = AgentMetrics(agent_name=config.name)
        self.state = AgentState()
        self._cache: Dict[str, ExtractionResult] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._callbacks: List[BaseCallbackHandler] = [
            AgentCallbackHandler(config.name)
        ]

    # -------------------------------------------------------------------------
    # Lifecycle Methods
    # -------------------------------------------------------------------------

    async def initialize(self) -> None:
        """Initialize agent resources."""
        if self.state.is_initialized:
            return

        self.logger.info(f"Initializing agent: {self.config.name}")
        try:
            await self._setup_resources()
            self.state.is_initialized = True
            self.logger.info(f"Agent {self.config.name} initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize agent: {e}")
            raise

    async def _setup_resources(self) -> None:
        """Override to set up agent-specific resources."""
        pass

    async def shutdown(self) -> None:
        """Clean up agent resources."""
        self.logger.info(f"Shutting down agent: {self.config.name}")
        try:
            await self._cleanup_resources()
            self.state.is_initialized = False
            self._cache.clear()
            self._cache_timestamps.clear()
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            raise

    async def _cleanup_resources(self) -> None:
        """Override to clean up agent-specific resources."""
        pass

    # -------------------------------------------------------------------------
    # Abstract Methods
    # -------------------------------------------------------------------------

    @abstractmethod
    async def extract(
        self,
        verse_id: str,
        text: str,
        context: ExtractionContext,
    ) -> ExtractionResult:
        """
        Perform extraction on a verse.

        Args:
            verse_id: Canonical verse identifier (e.g., "GEN.1.1")
            text: Verse text in original language
            context: Typed extraction context

        Returns:
            ExtractionResult with extracted data
        """
        pass

    @abstractmethod
    async def validate(self, result: ExtractionResult) -> bool:
        """
        Validate extraction result.

        Args:
            result: Extraction result to validate

        Returns:
            True if valid, False otherwise
        """
        pass

    @abstractmethod
    def get_dependencies(self) -> List[str]:
        """
        Get list of agent dependencies.

        Returns:
            List of agent names this agent depends on
        """
        pass

    # -------------------------------------------------------------------------
    # Processing Methods
    # -------------------------------------------------------------------------

    async def process(
        self,
        verse_id: str,
        text: str,
        context: Union[ExtractionContext, Dict[str, Any]],
        config: Optional[RunnableConfig] = None,
    ) -> ExtractionResult:
        """
        Process a verse with caching, validation, and metrics.

        Args:
            verse_id: Canonical verse identifier
            text: Verse text
            context: Extraction context (dict or typed)
            config: Optional LangChain runnable config

        Returns:
            ExtractionResult
        """
        # Convert dict context to typed
        if isinstance(context, dict):
            context = ExtractionContext(**context)

        # Check cache
        cache_key = self._make_cache_key(verse_id, text)
        if self.config.enable_caching:
            cached = self._get_cached(cache_key)
            if cached is not None:
                self.state.cache_hits += 1
                self.logger.debug(f"Cache hit for {verse_id}")
                return cached
            self.state.cache_misses += 1

        # Perform extraction with timing
        start_time = time.perf_counter()
        self.state.is_processing = True
        self.state.current_verse_id = verse_id

        try:
            result = await self._execute_with_retry(verse_id, text, context)
            result.processing_time_ms = (time.perf_counter() - start_time) * 1000

            # Validate if enabled
            if self.config.enable_validation:
                is_valid = await self.validate(result)
                if not is_valid:
                    result.status = ProcessingStatus.NEEDS_REVIEW
                    result.warnings.append("Validation failed")

            # Record success
            self.state.record_success(result)

        except Exception as e:
            self.logger.error(f"Extraction failed for {verse_id}: {e}")
            self.logger.debug(traceback.format_exc())
            result = ExtractionResult.failure(
                agent_name=self.config.name,
                extraction_type=self.config.extraction_type,
                verse_id=verse_id,
                error=str(e),
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
            )
            self.state.record_failure(str(e))

        finally:
            self.state.is_processing = False
            self.state.current_verse_id = None

        # Record metrics
        if self.config.enable_metrics:
            self.metrics.record(result)

        # Cache successful results
        if (
            self.config.enable_caching
            and result.status == ProcessingStatus.COMPLETED
        ):
            self._set_cached(cache_key, result)

        return result

    async def _execute_with_retry(
        self,
        verse_id: str,
        text: str,
        context: ExtractionContext,
    ) -> ExtractionResult:
        """Execute extraction with retry logic."""
        last_error: Optional[Exception] = None

        for attempt in range(self.config.max_retries + 1):
            try:
                return await asyncio.wait_for(
                    self.extract(verse_id, text, context),
                    timeout=self.config.timeout_seconds,
                )
            except asyncio.TimeoutError:
                last_error = TimeoutError(
                    f"Extraction timed out after {self.config.timeout_seconds}s"
                )
                self.logger.warning(
                    f"Attempt {attempt + 1} timed out for {verse_id}"
                )
            except Exception as e:
                last_error = e
                self.logger.warning(
                    f"Attempt {attempt + 1} failed for {verse_id}: {e}"
                )

            if attempt < self.config.max_retries:
                # Exponential backoff
                await asyncio.sleep(2**attempt * 0.5)

        raise last_error or RuntimeError("Extraction failed with no error captured")

    # -------------------------------------------------------------------------
    # Batch Processing
    # -------------------------------------------------------------------------

    async def process_batch(
        self,
        verses: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        parallel: bool = True,
    ) -> List[ExtractionResult]:
        """
        Process a batch of verses.

        Args:
            verses: List of {verse_id, text, context} dicts
            progress_callback: Optional callback(current, total)
            parallel: Whether to process in parallel

        Returns:
            List of ExtractionResult
        """
        total = len(verses)
        results: List[ExtractionResult] = []

        if parallel:
            # Process in parallel batches
            semaphore = asyncio.Semaphore(self.config.parallel_workers)

            async def process_with_semaphore(verse: Dict[str, Any]) -> ExtractionResult:
                async with semaphore:
                    return await self.process(
                        verse["verse_id"],
                        verse["text"],
                        verse.get("context", {}),
                    )

            tasks = [process_with_semaphore(verse) for verse in verses]

            for i, task in enumerate(asyncio.as_completed(tasks)):
                result = await task
                results.append(result)

                if progress_callback:
                    progress_callback(i + 1, total)

        else:
            # Sequential processing
            for i, verse in enumerate(verses):
                result = await self.process(
                    verse["verse_id"],
                    verse["text"],
                    verse.get("context", {}),
                )
                results.append(result)

                if progress_callback:
                    progress_callback(i + 1, total)

                # Checkpoint
                if (i + 1) % self.config.checkpoint_interval == 0:
                    self.logger.info(f"Checkpoint: {i + 1}/{total} processed")

        return results

    # -------------------------------------------------------------------------
    # Caching
    # -------------------------------------------------------------------------

    def _make_cache_key(self, verse_id: str, text: str) -> str:
        """Generate cache key for verse."""
        content = f"{self.config.name}:{verse_id}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def _get_cached(
        self, cache_key: str, max_age_seconds: float = 3600
    ) -> Optional[ExtractionResult]:
        """Get cached result if not expired."""
        if cache_key not in self._cache:
            return None

        timestamp = self._cache_timestamps.get(cache_key, 0)
        if time.time() - timestamp > max_age_seconds:
            # Expired
            del self._cache[cache_key]
            del self._cache_timestamps[cache_key]
            return None

        return self._cache[cache_key]

    def _set_cached(self, cache_key: str, result: ExtractionResult) -> None:
        """Cache a result."""
        self._cache[cache_key] = result
        self._cache_timestamps[cache_key] = time.time()

    def clear_cache(self) -> int:
        """Clear all cached results. Returns count of cleared items."""
        count = len(self._cache)
        self._cache.clear()
        self._cache_timestamps.clear()
        return count

    # -------------------------------------------------------------------------
    # LangChain Tool Integration
    # -------------------------------------------------------------------------

    def as_langchain_tool(self) -> BaseTool:
        """Convert agent to LangChain tool for use in chains/agents."""
        return AgentTool.from_agent(self)

    def get_tool_schema(self) -> Type[BaseModel]:
        """Get the Pydantic schema for tool inputs."""
        return AgentToolInput


# =============================================================================
# LANGCHAIN TOOL WRAPPER
# =============================================================================


class AgentToolInput(BaseModel):
    """Input schema for agent LangChain tool."""

    model_config = ConfigDict(extra="forbid")

    verse_id: str = Field(
        ...,
        description="Canonical verse ID (e.g., GEN.1.1)",
        pattern=r"^[A-Z0-9]{3}\.\d+\.\d+$",
    )
    text: str = Field(..., description="Verse text in original language", min_length=1)
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context including book info and prior results",
    )


class AgentToolOutput(BaseModel):
    """Output schema for agent LangChain tool."""

    model_config = ConfigDict(extra="forbid")

    verse_id: str
    status: str
    confidence: float
    data: Dict[str, Any]
    errors: List[str]
    processing_time_ms: float


class AgentTool(BaseTool):
    """LangChain tool wrapper for extraction agents."""

    name: str = ""
    description: str = ""
    args_schema: Type[BaseModel] = AgentToolInput
    return_direct: bool = False
    agent: Any = None  # BaseExtractionAgent

    @classmethod
    def from_agent(cls, agent: BaseExtractionAgent) -> "AgentTool":
        """Create tool from agent instance."""
        return cls(
            name=f"extract_{agent.config.name}",
            description=(
                f"Extract {agent.config.extraction_type.value} information "
                f"from biblical text using the {agent.config.name} agent. "
                f"Returns structured extraction results with confidence scores."
            ),
            agent=agent,
        )

    def _run(
        self,
        verse_id: str,
        text: str,
        context: Dict[str, Any] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Synchronous execution - creates event loop if needed."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is None:
            # No running loop, create one
            result = asyncio.run(
                self._arun(verse_id, text, context or {}, run_manager=None)
            )
        else:
            # Running loop exists, schedule coroutine
            import nest_asyncio

            nest_asyncio.apply()
            result = loop.run_until_complete(
                self._arun(verse_id, text, context or {}, run_manager=None)
            )

        return result

    async def _arun(
        self,
        verse_id: str,
        text: str,
        context: Dict[str, Any] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Asynchronous execution."""
        result = await self.agent.process(verse_id, text, context or {})

        output = AgentToolOutput(
            verse_id=result.verse_id,
            status=result.status.value,
            confidence=result.confidence,
            data=result.data,
            errors=result.errors,
            processing_time_ms=result.processing_time_ms,
        )

        return output.model_dump_json(indent=2)


# =============================================================================
# CONTEXT WINDOW MANAGEMENT
# =============================================================================


class ContextManager:
    """Manage context window for long biblical texts."""

    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        # ~4 characters per token for English, ~3 for Greek/Hebrew
        return len(text) // 3

    def compress_context(
        self,
        context: ExtractionContext,
        priority_fields: Optional[List[str]] = None,
    ) -> ExtractionContext:
        """Compress context to fit within token limits."""
        priority_fields = priority_fields or [
            "book",
            "chapter",
            "testament",
            "language",
            "linguistic_results",
        ]

        # Start with priority fields
        compressed = ExtractionContext()
        for field in priority_fields:
            if hasattr(context, field):
                setattr(compressed, field, getattr(context, field))

        # Add remaining fields if space permits
        token_count = self.estimate_tokens(compressed.model_dump_json())

        for field, value in context.model_dump().items():
            if field in priority_fields:
                continue

            field_tokens = self.estimate_tokens(str(value))
            if token_count + field_tokens < self.max_tokens:
                setattr(compressed, field, value)
                token_count += field_tokens

        return compressed

    def split_long_text(
        self,
        text: str,
        chunk_size: int = 1000,
        overlap: int = 100,
    ) -> List[str]:
        """Split long text into overlapping chunks."""
        if self.estimate_tokens(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        char_chunk_size = chunk_size * 3  # Convert tokens to approx chars
        char_overlap = overlap * 3

        while start < len(text):
            end = min(start + char_chunk_size, len(text))
            chunks.append(text[start:end])
            start = end - char_overlap

        return chunks


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def require_initialized(method: Callable) -> Callable:
    """Decorator to ensure agent is initialized before method execution."""

    @wraps(method)
    async def wrapper(self: BaseExtractionAgent, *args, **kwargs):
        if not self.state.is_initialized:
            await self.initialize()
        return await method(self, *args, **kwargs)

    return wrapper


def with_observability(method: Callable) -> Callable:
    """Decorator to add observability to agent methods."""

    @wraps(method)
    async def wrapper(self: BaseExtractionAgent, *args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = await method(self, *args, **kwargs)
            self.logger.debug(
                f"{method.__name__} completed in "
                f"{(time.perf_counter() - start_time) * 1000:.2f}ms"
            )
            return result
        except Exception as e:
            self.logger.error(
                f"{method.__name__} failed after "
                f"{(time.perf_counter() - start_time) * 1000:.2f}ms: {e}"
            )
            raise

    return wrapper
