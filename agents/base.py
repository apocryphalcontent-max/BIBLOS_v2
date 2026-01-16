"""
BIBLOS v2 - Refactored Base Extraction Agent

Production-grade base class combining best patterns from base.py and base_v2.py:
- OpenTelemetry distributed tracing
- Pydantic models for type safety
- Async context managers for resource management
- LRU cache with TTL
- Factory methods for ExtractionResult
- Comprehensive error handling
- Structured logging and observability
"""
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
)
from collections import OrderedDict
import asyncio
import hashlib
import json
import time
import traceback

from pydantic import BaseModel, Field, ConfigDict, field_validator
from langchain_core.tools import BaseTool
from langchain_core.callbacks import (
    CallbackManagerForToolRun,
    AsyncCallbackManagerForToolRun,
)
from opentelemetry.trace import SpanKind, Status, StatusCode

# Import core error types
from core.errors import (
    BiblosError,
    BiblosAgentError,
    BiblosValidationError,
    BiblosTimeoutError,
    BiblosResourceError,
)

# Import centralized schemas
from data.schemas import (
    ProcessingStatus,
    ExtractionResultSchema,
    validate_verse_id,
    normalize_verse_id,
)

# Import observability
from observability import get_tracer, get_logger
from observability.metrics import record_agent_duration, record_cache_access
from observability.logging import AgentLogger


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
    THEOLOGICAL = "theological"
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
# PYDANTIC MODELS
# =============================================================================


class AgentConfig(BaseModel):
    """Configuration for an extraction agent."""

    model_config = ConfigDict(frozen=False, extra="allow")

    # Identity
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

    # Resource limits
    max_memory_mb: int = Field(default=4096, ge=256)
    max_cpu_percent: float = Field(default=80.0, ge=0.0, le=100.0)

    # Feature flags
    enable_caching: bool = Field(default=True)
    enable_validation: bool = Field(default=True)
    enable_metrics: bool = Field(default=True)
    enable_tracing: bool = Field(default=True)

    # LLM settings
    llm_model: str = Field(default="gpt-4-turbo-preview")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)

    # Caching
    cache_ttl_seconds: float = Field(default=3600, ge=0)
    cache_max_size: int = Field(default=1000, ge=0)

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

    Pydantic model for type safety and validation.
    Aligned with ExtractionResultSchema for system-wide uniformity.
    """

    model_config = ConfigDict(extra="forbid")

    # Required fields
    agent_name: str = Field(..., description="Name of the agent")
    extraction_type: ExtractionType = Field(..., description="Type of extraction")
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
    trace_id: Optional[str] = Field(default=None, description="OpenTelemetry trace ID")

    @field_validator("verse_id", mode="before")
    @classmethod
    def normalize_verse_id_validator(cls, v: str) -> str:
        """Normalize verse ID on creation."""
        if v and isinstance(v, str):
            return normalize_verse_id(v)
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "agent_name": self.agent_name,
            "extraction_type": self.extraction_type.value,
            "verse_id": self.verse_id,
            "status": self.status.value,
            "data": self.data,
            "confidence": self.confidence,
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp.isoformat(),
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata,
            "trace_id": self.trace_id,
        }

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
    def success(
        cls,
        agent_name: str,
        extraction_type: ExtractionType,
        verse_id: str,
        data: Dict[str, Any],
        confidence: float,
        processing_time_ms: float = 0.0,
        **kwargs,
    ) -> "ExtractionResult":
        """Factory method: Create a successful result."""
        return cls(
            agent_name=agent_name,
            extraction_type=extraction_type,
            verse_id=verse_id,
            status=ProcessingStatus.COMPLETED,
            data=data,
            confidence=confidence,
            processing_time_ms=processing_time_ms,
            **kwargs,
        )

    @classmethod
    def failure(
        cls,
        agent_name: str,
        extraction_type: ExtractionType,
        verse_id: str,
        error: str,
        processing_time_ms: float = 0.0,
        **kwargs,
    ) -> "ExtractionResult":
        """Factory method: Create a failure result."""
        return cls(
            agent_name=agent_name,
            extraction_type=extraction_type,
            verse_id=verse_id,
            status=ProcessingStatus.FAILED,
            data={},
            confidence=0.0,
            processing_time_ms=processing_time_ms,
            errors=[error],
            **kwargs,
        )

    @classmethod
    def needs_review(
        cls,
        agent_name: str,
        extraction_type: ExtractionType,
        verse_id: str,
        data: Dict[str, Any],
        confidence: float,
        warning: str,
        processing_time_ms: float = 0.0,
        **kwargs,
    ) -> "ExtractionResult":
        """Factory method: Create a needs-review result (uses FAILED status with warning)."""
        return cls(
            agent_name=agent_name,
            extraction_type=extraction_type,
            verse_id=verse_id,
            status=ProcessingStatus.FAILED,  # Use FAILED status for review cases
            data=data,
            confidence=confidence,
            processing_time_ms=processing_time_ms,
            warnings=[warning],
            **kwargs,
        )


class AgentState(BaseModel):
    """State container for agent execution."""

    model_config = ConfigDict(extra="allow")

    # Lifecycle state
    is_initialized: bool = False
    is_processing: bool = False
    current_verse_id: Optional[str] = None

    # Statistics
    total_processed: int = 0
    successful: int = 0
    failed: int = 0
    total_time_ms: float = 0.0
    avg_confidence: float = 0.0
    _confidence_sum: float = 0.0

    # Cache statistics
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
        self._confidence_sum += result.confidence
        self.avg_confidence = self._confidence_sum / self.successful
        self.consecutive_errors = 0

    def record_failure(self, error: str) -> None:
        """Record a failed extraction."""
        self.total_processed += 1
        self.failed += 1
        self.last_error = error
        self.consecutive_errors += 1

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.successful / max(1, self.total_processed)


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
# LRU CACHE WITH TTL
# =============================================================================


class LRUCacheWithTTL:
    """LRU cache with time-to-live expiration."""

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, ExtractionResult] = OrderedDict()
        self._timestamps: Dict[str, float] = {}

    def get(self, key: str) -> Optional[ExtractionResult]:
        """Get cached result if not expired."""
        if key not in self._cache:
            return None

        timestamp = self._timestamps.get(key, 0)
        if time.time() - timestamp > self.ttl_seconds:
            # Expired - remove it
            del self._cache[key]
            del self._timestamps[key]
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        return self._cache[key]

    def set(self, key: str, value: ExtractionResult) -> None:
        """Cache a result."""
        # Remove if already exists
        if key in self._cache:
            del self._cache[key]

        # Add to end
        self._cache[key] = value
        self._timestamps[key] = time.time()

        # Evict oldest if over max size
        if len(self._cache) > self.max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            del self._timestamps[oldest_key]

    def clear(self) -> int:
        """Clear all cached results. Returns count of cleared items."""
        count = len(self._cache)
        self._cache.clear()
        self._timestamps.clear()
        return count

    def __len__(self) -> int:
        """Return cache size."""
        return len(self._cache)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_current_trace_id() -> Optional[str]:
    """Get current OpenTelemetry trace ID as hex string."""
    from opentelemetry import trace

    span = trace.get_current_span()
    if span and span.is_recording():
        ctx = span.get_span_context()
        if ctx.is_valid:
            return format(ctx.trace_id, "032x")
    return None


# =============================================================================
# BASE EXTRACTION AGENT
# =============================================================================

T = TypeVar("T", bound="BaseExtractionAgent")


class BaseExtractionAgent(ABC):
    """
    Abstract base class for SDES extraction agents.

    Production-grade implementation combining best patterns:
    - OpenTelemetry distributed tracing
    - Pydantic models for type safety
    - Async context managers for resource management
    - LRU cache with TTL
    - Factory methods for results
    - Comprehensive error handling

    All extraction agents must inherit from this class and implement:
    - extract(): Core extraction logic
    - validate(): Result validation
    - get_dependencies(): Agent dependencies
    """

    def __init__(self, config: AgentConfig):
        """Initialize agent with configuration."""
        self.config = config
        self.state = AgentState()
        self.metrics = AgentMetrics(agent_name=config.name)

        # Caching
        self._cache = LRUCacheWithTTL(
            max_size=config.cache_max_size,
            ttl_seconds=config.cache_ttl_seconds,
        )

        # Observability
        self._tracer = get_tracer(f"biblos.agents.{config.name}")
        self._logger = get_logger(f"biblos.agents.{config.name}")
        self._agent_logger = AgentLogger(config.name)

    # -------------------------------------------------------------------------
    # Context Manager Protocol
    # -------------------------------------------------------------------------

    async def __aenter__(self: T) -> T:
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit with cleanup."""
        await self.shutdown()

    # -------------------------------------------------------------------------
    # Lifecycle Methods
    # -------------------------------------------------------------------------

    async def initialize(self) -> None:
        """Initialize agent resources with tracing."""
        if self.state.is_initialized:
            return

        with self._tracer.start_as_current_span(
            f"agent.{self.config.name}.initialize",
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute("agent.name", self.config.name)
            span.set_attribute("agent.extraction_type", self.config.extraction_type.value)

            self._logger.info(
                f"Initializing agent: {self.config.name}",
                agent=self.config.name,
            )

            try:
                await self._setup_resources()
                self.state.is_initialized = True
                span.set_status(Status(StatusCode.OK))
                self._logger.info(
                    f"Agent {self.config.name} initialized successfully",
                    agent=self.config.name,
                )
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                self._logger.error(
                    f"Failed to initialize agent: {e}",
                    agent=self.config.name,
                    error=str(e),
                )
                raise

    async def _setup_resources(self) -> None:
        """Override to set up agent-specific resources."""
        pass

    async def shutdown(self) -> None:
        """Clean up agent resources with tracing."""
        with self._tracer.start_as_current_span(
            f"agent.{self.config.name}.shutdown",
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute("agent.name", self.config.name)

            self._logger.info(
                f"Shutting down agent: {self.config.name}",
                agent=self.config.name,
            )

            try:
                await self._cleanup_resources()
                self.state.is_initialized = False
                self._cache.clear()
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                self._logger.error(
                    f"Error during shutdown: {e}",
                    agent=self.config.name,
                    error=str(e),
                )
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
    ) -> ExtractionResult:
        """
        Process a verse with caching, validation, metrics, and tracing.

        Creates a span for the entire agent execution, with child spans
        for cache lookup, extraction, and validation.

        Args:
            verse_id: Canonical verse identifier
            text: Verse text
            context: Extraction context (dict or typed)

        Returns:
            ExtractionResult
        """
        # Convert dict context to typed
        if isinstance(context, dict):
            context = ExtractionContext(**context)

        # Create parent span for agent execution
        with self._tracer.start_as_current_span(
            f"agent.{self.config.name}.process",
            kind=SpanKind.INTERNAL,
        ) as agent_span:
            # Set span attributes
            agent_span.set_attribute("agent.name", self.config.name)
            agent_span.set_attribute("agent.extraction_type", self.config.extraction_type.value)
            agent_span.set_attribute("verse.id", verse_id)
            agent_span.set_attribute("verse.text_length", len(text))

            self._agent_logger.start_extraction(verse_id)

            # Check cache
            cache_key = self._make_cache_key(verse_id, text)
            if self.config.enable_caching:
                with self._tracer.start_as_current_span(
                    f"agent.{self.config.name}.cache_lookup"
                ) as cache_span:
                    cache_span.set_attribute("cache.key", cache_key[:16])

                    cached = self._cache.get(cache_key)
                    if cached is not None:
                        cache_span.set_attribute("cache.hit", True)
                        record_cache_access(hit=True, cache_type=f"agent.{self.config.name}")
                        self.state.cache_hits += 1
                        self._logger.debug(
                            f"Cache hit for {verse_id}",
                            verse_id=verse_id,
                            agent=self.config.name,
                        )
                        # Update trace ID
                        cached.trace_id = get_current_trace_id()
                        return cached
                    else:
                        cache_span.set_attribute("cache.hit", False)
                        record_cache_access(hit=False, cache_type=f"agent.{self.config.name}")
                        self.state.cache_misses += 1

            # Perform extraction with timing
            start_time = time.perf_counter()
            self.state.is_processing = True
            self.state.current_verse_id = verse_id

            try:
                # Execute with retry and timeout
                with self._tracer.start_as_current_span(
                    f"agent.{self.config.name}.extract",
                    kind=SpanKind.INTERNAL,
                ) as extract_span:
                    extract_span.set_attribute("verse.id", verse_id)

                    result = await self._execute_with_retry(verse_id, text, context)
                    result.processing_time_ms = (time.perf_counter() - start_time) * 1000
                    result.trace_id = get_current_trace_id()

                    extract_span.set_attribute("result.confidence", result.confidence)
                    extract_span.set_attribute("result.status", result.status.value)
                    extract_span.set_attribute("result.data_keys", list(result.data.keys()))

                # Validate if enabled
                if self.config.enable_validation:
                    with self._tracer.start_as_current_span(
                        f"agent.{self.config.name}.validate"
                    ) as validate_span:
                        is_valid = await self.validate(result)
                        validate_span.set_attribute("validation.passed", is_valid)

                        if not is_valid:
                            # Keep status as COMPLETED but add warning
                            result.warnings.append("Validation failed")
                            validate_span.set_attribute("validation.warning", "Failed validation")

                # Record success
                self.state.record_success(result)

            except asyncio.TimeoutError as e:
                processing_time = (time.perf_counter() - start_time) * 1000
                error_msg = f"Extraction timed out after {self.config.timeout_seconds}s"

                agent_span.set_status(Status(StatusCode.ERROR, error_msg))
                agent_span.set_attribute("error.type", "timeout")

                self._logger.warning(
                    f"Extraction timeout for {verse_id}",
                    verse_id=verse_id,
                    agent=self.config.name,
                    timeout_seconds=self.config.timeout_seconds,
                )

                result = ExtractionResult.failure(
                    agent_name=self.config.name,
                    extraction_type=self.config.extraction_type,
                    verse_id=verse_id,
                    error=error_msg,
                    processing_time_ms=processing_time,
                    trace_id=get_current_trace_id(),
                )
                self.state.record_failure(error_msg)

            except BiblosAgentError as e:
                processing_time = (time.perf_counter() - start_time) * 1000

                agent_span.set_status(Status(StatusCode.ERROR, e.message))
                agent_span.set_attribute("error.type", "agent_error")
                agent_span.set_attribute("error.code", e.error_code)

                self._logger.error(
                    f"Agent error for {verse_id}: {e.message}",
                    verse_id=verse_id,
                    agent=self.config.name,
                    error_code=e.error_code,
                )

                result = ExtractionResult.failure(
                    agent_name=self.config.name,
                    extraction_type=self.config.extraction_type,
                    verse_id=verse_id,
                    error=e.message,
                    processing_time_ms=processing_time,
                    trace_id=get_current_trace_id(),
                )
                self.state.record_failure(e.message)

            except BiblosValidationError as e:
                processing_time = (time.perf_counter() - start_time) * 1000

                agent_span.set_status(Status(StatusCode.OK))
                agent_span.set_attribute("error.type", "validation_error")

                self._logger.warning(
                    f"Validation error for {verse_id}: {e.message}",
                    verse_id=verse_id,
                    agent=self.config.name,
                    field=e.field_name,
                )

                # Validation errors result in completed status with warnings
                result = ExtractionResult.success(
                    agent_name=self.config.name,
                    extraction_type=self.config.extraction_type,
                    verse_id=verse_id,
                    data={},
                    confidence=0.5,  # Lower confidence for validation issues
                    processing_time_ms=processing_time,
                    trace_id=get_current_trace_id(),
                )
                result.warnings.append(e.message)
                self.state.record_success(result)

            except (MemoryError, BiblosResourceError) as e:
                processing_time = (time.perf_counter() - start_time) * 1000
                error_msg = str(e) if isinstance(e, MemoryError) else e.message

                agent_span.set_status(Status(StatusCode.ERROR, error_msg))
                agent_span.set_attribute("error.type", "resource_error")

                self._logger.error(
                    f"Resource error for {verse_id}: {error_msg}",
                    verse_id=verse_id,
                    agent=self.config.name,
                )

                result = ExtractionResult.failure(
                    agent_name=self.config.name,
                    extraction_type=self.config.extraction_type,
                    verse_id=verse_id,
                    error=f"Resource error: {error_msg}",
                    processing_time_ms=processing_time,
                    trace_id=get_current_trace_id(),
                )
                self.state.record_failure(error_msg)

            except BiblosError as e:
                processing_time = (time.perf_counter() - start_time) * 1000

                agent_span.set_status(Status(StatusCode.ERROR, e.message))
                agent_span.record_exception(e)

                self._logger.error(
                    f"BIBLOS error for {verse_id}: {e.message}",
                    verse_id=verse_id,
                    agent=self.config.name,
                    error_code=e.error_code,
                )

                result = ExtractionResult.failure(
                    agent_name=self.config.name,
                    extraction_type=self.config.extraction_type,
                    verse_id=verse_id,
                    error=e.message,
                    processing_time_ms=processing_time,
                    trace_id=get_current_trace_id(),
                )
                self.state.record_failure(e.message)

            except Exception as e:
                processing_time = (time.perf_counter() - start_time) * 1000

                agent_span.set_status(Status(StatusCode.ERROR, str(e)))
                agent_span.record_exception(e)
                agent_span.set_attribute("error.type", "unexpected")

                self._logger.error(
                    f"Unexpected error for {verse_id}: {e}",
                    verse_id=verse_id,
                    agent=self.config.name,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                self._logger.debug(traceback.format_exc())

                result = ExtractionResult.failure(
                    agent_name=self.config.name,
                    extraction_type=self.config.extraction_type,
                    verse_id=verse_id,
                    error=f"Unexpected error: {type(e).__name__}: {e}",
                    processing_time_ms=processing_time,
                    trace_id=get_current_trace_id(),
                )
                self.state.record_failure(str(e))

            finally:
                self.state.is_processing = False
                self.state.current_verse_id = None

            # Record metrics
            if self.config.enable_metrics:
                self.metrics.record(result)

                # Record to OpenTelemetry metrics
                status_str = "completed" if result.status == ProcessingStatus.COMPLETED else "failed"
                record_agent_duration(
                    self.config.name,
                    self.config.extraction_type.value,
                    verse_id,
                    result.processing_time_ms / 1000.0,
                    result.confidence,
                    status_str,
                )

            # Set final span attributes
            agent_span.set_attribute("result.status", result.status.value)
            agent_span.set_attribute("result.confidence", result.confidence)
            agent_span.set_attribute("processing_time_ms", result.processing_time_ms)
            agent_span.set_attribute("result.error_count", len(result.errors))
            agent_span.set_attribute("result.warning_count", len(result.warnings))

            if result.status == ProcessingStatus.COMPLETED:
                agent_span.set_status(Status(StatusCode.OK))
            elif result.status == ProcessingStatus.FAILED:
                agent_span.set_status(Status(StatusCode.ERROR, "; ".join(result.errors)))
            else:
                agent_span.set_status(Status(StatusCode.OK))

            self._agent_logger.end_extraction(
                verse_id,
                result.status.value,
                result.confidence,
                result.processing_time_ms,
            )

            # Cache successful results
            if self.config.enable_caching and result.status == ProcessingStatus.COMPLETED:
                self._cache.set(cache_key, result)

            return result

    async def _execute_with_retry(
        self,
        verse_id: str,
        text: str,
        context: ExtractionContext,
    ) -> ExtractionResult:
        """Execute extraction with retry logic and timeout."""
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
                self._logger.warning(
                    f"Attempt {attempt + 1} timed out for {verse_id}",
                    verse_id=verse_id,
                    attempt=attempt + 1,
                )
            except Exception as e:
                last_error = e
                self._logger.warning(
                    f"Attempt {attempt + 1} failed for {verse_id}: {e}",
                    verse_id=verse_id,
                    attempt=attempt + 1,
                    error=str(e),
                )

            if attempt < self.config.max_retries:
                # Exponential backoff
                backoff_time = 2**attempt * 0.5
                await asyncio.sleep(backoff_time)

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
        Process a batch of verses with optional parallelization.

        Args:
            verses: List of {verse_id, text, context} dicts
            progress_callback: Optional callback(current, total)
            parallel: Whether to process in parallel

        Returns:
            List of ExtractionResult
        """
        with self._tracer.start_as_current_span(
            f"agent.{self.config.name}.process_batch",
            kind=SpanKind.INTERNAL,
        ) as batch_span:
            batch_span.set_attribute("batch.size", len(verses))
            batch_span.set_attribute("agent.name", self.config.name)
            batch_span.set_attribute("batch.parallel", parallel)

            total = len(verses)
            results: List[ExtractionResult] = []

            if parallel:
                # Process in parallel with semaphore
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
                        self._logger.info(
                            f"Checkpoint: {i + 1}/{total} processed",
                            processed=i + 1,
                            total=total,
                            agent=self.config.name,
                        )

            # Set batch completion attributes
            successful = sum(1 for r in results if r.status == ProcessingStatus.COMPLETED)
            batch_span.set_attribute("batch.successful", successful)
            batch_span.set_attribute("batch.failed", len(results) - successful)

            return results

    # -------------------------------------------------------------------------
    # Caching
    # -------------------------------------------------------------------------

    def _make_cache_key(self, verse_id: str, text: str) -> str:
        """Generate cache key for verse."""
        content = f"{self.config.name}:{verse_id}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def clear_cache(self) -> int:
        """Clear all cached results. Returns count of cleared items."""
        return self._cache.clear()

    # -------------------------------------------------------------------------
    # LangChain Tool Integration
    # -------------------------------------------------------------------------

    def as_langchain_tool(self) -> BaseTool:
        """Convert agent to LangChain tool for use in chains/agents."""
        return AgentTool.from_agent(self)


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


class AgentTool(BaseTool):
    """LangChain tool wrapper for extraction agents with tracing."""

    name: str = ""
    description: str = ""
    args_schema: Type[BaseModel] = AgentToolInput
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
        """Synchronous execution with tracing."""
        tracer = get_tracer("biblos.agents.langchain")
        with tracer.start_as_current_span(
            f"langchain.tool.{self.name}",
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute("tool.name", self.name)
            span.set_attribute("verse.id", verse_id)

            result = asyncio.run(self.agent.process(verse_id, text, context or {}))
            return json.dumps(result.to_dict(), indent=2)

    async def _arun(
        self,
        verse_id: str,
        text: str,
        context: Dict[str, Any] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Asynchronous execution with tracing."""
        tracer = get_tracer("biblos.agents.langchain")
        with tracer.start_as_current_span(
            f"langchain.tool.{self.name}",
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute("tool.name", self.name)
            span.set_attribute("verse.id", verse_id)

            result = await self.agent.process(verse_id, text, context or {})
            return json.dumps(result.to_dict(), indent=2)
