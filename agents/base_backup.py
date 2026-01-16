"""
BIBLOS v2 - Base Extraction Agent with LangChain Integration

This module provides the abstract base class for all SDES extraction agents,
with integrated LangChain tool capabilities for AI-augmented extraction
and comprehensive OpenTelemetry distributed tracing.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, TypeVar, Generic, Callable
from pathlib import Path
import asyncio
import hashlib
import json
import time

from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

# Import core error types for specific exception handling
from core.errors import (
    BiblosError,
    BiblosAgentError,
    BiblosValidationError,
    BiblosTimeoutError,
    BiblosResourceError,
    ErrorContext,
)

# Import centralized schemas for system-wide uniformity
from data.schemas import (
    ProcessingStatus,
    ExtractionResultSchema,
    validate_verse_id,
    normalize_verse_id
)

# Import observability
from observability import get_tracer, get_logger
from observability.metrics import (
    record_agent_duration,
    record_cache_access,
    timed_agent,
)
from observability.logging import AgentLogger

T = TypeVar('T')


class ExtractionType(Enum):
    """Categories of extraction operations."""
    PHONOLOGICAL = auto()
    MORPHOLOGICAL = auto()
    SYNTACTIC = auto()
    SEMANTIC = auto()
    DISCOURSE = auto()
    PRAGMATIC = auto()
    LEXICAL = auto()
    ETYMOLOGICAL = auto()
    PATRISTIC = auto()
    THEOLOGICAL = auto()
    TYPOLOGICAL = auto()
    LITURGICAL = auto()
    INTERTEXTUAL = auto()
    STRUCTURAL = auto()
    VALIDATION = auto()


@dataclass
class AgentConfig:
    """Configuration for an extraction agent."""
    name: str
    extraction_type: ExtractionType
    batch_size: int = 1000
    max_retries: int = 3
    timeout_seconds: int = 300
    checkpoint_interval: int = 500
    parallel_workers: int = 4

    # Quality thresholds
    min_confidence: float = 0.7
    min_coverage: float = 0.9

    # Resource limits
    max_memory_mb: int = 4096
    max_cpu_percent: float = 80.0

    # Feature flags
    enable_caching: bool = True
    enable_validation: bool = True
    enable_metrics: bool = True
    enable_langchain: bool = True
    enable_tracing: bool = True

    # LangChain settings
    llm_model: str = "gpt-4-turbo-preview"
    temperature: float = 0.0
    max_tokens: int = 4096

    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """
    Result from an extraction operation.

    Aligned with ExtractionResultSchema for system-wide uniformity.
    Uses centralized ProcessingStatus enum from data.schemas.
    """
    agent_name: str
    extraction_type: ExtractionType
    verse_id: str
    status: ProcessingStatus
    data: Dict[str, Any]
    confidence: float
    processing_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None

    def __post_init__(self):
        """Validate and normalize verse_id on creation."""
        if self.verse_id and not validate_verse_id(self.verse_id):
            self.warnings.append(f"Invalid verse_id format: {self.verse_id}")
        elif self.verse_id:
            self.verse_id = normalize_verse_id(self.verse_id)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "agent_name": self.agent_name,
            "extraction_type": self.extraction_type.name,
            "verse_id": self.verse_id,
            "status": self.status.value,
            "data": self.data,
            "confidence": self.confidence,
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp.isoformat(),
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata,
            "trace_id": self.trace_id
        }

    def to_schema(self) -> ExtractionResultSchema:
        """Convert to centralized ExtractionResultSchema."""
        return ExtractionResultSchema(
            agent_name=self.agent_name,
            extraction_type=self.extraction_type.name.lower(),
            verse_id=self.verse_id,
            status=self.status.value,
            confidence=self.confidence,
            data=self.data,
            error="; ".join(self.errors) if self.errors else None,
            processing_time=self.processing_time_ms / 1000.0  # Convert to seconds
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractionResult":
        """Deserialize from dictionary."""
        return cls(
            agent_name=data["agent_name"],
            extraction_type=ExtractionType[data["extraction_type"]],
            verse_id=data["verse_id"],
            status=ProcessingStatus(data["status"]),
            data=data["data"],
            confidence=data["confidence"],
            processing_time_ms=data["processing_time_ms"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            errors=data.get("errors", []),
            warnings=data.get("warnings", []),
            metadata=data.get("metadata", {}),
            trace_id=data.get("trace_id")
        )

    @classmethod
    def from_schema(cls, schema: ExtractionResultSchema, extraction_type: ExtractionType) -> "ExtractionResult":
        """Create from ExtractionResultSchema."""
        return cls(
            agent_name=schema.agent_name,
            extraction_type=extraction_type,
            verse_id=schema.verse_id,
            status=ProcessingStatus(schema.status),
            data=schema.data,
            confidence=schema.confidence,
            processing_time_ms=schema.processing_time * 1000.0,  # Convert from seconds
            errors=[schema.error] if schema.error else [],
            warnings=[],
            metadata={}
        )

    @classmethod
    def create_error(
        cls,
        agent_name: str,
        extraction_type: ExtractionType,
        verse_id: str,
        error_message: str,
        processing_time_ms: float = 0.0,
        trace_id: Optional[str] = None,
        status: ProcessingStatus = ProcessingStatus.FAILED,
    ) -> "ExtractionResult":
        """
        Factory method for creating error results.

        Consolidates the repeated error result creation pattern
        used throughout exception handling.

        Args:
            agent_name: Name of the agent that failed
            extraction_type: Type of extraction being performed
            verse_id: Verse being processed when error occurred
            error_message: Description of the error
            processing_time_ms: Time spent before failure
            trace_id: OpenTelemetry trace ID for correlation
            status: Processing status (FAILED or NEEDS_REVIEW)

        Returns:
            ExtractionResult configured for the error condition
        """
        return cls(
            agent_name=agent_name,
            extraction_type=extraction_type,
            verse_id=verse_id,
            status=status,
            data={},
            confidence=0.0,
            processing_time_ms=processing_time_ms,
            errors=[error_message],
            trace_id=trace_id,
        )

    @classmethod
    def create_validation_warning(
        cls,
        agent_name: str,
        extraction_type: ExtractionType,
        verse_id: str,
        warning_message: str,
        processing_time_ms: float = 0.0,
        trace_id: Optional[str] = None,
    ) -> "ExtractionResult":
        """
        Factory method for creating validation warning results.

        Used when validation fails but the result may be recoverable.

        Args:
            agent_name: Name of the agent
            extraction_type: Type of extraction
            verse_id: Verse being processed
            warning_message: Description of the validation issue
            processing_time_ms: Time spent processing
            trace_id: OpenTelemetry trace ID

        Returns:
            ExtractionResult configured for needs-review status
        """
        return cls(
            agent_name=agent_name,
            extraction_type=extraction_type,
            verse_id=verse_id,
            status=ProcessingStatus.NEEDS_REVIEW,
            data={},
            confidence=0.0,
            processing_time_ms=processing_time_ms,
            warnings=[warning_message],
            trace_id=trace_id,
        )


class AgentMetrics:
    """Metrics collector for agent performance monitoring."""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.total_processed = 0
        self.successful = 0
        self.failed = 0
        self.total_time_ms = 0.0
        self.avg_confidence = 0.0
        self._confidence_sum = 0.0

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
        return {
            "agent_name": self.agent_name,
            "total_processed": self.total_processed,
            "successful": self.successful,
            "failed": self.failed,
            "success_rate": self.successful / max(1, self.total_processed),
            "total_time_ms": self.total_time_ms,
            "avg_time_ms": self.total_time_ms / max(1, self.total_processed),
            "avg_confidence": self.avg_confidence
        }


def get_current_trace_id() -> Optional[str]:
    """Get current trace ID as hex string."""
    span = trace.get_current_span()
    if span and span.is_recording():
        ctx = span.get_span_context()
        if ctx.is_valid:
            return format(ctx.trace_id, "032x")
    return None


class BaseExtractionAgent(ABC):
    """
    Abstract base class for SDES extraction agents.

    All extraction agents must inherit from this class and implement:
    - extract(): Core extraction logic
    - validate(): Result validation
    - get_dependencies(): Agent dependencies

    Includes comprehensive OpenTelemetry tracing for flame graph visualization.
    Each agent execution creates child spans under the parent phase span.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.metrics = AgentMetrics(config.name)
        self._cache: Dict[str, ExtractionResult] = {}
        self._initialized = False

        # Initialize observability components
        self._tracer = get_tracer(f"biblos.agents.{config.name}")
        self._logger = get_logger(f"biblos.agents.{config.name}")
        self._agent_logger = AgentLogger(config.name)

    async def initialize(self) -> None:
        """Initialize agent resources with tracing."""
        if self._initialized:
            return

        with self._tracer.start_as_current_span(
            f"agent.{self.config.name}.initialize",
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute("agent.name", self.config.name)
            span.set_attribute("agent.extraction_type", self.config.extraction_type.name)

            self._logger.info(
                f"Initializing agent: {self.config.name}",
                agent=self.config.name,
            )

            try:
                await self._setup_resources()
                self._initialized = True
                span.set_attribute("status", "success")
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                self._logger.error(
                    f"Failed to initialize agent: {self.config.name}",
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

            await self._cleanup_resources()
            self._initialized = False

    async def _cleanup_resources(self) -> None:
        """Override to clean up agent-specific resources."""
        pass

    @abstractmethod
    async def extract(
        self,
        verse_id: str,
        text: str,
        context: Dict[str, Any]
    ) -> ExtractionResult:
        """
        Perform extraction on a verse.

        Args:
            verse_id: Canonical verse identifier (e.g., "GEN.1.1")
            text: Verse text in original language
            context: Additional context (book, chapter, parallel texts, etc.)

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

    async def process(
        self,
        verse_id: str,
        text: str,
        context: Dict[str, Any]
    ) -> ExtractionResult:
        """
        Process a verse with caching, metrics, and comprehensive tracing.

        Creates a span for the entire agent execution, with child spans
        for cache lookup, extraction, and validation. This enables
        detailed flame graph visualization of agent performance.

        Args:
            verse_id: Canonical verse identifier
            text: Verse text
            context: Additional context

        Returns:
            ExtractionResult
        """
        # Create parent span for agent execution
        with self._tracer.start_as_current_span(
            f"agent.{self.config.name}.process",
            kind=SpanKind.INTERNAL,
        ) as agent_span:
            # Set span attributes
            agent_span.set_attribute("agent.name", self.config.name)
            agent_span.set_attribute("agent.extraction_type", self.config.extraction_type.name)
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

                    if cache_key in self._cache:
                        cache_span.set_attribute("cache.hit", True)
                        record_cache_access(hit=True, cache_type=f"agent.{self.config.name}")
                        self._logger.debug(
                            f"Cache hit for {verse_id}",
                            verse_id=verse_id,
                            agent=self.config.name,
                        )
                        cached_result = self._cache[cache_key]
                        cached_result.trace_id = get_current_trace_id()
                        return cached_result
                    else:
                        cache_span.set_attribute("cache.hit", False)
                        record_cache_access(hit=False, cache_type=f"agent.{self.config.name}")

            # Perform extraction
            start_time = time.perf_counter()
            try:
                with self._tracer.start_as_current_span(
                    f"agent.{self.config.name}.extract",
                    kind=SpanKind.INTERNAL,
                ) as extract_span:
                    extract_span.set_attribute("verse.id", verse_id)

                    result = await self.extract(verse_id, text, context)
                    result.processing_time_ms = (time.perf_counter() - start_time) * 1000
                    result.trace_id = get_current_trace_id()

                    extract_span.set_attribute("result.confidence", result.confidence)
                    extract_span.set_attribute("result.status", result.status.value)
                    extract_span.set_attribute("result.data_keys", list(result.data.keys()))

                # Validate
                if self.config.enable_validation:
                    with self._tracer.start_as_current_span(
                        f"agent.{self.config.name}.validate"
                    ) as validate_span:
                        is_valid = await self.validate(result)
                        validate_span.set_attribute("validation.passed", is_valid)

                        if not is_valid:
                            result.status = ProcessingStatus.NEEDS_REVIEW
                            result.warnings.append("Validation failed")
                            validate_span.set_attribute("validation.warning", "Failed validation")

            except asyncio.TimeoutError as e:
                # Handle timeout specifically
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

                result = ExtractionResult(
                    agent_name=self.config.name,
                    extraction_type=self.config.extraction_type,
                    verse_id=verse_id,
                    status=ProcessingStatus.FAILED,
                    data={},
                    confidence=0.0,
                    processing_time_ms=processing_time,
                    errors=[error_msg],
                    trace_id=get_current_trace_id()
                )

            except BiblosAgentError as e:
                # Handle agent-specific errors
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

                result = ExtractionResult(
                    agent_name=self.config.name,
                    extraction_type=self.config.extraction_type,
                    verse_id=verse_id,
                    status=ProcessingStatus.FAILED,
                    data={},
                    confidence=0.0,
                    processing_time_ms=processing_time,
                    errors=[e.message],
                    trace_id=get_current_trace_id()
                )

            except BiblosValidationError as e:
                # Handle validation errors - mark for review instead of failed
                processing_time = (time.perf_counter() - start_time) * 1000

                agent_span.set_status(Status(StatusCode.OK))  # Validation errors are recoverable
                agent_span.set_attribute("error.type", "validation_error")

                self._logger.warning(
                    f"Validation error for {verse_id}: {e.message}",
                    verse_id=verse_id,
                    agent=self.config.name,
                    field=e.field_name,
                )

                result = ExtractionResult(
                    agent_name=self.config.name,
                    extraction_type=self.config.extraction_type,
                    verse_id=verse_id,
                    status=ProcessingStatus.NEEDS_REVIEW,
                    data={},
                    confidence=0.0,
                    processing_time_ms=processing_time,
                    warnings=[e.message],
                    trace_id=get_current_trace_id()
                )

            except (MemoryError, BiblosResourceError) as e:
                # Handle resource exhaustion
                processing_time = (time.perf_counter() - start_time) * 1000
                error_msg = str(e) if isinstance(e, MemoryError) else e.message

                agent_span.set_status(Status(StatusCode.ERROR, error_msg))
                agent_span.set_attribute("error.type", "resource_error")

                self._logger.error(
                    f"Resource error for {verse_id}: {error_msg}",
                    verse_id=verse_id,
                    agent=self.config.name,
                )

                result = ExtractionResult(
                    agent_name=self.config.name,
                    extraction_type=self.config.extraction_type,
                    verse_id=verse_id,
                    status=ProcessingStatus.FAILED,
                    data={},
                    confidence=0.0,
                    processing_time_ms=processing_time,
                    errors=[f"Resource error: {error_msg}"],
                    trace_id=get_current_trace_id()
                )

            except BiblosError as e:
                # Handle other BIBLOS errors
                processing_time = (time.perf_counter() - start_time) * 1000

                agent_span.set_status(Status(StatusCode.ERROR, e.message))
                agent_span.record_exception(e)

                self._logger.error(
                    f"BIBLOS error for {verse_id}: {e.message}",
                    verse_id=verse_id,
                    agent=self.config.name,
                    error_code=e.error_code,
                )

                result = ExtractionResult(
                    agent_name=self.config.name,
                    extraction_type=self.config.extraction_type,
                    verse_id=verse_id,
                    status=ProcessingStatus.FAILED,
                    data={},
                    confidence=0.0,
                    processing_time_ms=processing_time,
                    errors=[e.message],
                    trace_id=get_current_trace_id()
                )

            except Exception as e:
                # Catch-all for unexpected errors (should be rare)
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

                result = ExtractionResult(
                    agent_name=self.config.name,
                    extraction_type=self.config.extraction_type,
                    verse_id=verse_id,
                    status=ProcessingStatus.FAILED,
                    data={},
                    confidence=0.0,
                    processing_time_ms=processing_time,
                    errors=[f"Unexpected error: {type(e).__name__}: {e}"],
                    trace_id=get_current_trace_id()
                )

            # Record metrics
            if self.config.enable_metrics:
                self.metrics.record(result)

                # Record to OpenTelemetry metrics
                status_str = "completed" if result.status == ProcessingStatus.COMPLETED else "failed"
                record_agent_duration(
                    self.config.name,
                    self.config.extraction_type.name.lower(),
                    verse_id,
                    result.processing_time_ms / 1000.0,  # Convert to seconds
                    result.confidence,
                    status_str
                )

            # Set final span attributes
            agent_span.set_attribute("result.status", result.status.value)
            agent_span.set_attribute("result.confidence", result.confidence)
            agent_span.set_attribute("processing_time_ms", result.processing_time_ms)
            agent_span.set_attribute("result.error_count", len(result.errors))
            agent_span.set_attribute("result.warning_count", len(result.warnings))

            if result.status == ProcessingStatus.COMPLETED:
                agent_span.set_status(Status(StatusCode.OK))
            elif result.status == ProcessingStatus.NEEDS_REVIEW:
                agent_span.set_status(Status(StatusCode.OK))  # Needs review is not an error
            else:
                agent_span.set_status(Status(StatusCode.ERROR, "; ".join(result.errors)))

            self._agent_logger.end_extraction(
                verse_id,
                result.status.value,
                result.confidence,
                result.processing_time_ms
            )

            # Cache result
            if self.config.enable_caching and result.status == ProcessingStatus.COMPLETED:
                self._cache[cache_key] = result

            return result

    def _make_cache_key(self, verse_id: str, text: str) -> str:
        """Generate cache key for verse."""
        content = f"{self.config.name}:{verse_id}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    async def process_batch(
        self,
        verses: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[ExtractionResult]:
        """
        Process a batch of verses with tracing.

        Args:
            verses: List of {verse_id, text, context} dicts
            progress_callback: Optional callback(current, total)

        Returns:
            List of ExtractionResult
        """
        with self._tracer.start_as_current_span(
            f"agent.{self.config.name}.process_batch",
            kind=SpanKind.INTERNAL,
        ) as batch_span:
            batch_span.set_attribute("batch.size", len(verses))
            batch_span.set_attribute("agent.name", self.config.name)

            results = []
            total = len(verses)

            for i, verse in enumerate(verses):
                result = await self.process(
                    verse["verse_id"],
                    verse["text"],
                    verse.get("context", {})
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

    def as_langchain_tool(self) -> BaseTool:
        """Convert agent to LangChain tool for use in chains/agents."""
        return AgentLangChainTool(agent=self)


class AgentToolInput(BaseModel):
    """Input schema for agent LangChain tool."""
    verse_id: str = Field(description="Canonical verse ID (e.g., GEN.1.1)")
    text: str = Field(description="Verse text in original language")
    context: Dict[str, Any] = Field(default={}, description="Additional context")


class AgentLangChainTool(BaseTool):
    """LangChain tool wrapper for extraction agents with tracing."""

    name: str = ""
    description: str = ""
    args_schema: type = AgentToolInput
    agent: Any = None  # BaseExtractionAgent

    def __init__(self, agent: BaseExtractionAgent, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent
        self.name = f"extract_{agent.config.name}"
        self.description = f"Extract {agent.config.extraction_type.name.lower()} " \
                          f"information from biblical text using {agent.config.name} agent"

    def _run(
        self,
        verse_id: str,
        text: str,
        context: Dict[str, Any] = {},
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Synchronous execution with tracing."""
        tracer = get_tracer("biblos.agents.langchain")
        with tracer.start_as_current_span(
            f"langchain.tool.{self.name}",
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute("tool.name", self.name)
            span.set_attribute("verse.id", verse_id)

            result = asyncio.run(self.agent.process(verse_id, text, context))
            return json.dumps(result.to_dict(), indent=2)

    async def _arun(
        self,
        verse_id: str,
        text: str,
        context: Dict[str, Any] = {},
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Asynchronous execution with tracing."""
        tracer = get_tracer("biblos.agents.langchain")
        with tracer.start_as_current_span(
            f"langchain.tool.{self.name}",
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute("tool.name", self.name)
            span.set_attribute("verse.id", verse_id)

            result = await self.agent.process(verse_id, text, context)
            return json.dumps(result.to_dict(), indent=2)
