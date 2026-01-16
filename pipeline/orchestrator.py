"""
BIBLOS v2 - Pipeline Orchestrator

Coordinates execution of all pipeline phases with comprehensive
OpenTelemetry distributed tracing for flame graph visualization.
"""
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone

from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from pipeline.base import (
    BasePipelinePhase,
    PhaseResult,
    PhaseStatus,
    PipelineContext
)
from pipeline.linguistic import LinguisticPhase
from pipeline.theological import TheologicalPhase
from pipeline.intertextual import IntertextualPhase
from pipeline.validation import ValidationPhase
from pipeline.finalization import FinalizationPhase

# Import core error types for specific exception handling
from core.errors import (
    BiblosError,
    BiblosPipelineError,
    BiblosAgentError,
    BiblosValidationError,
    BiblosTimeoutError,
    BiblosResourceError,
    ErrorSeverity,
)

# Import observability
from observability import get_tracer, get_logger
from observability.metrics import (
    record_pipeline_duration,
    record_phase_duration,
    timed_pipeline,
    timed_phase,
)
from observability.logging import PipelineLogger

# Get module-level tracer and logger
tracer = get_tracer(__name__)
logger = get_logger(__name__)
pipeline_logger = PipelineLogger()


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""
    phases: List[str] = field(default_factory=lambda: [
        "linguistic", "theological", "intertextual", "validation", "finalization"
    ])
    parallel_phases: bool = False  # Run independent phases in parallel
    fail_fast: bool = False  # Stop on first failure
    timeout_seconds: int = 600
    min_confidence: float = 0.5


@dataclass
class PipelineResult:
    """Result of complete pipeline execution."""
    verse_id: str
    status: str
    phase_results: Dict[str, PhaseResult]
    golden_record: Optional[Dict[str, Any]]
    start_time: float
    end_time: float
    errors: List[str]
    trace_id: Optional[str] = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verse_id": self.verse_id,
            "status": self.status,
            "duration": self.duration,
            "phases": {
                name: result.to_dict()
                for name, result in self.phase_results.items()
            },
            "golden_record": self.golden_record,
            "errors": self.errors,
            "trace_id": self.trace_id
        }


def get_current_trace_id() -> Optional[str]:
    """Get current trace ID as hex string."""
    span = trace.get_current_span()
    if span and span.is_recording():
        ctx = span.get_span_context()
        if ctx.is_valid:
            return format(ctx.trace_id, "032x")
    return None


class PipelineOrchestrator:
    """
    Orchestrates the complete SDES extraction pipeline.

    Coordinates all phases with comprehensive distributed tracing:
    1. Linguistic - Text analysis foundation
    2. Theological - Theological interpretation
    3. Intertextual - Cross-reference analysis
    4. Validation - Quality assurance
    5. Finalization - Golden record creation

    Each phase and agent execution creates child spans, enabling
    flame graph visualization in Jaeger/Tempo.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._phases: Dict[str, BasePipelinePhase] = {}
        self._initialized = False

    # -------------------------------------------------------------------------
    # Error Handling Helpers (Single Responsibility)
    # -------------------------------------------------------------------------

    def _create_failed_phase_result(
        self,
        phase_name: str,
        error: str,
    ) -> PhaseResult:
        """
        Factory method for creating failed PhaseResult.

        Consolidates the repeated PhaseResult creation pattern
        in exception handlers.

        Args:
            phase_name: Name of the failed phase
            error: Error message

        Returns:
            PhaseResult configured for failure
        """
        now = datetime.now(timezone.utc).timestamp()
        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.FAILED,
            agent_results={},
            start_time=now,
            end_time=now,
            error=error,
        )

    def _create_error_pipeline_result(
        self,
        verse_id: str,
        status: str,
        error_message: str,
        phase_results: Dict[str, PhaseResult],
        start_time: float,
    ) -> PipelineResult:
        """
        Factory method for creating error PipelineResult.

        Consolidates the repeated PipelineResult creation pattern
        in exception handlers.

        Args:
            verse_id: Verse being processed
            status: Error status string
            error_message: Description of the error
            phase_results: Results from completed phases
            start_time: Pipeline start timestamp

        Returns:
            PipelineResult configured for error
        """
        end_time = datetime.now(timezone.utc).timestamp()
        return PipelineResult(
            verse_id=verse_id,
            status=status,
            phase_results=phase_results,
            golden_record=None,
            start_time=start_time,
            end_time=end_time,
            errors=[error_message],
            trace_id=get_current_trace_id(),
        )

    def _set_pipeline_error_span(
        self,
        span: Any,
        status: str,
        error_type: str,
        error_message: str,
        duration: float,
        error_code: Optional[str] = None,
        exception: Optional[Exception] = None,
    ) -> None:
        """
        Set standard error attributes on a pipeline span.

        Consolidates the repeated span configuration pattern
        in exception handlers.

        Args:
            span: OpenTelemetry span
            status: Pipeline status string
            error_type: Type of error for categorization
            error_message: Error description
            duration: Time elapsed
            error_code: Optional error code
            exception: Optional exception to record
        """
        span.set_status(Status(StatusCode.ERROR, error_message))
        span.set_attribute("pipeline.status", status)
        span.set_attribute("pipeline.duration_seconds", duration)
        span.set_attribute("error.type", error_type)
        if error_code:
            span.set_attribute("error.code", error_code)
        if exception:
            span.record_exception(exception)

    def _set_phase_error_span(
        self,
        span: Any,
        error_type: str,
        error_message: str,
        error_code: Optional[str] = None,
        exception: Optional[Exception] = None,
    ) -> None:
        """
        Set standard error attributes on a phase span.

        Args:
            span: OpenTelemetry span
            error_type: Type of error for categorization
            error_message: Error description
            error_code: Optional error code
            exception: Optional exception to record
        """
        span.set_status(Status(StatusCode.ERROR, error_message))
        span.set_attribute("error.type", error_type)
        if error_code:
            span.set_attribute("error.code", error_code)
        if exception:
            span.record_exception(exception)

    async def initialize(self) -> None:
        """Initialize all pipeline phases with tracing."""
        with tracer.start_as_current_span(
            "pipeline.initialize",
            kind=SpanKind.INTERNAL,
        ) as span:
            logger.info("Initializing pipeline orchestrator")
            span.set_attribute("pipeline.phase_count", len(self.config.phases))

            phase_classes = {
                "linguistic": LinguisticPhase,
                "theological": TheologicalPhase,
                "intertextual": IntertextualPhase,
                "validation": ValidationPhase,
                "finalization": FinalizationPhase
            }

            for phase_name in self.config.phases:
                if phase_name in phase_classes:
                    with tracer.start_as_current_span(
                        f"pipeline.initialize.{phase_name}"
                    ) as phase_span:
                        try:
                            phase = phase_classes[phase_name]()
                            await phase.initialize()
                            self._phases[phase_name] = phase
                            phase_span.set_attribute("status", "success")
                            logger.info(
                                f"Initialized phase: {phase_name}",
                                phase=phase_name,
                            )
                        except asyncio.TimeoutError as e:
                            # Handle initialization timeout
                            phase_span.set_status(Status(StatusCode.ERROR, f"Timeout initializing phase"))
                            phase_span.set_attribute("error.type", "timeout")
                            logger.error(
                                f"Timeout initializing phase: {phase_name}",
                                phase=phase_name,
                            )
                        except BiblosPipelineError as e:
                            # Handle pipeline-specific errors
                            phase_span.set_status(Status(StatusCode.ERROR, str(e)))
                            phase_span.record_exception(e)
                            phase_span.set_attribute("error.type", "pipeline")
                            logger.error(
                                f"Pipeline error initializing phase: {phase_name}",
                                phase=phase_name,
                                error=str(e),
                                error_code=e.error_code if hasattr(e, 'error_code') else None,
                            )
                        except BiblosResourceError as e:
                            # Handle resource exhaustion (memory, connections)
                            phase_span.set_status(Status(StatusCode.ERROR, str(e)))
                            phase_span.record_exception(e)
                            phase_span.set_attribute("error.type", "resource")
                            logger.error(
                                f"Resource error initializing phase: {phase_name}",
                                phase=phase_name,
                                error=str(e),
                            )
                        except BiblosError as e:
                            # Handle other BIBLOS-specific errors
                            phase_span.set_status(Status(StatusCode.ERROR, str(e)))
                            phase_span.record_exception(e)
                            phase_span.set_attribute("error.type", "biblos")
                            logger.error(
                                f"BIBLOS error initializing phase: {phase_name}",
                                phase=phase_name,
                                error=str(e),
                            )
                        except Exception as e:
                            # Catch-all for unexpected errors - should be rare
                            phase_span.set_status(Status(StatusCode.ERROR, str(e)))
                            phase_span.record_exception(e)
                            phase_span.set_attribute("error.type", "unexpected")
                            logger.error(
                                f"Unexpected error initializing phase: {phase_name}",
                                phase=phase_name,
                                error=str(e),
                                error_type=type(e).__name__,
                            )

            self._initialized = True
            span.set_attribute("pipeline.initialized_phases", len(self._phases))
            logger.info(
                f"Pipeline initialized with {len(self._phases)} phases",
                phase_count=len(self._phases),
            )

    async def execute(
        self,
        verse_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PipelineResult:
        """
        Execute the complete pipeline on a verse with full tracing.

        Creates a parent span for the entire pipeline, with child spans
        for each phase. Enables flame graph visualization showing:
        - Total pipeline duration
        - Time spent in each phase
        - Time spent in each agent (nested under phases)
        """
        if not self._initialized:
            await self.initialize()

        # Start parent span for entire pipeline
        with tracer.start_as_current_span(
            "pipeline.execute",
            kind=SpanKind.INTERNAL,
        ) as pipeline_span:
            # Set span attributes
            pipeline_span.set_attribute("verse.id", verse_id)
            pipeline_span.set_attribute("verse.book", verse_id.split(".")[0] if "." in verse_id else verse_id)
            pipeline_span.set_attribute("pipeline.phase_count", len(self.config.phases))
            pipeline_span.set_attribute("pipeline.parallel", self.config.parallel_phases)
            pipeline_span.set_attribute("pipeline.fail_fast", self.config.fail_fast)

            start_time = datetime.now(timezone.utc).timestamp()
            context = PipelineContext()
            context.verse_id = verse_id
            context.text = text
            context.metadata = metadata or {}

            pipeline_logger.start_pipeline(verse_id, len(self.config.phases))

            try:
                # Execute phases in order
                for phase_name in self.config.phases:
                    if phase_name not in self._phases:
                        continue

                    phase = self._phases[phase_name]
                    agent_count = len(getattr(phase.config, 'agents', []))

                    # Create child span for each phase
                    with tracer.start_as_current_span(
                        f"phase.{phase_name}",
                        kind=SpanKind.INTERNAL,
                    ) as phase_span:
                        phase_span.set_attribute("phase.name", phase_name)
                        phase_span.set_attribute("verse.id", verse_id)
                        phase_span.set_attribute("phase.agent_count", agent_count)

                        pipeline_logger.start_phase(phase_name, verse_id, agent_count)
                        phase_start = datetime.now(timezone.utc).timestamp()

                        # Execute phase with timeout
                        result = await self._execute_phase_with_timeout(
                            phase, verse_id, text, context.to_dict()
                        )

                        phase_duration = datetime.now(timezone.utc).timestamp() - phase_start

                        # Record phase metrics
                        phase_status = "completed" if result.status == PhaseStatus.COMPLETED else "failed"
                        record_phase_duration(
                            phase_name,
                            verse_id,
                            phase_duration,
                            phase_status,
                            agent_count
                        )

                        # Set phase span attributes
                        phase_span.set_attribute("phase.status", result.status.value)
                        phase_span.set_attribute("phase.duration_seconds", phase_duration)
                        phase_span.set_attribute("phase.agent_results_count", len(result.agent_results))

                        if result.status == PhaseStatus.FAILED:
                            phase_span.set_status(Status(StatusCode.ERROR, result.error or "Phase failed"))
                            if result.error:
                                phase_span.set_attribute("error.message", result.error)
                        else:
                            phase_span.set_status(Status(StatusCode.OK))

                        pipeline_logger.end_phase(phase_name, verse_id, phase_status, phase_duration)

                        # Update context
                        context.update_from_result(result)
                        context.phase_results[phase_name] = result

                        # Check for failure
                        if result.status == PhaseStatus.FAILED:
                            if self.config.fail_fast:
                                logger.error(
                                    f"Phase {phase_name} failed, stopping pipeline",
                                    phase=phase_name,
                                    verse_id=verse_id,
                                )
                                break
                            else:
                                logger.warning(
                                    f"Phase {phase_name} failed, continuing",
                                    phase=phase_name,
                                    verse_id=verse_id,
                                )

                # Get golden record
                golden_record = None
                if "finalization" in context.phase_results:
                    finalization = context.phase_results["finalization"]
                    if finalization.status == PhaseStatus.COMPLETED:
                        golden_record = finalization.agent_results.get("golden_record")

                # Determine overall status
                failed_phases = [
                    name for name, result in context.phase_results.items()
                    if result.status == PhaseStatus.FAILED
                ]

                if not failed_phases:
                    status = "completed"
                    pipeline_span.set_status(Status(StatusCode.OK))
                elif len(failed_phases) == len(context.phase_results):
                    status = "failed"
                    pipeline_span.set_status(Status(StatusCode.ERROR, "All phases failed"))
                else:
                    status = "partial"
                    pipeline_span.set_status(Status(StatusCode.OK))  # Partial success is OK

                end_time = datetime.now(timezone.utc).timestamp()
                duration = end_time - start_time

                # Record pipeline metrics
                record_pipeline_duration(verse_id, duration, status, len(self.config.phases))

                # Set final span attributes
                pipeline_span.set_attribute("pipeline.status", status)
                pipeline_span.set_attribute("pipeline.duration_seconds", duration)
                pipeline_span.set_attribute("pipeline.failed_phases", len(failed_phases))
                pipeline_span.set_attribute("pipeline.has_golden_record", golden_record is not None)

                pipeline_logger.end_pipeline(verse_id, status, duration)

                return PipelineResult(
                    verse_id=verse_id,
                    status=status,
                    phase_results=context.phase_results,
                    golden_record=golden_record,
                    start_time=start_time,
                    end_time=end_time,
                    errors=context.errors,
                    trace_id=get_current_trace_id()
                )

            except asyncio.TimeoutError:
                # Handle overall pipeline timeout
                duration = datetime.now(timezone.utc).timestamp() - start_time
                error_msg = "Pipeline execution timed out"

                self._set_pipeline_error_span(
                    pipeline_span, "timeout", "timeout", error_msg, duration
                )
                logger.error(
                    f"Pipeline execution timed out for {verse_id}",
                    verse_id=verse_id,
                    duration=duration,
                )

                record_pipeline_duration(verse_id, duration, "timeout")
                pipeline_logger.end_pipeline(verse_id, "timeout", duration, error_msg)

                return self._create_error_pipeline_result(
                    verse_id, "timeout", error_msg, context.phase_results, start_time
                )

            except BiblosPipelineError as e:
                # Handle pipeline-specific errors
                duration = datetime.now(timezone.utc).timestamp() - start_time

                self._set_pipeline_error_span(
                    pipeline_span, "pipeline_error", "pipeline", str(e), duration,
                    e.error_code, e
                )
                logger.error(
                    f"Pipeline error: {e}",
                    verse_id=verse_id,
                    error=str(e),
                    error_code=e.error_code,
                )

                record_pipeline_duration(verse_id, duration, "pipeline_error")
                pipeline_logger.end_pipeline(verse_id, "pipeline_error", duration, str(e))

                return self._create_error_pipeline_result(
                    verse_id, "pipeline_error", str(e), context.phase_results, start_time
                )

            except BiblosAgentError as e:
                # Handle agent-specific errors
                duration = datetime.now(timezone.utc).timestamp() - start_time

                self._set_pipeline_error_span(
                    pipeline_span, "agent_error", "agent", str(e), duration,
                    exception=e
                )
                logger.error(
                    f"Agent error in pipeline: {e}",
                    verse_id=verse_id,
                    error=str(e),
                )

                record_pipeline_duration(verse_id, duration, "agent_error")
                pipeline_logger.end_pipeline(verse_id, "agent_error", duration, str(e))

                return self._create_error_pipeline_result(
                    verse_id, "agent_error", str(e), context.phase_results, start_time
                )

            except (MemoryError, BiblosResourceError) as e:
                # Handle resource exhaustion
                duration = datetime.now(timezone.utc).timestamp() - start_time
                error_msg = f"Resource exhaustion: {e}"

                self._set_pipeline_error_span(
                    pipeline_span, "resource_error", "resource", str(e), duration,
                    exception=e
                )
                logger.critical(
                    f"Resource exhaustion during pipeline: {e}",
                    verse_id=verse_id,
                    error=str(e),
                    error_type=type(e).__name__,
                )

                record_pipeline_duration(verse_id, duration, "resource_error")
                pipeline_logger.end_pipeline(verse_id, "resource_error", duration, str(e))

                return self._create_error_pipeline_result(
                    verse_id, "resource_error", error_msg, context.phase_results, start_time
                )

            except BiblosError as e:
                # Handle other BIBLOS-specific errors
                duration = datetime.now(timezone.utc).timestamp() - start_time

                self._set_pipeline_error_span(
                    pipeline_span, "biblos_error", "biblos", str(e), duration,
                    exception=e
                )
                logger.error(
                    f"BIBLOS error during pipeline: {e}",
                    verse_id=verse_id,
                    error=str(e),
                )

                record_pipeline_duration(verse_id, duration, "biblos_error")
                pipeline_logger.end_pipeline(verse_id, "biblos_error", duration, str(e))

                return self._create_error_pipeline_result(
                    verse_id, "biblos_error", str(e), context.phase_results, start_time
                )

            except Exception as e:
                # Catch-all for truly unexpected errors
                duration = datetime.now(timezone.utc).timestamp() - start_time

                self._set_pipeline_error_span(
                    pipeline_span, "error", "unexpected", str(e), duration,
                    exception=e
                )
                logger.error(
                    f"Unexpected error during pipeline execution: {e}",
                    verse_id=verse_id,
                    error=str(e),
                    error_type=type(e).__name__,
                )

                record_pipeline_duration(verse_id, duration, "error")
                pipeline_logger.end_pipeline(verse_id, "error", duration, str(e))

                return self._create_error_pipeline_result(
                    verse_id, "error", str(e), context.phase_results, start_time
                )

    async def _execute_phase_with_timeout(
        self,
        phase: BasePipelinePhase,
        verse_id: str,
        text: str,
        context: Dict[str, Any]
    ) -> PhaseResult:
        """Execute phase with timeout and tracing."""
        with tracer.start_as_current_span(
            f"phase.{phase.config.name}.execute",
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute("phase.name", phase.config.name)
            span.set_attribute("phase.timeout", phase.config.timeout_seconds)

            try:
                result = await asyncio.wait_for(
                    phase.execute(verse_id, text, context),
                    timeout=phase.config.timeout_seconds
                )
                span.set_attribute("phase.result_status", result.status.value)
                return result

            except asyncio.TimeoutError:
                error_msg = f"Timeout after {phase.config.timeout_seconds}s"
                self._set_phase_error_span(span, "timeout", error_msg)
                logger.error(
                    f"Phase {phase.config.name} timed out",
                    phase=phase.config.name,
                    timeout_seconds=phase.config.timeout_seconds,
                )
                return self._create_failed_phase_result(phase.config.name, error_msg)

            except BiblosPipelineError as e:
                self._set_phase_error_span(span, "pipeline", str(e), e.error_code, e)
                logger.error(
                    f"Pipeline error in phase {phase.config.name}",
                    phase=phase.config.name,
                    error=str(e),
                    error_code=e.error_code,
                )
                return self._create_failed_phase_result(phase.config.name, str(e))

            except BiblosAgentError as e:
                self._set_phase_error_span(span, "agent", str(e), exception=e)
                logger.error(
                    f"Agent error in phase {phase.config.name}",
                    phase=phase.config.name,
                    error=str(e),
                )
                return self._create_failed_phase_result(phase.config.name, str(e))

            except BiblosValidationError as e:
                self._set_phase_error_span(span, "validation", str(e), exception=e)
                logger.warning(
                    f"Validation error in phase {phase.config.name}",
                    phase=phase.config.name,
                    error=str(e),
                )
                return self._create_failed_phase_result(
                    phase.config.name, f"Validation error: {e}"
                )

            except (MemoryError, BiblosResourceError) as e:
                error_msg = str(e)
                self._set_phase_error_span(span, "resource", error_msg, exception=e)
                logger.critical(
                    f"Resource exhaustion in phase {phase.config.name}",
                    phase=phase.config.name,
                    error=error_msg,
                    error_type=type(e).__name__,
                )
                return self._create_failed_phase_result(
                    phase.config.name, f"Resource exhaustion: {e}"
                )

            except BiblosError as e:
                self._set_phase_error_span(span, "biblos", str(e), exception=e)
                logger.error(
                    f"BIBLOS error in phase {phase.config.name}",
                    phase=phase.config.name,
                    error=str(e),
                )
                return self._create_failed_phase_result(phase.config.name, str(e))

            except Exception as e:
                # Catch-all for truly unexpected errors
                self._set_phase_error_span(span, "unexpected", str(e), exception=e)
                logger.error(
                    f"Unexpected error in phase {phase.config.name}",
                    phase=phase.config.name,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                return self._create_failed_phase_result(phase.config.name, str(e))

    async def execute_batch(
        self,
        verses: List[Dict[str, Any]],
        parallel: int = 4
    ) -> List[PipelineResult]:
        """Execute pipeline on a batch of verses with tracing."""
        with tracer.start_as_current_span(
            "pipeline.execute_batch",
            kind=SpanKind.INTERNAL,
        ) as batch_span:
            batch_span.set_attribute("batch.size", len(verses))
            batch_span.set_attribute("batch.parallelism", parallel)

            logger.info(
                f"Starting batch execution of {len(verses)} verses",
                verse_count=len(verses),
                parallelism=parallel,
            )

            results = []

            # Process in chunks
            for i in range(0, len(verses), parallel):
                chunk = verses[i:i + parallel]
                chunk_index = i // parallel

                with tracer.start_as_current_span(
                    f"batch.chunk_{chunk_index}",
                    kind=SpanKind.INTERNAL,
                ) as chunk_span:
                    chunk_span.set_attribute("chunk.index", chunk_index)
                    chunk_span.set_attribute("chunk.size", len(chunk))

                    tasks = [
                        self.execute(
                            verse["verse_id"],
                            verse["text"],
                            verse.get("metadata")
                        )
                        for verse in chunk
                    ]

                    chunk_results = await asyncio.gather(*tasks, return_exceptions=True)

                    for j, result in enumerate(chunk_results):
                        if isinstance(result, Exception):
                            logger.error(
                                f"Verse processing failed",
                                verse_id=chunk[j]["verse_id"],
                                error=str(result),
                            )
                            results.append(PipelineResult(
                                verse_id=chunk[j]["verse_id"],
                                status="error",
                                phase_results={},
                                golden_record=None,
                                start_time=0,
                                end_time=0,
                                errors=[str(result)],
                                trace_id=get_current_trace_id()
                            ))
                        else:
                            results.append(result)

                    logger.info(
                        f"Processed {i + len(chunk)}/{len(verses)} verses",
                        processed=i + len(chunk),
                        total=len(verses),
                    )

            # Set batch completion attributes
            successful = sum(1 for r in results if r.status == "completed")
            failed = sum(1 for r in results if r.status in ["failed", "error"])

            batch_span.set_attribute("batch.successful", successful)
            batch_span.set_attribute("batch.failed", failed)
            batch_span.set_attribute("batch.partial", len(results) - successful - failed)

            return results

    async def cleanup(self) -> None:
        """Cleanup all phases with tracing."""
        with tracer.start_as_current_span(
            "pipeline.cleanup",
            kind=SpanKind.INTERNAL,
        ) as span:
            for phase_name, phase in self._phases.items():
                with tracer.start_as_current_span(
                    f"pipeline.cleanup.{phase_name}"
                ):
                    await phase.cleanup()
                    logger.info(f"Cleaned up phase: {phase_name}", phase=phase_name)

            self._phases.clear()
            self._initialized = False
            span.set_attribute("cleanup.complete", True)
            logger.info("Pipeline orchestrator cleaned up")

    def get_phase_status(self) -> Dict[str, str]:
        """Get status of all phases."""
        return {
            name: "initialized" if phase else "not_initialized"
            for name, phase in self._phases.items()
        }
