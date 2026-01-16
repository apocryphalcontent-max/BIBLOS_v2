"""
BIBLOS v2 - Distributed Tracing with OpenTelemetry

Provides comprehensive distributed tracing capabilities for the BIBLOS v2 system,
enabling flame graphs, latency analysis, and request correlation across all
pipeline phases, agents, and ML inference.

Features:
- OTLP export to Jaeger, Tempo, or any OTLP-compatible backend
- Auto-instrumentation for FastAPI, SQLAlchemy, Redis, aiohttp
- Manual instrumentation decorators and context managers
- Configurable sampling strategies
- Resource attributes for service identification

Usage:
    from observability.tracing import setup_tracing, get_tracer, span_decorator

    # Setup at startup
    setup_tracing(TracingConfig(service_name="biblos-v2"))

    # Manual instrumentation
    tracer = get_tracer(__name__)

    @span_decorator("process_verse")
    async def process_verse(verse_id: str):
        with tracer.start_as_current_span("inner_operation") as span:
            span.set_attribute("verse.id", verse_id)
            # ... processing logic
"""
from __future__ import annotations

import functools
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, TypeVar, ParamSpec

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.propagators.b3 import B3MultiFormat
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.sdk.trace import TracerProvider, Span
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.sdk.trace.sampling import (
    ParentBased,
    TraceIdRatioBased,
    ALWAYS_ON,
    ALWAYS_OFF,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.trace import Status, StatusCode, SpanKind

# Type variables for decorators
P = ParamSpec("P")
T = TypeVar("T")

# Global state
_tracer_provider: Optional[TracerProvider] = None
_initialized: bool = False


@dataclass
class TracingConfig:
    """Configuration for OpenTelemetry tracing."""

    service_name: str = "biblos-v2"
    service_version: str = "2.0.0"
    otlp_endpoint: str = field(
        default_factory=lambda: os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    )
    enabled: bool = field(
        default_factory=lambda: os.getenv("OTEL_TRACING_ENABLED", "true").lower() == "true"
    )
    sample_rate: float = field(
        default_factory=lambda: float(os.getenv("OTEL_SAMPLE_RATE", "1.0"))
    )
    environment: str = field(
        default_factory=lambda: os.getenv("ENVIRONMENT", "development")
    )
    console_export: bool = field(
        default_factory=lambda: os.getenv("OTEL_CONSOLE_EXPORT", "false").lower() == "true"
    )
    batch_export: bool = True
    max_queue_size: int = 2048
    schedule_delay_millis: int = 5000
    max_export_batch_size: int = 512
    export_timeout_millis: int = 30000

    # Additional resource attributes
    extra_attributes: Dict[str, str] = field(default_factory=dict)


def setup_tracing(config: Optional[TracingConfig] = None) -> TracerProvider:
    """
    Configure OpenTelemetry tracing with OTLP export.

    Args:
        config: Tracing configuration. Uses defaults if not provided.

    Returns:
        Configured TracerProvider

    Example:
        >>> config = TracingConfig(
        ...     service_name="biblos-v2",
        ...     otlp_endpoint="http://jaeger:4317",
        ...     sample_rate=0.1  # 10% sampling in production
        ... )
        >>> provider = setup_tracing(config)
    """
    global _tracer_provider, _initialized

    if _initialized and _tracer_provider:
        return _tracer_provider

    config = config or TracingConfig()

    if not config.enabled:
        # Set a no-op provider
        trace.set_tracer_provider(trace.NoOpTracerProvider())
        _initialized = True
        return trace.get_tracer_provider()

    # Build resource with service information
    resource_attributes = {
        SERVICE_NAME: config.service_name,
        SERVICE_VERSION: config.service_version,
        "deployment.environment": config.environment,
        "telemetry.sdk.language": "python",
        "telemetry.sdk.name": "opentelemetry",
        "service.namespace": "biblos",
        **config.extra_attributes,
    }
    resource = Resource.create(resource_attributes)

    # Configure sampler
    if config.sample_rate <= 0.0:
        sampler = ALWAYS_OFF
    elif config.sample_rate >= 1.0:
        sampler = ALWAYS_ON
    else:
        sampler = ParentBased(root=TraceIdRatioBased(config.sample_rate))

    # Create tracer provider
    _tracer_provider = TracerProvider(resource=resource, sampler=sampler)

    # Configure OTLP exporter
    try:
        otlp_exporter = OTLPSpanExporter(
            endpoint=config.otlp_endpoint,
            insecure=True,  # Use insecure for local development
        )

        if config.batch_export:
            processor = BatchSpanProcessor(
                otlp_exporter,
                max_queue_size=config.max_queue_size,
                schedule_delay_millis=config.schedule_delay_millis,
                max_export_batch_size=config.max_export_batch_size,
                export_timeout_millis=config.export_timeout_millis,
            )
        else:
            processor = SimpleSpanProcessor(otlp_exporter)

        _tracer_provider.add_span_processor(processor)
    except Exception as e:
        print(f"Warning: Failed to initialize OTLP exporter: {e}")

    # Optional console export for debugging
    if config.console_export:
        console_exporter = ConsoleSpanExporter()
        _tracer_provider.add_span_processor(SimpleSpanProcessor(console_exporter))

    # Set as global tracer provider
    trace.set_tracer_provider(_tracer_provider)

    # Configure propagators (W3C TraceContext + B3 for compatibility)
    propagator = CompositePropagator([TraceContextTextMapPropagator(), B3MultiFormat()])
    set_global_textmap(propagator)

    _initialized = True

    return _tracer_provider


def get_tracer_provider() -> TracerProvider:
    """Get the global tracer provider, initializing if necessary."""
    global _tracer_provider, _initialized
    if not _initialized:
        setup_tracing()
    return _tracer_provider or trace.get_tracer_provider()


def get_tracer(name: str, version: str = "1.0.0") -> trace.Tracer:
    """
    Get a tracer instance for manual instrumentation.

    Args:
        name: Tracer name, typically __name__ of the module
        version: Tracer version string

    Returns:
        Tracer instance for creating spans

    Example:
        >>> tracer = get_tracer(__name__)
        >>> with tracer.start_as_current_span("my_operation") as span:
        ...     span.set_attribute("key", "value")
        ...     # ... operation logic
    """
    provider = get_tracer_provider()
    return provider.get_tracer(name, version)


def shutdown_tracing() -> None:
    """
    Gracefully shutdown tracing, flushing any pending spans.

    Call this during application shutdown.
    """
    global _tracer_provider, _initialized
    if _tracer_provider and hasattr(_tracer_provider, "shutdown"):
        _tracer_provider.shutdown()
    _initialized = False
    _tracer_provider = None


@contextmanager
def create_span(
    name: str,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Optional[Dict[str, Any]] = None,
    tracer_name: str = "biblos.observability",
):
    """
    Context manager for creating spans with automatic error handling.

    Args:
        name: Span name
        kind: Span kind (INTERNAL, SERVER, CLIENT, PRODUCER, CONSUMER)
        attributes: Initial span attributes
        tracer_name: Name of the tracer to use

    Yields:
        Active Span instance

    Example:
        >>> with create_span("process_verse", attributes={"verse.id": "GEN.1.1"}) as span:
        ...     result = await process(verse_id)
        ...     span.set_attribute("result.confidence", result.confidence)
    """
    tracer = get_tracer(tracer_name)
    with tracer.start_as_current_span(name, kind=kind) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


def span_decorator(
    name: Optional[str] = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Optional[Dict[str, Any]] = None,
    record_args: bool = False,
    record_result: bool = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for automatic span creation around functions.

    Args:
        name: Span name (defaults to function name)
        kind: Span kind
        attributes: Static attributes to add to all spans
        record_args: Whether to record function arguments as attributes
        record_result: Whether to record return value as attribute

    Returns:
        Decorated function

    Example:
        >>> @span_decorator("process_verse", record_args=True)
        ... async def process_verse(verse_id: str, text: str):
        ...     # ... processing logic
        ...     return result

        >>> @span_decorator(attributes={"component": "ml.inference"})
        ... async def run_inference(text: str):
        ...     # ... ML inference
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        span_name = name or func.__name__
        tracer = get_tracer(func.__module__)

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            with tracer.start_as_current_span(span_name, kind=kind) as span:
                # Add static attributes
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)

                # Record arguments
                if record_args:
                    _record_function_args(span, func, args, kwargs)

                try:
                    result = await func(*args, **kwargs)

                    # Record result
                    if record_result and result is not None:
                        _record_result(span, result)

                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            with tracer.start_as_current_span(span_name, kind=kind) as span:
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)

                if record_args:
                    _record_function_args(span, func, args, kwargs)

                try:
                    result = func(*args, **kwargs)

                    if record_result and result is not None:
                        _record_result(span, result)

                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        # Return appropriate wrapper based on function type
        if _is_async_function(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def _is_async_function(func: Callable) -> bool:
    """Check if a function is async."""
    import asyncio
    return asyncio.iscoroutinefunction(func)


def _record_function_args(span: Span, func: Callable, args: tuple, kwargs: dict) -> None:
    """Record function arguments as span attributes."""
    import inspect

    sig = inspect.signature(func)
    params = list(sig.parameters.keys())

    # Record positional args
    for i, (param, value) in enumerate(zip(params, args)):
        if i < len(params):
            _set_safe_attribute(span, f"arg.{param}", value)

    # Record keyword args
    for key, value in kwargs.items():
        _set_safe_attribute(span, f"arg.{key}", value)


def _record_result(span: Span, result: Any) -> None:
    """Record function result as span attribute."""
    if hasattr(result, "to_dict"):
        try:
            result_dict = result.to_dict()
            for key, value in result_dict.items():
                if key not in ["data", "embeddings"]:  # Skip large fields
                    _set_safe_attribute(span, f"result.{key}", value)
        except Exception:
            pass
    elif isinstance(result, (str, int, float, bool)):
        span.set_attribute("result.value", result)


def _set_safe_attribute(span: Span, key: str, value: Any) -> None:
    """Set span attribute with type coercion for safety."""
    if value is None:
        return

    if isinstance(value, (str, int, float, bool)):
        span.set_attribute(key, value)
    elif isinstance(value, (list, tuple)):
        if len(value) <= 10:  # Limit array size
            try:
                # Convert to string representation for complex types
                str_values = [str(v)[:100] for v in value]
                span.set_attribute(key, str_values)
            except Exception:
                span.set_attribute(key, str(value)[:500])
    elif isinstance(value, dict):
        span.set_attribute(key, str(value)[:500])
    else:
        span.set_attribute(key, str(value)[:200])


# Auto-instrumentation helpers
def instrument_fastapi(app) -> None:
    """
    Auto-instrument a FastAPI application.

    Args:
        app: FastAPI application instance

    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> instrument_fastapi(app)
    """
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(
            app,
            excluded_urls="health,metrics,ready",
            tracer_provider=get_tracer_provider(),
        )
    except ImportError:
        print("Warning: opentelemetry-instrumentation-fastapi not installed")


def instrument_sqlalchemy(engine) -> None:
    """
    Auto-instrument SQLAlchemy for database tracing.

    Args:
        engine: SQLAlchemy engine instance
    """
    try:
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

        SQLAlchemyInstrumentor().instrument(
            engine=engine,
            tracer_provider=get_tracer_provider(),
        )
    except ImportError:
        print("Warning: opentelemetry-instrumentation-sqlalchemy not installed")


def instrument_redis(client=None) -> None:
    """
    Auto-instrument Redis for cache tracing.

    Args:
        client: Optional specific Redis client to instrument
    """
    try:
        from opentelemetry.instrumentation.redis import RedisInstrumentor

        RedisInstrumentor().instrument(tracer_provider=get_tracer_provider())
    except ImportError:
        print("Warning: opentelemetry-instrumentation-redis not installed")


def instrument_aiohttp() -> None:
    """Auto-instrument aiohttp for HTTP client tracing."""
    try:
        from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor

        AioHttpClientInstrumentor().instrument(tracer_provider=get_tracer_provider())
    except ImportError:
        print("Warning: opentelemetry-instrumentation-aiohttp-client not installed")


def instrument_httpx() -> None:
    """Auto-instrument httpx for HTTP client tracing."""
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().instrument(tracer_provider=get_tracer_provider())
    except ImportError:
        print("Warning: opentelemetry-instrumentation-httpx not installed")


def instrument_all(
    app=None,
    sqlalchemy_engine=None,
    include_redis: bool = True,
    include_http: bool = True,
) -> None:
    """
    Instrument all available integrations.

    Args:
        app: Optional FastAPI app to instrument
        sqlalchemy_engine: Optional SQLAlchemy engine to instrument
        include_redis: Whether to instrument Redis
        include_http: Whether to instrument HTTP clients
    """
    if app:
        instrument_fastapi(app)

    if sqlalchemy_engine:
        instrument_sqlalchemy(sqlalchemy_engine)

    if include_redis:
        instrument_redis()

    if include_http:
        instrument_aiohttp()
        instrument_httpx()


# BIBLOS-specific span helpers
def start_pipeline_span(
    verse_id: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Span:
    """
    Start a span for pipeline execution.

    Args:
        verse_id: Verse being processed
        metadata: Additional pipeline metadata

    Returns:
        Active span for the pipeline
    """
    tracer = get_tracer("biblos.pipeline")
    span = tracer.start_span(
        "pipeline.execute",
        kind=SpanKind.INTERNAL,
    )
    span.set_attribute("verse.id", verse_id)
    span.set_attribute("component", "pipeline")

    if metadata:
        for key, value in metadata.items():
            _set_safe_attribute(span, f"pipeline.{key}", value)

    return span


def start_phase_span(
    phase_name: str,
    verse_id: str,
    agent_count: int,
) -> Span:
    """
    Start a span for a pipeline phase.

    Args:
        phase_name: Name of the phase (linguistic, theological, etc.)
        verse_id: Verse being processed
        agent_count: Number of agents in the phase

    Returns:
        Active span for the phase
    """
    tracer = get_tracer("biblos.pipeline.phase")
    span = tracer.start_span(
        f"phase.{phase_name}",
        kind=SpanKind.INTERNAL,
    )
    span.set_attribute("phase.name", phase_name)
    span.set_attribute("verse.id", verse_id)
    span.set_attribute("phase.agent_count", agent_count)
    span.set_attribute("component", "phase")

    return span


def start_agent_span(
    agent_name: str,
    extraction_type: str,
    verse_id: str,
) -> Span:
    """
    Start a span for an agent extraction.

    Args:
        agent_name: Name of the agent
        extraction_type: Type of extraction being performed
        verse_id: Verse being processed

    Returns:
        Active span for the agent
    """
    tracer = get_tracer("biblos.agents")
    span = tracer.start_span(
        f"agent.{agent_name}.extract",
        kind=SpanKind.INTERNAL,
    )
    span.set_attribute("agent.name", agent_name)
    span.set_attribute("agent.extraction_type", extraction_type)
    span.set_attribute("verse.id", verse_id)
    span.set_attribute("component", "agent")

    return span


def start_ml_span(
    operation: str,
    model_name: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> Span:
    """
    Start a span for ML inference.

    Args:
        operation: ML operation (embed, infer, classify)
        model_name: Name of the model being used
        batch_size: Batch size for the operation

    Returns:
        Active span for ML inference
    """
    tracer = get_tracer("biblos.ml")
    span = tracer.start_span(
        f"ml.{operation}",
        kind=SpanKind.INTERNAL,
    )
    span.set_attribute("ml.operation", operation)
    span.set_attribute("component", "ml")

    if model_name:
        span.set_attribute("ml.model", model_name)
    if batch_size:
        span.set_attribute("ml.batch_size", batch_size)

    return span
