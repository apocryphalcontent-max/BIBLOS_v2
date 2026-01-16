"""
BIBLOS v2 - Observability Package

Comprehensive distributed tracing, metrics, and logging for the BIBLOS v2 system.
Provides OpenTelemetry integration for flame graphs, latency analysis, and
production-grade monitoring.

Components:
- tracing: OpenTelemetry distributed tracing with OTLP export
- metrics: Custom BIBLOS metrics for pipeline, agents, and ML inference
- logging: Structlog integration with trace context propagation

Usage:
    from observability import setup_observability, get_tracer, get_meter

    # Initialize at application startup
    setup_observability(service_name="biblos-v2")

    # Get tracer for manual instrumentation
    tracer = get_tracer(__name__)

    # Get meter for custom metrics
    meter = get_meter(__name__)
"""
try:
    from .tracing import (
        setup_tracing,
        get_tracer,
        get_tracer_provider,
        create_span,
        span_decorator,
        TracingConfig,
        shutdown_tracing,
    )
except ImportError:
    # Tracing module not yet created - provide stubs
    setup_tracing = lambda *args, **kwargs: None
    get_tracer = lambda name: None
    get_tracer_provider = lambda: None
    create_span = lambda *args, **kwargs: None
    span_decorator = lambda *args, **kwargs: lambda fn: fn
    TracingConfig = type('TracingConfig', (), {})
    shutdown_tracing = lambda: None

try:
    from .metrics import (
        setup_metrics,
        get_meter,
        get_meter_provider,
        MetricsConfig,
        BiblosMetrics,
        record_pipeline_duration,
        record_phase_duration,
        record_agent_duration,
        record_verse_processed,
        record_crossref_discovered,
        record_ml_inference_duration,
        shutdown_metrics,
    )
except ImportError:
    # Metrics module not yet created - provide stubs
    setup_metrics = lambda *args, **kwargs: None
    get_meter = lambda name: None
    get_meter_provider = lambda: None
    MetricsConfig = type('MetricsConfig', (), {})
    BiblosMetrics = type('BiblosMetrics', (), {})
    record_pipeline_duration = lambda *args, **kwargs: None
    record_phase_duration = lambda *args, **kwargs: None
    record_agent_duration = lambda *args, **kwargs: None
    record_verse_processed = lambda *args, **kwargs: None
    record_crossref_discovered = lambda *args, **kwargs: None
    record_ml_inference_duration = lambda *args, **kwargs: None
    shutdown_metrics = lambda: None

try:
    from .logging import (
        setup_logging,
        get_logger,
        LoggingConfig,
        configure_structlog,
        shutdown_logging,
    )
except ImportError:
    # Logging module not yet created - provide stubs
    import logging as _logging
    setup_logging = lambda *args, **kwargs: None
    get_logger = _logging.getLogger
    LoggingConfig = type('LoggingConfig', (), {})
    configure_structlog = lambda *args, **kwargs: None
    shutdown_logging = lambda: None

__all__ = [
    # Tracing
    "setup_tracing",
    "get_tracer",
    "get_tracer_provider",
    "create_span",
    "span_decorator",
    "TracingConfig",
    "shutdown_tracing",
    # Metrics
    "setup_metrics",
    "get_meter",
    "get_meter_provider",
    "MetricsConfig",
    "BiblosMetrics",
    "record_pipeline_duration",
    "record_phase_duration",
    "record_agent_duration",
    "record_verse_processed",
    "record_crossref_discovered",
    "record_ml_inference_duration",
    "shutdown_metrics",
    # Logging
    "setup_logging",
    "get_logger",
    "LoggingConfig",
    "configure_structlog",
    "shutdown_logging",
    # Combined setup
    "setup_observability",
    "shutdown_observability",
]

__version__ = "1.0.0"


def setup_observability(
    service_name: str = "biblos-v2",
    otlp_endpoint: str = "http://localhost:4317",
    enabled: bool = True,
    sample_rate: float = 1.0,
    log_level: str = "INFO",
    environment: str = "development",
) -> None:
    """
    Initialize all observability components for BIBLOS v2.

    This sets up:
    - OpenTelemetry distributed tracing with OTLP export
    - Custom BIBLOS metrics (pipeline, agents, ML inference)
    - Structlog with trace context integration

    Args:
        service_name: Name of the service for telemetry
        otlp_endpoint: OTLP collector endpoint (gRPC)
        enabled: Enable/disable observability
        sample_rate: Trace sampling rate (0.0-1.0)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        environment: Deployment environment (development, staging, production)

    Example:
        >>> from observability import setup_observability
        >>> setup_observability(
        ...     service_name="biblos-v2",
        ...     otlp_endpoint="http://localhost:4317",
        ...     sample_rate=0.1  # 10% sampling in production
        ... )
    """
    if not enabled:
        return

    # Configure tracing
    tracing_config = TracingConfig(
        service_name=service_name,
        otlp_endpoint=otlp_endpoint,
        enabled=enabled,
        sample_rate=sample_rate,
        environment=environment,
    )
    setup_tracing(tracing_config)

    # Configure metrics
    metrics_config = MetricsConfig(
        service_name=service_name,
        otlp_endpoint=otlp_endpoint,
        enabled=enabled,
        environment=environment,
    )
    setup_metrics(metrics_config)

    # Configure logging with trace context
    logging_config = LoggingConfig(
        service_name=service_name,
        level=log_level,
        enable_trace_context=True,
        json_format=True,
    )
    setup_logging(logging_config)


def shutdown_observability() -> None:
    """
    Gracefully shutdown all observability components.

    Call this during application shutdown to ensure all telemetry
    data is flushed to the collector.
    """
    shutdown_tracing()
    shutdown_metrics()
    shutdown_logging()
