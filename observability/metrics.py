"""
BIBLOS v2 - OpenTelemetry Metrics

Custom metrics for monitoring BIBLOS v2 pipeline performance, agent execution,
and ML inference. Provides histograms, counters, and gauges for comprehensive
observability.

Key Metrics:
- biblos_pipeline_duration_seconds: Total pipeline execution time
- biblos_phase_duration_seconds: Duration by phase (linguistic, theological, etc.)
- biblos_agent_duration_seconds: Duration by agent
- biblos_verses_processed_total: Counter of processed verses
- biblos_cross_references_discovered_total: Counter of discovered cross-references
- biblos_ml_inference_duration_seconds: ML inference latency
- biblos_embedding_generation_duration_seconds: Embedding generation time
- biblos_agent_confidence: Histogram of agent confidence scores
- biblos_cache_hits_total / biblos_cache_misses_total: Cache effectiveness

Usage:
    from observability.metrics import setup_metrics, record_pipeline_duration

    # Setup at startup
    setup_metrics(MetricsConfig(service_name="biblos-v2"))

    # Record metrics
    record_pipeline_duration("GEN.1.1", 1.25, "completed")
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar

from opentelemetry import metrics
from opentelemetry.metrics import (
    Counter,
    Histogram,
    UpDownCounter,
    Meter,
    MeterProvider,
    CallbackOptions,
    Observation,
)
from opentelemetry.sdk.metrics import MeterProvider as SDKMeterProvider
from opentelemetry.sdk.metrics.export import (
    PeriodicExportingMetricReader,
    ConsoleMetricExporter,
)
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

# Type variables
T = TypeVar("T")

# Global state
_meter_provider: Optional[SDKMeterProvider] = None
_meters: Dict[str, Meter] = {}
_initialized: bool = False

# Global metrics instances
_biblos_metrics: Optional["BiblosMetrics"] = None


@dataclass
class MetricsConfig:
    """Configuration for OpenTelemetry metrics."""

    service_name: str = "biblos-v2"
    service_version: str = "2.0.0"
    otlp_endpoint: str = field(
        default_factory=lambda: os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    )
    enabled: bool = field(
        default_factory=lambda: os.getenv("OTEL_METRICS_ENABLED", "true").lower() == "true"
    )
    environment: str = field(
        default_factory=lambda: os.getenv("ENVIRONMENT", "development")
    )
    console_export: bool = field(
        default_factory=lambda: os.getenv("OTEL_CONSOLE_EXPORT", "false").lower() == "true"
    )
    export_interval_millis: int = 60000  # 1 minute
    export_timeout_millis: int = 30000

    # Histogram bucket boundaries
    duration_buckets: List[float] = field(
        default_factory=lambda: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
    )
    confidence_buckets: List[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
    )


class BiblosMetrics:
    """
    Central metrics collector for BIBLOS v2.

    Provides all custom metrics used throughout the system, with
    convenient methods for recording values.
    """

    def __init__(self, meter: Meter, config: MetricsConfig):
        self.meter = meter
        self.config = config

        # Pipeline metrics
        self.pipeline_duration = meter.create_histogram(
            name="biblos_pipeline_duration_seconds",
            description="Duration of complete pipeline execution",
            unit="s",
        )

        self.phase_duration = meter.create_histogram(
            name="biblos_phase_duration_seconds",
            description="Duration of pipeline phase execution",
            unit="s",
        )

        self.agent_duration = meter.create_histogram(
            name="biblos_agent_duration_seconds",
            description="Duration of agent extraction",
            unit="s",
        )

        # Counters
        self.verses_processed = meter.create_counter(
            name="biblos_verses_processed_total",
            description="Total number of verses processed",
            unit="1",
        )

        self.crossrefs_discovered = meter.create_counter(
            name="biblos_cross_references_discovered_total",
            description="Total number of cross-references discovered",
            unit="1",
        )

        self.extraction_errors = meter.create_counter(
            name="biblos_extraction_errors_total",
            description="Total extraction errors",
            unit="1",
        )

        self.validation_failures = meter.create_counter(
            name="biblos_validation_failures_total",
            description="Total validation failures",
            unit="1",
        )

        # ML metrics
        self.ml_inference_duration = meter.create_histogram(
            name="biblos_ml_inference_duration_seconds",
            description="Duration of ML inference operations",
            unit="s",
        )

        self.embedding_duration = meter.create_histogram(
            name="biblos_embedding_generation_duration_seconds",
            description="Duration of embedding generation",
            unit="s",
        )

        self.gnn_inference_duration = meter.create_histogram(
            name="biblos_gnn_inference_duration_seconds",
            description="Duration of GNN inference",
            unit="s",
        )

        # Quality metrics
        self.agent_confidence = meter.create_histogram(
            name="biblos_agent_confidence",
            description="Distribution of agent confidence scores",
            unit="1",
        )

        self.crossref_confidence = meter.create_histogram(
            name="biblos_crossref_confidence",
            description="Distribution of cross-reference confidence scores",
            unit="1",
        )

        # Cache metrics
        self.cache_hits = meter.create_counter(
            name="biblos_cache_hits_total",
            description="Total cache hits",
            unit="1",
        )

        self.cache_misses = meter.create_counter(
            name="biblos_cache_misses_total",
            description="Total cache misses",
            unit="1",
        )

        # Database metrics
        self.db_query_duration = meter.create_histogram(
            name="biblos_db_query_duration_seconds",
            description="Duration of database queries",
            unit="s",
        )

        self.db_connections_active = meter.create_up_down_counter(
            name="biblos_db_connections_active",
            description="Number of active database connections",
            unit="1",
        )

        # Resource metrics
        self.batch_size = meter.create_histogram(
            name="biblos_batch_size",
            description="Size of processing batches",
            unit="1",
        )

        # API metrics
        self.api_request_duration = meter.create_histogram(
            name="biblos_api_request_duration_seconds",
            description="Duration of API requests",
            unit="s",
        )

        self.api_requests_total = meter.create_counter(
            name="biblos_api_requests_total",
            description="Total API requests",
            unit="1",
        )

    # Convenient recording methods
    def record_pipeline_execution(
        self,
        verse_id: str,
        duration: float,
        status: str,
        phase_count: int = 0,
    ) -> None:
        """Record pipeline execution metrics."""
        attributes = {
            "status": status,
            "book": _extract_book(verse_id),
        }
        self.pipeline_duration.record(duration, attributes)
        self.verses_processed.add(1, attributes)

    def record_phase_execution(
        self,
        phase_name: str,
        verse_id: str,
        duration: float,
        status: str,
        agent_count: int = 0,
    ) -> None:
        """Record phase execution metrics."""
        attributes = {
            "phase": phase_name,
            "status": status,
            "book": _extract_book(verse_id),
        }
        self.phase_duration.record(duration, attributes)

    def record_agent_execution(
        self,
        agent_name: str,
        extraction_type: str,
        verse_id: str,
        duration: float,
        confidence: float,
        status: str,
    ) -> None:
        """Record agent execution metrics."""
        attributes = {
            "agent": agent_name,
            "extraction_type": extraction_type,
            "status": status,
            "book": _extract_book(verse_id),
        }
        self.agent_duration.record(duration, attributes)
        self.agent_confidence.record(confidence, {"agent": agent_name})

        if status == "failed":
            self.extraction_errors.add(1, {"agent": agent_name})

    def record_crossref_discovery(
        self,
        connection_type: str,
        confidence: float,
        source_book: str,
        target_book: str,
    ) -> None:
        """Record cross-reference discovery metrics."""
        attributes = {
            "connection_type": connection_type,
            "source_book": source_book,
            "target_book": target_book,
        }
        self.crossrefs_discovered.add(1, attributes)
        self.crossref_confidence.record(confidence, {"connection_type": connection_type})

    def record_ml_inference(
        self,
        operation: str,
        duration: float,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        """Record ML inference metrics."""
        attributes = {"operation": operation}
        if model_name:
            attributes["model"] = model_name

        self.ml_inference_duration.record(duration, attributes)

        if operation == "embedding":
            self.embedding_duration.record(duration, attributes)
        elif operation == "gnn":
            self.gnn_inference_duration.record(duration, attributes)

        if batch_size:
            self.batch_size.record(batch_size, {"operation": operation})

    def record_cache_access(self, hit: bool, cache_type: str = "embedding") -> None:
        """Record cache hit/miss."""
        attributes = {"cache_type": cache_type}
        if hit:
            self.cache_hits.add(1, attributes)
        else:
            self.cache_misses.add(1, attributes)

    def record_db_query(
        self,
        operation: str,
        duration: float,
        database: str = "postgres",
    ) -> None:
        """Record database query metrics."""
        attributes = {
            "operation": operation,
            "database": database,
        }
        self.db_query_duration.record(duration, attributes)

    def record_api_request(
        self,
        endpoint: str,
        method: str,
        duration: float,
        status_code: int,
    ) -> None:
        """Record API request metrics."""
        attributes = {
            "endpoint": endpoint,
            "method": method,
            "status_code": str(status_code),
        }
        self.api_request_duration.record(duration, attributes)
        self.api_requests_total.add(1, attributes)


def setup_metrics(config: Optional[MetricsConfig] = None) -> SDKMeterProvider:
    """
    Configure OpenTelemetry metrics with OTLP export.

    Args:
        config: Metrics configuration. Uses defaults if not provided.

    Returns:
        Configured MeterProvider
    """
    global _meter_provider, _biblos_metrics, _initialized

    if _initialized and _meter_provider:
        return _meter_provider

    config = config or MetricsConfig()

    if not config.enabled:
        _initialized = True
        return metrics.get_meter_provider()

    # Build resource
    resource = Resource.create({
        SERVICE_NAME: config.service_name,
        SERVICE_VERSION: config.service_version,
        "deployment.environment": config.environment,
    })

    # Configure metric readers
    readers = []

    # OTLP exporter
    try:
        otlp_exporter = OTLPMetricExporter(
            endpoint=config.otlp_endpoint,
            insecure=True,
        )
        readers.append(
            PeriodicExportingMetricReader(
                otlp_exporter,
                export_interval_millis=config.export_interval_millis,
                export_timeout_millis=config.export_timeout_millis,
            )
        )
    except Exception as e:
        print(f"Warning: Failed to initialize OTLP metric exporter: {e}")

    # Console exporter for debugging
    if config.console_export:
        console_exporter = ConsoleMetricExporter()
        readers.append(
            PeriodicExportingMetricReader(
                console_exporter,
                export_interval_millis=config.export_interval_millis,
            )
        )

    # Create meter provider
    _meter_provider = SDKMeterProvider(
        resource=resource,
        metric_readers=readers,
    )

    # Set as global
    metrics.set_meter_provider(_meter_provider)

    # Initialize BIBLOS metrics
    meter = _meter_provider.get_meter(config.service_name, config.service_version)
    _biblos_metrics = BiblosMetrics(meter, config)

    _initialized = True
    return _meter_provider


def get_meter_provider() -> MeterProvider:
    """Get the global meter provider."""
    global _meter_provider, _initialized
    if not _initialized:
        setup_metrics()
    return _meter_provider or metrics.get_meter_provider()


def get_meter(name: str, version: str = "1.0.0") -> Meter:
    """
    Get a meter instance for custom metrics.

    Args:
        name: Meter name, typically module name
        version: Meter version

    Returns:
        Meter instance
    """
    global _meters

    if name not in _meters:
        provider = get_meter_provider()
        _meters[name] = provider.get_meter(name, version)

    return _meters[name]


def get_biblos_metrics() -> Optional[BiblosMetrics]:
    """Get the global BiblosMetrics instance."""
    global _biblos_metrics
    return _biblos_metrics


def shutdown_metrics() -> None:
    """Gracefully shutdown metrics collection."""
    global _meter_provider, _biblos_metrics, _initialized

    if _meter_provider and hasattr(_meter_provider, "shutdown"):
        _meter_provider.shutdown()

    _meter_provider = None
    _biblos_metrics = None
    _meters.clear()
    _initialized = False


# Convenience functions for direct metric recording
def record_pipeline_duration(
    verse_id: str,
    duration: float,
    status: str,
    phase_count: int = 0,
) -> None:
    """Record pipeline execution duration."""
    m = get_biblos_metrics()
    if m:
        m.record_pipeline_execution(verse_id, duration, status, phase_count)


def record_phase_duration(
    phase_name: str,
    verse_id: str,
    duration: float,
    status: str,
    agent_count: int = 0,
) -> None:
    """Record phase execution duration."""
    m = get_biblos_metrics()
    if m:
        m.record_phase_execution(phase_name, verse_id, duration, status, agent_count)


def record_agent_duration(
    agent_name: str,
    extraction_type: str,
    verse_id: str,
    duration: float,
    confidence: float,
    status: str,
) -> None:
    """Record agent execution duration."""
    m = get_biblos_metrics()
    if m:
        m.record_agent_execution(
            agent_name, extraction_type, verse_id, duration, confidence, status
        )


def record_verse_processed(verse_id: str, status: str = "completed") -> None:
    """Record a verse being processed."""
    m = get_biblos_metrics()
    if m:
        attributes = {
            "status": status,
            "book": _extract_book(verse_id),
        }
        m.verses_processed.add(1, attributes)


def record_crossref_discovered(
    connection_type: str,
    confidence: float,
    source_ref: str,
    target_ref: str,
) -> None:
    """Record cross-reference discovery."""
    m = get_biblos_metrics()
    if m:
        m.record_crossref_discovery(
            connection_type,
            confidence,
            _extract_book(source_ref),
            _extract_book(target_ref),
        )


def record_ml_inference_duration(
    operation: str,
    duration: float,
    model_name: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> None:
    """Record ML inference duration."""
    m = get_biblos_metrics()
    if m:
        m.record_ml_inference(operation, duration, model_name, batch_size)


def record_cache_access(hit: bool, cache_type: str = "embedding") -> None:
    """Record cache hit/miss."""
    m = get_biblos_metrics()
    if m:
        m.record_cache_access(hit, cache_type)


def record_db_query(
    operation: str,
    duration: float,
    database: str = "postgres",
) -> None:
    """Record database query metrics."""
    m = get_biblos_metrics()
    if m:
        m.record_db_query(operation, duration, database)


# Timing context managers
@contextmanager
def timed_operation(
    operation_type: str,
    record_func: Callable[[float], None],
):
    """
    Context manager for timing operations.

    Args:
        operation_type: Type of operation (for logging)
        record_func: Function to call with duration

    Example:
        >>> with timed_operation("ml.inference", lambda d: record_ml_inference_duration("infer", d)):
        ...     result = model.infer(data)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        record_func(duration)


@contextmanager
def timed_pipeline(verse_id: str):
    """
    Context manager for timing pipeline execution.

    Args:
        verse_id: Verse being processed

    Example:
        >>> with timed_pipeline("GEN.1.1") as ctx:
        ...     result = await pipeline.execute(verse_id, text)
        ...     ctx["status"] = result.status
    """
    context = {"status": "unknown"}
    start = time.perf_counter()
    try:
        yield context
    except Exception:
        context["status"] = "error"
        raise
    finally:
        duration = time.perf_counter() - start
        record_pipeline_duration(verse_id, duration, context["status"])


@contextmanager
def timed_phase(phase_name: str, verse_id: str, agent_count: int = 0):
    """
    Context manager for timing phase execution.

    Args:
        phase_name: Name of the phase
        verse_id: Verse being processed
        agent_count: Number of agents in the phase
    """
    context = {"status": "unknown"}
    start = time.perf_counter()
    try:
        yield context
    except Exception:
        context["status"] = "error"
        raise
    finally:
        duration = time.perf_counter() - start
        record_phase_duration(phase_name, verse_id, duration, context["status"], agent_count)


@contextmanager
def timed_agent(
    agent_name: str,
    extraction_type: str,
    verse_id: str,
):
    """
    Context manager for timing agent execution.

    Args:
        agent_name: Name of the agent
        extraction_type: Type of extraction
        verse_id: Verse being processed
    """
    context = {"status": "unknown", "confidence": 0.0}
    start = time.perf_counter()
    try:
        yield context
    except Exception:
        context["status"] = "failed"
        raise
    finally:
        duration = time.perf_counter() - start
        record_agent_duration(
            agent_name,
            extraction_type,
            verse_id,
            duration,
            context["confidence"],
            context["status"],
        )


@contextmanager
def timed_ml_inference(
    operation: str,
    model_name: Optional[str] = None,
    batch_size: Optional[int] = None,
):
    """
    Context manager for timing ML inference.

    Args:
        operation: Type of ML operation
        model_name: Name of the model
        batch_size: Batch size
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        record_ml_inference_duration(operation, duration, model_name, batch_size)


# Utility functions
def _extract_book(verse_id: str) -> str:
    """Extract book code from verse ID."""
    if not verse_id:
        return "UNKNOWN"
    parts = verse_id.split(".")
    return parts[0] if parts else "UNKNOWN"
