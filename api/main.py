"""
BIBLOS v2 - FastAPI Application

Production-ready API for biblical cross-reference discovery,
text extraction, and ML inference.

Includes comprehensive OpenTelemetry instrumentation for distributed tracing,
metrics collection, and structured logging with trace context.
"""
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
import time
import uuid

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Response, Depends
from pydantic import BaseModel, Field

# Import security components
from api.security.auth import (
    AuthConfig,
    configure_auth,
    get_current_user,
    require_auth,
    User,
)
from api.security.rate_limit import RateLimitConfig, RateLimitMiddleware
from api.security.cors import CORSConfig, add_cors_middleware
from api.security.headers import SecurityHeadersMiddleware, SecurityHeadersConfig
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

# Import observability components
from observability import (
    setup_observability,
    shutdown_observability,
    get_tracer,
    get_logger,
)
from observability.tracing import instrument_fastapi, create_span
from observability.metrics import (
    get_biblos_metrics,
    timed_ml_inference,
    record_api_request,
)
from observability.logging import LogContext, bind_context, clear_context

# Initialize observability on module load
setup_observability(
    service_name="biblos-api",
    sample_rate=1.0,  # 100% sampling in development
    log_level="INFO",
)

# Get logger and tracer
logger = get_logger(__name__)
tracer = get_tracer(__name__)


# Pydantic models for API
class VerseRequest(BaseModel):
    """Request for verse processing."""
    verse_id: str = Field(..., description="Canonical verse ID (e.g., GEN.1.1)")
    text: str = Field(..., description="Verse text")
    context: Optional[Dict[str, Any]] = Field(default={}, description="Additional context")


class CrossRefDiscoveryRequest(BaseModel):
    """Request for cross-reference discovery."""
    source_verse: str = Field(..., description="Source verse reference")
    top_k: int = Field(default=10, description="Number of results")
    min_confidence: float = Field(default=0.5, description="Minimum confidence threshold")
    connection_types: Optional[List[str]] = Field(default=None, description="Filter by types")


class EmbeddingRequest(BaseModel):
    """Request for text embedding."""
    texts: List[str] = Field(..., description="Texts to embed")
    models: Optional[List[str]] = Field(default=None, description="Specific models to use")


class ExtractionRequest(BaseModel):
    """Request for multi-agent extraction."""
    verse_id: str
    text: str
    agents: Optional[List[str]] = Field(default=None, description="Specific agents to run")
    parallel: bool = Field(default=True, description="Run agents in parallel")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    components: Dict[str, str]
    trace_id: Optional[str] = None


class CrossRefResult(BaseModel):
    """Cross-reference discovery result."""
    source_ref: str
    target_ref: str
    connection_type: str
    confidence: float
    features: Dict[str, Any]


class ExtractionResponse(BaseModel):
    """Extraction response."""
    verse_id: str
    results: Dict[str, Any]
    overall_confidence: float
    processing_time_ms: float
    trace_id: Optional[str] = None


# Global app state
class AppState:
    embedder = None
    gnn_model = None
    orchestrator = None
    db_pool = None


state = AppState()


def get_current_trace_id() -> Optional[str]:
    """Get current trace ID as hex string."""
    span = trace.get_current_span()
    if span and span.is_recording():
        ctx = span.get_span_context()
        if ctx.is_valid:
            return format(ctx.trace_id, "032x")
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler with observability setup."""
    logger.info("Starting BIBLOS v2 API", event="startup")

    # Initialize ML components
    try:
        from ml.embeddings.ensemble import EnsembleEmbedder
        with tracer.start_as_current_span("startup.embedder") as span:
            state.embedder = EnsembleEmbedder(device="cpu")
            span.set_attribute("status", "success")
        logger.info("Initialized embedding ensemble", component="embedder")
    except Exception as e:
        logger.warning("Failed to initialize embedder", error=str(e), component="embedder")

    try:
        from agents.orchestrator import AgentOrchestrator
        with tracer.start_as_current_span("startup.orchestrator") as span:
            state.orchestrator = AgentOrchestrator()
            span.set_attribute("status", "success")
        logger.info("Initialized agent orchestrator", component="orchestrator")
    except Exception as e:
        logger.warning("Failed to initialize orchestrator", error=str(e), component="orchestrator")

    yield

    # Cleanup with observability
    logger.info("Shutting down BIBLOS v2 API", event="shutdown")
    if state.orchestrator:
        try:
            from agents.registry import registry
            await registry.shutdown_all()
        except Exception as e:
            logger.error("Error during shutdown", error=str(e))

    # Shutdown observability
    shutdown_observability()


def create_app() -> FastAPI:
    """Create and configure FastAPI application with observability and security."""
    app = FastAPI(
        title="BIBLOS v2 API",
        description="Biblical Cross-Reference Discovery and Text Extraction API",
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # Configure authentication
    configure_auth(AuthConfig(
        enabled=True,
        public_endpoints={"/health", "/docs", "/redoc", "/openapi.json", "/metrics"},
    ))

    # Secure CORS configuration (no wildcard + credentials vulnerability)
    add_cors_middleware(app, CORSConfig(
        allow_origins=[
            "http://localhost:3000",
            "http://localhost:8000",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8000",
        ],
        allow_credentials=True,
    ))

    # Security headers middleware
    app.add_middleware(
        SecurityHeadersMiddleware,
        config=SecurityHeadersConfig(
            environment="development",
            csp_enabled=True,
        )
    )

    # Rate limiting middleware
    app.add_middleware(
        RateLimitMiddleware,
        config=RateLimitConfig(
            default_limit=100,
            default_window_seconds=60,
            endpoint_limits={
                "/api/v1/extract": (20, 60),  # More restrictive for heavy endpoints
                "/api/v1/batch/extract": (5, 60),
            },
        )
    )

    # Instrument FastAPI with OpenTelemetry
    instrument_fastapi(app)

    return app


app = create_app()


# Request tracking middleware
@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    """Add request tracking with trace context."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start_time = time.perf_counter()

    # Bind request context to all logs
    bind_context(
        request_id=request_id,
        method=request.method,
        path=request.url.path,
    )

    try:
        response: Response = await call_next(request)
        duration = time.perf_counter() - start_time

        # Record API metrics
        metrics = get_biblos_metrics()
        if metrics:
            metrics.record_api_request(
                endpoint=request.url.path,
                method=request.method,
                duration=duration,
                status_code=response.status_code,
            )

        # Add trace ID to response headers
        trace_id = get_current_trace_id()
        if trace_id:
            response.headers["X-Trace-ID"] = trace_id

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration * 1000:.2f}ms"

        logger.info(
            "Request completed",
            status_code=response.status_code,
            duration_ms=duration * 1000,
        )

        return response
    except Exception as e:
        duration = time.perf_counter() - start_time
        logger.error("Request failed", error=str(e), duration_ms=duration * 1000)
        raise
    finally:
        clear_context()


# Health endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and component status."""
    with create_span("health_check", attributes={"endpoint": "/health"}) as span:
        components = {
            "embedder": "healthy" if state.embedder else "unavailable",
            "orchestrator": "healthy" if state.orchestrator else "unavailable",
            "gnn_model": "healthy" if state.gnn_model else "unavailable",
            "database": "healthy" if state.db_pool else "unavailable"
        }

        overall = "healthy" if all(
            v == "healthy" for k, v in components.items()
            if k in ["embedder", "orchestrator"]
        ) else "degraded"

        span.set_attribute("health.status", overall)
        for component, status in components.items():
            span.set_attribute(f"health.{component}", status)

        return HealthResponse(
            status=overall,
            version="2.0.0",
            components=components,
            trace_id=get_current_trace_id()
        )


# Embedding endpoints
@app.post("/api/v1/embed")
async def embed_texts(request: EmbeddingRequest):
    """Generate embeddings for texts with full tracing."""
    if not state.embedder:
        raise HTTPException(status_code=503, detail="Embedding service unavailable")

    with tracer.start_as_current_span(
        "embed_texts",
        kind=SpanKind.SERVER,
    ) as span:
        span.set_attribute("embedding.text_count", len(request.texts))
        span.set_attribute("embedding.models", str(request.models))

        start_time = time.perf_counter()
        results = []

        for i, text in enumerate(request.texts):
            with tracer.start_as_current_span(f"embed_single_{i}") as embed_span:
                embed_span.set_attribute("text.length", len(text))

                with timed_ml_inference("embedding", model_name="ensemble"):
                    result = await state.embedder.embed(text)

                results.append({
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "embedding_shape": list(result.fused_embedding.shape),
                    "detected_language": result.detected_language,
                    "weights": result.weights
                })

        processing_time = (time.perf_counter() - start_time) * 1000
        span.set_attribute("processing_time_ms", processing_time)

        logger.info(
            "Embeddings generated",
            text_count=len(request.texts),
            processing_time_ms=processing_time,
        )

        return {
            "results": results,
            "processing_time_ms": processing_time,
            "trace_id": get_current_trace_id()
        }


# Cross-reference discovery endpoints
@app.post("/api/v1/crossref/discover", response_model=List[CrossRefResult])
async def discover_crossrefs(request: CrossRefDiscoveryRequest):
    """Discover potential cross-references for a verse."""
    with tracer.start_as_current_span(
        "discover_crossrefs",
        kind=SpanKind.SERVER,
    ) as span:
        span.set_attribute("crossref.source_verse", request.source_verse)
        span.set_attribute("crossref.top_k", request.top_k)
        span.set_attribute("crossref.min_confidence", request.min_confidence)

        if not state.gnn_model:
            if not state.embedder:
                span.set_status(Status(StatusCode.ERROR, "ML services unavailable"))
                raise HTTPException(status_code=503, detail="ML services unavailable")

            span.set_attribute("crossref.fallback", True)
            logger.warning(
                "Using fallback discovery",
                source_verse=request.source_verse,
            )

            return [{
                "source_ref": request.source_verse,
                "target_ref": "Discovery requires initialized GNN model",
                "connection_type": "unknown",
                "confidence": 0.0,
                "features": {}
            }]

        # Use GNN model for discovery
        with timed_ml_inference("gnn_discovery", model_name="gnn"):
            # This would be implemented with the actual model
            pass

        return []


# Extraction endpoints (protected)
@app.post("/api/v1/extract", response_model=ExtractionResponse)
async def extract_verse(
    request: ExtractionRequest,
    user: User = Depends(require_auth),
):
    """Run multi-agent extraction on a verse with comprehensive tracing.

    Requires authentication via API key or JWT token.
    """
    if not state.orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator unavailable")

    with tracer.start_as_current_span(
        "extract_verse",
        kind=SpanKind.SERVER,
    ) as span:
        span.set_attribute("verse.id", request.verse_id)
        span.set_attribute("extraction.parallel", request.parallel)
        if request.agents:
            span.set_attribute("extraction.agents", request.agents)

        start_time = time.perf_counter()

        try:
            with LogContext(verse_id=request.verse_id, operation="extraction"):
                logger.info("Starting verse extraction")

                result = await state.orchestrator.process_verse(
                    request.verse_id,
                    request.text,
                    {"agents": request.agents}
                )

                processing_time = (time.perf_counter() - start_time) * 1000
                overall_confidence = result["metadata"].get("overall_confidence", 0.0)

                span.set_attribute("result.confidence", overall_confidence)
                span.set_attribute("processing_time_ms", processing_time)

                logger.info(
                    "Extraction completed",
                    confidence=overall_confidence,
                    processing_time_ms=processing_time,
                )

                return ExtractionResponse(
                    verse_id=request.verse_id,
                    results=result["results"],
                    overall_confidence=overall_confidence,
                    processing_time_ms=processing_time,
                    trace_id=get_current_trace_id()
                )

        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            logger.error("Extraction failed", error=str(e), verse_id=request.verse_id)
            raise HTTPException(status_code=500, detail=str(e))


# Batch endpoints
@app.post("/api/v1/batch/extract")
async def batch_extract(
    verses: List[VerseRequest],
    background_tasks: BackgroundTasks
):
    """Queue batch extraction job with tracing."""
    if not state.orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator unavailable")

    with tracer.start_as_current_span(
        "batch_extract_queue",
        kind=SpanKind.SERVER,
    ) as span:
        job_id = str(uuid.uuid4())
        span.set_attribute("batch.job_id", job_id)
        span.set_attribute("batch.verse_count", len(verses))

        # Capture trace context for background task
        current_context = trace.get_current_span().get_span_context()

        async def run_batch():
            """Background batch processing with linked traces."""
            with tracer.start_as_current_span(
                "batch_extract_process",
                kind=SpanKind.CONSUMER,
                links=[trace.Link(current_context)],
            ) as batch_span:
                batch_span.set_attribute("batch.job_id", job_id)

                for i, verse in enumerate(verses):
                    with tracer.start_as_current_span(f"batch_verse_{i}") as verse_span:
                        verse_span.set_attribute("verse.id", verse.verse_id)
                        try:
                            await state.orchestrator.process_verse(
                                verse.verse_id,
                                verse.text,
                                verse.context
                            )
                            verse_span.set_status(Status(StatusCode.OK))
                        except Exception as e:
                            verse_span.set_status(Status(StatusCode.ERROR, str(e)))
                            verse_span.record_exception(e)
                            logger.error(
                                "Batch verse processing failed",
                                verse_id=verse.verse_id,
                                error=str(e),
                            )

        background_tasks.add_task(run_batch)

        logger.info(
            "Batch job queued",
            job_id=job_id,
            verse_count=len(verses),
        )

        return {
            "job_id": job_id,
            "status": "queued",
            "total_verses": len(verses),
            "trace_id": get_current_trace_id()
        }


# Stats endpoints
@app.get("/api/v1/stats")
async def get_stats():
    """Get API statistics with tracing."""
    with create_span("get_stats", attributes={"endpoint": "/stats"}) as span:
        stats = {
            "api_version": "2.0.0",
            "models_loaded": [],
            "cache_stats": {},
            "trace_id": get_current_trace_id()
        }

        if state.embedder:
            stats["models_loaded"].append("embedder")
            try:
                stats["cache_stats"]["embeddings"] = state.embedder.get_cache_stats()
            except Exception:
                pass

        if state.gnn_model:
            stats["models_loaded"].append("gnn")

        span.set_attribute("stats.models_loaded", len(stats["models_loaded"]))

        return stats


# Metrics endpoint for Prometheus scraping
@app.get("/metrics")
async def get_metrics():
    """Expose Prometheus metrics endpoint."""
    # This would typically use prometheus_client
    # For OpenTelemetry, metrics are pushed to the collector
    return {
        "message": "Metrics are exported via OTLP to the configured collector",
        "trace_id": get_current_trace_id()
    }


def run_server():
    """Run the API server."""
    import uvicorn

    logger.info("Starting uvicorn server", host="0.0.0.0", port=8000)
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=4
    )


if __name__ == "__main__":
    run_server()
