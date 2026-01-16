# BIBLOS v2 System Architecture

## Overview

BIBLOS v2 is a comprehensive ML pipeline orchestration system for biblical scholarship, featuring:
- 24-agent SDES (Scripture Data Extraction System) pipeline
- Cross-reference discovery using GNN and embedding models
- Orthodox Christian patristic integration
- LangChain/LangGraph multi-agent framework

## Directory Structure

```
BIBLOS_v2/
├── agents/                    # 24 SDES extraction agents
│   ├── base.py               # BaseExtractionAgent foundation
│   ├── base_v2.py            # Enhanced agent with resilience
│   ├── orchestrator.py       # LangGraph workflow orchestrator
│   ├── langgraph_orchestrator.py  # Extended orchestrator
│   ├── registry.py           # Agent registry
│   ├── linguistic/           # Grammatical analysis agents
│   ├── theological/          # Patristic & typological agents
│   ├── intertextual/        # Cross-reference agents
│   ├── validation/          # Quality assurance agents
│   └── tools/               # LangChain tool implementations
│
├── api/                      # FastAPI REST endpoints
│   ├── main.py              # Application entry point
│   └── security/            # Security middleware
│       ├── auth.py          # API key & JWT authentication
│       ├── cors.py          # Secure CORS configuration
│       ├── rate_limit.py    # Rate limiting middleware
│       └── headers.py       # Security headers
│
├── core/                     # Core infrastructure
│   ├── __init__.py          # Central exports
│   ├── errors.py            # Unified error hierarchy
│   ├── resilience.py        # Circuit breaker, retry, bulkhead
│   ├── async_utils.py       # Advanced async patterns
│   ├── config_validator.py  # Configuration validation
│   └── exports.py           # Unified export point
│
├── db/                       # Database clients
│   ├── models.py            # SQLAlchemy models
│   ├── postgres.py          # PostgreSQL client
│   ├── postgres_optimized.py # Optimized client
│   ├── neo4j_client.py      # Neo4j graph database
│   ├── qdrant_client.py     # Vector similarity search
│   └── connection_pool*.py  # Connection pooling
│
├── di/                       # Dependency injection
│   ├── __init__.py          # DI exports
│   └── container.py         # IoC container
│
├── ml/                       # Machine learning pipeline
│   ├── cache.py             # O(1) LRU cache with TTL
│   ├── batch_processor.py   # Parallel batch processing
│   ├── embeddings/          # Sentence embeddings
│   ├── models/              # GNN & classifiers
│   ├── inference/           # Inference pipeline
│   └── training/            # Training infrastructure
│
├── observability/            # Monitoring & tracing
│   ├── __init__.py          # Setup functions
│   ├── tracing.py           # OpenTelemetry distributed tracing
│   ├── metrics.py           # Custom BIBLOS metrics
│   ├── logging.py           # Structlog integration
│   └── instrumentation.py   # Auto-instrumentation
│
├── pipeline/                 # Pipeline orchestration
│   ├── base.py              # BasePipelinePhase
│   ├── orchestrator.py      # Main pipeline coordinator
│   ├── event_bus.py         # Redis Streams event bus
│   ├── stream_*.py          # Stream processing
│   ├── recovery.py          # Error recovery
│   ├── linguistic.py        # Linguistic phase
│   ├── theological.py       # Theological phase
│   ├── intertextual.py      # Intertextual phase
│   ├── validation.py        # Validation phase
│   └── finalization.py      # Finalization phase
│
├── data/                     # Data handling
│   ├── schemas.py           # Normalized data schemas
│   ├── dataset.py           # PyTorch datasets
│   ├── loaders.py           # DataLoader factories
│   └── polars_schemas.py    # Polars dataframe schemas
│
├── integrations/             # External integrations
│   ├── text_fabric.py       # Text-Fabric (BHSA, SBLGNT)
│   └── macula.py            # Macula Greek/Hebrew
│
├── evaluation/               # Evaluation framework
│   └── framework.py         # Quality evaluation
│
├── tests/                    # Test suite
│   ├── agents/
│   ├── pipeline/
│   ├── ml/
│   ├── api/
│   └── property/            # Property-based tests
│
└── scripts/                  # Utility scripts
    ├── populate_data.py
    └── run_*.py
```

## Core Components

### Error Handling (`core/errors.py`)

Unified error hierarchy:
```python
BiblosError (base)
├── BiblosConfigError      # Configuration errors
├── BiblosDatabaseError    # Database operations
├── BiblosMLError          # ML inference/training
├── BiblosPipelineError    # Pipeline execution
├── BiblosAgentError       # Agent processing
├── BiblosValidationError  # Data validation
├── BiblosTimeoutError     # Timeout conditions
└── BiblosResourceError    # Resource exhaustion
```

### Resilience Patterns (`core/resilience.py`)

- **CircuitBreaker**: Prevents cascading failures
- **RetryPolicy**: Configurable retry with exponential backoff
- **Bulkhead**: Resource isolation for parallel operations
- **@resilient**: Combined decorator for all patterns

### Async Utilities (`core/async_utils.py`)

- **AsyncTaskGroup**: Structured concurrency for task groups
- **AsyncBatcher**: Automatic request batching
- **AsyncThrottler**: Rate limiting for external calls
- **gather_with_concurrency**: Controlled parallel execution

### DI Container (`di/container.py`)

IoC container supporting:
- Singleton lifetime (one instance per container)
- Scoped lifetime (one instance per scope)
- Transient lifetime (new instance each resolve)
- Request-scoped dependencies for web apps

## Pipeline Architecture

### 5-Phase Extraction Pipeline

1. **Linguistic Phase**: phonology, morphology, syntax, semantics
2. **Theological Phase**: patristic, typological, liturgical
3. **Intertextual Phase**: cross-references, parallels, allusions
4. **Validation Phase**: verification, correction, adversarial testing
5. **Finalization Phase**: golden record creation

### Event-Driven Communication

Redis Streams provides:
- At-least-once delivery
- Consumer groups for parallel processing
- Message replay for recovery
- Dead letter handling

## Database Architecture

### PostgreSQL (pgvector)
- Relational data storage
- Vector embeddings for similarity search
- Full-text search indexes

### Neo4j (SPIDERWEB Schema)
- Graph relationships between verses
- Patristic citation networks
- Thematic connections

### Qdrant
- High-performance vector similarity
- Multi-collection support
- Filtered searches

### Redis
- Caching layer
- Event bus (Streams)
- Rate limiting counters

## ML Components

### Embedding System
- Multi-model ensemble (multilingual, biblical-specific)
- O(1) LRU cache with TTL
- GPU-optimized batch processing

### Cross-Reference Discovery
- GNN-based relationship discovery
- Semantic similarity scoring
- Confidence-based filtering

## Observability

### Distributed Tracing
- OpenTelemetry with OTLP export
- Flame graph visualization in Jaeger/Tempo
- Trace context propagation

### Metrics
- Pipeline duration histograms
- Agent processing times
- Cache hit rates
- ML inference latency

### Logging
- Structlog for structured logging
- Trace context in log messages
- JSON format for log aggregation

## Security

### Authentication
- API key validation (SHA256 hashed)
- JWT token support
- Request-scoped user context

### Rate Limiting
- Sliding window algorithm
- Redis backend for distributed deployments
- Per-endpoint configuration

### CORS
- Secure origin validation
- No wildcard with credentials
- Preflight caching

## Configuration

Environment-based configuration with validation:
- Database connections
- ML model parameters
- API settings
- Observability endpoints

## Import Conventions

```python
# Core imports
from core import (
    BiblosError, CircuitBreaker, AsyncTaskGroup,
    ConfigValidator, resilient,
)

# Database imports
from db import Neo4jClient, QdrantVectorStore, PostgresClient

# ML imports
from ml import LRUCache, AsyncLRUCache, EmbeddingBatcher

# Observability imports
from observability import setup_observability, get_tracer, get_meter
```

## Development Guidelines

1. **Error Handling**: Always use specific exception types
2. **Resilience**: Apply circuit breakers to external calls
3. **Async**: Use structured concurrency patterns
4. **Testing**: Property-based tests for data transformations
5. **Observability**: Add spans for significant operations
