"""
BIBLOS v2 - Core Module

Provides foundational components for the entire system:
- Unified error handling
- Resilience patterns (circuit breaker, retry)
- Async utilities
- Configuration validation
- Type definitions

All modules should import from core for consistent behavior.
"""

from core.errors import (
    BiblosError,
    BiblosConfigError,
    BiblosDatabaseError,
    BiblosMLError,
    BiblosPipelineError,
    BiblosAgentError,
    BiblosValidationError,
    BiblosTimeoutError,
    BiblosResourceError,
    ErrorContext,
    ErrorSeverity,
    error_handler,
    safe_execute,
)
from core.resilience import (
    CircuitBreaker,
    CircuitState,
    RetryPolicy,
    Bulkhead,
    RateLimiter,
    RateLimiterConfig,
    FallbackResult,
    BatchResult,
    BatchError,
    HealthStatus,
    HealthCheck,
    HealthMonitor,
    with_retry,
    with_circuit_breaker,
    with_bulkhead,
    with_rate_limit,
    with_fallback,
    batch_execute,
    resilient,
    get_health_monitor,
)
from core.async_utils import (
    AsyncTaskGroup,
    AsyncBatcher,
    AsyncThrottler,
    LazyAsync,
    PriorityTaskQueue,
    AsyncContextStack,
    gather_with_concurrency,
    timeout_with_cleanup,
    cancel_scope,
    async_cached,
    async_chunked_iter,
    async_buffered_iter,
    debounce,
    coalesce,
)
from core.config_validator import (
    ConfigValidator,
    ValidationResult,
    validate_config,
    require_env,
)
from core.validation import (
    validate_verse_id,
    is_valid_verse_id,
    normalize_verse_id,
    validate_book_code,
    parse_verse_range,
    VerseIdValidationError,
    VALID_BOOK_CODES,
)
from core.types import (
    # Type aliases
    VerseId,
    BookCode,
    WordId,
    ConnectionTypeLiteral,
    StrengthLiteral,
    StatusLiteral,
    Confidence,
    # TypedDicts
    VerseDict,
    WordDict,
    CrossReferenceDict,
    ExtractionResultDict,
    GoldenRecordDict,
    InferenceCandidateDict,
    # Protocols
    Validatable,
    Serializable,
    ExtractionAgent,
    DatabaseClient,
    VectorStore,
    EmbeddingModel,
    PipelinePhase,
    # Result type
    Result,
    # Type guards
    is_verse_id,
    is_connection_type,
    is_strength,
    is_confidence,
    # Sentinels
    MISSING,
    UNSET,
)
# Bootstrap and factories
from core.bootstrap import (
    Application,
    ApplicationConfig,
    ApplicationPhase,
    ApplicationLifecycleHook,
    bootstrap,
    get_application,
    run_application,
)
from core.factories import (
    Factory,
    Builder,
    PipelineFactory,
    PipelineConfig,
    AgentFactory,
    AgentConfig,
    DatabaseClientFactory,
    DatabaseConfig,
    MLEngineFactory,
    MLEngineConfig,
    GoldenRecordBuilder,
)

__all__ = [
    # Errors
    "BiblosError",
    "BiblosConfigError",
    "BiblosDatabaseError",
    "BiblosMLError",
    "BiblosPipelineError",
    "BiblosAgentError",
    "BiblosValidationError",
    "BiblosTimeoutError",
    "BiblosResourceError",
    "ErrorContext",
    "ErrorSeverity",
    "error_handler",
    "safe_execute",
    # Resilience
    "CircuitBreaker",
    "CircuitState",
    "RetryPolicy",
    "Bulkhead",
    "RateLimiter",
    "RateLimiterConfig",
    "FallbackResult",
    "BatchResult",
    "BatchError",
    "HealthStatus",
    "HealthCheck",
    "HealthMonitor",
    "with_retry",
    "with_circuit_breaker",
    "with_bulkhead",
    "with_rate_limit",
    "with_fallback",
    "batch_execute",
    "resilient",
    "get_health_monitor",
    # Async
    "AsyncTaskGroup",
    "AsyncBatcher",
    "AsyncThrottler",
    "LazyAsync",
    "PriorityTaskQueue",
    "AsyncContextStack",
    "gather_with_concurrency",
    "timeout_with_cleanup",
    "cancel_scope",
    "async_cached",
    "async_chunked_iter",
    "async_buffered_iter",
    "debounce",
    "coalesce",
    # Config
    "ConfigValidator",
    "ValidationResult",
    "validate_config",
    "require_env",
    # Input validation
    "validate_verse_id",
    "is_valid_verse_id",
    "normalize_verse_id",
    "validate_book_code",
    "parse_verse_range",
    "VerseIdValidationError",
    "VALID_BOOK_CODES",
    # Bootstrap
    "Application",
    "ApplicationConfig",
    "ApplicationPhase",
    "ApplicationLifecycleHook",
    "bootstrap",
    "get_application",
    "run_application",
    # Factories
    "Factory",
    "Builder",
    "PipelineFactory",
    "PipelineConfig",
    "AgentFactory",
    "AgentConfig",
    "DatabaseClientFactory",
    "DatabaseConfig",
    "MLEngineFactory",
    "MLEngineConfig",
    "GoldenRecordBuilder",
]
