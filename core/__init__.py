"""
BIBLOS v2 - Core Module (The Nervous System)

The core module is the organism's central nervous system - coordinating signals
between all organs, managing lifecycle, and ensuring coherent behavior.

Provides foundational components for the entire system:
- Unified error handling (pain signals)
- Resilience patterns (immune system: circuit breaker, retry)
- Async utilities (neural pathways)
- Configuration validation (sensory validation)
- Type definitions (cellular blueprints)
- Application lifecycle (autonomic nervous system)
- Factory abstractions (stem cells)

All modules should import from core for consistent behavior.

Architectural Role:
    The core doesn't DO work - it COORDINATES work. Like the nervous system,
    it doesn't digest food or pump blood, but without it, nothing functions.
    Each component here is designed to be dependency-free from other BIBLOS
    modules, making it the stable foundation everything else builds upon.

Usage:
    from core import (
        # Lifecycle
        Application, ApplicationBuilder,
        # Error handling
        BiblosError, safe_execute,
        # Resilience
        CircuitBreaker, with_retry,
        # Factories
        SystemFactory, GoldenRecordBuilder,
    )

    # Create application with builder
    app = await (
        ApplicationBuilder()
        .with_environment("production")
        .with_module(DatabaseModule())
        .with_module(PipelineModule())
        .build()
    )
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

# ============================================================================
# Bootstrap - The Autonomic Nervous System
# ============================================================================
# These components manage application lifecycle, from heartbeat to shutdown.
# Like the autonomic nervous system, they operate without conscious thought,
# ensuring the organism functions even when attention is elsewhere.
# ============================================================================

from core.bootstrap import (
    # Core Application
    Application,
    ApplicationConfig,
    ApplicationPhase,
    # Lifecycle Stages (fine-grained control)
    StartupStage,
    ShutdownStage,
    LifecycleEvent,
    # Lifecycle Hook Interfaces
    ILifecycleHook,
    LifecycleHookBase,
    # Service Module Interfaces
    IServiceModule,
    ServiceModuleBase,
    # Background Service Interfaces
    IBackgroundService,
    BackgroundServiceBase,
    # Health Aggregation
    AggregateHealthStatus,
    AggregateHealthReport,
    HealthAggregator,
    # Feature Flags
    FeatureFlag,
    # Builder Pattern
    ApplicationBuilder,
    # Global Application Access
    get_application,
    set_application,
    # Built-in Modules
    DatabaseModule,
    EventSourcingModule,
    PipelineModule,
    MediatorModule,
    # Built-in Lifecycle Hooks
    LoggingLifecycleHook,
    TelemetryLifecycleHook,
    # Built-in Background Services
    HealthMonitorService,
)

# Legacy aliases for backward compatibility
ApplicationLifecycleHook = LifecycleHookBase
bootstrap = ApplicationBuilder  # Alias for old bootstrap function


def run_application(app: Application) -> None:
    """
    Synchronous wrapper to run an application.

    This is a convenience function for scripts that don't use async/await.
    For production use, prefer the async pattern with ApplicationBuilder.
    """
    import asyncio
    asyncio.run(app.run())


# ============================================================================
# Factories - The Stem Cells
# ============================================================================
# Factories create the specialized cells (components) of the organism.
# Like stem cells, they can produce any type needed, following the blueprints
# (configurations) they're given. The SystemFactory is the bone marrow,
# coordinating all production.
# ============================================================================

from core.factories import (
    # Factory Interfaces (ISP-compliant protocols)
    IFactory,
    IAsyncFactory,
    IConfigurableFactory,
    IPooledFactory,
    IBuilder,
    # Factory Base Classes
    FactoryBase,
    AsyncFactoryBase,
    BuilderBase,
    # Object Pooling (reusable expensive objects)
    ObjectPoolState,
    PooledObject,
    ObjectPool,
    # Factory Registry (factory discovery)
    FactoryRegistry,
    get_factory_registry,
    # Pipeline Factory
    PipelinePhaseSet,
    PipelineConfig,
    IPipelineFactory,
    PipelineFactory,
    # Agent Factory
    AgentCategory,
    AgentConfig,
    IAgentFactory,
    AgentFactory,
    # Database Client Factory
    DatabaseType,
    DatabaseConfig,
    IDatabaseClientFactory,
    DatabaseClientFactory,
    # ML Engine Factory
    MLDeviceType,
    MLEngineConfig,
    IMLEngineFactory,
    MLEngineFactory,
    # Golden Record (the sacred output)
    QualityTier,
    GoldenRecord,
    GoldenRecordBuilder,
    # System Factory (composite factory)
    SystemFactory,
)

# Legacy aliases for backward compatibility
Factory = FactoryBase
Builder = BuilderBase


__all__ = [
    # ========================================================================
    # ERRORS - Pain Signals
    # ========================================================================
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

    # ========================================================================
    # RESILIENCE - Immune System
    # ========================================================================
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

    # ========================================================================
    # ASYNC UTILITIES - Neural Pathways
    # ========================================================================
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

    # ========================================================================
    # CONFIGURATION - Sensory Validation
    # ========================================================================
    "ConfigValidator",
    "ValidationResult",
    "validate_config",
    "require_env",

    # ========================================================================
    # INPUT VALIDATION - Gating Functions
    # ========================================================================
    "validate_verse_id",
    "is_valid_verse_id",
    "normalize_verse_id",
    "validate_book_code",
    "parse_verse_range",
    "VerseIdValidationError",
    "VALID_BOOK_CODES",

    # ========================================================================
    # TYPES - Cellular Blueprints
    # ========================================================================
    "VerseId",
    "BookCode",
    "WordId",
    "ConnectionTypeLiteral",
    "StrengthLiteral",
    "StatusLiteral",
    "Confidence",
    "VerseDict",
    "WordDict",
    "CrossReferenceDict",
    "ExtractionResultDict",
    "GoldenRecordDict",
    "InferenceCandidateDict",
    "Validatable",
    "Serializable",
    "ExtractionAgent",
    "DatabaseClient",
    "VectorStore",
    "EmbeddingModel",
    "PipelinePhase",
    "Result",
    "is_verse_id",
    "is_connection_type",
    "is_strength",
    "is_confidence",
    "MISSING",
    "UNSET",

    # ========================================================================
    # BOOTSTRAP - Autonomic Nervous System
    # ========================================================================
    # Core Application
    "Application",
    "ApplicationConfig",
    "ApplicationPhase",
    # Lifecycle Stages
    "StartupStage",
    "ShutdownStage",
    "LifecycleEvent",
    # Lifecycle Hooks
    "ILifecycleHook",
    "LifecycleHookBase",
    "ApplicationLifecycleHook",  # Legacy alias
    # Service Modules
    "IServiceModule",
    "ServiceModuleBase",
    # Background Services
    "IBackgroundService",
    "BackgroundServiceBase",
    # Health Aggregation
    "AggregateHealthStatus",
    "AggregateHealthReport",
    "HealthAggregator",
    # Feature Flags
    "FeatureFlag",
    # Builder
    "ApplicationBuilder",
    "bootstrap",  # Legacy alias
    # Global Access
    "get_application",
    "set_application",
    "run_application",
    # Built-in Modules
    "DatabaseModule",
    "EventSourcingModule",
    "PipelineModule",
    "MediatorModule",
    # Built-in Hooks
    "LoggingLifecycleHook",
    "TelemetryLifecycleHook",
    # Built-in Services
    "HealthMonitorService",

    # ========================================================================
    # FACTORIES - Stem Cells
    # ========================================================================
    # Factory Interfaces
    "IFactory",
    "IAsyncFactory",
    "IConfigurableFactory",
    "IPooledFactory",
    "IBuilder",
    # Factory Base Classes
    "FactoryBase",
    "AsyncFactoryBase",
    "BuilderBase",
    "Factory",  # Legacy alias
    "Builder",  # Legacy alias
    # Object Pooling
    "ObjectPoolState",
    "PooledObject",
    "ObjectPool",
    # Factory Registry
    "FactoryRegistry",
    "get_factory_registry",
    # Pipeline Factory
    "PipelinePhaseSet",
    "PipelineConfig",
    "IPipelineFactory",
    "PipelineFactory",
    # Agent Factory
    "AgentCategory",
    "AgentConfig",
    "IAgentFactory",
    "AgentFactory",
    # Database Client Factory
    "DatabaseType",
    "DatabaseConfig",
    "IDatabaseClientFactory",
    "DatabaseClientFactory",
    # ML Engine Factory
    "MLDeviceType",
    "MLEngineConfig",
    "IMLEngineFactory",
    "MLEngineFactory",
    # Golden Record
    "QualityTier",
    "GoldenRecord",
    "GoldenRecordBuilder",
    # System Factory
    "SystemFactory",
]
