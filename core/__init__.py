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
    with_retry,
    with_circuit_breaker,
    with_bulkhead,
    resilient,
)
from core.async_utils import (
    AsyncTaskGroup,
    AsyncBatcher,
    AsyncThrottler,
    gather_with_concurrency,
    timeout_with_cleanup,
    cancel_scope,
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
    "with_retry",
    "with_circuit_breaker",
    "with_bulkhead",
    "resilient",
    # Async
    "AsyncTaskGroup",
    "AsyncBatcher",
    "AsyncThrottler",
    "gather_with_concurrency",
    "timeout_with_cleanup",
    "cancel_scope",
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
]
