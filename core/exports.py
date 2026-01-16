"""
BIBLOS v2 - Core Exports Module

Central export point for all core functionality. Import from this module
to access error types, resilience patterns, async utilities, and configuration
validators with a single import statement.

Usage:
    from core.exports import (
        # Errors
        BiblosError, BiblosDatabaseError, BiblosMLError, BiblosPipelineError,
        BiblosAgentError, BiblosValidationError, BiblosTimeoutError,
        BiblosResourceError, BiblosConfigError,
        # Resilience
        CircuitBreaker, RetryPolicy, Bulkhead, resilient,
        # Async utilities
        AsyncTaskGroup, AsyncBatcher, AsyncThrottler,
        gather_with_concurrency, timeout_with_cleanup,
        # Config validation
        ConfigValidator, FieldValidator,
    )
"""

# Re-export all error types
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
    ErrorCode,
    ErrorSeverity,
    ErrorContext,
    error_handler,
    safe_execute,
)

# Re-export resilience patterns
from core.resilience import (
    CircuitState,
    CircuitBreaker,
    RetryPolicy,
    Bulkhead,
    resilient,
    with_circuit_breaker,
    with_retry,
    with_bulkhead,
)

# Re-export async utilities
from core.async_utils import (
    AsyncTaskGroup,
    AsyncBatcher,
    AsyncThrottler,
    gather_with_concurrency,
    timeout_with_cleanup,
    cancel_scope,
    run_with_timeout,
    debounce_async,
    throttle_async,
)

# Re-export configuration validation
from core.config_validator import (
    FieldValidator,
    ConfigValidator,
    ValidationResult,
    DatabaseConfigValidator,
    MLConfigValidator,
    APIConfigValidator,
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
    "ErrorCode",
    "ErrorSeverity",
    "ErrorContext",
    "error_handler",
    "safe_execute",
    # Resilience
    "CircuitState",
    "CircuitBreaker",
    "RetryPolicy",
    "Bulkhead",
    "resilient",
    "with_circuit_breaker",
    "with_retry",
    "with_bulkhead",
    # Async utilities
    "AsyncTaskGroup",
    "AsyncBatcher",
    "AsyncThrottler",
    "gather_with_concurrency",
    "timeout_with_cleanup",
    "cancel_scope",
    "run_with_timeout",
    "debounce_async",
    "throttle_async",
    # Config validation
    "FieldValidator",
    "ConfigValidator",
    "ValidationResult",
    "DatabaseConfigValidator",
    "MLConfigValidator",
    "APIConfigValidator",
]
