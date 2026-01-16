"""
BIBLOS v2 - Unified Error Handling

Provides a comprehensive error hierarchy and handling utilities
for consistent error management across the entire system.

Features:
- Hierarchical exception classes with context preservation
- Error severity levels for prioritized handling
- Structured error context for debugging
- Safe execution wrappers with automatic recovery
- OpenTelemetry integration for error tracing
"""

from __future__ import annotations

import asyncio
import functools
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    ParamSpec,
)

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

# Type variables for generic functions
T = TypeVar("T")
P = ParamSpec("P")


class ErrorSeverity(Enum):
    """Error severity levels for prioritized handling."""

    DEBUG = "debug"      # Non-critical, informational
    INFO = "info"        # Minor issue, operation continues
    WARNING = "warning"  # Potential problem, degraded operation
    ERROR = "error"      # Significant failure, operation failed
    CRITICAL = "critical"  # System-level failure, requires immediate attention
    FATAL = "fatal"      # Unrecoverable, system shutdown required


@dataclass
class ErrorContext:
    """Structured context for error debugging and tracing."""

    operation: str
    component: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    verse_id: Optional[str] = None
    agent_name: Optional[str] = None
    phase_name: Optional[str] = None
    input_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for serialization."""
        return {
            "operation": self.operation,
            "component": self.component,
            "timestamp": self.timestamp.isoformat(),
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "verse_id": self.verse_id,
            "agent_name": self.agent_name,
            "phase_name": self.phase_name,
            "input_data": self.input_data,
            "metadata": self.metadata,
            "stack_trace": self.stack_trace,
        }

    @classmethod
    def from_current_span(
        cls,
        operation: str,
        component: str,
        **kwargs: Any
    ) -> "ErrorContext":
        """Create context from current OpenTelemetry span."""
        span = trace.get_current_span()
        trace_id = None
        span_id = None

        if span and span.is_recording():
            ctx = span.get_span_context()
            if ctx.is_valid:
                trace_id = format(ctx.trace_id, "032x")
                span_id = format(ctx.span_id, "016x")

        return cls(
            operation=operation,
            component=component,
            trace_id=trace_id,
            span_id=span_id,
            stack_trace=traceback.format_exc(),
            **kwargs
        )


class BiblosError(Exception):
    """
    Base exception for all BIBLOS-specific errors.

    Provides:
    - Structured error context
    - Severity level
    - Chained exception support
    - OpenTelemetry span recording
    """

    default_severity: ErrorSeverity = ErrorSeverity.ERROR
    error_code: str = "BIBLOS_ERROR"

    def __init__(
        self,
        message: str,
        context: Optional[ErrorContext] = None,
        severity: Optional[ErrorSeverity] = None,
        cause: Optional[Exception] = None,
        recoverable: bool = False,
        suggestions: Optional[List[str]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.context = context
        self.severity = severity or self.default_severity
        self.cause = cause
        self.recoverable = recoverable
        self.suggestions = suggestions or []
        self.timestamp = datetime.now(timezone.utc)

        # Record to current span if available
        self._record_to_span()

    def _record_to_span(self) -> None:
        """Record exception to current OpenTelemetry span."""
        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_status(Status(StatusCode.ERROR, self.message))
            span.record_exception(self)
            span.set_attribute("error.code", self.error_code)
            span.set_attribute("error.severity", self.severity.value)
            span.set_attribute("error.recoverable", self.recoverable)
            if self.context:
                span.set_attribute("error.component", self.context.component)
                span.set_attribute("error.operation", self.context.operation)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "severity": self.severity.value,
            "recoverable": self.recoverable,
            "suggestions": self.suggestions,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context.to_dict() if self.context else None,
            "cause": str(self.cause) if self.cause else None,
        }

    def __str__(self) -> str:
        parts = [f"[{self.error_code}] {self.message}"]
        if self.context:
            parts.append(f" (component: {self.context.component})")
        if self.cause:
            parts.append(f" [caused by: {self.cause}]")
        return "".join(parts)

    def with_context(self, **kwargs: Any) -> "BiblosError":
        """Add additional context to the error."""
        if self.context:
            self.context.metadata.update(kwargs)
        else:
            self.context = ErrorContext(
                operation="unknown",
                component="unknown",
                metadata=kwargs
            )
        return self


class BiblosConfigError(BiblosError):
    """Configuration-related errors."""

    error_code = "CONFIG_ERROR"
    default_severity = ErrorSeverity.CRITICAL

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        expected_type: Optional[Type] = None,
        actual_value: Any = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        self.config_key = config_key
        self.expected_type = expected_type
        self.actual_value = actual_value


class BiblosDatabaseError(BiblosError):
    """Database operation errors."""

    error_code = "DATABASE_ERROR"
    default_severity = ErrorSeverity.ERROR

    def __init__(
        self,
        message: str,
        database: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        self.database = database
        self.query = query


class BiblosMLError(BiblosError):
    """Machine learning operation errors."""

    error_code = "ML_ERROR"
    default_severity = ErrorSeverity.ERROR

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        input_shape: Optional[tuple] = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        self.model_name = model_name
        self.input_shape = input_shape


class BiblosPipelineError(BiblosError):
    """Pipeline execution errors."""

    error_code = "PIPELINE_ERROR"
    default_severity = ErrorSeverity.ERROR

    def __init__(
        self,
        message: str,
        phase_name: Optional[str] = None,
        verse_id: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        self.phase_name = phase_name
        self.verse_id = verse_id


class BiblosAgentError(BiblosError):
    """Agent execution errors."""

    error_code = "AGENT_ERROR"
    default_severity = ErrorSeverity.WARNING

    def __init__(
        self,
        message: str,
        agent_name: Optional[str] = None,
        extraction_type: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        self.agent_name = agent_name
        self.extraction_type = extraction_type


class BiblosValidationError(BiblosError):
    """Data validation errors."""

    error_code = "VALIDATION_ERROR"
    default_severity = ErrorSeverity.WARNING

    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        expected_format: Optional[str] = None,
        actual_value: Any = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        self.field_name = field_name
        self.expected_format = expected_format
        self.actual_value = actual_value


class BiblosTimeoutError(BiblosError):
    """Timeout-related errors."""

    error_code = "TIMEOUT_ERROR"
    default_severity = ErrorSeverity.WARNING

    def __init__(
        self,
        message: str,
        timeout_seconds: Optional[float] = None,
        operation: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(message, recoverable=True, **kwargs)
        self.timeout_seconds = timeout_seconds
        self.operation_name = operation


class BiblosResourceError(BiblosError):
    """Resource availability errors (memory, connections, etc.)."""

    error_code = "RESOURCE_ERROR"
    default_severity = ErrorSeverity.ERROR

    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        current_usage: Optional[float] = None,
        limit: Optional[float] = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        self.resource_type = resource_type
        self.current_usage = current_usage
        self.limit = limit


# Error mapping for automatic classification
ERROR_TYPE_MAP: Dict[Type[Exception], Type[BiblosError]] = {
    asyncio.TimeoutError: BiblosTimeoutError,
    TimeoutError: BiblosTimeoutError,
    ConnectionError: BiblosDatabaseError,
    MemoryError: BiblosResourceError,
    ValueError: BiblosValidationError,
}


def classify_error(error: Exception) -> BiblosError:
    """Classify a generic exception into the appropriate BiblosError type."""
    for error_type, biblos_type in ERROR_TYPE_MAP.items():
        if isinstance(error, error_type):
            return biblos_type(
                message=str(error),
                cause=error,
            )
    return BiblosError(
        message=str(error),
        cause=error,
    )


def error_handler(
    *error_types: Type[Exception],
    default_return: Any = None,
    reraise_as: Optional[Type[BiblosError]] = None,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    log_error: bool = True,
) -> Callable[[Callable[P, T]], Callable[P, Union[T, Any]]]:
    """
    Decorator for standardized error handling.

    Args:
        error_types: Exception types to catch (default: Exception)
        default_return: Value to return on error
        reraise_as: Convert caught exceptions to this type and reraise
        severity: Severity level for logging
        log_error: Whether to log the error

    Usage:
        @error_handler(ValueError, KeyError, default_return=None)
        def my_function():
            ...
    """
    if not error_types:
        error_types = (Exception,)

    def decorator(func: Callable[P, T]) -> Callable[P, Union[T, Any]]:
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> Union[T, Any]:
            try:
                return func(*args, **kwargs)
            except error_types as e:
                if log_error:
                    # Log would go here
                    pass
                if reraise_as:
                    raise reraise_as(
                        message=str(e),
                        cause=e,
                        severity=severity,
                    ) from e
                return default_return

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Union[T, Any]:
            try:
                return await func(*args, **kwargs)
            except error_types as e:
                if log_error:
                    # Log would go here
                    pass
                if reraise_as:
                    raise reraise_as(
                        message=str(e),
                        cause=e,
                        severity=severity,
                    ) from e
                return default_return

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


async def safe_execute(
    func: Callable[..., T],
    *args: Any,
    default: Optional[T] = None,
    error_handler: Optional[Callable[[Exception], T]] = None,
    **kwargs: Any,
) -> T:
    """
    Safely execute a function with automatic error handling.

    Args:
        func: Function to execute
        *args: Positional arguments
        default: Default value on error
        error_handler: Custom error handler
        **kwargs: Keyword arguments

    Returns:
        Function result or default value
    """
    try:
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        return func(*args, **kwargs)
    except Exception as e:
        if error_handler:
            return error_handler(e)
        return default  # type: ignore
