"""
BIBLOS v2 - Structured Logging with Trace Context

Integrates structlog with OpenTelemetry trace context propagation,
ensuring all log messages include trace_id and span_id for correlation
with distributed traces.

Features:
- Structured JSON logging for log aggregation (ELK, Loki)
- Automatic trace context injection (trace_id, span_id)
- Configurable log levels and output formats
- Async-safe logging
- Request context enrichment

Usage:
    from observability.logging import setup_logging, get_logger

    # Setup at startup
    setup_logging(LoggingConfig(level="INFO", json_format=True))

    # Get logger
    logger = get_logger(__name__)
    logger.info("Processing verse", verse_id="GEN.1.1", phase="linguistic")
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, TextIO

import structlog
from structlog.types import EventDict, WrappedLogger

# Global state
_configured: bool = False


@dataclass
class LoggingConfig:
    """Configuration for structured logging."""

    service_name: str = "biblos-v2"
    level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )
    json_format: bool = field(
        default_factory=lambda: os.getenv("LOG_FORMAT", "json").lower() == "json"
    )
    enable_trace_context: bool = True
    log_to_console: bool = True
    log_to_file: bool = field(
        default_factory=lambda: os.getenv("LOG_TO_FILE", "false").lower() == "true"
    )
    log_file_path: Path = field(
        default_factory=lambda: Path(os.getenv("LOG_FILE", "./logs/biblos.log"))
    )
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    include_timestamp: bool = True
    include_caller_info: bool = True
    environment: str = field(
        default_factory=lambda: os.getenv("ENVIRONMENT", "development")
    )


def add_trace_context(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """
    Structlog processor that adds OpenTelemetry trace context to log events.

    Adds trace_id and span_id from the current span context, enabling
    correlation between logs and traces in observability backends.
    """
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span and span.is_recording():
            ctx = span.get_span_context()
            if ctx.is_valid:
                # Format as hex strings for compatibility with most backends
                event_dict["trace_id"] = format(ctx.trace_id, "032x")
                event_dict["span_id"] = format(ctx.span_id, "016x")
                event_dict["trace_flags"] = ctx.trace_flags
    except ImportError:
        pass
    except Exception:
        pass

    return event_dict


def add_service_context(
    service_name: str,
    environment: str,
) -> structlog.types.Processor:
    """
    Create a processor that adds service context to all log events.

    Args:
        service_name: Name of the service
        environment: Deployment environment
    """

    def processor(
        logger: WrappedLogger,
        method_name: str,
        event_dict: EventDict,
    ) -> EventDict:
        event_dict["service"] = service_name
        event_dict["environment"] = environment
        return event_dict

    return processor


def add_caller_info(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Add caller module and function information."""
    record = event_dict.get("_record")
    if record:
        event_dict["module"] = record.module
        event_dict["function"] = record.funcName
        event_dict["line"] = record.lineno
    return event_dict


def add_timestamp(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Add ISO8601 timestamp."""
    event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
    return event_dict


def format_exception(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Format exception information for structured output."""
    exc_info = event_dict.pop("exc_info", None)
    if exc_info:
        if isinstance(exc_info, tuple):
            event_dict["exception"] = {
                "type": exc_info[0].__name__ if exc_info[0] else None,
                "message": str(exc_info[1]) if exc_info[1] else None,
            }
        elif isinstance(exc_info, BaseException):
            event_dict["exception"] = {
                "type": type(exc_info).__name__,
                "message": str(exc_info),
            }
    return event_dict


def setup_logging(config: Optional[LoggingConfig] = None) -> None:
    """
    Configure structlog with OpenTelemetry trace context integration.

    Args:
        config: Logging configuration. Uses defaults if not provided.

    Example:
        >>> setup_logging(LoggingConfig(
        ...     level="DEBUG",
        ...     json_format=True,
        ...     enable_trace_context=True
        ... ))
    """
    global _configured

    if _configured:
        return

    config = config or LoggingConfig()

    # Build processor chain
    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
    ]

    # Add service context
    processors.append(add_service_context(config.service_name, config.environment))

    # Add timestamp
    if config.include_timestamp:
        processors.append(add_timestamp)

    # Add trace context
    if config.enable_trace_context:
        processors.append(add_trace_context)

    # Add caller info
    if config.include_caller_info:
        processors.append(add_caller_info)

    # Format exceptions
    processors.append(format_exception)

    # Add positional args
    processors.append(structlog.stdlib.PositionalArgumentsFormatter())

    # Stack info for debugging
    processors.append(structlog.processors.StackInfoRenderer())

    # Exception formatting
    processors.append(structlog.processors.format_exc_info)

    # Unicode decode errors
    processors.append(structlog.processors.UnicodeDecoder())

    # Final rendering
    if config.json_format:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
    _configure_stdlib_logging(config)

    _configured = True


def _configure_stdlib_logging(config: LoggingConfig) -> None:
    """Configure Python standard library logging."""
    handlers: list[logging.Handler] = []

    # Console handler
    if config.log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, config.level))

        if config.json_format:
            console_handler.setFormatter(_JsonFormatter())
        else:
            console_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
        handlers.append(console_handler)

    # File handler
    if config.log_to_file:
        from logging.handlers import RotatingFileHandler

        # Ensure log directory exists
        config.log_file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            config.log_file_path,
            maxBytes=config.max_file_size,
            backupCount=config.backup_count,
        )
        file_handler.setLevel(getattr(logging, config.level))
        file_handler.setFormatter(_JsonFormatter())
        handlers.append(file_handler)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.level))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add new handlers
    for handler in handlers:
        root_logger.addHandler(handler)

    # Set levels for noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("opentelemetry").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


class _JsonFormatter(logging.Formatter):
    """JSON formatter for stdlib logging."""

    def format(self, record: logging.LogRecord) -> str:
        import json

        log_record: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add trace context if available
        try:
            from opentelemetry import trace

            span = trace.get_current_span()
            if span and span.is_recording():
                ctx = span.get_span_context()
                if ctx.is_valid:
                    log_record["trace_id"] = format(ctx.trace_id, "032x")
                    log_record["span_id"] = format(ctx.span_id, "016x")
        except Exception:
            pass

        # Add exception info
        if record.exc_info:
            log_record["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info),
            }

        return json.dumps(log_record, default=str)


def configure_structlog(config: Optional[LoggingConfig] = None) -> None:
    """Alias for setup_logging for backward compatibility."""
    setup_logging(config)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structlog logger instance.

    Args:
        name: Logger name, typically __name__

    Returns:
        Bound logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started", verse_id="GEN.1.1")
        >>> logger.error("Processing failed", verse_id="GEN.1.1", error=str(e))
    """
    global _configured
    if not _configured:
        setup_logging()

    return structlog.get_logger(name)


def shutdown_logging() -> None:
    """Shutdown logging handlers."""
    global _configured

    # Flush and close all handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.flush()
        handler.close()

    _configured = False


# Context manager for request logging
class LogContext:
    """
    Context manager for adding contextual information to all logs.

    Example:
        >>> with LogContext(verse_id="GEN.1.1", phase="linguistic"):
        ...     logger.info("Starting processing")
        ...     # All logs will include verse_id and phase
    """

    def __init__(self, **kwargs: Any):
        self.context = kwargs
        self._token = None

    def __enter__(self) -> "LogContext":
        self._token = structlog.contextvars.bind_contextvars(**self.context)
        return self

    def __exit__(self, *args: Any) -> None:
        structlog.contextvars.unbind_contextvars(*self.context.keys())

    async def __aenter__(self) -> "LogContext":
        return self.__enter__()

    async def __aexit__(self, *args: Any) -> None:
        self.__exit__(*args)


def bind_context(**kwargs: Any) -> None:
    """
    Bind contextual variables to all subsequent log messages.

    Args:
        **kwargs: Key-value pairs to add to log context

    Example:
        >>> bind_context(request_id="abc123", user_id="user1")
        >>> logger.info("Request started")  # Includes request_id and user_id
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def unbind_context(*keys: str) -> None:
    """
    Remove contextual variables from log context.

    Args:
        *keys: Keys to remove from context
    """
    structlog.contextvars.unbind_contextvars(*keys)


def clear_context() -> None:
    """Clear all bound contextual variables."""
    structlog.contextvars.clear_contextvars()


# Specialized loggers for BIBLOS components
class PipelineLogger:
    """Logger specialized for pipeline operations."""

    def __init__(self):
        self._logger = get_logger("biblos.pipeline")

    def start_pipeline(self, verse_id: str, phase_count: int) -> None:
        self._logger.info(
            "Pipeline started",
            verse_id=verse_id,
            phase_count=phase_count,
            component="pipeline",
        )

    def end_pipeline(
        self,
        verse_id: str,
        status: str,
        duration: float,
        error: Optional[str] = None,
    ) -> None:
        self._logger.info(
            "Pipeline completed",
            verse_id=verse_id,
            status=status,
            duration_seconds=duration,
            error=error,
            component="pipeline",
        )

    def start_phase(self, phase_name: str, verse_id: str, agent_count: int) -> None:
        self._logger.info(
            "Phase started",
            phase=phase_name,
            verse_id=verse_id,
            agent_count=agent_count,
            component="phase",
        )

    def end_phase(
        self,
        phase_name: str,
        verse_id: str,
        status: str,
        duration: float,
    ) -> None:
        self._logger.info(
            "Phase completed",
            phase=phase_name,
            verse_id=verse_id,
            status=status,
            duration_seconds=duration,
            component="phase",
        )


class AgentLogger:
    """Logger specialized for agent operations."""

    def __init__(self, agent_name: str):
        self._logger = get_logger(f"biblos.agents.{agent_name}")
        self.agent_name = agent_name

    def start_extraction(self, verse_id: str) -> None:
        self._logger.debug(
            "Extraction started",
            agent=self.agent_name,
            verse_id=verse_id,
            component="agent",
        )

    def end_extraction(
        self,
        verse_id: str,
        status: str,
        confidence: float,
        duration: float,
    ) -> None:
        self._logger.info(
            "Extraction completed",
            agent=self.agent_name,
            verse_id=verse_id,
            status=status,
            confidence=confidence,
            duration_ms=duration,
            component="agent",
        )

    def extraction_error(self, verse_id: str, error: str) -> None:
        self._logger.error(
            "Extraction failed",
            agent=self.agent_name,
            verse_id=verse_id,
            error=error,
            component="agent",
        )


class MLLogger:
    """Logger specialized for ML operations."""

    def __init__(self):
        self._logger = get_logger("biblos.ml")

    def start_inference(self, operation: str, model: Optional[str] = None) -> None:
        self._logger.debug(
            "ML inference started",
            operation=operation,
            model=model,
            component="ml",
        )

    def end_inference(
        self,
        operation: str,
        duration: float,
        batch_size: Optional[int] = None,
    ) -> None:
        self._logger.info(
            "ML inference completed",
            operation=operation,
            duration_seconds=duration,
            batch_size=batch_size,
            component="ml",
        )

    def model_loaded(self, model_name: str, load_time: float) -> None:
        self._logger.info(
            "Model loaded",
            model=model_name,
            load_time_seconds=load_time,
            component="ml",
        )
