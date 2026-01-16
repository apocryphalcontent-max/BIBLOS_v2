"""
BIBLOS v2 - Audit Logging

Security and compliance audit logging for tracking sensitive operations,
data access, and system changes.

Features:
- Immutable audit log entries
- Data access tracking
- Configuration change logging
- Security event tracking
- Compliance reporting support (SOC2, GDPR)

Usage:
    from observability.audit import audit_logger, AuditEvent

    # Log data access
    audit_logger.log_data_access(
        user_id="system",
        resource="verses",
        action="read",
        verse_ids=["GEN.1.1", "GEN.1.2"],
    )
"""
from __future__ import annotations

import json
import os
import hashlib
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog


class AuditAction(Enum):
    """Types of auditable actions."""

    # Data operations
    DATA_READ = "data_read"
    DATA_WRITE = "data_write"
    DATA_DELETE = "data_delete"
    DATA_EXPORT = "data_export"

    # Configuration
    CONFIG_CHANGE = "config_change"
    CONFIG_VIEW = "config_view"

    # Authentication/Authorization
    AUTH_LOGIN = "auth_login"
    AUTH_LOGOUT = "auth_logout"
    AUTH_FAILED = "auth_failed"
    AUTH_TOKEN_ISSUED = "auth_token_issued"
    AUTH_PERMISSION_DENIED = "auth_permission_denied"

    # System operations
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    SYSTEM_ERROR = "system_error"

    # Pipeline operations
    PIPELINE_START = "pipeline_start"
    PIPELINE_COMPLETE = "pipeline_complete"
    PIPELINE_FAILED = "pipeline_failed"

    # API operations
    API_REQUEST = "api_request"
    API_ERROR = "api_error"


class AuditSeverity(Enum):
    """Severity levels for audit events."""

    INFO = "info"
    WARNING = "warning"
    ALERT = "alert"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Immutable audit log entry."""

    event_id: str
    timestamp: str
    action: str
    severity: str
    actor_id: str
    actor_type: str
    resource_type: str
    resource_id: Optional[str]
    description: str
    metadata: Dict[str, Any]
    source_ip: Optional[str]
    user_agent: Optional[str]
    trace_id: Optional[str]
    span_id: Optional[str]
    checksum: str = field(init=False)

    def __post_init__(self) -> None:
        """Calculate checksum for integrity verification."""
        data = f"{self.event_id}:{self.timestamp}:{self.action}:{self.actor_id}:{self.resource_type}"
        self.checksum = hashlib.sha256(data.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class AuditLogger:
    """
    Secure audit logger for compliance and security tracking.

    Provides immutable, timestamped records of sensitive operations
    with integrity checksums.
    """

    def __init__(
        self,
        service_name: str = "biblos-v2",
        log_to_file: bool = True,
        log_file_path: Optional[Path] = None,
    ):
        self.service_name = service_name
        self.log_to_file = log_to_file
        self.log_file_path = log_file_path or Path("./logs/audit.log")
        self._logger = structlog.get_logger("biblos.audit")

        if self.log_to_file:
            self.log_file_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_trace_context(self) -> tuple[Optional[str], Optional[str]]:
        """Get current trace context if available."""
        try:
            from opentelemetry import trace

            span = trace.get_current_span()
            if span and span.is_recording():
                ctx = span.get_span_context()
                if ctx.is_valid:
                    return (
                        format(ctx.trace_id, "032x"),
                        format(ctx.span_id, "016x"),
                    )
        except Exception:
            pass
        return None, None

    def log(
        self,
        action: AuditAction,
        actor_id: str,
        actor_type: str = "system",
        resource_type: str = "unknown",
        resource_id: Optional[str] = None,
        description: str = "",
        severity: AuditSeverity = AuditSeverity.INFO,
        metadata: Optional[Dict[str, Any]] = None,
        source_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> AuditEvent:
        """
        Log an audit event.

        Args:
            action: Type of action being audited
            actor_id: ID of the entity performing the action
            actor_type: Type of actor (user, system, api)
            resource_type: Type of resource being accessed
            resource_id: Specific resource identifier
            description: Human-readable description
            severity: Severity level
            metadata: Additional context
            source_ip: Source IP address
            user_agent: User agent string

        Returns:
            Created AuditEvent
        """
        trace_id, span_id = self._get_trace_context()

        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            action=action.value,
            severity=severity.value,
            actor_id=actor_id,
            actor_type=actor_type,
            resource_type=resource_type,
            resource_id=resource_id,
            description=description,
            metadata=metadata or {},
            source_ip=source_ip,
            user_agent=user_agent,
            trace_id=trace_id,
            span_id=span_id,
        )

        # Log to structlog
        self._logger.info(
            "audit_event",
            **event.to_dict(),
        )

        # Append to audit file
        if self.log_to_file:
            self._write_to_file(event)

        return event

    def _write_to_file(self, event: AuditEvent) -> None:
        """Write audit event to file."""
        try:
            with open(self.log_file_path, "a") as f:
                f.write(event.to_json() + "\n")
        except Exception as e:
            self._logger.error("Failed to write audit log", error=str(e))

    # Convenience methods for common operations

    def log_data_access(
        self,
        actor_id: str,
        resource: str,
        action: str = "read",
        verse_ids: Optional[List[str]] = None,
        query: Optional[str] = None,
    ) -> AuditEvent:
        """Log data access operation."""
        audit_action = (
            AuditAction.DATA_READ if action == "read"
            else AuditAction.DATA_WRITE if action == "write"
            else AuditAction.DATA_DELETE
        )

        return self.log(
            action=audit_action,
            actor_id=actor_id,
            resource_type=resource,
            description=f"Data {action} on {resource}",
            metadata={
                "verse_ids": verse_ids,
                "query": query,
                "record_count": len(verse_ids) if verse_ids else None,
            },
        )

    def log_config_change(
        self,
        actor_id: str,
        config_key: str,
        old_value: Any,
        new_value: Any,
    ) -> AuditEvent:
        """Log configuration change."""
        return self.log(
            action=AuditAction.CONFIG_CHANGE,
            actor_id=actor_id,
            resource_type="configuration",
            resource_id=config_key,
            description=f"Configuration change: {config_key}",
            severity=AuditSeverity.WARNING,
            metadata={
                "old_value": str(old_value),
                "new_value": str(new_value),
            },
        )

    def log_pipeline_execution(
        self,
        actor_id: str,
        verse_id: str,
        status: str,
        duration: Optional[float] = None,
        error: Optional[str] = None,
    ) -> AuditEvent:
        """Log pipeline execution."""
        action = (
            AuditAction.PIPELINE_COMPLETE if status == "completed"
            else AuditAction.PIPELINE_FAILED if status == "failed"
            else AuditAction.PIPELINE_START
        )

        severity = (
            AuditSeverity.ALERT if status == "failed"
            else AuditSeverity.INFO
        )

        return self.log(
            action=action,
            actor_id=actor_id,
            resource_type="verse",
            resource_id=verse_id,
            description=f"Pipeline {status} for {verse_id}",
            severity=severity,
            metadata={
                "duration_seconds": duration,
                "error": error,
            },
        )

    def log_api_request(
        self,
        actor_id: str,
        endpoint: str,
        method: str,
        status_code: int,
        duration: float,
        source_ip: Optional[str] = None,
    ) -> AuditEvent:
        """Log API request."""
        severity = (
            AuditSeverity.ALERT if status_code >= 500
            else AuditSeverity.WARNING if status_code >= 400
            else AuditSeverity.INFO
        )

        action = (
            AuditAction.API_ERROR if status_code >= 400
            else AuditAction.API_REQUEST
        )

        return self.log(
            action=action,
            actor_id=actor_id,
            resource_type="api",
            resource_id=endpoint,
            description=f"{method} {endpoint} -> {status_code}",
            severity=severity,
            source_ip=source_ip,
            metadata={
                "method": method,
                "status_code": status_code,
                "duration_seconds": duration,
            },
        )

    def log_security_event(
        self,
        actor_id: str,
        event_type: str,
        description: str,
        severity: AuditSeverity = AuditSeverity.ALERT,
        source_ip: Optional[str] = None,
    ) -> AuditEvent:
        """Log security-related event."""
        action = (
            AuditAction.AUTH_FAILED if "failed" in event_type.lower()
            else AuditAction.AUTH_PERMISSION_DENIED if "denied" in event_type.lower()
            else AuditAction.AUTH_LOGIN
        )

        return self.log(
            action=action,
            actor_id=actor_id,
            resource_type="security",
            description=description,
            severity=severity,
            source_ip=source_ip,
            metadata={"event_type": event_type},
        )


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get or create global audit logger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


# Convenience alias
audit_logger = get_audit_logger()
