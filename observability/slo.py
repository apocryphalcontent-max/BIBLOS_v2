"""
BIBLOS v2 - SLI/SLO Tracking

Service Level Indicator (SLI) and Service Level Objective (SLO) tracking
for monitoring system performance against defined targets.

Features:
- Define SLOs for latency, availability, and throughput
- Track SLI metrics in real-time
- Error budget calculation
- Alerting threshold support
- SLO compliance reporting

Usage:
    from observability.slo import slo_tracker, SLO, SLOType

    # Define an SLO
    slo_tracker.define_slo(SLO(
        name="pipeline_latency_p99",
        slo_type=SLOType.LATENCY,
        target=5.0,
        window_seconds=3600,
    ))

    # Record observations
    slo_tracker.record_latency("pipeline_latency_p99", 1.25)

    # Check compliance
    status = slo_tracker.get_slo_status("pipeline_latency_p99")
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from threading import Lock
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import structlog

logger = structlog.get_logger("biblos.slo")


class SLOType(Enum):
    """Types of Service Level Objectives."""

    LATENCY = "latency"  # Response time percentile
    AVAILABILITY = "availability"  # Uptime percentage
    THROUGHPUT = "throughput"  # Requests per second
    ERROR_RATE = "error_rate"  # Percentage of errors
    SUCCESS_RATE = "success_rate"  # Percentage of successes


class SLOStatus(Enum):
    """SLO compliance status."""

    HEALTHY = "healthy"  # Within budget
    WARNING = "warning"  # Approaching budget exhaustion
    CRITICAL = "critical"  # Budget exhausted or exceeded
    UNKNOWN = "unknown"  # Insufficient data


@dataclass
class SLO:
    """Service Level Objective definition."""

    name: str
    slo_type: SLOType
    target: float  # Target value (e.g., 99.9 for availability, 5.0 for latency)
    window_seconds: int = 3600  # Time window for calculation (1 hour default)
    percentile: float = 0.99  # For latency SLOs
    warning_threshold: float = 0.8  # Warning when 80% of budget consumed
    description: str = ""

    def __post_init__(self) -> None:
        if self.target <= 0:
            raise ValueError("SLO target must be positive")
        if self.window_seconds <= 0:
            raise ValueError("Window must be positive")


@dataclass
class SLOStatusReport:
    """Current status of an SLO."""

    slo_name: str
    status: SLOStatus
    current_value: float
    target_value: float
    compliance_percentage: float
    error_budget_remaining: float
    window_start: datetime
    window_end: datetime
    observation_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "slo_name": self.slo_name,
            "status": self.status.value,
            "current_value": self.current_value,
            "target_value": self.target_value,
            "compliance_percentage": self.compliance_percentage,
            "error_budget_remaining": self.error_budget_remaining,
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "observation_count": self.observation_count,
            "metadata": self.metadata,
        }


@dataclass
class Observation:
    """Single observation for SLI tracking."""

    timestamp: float
    value: float
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class SLOTracker:
    """
    Tracks Service Level Indicators and calculates SLO compliance.

    Maintains sliding window of observations for each SLO and
    calculates current status in real-time.
    """

    def __init__(self) -> None:
        self._slos: Dict[str, SLO] = {}
        self._observations: Dict[str, Deque[Observation]] = {}
        self._lock = Lock()
        self._alert_callbacks: List[Callable[[str, SLOStatus, SLOStatusReport], None]] = []

    def define_slo(self, slo: SLO) -> None:
        """
        Define a new SLO to track.

        Args:
            slo: SLO definition
        """
        with self._lock:
            self._slos[slo.name] = slo
            self._observations[slo.name] = deque()
            logger.info(
                "SLO defined",
                slo_name=slo.name,
                slo_type=slo.slo_type.value,
                target=slo.target,
                window_seconds=slo.window_seconds,
            )

    def remove_slo(self, name: str) -> None:
        """Remove an SLO."""
        with self._lock:
            self._slos.pop(name, None)
            self._observations.pop(name, None)

    def record_latency(
        self,
        slo_name: str,
        latency_seconds: float,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a latency observation.

        Args:
            slo_name: Name of the SLO
            latency_seconds: Observed latency
            success: Whether the request succeeded
            metadata: Additional context
        """
        self._record(slo_name, latency_seconds, success, metadata)

    def record_success(
        self,
        slo_name: str,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a success/failure observation for availability/error rate SLOs.

        Args:
            slo_name: Name of the SLO
            success: Whether the operation succeeded
            metadata: Additional context
        """
        self._record(slo_name, 1.0 if success else 0.0, success, metadata)

    def record_count(
        self,
        slo_name: str,
        count: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a count observation for throughput SLOs.

        Args:
            slo_name: Name of the SLO
            count: Number of items processed
            metadata: Additional context
        """
        self._record(slo_name, float(count), True, metadata)

    def _record(
        self,
        slo_name: str,
        value: float,
        success: bool,
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        """Record an observation."""
        with self._lock:
            if slo_name not in self._slos:
                return

            obs = Observation(
                timestamp=time.time(),
                value=value,
                success=success,
                metadata=metadata or {},
            )
            self._observations[slo_name].append(obs)

            # Cleanup old observations
            self._cleanup_observations(slo_name)

            # Check for status change and alert
            self._check_alerts(slo_name)

    def _cleanup_observations(self, slo_name: str) -> None:
        """Remove observations outside the window."""
        slo = self._slos[slo_name]
        cutoff = time.time() - slo.window_seconds
        observations = self._observations[slo_name]

        while observations and observations[0].timestamp < cutoff:
            observations.popleft()

    def _check_alerts(self, slo_name: str) -> None:
        """Check SLO status and trigger alerts if needed."""
        status_report = self._calculate_status(slo_name)
        if status_report.status in (SLOStatus.WARNING, SLOStatus.CRITICAL):
            for callback in self._alert_callbacks:
                try:
                    callback(slo_name, status_report.status, status_report)
                except Exception as e:
                    logger.error("Alert callback failed", error=str(e))

    def get_slo_status(self, slo_name: str) -> SLOStatusReport:
        """
        Get current status of an SLO.

        Args:
            slo_name: Name of the SLO

        Returns:
            Current SLO status report
        """
        with self._lock:
            return self._calculate_status(slo_name)

    def _calculate_status(self, slo_name: str) -> SLOStatusReport:
        """Calculate current SLO status."""
        if slo_name not in self._slos:
            return SLOStatusReport(
                slo_name=slo_name,
                status=SLOStatus.UNKNOWN,
                current_value=0.0,
                target_value=0.0,
                compliance_percentage=0.0,
                error_budget_remaining=0.0,
                window_start=datetime.now(timezone.utc),
                window_end=datetime.now(timezone.utc),
                observation_count=0,
            )

        slo = self._slos[slo_name]
        observations = list(self._observations[slo_name])

        if not observations:
            return SLOStatusReport(
                slo_name=slo_name,
                status=SLOStatus.UNKNOWN,
                current_value=0.0,
                target_value=slo.target,
                compliance_percentage=0.0,
                error_budget_remaining=100.0,
                window_start=datetime.now(timezone.utc) - timedelta(seconds=slo.window_seconds),
                window_end=datetime.now(timezone.utc),
                observation_count=0,
            )

        # Calculate current value based on SLO type
        current_value = self._calculate_current_value(slo, observations)

        # Calculate compliance
        compliance = self._calculate_compliance(slo, current_value)

        # Calculate error budget remaining
        error_budget = self._calculate_error_budget(slo, compliance)

        # Determine status
        status = self._determine_status(slo, compliance, error_budget)

        return SLOStatusReport(
            slo_name=slo_name,
            status=status,
            current_value=current_value,
            target_value=slo.target,
            compliance_percentage=compliance,
            error_budget_remaining=error_budget,
            window_start=datetime.fromtimestamp(observations[0].timestamp, timezone.utc),
            window_end=datetime.fromtimestamp(observations[-1].timestamp, timezone.utc),
            observation_count=len(observations),
        )

    def _calculate_current_value(
        self,
        slo: SLO,
        observations: List[Observation],
    ) -> float:
        """Calculate current SLI value."""
        if not observations:
            return 0.0

        if slo.slo_type == SLOType.LATENCY:
            # Calculate percentile latency
            values = sorted(obs.value for obs in observations)
            index = int(len(values) * slo.percentile)
            return values[min(index, len(values) - 1)]

        elif slo.slo_type in (SLOType.AVAILABILITY, SLOType.SUCCESS_RATE):
            # Calculate success rate
            successes = sum(1 for obs in observations if obs.success)
            return (successes / len(observations)) * 100.0

        elif slo.slo_type == SLOType.ERROR_RATE:
            # Calculate error rate
            failures = sum(1 for obs in observations if not obs.success)
            return (failures / len(observations)) * 100.0

        elif slo.slo_type == SLOType.THROUGHPUT:
            # Calculate throughput (sum of counts / window duration)
            total = sum(obs.value for obs in observations)
            duration = observations[-1].timestamp - observations[0].timestamp
            if duration > 0:
                return total / duration
            return total

        return 0.0

    def _calculate_compliance(self, slo: SLO, current_value: float) -> float:
        """Calculate compliance percentage."""
        if slo.slo_type == SLOType.LATENCY:
            # For latency, lower is better
            if current_value <= slo.target:
                return 100.0
            return max(0.0, (slo.target / current_value) * 100.0)

        elif slo.slo_type in (SLOType.AVAILABILITY, SLOType.SUCCESS_RATE):
            # For availability/success rate, current value is already a percentage
            return min(100.0, (current_value / slo.target) * 100.0)

        elif slo.slo_type == SLOType.ERROR_RATE:
            # For error rate, lower is better
            if current_value <= slo.target:
                return 100.0
            return max(0.0, (slo.target / current_value) * 100.0)

        elif slo.slo_type == SLOType.THROUGHPUT:
            # For throughput, higher is better
            return min(100.0, (current_value / slo.target) * 100.0)

        return 0.0

    def _calculate_error_budget(self, slo: SLO, compliance: float) -> float:
        """Calculate remaining error budget."""
        if slo.slo_type in (SLOType.AVAILABILITY, SLOType.SUCCESS_RATE):
            # Error budget is 100% - target
            # e.g., 99.9% availability = 0.1% error budget
            error_budget_total = 100.0 - slo.target
            if error_budget_total <= 0:
                return 0.0
            used = max(0.0, 100.0 - compliance)
            return max(0.0, 100.0 - (used / error_budget_total * 100.0))

        # For other types, use compliance as proxy
        return compliance

    def _determine_status(
        self,
        slo: SLO,
        compliance: float,
        error_budget: float,
    ) -> SLOStatus:
        """Determine SLO status based on compliance and error budget."""
        if compliance >= 100.0:
            return SLOStatus.HEALTHY
        elif error_budget <= 0:
            return SLOStatus.CRITICAL
        elif error_budget <= (1.0 - slo.warning_threshold) * 100.0:
            return SLOStatus.WARNING
        else:
            return SLOStatus.HEALTHY

    def get_all_slo_status(self) -> Dict[str, SLOStatusReport]:
        """Get status of all defined SLOs."""
        with self._lock:
            return {
                name: self._calculate_status(name)
                for name in self._slos
            }

    def register_alert_callback(
        self,
        callback: Callable[[str, SLOStatus, SLOStatusReport], None],
    ) -> None:
        """
        Register a callback for SLO alerts.

        Args:
            callback: Function called when SLO status changes to warning/critical
        """
        self._alert_callbacks.append(callback)

    def get_slo_summary(self) -> Dict[str, Any]:
        """Get summary of all SLOs for dashboards."""
        all_status = self.get_all_slo_status()

        healthy = sum(1 for s in all_status.values() if s.status == SLOStatus.HEALTHY)
        warning = sum(1 for s in all_status.values() if s.status == SLOStatus.WARNING)
        critical = sum(1 for s in all_status.values() if s.status == SLOStatus.CRITICAL)
        unknown = sum(1 for s in all_status.values() if s.status == SLOStatus.UNKNOWN)

        return {
            "total_slos": len(all_status),
            "healthy": healthy,
            "warning": warning,
            "critical": critical,
            "unknown": unknown,
            "overall_status": (
                "critical" if critical > 0
                else "warning" if warning > 0
                else "healthy" if healthy > 0
                else "unknown"
            ),
            "slos": {name: status.to_dict() for name, status in all_status.items()},
        }


# Global SLO tracker instance
_slo_tracker: Optional[SLOTracker] = None


def get_slo_tracker() -> SLOTracker:
    """Get or create global SLO tracker."""
    global _slo_tracker
    if _slo_tracker is None:
        _slo_tracker = SLOTracker()
    return _slo_tracker


# Convenience alias
slo_tracker = get_slo_tracker()


# Default BIBLOS SLOs
def setup_default_slos() -> None:
    """Set up default SLOs for BIBLOS system."""
    tracker = get_slo_tracker()

    # Pipeline latency SLO - 99th percentile under 5 seconds
    tracker.define_slo(SLO(
        name="pipeline_latency_p99",
        slo_type=SLOType.LATENCY,
        target=5.0,
        percentile=0.99,
        window_seconds=3600,
        description="99th percentile pipeline execution latency",
    ))

    # Pipeline availability SLO - 99.9% success rate
    tracker.define_slo(SLO(
        name="pipeline_availability",
        slo_type=SLOType.AVAILABILITY,
        target=99.9,
        window_seconds=86400,  # 24 hours
        description="Pipeline execution success rate",
    ))

    # API latency SLO - 99th percentile under 1 second
    tracker.define_slo(SLO(
        name="api_latency_p99",
        slo_type=SLOType.LATENCY,
        target=1.0,
        percentile=0.99,
        window_seconds=3600,
        description="99th percentile API response latency",
    ))

    # Agent error rate SLO - under 5% errors
    tracker.define_slo(SLO(
        name="agent_error_rate",
        slo_type=SLOType.ERROR_RATE,
        target=5.0,
        window_seconds=3600,
        description="Agent extraction error rate",
    ))

    # ML inference latency SLO - 99th percentile under 500ms
    tracker.define_slo(SLO(
        name="ml_inference_latency_p99",
        slo_type=SLOType.LATENCY,
        target=0.5,
        percentile=0.99,
        window_seconds=3600,
        description="99th percentile ML inference latency",
    ))

    logger.info("Default SLOs configured", slo_count=len(tracker._slos))
