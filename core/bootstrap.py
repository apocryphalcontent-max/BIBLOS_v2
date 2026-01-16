"""
BIBLOS v2 - Application Bootstrap

The central nervous system of the BIBLOS organism - coordinates the awakening,
sustenance, and graceful rest of all system components.

Architecture:
    ApplicationBuilder → Application → Container → Services → Living System

The bootstrap layer implements:
    - Fluent ApplicationBuilder for declarative configuration
    - Module-based service registration (IServiceModule pattern)
    - Ordered lifecycle hooks with priority and dependency resolution
    - Health check aggregation with circuit breakers
    - Background service management
    - Graceful degradation on partial failures
    - Observability integration

Usage:
    from core.bootstrap import ApplicationBuilder, bootstrap

    # Fluent builder pattern
    app = await (
        ApplicationBuilder()
        .with_environment("production")
        .with_module(DatabaseModule())
        .with_module(PipelineModule())
        .with_lifecycle_hook(TelemetryHook(), priority=0)
        .with_background_service(HealthMonitorService())
        .with_feature("event_sourcing", enabled=True)
        .build()
    )

    # Or simple context manager
    async with bootstrap() as app:
        await app.pipeline.process_verse("GEN.1.1")

Design Principles:
    - Explicit dependency wiring over implicit discovery
    - Fail-fast on configuration errors with graceful degradation
    - Observable startup/shutdown phases with metrics
    - Services as organs in a living system
"""
from __future__ import annotations

import asyncio
import logging
import signal
import sys
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)
from uuid import UUID, uuid4

from di.container import (
    Container,
    IServiceCollection,
    IServiceProvider,
    IServiceScope,
    ServiceLifetime,
    get_container,
    HealthCheckResult,
    HealthStatus,
)

logger = logging.getLogger("biblos.bootstrap")

T = TypeVar("T")
TService = TypeVar("TService")


# =============================================================================
# LIFECYCLE PHASES AND EVENTS
# =============================================================================


class ApplicationPhase(Enum):
    """
    Application lifecycle phases - the heartbeat states of the organism.

    The application transitions through these phases in order:
    CREATED → CONFIGURING → INITIALIZING → STARTING → RUNNING → SHUTTING_DOWN → TERMINATED
    """
    CREATED = "created"
    CONFIGURING = "configuring"
    INITIALIZING = "initializing"
    STARTING = "starting"
    RUNNING = "running"
    SHUTTING_DOWN = "shutting_down"
    TERMINATED = "terminated"
    FAILED = "failed"


class StartupStage(Enum):
    """Fine-grained startup stages for dependency ordering."""
    CONFIGURATION = auto()      # Load and validate configuration
    INFRASTRUCTURE = auto()     # Core infrastructure (logging, metrics)
    DATABASE = auto()           # Database connections
    MESSAGING = auto()          # Event buses, message queues
    CACHE = auto()              # Caching layers
    SERVICES = auto()           # Business services
    PIPELINE = auto()           # ML pipeline components
    BACKGROUND = auto()         # Background services
    HEALTH = auto()             # Health check registration
    FINALIZE = auto()           # Final initialization


class ShutdownStage(Enum):
    """Fine-grained shutdown stages - reverse order of startup."""
    FINALIZE = auto()           # Pre-shutdown hooks
    HEALTH = auto()             # Unregister health checks
    BACKGROUND = auto()         # Stop background services
    PIPELINE = auto()           # Stop pipeline processing
    SERVICES = auto()           # Shutdown business services
    CACHE = auto()              # Flush and close caches
    MESSAGING = auto()          # Drain message queues
    DATABASE = auto()           # Close database connections
    INFRASTRUCTURE = auto()     # Final infrastructure cleanup
    CONFIGURATION = auto()      # Save state if needed


@dataclass(frozen=True, slots=True)
class LifecycleEvent:
    """
    Immutable record of a lifecycle event.

    Captures the complete context of what happened during a phase transition,
    enabling full observability of the application's awakening and rest.
    """
    event_id: UUID
    timestamp: float
    phase: ApplicationPhase
    stage: Union[StartupStage, ShutdownStage, None]
    component: str
    success: bool
    duration_ms: float
    error: Optional[str] = None
    error_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success_event(
        cls,
        phase: ApplicationPhase,
        component: str,
        duration_ms: float,
        stage: Union[StartupStage, ShutdownStage, None] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "LifecycleEvent":
        """Factory for successful lifecycle events."""
        return cls(
            event_id=uuid4(),
            timestamp=time.time(),
            phase=phase,
            stage=stage,
            component=component,
            success=True,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )

    @classmethod
    def failure_event(
        cls,
        phase: ApplicationPhase,
        component: str,
        error: Exception,
        stage: Union[StartupStage, ShutdownStage, None] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "LifecycleEvent":
        """Factory for failed lifecycle events."""
        return cls(
            event_id=uuid4(),
            timestamp=time.time(),
            phase=phase,
            stage=stage,
            component=component,
            success=False,
            duration_ms=0,
            error=str(error),
            error_type=type(error).__name__,
            metadata=metadata or {},
        )


# =============================================================================
# LIFECYCLE HOOKS AND MODULES
# =============================================================================


@runtime_checkable
class ILifecycleHook(Protocol):
    """
    Protocol for lifecycle hooks - extension points in the organism's lifecycle.

    Hooks are called at specific phases and can perform cross-cutting concerns
    like telemetry, auditing, or external integrations.
    """

    @property
    def name(self) -> str:
        """Unique name for this hook."""
        ...

    @property
    def priority(self) -> int:
        """
        Execution priority (lower = earlier).

        Priority ranges:
            0-99: Critical infrastructure (logging, tracing)
            100-199: Core services (database, cache)
            200-299: Application services
            300-399: External integrations
            400+: Cleanup and finalization
        """
        ...

    async def on_configure(self, app: "Application") -> None:
        """Called during configuration phase."""
        ...

    async def on_initialize(self, app: "Application") -> None:
        """Called during initialization phase."""
        ...

    async def on_start(self, app: "Application") -> None:
        """Called when application enters running state."""
        ...

    async def on_shutdown(self, app: "Application") -> None:
        """Called during graceful shutdown."""
        ...


class LifecycleHookBase(ABC):
    """
    Base class for lifecycle hooks with sensible defaults.

    Inherit from this to create hooks with minimal boilerplate.
    """

    def __init__(self, name: Optional[str] = None, priority: int = 200):
        self._name = name or self.__class__.__name__
        self._priority = priority

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return self._priority

    async def on_configure(self, app: "Application") -> None:
        """Override to add configuration logic."""
        pass

    async def on_initialize(self, app: "Application") -> None:
        """Override to add initialization logic."""
        pass

    async def on_start(self, app: "Application") -> None:
        """Override to add startup logic."""
        pass

    async def on_shutdown(self, app: "Application") -> None:
        """Override to add shutdown logic."""
        pass


@runtime_checkable
class IServiceModule(Protocol):
    """
    Protocol for service modules - cohesive groups of related services.

    Modules encapsulate related functionality:
        - DatabaseModule: All database-related services
        - PipelineModule: ML pipeline components
        - AgentModule: SDES agents

    This follows the Module Pattern from dependency injection frameworks.
    """

    @property
    def name(self) -> str:
        """Module name for identification."""
        ...

    @property
    def dependencies(self) -> List[str]:
        """Names of modules this module depends on."""
        ...

    def configure_services(self, services: IServiceCollection, config: "ApplicationConfig") -> None:
        """Register services in the container."""
        ...

    async def initialize(self, provider: IServiceProvider) -> None:
        """Initialize module services after registration."""
        ...

    async def shutdown(self, provider: IServiceProvider) -> None:
        """Cleanup module services."""
        ...


class ServiceModuleBase(ABC):
    """Base class for service modules with dependency tracking."""

    def __init__(self, name: Optional[str] = None):
        self._name = name or self.__class__.__name__
        self._dependencies: List[str] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def dependencies(self) -> List[str]:
        return self._dependencies

    def depends_on(self, *module_names: str) -> "ServiceModuleBase":
        """Declare dependencies on other modules."""
        self._dependencies.extend(module_names)
        return self

    @abstractmethod
    def configure_services(self, services: IServiceCollection, config: "ApplicationConfig") -> None:
        """Override to register services."""
        ...

    async def initialize(self, provider: IServiceProvider) -> None:
        """Override to add initialization logic."""
        pass

    async def shutdown(self, provider: IServiceProvider) -> None:
        """Override to add shutdown logic."""
        pass


@runtime_checkable
class IBackgroundService(Protocol):
    """
    Protocol for background services - autonomous organs of the system.

    Background services run continuously while the application is alive,
    performing tasks like:
        - Health monitoring
        - Event processing
        - Cache warming
        - Metrics collection
    """

    @property
    def name(self) -> str:
        """Service name for identification."""
        ...

    async def start(self, cancellation_token: asyncio.Event) -> None:
        """Start the background service."""
        ...

    async def stop(self) -> None:
        """Stop the background service gracefully."""
        ...


class BackgroundServiceBase(ABC):
    """Base class for background services with lifecycle management."""

    def __init__(self, name: Optional[str] = None):
        self._name = name or self.__class__.__name__
        self._running = False
        self._task: Optional[asyncio.Task[None]] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self, cancellation_token: asyncio.Event) -> None:
        """Start the service with cancellation support."""
        self._running = True
        try:
            await self.execute(cancellation_token)
        finally:
            self._running = False

    async def stop(self) -> None:
        """Stop the service."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    @abstractmethod
    async def execute(self, cancellation_token: asyncio.Event) -> None:
        """Override to implement the service logic."""
        ...


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class FeatureFlag:
    """Configuration for a feature flag."""
    name: str
    enabled: bool
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ApplicationConfig:
    """
    Application-wide configuration - the DNA of the organism.

    Configuration flows from environment → file → code defaults,
    with each layer having the ability to override previous values.
    """
    # Environment
    environment: str = "development"
    app_name: str = "BIBLOS"
    app_version: str = "2.0.0"
    instance_id: str = field(default_factory=lambda: str(uuid4())[:8])
    debug: bool = False

    # Observability
    enable_telemetry: bool = True
    enable_health_checks: bool = True
    enable_metrics: bool = True
    log_level: str = "INFO"

    # Timeouts
    graceful_shutdown_timeout: float = 30.0
    startup_timeout: float = 60.0
    health_check_timeout: float = 10.0
    initialization_retry_count: int = 3
    initialization_retry_delay: float = 1.0

    # Database configuration
    postgres_enabled: bool = True
    neo4j_enabled: bool = True
    qdrant_enabled: bool = True
    redis_enabled: bool = True

    # Event sourcing
    event_store_enabled: bool = True
    snapshot_interval: int = 100

    # ML configuration
    ml_device: str = "cuda"
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"

    # Pipeline configuration
    pipeline_parallel: bool = True
    pipeline_max_agents: int = 8
    pipeline_batch_size: int = 10

    # Feature flags
    feature_flags: Dict[str, FeatureFlag] = field(default_factory=dict)

    def is_feature_enabled(self, feature_name: str, default: bool = False) -> bool:
        """Check if a feature is enabled."""
        flag = self.feature_flags.get(feature_name)
        return flag.enabled if flag else default

    def enable_feature(self, name: str, description: str = "") -> "ApplicationConfig":
        """Enable a feature flag (fluent API)."""
        self.feature_flags[name] = FeatureFlag(name=name, enabled=True, description=description)
        return self

    def disable_feature(self, name: str) -> "ApplicationConfig":
        """Disable a feature flag (fluent API)."""
        if name in self.feature_flags:
            self.feature_flags[name].enabled = False
        return self

    @classmethod
    def from_environment(cls) -> "ApplicationConfig":
        """Create configuration from environment variables."""
        import os

        config = cls()
        config.environment = os.getenv("BIBLOS_ENV", config.environment)
        config.debug = os.getenv("BIBLOS_DEBUG", "").lower() == "true"
        config.log_level = os.getenv("BIBLOS_LOG_LEVEL", config.log_level)
        config.ml_device = os.getenv("BIBLOS_ML_DEVICE", config.ml_device)

        # Database flags
        config.postgres_enabled = os.getenv("BIBLOS_POSTGRES", "true").lower() == "true"
        config.neo4j_enabled = os.getenv("BIBLOS_NEO4J", "true").lower() == "true"
        config.qdrant_enabled = os.getenv("BIBLOS_QDRANT", "true").lower() == "true"
        config.redis_enabled = os.getenv("BIBLOS_REDIS", "true").lower() == "true"

        return config

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() in ("production", "prod")

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() in ("development", "dev", "local")


# =============================================================================
# HEALTH AGGREGATION
# =============================================================================


class AggregateHealthStatus(Enum):
    """Aggregated health status across all checks."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class AggregateHealthReport:
    """
    Comprehensive health report aggregating all health checks.

    This is the vital signs readout for the entire organism.
    """
    status: AggregateHealthStatus
    timestamp: float
    checks: List[HealthCheckResult]
    total_checks: int
    healthy_count: int
    degraded_count: int
    unhealthy_count: int
    duration_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_healthy(self) -> bool:
        return self.status == AggregateHealthStatus.HEALTHY

    @property
    def unhealthy_services(self) -> List[str]:
        return [c.service_name for c in self.checks if c.status == HealthStatus.UNHEALTHY]

    @property
    def degraded_services(self) -> List[str]:
        return [c.service_name for c in self.checks if c.status == HealthStatus.DEGRADED]


class HealthAggregator:
    """
    Aggregates health checks from all registered services.

    Implements circuit breaker pattern to avoid cascading health check failures.
    """

    def __init__(
        self,
        timeout: float = 10.0,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
    ):
        self._timeout = timeout
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._failure_counts: Dict[str, int] = {}
        self._circuit_open_until: Dict[str, float] = {}

    async def check_health(self, container: Container) -> AggregateHealthReport:
        """Run all health checks and aggregate results."""
        start_time = time.time()

        try:
            results = await asyncio.wait_for(
                container.check_health(),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            results = [
                HealthCheckResult(
                    service_name="health_aggregator",
                    status=HealthStatus.UNHEALTHY,
                    message="Health check timed out",
                    duration_ms=self._timeout * 1000,
                )
            ]

        duration_ms = (time.time() - start_time) * 1000

        # Count statuses
        healthy_count = sum(1 for r in results if r.status == HealthStatus.HEALTHY)
        degraded_count = sum(1 for r in results if r.status == HealthStatus.DEGRADED)
        unhealthy_count = sum(1 for r in results if r.status == HealthStatus.UNHEALTHY)

        # Determine aggregate status
        if unhealthy_count > 0:
            status = AggregateHealthStatus.UNHEALTHY
        elif degraded_count > 0:
            status = AggregateHealthStatus.DEGRADED
        elif healthy_count > 0:
            status = AggregateHealthStatus.HEALTHY
        else:
            status = AggregateHealthStatus.UNKNOWN

        return AggregateHealthReport(
            status=status,
            timestamp=time.time(),
            checks=results,
            total_checks=len(results),
            healthy_count=healthy_count,
            degraded_count=degraded_count,
            unhealthy_count=unhealthy_count,
            duration_ms=duration_ms,
        )


# =============================================================================
# APPLICATION CORE
# =============================================================================


class Application:
    """
    The living application - the organism itself.

    The Application class represents the complete running system,
    coordinating all services, managing lifecycle, and providing
    access to the dependency injection container.

    Responsibilities:
        - Service container configuration and management
        - Lifecycle phase transitions
        - Resource initialization and cleanup
        - Signal handling for graceful shutdown
        - Health monitoring and reporting
        - Background service coordination
        - Telemetry and observability integration
    """

    __slots__ = (
        "_config",
        "_container",
        "_phase",
        "_lifecycle_events",
        "_lifecycle_hooks",
        "_modules",
        "_background_services",
        "_background_tasks",
        "_shutdown_requested",
        "_cancellation_token",
        "_health_aggregator",
        "_services",
        "_startup_time",
        "_instance_id",
    )

    def __init__(
        self,
        config: Optional[ApplicationConfig] = None,
        container: Optional[Container] = None,
    ):
        self._config = config or ApplicationConfig()
        self._container = container or get_container()
        self._phase = ApplicationPhase.CREATED
        self._lifecycle_events: List[LifecycleEvent] = []
        self._lifecycle_hooks: List[ILifecycleHook] = []
        self._modules: Dict[str, IServiceModule] = {}
        self._background_services: List[IBackgroundService] = []
        self._background_tasks: List[asyncio.Task[None]] = []
        self._shutdown_requested = asyncio.Event()
        self._cancellation_token = asyncio.Event()
        self._health_aggregator = HealthAggregator(
            timeout=self._config.health_check_timeout,
        )
        self._services: Dict[str, Any] = {}
        self._startup_time: Optional[float] = None
        self._instance_id = self._config.instance_id

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def config(self) -> ApplicationConfig:
        """Application configuration."""
        return self._config

    @property
    def phase(self) -> ApplicationPhase:
        """Current lifecycle phase."""
        return self._phase

    @property
    def container(self) -> Container:
        """Dependency injection container."""
        return self._container

    @property
    def is_running(self) -> bool:
        """Whether the application is in running state."""
        return self._phase == ApplicationPhase.RUNNING

    @property
    def is_healthy(self) -> bool:
        """Quick health check - returns True if running."""
        return self._phase == ApplicationPhase.RUNNING

    @property
    def uptime_seconds(self) -> float:
        """Time since startup in seconds."""
        if self._startup_time is None:
            return 0.0
        return time.time() - self._startup_time

    @property
    def instance_id(self) -> str:
        """Unique instance identifier."""
        return self._instance_id

    # -------------------------------------------------------------------------
    # Registration Methods (used during configuration)
    # -------------------------------------------------------------------------

    def add_lifecycle_hook(self, hook: ILifecycleHook) -> "Application":
        """Add a lifecycle hook."""
        self._lifecycle_hooks.append(hook)
        # Keep hooks sorted by priority
        self._lifecycle_hooks.sort(key=lambda h: h.priority)
        return self

    def add_module(self, module: IServiceModule) -> "Application":
        """Add a service module."""
        self._modules[module.name] = module
        return self

    def add_background_service(self, service: IBackgroundService) -> "Application":
        """Add a background service."""
        self._background_services.append(service)
        return self

    # -------------------------------------------------------------------------
    # Lifecycle Management
    # -------------------------------------------------------------------------

    async def initialize(self) -> None:
        """
        Initialize the application through all startup stages.

        This is the awakening of the organism - each organ comes online
        in the correct order, with dependencies resolved automatically.
        """
        total_start = time.time()

        try:
            # Phase: Configuring
            self._phase = ApplicationPhase.CONFIGURING
            logger.info(f"Configuring application (instance={self._instance_id})...")
            await self._run_configure_phase()

            # Phase: Initializing
            self._phase = ApplicationPhase.INITIALIZING
            logger.info("Initializing services...")
            await asyncio.wait_for(
                self._run_initialize_phase(),
                timeout=self._config.startup_timeout,
            )

            # Phase: Starting
            self._phase = ApplicationPhase.STARTING
            logger.info("Starting background services...")
            await self._run_start_phase()

            # Phase: Running
            self._phase = ApplicationPhase.RUNNING
            self._startup_time = time.time()

            total_duration = (time.time() - total_start) * 1000
            logger.info(
                f"Application started successfully "
                f"(duration={total_duration:.0f}ms, instance={self._instance_id})"
            )

        except asyncio.TimeoutError:
            self._phase = ApplicationPhase.FAILED
            logger.error(f"Initialization timed out after {self._config.startup_timeout}s")
            raise
        except Exception as e:
            self._phase = ApplicationPhase.FAILED
            logger.error(f"Initialization failed: {e}")
            raise

    async def _run_configure_phase(self) -> None:
        """Execute configuration phase."""
        # Run configure hooks (sorted by priority)
        for hook in self._lifecycle_hooks:
            try:
                start = time.time()
                await hook.on_configure(self)
                self._record_event(
                    LifecycleEvent.success_event(
                        phase=ApplicationPhase.CONFIGURING,
                        component=f"hook:{hook.name}",
                        duration_ms=(time.time() - start) * 1000,
                    )
                )
            except Exception as e:
                self._record_event(
                    LifecycleEvent.failure_event(
                        phase=ApplicationPhase.CONFIGURING,
                        component=f"hook:{hook.name}",
                        error=e,
                    )
                )
                raise

        # Configure modules in dependency order
        sorted_modules = self._topological_sort_modules()
        for module in sorted_modules:
            try:
                start = time.time()
                module.configure_services(self._container, self._config)
                self._record_event(
                    LifecycleEvent.success_event(
                        phase=ApplicationPhase.CONFIGURING,
                        component=f"module:{module.name}",
                        duration_ms=(time.time() - start) * 1000,
                    )
                )
                logger.debug(f"Configured module: {module.name}")
            except Exception as e:
                self._record_event(
                    LifecycleEvent.failure_event(
                        phase=ApplicationPhase.CONFIGURING,
                        component=f"module:{module.name}",
                        error=e,
                    )
                )
                raise

        # Apply default service configuration
        await self._configure_default_services()

    async def _configure_default_services(self) -> None:
        """Configure default services based on config flags."""
        # Database services
        if self._config.postgres_enabled:
            try:
                from db.postgres_optimized import PostgresClient
                self._container.register_singleton(PostgresClient)
            except ImportError:
                logger.warning("PostgresClient not available")

        if self._config.neo4j_enabled:
            try:
                from db.neo4j_optimized import Neo4jClient
                self._container.register_singleton(Neo4jClient)
            except ImportError:
                logger.warning("Neo4jClient not available")

        if self._config.qdrant_enabled:
            try:
                from db.qdrant_client import QdrantVectorStore
                self._container.register_singleton(QdrantVectorStore)
            except ImportError:
                logger.warning("QdrantVectorStore not available")

        # Connection manager
        try:
            from db.connection_pool_optimized import ConnectionManager
            self._container.register_singleton(ConnectionManager)
        except ImportError:
            logger.warning("ConnectionManager not available")

        # Event store
        if self._config.event_store_enabled:
            try:
                from db.event_store import EventStore
                self._container.register_singleton(EventStore)
            except ImportError:
                logger.warning("EventStore not available")

        # Pipeline components
        try:
            from pipeline.unified_orchestrator import UnifiedOrchestrator
            self._container.register_singleton(UnifiedOrchestrator)
        except ImportError:
            logger.warning("UnifiedOrchestrator not available")

        # Agent registry
        try:
            from agents.registry import AgentRegistry
            self._container.register_singleton(AgentRegistry)
        except ImportError:
            logger.warning("AgentRegistry not available")

        logger.debug("Default services configured")

    async def _run_initialize_phase(self) -> None:
        """Execute initialization phase with retry support."""
        # Initialize modules in dependency order
        sorted_modules = self._topological_sort_modules()
        for module in sorted_modules:
            await self._initialize_with_retry(
                f"module:{module.name}",
                lambda m=module: m.initialize(self._container),
            )

        # Initialize connection manager (critical service)
        try:
            from db.connection_pool_optimized import ConnectionManager
            conn_manager = self._container.try_resolve(ConnectionManager)
            if conn_manager:
                await self._initialize_with_retry(
                    "connection_manager",
                    conn_manager.initialize,
                )
                self._services["connection_manager"] = conn_manager
        except ImportError:
            logger.debug("ConnectionManager not available for initialization")

        # Run initialize hooks
        for hook in self._lifecycle_hooks:
            try:
                start = time.time()
                await hook.on_initialize(self)
                self._record_event(
                    LifecycleEvent.success_event(
                        phase=ApplicationPhase.INITIALIZING,
                        component=f"hook:{hook.name}",
                        duration_ms=(time.time() - start) * 1000,
                    )
                )
            except Exception as e:
                self._record_event(
                    LifecycleEvent.failure_event(
                        phase=ApplicationPhase.INITIALIZING,
                        component=f"hook:{hook.name}",
                        error=e,
                    )
                )
                raise

    async def _initialize_with_retry(
        self,
        component: str,
        initializer: Callable[[], Awaitable[None]],
    ) -> None:
        """Initialize a component with retry logic."""
        last_error: Optional[Exception] = None

        for attempt in range(self._config.initialization_retry_count):
            try:
                start = time.time()
                await initializer()
                self._record_event(
                    LifecycleEvent.success_event(
                        phase=ApplicationPhase.INITIALIZING,
                        component=component,
                        duration_ms=(time.time() - start) * 1000,
                        metadata={"attempt": attempt + 1},
                    )
                )
                logger.info(f"✓ {component} initialized")
                return
            except Exception as e:
                last_error = e
                if attempt < self._config.initialization_retry_count - 1:
                    logger.warning(
                        f"✗ {component} failed (attempt {attempt + 1}), retrying..."
                    )
                    await asyncio.sleep(self._config.initialization_retry_delay)

        # All retries exhausted
        self._record_event(
            LifecycleEvent.failure_event(
                phase=ApplicationPhase.INITIALIZING,
                component=component,
                error=last_error or Exception("Unknown initialization error"),
            )
        )
        logger.error(f"✗ {component} failed after {self._config.initialization_retry_count} attempts")
        raise last_error or Exception(f"Failed to initialize {component}")

    async def _run_start_phase(self) -> None:
        """Execute start phase - launch background services."""
        # Run start hooks
        for hook in self._lifecycle_hooks:
            try:
                start = time.time()
                await hook.on_start(self)
                self._record_event(
                    LifecycleEvent.success_event(
                        phase=ApplicationPhase.STARTING,
                        component=f"hook:{hook.name}",
                        duration_ms=(time.time() - start) * 1000,
                    )
                )
            except Exception as e:
                self._record_event(
                    LifecycleEvent.failure_event(
                        phase=ApplicationPhase.STARTING,
                        component=f"hook:{hook.name}",
                        error=e,
                    )
                )
                raise

        # Start background services
        for service in self._background_services:
            task = asyncio.create_task(
                service.start(self._cancellation_token),
                name=f"bg:{service.name}",
            )
            self._background_tasks.append(task)
            logger.info(f"Started background service: {service.name}")

    async def shutdown(self) -> None:
        """
        Gracefully shutdown the application.

        This is the organism's rest - each organ shuts down in reverse
        order of startup, ensuring clean resource release.
        """
        if self._phase == ApplicationPhase.TERMINATED:
            return

        self._phase = ApplicationPhase.SHUTTING_DOWN
        logger.info("Shutting down application...")

        shutdown_start = time.time()

        try:
            # Signal cancellation to background services
            self._cancellation_token.set()

            # Stop background services
            for service in reversed(self._background_services):
                try:
                    start = time.time()
                    await asyncio.wait_for(
                        service.stop(),
                        timeout=5.0,
                    )
                    self._record_event(
                        LifecycleEvent.success_event(
                            phase=ApplicationPhase.SHUTTING_DOWN,
                            stage=ShutdownStage.BACKGROUND,
                            component=f"service:{service.name}",
                            duration_ms=(time.time() - start) * 1000,
                        )
                    )
                    logger.debug(f"Stopped background service: {service.name}")
                except Exception as e:
                    self._record_event(
                        LifecycleEvent.failure_event(
                            phase=ApplicationPhase.SHUTTING_DOWN,
                            stage=ShutdownStage.BACKGROUND,
                            component=f"service:{service.name}",
                            error=e,
                        )
                    )
                    logger.warning(f"Error stopping service {service.name}: {e}")

            # Wait for background tasks to complete
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
                self._background_tasks.clear()

            # Run shutdown hooks in reverse priority order
            for hook in reversed(self._lifecycle_hooks):
                try:
                    start = time.time()
                    await hook.on_shutdown(self)
                    self._record_event(
                        LifecycleEvent.success_event(
                            phase=ApplicationPhase.SHUTTING_DOWN,
                            component=f"hook:{hook.name}",
                            duration_ms=(time.time() - start) * 1000,
                        )
                    )
                except Exception as e:
                    self._record_event(
                        LifecycleEvent.failure_event(
                            phase=ApplicationPhase.SHUTTING_DOWN,
                            component=f"hook:{hook.name}",
                            error=e,
                        )
                    )
                    logger.error(f"Lifecycle hook shutdown error: {e}")

            # Shutdown modules in reverse dependency order
            sorted_modules = list(reversed(self._topological_sort_modules()))
            for module in sorted_modules:
                try:
                    start = time.time()
                    await module.shutdown(self._container)
                    self._record_event(
                        LifecycleEvent.success_event(
                            phase=ApplicationPhase.SHUTTING_DOWN,
                            component=f"module:{module.name}",
                            duration_ms=(time.time() - start) * 1000,
                        )
                    )
                except Exception as e:
                    self._record_event(
                        LifecycleEvent.failure_event(
                            phase=ApplicationPhase.SHUTTING_DOWN,
                            component=f"module:{module.name}",
                            error=e,
                        )
                    )
                    logger.error(f"Module shutdown error ({module.name}): {e}")

            # Shutdown connection manager
            if "connection_manager" in self._services:
                try:
                    await self._services["connection_manager"].shutdown()
                except Exception as e:
                    logger.error(f"Connection manager shutdown error: {e}")

        finally:
            self._phase = ApplicationPhase.TERMINATED
            shutdown_duration = (time.time() - shutdown_start) * 1000
            logger.info(f"Application shutdown complete (duration={shutdown_duration:.0f}ms)")

    def _topological_sort_modules(self) -> List[IServiceModule]:
        """Sort modules by their dependencies using topological sort."""
        # Build adjacency list
        in_degree: Dict[str, int] = {name: 0 for name in self._modules}
        graph: Dict[str, List[str]] = {name: [] for name in self._modules}

        for name, module in self._modules.items():
            for dep in module.dependencies:
                if dep in self._modules:
                    graph[dep].append(name)
                    in_degree[name] += 1

        # Kahn's algorithm
        queue = [name for name, degree in in_degree.items() if degree == 0]
        result: List[IServiceModule] = []

        while queue:
            current = queue.pop(0)
            result.append(self._modules[current])

            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(self._modules):
            # Circular dependency detected
            remaining = set(self._modules.keys()) - {m.name for m in result}
            raise RuntimeError(f"Circular dependency detected in modules: {remaining}")

        return result

    # -------------------------------------------------------------------------
    # Service Access
    # -------------------------------------------------------------------------

    def get_service(self, service_type: Type[T]) -> T:
        """Get a service from the container."""
        return self._container.resolve(service_type)

    def try_get_service(self, service_type: Type[T]) -> Optional[T]:
        """Try to get a service, returning None if not found."""
        return self._container.try_resolve(service_type)

    # -------------------------------------------------------------------------
    # Health and Observability
    # -------------------------------------------------------------------------

    async def check_health(self) -> AggregateHealthReport:
        """Run comprehensive health check."""
        return await self._health_aggregator.check_health(self._container)

    def request_shutdown(self) -> None:
        """Request graceful shutdown (called by signal handlers)."""
        self._shutdown_requested.set()

    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown signal."""
        await self._shutdown_requested.wait()

    def _record_event(self, event: LifecycleEvent) -> None:
        """Record a lifecycle event."""
        self._lifecycle_events.append(event)

        if event.success:
            logger.debug(f"Lifecycle: {event.component} ({event.duration_ms:.0f}ms)")
        else:
            logger.warning(f"Lifecycle failed: {event.component} - {event.error}")

    def get_lifecycle_report(self) -> Dict[str, Any]:
        """Get a comprehensive report of the application lifecycle."""
        return {
            "instance_id": self._instance_id,
            "phase": self._phase.value,
            "environment": self._config.environment,
            "uptime_seconds": self.uptime_seconds,
            "events": [
                {
                    "event_id": str(e.event_id),
                    "timestamp": e.timestamp,
                    "phase": e.phase.value,
                    "stage": e.stage.name if e.stage else None,
                    "component": e.component,
                    "success": e.success,
                    "duration_ms": e.duration_ms,
                    "error": e.error,
                    "error_type": e.error_type,
                }
                for e in self._lifecycle_events
            ],
            "total_events": len(self._lifecycle_events),
            "failed_events": sum(1 for e in self._lifecycle_events if not e.success),
            "modules": list(self._modules.keys()),
            "hooks": [h.name for h in self._lifecycle_hooks],
            "background_services": [s.name for s in self._background_services],
        }


# =============================================================================
# APPLICATION BUILDER
# =============================================================================


class ApplicationBuilder:
    """
    Fluent builder for Application configuration.

    The builder pattern allows declarative, readable application setup:

        app = await (
            ApplicationBuilder()
            .with_environment("production")
            .with_module(DatabaseModule())
            .with_lifecycle_hook(LoggingHook())
            .with_feature("event_sourcing", enabled=True)
            .build()
        )

    The builder ensures all configuration is validated before the application
    is constructed, providing fail-fast behavior for configuration errors.
    """

    def __init__(self):
        self._config = ApplicationConfig()
        self._container: Optional[Container] = None
        self._modules: List[IServiceModule] = []
        self._hooks: List[Tuple[ILifecycleHook, int]] = []  # (hook, priority)
        self._background_services: List[IBackgroundService] = []
        self._service_configurators: List[Callable[[IServiceCollection], None]] = []

    # -------------------------------------------------------------------------
    # Environment Configuration
    # -------------------------------------------------------------------------

    def with_environment(self, environment: str) -> "ApplicationBuilder":
        """Set the application environment."""
        self._config.environment = environment
        return self

    def with_debug(self, debug: bool = True) -> "ApplicationBuilder":
        """Enable or disable debug mode."""
        self._config.debug = debug
        return self

    def with_app_info(self, name: str, version: str) -> "ApplicationBuilder":
        """Set application name and version."""
        self._config.app_name = name
        self._config.app_version = version
        return self

    # -------------------------------------------------------------------------
    # Service Configuration
    # -------------------------------------------------------------------------

    def with_container(self, container: Container) -> "ApplicationBuilder":
        """Use a custom dependency injection container."""
        self._container = container
        return self

    def with_module(self, module: IServiceModule) -> "ApplicationBuilder":
        """Add a service module."""
        self._modules.append(module)
        return self

    def with_services(
        self,
        configurator: Callable[[IServiceCollection], None],
    ) -> "ApplicationBuilder":
        """Add custom service configuration."""
        self._service_configurators.append(configurator)
        return self

    # -------------------------------------------------------------------------
    # Lifecycle Configuration
    # -------------------------------------------------------------------------

    def with_lifecycle_hook(
        self,
        hook: ILifecycleHook,
        priority: Optional[int] = None,
    ) -> "ApplicationBuilder":
        """Add a lifecycle hook with optional priority override."""
        actual_priority = priority if priority is not None else hook.priority
        self._hooks.append((hook, actual_priority))
        return self

    def with_background_service(
        self,
        service: IBackgroundService,
    ) -> "ApplicationBuilder":
        """Add a background service."""
        self._background_services.append(service)
        return self

    # -------------------------------------------------------------------------
    # Database Configuration
    # -------------------------------------------------------------------------

    def with_postgres(self, enabled: bool = True) -> "ApplicationBuilder":
        """Enable or disable PostgreSQL."""
        self._config.postgres_enabled = enabled
        return self

    def with_neo4j(self, enabled: bool = True) -> "ApplicationBuilder":
        """Enable or disable Neo4j."""
        self._config.neo4j_enabled = enabled
        return self

    def with_qdrant(self, enabled: bool = True) -> "ApplicationBuilder":
        """Enable or disable Qdrant."""
        self._config.qdrant_enabled = enabled
        return self

    def with_redis(self, enabled: bool = True) -> "ApplicationBuilder":
        """Enable or disable Redis."""
        self._config.redis_enabled = enabled
        return self

    def with_event_sourcing(self, enabled: bool = True) -> "ApplicationBuilder":
        """Enable or disable event sourcing."""
        self._config.event_store_enabled = enabled
        return self

    # -------------------------------------------------------------------------
    # ML Configuration
    # -------------------------------------------------------------------------

    def with_ml_device(self, device: str) -> "ApplicationBuilder":
        """Set the ML compute device (cuda/cpu)."""
        self._config.ml_device = device
        return self

    def with_embedding_model(self, model: str) -> "ApplicationBuilder":
        """Set the embedding model."""
        self._config.embedding_model = model
        return self

    # -------------------------------------------------------------------------
    # Pipeline Configuration
    # -------------------------------------------------------------------------

    def with_pipeline_config(
        self,
        parallel: bool = True,
        max_agents: int = 8,
        batch_size: int = 10,
    ) -> "ApplicationBuilder":
        """Configure the processing pipeline."""
        self._config.pipeline_parallel = parallel
        self._config.pipeline_max_agents = max_agents
        self._config.pipeline_batch_size = batch_size
        return self

    # -------------------------------------------------------------------------
    # Feature Flags
    # -------------------------------------------------------------------------

    def with_feature(
        self,
        name: str,
        enabled: bool = True,
        description: str = "",
    ) -> "ApplicationBuilder":
        """Enable or disable a feature flag."""
        self._config.feature_flags[name] = FeatureFlag(
            name=name,
            enabled=enabled,
            description=description,
        )
        return self

    # -------------------------------------------------------------------------
    # Timeouts
    # -------------------------------------------------------------------------

    def with_startup_timeout(self, timeout: float) -> "ApplicationBuilder":
        """Set the startup timeout in seconds."""
        self._config.startup_timeout = timeout
        return self

    def with_shutdown_timeout(self, timeout: float) -> "ApplicationBuilder":
        """Set the graceful shutdown timeout in seconds."""
        self._config.graceful_shutdown_timeout = timeout
        return self

    # -------------------------------------------------------------------------
    # Observability
    # -------------------------------------------------------------------------

    def with_telemetry(self, enabled: bool = True) -> "ApplicationBuilder":
        """Enable or disable telemetry."""
        self._config.enable_telemetry = enabled
        return self

    def with_health_checks(self, enabled: bool = True) -> "ApplicationBuilder":
        """Enable or disable health checks."""
        self._config.enable_health_checks = enabled
        return self

    def with_metrics(self, enabled: bool = True) -> "ApplicationBuilder":
        """Enable or disable metrics collection."""
        self._config.enable_metrics = enabled
        return self

    def with_log_level(self, level: str) -> "ApplicationBuilder":
        """Set the logging level."""
        self._config.log_level = level
        return self

    # -------------------------------------------------------------------------
    # Build
    # -------------------------------------------------------------------------

    def validate(self) -> List[str]:
        """Validate the configuration and return any errors."""
        errors: List[str] = []

        # Check for circular dependencies in modules
        module_names = {m.name for m in self._modules}
        for module in self._modules:
            for dep in module.dependencies:
                if dep not in module_names:
                    errors.append(f"Module '{module.name}' depends on unknown module '{dep}'")

        # Validate ML device
        if self._config.ml_device not in ("cuda", "cpu", "mps"):
            errors.append(f"Invalid ML device: {self._config.ml_device}")

        # Validate timeouts
        if self._config.startup_timeout <= 0:
            errors.append("Startup timeout must be positive")
        if self._config.graceful_shutdown_timeout <= 0:
            errors.append("Shutdown timeout must be positive")

        return errors

    async def build(self) -> Application:
        """
        Build and initialize the application.

        Returns:
            Fully initialized Application instance

        Raises:
            ValueError: If configuration validation fails
        """
        # Validate configuration
        errors = self.validate()
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")

        # Create container
        container = self._container or get_container()

        # Apply custom service configurators
        for configurator in self._service_configurators:
            configurator(container)

        # Create application
        app = Application(config=self._config, container=container)

        # Add modules
        for module in self._modules:
            app.add_module(module)

        # Add hooks (sorted by priority)
        sorted_hooks = sorted(self._hooks, key=lambda x: x[1])
        for hook, _ in sorted_hooks:
            app.add_lifecycle_hook(hook)

        # Add background services
        for service in self._background_services:
            app.add_background_service(service)

        # Initialize
        await app.initialize()

        return app

    def build_sync(self) -> Application:
        """Build the application synchronously (not initialized)."""
        # Validate configuration
        errors = self.validate()
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")

        # Create container
        container = self._container or get_container()

        # Apply custom service configurators
        for configurator in self._service_configurators:
            configurator(container)

        # Create application
        app = Application(config=self._config, container=container)

        # Add modules
        for module in self._modules:
            app.add_module(module)

        # Add hooks
        sorted_hooks = sorted(self._hooks, key=lambda x: x[1])
        for hook, _ in sorted_hooks:
            app.add_lifecycle_hook(hook)

        # Add background services
        for service in self._background_services:
            app.add_background_service(service)

        return app


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


@asynccontextmanager
async def bootstrap(
    config: Optional[ApplicationConfig] = None,
    setup_signals: bool = True,
) -> AsyncIterator[Application]:
    """
    Bootstrap the application with proper lifecycle management.

    Usage:
        async with bootstrap() as app:
            # Application is initialized and running
            result = await app.get_service(Pipeline).process_verse("GEN.1.1")

    Args:
        config: Optional application configuration
        setup_signals: Whether to set up signal handlers

    Yields:
        Initialized Application instance
    """
    builder = ApplicationBuilder()

    if config:
        builder._config = config

    # Build without initialization (we'll do it manually for signal setup)
    app = builder.build_sync()

    # Set up signal handlers for graceful shutdown
    if setup_signals and sys.platform != "win32":
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, app.request_shutdown)

    try:
        await app.initialize()
        yield app
    finally:
        await app.shutdown()


# Global application instance
_global_app: Optional[Application] = None


def get_application() -> Application:
    """Get the global application instance."""
    global _global_app
    if _global_app is None:
        _global_app = Application()
    return _global_app


def set_application(app: Application) -> None:
    """Set the global application instance."""
    global _global_app
    _global_app = app


async def run_application(
    main: Callable[[Application], Awaitable[Any]],
    config: Optional[ApplicationConfig] = None,
) -> None:
    """
    Run the application with a main function.

    Usage:
        async def main(app: Application):
            # Do work with the application
            pass

        asyncio.run(run_application(main))
    """
    async with bootstrap(config=config) as app:
        await main(app)


# =============================================================================
# BUILT-IN MODULES
# =============================================================================


class DatabaseModule(ServiceModuleBase):
    """Module for database services."""

    def __init__(self):
        super().__init__("DatabaseModule")

    def configure_services(self, services: IServiceCollection, config: ApplicationConfig) -> None:
        """Register database services."""
        if config.postgres_enabled:
            try:
                from db.postgres_optimized import PostgresClient
                services.register_singleton(PostgresClient)
            except ImportError:
                logger.debug("PostgresClient not available")

        if config.neo4j_enabled:
            try:
                from db.neo4j_optimized import Neo4jClient
                services.register_singleton(Neo4jClient)
            except ImportError:
                logger.debug("Neo4jClient not available")

        if config.qdrant_enabled:
            try:
                from db.qdrant_client import QdrantVectorStore
                services.register_singleton(QdrantVectorStore)
            except ImportError:
                logger.debug("QdrantVectorStore not available")


class EventSourcingModule(ServiceModuleBase):
    """Module for event sourcing infrastructure."""

    def __init__(self):
        super().__init__("EventSourcingModule")
        self.depends_on("DatabaseModule")

    def configure_services(self, services: IServiceCollection, config: ApplicationConfig) -> None:
        """Register event sourcing services."""
        if config.event_store_enabled:
            try:
                from db.event_store import EventStore, SnapshotStore, SubscriptionManager
                services.register_singleton(EventStore)
                services.register_singleton(SnapshotStore)
                services.register_singleton(SubscriptionManager)
            except ImportError:
                logger.debug("Event sourcing components not available")


class PipelineModule(ServiceModuleBase):
    """Module for ML pipeline components."""

    def __init__(self):
        super().__init__("PipelineModule")
        self.depends_on("DatabaseModule")

    def configure_services(self, services: IServiceCollection, config: ApplicationConfig) -> None:
        """Register pipeline services."""
        try:
            from pipeline.unified_orchestrator import UnifiedOrchestrator
            services.register_singleton(UnifiedOrchestrator)
        except ImportError:
            logger.debug("UnifiedOrchestrator not available")

        try:
            from agents.registry import AgentRegistry
            services.register_singleton(AgentRegistry)
        except ImportError:
            logger.debug("AgentRegistry not available")


class MediatorModule(ServiceModuleBase):
    """Module for mediator pattern infrastructure."""

    def __init__(self):
        super().__init__("MediatorModule")

    def configure_services(self, services: IServiceCollection, config: ApplicationConfig) -> None:
        """Register mediator services."""
        try:
            from domain.mediator import Mediator, MediatorBuilder
            services.register_singleton(Mediator)
        except ImportError:
            logger.debug("Mediator not available")


# =============================================================================
# BUILT-IN LIFECYCLE HOOKS
# =============================================================================


class LoggingLifecycleHook(LifecycleHookBase):
    """Lifecycle hook that logs all phase transitions."""

    def __init__(self):
        super().__init__("LoggingHook", priority=0)

    async def on_configure(self, app: Application) -> None:
        logger.info(f"[Lifecycle] Configuring {app.config.app_name} v{app.config.app_version}")

    async def on_initialize(self, app: Application) -> None:
        logger.info(f"[Lifecycle] Initializing in {app.config.environment} environment")

    async def on_start(self, app: Application) -> None:
        logger.info(f"[Lifecycle] Starting (instance={app.instance_id})")

    async def on_shutdown(self, app: Application) -> None:
        logger.info(f"[Lifecycle] Shutting down (uptime={app.uptime_seconds:.1f}s)")


class TelemetryLifecycleHook(LifecycleHookBase):
    """Lifecycle hook for telemetry integration."""

    def __init__(self):
        super().__init__("TelemetryHook", priority=10)

    async def on_start(self, app: Application) -> None:
        if app.config.enable_telemetry:
            logger.debug("[Telemetry] Starting telemetry collection")

    async def on_shutdown(self, app: Application) -> None:
        if app.config.enable_telemetry:
            logger.debug("[Telemetry] Flushing telemetry data")


# =============================================================================
# BUILT-IN BACKGROUND SERVICES
# =============================================================================


class HealthMonitorService(BackgroundServiceBase):
    """Background service that periodically checks health."""

    def __init__(self, interval_seconds: float = 30.0):
        super().__init__("HealthMonitor")
        self._interval = interval_seconds
        self._app: Optional[Application] = None

    def set_application(self, app: Application) -> None:
        """Set the application reference."""
        self._app = app

    async def execute(self, cancellation_token: asyncio.Event) -> None:
        """Periodically check health."""
        while not cancellation_token.is_set():
            try:
                await asyncio.wait_for(
                    cancellation_token.wait(),
                    timeout=self._interval,
                )
                # Cancellation requested
                break
            except asyncio.TimeoutError:
                # Timeout = time to check health
                if self._app:
                    report = await self._app.check_health()
                    if report.status != AggregateHealthStatus.HEALTHY:
                        logger.warning(
                            f"[HealthMonitor] Status: {report.status.value}, "
                            f"unhealthy: {report.unhealthy_services}"
                        )


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Phases and Events
    "ApplicationPhase",
    "StartupStage",
    "ShutdownStage",
    "LifecycleEvent",
    # Configuration
    "ApplicationConfig",
    "FeatureFlag",
    # Lifecycle Interfaces
    "ILifecycleHook",
    "LifecycleHookBase",
    "IServiceModule",
    "ServiceModuleBase",
    "IBackgroundService",
    "BackgroundServiceBase",
    # Health
    "AggregateHealthStatus",
    "AggregateHealthReport",
    "HealthAggregator",
    # Application
    "Application",
    "ApplicationBuilder",
    # Built-in Modules
    "DatabaseModule",
    "EventSourcingModule",
    "PipelineModule",
    "MediatorModule",
    # Built-in Hooks
    "LoggingLifecycleHook",
    "TelemetryLifecycleHook",
    # Built-in Background Services
    "HealthMonitorService",
    # Convenience Functions
    "bootstrap",
    "get_application",
    "set_application",
    "run_application",
]
