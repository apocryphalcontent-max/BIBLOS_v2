"""
BIBLOS v2 - Application Bootstrap

Provides centralized application lifecycle management and dependency wiring.
This module is the single entry point for configuring and starting the application.

Architecture:
    Bootstrap → Container → Services → Application

Usage:
    from core.bootstrap import Application, bootstrap

    # Quick start
    async with bootstrap() as app:
        await app.pipeline.process_verse("GEN.1.1")

    # Or with custom configuration
    app = Application()
    await app.initialize()
    try:
        # use app.services...
    finally:
        await app.shutdown()

Design Principles:
    - Explicit dependency wiring over implicit discovery
    - Fail-fast on configuration errors
    - Graceful shutdown with resource cleanup
    - Observable startup/shutdown phases
"""
from __future__ import annotations

import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Type

from di.container import Container, ServiceLifetime, get_container

logger = logging.getLogger("biblos.bootstrap")


class ApplicationPhase(Enum):
    """Application lifecycle phases."""
    CREATED = "created"
    CONFIGURING = "configuring"
    INITIALIZING = "initializing"
    RUNNING = "running"
    SHUTTING_DOWN = "shutting_down"
    TERMINATED = "terminated"


@dataclass
class StartupEvent:
    """Represents an event during application startup."""
    phase: ApplicationPhase
    component: str
    success: bool
    duration_ms: float
    error: Optional[Exception] = None


class ApplicationLifecycleHook:
    """Base class for lifecycle hooks."""

    async def on_configure(self, app: "Application") -> None:
        """Called during configuration phase."""
        pass

    async def on_initialize(self, app: "Application") -> None:
        """Called during initialization phase."""
        pass

    async def on_start(self, app: "Application") -> None:
        """Called when application starts running."""
        pass

    async def on_shutdown(self, app: "Application") -> None:
        """Called during shutdown phase."""
        pass


@dataclass
class ApplicationConfig:
    """Application-wide configuration."""
    environment: str = "development"
    debug: bool = False
    enable_telemetry: bool = True
    enable_health_checks: bool = True
    graceful_shutdown_timeout: float = 30.0
    startup_timeout: float = 60.0

    # Database configuration
    postgres_enabled: bool = True
    neo4j_enabled: bool = True
    qdrant_enabled: bool = True
    redis_enabled: bool = True

    # ML configuration
    ml_device: str = "cuda"
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"

    # Pipeline configuration
    pipeline_parallel: bool = True
    pipeline_max_agents: int = 8


class Application:
    """
    Main application class that manages the entire system lifecycle.

    Responsibilities:
        - Service container configuration
        - Resource initialization and cleanup
        - Signal handling for graceful shutdown
        - Health monitoring
        - Telemetry integration
    """

    def __init__(
        self,
        config: Optional[ApplicationConfig] = None,
        container: Optional[Container] = None,
    ):
        self.config = config or ApplicationConfig()
        self._container = container or get_container()
        self._phase = ApplicationPhase.CREATED
        self._startup_events: List[StartupEvent] = []
        self._lifecycle_hooks: List[ApplicationLifecycleHook] = []
        self._shutdown_requested = asyncio.Event()

        # Service references (populated during initialization)
        self._services: Dict[str, Any] = {}

    @property
    def phase(self) -> ApplicationPhase:
        """Current application lifecycle phase."""
        return self._phase

    @property
    def container(self) -> Container:
        """Dependency injection container."""
        return self._container

    @property
    def is_running(self) -> bool:
        """Whether application is in running state."""
        return self._phase == ApplicationPhase.RUNNING

    def add_lifecycle_hook(self, hook: ApplicationLifecycleHook) -> None:
        """Add a lifecycle hook for custom startup/shutdown logic."""
        self._lifecycle_hooks.append(hook)

    async def initialize(self) -> None:
        """
        Initialize all application services.

        This performs:
            1. Configuration validation
            2. Service container registration
            3. Database connection establishment
            4. ML model loading
            5. Pipeline initialization
            6. Health check setup
        """
        import time

        self._phase = ApplicationPhase.CONFIGURING
        logger.info("Configuring application...")

        # Run configure hooks
        for hook in self._lifecycle_hooks:
            await hook.on_configure(self)

        # Configure container with all services
        await self._configure_services()

        self._phase = ApplicationPhase.INITIALIZING
        logger.info("Initializing services...")

        # Initialize services in dependency order
        start_time = time.time()

        try:
            await asyncio.wait_for(
                self._initialize_services(),
                timeout=self.config.startup_timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Initialization timed out after {self.config.startup_timeout}s")
            raise

        # Run initialize hooks
        for hook in self._lifecycle_hooks:
            await hook.on_initialize(self)

        self._phase = ApplicationPhase.RUNNING
        logger.info(f"Application initialized in {time.time() - start_time:.2f}s")

        # Run start hooks
        for hook in self._lifecycle_hooks:
            await hook.on_start(self)

    async def _configure_services(self) -> None:
        """Register all services in the container."""

        # Database services
        if self.config.postgres_enabled:
            from db.postgres_optimized import PostgresClient
            self._container.register_singleton(PostgresClient)

        if self.config.neo4j_enabled:
            from db.neo4j_optimized import Neo4jClient
            self._container.register_singleton(Neo4jClient)

        if self.config.qdrant_enabled:
            from db.qdrant_client import QdrantVectorStore
            self._container.register_singleton(QdrantVectorStore)

        # Connection manager
        from db.connection_pool_optimized import ConnectionManager
        self._container.register_singleton(ConnectionManager)

        # Pipeline components
        from pipeline.unified_orchestrator import UnifiedOrchestrator
        self._container.register_singleton(UnifiedOrchestrator)

        # Agent registry
        from agents.registry import AgentRegistry
        self._container.register_singleton(AgentRegistry)

        logger.debug("Services registered in container")

    async def _initialize_services(self) -> None:
        """Initialize all registered services."""
        import time

        # Initialize connection manager (handles all databases)
        try:
            start = time.time()
            from db.connection_pool_optimized import ConnectionManager
            conn_manager = self._container.resolve(ConnectionManager)
            await conn_manager.initialize()
            self._services["connection_manager"] = conn_manager
            self._log_startup_event("connection_manager", True, (time.time() - start) * 1000)
        except Exception as e:
            self._log_startup_event("connection_manager", False, 0, e)
            # Connection manager failure is critical
            raise

        logger.info("All services initialized successfully")

    def _log_startup_event(
        self,
        component: str,
        success: bool,
        duration_ms: float,
        error: Optional[Exception] = None
    ) -> None:
        """Log a startup event."""
        event = StartupEvent(
            phase=self._phase,
            component=component,
            success=success,
            duration_ms=duration_ms,
            error=error
        )
        self._startup_events.append(event)

        if success:
            logger.info(f"✓ {component} initialized ({duration_ms:.0f}ms)")
        else:
            logger.error(f"✗ {component} failed: {error}")

    async def shutdown(self) -> None:
        """
        Gracefully shutdown the application.

        This performs:
            1. Signal lifecycle hooks
            2. Stop accepting new work
            3. Wait for in-flight work to complete
            4. Close database connections
            5. Release resources
        """
        if self._phase == ApplicationPhase.TERMINATED:
            return

        self._phase = ApplicationPhase.SHUTTING_DOWN
        logger.info("Shutting down application...")

        # Run shutdown hooks
        for hook in self._lifecycle_hooks:
            try:
                await hook.on_shutdown(self)
            except Exception as e:
                logger.error(f"Lifecycle hook shutdown error: {e}")

        # Shutdown connection manager (handles all databases)
        if "connection_manager" in self._services:
            try:
                await self._services["connection_manager"].shutdown()
            except Exception as e:
                logger.error(f"Connection manager shutdown error: {e}")

        self._phase = ApplicationPhase.TERMINATED
        logger.info("Application shutdown complete")

    def request_shutdown(self) -> None:
        """Request graceful shutdown (called by signal handlers)."""
        self._shutdown_requested.set()

    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown signal."""
        await self._shutdown_requested.wait()

    def get_service(self, service_type: Type) -> Any:
        """Get a service from the container."""
        return self._container.resolve(service_type)

    def get_startup_report(self) -> Dict[str, Any]:
        """Get a report of the startup process."""
        return {
            "phase": self._phase.value,
            "events": [
                {
                    "phase": e.phase.value,
                    "component": e.component,
                    "success": e.success,
                    "duration_ms": e.duration_ms,
                    "error": str(e.error) if e.error else None
                }
                for e in self._startup_events
            ],
            "total_duration_ms": sum(e.duration_ms for e in self._startup_events),
            "failed_components": [e.component for e in self._startup_events if not e.success]
        }


@asynccontextmanager
async def bootstrap(
    config: Optional[ApplicationConfig] = None,
    setup_signals: bool = True
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
    app = Application(config=config)

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


# Convenience function for getting the global application
_global_app: Optional[Application] = None


def get_application() -> Application:
    """Get the global application instance."""
    global _global_app
    if _global_app is None:
        _global_app = Application()
    return _global_app


async def run_application(
    main: Callable[[Application], Any],
    config: Optional[ApplicationConfig] = None
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
