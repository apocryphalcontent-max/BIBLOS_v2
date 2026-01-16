"""
BIBLOS v2 - Service Factories

The generative organs of the BIBLOS organism - responsible for the creation
of all complex objects with proper dependency injection and lifecycle management.

This module implements multiple factory patterns that integrate seamlessly
with the dependency injection container:
    - Abstract Factory: Creates families of related objects
    - Factory Method: Defers instantiation to subclasses
    - Builder: Step-by-step construction of complex objects
    - Object Pool: Reuses expensive-to-create objects
    - Prototype: Creates objects by cloning prototypes

Architecture:
    IFactory[T] → FactoryRegistry → Container → Services

All factories are registered in the DI container and can be resolved
with proper dependency injection, enabling loose coupling and testability.

Usage:
    from core.factories import PipelineFactory, AgentFactory

    # Resolve factory from container
    pipeline_factory = container.resolve(IPipelineFactory)
    pipeline = await pipeline_factory.create(config)

    # Or use builder pattern
    record = (
        GoldenRecordBuilder()
        .with_verse("GEN.1.1")
        .with_linguistic_data(data)
        .validate()
        .build()
    )
"""
from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    AsyncIterator,
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

logger = logging.getLogger("biblos.factories")

T = TypeVar("T")
TProduct = TypeVar("TProduct")
TConfig = TypeVar("TConfig")


# =============================================================================
# FACTORY INTERFACES
# =============================================================================


@runtime_checkable
class IFactory(Protocol[TProduct]):
    """
    Synchronous factory interface for creating products.

    Factories encapsulate the creation logic, allowing the system to create
    objects without knowing their concrete types or construction details.
    """

    def create(self, **kwargs: Any) -> TProduct:
        """Create a product instance."""
        ...


@runtime_checkable
class IAsyncFactory(Protocol[TProduct]):
    """
    Asynchronous factory interface for creating products.

    Use this when product creation requires async operations like
    database connections, network calls, or resource allocation.
    """

    async def create(self, **kwargs: Any) -> TProduct:
        """Create a product instance asynchronously."""
        ...


@runtime_checkable
class IConfigurableFactory(Protocol[TProduct, TConfig]):
    """
    Factory that accepts a configuration object.

    Provides type-safe configuration for product creation.
    """

    async def create(self, config: TConfig) -> TProduct:
        """Create a product with the given configuration."""
        ...


@runtime_checkable
class IPooledFactory(Protocol[TProduct]):
    """
    Factory that manages a pool of reusable objects.

    Use for expensive-to-create objects that can be reused,
    like database connections or ML model instances.
    """

    async def acquire(self) -> TProduct:
        """Acquire an object from the pool."""
        ...

    async def release(self, obj: TProduct) -> None:
        """Release an object back to the pool."""
        ...

    @property
    def pool_size(self) -> int:
        """Current pool size."""
        ...

    @property
    def available(self) -> int:
        """Number of available objects."""
        ...


class IBuilder(Protocol[TProduct]):
    """
    Builder interface for step-by-step construction.

    Builders separate the construction of complex objects from their
    representation, allowing the same construction process to create
    different representations.
    """

    def reset(self) -> "IBuilder[TProduct]":
        """Reset the builder to initial state."""
        ...

    def validate(self) -> "IBuilder[TProduct]":
        """Validate the current build state."""
        ...

    def build(self) -> TProduct:
        """Build and return the final product."""
        ...


# =============================================================================
# FACTORY BASE CLASSES
# =============================================================================


class FactoryBase(ABC, Generic[TProduct]):
    """
    Base class for synchronous factories.

    Provides common functionality like logging, metrics, and validation.
    """

    def __init__(self, name: Optional[str] = None):
        self._name = name or self.__class__.__name__
        self._creation_count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def creation_count(self) -> int:
        return self._creation_count

    @abstractmethod
    def _create_instance(self, **kwargs: Any) -> TProduct:
        """Override to implement creation logic."""
        ...

    def create(self, **kwargs: Any) -> TProduct:
        """Create a product with logging and metrics."""
        start_time = time.time()
        try:
            product = self._create_instance(**kwargs)
            self._creation_count += 1
            duration = (time.time() - start_time) * 1000
            logger.debug(f"{self._name} created product in {duration:.2f}ms")
            return product
        except Exception as e:
            logger.error(f"{self._name} creation failed: {e}")
            raise


class AsyncFactoryBase(ABC, Generic[TProduct]):
    """
    Base class for asynchronous factories.

    Provides common async functionality like logging, metrics, and timeouts.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self._name = name or self.__class__.__name__
        self._timeout = timeout
        self._creation_count = 0
        self._total_creation_time_ms = 0.0

    @property
    def name(self) -> str:
        return self._name

    @property
    def creation_count(self) -> int:
        return self._creation_count

    @property
    def average_creation_time_ms(self) -> float:
        if self._creation_count == 0:
            return 0.0
        return self._total_creation_time_ms / self._creation_count

    @abstractmethod
    async def _create_instance(self, **kwargs: Any) -> TProduct:
        """Override to implement creation logic."""
        ...

    async def create(self, **kwargs: Any) -> TProduct:
        """Create a product with logging, metrics, and timeout."""
        start_time = time.time()
        try:
            product = await asyncio.wait_for(
                self._create_instance(**kwargs),
                timeout=self._timeout,
            )
            self._creation_count += 1
            duration = (time.time() - start_time) * 1000
            self._total_creation_time_ms += duration
            logger.debug(f"{self._name} created product in {duration:.2f}ms")
            return product
        except asyncio.TimeoutError:
            logger.error(f"{self._name} creation timed out after {self._timeout}s")
            raise
        except Exception as e:
            logger.error(f"{self._name} creation failed: {e}")
            raise


class BuilderBase(ABC, Generic[TProduct]):
    """
    Base class for builders with validation support.

    Provides common builder functionality like validation, reset, and build.
    """

    def __init__(self):
        self._validation_errors: List[str] = []
        self._validated = False

    @property
    def validation_errors(self) -> List[str]:
        """Get validation errors from last validation."""
        return self._validation_errors.copy()

    @property
    def is_valid(self) -> bool:
        """Check if last validation passed."""
        return self._validated and len(self._validation_errors) == 0

    def reset(self) -> "BuilderBase[TProduct]":
        """Reset the builder to initial state."""
        self._validation_errors.clear()
        self._validated = False
        self._reset_state()
        return self

    @abstractmethod
    def _reset_state(self) -> None:
        """Override to reset product-specific state."""
        ...

    def validate(self) -> "BuilderBase[TProduct]":
        """Validate the current build state."""
        self._validation_errors.clear()
        self._validate_state()
        self._validated = True
        return self

    @abstractmethod
    def _validate_state(self) -> None:
        """Override to implement validation logic. Add errors to _validation_errors."""
        ...

    def build(self) -> TProduct:
        """Build and return the final product."""
        if not self._validated:
            self.validate()

        if self._validation_errors:
            raise ValueError(
                f"Build failed with {len(self._validation_errors)} validation errors: "
                f"{'; '.join(self._validation_errors)}"
            )

        product = self._build_product()
        self.reset()
        return product

    @abstractmethod
    def _build_product(self) -> TProduct:
        """Override to implement the actual build logic."""
        ...


# =============================================================================
# OBJECT POOL
# =============================================================================


class ObjectPoolState(Enum):
    """State of a pooled object."""
    AVAILABLE = auto()
    IN_USE = auto()
    DISPOSED = auto()


@dataclass
class PooledObject(Generic[T]):
    """Wrapper for pooled objects with metadata."""
    obj: T
    created_at: float
    last_used_at: float
    use_count: int
    state: ObjectPoolState


class ObjectPool(Generic[T]):
    """
    Generic object pool for managing expensive-to-create objects.

    Implements the Object Pool pattern with:
        - Lazy initialization
        - Maximum pool size
        - Object validation before reuse
        - Automatic cleanup of stale objects
        - Async context manager support

    Usage:
        pool = ObjectPool(factory=lambda: ExpensiveObject(), max_size=10)

        async with pool.acquire() as obj:
            # Use obj...
        # Object automatically released
    """

    def __init__(
        self,
        factory: Callable[[], T],
        max_size: int = 10,
        validation: Optional[Callable[[T], bool]] = None,
        max_idle_time: float = 300.0,
    ):
        self._factory = factory
        self._max_size = max_size
        self._validation = validation
        self._max_idle_time = max_idle_time

        self._pool: List[PooledObject[T]] = []
        self._in_use: Dict[int, PooledObject[T]] = {}
        self._lock = asyncio.Lock()
        self._total_created = 0
        self._total_reused = 0

    @property
    def pool_size(self) -> int:
        return len(self._pool) + len(self._in_use)

    @property
    def available(self) -> int:
        return len(self._pool)

    @property
    def in_use(self) -> int:
        return len(self._in_use)

    @property
    def reuse_ratio(self) -> float:
        total = self._total_created + self._total_reused
        return self._total_reused / total if total > 0 else 0.0

    async def acquire(self) -> T:
        """Acquire an object from the pool."""
        async with self._lock:
            # Try to get from pool
            current_time = time.time()

            while self._pool:
                pooled = self._pool.pop()

                # Check if stale
                if current_time - pooled.last_used_at > self._max_idle_time:
                    pooled.state = ObjectPoolState.DISPOSED
                    continue

                # Validate if validator provided
                if self._validation and not self._validation(pooled.obj):
                    pooled.state = ObjectPoolState.DISPOSED
                    continue

                # Valid object found
                pooled.state = ObjectPoolState.IN_USE
                pooled.last_used_at = current_time
                pooled.use_count += 1
                self._in_use[id(pooled.obj)] = pooled
                self._total_reused += 1
                return pooled.obj

            # Create new if pool has room
            if self.pool_size < self._max_size:
                obj = self._factory()
                pooled = PooledObject(
                    obj=obj,
                    created_at=current_time,
                    last_used_at=current_time,
                    use_count=1,
                    state=ObjectPoolState.IN_USE,
                )
                self._in_use[id(obj)] = pooled
                self._total_created += 1
                return obj

            # Pool exhausted
            raise RuntimeError("Object pool exhausted")

    async def release(self, obj: T) -> None:
        """Release an object back to the pool."""
        async with self._lock:
            obj_id = id(obj)
            if obj_id not in self._in_use:
                logger.warning("Attempted to release object not from this pool")
                return

            pooled = self._in_use.pop(obj_id)
            pooled.state = ObjectPoolState.AVAILABLE
            pooled.last_used_at = time.time()
            self._pool.append(pooled)

    @asynccontextmanager
    async def managed(self) -> AsyncIterator[T]:
        """Context manager for automatic acquire/release."""
        obj = await self.acquire()
        try:
            yield obj
        finally:
            await self.release(obj)

    async def clear(self) -> None:
        """Clear all objects from the pool."""
        async with self._lock:
            for pooled in self._pool:
                pooled.state = ObjectPoolState.DISPOSED
            self._pool.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            "pool_size": self.pool_size,
            "available": self.available,
            "in_use": self.in_use,
            "max_size": self._max_size,
            "total_created": self._total_created,
            "total_reused": self._total_reused,
            "reuse_ratio": self.reuse_ratio,
        }


# =============================================================================
# FACTORY REGISTRY
# =============================================================================


class FactoryRegistry:
    """
    Central registry for all factories.

    The registry enables discovery and resolution of factories by type,
    supporting both sync and async factories.

    Usage:
        registry = FactoryRegistry()
        registry.register(IPipelineFactory, PipelineFactory())

        factory = registry.get(IPipelineFactory)
        product = await factory.create()
    """

    def __init__(self):
        self._factories: Dict[Type, Any] = {}
        self._async_factories: Dict[Type, Any] = {}

    def register(
        self,
        factory_type: Type[TProduct],
        factory: Union[IFactory[TProduct], IAsyncFactory[TProduct]],
    ) -> None:
        """Register a factory."""
        if isinstance(factory, IAsyncFactory):
            self._async_factories[factory_type] = factory
        else:
            self._factories[factory_type] = factory
        logger.debug(f"Registered factory for {factory_type.__name__}")

    def get(self, factory_type: Type[TProduct]) -> Optional[IFactory[TProduct]]:
        """Get a synchronous factory."""
        return self._factories.get(factory_type)

    def get_async(self, factory_type: Type[TProduct]) -> Optional[IAsyncFactory[TProduct]]:
        """Get an asynchronous factory."""
        return self._async_factories.get(factory_type)

    def has(self, factory_type: Type) -> bool:
        """Check if a factory is registered."""
        return factory_type in self._factories or factory_type in self._async_factories

    def get_registered_types(self) -> List[Type]:
        """Get all registered factory types."""
        return list(set(self._factories.keys()) | set(self._async_factories.keys()))


# Global factory registry
_global_registry: Optional[FactoryRegistry] = None


def get_factory_registry() -> FactoryRegistry:
    """Get the global factory registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = FactoryRegistry()
    return _global_registry


# =============================================================================
# PIPELINE FACTORY
# =============================================================================


class PipelinePhaseSet(Enum):
    """Predefined sets of pipeline phases."""
    FULL = auto()           # All phases
    LINGUISTIC = auto()     # Linguistic analysis only
    THEOLOGICAL = auto()    # Theological analysis only
    INTERTEXTUAL = auto()   # Cross-reference analysis only
    VALIDATION = auto()     # Validation only
    MINIMAL = auto()        # Minimum viable processing
    CUSTOM = auto()         # Custom phase selection


@dataclass
class PipelineConfig:
    """
    Comprehensive configuration for pipeline creation.

    This is the DNA specification for creating a pipeline instance,
    controlling which phases are active and how they execute.
    """
    # Phase selection
    phase_set: PipelinePhaseSet = PipelinePhaseSet.FULL
    custom_phases: List[str] = field(default_factory=list)

    # Phase toggles (for FULL phase set)
    enable_linguistic: bool = True
    enable_theological: bool = True
    enable_intertextual: bool = True
    enable_validation: bool = True

    # Execution configuration
    parallel_execution: bool = True
    max_parallel_agents: int = 8
    phase_timeout_seconds: float = 300.0

    # Reliability configuration
    enable_event_sourcing: bool = True
    enable_circuit_breakers: bool = True
    enable_retries: bool = True
    retry_count: int = 3
    retry_delay_seconds: float = 1.0

    # Observability
    enable_tracing: bool = True
    enable_metrics: bool = True

    @classmethod
    def production(cls) -> "PipelineConfig":
        """Create production-ready configuration."""
        return cls(
            phase_set=PipelinePhaseSet.FULL,
            parallel_execution=True,
            max_parallel_agents=8,
            enable_event_sourcing=True,
            enable_circuit_breakers=True,
            enable_retries=True,
        )

    @classmethod
    def development(cls) -> "PipelineConfig":
        """Create development configuration with debugging features."""
        return cls(
            phase_set=PipelinePhaseSet.FULL,
            parallel_execution=False,  # Sequential for debugging
            max_parallel_agents=1,
            phase_timeout_seconds=600.0,  # Longer timeout
            enable_event_sourcing=False,
            enable_circuit_breakers=False,
        )

    @classmethod
    def testing(cls) -> "PipelineConfig":
        """Create minimal configuration for testing."""
        return cls(
            phase_set=PipelinePhaseSet.MINIMAL,
            enable_linguistic=True,
            enable_theological=False,
            enable_intertextual=False,
            enable_validation=False,
            parallel_execution=False,
            max_parallel_agents=1,
            phase_timeout_seconds=30.0,
            enable_event_sourcing=False,
            enable_circuit_breakers=False,
            enable_retries=False,
        )


class IPipelineFactory(Protocol):
    """Interface for pipeline factories."""

    async def create(self, config: PipelineConfig) -> Any:
        """Create a pipeline with the given configuration."""
        ...

    async def create_default(self) -> Any:
        """Create a pipeline with default configuration."""
        ...


class PipelineFactory(AsyncFactoryBase[Any]):
    """
    Factory for creating configured pipeline instances.

    Creates pipelines with proper phase configuration, database connections,
    and ML component wiring based on the provided configuration.
    """

    def __init__(self, container: Optional[Any] = None):
        super().__init__("PipelineFactory", timeout=60.0)
        self._container = container

    async def _create_instance(self, **kwargs: Any) -> Any:
        """Create pipeline instance."""
        config = kwargs.get("config", PipelineConfig())
        return await self._create_from_config(config)

    async def _create_from_config(self, config: PipelineConfig) -> Any:
        """Create pipeline from configuration."""
        phases = self._resolve_phases(config)

        try:
            from pipeline.unified_orchestrator import UnifiedOrchestrator

            orchestrator = UnifiedOrchestrator()
            logger.info(f"Created pipeline with {len(phases)} phases")
            return orchestrator
        except ImportError:
            logger.warning("UnifiedOrchestrator not available, returning mock")
            return {"phases": phases, "config": config}

    def _resolve_phases(self, config: PipelineConfig) -> List[str]:
        """Resolve phase names based on configuration."""
        if config.phase_set == PipelinePhaseSet.CUSTOM:
            return config.custom_phases

        phases = []

        if config.phase_set in (PipelinePhaseSet.FULL, PipelinePhaseSet.LINGUISTIC):
            if config.enable_linguistic:
                phases.append("LinguisticPhase")

        if config.phase_set in (PipelinePhaseSet.FULL, PipelinePhaseSet.THEOLOGICAL):
            if config.enable_theological:
                phases.append("TheologicalPhase")

        if config.phase_set in (PipelinePhaseSet.FULL, PipelinePhaseSet.INTERTEXTUAL):
            if config.enable_intertextual:
                phases.append("IntertextualPhase")
                phases.append("CrossReferencePhase")

        if config.phase_set in (PipelinePhaseSet.FULL, PipelinePhaseSet.VALIDATION):
            if config.enable_validation:
                phases.append("ValidationPhase")

        if config.phase_set == PipelinePhaseSet.MINIMAL:
            phases = ["LinguisticPhase"]

        return phases

    async def create_default(self) -> Any:
        """Create pipeline with default configuration."""
        return await self.create(config=PipelineConfig())

    async def create_production(self) -> Any:
        """Create production-ready pipeline."""
        return await self.create(config=PipelineConfig.production())

    async def create_development(self) -> Any:
        """Create development pipeline."""
        return await self.create(config=PipelineConfig.development())

    async def create_testing(self) -> Any:
        """Create testing pipeline."""
        return await self.create(config=PipelineConfig.testing())


# =============================================================================
# AGENT FACTORY
# =============================================================================


class AgentCategory(Enum):
    """Categories of SDES extraction agents."""
    LINGUISTIC = "linguistic"
    THEOLOGICAL = "theological"
    INTERTEXTUAL = "intertextual"
    VALIDATION = "validation"


@dataclass
class AgentConfig:
    """
    Configuration for agent creation.

    Specifies how an individual extraction agent should be configured,
    including its operational parameters and dependencies.
    """
    name: str
    category: AgentCategory = AgentCategory.LINGUISTIC
    extraction_type: str = "general"

    # Execution parameters
    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_delay_seconds: float = 1.0

    # Caching and optimization
    enable_caching: bool = True
    cache_ttl_seconds: float = 3600.0

    # Observability
    enable_tracing: bool = True
    enable_metrics: bool = True

    # LLM configuration (if agent uses LLM)
    llm_model: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 4096


class IAgentFactory(Protocol):
    """Interface for agent factories."""

    async def create(self, config: AgentConfig, **dependencies: Any) -> Any:
        """Create an agent with the given configuration."""
        ...

    def get_available_agents(self) -> List[str]:
        """Get list of available agent names."""
        ...

    def get_agents_by_category(self, category: AgentCategory) -> List[str]:
        """Get agents in a specific category."""
        ...


class AgentFactory(AsyncFactoryBase[Any]):
    """
    Factory for creating configured extraction agents.

    Handles agent instantiation with proper dependency injection
    for database clients, ML models, and observability.

    The factory maintains a registry of agent types and can create
    both individual agents and complete agent suites.
    """

    # Registry of agent types to their classes and categories
    _agent_registry: Dict[str, Tuple[Type, AgentCategory]] = {}

    def __init__(self, container: Optional[Any] = None):
        super().__init__("AgentFactory", timeout=30.0)
        self._container = container

    @classmethod
    def register(
        cls,
        name: str,
        agent_class: Type,
        category: AgentCategory = AgentCategory.LINGUISTIC,
    ) -> None:
        """Register an agent class."""
        cls._agent_registry[name.upper()] = (agent_class, category)
        logger.debug(f"Registered agent: {name} ({category.value})")

    @classmethod
    def get_available_agents(cls) -> List[str]:
        """Get list of available agent names."""
        return list(cls._agent_registry.keys())

    @classmethod
    def get_agents_by_category(cls, category: AgentCategory) -> List[str]:
        """Get agents in a specific category."""
        return [
            name for name, (_, cat) in cls._agent_registry.items()
            if cat == category
        ]

    async def _create_instance(self, **kwargs: Any) -> Any:
        """Create agent instance."""
        config = kwargs.pop("config", None)
        if config is None:
            raise ValueError("AgentConfig is required")

        return await self._create_from_config(config, kwargs)

    async def _create_from_config(
        self,
        config: AgentConfig,
        dependencies: Dict[str, Any],
    ) -> Any:
        """Create agent from configuration."""
        agent_entry = self._agent_registry.get(config.name.upper())
        if agent_entry is None:
            raise ValueError(f"Unknown agent type: {config.name}")

        agent_class, _ = agent_entry

        # Resolve dependencies from container if available
        if self._container and not dependencies:
            dependencies = self._resolve_dependencies(agent_class)

        # Create agent with injected dependencies
        agent = agent_class(**dependencies)
        logger.info(f"Created agent: {config.name}")
        return agent

    def _resolve_dependencies(self, agent_class: Type) -> Dict[str, Any]:
        """Resolve dependencies for an agent class from the container."""
        # This would integrate with the DI container to resolve dependencies
        # For now, return empty dict
        return {}

    async def create_suite(
        self,
        category: AgentCategory,
        **dependencies: Any,
    ) -> Dict[str, Any]:
        """Create all agents in a category."""
        agents = {}
        agent_names = self.get_agents_by_category(category)

        for name in agent_names:
            config = AgentConfig(
                name=name,
                category=category,
                extraction_type=category.value,
            )
            try:
                agents[name] = await self.create(config=config, **dependencies)
            except Exception as e:
                logger.warning(f"Failed to create agent {name}: {e}")

        return agents

    async def create_linguistic_suite(self, **dependencies: Any) -> Dict[str, Any]:
        """Create all linguistic analysis agents."""
        return await self.create_suite(AgentCategory.LINGUISTIC, **dependencies)

    async def create_theological_suite(self, **dependencies: Any) -> Dict[str, Any]:
        """Create all theological analysis agents."""
        return await self.create_suite(AgentCategory.THEOLOGICAL, **dependencies)

    async def create_intertextual_suite(self, **dependencies: Any) -> Dict[str, Any]:
        """Create all intertextual analysis agents."""
        return await self.create_suite(AgentCategory.INTERTEXTUAL, **dependencies)


# =============================================================================
# DATABASE CLIENT FACTORY
# =============================================================================


class DatabaseType(Enum):
    """Supported database types."""
    POSTGRES = "postgres"
    NEO4J = "neo4j"
    QDRANT = "qdrant"
    REDIS = "redis"


@dataclass
class DatabaseConfig:
    """
    Unified configuration for database client creation.

    Supports all database types used by BIBLOS with sensible defaults.
    """
    db_type: DatabaseType = DatabaseType.POSTGRES

    # Connection
    host: str = "localhost"
    port: int = 5432
    database: str = "biblos_v2"
    user: str = "biblos"
    password: str = ""

    # Connection pool
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout_seconds: float = 30.0

    # Timeouts
    connect_timeout_seconds: float = 10.0
    query_timeout_seconds: float = 30.0

    # SSL
    ssl_enabled: bool = False
    ssl_cert_path: Optional[str] = None

    @classmethod
    def postgres(
        cls,
        host: str = "localhost",
        port: int = 5432,
        database: str = "biblos_v2",
        **kwargs: Any,
    ) -> "DatabaseConfig":
        """Create PostgreSQL configuration."""
        return cls(
            db_type=DatabaseType.POSTGRES,
            host=host,
            port=port,
            database=database,
            **kwargs,
        )

    @classmethod
    def neo4j(
        cls,
        host: str = "localhost",
        port: int = 7687,
        **kwargs: Any,
    ) -> "DatabaseConfig":
        """Create Neo4j configuration."""
        return cls(
            db_type=DatabaseType.NEO4J,
            host=host,
            port=port,
            **kwargs,
        )

    @classmethod
    def qdrant(
        cls,
        host: str = "localhost",
        port: int = 6333,
        **kwargs: Any,
    ) -> "DatabaseConfig":
        """Create Qdrant configuration."""
        return cls(
            db_type=DatabaseType.QDRANT,
            host=host,
            port=port,
            **kwargs,
        )

    def get_connection_url(self) -> str:
        """Get the connection URL for this database."""
        if self.db_type == DatabaseType.POSTGRES:
            return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        elif self.db_type == DatabaseType.NEO4J:
            return f"bolt://{self.host}:{self.port}"
        elif self.db_type == DatabaseType.QDRANT:
            return f"http://{self.host}:{self.port}"
        else:
            return f"{self.host}:{self.port}"


class IDatabaseClientFactory(Protocol):
    """Interface for database client factories."""

    async def create(self, config: DatabaseConfig) -> Any:
        """Create a database client with the given configuration."""
        ...


class DatabaseClientFactory(AsyncFactoryBase[Any]):
    """
    Factory for creating database clients.

    Supports all database types used by BIBLOS with proper
    connection pooling and lifecycle management.
    """

    def __init__(self):
        super().__init__("DatabaseClientFactory", timeout=30.0)
        self._pools: Dict[str, ObjectPool[Any]] = {}

    async def _create_instance(self, **kwargs: Any) -> Any:
        """Create database client instance."""
        config = kwargs.get("config")
        if config is None:
            config = DatabaseConfig()

        return await self._create_from_config(config)

    async def _create_from_config(self, config: DatabaseConfig) -> Any:
        """Create client from configuration."""
        if config.db_type == DatabaseType.POSTGRES:
            return await self._create_postgres(config)
        elif config.db_type == DatabaseType.NEO4J:
            return await self._create_neo4j(config)
        elif config.db_type == DatabaseType.QDRANT:
            return await self._create_qdrant(config)
        else:
            raise ValueError(f"Unsupported database type: {config.db_type}")

    async def _create_postgres(self, config: DatabaseConfig) -> Any:
        """Create PostgreSQL client."""
        try:
            from db.postgres_optimized import PostgresClient

            client = PostgresClient(
                database_url=config.get_connection_url(),
                pool_size=config.pool_size,
                max_overflow=config.max_overflow,
            )
            await client.initialize()
            return client
        except ImportError:
            logger.warning("PostgresClient not available")
            return {"type": "postgres", "config": config}

    async def _create_neo4j(self, config: DatabaseConfig) -> Any:
        """Create Neo4j client."""
        try:
            from db.neo4j_optimized import Neo4jClient, Neo4jConfig

            neo4j_config = Neo4jConfig(
                uri=config.get_connection_url(),
                user=config.user,
                password=config.password,
            )
            client = Neo4jClient(config=neo4j_config)
            await client.connect()
            return client
        except ImportError:
            logger.warning("Neo4jClient not available")
            return {"type": "neo4j", "config": config}

    async def _create_qdrant(self, config: DatabaseConfig) -> Any:
        """Create Qdrant client."""
        try:
            from db.qdrant_client import QdrantVectorStore

            client = QdrantVectorStore(host=config.host, port=config.port)
            await client.connect()
            return client
        except ImportError:
            logger.warning("QdrantVectorStore not available")
            return {"type": "qdrant", "config": config}

    # Convenience methods
    async def create_postgres(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "biblos_v2",
        **kwargs: Any,
    ) -> Any:
        """Create a PostgreSQL client with specific parameters."""
        config = DatabaseConfig.postgres(host=host, port=port, database=database, **kwargs)
        return await self.create(config=config)

    async def create_neo4j(
        self,
        host: str = "localhost",
        port: int = 7687,
        **kwargs: Any,
    ) -> Any:
        """Create a Neo4j client with specific parameters."""
        config = DatabaseConfig.neo4j(host=host, port=port, **kwargs)
        return await self.create(config=config)

    async def create_qdrant(
        self,
        host: str = "localhost",
        port: int = 6333,
        **kwargs: Any,
    ) -> Any:
        """Create a Qdrant client with specific parameters."""
        config = DatabaseConfig.qdrant(host=host, port=port, **kwargs)
        return await self.create(config=config)


# =============================================================================
# ML ENGINE FACTORY
# =============================================================================


class MLDeviceType(Enum):
    """ML compute device types."""
    CUDA = "cuda"
    CPU = "cpu"
    MPS = "mps"  # Apple Silicon


@dataclass
class MLEngineConfig:
    """
    Configuration for ML engine creation.

    Controls device selection, model loading, and inference parameters.
    """
    device: MLDeviceType = MLDeviceType.CUDA
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    model_path: Optional[str] = None

    # Inference parameters
    batch_size: int = 32
    max_sequence_length: int = 512

    # Optimization
    enable_fp16: bool = True
    enable_quantization: bool = False
    compile_model: bool = False

    # Caching
    enable_embedding_cache: bool = True
    cache_size: int = 10000

    @classmethod
    def for_embedding(
        cls,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: MLDeviceType = MLDeviceType.CUDA,
    ) -> "MLEngineConfig":
        """Create configuration for embedding model."""
        return cls(device=device, model_name=model_name)

    @classmethod
    def for_gnn(
        cls,
        device: MLDeviceType = MLDeviceType.CUDA,
    ) -> "MLEngineConfig":
        """Create configuration for GNN model."""
        return cls(
            device=device,
            model_name="gnn_cross_ref",
            enable_fp16=True,
            compile_model=True,
        )


class IMLEngineFactory(Protocol):
    """Interface for ML engine factories."""

    async def create_embedding_model(self, config: MLEngineConfig) -> Any:
        """Create an embedding model."""
        ...

    async def create_gnn_model(self, config: MLEngineConfig) -> Any:
        """Create a GNN model."""
        ...

    async def create_inference_pipeline(
        self,
        embedding_model: Any,
        gnn_model: Any,
        vector_store: Any,
    ) -> Any:
        """Create the full inference pipeline."""
        ...


class MLEngineFactory(AsyncFactoryBase[Any]):
    """
    Factory for creating ML engine instances.

    Handles model loading, device selection, and optimization
    for all ML components in the BIBLOS system.
    """

    def __init__(self):
        super().__init__("MLEngineFactory", timeout=120.0)  # Models can take time to load
        self._model_cache: Dict[str, Any] = {}

    async def _create_instance(self, **kwargs: Any) -> Any:
        """Create ML engine instance."""
        config = kwargs.get("config", MLEngineConfig())
        model_type = kwargs.get("model_type", "embedding")

        if model_type == "embedding":
            return await self._create_embedding_model(config)
        elif model_type == "gnn":
            return await self._create_gnn_model(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    async def _create_embedding_model(self, config: MLEngineConfig) -> Any:
        """Create embedding model."""
        cache_key = f"embedding:{config.model_name}:{config.device.value}"

        if cache_key in self._model_cache:
            logger.debug(f"Using cached embedding model: {config.model_name}")
            return self._model_cache[cache_key]

        try:
            from ml.embeddings.domain_embedders import MultiDomainEmbedder

            embedder = MultiDomainEmbedder(
                model_name=config.model_name,
                device=config.device.value,
            )
            await embedder.initialize()

            self._model_cache[cache_key] = embedder
            return embedder
        except ImportError:
            logger.warning("MultiDomainEmbedder not available")
            return {"type": "embedding", "config": config}

    async def _create_gnn_model(self, config: MLEngineConfig) -> Any:
        """Create GNN model."""
        try:
            from ml.models.gnn_discovery import CrossRefGNN

            model = CrossRefGNN(device=config.device.value)
            return model
        except ImportError:
            logger.warning("CrossRefGNN not available")
            return {"type": "gnn", "config": config}

    async def create_embedding_model(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = "cuda",
    ) -> Any:
        """Create an embedding model with specific parameters."""
        config = MLEngineConfig.for_embedding(
            model_name=model_name,
            device=MLDeviceType(device),
        )
        return await self.create(config=config, model_type="embedding")

    async def create_gnn_model(self, device: str = "cuda") -> Any:
        """Create a GNN model for cross-reference prediction."""
        config = MLEngineConfig.for_gnn(device=MLDeviceType(device))
        return await self.create(config=config, model_type="gnn")

    async def create_inference_pipeline(
        self,
        embedding_model: Any,
        gnn_model: Any,
        vector_store: Any,
    ) -> Any:
        """Create the full inference pipeline."""
        try:
            from ml.inference.pipeline import InferencePipeline, InferenceConfig

            config = InferenceConfig()
            pipeline = InferencePipeline(config)
            await pipeline.initialize()
            return pipeline
        except ImportError:
            logger.warning("InferencePipeline not available")
            return {
                "type": "inference_pipeline",
                "embedding_model": embedding_model,
                "gnn_model": gnn_model,
                "vector_store": vector_store,
            }


# =============================================================================
# GOLDEN RECORD BUILDER
# =============================================================================


class QualityTier(Enum):
    """Quality certification tiers for Golden Records."""
    UNCERTIFIED = 0
    BRONZE = 1
    SILVER = 2
    GOLD = 3
    PLATINUM = 4


@dataclass(frozen=True)
class GoldenRecord:
    """
    Immutable Golden Record - the certified output of pipeline processing.

    A Golden Record represents the highest quality, verified extraction
    results for a biblical verse, combining all analysis dimensions.
    """
    record_id: UUID
    verse_id: str
    linguistic: Dict[str, Any]
    theological: Dict[str, Any]
    intertextual: Dict[str, Any]
    cross_references: Tuple[Dict[str, Any], ...]
    confidence: float
    quality_tier: QualityTier
    certified_at: float
    certified_by: str
    metadata: Dict[str, Any]


class GoldenRecordBuilder(BuilderBase[GoldenRecord]):
    """
    Builder for constructing Golden Record entries step by step.

    Provides a fluent API for building Golden Records with proper
    validation at each step.

    Usage:
        record = (
            GoldenRecordBuilder()
            .with_verse("GEN.1.1")
            .with_linguistic_data(linguistic_results)
            .with_theological_data(theological_results)
            .with_cross_references(crossrefs)
            .with_confidence(0.95)
            .validate()
            .build()
        )
    """

    def __init__(self) -> None:
        super().__init__()
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset all build state."""
        self._verse_id: Optional[str] = None
        self._linguistic: Dict[str, Any] = {}
        self._theological: Dict[str, Any] = {}
        self._intertextual: Dict[str, Any] = {}
        self._cross_references: List[Dict[str, Any]] = []
        self._confidence: float = 0.0
        self._quality_tier: QualityTier = QualityTier.UNCERTIFIED
        self._certified_by: str = "system"
        self._metadata: Dict[str, Any] = {}

    def _validate_state(self) -> None:
        """Validate the current build state."""
        if not self._verse_id:
            self._validation_errors.append("verse_id is required")

        if not self._verse_id or not self._is_valid_verse_reference(self._verse_id):
            self._validation_errors.append("verse_id must be a valid reference (e.g., GEN.1.1)")

        if self._confidence < 0.0 or self._confidence > 1.0:
            self._validation_errors.append("confidence must be between 0.0 and 1.0")

        # Validate cross-references
        for i, crossref in enumerate(self._cross_references):
            if "target_ref" not in crossref:
                self._validation_errors.append(f"cross_reference[{i}] missing target_ref")

    def _is_valid_verse_reference(self, ref: str) -> bool:
        """Validate verse reference format."""
        parts = ref.split(".")
        if len(parts) != 3:
            return False
        book, chapter, verse = parts
        return (
            len(book) == 3
            and book.isalpha()
            and chapter.isdigit()
            and verse.isdigit()
        )

    def _build_product(self) -> GoldenRecord:
        """Build the Golden Record."""
        return GoldenRecord(
            record_id=uuid4(),
            verse_id=self._verse_id or "",
            linguistic=self._linguistic.copy(),
            theological=self._theological.copy(),
            intertextual=self._intertextual.copy(),
            cross_references=tuple(self._cross_references),
            confidence=self._confidence,
            quality_tier=self._quality_tier,
            certified_at=time.time(),
            certified_by=self._certified_by,
            metadata=self._metadata.copy(),
        )

    # Fluent API methods

    def with_verse(self, verse_id: str) -> "GoldenRecordBuilder":
        """Set the verse ID."""
        self._verse_id = verse_id
        return self

    def with_linguistic_data(self, data: Dict[str, Any]) -> "GoldenRecordBuilder":
        """Add linguistic analysis data."""
        self._linguistic = data
        return self

    def with_theological_data(self, data: Dict[str, Any]) -> "GoldenRecordBuilder":
        """Add theological analysis data."""
        self._theological = data
        return self

    def with_intertextual_data(self, data: Dict[str, Any]) -> "GoldenRecordBuilder":
        """Add intertextual analysis data."""
        self._intertextual = data
        return self

    def with_cross_reference(
        self,
        target_ref: str,
        connection_type: str,
        strength: str = "moderate",
        confidence: float = 0.8,
        **kwargs: Any,
    ) -> "GoldenRecordBuilder":
        """Add a single cross-reference."""
        crossref = {
            "target_ref": target_ref,
            "connection_type": connection_type,
            "strength": strength,
            "confidence": confidence,
            **kwargs,
        }
        self._cross_references.append(crossref)
        return self

    def with_cross_references(self, crossrefs: List[Dict[str, Any]]) -> "GoldenRecordBuilder":
        """Add multiple cross-references."""
        self._cross_references.extend(crossrefs)
        return self

    def with_confidence(self, confidence: float) -> "GoldenRecordBuilder":
        """Set overall confidence score."""
        self._confidence = confidence
        return self

    def with_quality_tier(self, tier: QualityTier) -> "GoldenRecordBuilder":
        """Set quality certification tier."""
        self._quality_tier = tier
        return self

    def with_certification(
        self,
        tier: QualityTier,
        certified_by: str,
    ) -> "GoldenRecordBuilder":
        """Set certification details."""
        self._quality_tier = tier
        self._certified_by = certified_by
        return self

    def with_metadata(self, **kwargs: Any) -> "GoldenRecordBuilder":
        """Add metadata."""
        self._metadata.update(kwargs)
        return self


# =============================================================================
# COMPOSITE FACTORY
# =============================================================================


class SystemFactory:
    """
    Composite factory that provides access to all system factories.

    This is the central factory hub that coordinates the creation of
    all major system components, ensuring proper dependency wiring.

    Usage:
        system = SystemFactory()

        pipeline = await system.pipeline.create_production()
        agents = await system.agent.create_linguistic_suite()
        db = await system.database.create_postgres()
    """

    def __init__(self, container: Optional[Any] = None):
        self._container = container
        self._pipeline = PipelineFactory(container)
        self._agent = AgentFactory(container)
        self._database = DatabaseClientFactory()
        self._ml = MLEngineFactory()

    @property
    def pipeline(self) -> PipelineFactory:
        """Pipeline factory."""
        return self._pipeline

    @property
    def agent(self) -> AgentFactory:
        """Agent factory."""
        return self._agent

    @property
    def database(self) -> DatabaseClientFactory:
        """Database client factory."""
        return self._database

    @property
    def ml(self) -> MLEngineFactory:
        """ML engine factory."""
        return self._ml

    def golden_record_builder(self) -> GoldenRecordBuilder:
        """Create a new Golden Record builder."""
        return GoldenRecordBuilder()

    async def create_complete_stack(
        self,
        pipeline_config: Optional[PipelineConfig] = None,
    ) -> Dict[str, Any]:
        """
        Create a complete system stack.

        Returns all major components properly wired together.
        """
        # Create database clients
        postgres = await self._database.create_postgres()
        neo4j = await self._database.create_neo4j()
        qdrant = await self._database.create_qdrant()

        # Create ML components
        embedder = await self._ml.create_embedding_model()
        gnn = await self._ml.create_gnn_model()
        inference = await self._ml.create_inference_pipeline(embedder, gnn, qdrant)

        # Create agents with dependencies
        agents = await self._agent.create_linguistic_suite(
            db_client=postgres,
            graph_client=neo4j,
            vector_store=qdrant,
        )

        # Create pipeline
        config = pipeline_config or PipelineConfig.production()
        pipeline = await self._pipeline.create(config=config)

        return {
            "databases": {
                "postgres": postgres,
                "neo4j": neo4j,
                "qdrant": qdrant,
            },
            "ml": {
                "embedder": embedder,
                "gnn": gnn,
                "inference": inference,
            },
            "agents": agents,
            "pipeline": pipeline,
        }


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Interfaces
    "IFactory",
    "IAsyncFactory",
    "IConfigurableFactory",
    "IPooledFactory",
    "IBuilder",
    # Base Classes
    "FactoryBase",
    "AsyncFactoryBase",
    "BuilderBase",
    # Object Pool
    "ObjectPoolState",
    "PooledObject",
    "ObjectPool",
    # Registry
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
    # Database Factory
    "DatabaseType",
    "DatabaseConfig",
    "IDatabaseClientFactory",
    "DatabaseClientFactory",
    # ML Factory
    "MLDeviceType",
    "MLEngineConfig",
    "IMLEngineFactory",
    "MLEngineFactory",
    # Golden Record Builder
    "QualityTier",
    "GoldenRecord",
    "GoldenRecordBuilder",
    # Composite Factory
    "SystemFactory",
]
