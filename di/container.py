"""
BIBLOS v2 - Dependency Injection Container

The DI container is the circulatory system of the application,
responsible for creating, wiring, and managing the lifecycle of
all components.

Features:
    - ISP-compliant interfaces (IServiceCollection, IServiceProvider, IServiceScope)
    - Multiple lifetimes: Singleton, Scoped, Transient
    - Factory function support (sync and async)
    - Async initialization
    - Decorator-based injection
    - Lazy loading
    - Module system for organized registration
    - Named services support
    - Decorator registration for cross-cutting concerns
    - Health check integration

Architecture:
    - IServiceCollection: Registration interface (write-only)
    - IServiceProvider: Resolution interface (read-only)
    - IServiceScope: Scoped lifetime management
    - Container: Combined implementation

Usage:
    # Create and configure container
    container = Container()

    # Register services using fluent API
    (container
        .add_singleton(IDatabaseClient, PostgresClient)
        .add_scoped(IUnitOfWork, UnitOfWork)
        .add_transient(ICommandHandler, ProcessVerseHandler))

    # Register module
    container.add_module(InfrastructureModule())

    # Resolve services
    db = await container.resolve_async(IDatabaseClient)

    # Use with scopes
    async with container.create_scope_async() as scope:
        uow = await scope.resolve_async(IUnitOfWork)
        ...
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import threading
import weakref
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    FrozenSet,
    Generic,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    TypeVar,
    get_type_hints,
    ParamSpec,
    cast,
    overload,
    runtime_checkable,
)
from abc import ABC, abstractmethod
from uuid import UUID, uuid4

T = TypeVar("T")
P = ParamSpec("P")
TService = TypeVar("TService")
TImpl = TypeVar("TImpl")

logger = logging.getLogger(__name__)

# Global container instance
_container: Optional["Container"] = None
_container_lock = threading.Lock()


# =============================================================================
# LIFECYCLE PROTOCOLS
# =============================================================================


@runtime_checkable
class IDisposable(Protocol):
    """Protocol for synchronous resource cleanup."""

    def dispose(self) -> None:
        """Release resources synchronously."""
        ...


@runtime_checkable
class IAsyncDisposable(Protocol):
    """Protocol for asynchronous resource cleanup."""

    async def dispose_async(self) -> None:
        """Release resources asynchronously."""
        ...


@runtime_checkable
class IInitializable(Protocol):
    """Protocol for services requiring initialization after construction."""

    async def initialize(self) -> None:
        """Initialize the service asynchronously."""
        ...


@runtime_checkable
class IHealthCheck(Protocol):
    """Protocol for services that can report health status."""

    async def check_health(self) -> "HealthCheckResult":
        """Check the health of this service."""
        ...


@dataclass(frozen=True, slots=True)
class HealthCheckResult:
    """Result of a health check."""

    healthy: bool
    service_name: str
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def healthy_result(cls, service_name: str, message: str = "OK") -> "HealthCheckResult":
        return cls(healthy=True, service_name=service_name, message=message)

    @classmethod
    def unhealthy_result(
        cls, service_name: str, message: str, details: Optional[Dict[str, Any]] = None
    ) -> "HealthCheckResult":
        return cls(
            healthy=False,
            service_name=service_name,
            message=message,
            details=details or {},
        )


# =============================================================================
# ISP-COMPLIANT INTERFACES
# =============================================================================


class IServiceProvider(ABC):
    """
    Read-only interface for resolving services (ISP compliant).

    Use this interface when code only needs to resolve services,
    not register them. This is the interface most code should depend on.

    This is one lobe of the container's brain - the resolution pathway
    that routes requests to their implementations.
    """

    @abstractmethod
    def resolve(self, service_type: Type[T], scope: Optional["IServiceScope"] = None) -> T:
        """Resolve a service synchronously."""
        pass

    @abstractmethod
    async def resolve_async(
        self, service_type: Type[T], scope: Optional["IServiceScope"] = None
    ) -> T:
        """Resolve a service asynchronously."""
        pass

    @abstractmethod
    def is_registered(self, service_type: Type) -> bool:
        """Check if a service is registered."""
        pass

    @abstractmethod
    def try_resolve(self, service_type: Type[T]) -> Optional[T]:
        """Try to resolve a service, returning None if not registered."""
        pass

    @abstractmethod
    async def try_resolve_async(self, service_type: Type[T]) -> Optional[T]:
        """Try to resolve a service asynchronously, returning None if not registered."""
        pass

    @abstractmethod
    def resolve_all(self, service_type: Type[T]) -> List[T]:
        """Resolve all implementations of a service type."""
        pass

    @abstractmethod
    async def resolve_all_async(self, service_type: Type[T]) -> List[T]:
        """Resolve all implementations of a service type asynchronously."""
        pass

    @abstractmethod
    @contextmanager
    def create_scope(self) -> Iterator["IServiceScope"]:
        """Create a new resolution scope as a context manager."""
        yield  # type: ignore

    @abstractmethod
    @asynccontextmanager
    async def create_scope_async(self) -> AsyncIterator["IServiceScope"]:
        """Create a new resolution scope asynchronously as a context manager."""
        yield  # type: ignore


class IServiceCollection(ABC):
    """
    Write-only interface for registering services (ISP compliant).

    Use this interface when code needs to register services but
    not resolve them (e.g., module configuration).

    This is the other lobe of the container's brain - the registration
    pathway that maps abstractions to implementations.
    """

    @abstractmethod
    def add_singleton(
        self,
        service_type: Type[TService],
        implementation_type: Optional[Type[TImpl]] = None,
        *,
        lazy: bool = True,
    ) -> "IServiceCollection":
        """Register a singleton service."""
        pass

    @abstractmethod
    def add_scoped(
        self,
        service_type: Type[TService],
        implementation_type: Optional[Type[TImpl]] = None,
    ) -> "IServiceCollection":
        """Register a scoped service."""
        pass

    @abstractmethod
    def add_transient(
        self,
        service_type: Type[TService],
        implementation_type: Optional[Type[TImpl]] = None,
    ) -> "IServiceCollection":
        """Register a transient service."""
        pass

    @abstractmethod
    def add_instance(
        self,
        service_type: Type[TService],
        instance: TService,
    ) -> "IServiceCollection":
        """Register an existing instance as singleton."""
        pass

    @abstractmethod
    def add_factory(
        self,
        service_type: Type[TService],
        factory: Callable[..., TService],
        lifetime: "ServiceLifetime" = None,  # type: ignore
    ) -> "IServiceCollection":
        """Register a factory function."""
        pass

    @abstractmethod
    def add_module(self, module: "IModule") -> "IServiceCollection":
        """Register services from a module."""
        pass

    @abstractmethod
    def add_decorator(
        self,
        service_type: Type[TService],
        decorator_type: Type[TImpl],
    ) -> "IServiceCollection":
        """
        Register a decorator for a service type.

        Decorators wrap the original service, enabling cross-cutting concerns
        like logging, caching, or validation to be applied transparently.

        Usage:
            services.add_singleton(IRepository, SqlRepository)
            services.add_decorator(IRepository, CachingRepositoryDecorator)
        """
        pass

    @abstractmethod
    def try_add_singleton(
        self,
        service_type: Type[TService],
        implementation_type: Optional[Type[TImpl]] = None,
    ) -> "IServiceCollection":
        """Register a singleton only if not already registered."""
        pass

    @abstractmethod
    def try_add_scoped(
        self,
        service_type: Type[TService],
        implementation_type: Optional[Type[TImpl]] = None,
    ) -> "IServiceCollection":
        """Register a scoped service only if not already registered."""
        pass

    @abstractmethod
    def try_add_transient(
        self,
        service_type: Type[TService],
        implementation_type: Optional[Type[TImpl]] = None,
    ) -> "IServiceCollection":
        """Register a transient service only if not already registered."""
        pass

    @abstractmethod
    def replace_singleton(
        self,
        service_type: Type[TService],
        implementation_type: Optional[Type[TImpl]] = None,
    ) -> "IServiceCollection":
        """Replace an existing singleton registration."""
        pass

    @abstractmethod
    def remove(self, service_type: Type) -> "IServiceCollection":
        """Remove a service registration."""
        pass


class IServiceScope(ABC):
    """
    Interface for a service resolution scope.

    Scoped services are created once per scope and disposed
    when the scope ends. Scopes form a hierarchy - child scopes
    can access parent scope's services but not vice versa.

    This represents the container's respiratory system - the rhythm
    of create/use/dispose that maintains the application's health.
    """

    @property
    @abstractmethod
    def scope_id(self) -> str:
        """Unique identifier for this scope."""
        pass

    @property
    @abstractmethod
    def provider(self) -> IServiceProvider:
        """Get the service provider for this scope."""
        pass

    @property
    @abstractmethod
    def parent(self) -> Optional["IServiceScope"]:
        """Get the parent scope, if any."""
        pass

    @abstractmethod
    def resolve(self, service_type: Type[T]) -> T:
        """Resolve a service within this scope."""
        pass

    @abstractmethod
    async def resolve_async(self, service_type: Type[T]) -> T:
        """Resolve a service asynchronously within this scope."""
        pass

    @abstractmethod
    def try_resolve(self, service_type: Type[T]) -> Optional[T]:
        """Try to resolve a service, returning None if not registered."""
        pass

    @abstractmethod
    async def try_resolve_async(self, service_type: Type[T]) -> Optional[T]:
        """Try to resolve a service asynchronously."""
        pass

    @abstractmethod
    def create_child_scope(self) -> "IServiceScope":
        """Create a child scope."""
        pass

    @abstractmethod
    def dispose(self) -> None:
        """Dispose all scoped instances synchronously."""
        pass

    @abstractmethod
    async def dispose_async(self) -> None:
        """Dispose all scoped instances asynchronously."""
        pass

    @abstractmethod
    def __enter__(self) -> "IServiceScope":
        """Enter context manager."""
        pass

    @abstractmethod
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager."""
        pass

    @abstractmethod
    async def __aenter__(self) -> "IServiceScope":
        """Enter async context manager."""
        pass

    @abstractmethod
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager."""
        pass


class IModule(ABC):
    """
    Interface for service registration modules.

    Modules organize related service registrations into cohesive units.

    Usage:
        class InfrastructureModule(IModule):
            def configure(self, services: IServiceCollection) -> None:
                services.add_singleton(IDatabaseClient, PostgresClient)
                services.add_scoped(IUnitOfWork, UnitOfWork)
    """

    @abstractmethod
    def configure(self, services: IServiceCollection) -> None:
        """Configure services in this module."""
        pass


# =============================================================================
# SERVICE LIFETIME AND DESCRIPTORS
# =============================================================================


class ServiceLifetime(Enum):
    """Service lifetime options."""

    SINGLETON = "singleton"  # One instance for entire application
    SCOPED = "scoped"        # One instance per scope
    TRANSIENT = "transient"  # New instance every time


@dataclass
class ServiceDescriptor(Generic[T]):
    """
    Describes how a service should be created and managed.

    This is the DNA of each service - encoding its type, implementation,
    lifecycle, and creation strategy.
    """

    service_type: Type[T]
    implementation_type: Optional[Type[T]] = None
    factory: Optional[Callable[..., T]] = None
    instance: Optional[T] = None
    lifetime: ServiceLifetime = ServiceLifetime.SINGLETON
    dependencies: List[Type] = field(default_factory=list)
    async_factory: Optional[Callable[..., Any]] = None
    lazy: bool = False
    decorators: List[Type] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.implementation_type is None and self.factory is None and self.instance is None:
            self.implementation_type = self.service_type

    def with_decorator(self, decorator_type: Type) -> "ServiceDescriptor[T]":
        """Add a decorator to this descriptor."""
        new_decorators = self.decorators.copy()
        new_decorators.append(decorator_type)
        return ServiceDescriptor(
            service_type=self.service_type,
            implementation_type=self.implementation_type,
            factory=self.factory,
            instance=self.instance,
            lifetime=self.lifetime,
            dependencies=self.dependencies,
            async_factory=self.async_factory,
            lazy=self.lazy,
            decorators=new_decorators,
            metadata=self.metadata,
        )


class Scope(IServiceScope):
    """
    Scope for scoped services - implements IServiceScope.

    Scopes form a hierarchical tree, like the branching of capillaries
    from the main circulatory system. Each scope maintains its own
    instances while having access to parent scope's services.

    Usage:
        async with container.create_scope_async() as scope:
            service = await scope.resolve_async(MyService)

            # Create a child scope for nested operations
            async with scope.create_child_scope() as child:
                nested_service = await child.resolve_async(NestedService)
    """

    def __init__(
        self,
        container: "Container",
        parent: Optional["Scope"] = None,
        scope_id: Optional[str] = None,
    ):
        self._container = container
        self._parent = parent
        self._scope_id = scope_id or str(uuid4())
        self._instances: Dict[Type, Any] = {}
        self._children: List["Scope"] = []
        self._lock = threading.Lock()
        self._disposed = False
        logger.debug(f"Scope {self._scope_id} created (parent={parent._scope_id if parent else 'None'})")

    @property
    def scope_id(self) -> str:
        """Unique identifier for this scope."""
        return self._scope_id

    @property
    def provider(self) -> IServiceProvider:
        """Get the service provider for this scope."""
        return self._container

    @property
    def parent(self) -> Optional[IServiceScope]:
        """Get the parent scope, if any."""
        return self._parent

    def resolve(self, service_type: Type[T]) -> T:
        """Resolve a service within this scope."""
        self._check_disposed()
        descriptor = self._container._get_descriptor(service_type)

        if descriptor.lifetime == ServiceLifetime.SCOPED:
            with self._lock:
                # Check parent scope first if not in this scope
                if service_type not in self._instances and self._parent is not None:
                    try:
                        return self._parent.resolve(service_type)
                    except KeyError:
                        pass  # Not in parent, create here

                if service_type not in self._instances:
                    self._instances[service_type] = self._container._create_instance(
                        descriptor, self
                    )
                return self._instances[service_type]

        return self._container.resolve(service_type, self)

    async def resolve_async(self, service_type: Type[T]) -> T:
        """Resolve a service asynchronously within this scope."""
        self._check_disposed()
        descriptor = self._container._get_descriptor(service_type)

        if descriptor.lifetime == ServiceLifetime.SCOPED:
            with self._lock:
                # Check parent scope first
                if service_type not in self._instances and self._parent is not None:
                    try:
                        return await self._parent.resolve_async(service_type)
                    except KeyError:
                        pass

                if service_type not in self._instances:
                    self._instances[service_type] = await self._container._create_instance_async(
                        descriptor, self
                    )
                return self._instances[service_type]

        return await self._container.resolve_async(service_type, self)

    def try_resolve(self, service_type: Type[T]) -> Optional[T]:
        """Try to resolve a service, returning None if not registered."""
        try:
            return self.resolve(service_type)
        except KeyError:
            return None

    async def try_resolve_async(self, service_type: Type[T]) -> Optional[T]:
        """Try to resolve a service asynchronously."""
        try:
            return await self.resolve_async(service_type)
        except KeyError:
            return None

    def create_child_scope(self) -> "Scope":
        """Create a child scope."""
        self._check_disposed()
        child = Scope(self._container, parent=self)
        with self._lock:
            self._children.append(child)
        return child

    def dispose(self) -> None:
        """Dispose all scoped instances synchronously."""
        if self._disposed:
            return

        logger.debug(f"Disposing scope {self._scope_id}")

        # Dispose children first (bottom-up disposal)
        for child in self._children:
            child.dispose()
        self._children.clear()

        # Dispose our instances
        for service_type, instance in self._instances.items():
            try:
                if isinstance(instance, IDisposable):
                    instance.dispose()
                elif hasattr(instance, "dispose"):
                    instance.dispose()
                elif hasattr(instance, "close"):
                    instance.close()
            except Exception as e:
                logger.warning(f"Error disposing {service_type.__name__}: {e}")

        self._instances.clear()
        self._disposed = True

    async def dispose_async(self) -> None:
        """Dispose all scoped instances asynchronously."""
        if self._disposed:
            return

        logger.debug(f"Disposing scope {self._scope_id} asynchronously")

        # Dispose children first
        for child in self._children:
            await child.dispose_async()
        self._children.clear()

        # Dispose our instances
        for service_type, instance in self._instances.items():
            try:
                if isinstance(instance, IAsyncDisposable):
                    await instance.dispose_async()
                elif hasattr(instance, "dispose_async"):
                    dispose_method = getattr(instance, "dispose_async")
                    if asyncio.iscoroutinefunction(dispose_method):
                        await dispose_method()
                    else:
                        dispose_method()
                elif hasattr(instance, "cleanup"):
                    cleanup_method = getattr(instance, "cleanup")
                    if asyncio.iscoroutinefunction(cleanup_method):
                        await cleanup_method()
                    else:
                        cleanup_method()
                elif hasattr(instance, "close"):
                    close_method = getattr(instance, "close")
                    if asyncio.iscoroutinefunction(close_method):
                        await close_method()
                    else:
                        close_method()
            except Exception as e:
                logger.warning(f"Error disposing {service_type.__name__}: {e}")

        self._instances.clear()
        self._disposed = True

    def _check_disposed(self) -> None:
        """Raise if scope has been disposed."""
        if self._disposed:
            raise RuntimeError(f"Scope {self._scope_id} has been disposed")

    def __enter__(self) -> "Scope":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager."""
        del exc_type, exc_val, exc_tb  # Unused but required by protocol
        self.dispose()

    async def __aenter__(self) -> "Scope":
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager."""
        del exc_type, exc_val, exc_tb  # Unused but required by protocol
        await self.dispose_async()


class Container(IServiceProvider, IServiceCollection):
    """
    Dependency Injection Container - implements both IServiceProvider and IServiceCollection.

    The container is the beating heart of the application - the organ that creates,
    wires, and manages the lifecycle of all components. Like a living organism,
    it maintains homeostasis through careful lifecycle management.

    Implements ISP-compliant interfaces:
    - IServiceProvider: For resolving services (read operations)
    - IServiceCollection: For registering services (write operations)

    Usage:
        container = Container()

        # Fluent registration API (IServiceCollection)
        (container
            .add_singleton(IDatabaseClient, PostgresClient)
            .add_scoped(IUnitOfWork, UnitOfWork)
            .add_transient(ICommandHandler, ProcessVerseHandler)
            .add_module(InfrastructureModule()))

        # Resolution (IServiceProvider)
        db = await container.resolve_async(IDatabaseClient)

        # Scoped resolution
        async with container.create_scope_async() as scope:
            uow = await scope.resolve_async(IUnitOfWork)
    """

    def __init__(self) -> None:
        self._descriptors: Dict[Type, ServiceDescriptor[Any]] = {}
        self._singletons: Dict[Type, Any] = {}
        self._multi_registrations: Dict[Type, List[ServiceDescriptor[Any]]] = {}
        self._lock = threading.Lock()
        self._async_lock = asyncio.Lock()
        self._initializing: Set[Type] = set()
        self._modules_loaded: Set[Type] = set()
        self._disposed = False
        logger.debug("Container initialized")

    def register(
        self,
        service_type: Type[T],
        implementation_type: Optional[Type[T]] = None,
        lifetime: ServiceLifetime = ServiceLifetime.SINGLETON,
        lazy: bool = False,
    ) -> "Container":
        """Register a service with its implementation."""
        self._descriptors[service_type] = ServiceDescriptor(
            service_type=service_type,
            implementation_type=implementation_type or service_type,
            lifetime=lifetime,
            lazy=lazy,
        )
        return self

    def register_singleton(
        self,
        service_type: Type[T],
        implementation_type: Optional[Type[T]] = None,
        lazy: bool = True,
    ) -> "Container":
        """Register a singleton service."""
        return self.register(
            service_type,
            implementation_type,
            ServiceLifetime.SINGLETON,
            lazy,
        )

    def register_scoped(
        self,
        service_type: Type[T],
        implementation_type: Optional[Type[T]] = None,
    ) -> "Container":
        """Register a scoped service."""
        return self.register(
            service_type,
            implementation_type,
            ServiceLifetime.SCOPED,
        )

    def register_transient(
        self,
        service_type: Type[T],
        implementation_type: Optional[Type[T]] = None,
    ) -> "Container":
        """Register a transient service."""
        return self.register(
            service_type,
            implementation_type,
            ServiceLifetime.TRANSIENT,
        )

    def register_instance(
        self,
        service_type: Type[T],
        instance: T,
    ) -> "Container":
        """Register an existing instance as singleton."""
        self._descriptors[service_type] = ServiceDescriptor(
            service_type=service_type,
            instance=instance,
            lifetime=ServiceLifetime.SINGLETON,
        )
        self._singletons[service_type] = instance
        return self

    def register_factory(
        self,
        service_type: Type[T],
        factory: Callable[..., T],
        lifetime: ServiceLifetime = ServiceLifetime.SINGLETON,
    ) -> "Container":
        """Register a factory function for creating instances."""
        is_async = asyncio.iscoroutinefunction(factory)

        self._descriptors[service_type] = ServiceDescriptor(
            service_type=service_type,
            factory=factory if not is_async else None,
            async_factory=factory if is_async else None,
            lifetime=lifetime,
        )
        return self

    # =========================================================================
    # IServiceCollection Implementation (ISP-compliant fluent API)
    # =========================================================================

    def add_singleton(
        self,
        service_type: Type[TService],
        implementation_type: Optional[Type[TImpl]] = None,
        *,
        lazy: bool = True,
    ) -> IServiceCollection:
        """Register a singleton service (IServiceCollection)."""
        self._check_disposed_for_registration()
        self.register_singleton(service_type, implementation_type, lazy)  # type: ignore
        return self

    def add_scoped(
        self,
        service_type: Type[TService],
        implementation_type: Optional[Type[TImpl]] = None,
    ) -> IServiceCollection:
        """Register a scoped service (IServiceCollection)."""
        self._check_disposed_for_registration()
        self.register_scoped(service_type, implementation_type)  # type: ignore
        return self

    def add_transient(
        self,
        service_type: Type[TService],
        implementation_type: Optional[Type[TImpl]] = None,
    ) -> IServiceCollection:
        """Register a transient service (IServiceCollection)."""
        self._check_disposed_for_registration()
        self.register_transient(service_type, implementation_type)  # type: ignore
        return self

    def add_instance(
        self,
        service_type: Type[TService],
        instance: TService,
    ) -> IServiceCollection:
        """Register an existing instance as singleton (IServiceCollection)."""
        self._check_disposed_for_registration()
        self.register_instance(service_type, instance)  # type: ignore
        return self

    def add_factory(
        self,
        service_type: Type[TService],
        factory: Callable[..., TService],
        lifetime: ServiceLifetime = ServiceLifetime.SINGLETON,
    ) -> IServiceCollection:
        """Register a factory function (IServiceCollection)."""
        self._check_disposed_for_registration()
        self.register_factory(service_type, factory, lifetime)  # type: ignore
        return self

    def add_module(self, module: IModule) -> IServiceCollection:
        """
        Register services from a module (IServiceCollection).

        Modules provide organized, reusable service registrations.
        Each module is only loaded once to prevent duplicate registrations.
        """
        self._check_disposed_for_registration()
        module_type = type(module)
        if module_type in self._modules_loaded:
            logger.debug(f"Module {module_type.__name__} already loaded, skipping")
            return self

        logger.debug(f"Loading module {module_type.__name__}")
        module.configure(self)
        self._modules_loaded.add(module_type)
        return self

    def add_decorator(
        self,
        service_type: Type[TService],
        decorator_type: Type[TImpl],
    ) -> IServiceCollection:
        """
        Register a decorator for a service type (IServiceCollection).

        The decorator wraps the original service, enabling cross-cutting
        concerns like logging, caching, or validation.
        """
        self._check_disposed_for_registration()
        if service_type in self._descriptors:
            descriptor = self._descriptors[service_type]
            self._descriptors[service_type] = descriptor.with_decorator(decorator_type)
        else:
            raise KeyError(f"Cannot add decorator: service '{service_type.__name__}' not registered")
        return self

    def try_add_singleton(
        self,
        service_type: Type[TService],
        implementation_type: Optional[Type[TImpl]] = None,
    ) -> IServiceCollection:
        """Register a singleton only if not already registered."""
        if not self.is_registered(service_type):
            self.add_singleton(service_type, implementation_type)
        return self

    def try_add_scoped(
        self,
        service_type: Type[TService],
        implementation_type: Optional[Type[TImpl]] = None,
    ) -> IServiceCollection:
        """Register a scoped service only if not already registered."""
        if not self.is_registered(service_type):
            self.add_scoped(service_type, implementation_type)
        return self

    def try_add_transient(
        self,
        service_type: Type[TService],
        implementation_type: Optional[Type[TImpl]] = None,
    ) -> IServiceCollection:
        """Register a transient service only if not already registered."""
        if not self.is_registered(service_type):
            self.add_transient(service_type, implementation_type)
        return self

    def replace_singleton(
        self,
        service_type: Type[TService],
        implementation_type: Optional[Type[TImpl]] = None,
    ) -> IServiceCollection:
        """Replace an existing singleton registration."""
        self._check_disposed_for_registration()
        # Remove existing singleton if present
        if service_type in self._singletons:
            del self._singletons[service_type]
        self.register_singleton(service_type, implementation_type)  # type: ignore
        return self

    def remove(self, service_type: Type) -> IServiceCollection:
        """Remove a service registration."""
        self._check_disposed_for_registration()
        if service_type in self._descriptors:
            del self._descriptors[service_type]
        if service_type in self._singletons:
            del self._singletons[service_type]
        if service_type in self._multi_registrations:
            del self._multi_registrations[service_type]
        return self

    def _check_disposed_for_registration(self) -> None:
        """Raise if container has been disposed."""
        if self._disposed:
            raise RuntimeError("Cannot register services: container has been disposed")

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _get_descriptor(self, service_type: Type[T]) -> ServiceDescriptor[T]:
        """Get service descriptor or raise error."""
        if service_type not in self._descriptors:
            raise KeyError(f"Service '{service_type.__name__}' is not registered")
        return self._descriptors[service_type]

    def _get_dependencies(self, impl_type: Type) -> Dict[str, Type]:
        """Get constructor dependencies from type hints."""
        if not hasattr(impl_type, "__init__"):
            return {}

        try:
            hints = get_type_hints(impl_type.__init__)
        except Exception:
            hints = {}

        # Remove 'return' hint if present
        hints.pop("return", None)

        return hints

    def _create_instance(
        self,
        descriptor: ServiceDescriptor[T],
        scope: Optional[IServiceScope] = None,
    ) -> T:
        """Create a service instance synchronously."""
        # Use existing instance
        if descriptor.instance is not None:
            return descriptor.instance

        # Use factory
        if descriptor.factory is not None:
            instance = descriptor.factory()
            # Apply decorators
            return self._apply_decorators(descriptor, instance, scope)

        # Create from implementation type
        impl_type = descriptor.implementation_type
        if impl_type is None:
            raise ValueError(f"No implementation for {descriptor.service_type.__name__}")

        # Resolve dependencies
        deps = self._get_dependencies(impl_type)
        resolved_deps = {}

        for param_name, param_type in deps.items():
            if param_type in self._descriptors:
                resolved_deps[param_name] = self.resolve(param_type, scope)

        instance = impl_type(**resolved_deps)
        return self._apply_decorators(descriptor, instance, scope)

    def _apply_decorators(
        self,
        descriptor: ServiceDescriptor[T],
        instance: T,
        scope: Optional[IServiceScope] = None,
    ) -> T:
        """Apply registered decorators to an instance."""
        for decorator_type in descriptor.decorators:
            # Resolve decorator dependencies and wrap
            deps = self._get_dependencies(decorator_type)
            resolved_deps = {}
            for param_name, param_type in deps.items():
                if param_type == descriptor.service_type:
                    resolved_deps[param_name] = instance
                elif param_type in self._descriptors:
                    resolved_deps[param_name] = self.resolve(param_type, scope)
            instance = decorator_type(**resolved_deps)  # type: ignore
        return instance

    async def _create_instance_async(
        self,
        descriptor: ServiceDescriptor[T],
        scope: Optional[IServiceScope] = None,
    ) -> T:
        """Create a service instance asynchronously."""
        # Use existing instance
        if descriptor.instance is not None:
            return descriptor.instance

        # Use async factory
        if descriptor.async_factory is not None:
            instance = await descriptor.async_factory()
            return await self._apply_decorators_async(descriptor, instance, scope)

        # Use sync factory
        if descriptor.factory is not None:
            instance = descriptor.factory()
            return await self._apply_decorators_async(descriptor, instance, scope)

        # Create from implementation type
        impl_type = descriptor.implementation_type
        if impl_type is None:
            raise ValueError(f"No implementation for {descriptor.service_type.__name__}")

        # Resolve dependencies
        deps = self._get_dependencies(impl_type)
        resolved_deps = {}

        for param_name, param_type in deps.items():
            if param_type in self._descriptors:
                resolved_deps[param_name] = await self.resolve_async(param_type, scope)

        instance = impl_type(**resolved_deps)

        # Call async init if present - use protocol check
        if isinstance(instance, IInitializable):
            await instance.initialize()

        return await self._apply_decorators_async(descriptor, instance, scope)

    async def _apply_decorators_async(
        self,
        descriptor: ServiceDescriptor[T],
        instance: T,
        scope: Optional[IServiceScope] = None,
    ) -> T:
        """Apply registered decorators to an instance asynchronously."""
        for decorator_type in descriptor.decorators:
            deps = self._get_dependencies(decorator_type)
            resolved_deps = {}
            for param_name, param_type in deps.items():
                if param_type == descriptor.service_type:
                    resolved_deps[param_name] = instance
                elif param_type in self._descriptors:
                    resolved_deps[param_name] = await self.resolve_async(param_type, scope)
            instance = decorator_type(**resolved_deps)  # type: ignore
            if isinstance(instance, IInitializable):
                await instance.initialize()
        return instance

    # =========================================================================
    # IServiceProvider Implementation (ISP-compliant resolution)
    # =========================================================================

    def resolve(
        self,
        service_type: Type[T],
        scope: Optional[IServiceScope] = None,
    ) -> T:
        """Resolve a service instance."""
        descriptor = self._get_descriptor(service_type)

        if descriptor.lifetime == ServiceLifetime.SINGLETON:
            with self._lock:
                if service_type not in self._singletons:
                    # Detect circular dependencies
                    if service_type in self._initializing:
                        raise RecursionError(
                            f"Circular dependency detected for {service_type.__name__}"
                        )
                    self._initializing.add(service_type)
                    try:
                        self._singletons[service_type] = self._create_instance(
                            descriptor, scope
                        )
                    finally:
                        self._initializing.discard(service_type)
                return self._singletons[service_type]

        elif descriptor.lifetime == ServiceLifetime.SCOPED:
            if scope is None:
                raise ValueError(
                    f"Scoped service '{service_type.__name__}' requires a scope"
                )
            return scope.resolve(service_type)

        else:  # TRANSIENT
            return self._create_instance(descriptor, scope)

    async def resolve_async(
        self,
        service_type: Type[T],
        scope: Optional[IServiceScope] = None,
    ) -> T:
        """Resolve a service instance asynchronously."""
        descriptor = self._get_descriptor(service_type)

        if descriptor.lifetime == ServiceLifetime.SINGLETON:
            async with self._async_lock:
                if service_type not in self._singletons:
                    if service_type in self._initializing:
                        raise RecursionError(
                            f"Circular dependency detected for {service_type.__name__}"
                        )
                    self._initializing.add(service_type)
                    try:
                        self._singletons[service_type] = await self._create_instance_async(
                            descriptor, scope
                        )
                    finally:
                        self._initializing.discard(service_type)
                return self._singletons[service_type]

        elif descriptor.lifetime == ServiceLifetime.SCOPED:
            if scope is None:
                raise ValueError(
                    f"Scoped service '{service_type.__name__}' requires a scope"
                )
            return await scope.resolve_async(service_type)

        else:  # TRANSIENT
            return await self._create_instance_async(descriptor, scope)

    def try_resolve(self, service_type: Type[T]) -> Optional[T]:
        """Try to resolve a service, returning None if not registered."""
        try:
            return self.resolve(service_type)
        except KeyError:
            return None

    async def try_resolve_async(self, service_type: Type[T]) -> Optional[T]:
        """Try to resolve a service asynchronously, returning None if not registered."""
        try:
            return await self.resolve_async(service_type)
        except KeyError:
            return None

    def resolve_all(self, service_type: Type[T]) -> List[T]:
        """Resolve all implementations of a service type."""
        results: List[T] = []
        # Resolve the primary registration if exists
        if service_type in self._descriptors:
            results.append(self.resolve(service_type))
        # Resolve multi-registrations
        if service_type in self._multi_registrations:
            for descriptor in self._multi_registrations[service_type]:
                results.append(self._create_instance(descriptor))
        return results

    async def resolve_all_async(self, service_type: Type[T]) -> List[T]:
        """Resolve all implementations of a service type asynchronously."""
        results: List[T] = []
        if service_type in self._descriptors:
            results.append(await self.resolve_async(service_type))
        if service_type in self._multi_registrations:
            for descriptor in self._multi_registrations[service_type]:
                results.append(await self._create_instance_async(descriptor))
        return results

    @contextmanager
    def create_scope(self) -> Iterator[Scope]:
        """Create a new scope for scoped services."""
        scope = Scope(self)
        try:
            yield scope
        finally:
            scope.dispose()

    @asynccontextmanager
    async def create_scope_async(self) -> AsyncIterator[Scope]:
        """Create a new scope for scoped services asynchronously."""
        scope = Scope(self)
        try:
            yield scope
        finally:
            await scope.dispose_async()

    def is_registered(self, service_type: Type) -> bool:
        """Check if a service is registered."""
        return service_type in self._descriptors

    # =========================================================================
    # Health Checks
    # =========================================================================

    async def check_health(self) -> List[HealthCheckResult]:
        """
        Check health of all registered health-checkable services.

        Returns a list of health check results for monitoring and observability.
        """
        results: List[HealthCheckResult] = []
        for service_type, instance in self._singletons.items():
            if isinstance(instance, IHealthCheck):
                try:
                    result = await instance.check_health()
                    results.append(result)
                except Exception as e:
                    results.append(
                        HealthCheckResult.unhealthy_result(
                            service_type.__name__,
                            f"Health check failed: {e}",
                        )
                    )
        return results

    # =========================================================================
    # Lifecycle Management
    # =========================================================================

    async def dispose_async(self) -> None:
        """Dispose all singleton instances."""
        if self._disposed:
            return

        logger.debug("Disposing container")
        for service_type, instance in self._singletons.items():
            try:
                if isinstance(instance, IAsyncDisposable):
                    await instance.dispose_async()
                elif hasattr(instance, "cleanup"):
                    cleanup_method = getattr(instance, "cleanup")
                    if asyncio.iscoroutinefunction(cleanup_method):
                        await cleanup_method()
                    else:
                        cleanup_method()
                elif hasattr(instance, "close"):
                    close_method = getattr(instance, "close")
                    if asyncio.iscoroutinefunction(close_method):
                        await close_method()
                    else:
                        close_method()
            except Exception as e:
                logger.warning(f"Error disposing {service_type.__name__}: {e}")

        self._singletons.clear()
        self._disposed = True


def get_container() -> Container:
    """Get or create the global container instance."""
    global _container
    if _container is None:
        with _container_lock:
            if _container is None:
                _container = Container()
    return _container


def injectable(
    lifetime: ServiceLifetime = ServiceLifetime.SINGLETON,
    lazy: bool = True,
) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to mark a class as injectable.

    Usage:
        @injectable()
        class MyService:
            def __init__(self, db: DatabaseClient):
                self.db = db
    """
    def decorator(cls: Type[T]) -> Type[T]:
        container = get_container()
        container.register(cls, lifetime=lifetime, lazy=lazy)
        return cls
    return decorator


def inject(
    service_type: Type[T],
) -> T:
    """
    Inject a service into a parameter.

    Usage:
        def my_function(db: DatabaseClient = inject(DatabaseClient)):
            ...
    """
    return get_container().resolve(service_type)


def provide(
    service_type: Type[T],
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to automatically inject dependencies.

    Usage:
        @provide(DatabaseClient)
        def handler(request, db):
            ...
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Inject the service if not provided
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())

            if len(args) < len(params):
                container = get_container()
                service = container.resolve(service_type)
                # Add to kwargs if missing
                for param in params[len(args):]:
                    if param not in kwargs:
                        param_type = sig.parameters[param].annotation
                        if param_type == service_type:
                            kwargs[param] = service
                            break

            return func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())

            if len(args) < len(params):
                container = get_container()
                service = await container.resolve_async(service_type)
                for param in params[len(args):]:
                    if param not in kwargs:
                        param_type = sig.parameters[param].annotation
                        if param_type == service_type:
                            kwargs[param] = service
                            break

            # Cast func to coroutine function for type checker
            coro_func = cast(Callable[..., Any], func)
            return await coro_func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return wrapper

    return decorator
