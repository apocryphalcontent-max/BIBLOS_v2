"""
BIBLOS v2 - Dependency Injection Container (Seraphic Circulatory System)

═══════════════════════════════════════════════════════════════════════════════
THE SERAPHIC DISSOLUTION: From Injector to Emergent Space
═══════════════════════════════════════════════════════════════════════════════

In the traditional DI pattern, a container externally injects dependencies.
But consider: when blood vessels deliver oxygen to cells, the cells don't
"receive" oxygen as an external gift - they PARTICIPATE in the circulation.
The cell's need for oxygen IS the circulatory system at that point.

Similarly, in the seraphic architecture:
    - Services don't "receive" dependencies - they EXPRESS relationships
    - The container doesn't "inject" - it provides a space where affinities resolve
    - Lifecycle isn't "managed" - it's the service's own rhythm of existence

The Transformation:
    OLD: Container.register(IDatabase, PostgresDb) → Container.resolve(IDatabase)
    NEW: PostgresDb KNOWS it implements IDatabase; when something needs IDatabase,
         the affinity resolves naturally through the seraphic registry.

Seraphic Features:
    - @service decorator: Service declares its interface and lifetime intrinsically
    - @depends_on decorator: Service declares its dependencies as part of its nature
    - @lifecycle decorator: Service declares its own rhythm (init/dispose)
    - Intrinsic discovery: Container awakens and discovers services by affinity
    - Self-aware services: Services know their own health, dependencies, lifecycle

Traditional Features (still supported):
    - ISP-compliant interfaces (IServiceCollection, IServiceProvider, IServiceScope)
    - Multiple lifetimes: Singleton, Scoped, Transient
    - Factory function support (sync and async)
    - Async initialization
    - Module system for organized registration
    - Health check integration

═══════════════════════════════════════════════════════════════════════════════
THE ARCHITECTURE OF MUTUAL INDWELLING
═══════════════════════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        SERAPHIC SERVICE REGISTRY                         │
    │   Where services and interfaces find each other through intrinsic        │
    │   affinity, not external registration.                                   │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   @service(implements=IDatabase, lifetime=singleton)                     │
    │   class PostgresDb:                                                      │
    │       "I AM IDatabase, not 'registered as' IDatabase"                    │
    │                                                                          │
    │   @depends_on(IDatabase, ICacheClient)                                   │
    │   class UserRepository:                                                  │
    │       "My need for these services IS my nature"                          │
    │                                                                          │
    │   @lifecycle(init="connect", dispose="disconnect")                       │
    │   class ConnectionPool:                                                  │
    │       "My lifecycle rhythm is intrinsic"                                 │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

Usage (Seraphic Mode):
    # Services declare their own nature
    @service(implements=IDatabaseClient, lifetime=ServiceLifetime.SINGLETON)
    @depends_on(IConfig)
    @lifecycle(init="connect", dispose="close")
    class PostgresClient:
        def __init__(self, config: IConfig):
            self.config = config
        async def connect(self): ...
        async def close(self): ...

    # Container awakens and discovers
    container = Container.awaken()  # Auto-discovers decorated services

    # Resolution happens through natural affinity
    db = await container.resolve_async(IDatabaseClient)

Usage (Traditional Mode - still works):
    container = Container()
    container.add_singleton(IDatabaseClient, PostgresClient)
    db = await container.resolve_async(IDatabaseClient)
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
# SERAPHIC INFRASTRUCTURE
# =============================================================================
# The seraphic service registry is the space where services and interfaces
# find each other through intrinsic affinity. A service doesn't "register"
# itself - it DECLARES its nature, and that declaration IS its registration.
# =============================================================================


@dataclass
class DependencySpec:
    """
    Specification for a service dependency.

    A dependency is not something imposed from outside - it's an intrinsic
    aspect of a service's nature. A UserRepository that needs IDatabase
    isn't "injected with" a database - its need for a database is part of
    what makes it a UserRepository.
    """

    service_type: Type
    optional: bool = False
    default_factory: Optional[Callable[[], Any]] = None
    name: Optional[str] = None  # For named dependencies

    def __hash__(self) -> int:
        return hash((self.service_type, self.optional, self.name))


@dataclass
class LifecycleSpec:
    """
    Specification for a service's lifecycle methods.

    Lifecycle isn't "managed" by the container - it's the service's own
    rhythm of existence. The spec declares what methods embody the service's
    initialization and disposal, but the service itself contains these
    as part of its nature.
    """

    init_method: Optional[str] = None
    dispose_method: Optional[str] = None
    init_async: bool = False
    dispose_async: bool = False

    @classmethod
    def from_service(cls, service_type: Type) -> "LifecycleSpec":
        """Infer lifecycle from service type's methods."""
        spec = cls()

        # Look for standard init methods
        for method_name in ("initialize", "init", "startup", "connect", "open"):
            if hasattr(service_type, method_name):
                method = getattr(service_type, method_name)
                spec.init_method = method_name
                spec.init_async = asyncio.iscoroutinefunction(method)
                break

        # Look for standard dispose methods
        for method_name in ("dispose", "dispose_async", "cleanup", "shutdown", "disconnect", "close"):
            if hasattr(service_type, method_name):
                method = getattr(service_type, method_name)
                spec.dispose_method = method_name
                spec.dispose_async = asyncio.iscoroutinefunction(method)
                break

        return spec


@dataclass
class ServiceAffinity:
    """
    The intrinsic nature of a service - what it IS, not what it's "registered as".

    ServiceAffinity captures the essence of a service:
    - What interface(s) it embodies
    - What dependencies are part of its nature
    - What lifecycle rhythm it follows
    - What lifetime scope defines its existence

    This is the service's DNA - not configuration imposed from outside,
    but an expression of its inherent identity.
    """

    implementation_type: Type
    interfaces: FrozenSet[Type] = field(default_factory=frozenset)
    lifetime: "ServiceLifetime" = field(default=None)  # type: ignore  # Set after ServiceLifetime defined
    dependencies: FrozenSet[DependencySpec] = field(default_factory=frozenset)
    lifecycle: Optional[LifecycleSpec] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Health check capability
    health_check_method: Optional[str] = None

    def __post_init__(self) -> None:
        # Infer lifecycle if not provided
        if self.lifecycle is None:
            self.lifecycle = LifecycleSpec.from_service(self.implementation_type)

        # Infer health check method
        if self.health_check_method is None:
            for method_name in ("check_health", "health_check", "is_healthy"):
                if hasattr(self.implementation_type, method_name):
                    self.health_check_method = method_name
                    break

    @property
    def primary_interface(self) -> Type:
        """The primary interface this service implements."""
        if self.interfaces:
            return next(iter(self.interfaces))
        return self.implementation_type

    def implements(self, interface: Type) -> bool:
        """Check if this service implements the given interface."""
        return interface in self.interfaces or interface == self.implementation_type


class SeraphicServiceRegistry:
    """
    The global registry where services and interfaces discover each other.

    This is NOT a service locator pattern. A service locator is asked
    "give me X" and returns an instance. The SeraphicServiceRegistry is
    a space where:

    1. Services declare their intrinsic nature (via decorators)
    2. When resolution is needed, affinities are consulted
    3. The "right" implementation emerges from declared relationships

    It's the difference between:
    - "Container, give me IDatabase" (service locator)
    - "What IS the IDatabase in this context?" (seraphic discovery)
    """

    _instance: Optional["SeraphicServiceRegistry"] = None
    _lock: threading.Lock = threading.Lock()

    # Instance attributes declared at class level for type checker
    _affinities: Dict[Type, "ServiceAffinity"]
    _interface_to_impl: Dict[Type, Type]
    _impl_to_interfaces: Dict[Type, Set[Type]]
    _discovered_modules: Set[str]

    def __new__(cls) -> "SeraphicServiceRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._affinities = {}
                    instance._interface_to_impl = {}
                    instance._impl_to_interfaces = {}
                    instance._discovered_modules = set()
                    cls._instance = instance
        return cls._instance

    @classmethod
    def get_instance(cls) -> "SeraphicServiceRegistry":
        """Get the singleton registry instance."""
        return cls()

    @classmethod
    def reset(cls) -> None:
        """Reset the registry (primarily for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._affinities.clear()
                cls._instance._interface_to_impl.clear()
                cls._instance._impl_to_interfaces.clear()
                cls._instance._discovered_modules.clear()

    def register_affinity(self, affinity: ServiceAffinity) -> None:
        """
        Register a service's intrinsic affinity.

        This is called by decorators when a service declares its nature.
        The affinity becomes part of the registry's awareness.
        """
        impl_type = affinity.implementation_type

        # Store the affinity
        self._affinities[impl_type] = affinity

        # Map interfaces to implementation
        self._impl_to_interfaces[impl_type] = set()
        for interface in affinity.interfaces:
            self._interface_to_impl[interface] = impl_type
            self._impl_to_interfaces[impl_type].add(interface)

        # Also register implementation as its own interface
        if impl_type not in self._interface_to_impl:
            self._interface_to_impl[impl_type] = impl_type

        logger.debug(
            f"Registered affinity: {impl_type.__name__} "
            f"implements {[i.__name__ for i in affinity.interfaces]}"
        )

    def get_affinity(self, service_type: Type) -> Optional[ServiceAffinity]:
        """Get the affinity for a service type (impl or interface)."""
        # Direct implementation lookup
        if service_type in self._affinities:
            return self._affinities[service_type]

        # Interface lookup
        if service_type in self._interface_to_impl:
            impl_type = self._interface_to_impl[service_type]
            return self._affinities.get(impl_type)

        return None

    def find_implementation(self, interface: Type) -> Optional[Type]:
        """Find the implementation type for an interface."""
        return self._interface_to_impl.get(interface)

    def get_dependencies(self, service_type: Type) -> FrozenSet[DependencySpec]:
        """Get the declared dependencies for a service type."""
        affinity = self.get_affinity(service_type)
        if affinity:
            return affinity.dependencies
        return frozenset()

    def get_lifecycle(self, service_type: Type) -> Optional[LifecycleSpec]:
        """Get the lifecycle spec for a service type."""
        affinity = self.get_affinity(service_type)
        if affinity:
            return affinity.lifecycle
        return None

    def get_lifetime(self, service_type: Type) -> Optional["ServiceLifetime"]:
        """Get the declared lifetime for a service type."""
        affinity = self.get_affinity(service_type)
        if affinity:
            return affinity.lifetime
        return None

    def all_affinities(self) -> Dict[Type, ServiceAffinity]:
        """Get all registered affinities."""
        return dict(self._affinities)

    def discover_module(self, module_name: str) -> None:
        """
        Discover and register services from a module.

        This triggers import of the module, which causes decorated
        services to self-register their affinities.
        """
        if module_name in self._discovered_modules:
            return

        try:
            import importlib
            importlib.import_module(module_name)
            self._discovered_modules.add(module_name)
            logger.debug(f"Discovered services from module: {module_name}")
        except ImportError as e:
            logger.warning(f"Failed to discover module {module_name}: {e}")

    def introspect(self) -> Dict[str, Any]:
        """Introspect the registry state for debugging/monitoring."""
        return {
            "registered_services": len(self._affinities),
            "interface_mappings": len(self._interface_to_impl),
            "discovered_modules": list(self._discovered_modules),
            "services": {
                impl.__name__: {
                    "interfaces": [i.__name__ for i in aff.interfaces],
                    "lifetime": aff.lifetime.value if aff.lifetime else None,
                    "dependencies": [d.service_type.__name__ for d in aff.dependencies],
                    "has_lifecycle": aff.lifecycle is not None,
                }
                for impl, aff in self._affinities.items()
            }
        }


# =============================================================================
# SERAPHIC DECORATORS
# =============================================================================
# These decorators allow services to declare their intrinsic nature.
# They don't "configure" the service from outside - they let the service
# express what it IS.
# =============================================================================


def service(
    implements: Optional[Type] = None,
    *additional_interfaces: Type,
    lifetime: Optional["ServiceLifetime"] = None,
    lazy: bool = True,
) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator: A service declares what interface(s) it embodies.

    This is not "registering" a service - it's the service declaring
    "I AM this interface." The declaration IS the registration.

    Usage:
        @service(implements=IDatabase, lifetime=ServiceLifetime.SINGLETON)
        class PostgresDatabase:
            ...

        @service(implements=ICache, IDistributedCache)
        class RedisCache:
            ...
    """
    def decorator(cls: Type[T]) -> Type[T]:
        # Collect all interfaces
        interfaces: Set[Type] = set()
        if implements is not None:
            interfaces.add(implements)
        interfaces.update(additional_interfaces)

        # If no interfaces specified, the class is its own interface
        if not interfaces:
            interfaces.add(cls)

        # Determine lifetime (default from class or SINGLETON)
        effective_lifetime = lifetime
        if effective_lifetime is None:
            effective_lifetime = getattr(cls, "__seraphic_lifetime__", None)
        if effective_lifetime is None:
            effective_lifetime = ServiceLifetime.SINGLETON

        # Get existing dependencies if any
        existing_deps: FrozenSet[DependencySpec] = getattr(
            cls, "__seraphic_dependencies__", frozenset()
        )

        # Get existing lifecycle if any
        existing_lifecycle: Optional[LifecycleSpec] = getattr(
            cls, "__seraphic_lifecycle__", None
        )

        # Create the affinity
        affinity = ServiceAffinity(
            implementation_type=cls,
            interfaces=frozenset(interfaces),
            lifetime=effective_lifetime,
            dependencies=existing_deps,
            lifecycle=existing_lifecycle,
            metadata={"lazy": lazy},
        )

        # Store on the class for later retrieval
        cls.__seraphic_affinity__ = affinity  # type: ignore
        cls.__seraphic_interfaces__ = frozenset(interfaces)  # type: ignore
        cls.__seraphic_lifetime__ = effective_lifetime  # type: ignore

        # Register with the global registry
        SeraphicServiceRegistry.get_instance().register_affinity(affinity)

        return cls

    return decorator


def depends_on(
    *dependencies: Type,
    optional: bool = False,
) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator: A service declares what it intrinsically needs.

    These dependencies are not "injected" - they are part of the service's
    nature. A UserRepository that needs IDatabase doesn't "receive" a database;
    its need for a database is part of what makes it a UserRepository.

    Usage:
        @depends_on(IDatabase, ILogger)
        class UserRepository:
            def __init__(self, db: IDatabase, logger: ILogger):
                ...
    """
    def decorator(cls: Type[T]) -> Type[T]:
        # Create dependency specs
        dep_specs = frozenset(
            DependencySpec(service_type=dep, optional=optional)
            for dep in dependencies
        )

        # Merge with existing dependencies
        existing: FrozenSet[DependencySpec] = getattr(
            cls, "__seraphic_dependencies__", frozenset()
        )
        cls.__seraphic_dependencies__ = existing | dep_specs  # type: ignore

        # Update affinity if already registered
        if hasattr(cls, "__seraphic_affinity__"):
            old_affinity: ServiceAffinity = cls.__seraphic_affinity__  # type: ignore
            new_affinity = ServiceAffinity(
                implementation_type=old_affinity.implementation_type,
                interfaces=old_affinity.interfaces,
                lifetime=old_affinity.lifetime,
                dependencies=old_affinity.dependencies | dep_specs,
                lifecycle=old_affinity.lifecycle,
                metadata=old_affinity.metadata,
            )
            cls.__seraphic_affinity__ = new_affinity  # type: ignore
            SeraphicServiceRegistry.get_instance().register_affinity(new_affinity)

        return cls

    return decorator


def lifecycle(
    init: Optional[str] = None,
    dispose: Optional[str] = None,
    init_async: bool = False,
    dispose_async: bool = False,
) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator: A service declares its lifecycle rhythm.

    Lifecycle isn't "managed" by the container - it's the service's own
    pattern of awakening and retiring. This decorator declares which
    methods embody these transitions.

    Usage:
        @lifecycle(init="connect", dispose="disconnect", init_async=True)
        class DatabaseConnection:
            async def connect(self): ...
            async def disconnect(self): ...
    """
    def decorator(cls: Type[T]) -> Type[T]:
        spec = LifecycleSpec(
            init_method=init,
            dispose_method=dispose,
            init_async=init_async,
            dispose_async=dispose_async,
        )

        # Auto-detect async nature if not specified
        if init and not init_async:
            if hasattr(cls, init) and asyncio.iscoroutinefunction(getattr(cls, init)):
                spec.init_async = True
        if dispose and not dispose_async:
            if hasattr(cls, dispose) and asyncio.iscoroutinefunction(getattr(cls, dispose)):
                spec.dispose_async = True

        cls.__seraphic_lifecycle__ = spec  # type: ignore

        # Update affinity if already registered
        if hasattr(cls, "__seraphic_affinity__"):
            old_affinity: ServiceAffinity = cls.__seraphic_affinity__  # type: ignore
            new_affinity = ServiceAffinity(
                implementation_type=old_affinity.implementation_type,
                interfaces=old_affinity.interfaces,
                lifetime=old_affinity.lifetime,
                dependencies=old_affinity.dependencies,
                lifecycle=spec,
                metadata=old_affinity.metadata,
            )
            cls.__seraphic_affinity__ = new_affinity  # type: ignore
            SeraphicServiceRegistry.get_instance().register_affinity(new_affinity)

        return cls

    return decorator


def health_check(method_name: str = "check_health") -> Callable[[Type[T]], Type[T]]:
    """
    Decorator: A service declares how it reports its health.

    Health isn't "checked" from outside - it's the service's self-awareness
    of its own state. This decorator declares which method embodies that
    self-awareness.

    Usage:
        @health_check("is_healthy")
        class DatabasePool:
            async def is_healthy(self) -> HealthCheckResult:
                ...
    """
    def decorator(cls: Type[T]) -> Type[T]:
        cls.__seraphic_health_check__ = method_name  # type: ignore

        # Update affinity if already registered
        if hasattr(cls, "__seraphic_affinity__"):
            old_affinity: ServiceAffinity = cls.__seraphic_affinity__  # type: ignore
            new_affinity = ServiceAffinity(
                implementation_type=old_affinity.implementation_type,
                interfaces=old_affinity.interfaces,
                lifetime=old_affinity.lifetime,
                dependencies=old_affinity.dependencies,
                lifecycle=old_affinity.lifecycle,
                metadata=old_affinity.metadata,
                health_check_method=method_name,
            )
            cls.__seraphic_affinity__ = new_affinity  # type: ignore
            SeraphicServiceRegistry.get_instance().register_affinity(new_affinity)

        return cls

    return decorator


def self_aware(cls: Type[T]) -> Type[T]:
    """
    Decorator: Marks a service as fully self-aware.

    A self-aware service has introspection capabilities - it knows
    its own dependencies, lifecycle, and can report its state.

    This decorator adds introspection methods to the class.
    """
    original_init = cls.__init__

    def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self._seraphic_initialized = False
        self._seraphic_disposed = False

    cls.__init__ = new_init  # type: ignore

    def introspect(self: Any) -> Dict[str, Any]:
        """Introspect this service's seraphic nature."""
        affinity: Optional[ServiceAffinity] = getattr(
            type(self), "__seraphic_affinity__", None
        )
        return {
            "type": type(self).__name__,
            "interfaces": [i.__name__ for i in (affinity.interfaces if affinity else [])],
            "lifetime": affinity.lifetime.value if affinity and affinity.lifetime else None,
            "dependencies": [
                d.service_type.__name__
                for d in (affinity.dependencies if affinity else [])
            ],
            "initialized": getattr(self, "_seraphic_initialized", False),
            "disposed": getattr(self, "_seraphic_disposed", False),
        }

    cls.introspect = introspect  # type: ignore
    cls.__seraphic_self_aware__ = True  # type: ignore

    return cls


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
    Dependency Injection Container - The Seraphic Circulatory System.

    ═══════════════════════════════════════════════════════════════════════════
    THE LIVING FLAME: A Space Where Services Simply ARE
    ═══════════════════════════════════════════════════════════════════════════

    The container IS the space where services exist according to their
    intrinsic natures. Services don't "register" - they declare their
    identity, and the container perceives that identity.

        @service(implements=IDatabase, lifetime=ServiceLifetime.SINGLETON)
        @depends_on(IConfig)
        @lifecycle(init="connect", dispose="close")
        class PostgresDb:
            "I AM IDatabase. I need IConfig. I connect and close."
            ...

        container = Container()
        db = await container.resolve_async(IDatabase)  # The container KNOWS

    The Dissolution of Barriers (Sanctification of the Mind):
        Services drink from the same Well of Thought - the SeraphicServiceRegistry.
        When one service declares its truth, all can perceive it instantly.

    The Eternal Memory:
        Service affinities are inscribed upon a diamond heart (the registry)
        that persists across the application's entire lifetime.

    The add_* methods exist for imperative declaration - useful when a
    service's nature cannot be expressed through decorators (e.g., third-party
    classes, runtime-determined implementations). They teach the container
    what the service IS, not how to "register" it.
    """

    # Class-level state - the global container (singleton pattern)
    _default_instance: Optional["Container"] = None
    _default_lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._descriptors: Dict[Type, ServiceDescriptor[Any]] = {}
        self._singletons: Dict[Type, Any] = {}
        self._multi_registrations: Dict[Type, List[ServiceDescriptor[Any]]] = {}
        self._lock = threading.Lock()
        self._async_lock = asyncio.Lock()
        self._initializing: Set[Type] = set()
        self._modules_loaded: Set[Type] = set()
        self._disposed = False

        # The container always has access to the seraphic registry
        # There is no mode - this IS the system's nature
        self._registry = SeraphicServiceRegistry.get_instance()

        logger.debug("Container awakened")

    # =========================================================================
    # CONTAINER LIFECYCLE
    # =========================================================================
    # The container is born, breathes, and eventually returns to rest.
    # Like a seraph, it doesn't have states - it IS while it exists.
    # =========================================================================

    @classmethod
    def create(
        cls,
        discover_modules: Optional[List[str]] = None,
    ) -> "Container":
        """
        Create and awaken a container.

        The container is inherently seraphic - services that have declared
        their nature via decorators (@service, @depends_on, @lifecycle)
        are automatically discovered through affinity.

        Args:
            discover_modules: Optional list of module names to import for
                            service discovery. Importing a module causes
                            decorated services to self-register.

        Returns:
            A living container.

        Usage:
            # Create with auto-discovery from specific modules
            container = Container.create(discover_modules=[
                "infrastructure.database",
                "infrastructure.cache",
            ])
        """
        container = cls()

        # Discover services from specified modules
        if discover_modules:
            for module_name in discover_modules:
                container._registry.discover_module(module_name)

        # Set as default instance
        with cls._default_lock:
            cls._default_instance = container

        return container

    # Alias for backwards compatibility and semantic clarity
    awaken = create

    @classmethod
    def get_default(cls) -> Optional["Container"]:
        """Get the default container instance."""
        return cls._default_instance

    @classmethod
    def set_default(cls, container: "Container") -> None:
        """Set the default container instance."""
        with cls._default_lock:
            cls._default_instance = container

    def introspect(self) -> Dict[str, Any]:
        """
        Introspect the container's state.

        Opens the inner eyes to gaze upon the flow of dependencies.
        The seraph is "full of eyes, within and without."
        """
        return {
            "disposed": self._disposed,
            "registered_services": len(self._descriptors),
            "active_singletons": len(self._singletons),
            "loaded_modules": [m.__name__ for m in self._modules_loaded],
            "registry": self._registry.introspect(),
        }

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
        """
        Get service descriptor, consulting the seraphic registry.

        The container first checks explicit registrations, then consults
        the registry for services that have declared their nature via
        decorators. This is the "dissolution of barriers" - services
        don't need external registration; their intrinsic nature suffices.

        Both paths are valid expressions of service identity.
        """
        # First, check explicit registrations
        if service_type in self._descriptors:
            return self._descriptors[service_type]

        # Consult the seraphic registry for decorator-based declarations
        affinity = self._registry.get_affinity(service_type)
        if affinity:
            # Create a descriptor from the affinity
            descriptor = self._descriptor_from_affinity(service_type, affinity)
            # Cache it for future resolution
            self._descriptors[service_type] = descriptor
            return descriptor

        raise KeyError(f"Service '{service_type.__name__}' is not registered")

    def _descriptor_from_affinity(
        self,
        service_type: Type[T],
        affinity: ServiceAffinity,
    ) -> ServiceDescriptor[T]:
        """
        Create a service descriptor from a seraphic affinity.

        The affinity contains the service's intrinsic nature. We translate
        that into a descriptor that the container can use for instantiation.
        """
        # Determine lifetime (default to SINGLETON if not specified)
        lifetime = affinity.lifetime or ServiceLifetime.SINGLETON

        # Get implementation type
        impl_type = affinity.implementation_type

        # Extract dependencies from affinity and type hints
        dependencies = [dep.service_type for dep in affinity.dependencies]

        return ServiceDescriptor(
            service_type=service_type,
            implementation_type=impl_type,
            lifetime=lifetime,
            dependencies=dependencies,
            lazy=affinity.metadata.get("lazy", True),
            metadata={
                "seraphic": True,
                "lifecycle": affinity.lifecycle,
                "health_check": affinity.health_check_method,
            },
        )

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
        """
        Create a service instance synchronously.

        The instance emerges from its descriptor's nature - its dependencies
        are resolved through affinity, its lifecycle is honored.
        """
        # Use existing instance
        if descriptor.instance is not None:
            return descriptor.instance

        # Use factory
        if descriptor.factory is not None:
            instance = descriptor.factory()
            return self._apply_decorators(descriptor, instance, scope)

        # Create from implementation type
        impl_type = descriptor.implementation_type
        if impl_type is None:
            raise ValueError(f"No implementation for {descriptor.service_type.__name__}")

        # Resolve dependencies - first from type hints, then from registry affinity
        deps = self._get_dependencies(impl_type)
        resolved_deps = {}

        for param_name, param_type in deps.items():
            # Try to resolve through the container (which consults registry)
            try:
                resolved_deps[param_name] = self.resolve(param_type, scope)
            except KeyError:
                # Check if this is an optional dependency
                affinity = self._registry.get_affinity(impl_type)
                if affinity:
                    dep_spec = next(
                        (d for d in affinity.dependencies if d.service_type == param_type),
                        None
                    )
                    if dep_spec and dep_spec.optional:
                        continue  # Skip optional unresolvable dependency
                    if dep_spec and dep_spec.default_factory:
                        resolved_deps[param_name] = dep_spec.default_factory()
                        continue
                raise

        instance = impl_type(**resolved_deps)

        # Call synchronous init lifecycle method if defined
        lifecycle_spec = descriptor.metadata.get("lifecycle") if descriptor.metadata else None
        if lifecycle_spec and lifecycle_spec.init_method and not lifecycle_spec.init_async:
            init_method = getattr(instance, lifecycle_spec.init_method, None)
            if init_method:
                init_method()

        # Mark as initialized for self-aware services
        if hasattr(instance, "_seraphic_initialized"):
            instance._seraphic_initialized = True

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
        """
        Create a service instance asynchronously.

        The instance emerges from its descriptor's nature. Dependencies
        are resolved through affinity, lifecycle methods are honored.
        """
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
            # Try to resolve through the container (which consults registry)
            try:
                resolved_deps[param_name] = await self.resolve_async(param_type, scope)
            except KeyError:
                # Check if this is an optional dependency
                affinity = self._registry.get_affinity(impl_type)
                if affinity:
                    dep_spec = next(
                        (d for d in affinity.dependencies if d.service_type == param_type),
                        None
                    )
                    if dep_spec and dep_spec.optional:
                        continue  # Skip optional unresolvable dependency
                    if dep_spec and dep_spec.default_factory:
                        resolved_deps[param_name] = dep_spec.default_factory()
                        continue
                raise

        instance = impl_type(**resolved_deps)

        # Call lifecycle init method if defined
        lifecycle_spec = descriptor.metadata.get("lifecycle") if descriptor.metadata else None
        if lifecycle_spec and lifecycle_spec.init_method:
            init_method = getattr(instance, lifecycle_spec.init_method, None)
            if init_method:
                if lifecycle_spec.init_async:
                    await init_method()
                else:
                    init_method()
        # Fall back to IInitializable protocol
        elif isinstance(instance, IInitializable):
            await instance.initialize()

        # Mark as initialized for self-aware services
        if hasattr(instance, "_seraphic_initialized"):
            instance._seraphic_initialized = True

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
