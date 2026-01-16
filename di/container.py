"""
BIBLOS v2 - Dependency Injection Container

Provides a lightweight IoC container for managing application
dependencies with support for various lifetimes and scopes.

Features:
- Service registration with multiple lifetimes
- Singleton, Scoped, and Transient services
- Factory function support
- Async initialization
- Decorator-based injection
- Lazy loading
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import threading
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    get_type_hints,
    ParamSpec,
)
from weakref import WeakValueDictionary

T = TypeVar("T")
P = ParamSpec("P")

# Global container instance
_container: Optional["Container"] = None
_container_lock = threading.Lock()


class ServiceLifetime(Enum):
    """Service lifetime options."""

    SINGLETON = "singleton"  # One instance for entire application
    SCOPED = "scoped"        # One instance per scope
    TRANSIENT = "transient"  # New instance every time


@dataclass
class ServiceDescriptor(Generic[T]):
    """Describes how a service should be created and managed."""

    service_type: Type[T]
    implementation_type: Optional[Type[T]] = None
    factory: Optional[Callable[..., T]] = None
    instance: Optional[T] = None
    lifetime: ServiceLifetime = ServiceLifetime.SINGLETON
    dependencies: List[Type] = field(default_factory=list)
    async_factory: Optional[Callable[..., Any]] = None
    lazy: bool = False

    def __post_init__(self) -> None:
        if self.implementation_type is None and self.factory is None and self.instance is None:
            self.implementation_type = self.service_type


class Scope:
    """
    Scope for scoped services.

    Usage:
        async with container.create_scope() as scope:
            service = scope.resolve(MyService)
    """

    def __init__(self, container: "Container", parent: Optional["Scope"] = None):
        self._container = container
        self._parent = parent
        self._instances: Dict[Type, Any] = {}
        self._lock = threading.Lock()

    def resolve(self, service_type: Type[T]) -> T:
        """Resolve a service within this scope."""
        descriptor = self._container._get_descriptor(service_type)

        if descriptor.lifetime == ServiceLifetime.SCOPED:
            with self._lock:
                if service_type not in self._instances:
                    self._instances[service_type] = self._container._create_instance(
                        descriptor, self
                    )
                return self._instances[service_type]

        return self._container.resolve(service_type, self)

    async def resolve_async(self, service_type: Type[T]) -> T:
        """Resolve a service asynchronously."""
        descriptor = self._container._get_descriptor(service_type)

        if descriptor.lifetime == ServiceLifetime.SCOPED:
            with self._lock:
                if service_type not in self._instances:
                    self._instances[service_type] = await self._container._create_instance_async(
                        descriptor, self
                    )
                return self._instances[service_type]

        return await self._container.resolve_async(service_type, self)

    def dispose(self) -> None:
        """Dispose all scoped instances."""
        for instance in self._instances.values():
            if hasattr(instance, "dispose"):
                instance.dispose()
            elif hasattr(instance, "close"):
                instance.close()
        self._instances.clear()

    async def dispose_async(self) -> None:
        """Dispose all scoped instances asynchronously."""
        for instance in self._instances.values():
            if hasattr(instance, "dispose_async"):
                await instance.dispose_async()
            elif hasattr(instance, "cleanup"):
                await instance.cleanup()
            elif hasattr(instance, "close"):
                if asyncio.iscoroutinefunction(instance.close):
                    await instance.close()
                else:
                    instance.close()
        self._instances.clear()


class Container:
    """
    Dependency Injection Container.

    Manages service registration, resolution, and lifecycle.

    Usage:
        container = Container()

        # Register services
        container.register_singleton(DatabaseClient)
        container.register_transient(RequestHandler)
        container.register_factory(Config, lambda: load_config())

        # Resolve services
        db = container.resolve(DatabaseClient)
    """

    def __init__(self) -> None:
        self._descriptors: Dict[Type, ServiceDescriptor] = {}
        self._singletons: Dict[Type, Any] = {}
        self._lock = threading.Lock()
        self._async_lock = asyncio.Lock()
        self._initializing: set = set()

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
        scope: Optional[Scope] = None,
    ) -> T:
        """Create a service instance synchronously."""
        # Use existing instance
        if descriptor.instance is not None:
            return descriptor.instance

        # Use factory
        if descriptor.factory is not None:
            return descriptor.factory()

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

        return impl_type(**resolved_deps)

    async def _create_instance_async(
        self,
        descriptor: ServiceDescriptor[T],
        scope: Optional[Scope] = None,
    ) -> T:
        """Create a service instance asynchronously."""
        # Use existing instance
        if descriptor.instance is not None:
            return descriptor.instance

        # Use async factory
        if descriptor.async_factory is not None:
            return await descriptor.async_factory()

        # Use sync factory
        if descriptor.factory is not None:
            return descriptor.factory()

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

        # Call async init if present
        if hasattr(instance, "initialize"):
            await instance.initialize()

        return instance

    def resolve(
        self,
        service_type: Type[T],
        scope: Optional[Scope] = None,
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
        scope: Optional[Scope] = None,
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

    async def dispose_async(self) -> None:
        """Dispose all singleton instances."""
        for instance in self._singletons.values():
            if hasattr(instance, "cleanup"):
                await instance.cleanup()
            elif hasattr(instance, "close"):
                if asyncio.iscoroutinefunction(instance.close):
                    await instance.close()
                else:
                    instance.close()
        self._singletons.clear()


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

            return await func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return wrapper

    return decorator
