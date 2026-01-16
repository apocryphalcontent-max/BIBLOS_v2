"""
BIBLOS v2 - Dependency Injection Module (The Blood Vessels)

The DI module is the organism's circulatory system - delivering resources
to every cell that needs them, managing the flow of dependencies, and
ensuring each component receives exactly what it requires to function.

Provides IoC (Inversion of Control) container for managing
application dependencies and lifecycle:
- ISP-compliant interfaces (small, focused contracts)
- Scoped lifetimes (singleton, transient, scoped)
- Factory functions support (lazy initialization)
- Async initialization support
- Module-based composition

Design Principles:
    1. Interface Segregation: Many small interfaces > few large ones
    2. Dependency Inversion: Depend on abstractions, not concretions
    3. Composition Root: All wiring happens at application startup
    4. Explicit Dependencies: No service locator anti-pattern

Architectural Role:
    Like blood vessels that don't choose what they carry but ensure
    delivery, the DI container doesn't dictate what services exist -
    it ensures they reach their destinations. The container knows
    HOW to create services but not WHEN they'll be needed.

Usage:
    from di import (
        Container, Scope, ServiceLifetime,
        IServiceProvider, IServiceCollection,
        injectable, inject,
    )

    # Define interface
    class IVerseService(Protocol):
        def get_verse(self, ref: str) -> Verse: ...

    # Register implementation
    container = Container()
    container.register(IVerseService, VerseService, ServiceLifetime.SCOPED)

    # Resolve in scope
    async with container.create_scope() as scope:
        service = scope.resolve(IVerseService)
        verse = service.get_verse("GEN.1.1")
"""

from di.container import (
    # ========================================================================
    # ISP-Compliant Interfaces (Interface Segregation Principle)
    # ========================================================================
    # Each interface has a single responsibility. Services implement
    # only the interfaces they need, not monolithic god-interfaces.
    # ========================================================================

    # Lifecycle interfaces
    IDisposable,
    IAsyncDisposable,
    IInitializable,

    # Health check interface
    IHealthCheck,
    HealthCheckResult,

    # Service provider abstraction
    IServiceProvider,

    # Service collection abstraction
    IServiceCollection,

    # Scope abstraction
    IServiceScope,

    # Module abstraction
    IModule,

    # ========================================================================
    # Core Container Components
    # ========================================================================

    # Lifetime enum
    ServiceLifetime,

    # Service descriptor
    ServiceDescriptor,

    # Scope implementation
    Scope,

    # Main container
    Container,

    # ========================================================================
    # Utilities and Decorators
    # ========================================================================

    # Global container access
    get_container,

    # Dependency injection decorators
    injectable,
    inject,
    provide,
)

__all__ = [
    # ========================================================================
    # ISP-COMPLIANT INTERFACES
    # ========================================================================
    # These small, focused interfaces follow the Interface Segregation
    # Principle - no client should be forced to depend on methods it
    # doesn't use.
    # ========================================================================

    # Lifecycle Interfaces
    "IDisposable",          # Synchronous cleanup
    "IAsyncDisposable",     # Async cleanup
    "IInitializable",       # Post-construction initialization

    # Health Check Interface
    "IHealthCheck",         # Health check capability
    "HealthCheckResult",    # Health check result data

    # Core Abstractions
    "IServiceProvider",     # Resolve services
    "IServiceCollection",   # Register services
    "IServiceScope",        # Scoped resolution
    "IModule",              # Module-based registration

    # ========================================================================
    # CORE COMPONENTS
    # ========================================================================

    # Lifetime management
    "ServiceLifetime",      # Singleton, Transient, Scoped
    "ServiceDescriptor",    # Service registration metadata

    # Container and Scope
    "Container",            # Main DI container
    "Scope",               # Scoped service resolution

    # ========================================================================
    # UTILITIES
    # ========================================================================

    # Global access
    "get_container",        # Get global container instance

    # Decorators
    "injectable",           # Mark class as injectable
    "inject",               # Mark parameter for injection
    "provide",             # Mark method as provider
]
