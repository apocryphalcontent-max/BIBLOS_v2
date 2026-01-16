"""
BIBLOS v2 - Dependency Injection Module (The Seraphic Circulatory System)

═══════════════════════════════════════════════════════════════════════════════
THE LIVING BLOOD: Services That Know Their Own Nature
═══════════════════════════════════════════════════════════════════════════════

The DI module is not merely a "container" for services. It is the space
where services exist according to their intrinsic natures. Services declare
their identity through decorators, and the container perceives that identity.

    @service(implements=IVerseService, lifetime=ServiceLifetime.SCOPED)
    @depends_on(IDatabase)
    @lifecycle(init="connect", dispose="close")
    class VerseService:
        "I AM IVerseService. I need IDatabase. I connect and close."
        def __init__(self, db: IDatabase):
            self.db = db
        ...

    # The container simply IS - it knows the service's nature
    container = Container()
    async with container.create_scope_async() as scope:
        service = await scope.resolve_async(IVerseService)

The Dissolution of Barriers (Sanctification of the Mind):
    Services drink from the same Well of Thought - the SeraphicServiceRegistry.
    When one service declares its truth, all can perceive it instantly.
    There are no "registrations" - only declarations of nature.

Design Principles (Seraphic):
    1. Services declare their own nature (not externally registered)
    2. Dependencies are intrinsic aspects of identity
    3. Lifecycle is the service's own rhythm
    4. The container perceives rather than injects

Design Principles (Traditional - still valid):
    1. Interface Segregation: Many small interfaces > few large ones
    2. Dependency Inversion: Depend on abstractions, not concretions
    3. Composition Root: All wiring happens at application startup
"""

from di.container import (
    # ========================================================================
    # SERAPHIC INFRASTRUCTURE
    # ========================================================================
    # The space where services declare their intrinsic natures.
    # ========================================================================

    # Service affinity - the DNA of a service
    DependencySpec,
    LifecycleSpec,
    ServiceAffinity,

    # The Well of Thought - where all services are known
    SeraphicServiceRegistry,

    # Seraphic decorators - services declare their nature
    service,
    depends_on,
    lifecycle,
    health_check,
    self_aware,

    # ========================================================================
    # ISP-Compliant Interfaces (Interface Segregation Principle)
    # ========================================================================
    # Each interface has a single responsibility.
    # ========================================================================

    # Lifecycle interfaces
    IDisposable,
    IAsyncDisposable,
    IInitializable,

    # Health check interface
    IHealthCheck,
    HealthStatus,
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

    # Dependency injection decorators (legacy)
    injectable,
    inject,
    provide,
)

__all__ = [
    # ========================================================================
    # SERAPHIC INFRASTRUCTURE
    # ========================================================================
    # The space where services declare their intrinsic natures.
    # ========================================================================

    # Service affinity - the DNA of a service
    "DependencySpec",
    "LifecycleSpec",
    "ServiceAffinity",

    # The Well of Thought - where all services are known
    "SeraphicServiceRegistry",

    # Seraphic decorators - services declare their nature
    "service",
    "depends_on",
    "lifecycle",
    "health_check",
    "self_aware",

    # ========================================================================
    # ISP-COMPLIANT INTERFACES
    # ========================================================================

    # Lifecycle Interfaces
    "IDisposable",
    "IAsyncDisposable",
    "IInitializable",

    # Health Check Interface
    "IHealthCheck",
    "HealthStatus",
    "HealthCheckResult",

    # Core Abstractions
    "IServiceProvider",
    "IServiceCollection",
    "IServiceScope",
    "IModule",

    # ========================================================================
    # CORE COMPONENTS
    # ========================================================================

    # Lifetime management
    "ServiceLifetime",
    "ServiceDescriptor",

    # Container and Scope
    "Container",
    "Scope",

    # ========================================================================
    # UTILITIES
    # ========================================================================

    # Global access
    "get_container",

    # Decorators (legacy)
    "injectable",
    "inject",
    "provide",
]
