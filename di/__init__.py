"""
BIBLOS v2 - Dependency Injection Module

Provides IoC (Inversion of Control) container for managing
application dependencies and lifecycle.

Features:
- Singleton and transient lifetimes
- Factory functions support
- Lazy initialization
- Async initialization support
- Scope management
"""

from di.container import (
    Container,
    Scope,
    ServiceLifetime,
    ServiceDescriptor,
    get_container,
    inject,
    injectable,
    provide,
)

__all__ = [
    "Container",
    "Scope",
    "ServiceLifetime",
    "ServiceDescriptor",
    "get_container",
    "inject",
    "injectable",
    "provide",
]
