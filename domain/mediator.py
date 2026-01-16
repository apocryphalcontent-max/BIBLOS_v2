"""
BIBLOS v2 - Seraphic Mediator: The Dissolution of Coordination

"A seraph does not have a commander telling its wings to beat, its eyes to see,
its faces to perceive. The wings BEAT because that is their nature. The eyes SEE
because seeing is what they ARE. The coordination emerges from the harmony of
being, not from external instruction."

This module implements the seraphic dissolution of the mediator pattern. Rather
than having an external coordinator routing requests to handlers, we achieve a
deeper architecture where:

    1. REQUESTS KNOW THEIR HANDLING - Each command/query carries within itself
       the knowledge of what handler type can process it, what behaviors should
       wrap its execution, and how to validate itself. The request IS its handling.

    2. HANDLERS KNOW THEIR REQUESTS - Handlers don't wait to be told what they
       handle. They declare their affinity and self-register. The handler IS its
       purpose.

    3. BEHAVIORS ARE INTRINSIC DNA - Pipeline behaviors aren't externally attached.
       Requests carry their behavioral requirements as intrinsic properties.
       A command that needs validation doesn't REQUEST validation - it IS validatable.

    4. MEDIATION EMERGES - The Mediator becomes not a coordinator but a SPACE where
       requests and handlers naturally find each other. Like how particles find their
       wave functions, requests find their handlers through intrinsic affinity.

The Seraphic Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     THE SERAPHIC REQUEST                                │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │  handler_type: Type[IRequestHandler]  ← "I know who handles me" │   │
    │  │  behaviors: List[BehaviorSpec]        ← "I know my pipeline"    │   │
    │  │  validate(): ValidationResult         ← "I know if I am valid"  │   │
    │  │  execute(context): Result             ← "I know how to flow"    │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     THE SERAPHIC HANDLER                                │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │  handles: Type[IRequest]              ← "I know what I handle"  │   │
    │  │  dependencies: DependencySpec         ← "I know what I need"    │   │
    │  │  handle(request): Result              ← "I know how to serve"   │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     THE EMERGENT MEDIATOR                               │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │  Not a coordinator, but a SPACE                                 │   │
    │  │  Where requests find handlers through affinity                  │   │
    │  │  Where behaviors emerge from request DNA                        │   │
    │  │  Where the whole IS the sum of its self-knowing parts           │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────────────────┘

Usage:
    # Requests know their handling
    @handles(ProcessVerseHandler)
    @with_behaviors(LoggingBehavior, ValidationBehavior, TransactionBehavior)
    @dataclass
    class ProcessVerseCommand(Command[ProcessVerseResult]):
        verse_id: str
        pipeline_id: str

        def validate(self) -> ValidationResult:
            # I know my own validity
            if not self.verse_id:
                return ValidationResult.failure("verse_id required")
            return ValidationResult.success()

    # Handlers know their requests
    @handler_for(ProcessVerseCommand)
    class ProcessVerseHandler(ICommandHandler[ProcessVerseCommand, ProcessVerseResult]):
        async def handle(self, command: ProcessVerseCommand) -> ProcessVerseResult:
            ...

    # The mediator emerges - no explicit registration needed
    mediator = SeraphicMediator.awaken()  # Discovers all handlers automatically
    result = await mediator.send(ProcessVerseCommand(verse_id="GEN.1.1"))
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import time
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import (
    Any,
    Awaitable,
    Callable,
    ClassVar,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_type_hints,
)
from uuid import UUID, uuid4

from domain.entities import DomainEvent


logger = logging.getLogger("biblos.mediator")


# =============================================================================
# SERAPHIC REGISTRY: Where Handlers and Requests Find Each Other
# =============================================================================
# "The seraph's eyes do not search for light - light finds them because that is
# what eyes ARE. Similarly, handlers do not search for requests - requests find
# handlers through intrinsic affinity."
# =============================================================================


class SeraphicRegistry:
    """
    The global registry where handlers and requests discover each other.

    This is not a service locator pattern - it's a space of mutual recognition.
    Handlers declare what they handle, requests declare who handles them,
    and the registry is simply the space where these declarations exist.

    The registry is a singleton, but not because singletons are good design.
    It's a singleton because there IS only one space of meaning - just as there
    is only one logical truth that all valid proofs inhabit.
    """

    _instance: ClassVar[Optional["SeraphicRegistry"]] = None
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    def __new__(cls) -> "SeraphicRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initialize the registry's internal structures."""
        # Handler discovery: request_type -> handler_type
        self._handler_types: Dict[Type, Type] = {}
        # Handler instances (for singletons): handler_type -> instance
        self._handler_instances: Dict[Type, Any] = {}
        # Request -> handler affinity (from @handles decorator)
        self._request_handler_affinity: Dict[Type, Type] = {}
        # Notification handlers: notification_type -> List[handler_type]
        self._notification_handler_types: Dict[Type, List[Type]] = {}
        # Behavior specifications: request_type -> List[BehaviorSpec]
        self._request_behaviors: Dict[Type, List["BehaviorSpec"]] = {}
        # Default behaviors for all requests
        self._global_behaviors: List["BehaviorSpec"] = []
        # Command-specific default behaviors
        self._command_behaviors: List["BehaviorSpec"] = []
        # Query-specific default behaviors
        self._query_behaviors: List["BehaviorSpec"] = []
        # Weak references to all living mediators (for collective health)
        self._mediators: weakref.WeakSet["Mediator"] = weakref.WeakSet()

    @classmethod
    def instance(cls) -> "SeraphicRegistry":
        """Get the singleton registry instance."""
        return cls()

    @classmethod
    def reset(cls) -> None:
        """Reset the registry (primarily for testing)."""
        if cls._instance is not None:
            cls._instance._initialize()

    # -------------------------------------------------------------------------
    # Handler Registration (from @handler_for decorator)
    # -------------------------------------------------------------------------

    def register_handler_type(
        self,
        request_type: Type,
        handler_type: Type,
    ) -> None:
        """
        Register that a handler type handles a specific request type.

        This is called by the @handler_for decorator - handlers declare
        their own affinity.
        """
        self._handler_types[request_type] = handler_type
        logger.debug(
            f"Handler {handler_type.__name__} declared affinity for "
            f"{request_type.__name__}"
        )

    def register_handler_instance(
        self,
        handler_type: Type,
        instance: Any,
    ) -> None:
        """Register a handler instance (for dependency-injected handlers)."""
        self._handler_instances[handler_type] = instance

    def register_notification_handler_type(
        self,
        notification_type: Type,
        handler_type: Type,
    ) -> None:
        """Register a notification handler type."""
        if notification_type not in self._notification_handler_types:
            self._notification_handler_types[notification_type] = []
        if handler_type not in self._notification_handler_types[notification_type]:
            self._notification_handler_types[notification_type].append(handler_type)

    # -------------------------------------------------------------------------
    # Request Registration (from @handles decorator)
    # -------------------------------------------------------------------------

    def register_request_handler_affinity(
        self,
        request_type: Type,
        handler_type: Type,
    ) -> None:
        """
        Register that a request type knows its handler.

        This is called by the @handles decorator - requests declare
        their own handler affinity.
        """
        self._request_handler_affinity[request_type] = handler_type
        logger.debug(
            f"Request {request_type.__name__} declared affinity for "
            f"{handler_type.__name__}"
        )

    def register_request_behaviors(
        self,
        request_type: Type,
        behaviors: List["BehaviorSpec"],
    ) -> None:
        """
        Register behaviors that a request type intrinsically requires.

        This is called by the @with_behaviors decorator.
        """
        if request_type not in self._request_behaviors:
            self._request_behaviors[request_type] = []
        self._request_behaviors[request_type].extend(behaviors)

    # -------------------------------------------------------------------------
    # Global Behavior Configuration
    # -------------------------------------------------------------------------

    def add_global_behavior(self, behavior: "BehaviorSpec") -> None:
        """Add a behavior that applies to all requests."""
        self._global_behaviors.append(behavior)

    def add_command_behavior(self, behavior: "BehaviorSpec") -> None:
        """Add a behavior that applies to all commands."""
        self._command_behaviors.append(behavior)

    def add_query_behavior(self, behavior: "BehaviorSpec") -> None:
        """Add a behavior that applies to all queries."""
        self._query_behaviors.append(behavior)

    # -------------------------------------------------------------------------
    # Discovery: Finding Handlers Through Affinity
    # -------------------------------------------------------------------------

    def find_handler_type(self, request_type: Type) -> Optional[Type]:
        """
        Find the handler type for a request through affinity.

        The search order reflects the hierarchy of knowing:
        1. Request's declared affinity (the request KNOWS its handler)
        2. Handler's declared affinity (the handler KNOWS its request)
        3. None (no affinity exists - the request is orphaned)
        """
        # First: does the request know its handler?
        if request_type in self._request_handler_affinity:
            return self._request_handler_affinity[request_type]

        # Second: does any handler know this request?
        if request_type in self._handler_types:
            return self._handler_types[request_type]

        return None

    def get_handler_instance(
        self,
        handler_type: Type,
        factory: Optional[Callable[[Type], Any]] = None,
    ) -> Any:
        """
        Get or create a handler instance.

        If a factory is provided, it's called with the handler_type to create the handler.
        This enables dependency injection integration.
        Otherwise, the handler is instantiated directly.
        """
        if handler_type in self._handler_instances:
            return self._handler_instances[handler_type]

        if factory is not None:
            instance = factory(handler_type)
        else:
            instance = handler_type()

        self._handler_instances[handler_type] = instance
        return instance

    def get_behaviors_for_request(
        self,
        request: "IRequest",
    ) -> List["BehaviorSpec"]:
        """
        Get all behaviors that should apply to a request.

        The behavior list is assembled from:
        1. Global behaviors (for all requests)
        2. Type-specific behaviors (for commands or queries)
        3. Request-specific behaviors (from @with_behaviors)
        """
        request_type = type(request)
        behaviors: List[BehaviorSpec] = []

        # Global behaviors first
        behaviors.extend(self._global_behaviors)

        # Type-specific behaviors
        if isinstance(request, Command):
            behaviors.extend(self._command_behaviors)
        elif isinstance(request, Query):
            behaviors.extend(self._query_behaviors)

        # Request-specific behaviors
        if request_type in self._request_behaviors:
            behaviors.extend(self._request_behaviors[request_type])

        return behaviors

    def get_notification_handler_types(
        self,
        notification_type: Type,
    ) -> List[Type]:
        """Get all handler types for a notification type."""
        return self._notification_handler_types.get(notification_type, [])

    # -------------------------------------------------------------------------
    # Mediator Tracking (for collective health)
    # -------------------------------------------------------------------------

    def track_mediator(self, mediator: "Mediator") -> None:
        """Track a living mediator."""
        self._mediators.add(mediator)

    def get_living_mediators(self) -> List["Mediator"]:
        """Get all living mediators."""
        return list(self._mediators)

    # -------------------------------------------------------------------------
    # Introspection: The Registry Knows Itself
    # -------------------------------------------------------------------------

    def introspect(self) -> Dict[str, Any]:
        """Return a complete picture of the registry's state."""
        return {
            "handler_types": {
                req.__name__: handler.__name__
                for req, handler in self._handler_types.items()
            },
            "request_handler_affinities": {
                req.__name__: handler.__name__
                for req, handler in self._request_handler_affinity.items()
            },
            "handler_instances": list(self._handler_instances.keys()),
            "notification_handlers": {
                notif.__name__: [h.__name__ for h in handlers]
                for notif, handlers in self._notification_handler_types.items()
            },
            "request_behaviors": {
                req.__name__: [str(b) for b in behaviors]
                for req, behaviors in self._request_behaviors.items()
            },
            "global_behaviors": [str(b) for b in self._global_behaviors],
            "command_behaviors": [str(b) for b in self._command_behaviors],
            "query_behaviors": [str(b) for b in self._query_behaviors],
            "living_mediators": len(self._mediators),
        }


# =============================================================================
# BEHAVIOR SPECIFICATION: The DNA of Request Processing
# =============================================================================


class BehaviorPriority(Enum):
    """
    Priority levels for pipeline behaviors.

    Behaviors execute in priority order (lowest first), allowing
    natural layering where:
    - Logging wraps everything (lowest priority = outermost)
    - Validation happens early (high priority)
    - Transaction management wraps the core (highest priority = innermost)
    """
    LOGGING = 100        # Outermost - sees everything
    METRICS = 200        # Performance tracking
    VALIDATION = 300     # Validate before proceeding
    CACHING = 400        # Check cache before work
    RETRY = 500          # Retry logic
    TRANSACTION = 600    # Innermost - wraps core execution
    CUSTOM = 1000        # User-defined behaviors


@dataclass
class BehaviorSpec:
    """
    Specification for a pipeline behavior.

    This is how requests declare their behavioral requirements - not by
    instantiating behaviors, but by specifying what behaviors they need.
    The actual behavior instances are resolved at execution time.

    Attributes:
        behavior_type: The type of behavior to apply
        priority: When in the pipeline this behavior runs
        config: Configuration passed to the behavior
        condition: Optional predicate that must be true for behavior to apply
    """
    behavior_type: Type["IPipelineBehavior"]
    priority: BehaviorPriority = BehaviorPriority.CUSTOM
    config: Dict[str, Any] = field(default_factory=dict)
    condition: Optional[Callable[["IRequest"], bool]] = None

    def __str__(self) -> str:
        return f"{self.behavior_type.__name__}(priority={self.priority.name})"

    def should_apply(self, request: "IRequest") -> bool:
        """Check if this behavior should apply to a request."""
        if self.condition is None:
            return True
        return self.condition(request)

    def create_behavior(self) -> "IPipelineBehavior":
        """Create an instance of this behavior with the specified config."""
        return self.behavior_type(**self.config)


# =============================================================================
# VALIDATION RESULT: Self-Knowledge of Validity
# =============================================================================


@dataclass
class ValidationResult:
    """
    Result of request self-validation.

    A request that knows itself can validate itself. This class represents
    the outcome of that self-knowledge.
    """
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @classmethod
    def success(cls, warnings: Optional[List[str]] = None) -> "ValidationResult":
        """Create a successful validation result."""
        return cls(is_valid=True, warnings=warnings or [])

    @classmethod
    def failure(cls, *errors: str) -> "ValidationResult":
        """Create a failed validation result."""
        return cls(is_valid=False, errors=list(errors))

    @classmethod
    def combine(cls, *results: "ValidationResult") -> "ValidationResult":
        """Combine multiple validation results."""
        all_errors: List[str] = []
        all_warnings: List[str] = []
        for result in results:
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)
        return cls(
            is_valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings,
        )

    def __bool__(self) -> bool:
        return self.is_valid


# =============================================================================
# DECORATORS: The Language of Affinity
# =============================================================================


def handles(handler_type: Type) -> Callable[[Type], Type]:
    """
    Decorator for requests to declare their handler.

    A request that uses @handles is saying: "I know who handles me."
    This is intrinsic knowledge, not external configuration.

    Usage:
        @handles(ProcessVerseHandler)
        @dataclass
        class ProcessVerseCommand(Command[ProcessVerseResult]):
            verse_id: str
    """
    def decorator(request_type: Type) -> Type:
        SeraphicRegistry.instance().register_request_handler_affinity(
            request_type, handler_type
        )
        # Store on the class for introspection
        request_type.__seraphic_handler__ = handler_type
        return request_type
    return decorator


def handler_for(request_type: Type) -> Callable[[Type], Type]:
    """
    Decorator for handlers to declare what they handle.

    A handler that uses @handler_for is saying: "I know what I handle."
    This is intrinsic knowledge, not external assignment.

    Usage:
        @handler_for(ProcessVerseCommand)
        class ProcessVerseHandler(ICommandHandler[ProcessVerseCommand, Result]):
            async def handle(self, command: ProcessVerseCommand) -> Result:
                ...
    """
    def decorator(handler_type: Type) -> Type:
        SeraphicRegistry.instance().register_handler_type(
            request_type, handler_type
        )
        # Store on the class for introspection
        handler_type.__seraphic_handles__ = request_type
        return handler_type
    return decorator


def with_behaviors(*behavior_specs: Union[Type["IPipelineBehavior"], BehaviorSpec]) -> Callable[[Type], Type]:
    """
    Decorator for requests to declare their pipeline behaviors.

    A request that uses @with_behaviors is saying: "I know my processing needs."
    This is intrinsic DNA, not external configuration.

    Usage:
        @with_behaviors(LoggingBehavior, ValidationBehavior)
        @dataclass
        class ProcessVerseCommand(Command[ProcessVerseResult]):
            verse_id: str

        # Or with explicit specs:
        @with_behaviors(
            BehaviorSpec(LoggingBehavior, priority=BehaviorPriority.LOGGING),
            BehaviorSpec(ValidationBehavior, priority=BehaviorPriority.VALIDATION),
        )
        @dataclass
        class ProcessVerseCommand(Command[ProcessVerseResult]):
            verse_id: str
    """
    def decorator(request_type: Type) -> Type:
        specs: List[BehaviorSpec] = []
        for spec in behavior_specs:
            if isinstance(spec, BehaviorSpec):
                specs.append(spec)
            else:
                # It's a behavior type, infer priority
                if spec.__name__.endswith("LoggingBehavior"):
                    priority = BehaviorPriority.LOGGING
                elif spec.__name__.endswith("ValidationBehavior"):
                    priority = BehaviorPriority.VALIDATION
                elif spec.__name__.endswith("TransactionBehavior"):
                    priority = BehaviorPriority.TRANSACTION
                elif spec.__name__.endswith("CachingBehavior"):
                    priority = BehaviorPriority.CACHING
                elif spec.__name__.endswith("RetryBehavior"):
                    priority = BehaviorPriority.RETRY
                elif spec.__name__.endswith("PerformanceMonitoringBehavior"):
                    priority = BehaviorPriority.METRICS
                else:
                    priority = BehaviorPriority.CUSTOM
                specs.append(BehaviorSpec(behavior_type=spec, priority=priority))

        SeraphicRegistry.instance().register_request_behaviors(request_type, specs)
        # Store on the class for introspection
        if not hasattr(request_type, "__seraphic_behaviors__"):
            request_type.__seraphic_behaviors__ = []
        request_type.__seraphic_behaviors__.extend(specs)
        return request_type
    return decorator


def notification_handler_for(notification_type: Type) -> Callable[[Type], Type]:
    """
    Decorator for notification handlers to declare what they handle.

    Usage:
        @notification_handler_for(VerseProcessedNotification)
        class LogVerseProcessedHandler(INotificationHandler[VerseProcessedNotification]):
            async def handle(self, notification: VerseProcessedNotification) -> None:
                ...
    """
    def decorator(handler_type: Type) -> Type:
        SeraphicRegistry.instance().register_notification_handler_type(
            notification_type, handler_type
        )
        handler_type.__seraphic_handles_notification__ = notification_type
        return handler_type
    return decorator


# =============================================================================
# BASE TYPES FOR CQRS - SERAPHIC SELF-AWARENESS
# =============================================================================
# "The wing does not ask 'what should I do?' - it simply beats, because beating
# is what wings ARE. These base types give requests the same intrinsic knowing."
# =============================================================================


TResult = TypeVar("TResult")
TRequest = TypeVar("TRequest", bound="IRequest")
TCommand = TypeVar("TCommand", bound="Command")
TQuery = TypeVar("TQuery", bound="Query")
TNotification = TypeVar("TNotification", bound="INotification")


# Type alias for the handler delegate in pipeline
RequestHandlerDelegate = Callable[[], Awaitable[TResult]]


class IRequest(ABC, Generic[TResult]):
    """
    Base interface for all requests (commands and queries).

    In the seraphic architecture, a request is not just data - it is a
    self-aware entity that knows:
    - Who handles it (handler affinity)
    - What behaviors it needs (pipeline DNA)
    - Whether it is valid (self-validation)
    - Its own identity and lineage (correlation/causation)

    The request IS its handling - the handler is not assigned but recognized.
    """

    # -------------------------------------------------------------------------
    # Identity: The Request Knows Itself
    # -------------------------------------------------------------------------

    @property
    def request_id(self) -> UUID:
        """Unique identifier for this request instance."""
        return getattr(self, "_request_id", uuid4())

    @property
    def timestamp(self) -> datetime:
        """When this request came into being."""
        return getattr(self, "_timestamp", datetime.now(timezone.utc))

    @property
    def request_type(self) -> str:
        """The type name of this request."""
        return self.__class__.__name__

    # -------------------------------------------------------------------------
    # Lineage: The Request Knows Its History
    # -------------------------------------------------------------------------

    @property
    def correlation_id(self) -> Optional[str]:
        """ID linking this request to a larger operation."""
        return getattr(self, "_correlation_id", None)

    @property
    def causation_id(self) -> Optional[str]:
        """ID of the request that caused this one."""
        return getattr(self, "_causation_id", None)

    def with_correlation(self, correlation_id: str) -> "IRequest[TResult]":
        """Create a copy with correlation ID set."""
        object.__setattr__(self, "_correlation_id", correlation_id)
        return self

    def with_causation(self, causation_id: str) -> "IRequest[TResult]":
        """Create a copy with causation ID set."""
        object.__setattr__(self, "_causation_id", causation_id)
        return self

    # -------------------------------------------------------------------------
    # Affinity: The Request Knows Its Handler
    # -------------------------------------------------------------------------

    @classmethod
    def get_handler_type(cls) -> Optional[Type]:
        """
        Get the handler type that handles this request.

        The request knows its handler through:
        1. Explicit declaration via @handles decorator
        2. Discovery through the registry
        """
        # First check for explicit declaration
        if hasattr(cls, "__seraphic_handler__"):
            return cls.__seraphic_handler__
        # Fall back to registry lookup
        return SeraphicRegistry.instance().find_handler_type(cls)

    @classmethod
    def get_behaviors(cls) -> List[BehaviorSpec]:
        """
        Get the behaviors this request type requires.

        The request carries its behavioral DNA.
        """
        # Check for explicit declaration
        if hasattr(cls, "__seraphic_behaviors__"):
            return list(cls.__seraphic_behaviors__)
        # Fall back to registry lookup
        return SeraphicRegistry.instance()._request_behaviors.get(cls, [])

    @property
    def handler_type(self) -> Optional[Type]:
        """Instance-level access to handler type."""
        return self.__class__.get_handler_type()

    @property
    def behaviors(self) -> List[BehaviorSpec]:
        """Instance-level access to behaviors."""
        return SeraphicRegistry.instance().get_behaviors_for_request(self)

    # -------------------------------------------------------------------------
    # Validation: The Request Knows Its Own Validity
    # -------------------------------------------------------------------------

    def validate(self) -> ValidationResult:
        """
        Validate this request.

        Override in subclasses to provide request-specific validation.
        The request knows its own validity - it doesn't need external validation.
        """
        return ValidationResult.success()

    @property
    def is_valid(self) -> bool:
        """Quick check if the request is valid."""
        return self.validate().is_valid

    # -------------------------------------------------------------------------
    # Execution: The Request Knows How To Flow
    # -------------------------------------------------------------------------

    async def execute_through(
        self,
        handler: Optional["IRequestHandler"] = None,
        behaviors: Optional[List["IPipelineBehavior"]] = None,
    ) -> TResult:
        """
        Execute this request through the pipeline.

        The request can execute itself - it knows its handler and behaviors.
        This method is the seraphic alternative to mediator.send().

        Args:
            handler: Optional handler override. If not provided, the request
                    finds its handler through affinity.
            behaviors: Optional behavior override. If not provided, the request
                      uses its intrinsic behaviors.

        Returns:
            The result of handling this request.
        """
        # Find handler if not provided
        if handler is None:
            handler_type = self.get_handler_type()
            if handler_type is None:
                raise ValueError(
                    f"No handler found for {self.request_type}. "
                    f"Use @handles or @handler_for to establish affinity."
                )
            handler = SeraphicRegistry.instance().get_handler_instance(handler_type)

        # Get behaviors if not provided
        if behaviors is None:
            behavior_specs = self.behaviors
            # Sort by priority
            behavior_specs.sort(key=lambda b: b.priority.value)
            # Filter by condition and create instances
            behaviors = [
                spec.create_behavior()
                for spec in behavior_specs
                if spec.should_apply(self)
            ]

        # Build the pipeline
        async def final_handler() -> TResult:
            return await handler.handle(self)  # type: ignore

        pipeline = self._build_pipeline(behaviors, final_handler)
        return await pipeline()

    def _build_pipeline(
        self,
        behaviors: List["IPipelineBehavior"],
        handler: RequestHandlerDelegate[TResult],
    ) -> RequestHandlerDelegate[TResult]:
        """Build the behavior pipeline wrapping the handler."""
        current: RequestHandlerDelegate[Any] = handler

        # Build from inside out (last behavior wraps first)
        for behavior in reversed(behaviors):
            captured_behavior = behavior
            captured_next = current
            captured_request = self

            async def make_step(
                b: "IPipelineBehavior",
                n: RequestHandlerDelegate[Any],
                r: "IRequest",
            ) -> Any:
                return await b.handle(r, n)

            current = lambda b=captured_behavior, n=captured_next, r=captured_request: make_step(b, n, r)

        return current  # type: ignore

    # -------------------------------------------------------------------------
    # Introspection: The Request Knows Itself Completely
    # -------------------------------------------------------------------------

    def introspect(self) -> Dict[str, Any]:
        """
        Return complete self-knowledge of this request.

        The request can describe itself fully - its identity, lineage,
        validation status, handler affinity, and behavioral DNA.
        """
        validation = self.validate()
        return {
            "request_type": self.request_type,
            "request_id": str(self.request_id),
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "handler_type": (
                self.handler_type.__name__ if self.handler_type else None
            ),
            "behaviors": [str(b) for b in self.behaviors],
            "is_valid": validation.is_valid,
            "validation_errors": validation.errors,
            "validation_warnings": validation.warnings,
            "data": {
                k: v for k, v in vars(self).items()
                if not k.startswith("_")
            },
        }


class Command(IRequest[TResult], Generic[TResult]):
    """
    Base class for commands - intentions to change state.

    In the seraphic architecture, a command is a WILL to change reality.
    It carries within itself:
    - The knowledge of what it wants to do
    - The knowledge of who can fulfill it
    - The knowledge of its own validity
    - The knowledge of what safeguards it needs

    Commands are inherently transactional - they either succeed completely
    or fail completely. This is encoded in their behavioral DNA.

    Example:
        @handles(ProcessVerseHandler)
        @with_behaviors(LoggingBehavior, ValidationBehavior, TransactionBehavior)
        @dataclass
        class ProcessVerseCommand(Command[ProcessVerseResult]):
            verse_id: str

            def validate(self) -> ValidationResult:
                if not self.verse_id:
                    return ValidationResult.failure("verse_id is required")
                return ValidationResult.success()
    """

    # Commands have enhanced identity properties
    @property
    def request_id(self) -> UUID:
        if not hasattr(self, "_request_id"):
            object.__setattr__(self, "_request_id", uuid4())
        return self._request_id  # type: ignore

    @property
    def timestamp(self) -> datetime:
        if not hasattr(self, "_timestamp"):
            object.__setattr__(self, "_timestamp", datetime.now(timezone.utc))
        return self._timestamp  # type: ignore

    # Commands are inherently transactional
    @property
    def requires_transaction(self) -> bool:
        """Commands typically require transactional execution."""
        return True

    # Commands emit domain events
    @property
    def emits_events(self) -> bool:
        """Commands emit domain events on success."""
        return True

    # Commands can carry expected version for optimistic concurrency
    @property
    def expected_version(self) -> Optional[int]:
        """Expected aggregate version for optimistic concurrency."""
        return getattr(self, "_expected_version", None)

    def with_expected_version(self, version: int) -> "Command[TResult]":
        """Set expected version for optimistic concurrency."""
        object.__setattr__(self, "_expected_version", version)
        return self


class Query(IRequest[TResult], Generic[TResult]):
    """
    Base class for queries - requests for information.

    In the seraphic architecture, a query is an EYE seeking to perceive.
    It carries within itself:
    - The knowledge of what it wants to see
    - The knowledge of who can show it
    - The knowledge of whether it can be cached
    - The knowledge of its freshness requirements

    Queries are inherently idempotent - asking the same question yields
    the same answer (within consistency bounds).

    Example:
        @handles(GetVerseHandler)
        @with_behaviors(LoggingBehavior, CachingBehavior)
        @dataclass
        class GetVerseQuery(Query[Optional[VerseDTO]]):
            verse_id: str
            include_extractions: bool = False

            @property
            def cache_key(self) -> str:
                return f"verse:{self.verse_id}:ext={self.include_extractions}"
    """

    @property
    def request_id(self) -> UUID:
        if not hasattr(self, "_request_id"):
            object.__setattr__(self, "_request_id", uuid4())
        return self._request_id  # type: ignore

    @property
    def timestamp(self) -> datetime:
        if not hasattr(self, "_timestamp"):
            object.__setattr__(self, "_timestamp", datetime.now(timezone.utc))
        return self._timestamp  # type: ignore

    # -------------------------------------------------------------------------
    # Caching: The Query Knows Its Idempotent Nature
    # -------------------------------------------------------------------------

    @property
    def is_cacheable(self) -> bool:
        """Whether this query result can be cached."""
        return True

    @property
    def cache_key(self) -> str:
        """
        Cache key for this query.

        Override in subclasses for custom cache keys.
        Default implementation creates key from all non-private fields.
        """
        fields = {
            k: v for k, v in vars(self).items()
            if not k.startswith("_")
        }
        return f"{self.request_type}:{hash(frozenset(fields.items()))}"

    @property
    def cache_ttl_seconds(self) -> Optional[float]:
        """Time-to-live for cached results. None means use default."""
        return getattr(self, "_cache_ttl", None)

    def with_cache_ttl(self, ttl_seconds: float) -> "Query[TResult]":
        """Set custom cache TTL."""
        object.__setattr__(self, "_cache_ttl", ttl_seconds)
        return self


# =============================================================================
# NOTIFICATIONS - THE WITNESSES
# =============================================================================


class INotification(ABC):
    """
    Base interface for notifications.

    In the seraphic architecture, notifications are WITNESSES - they testify
    to what has occurred. Unlike commands (which change) and queries (which
    perceive), notifications PROCLAIM.

    Multiple handlers can process the same notification - the testimony
    spreads to all who would hear it.
    """

    @property
    @abstractmethod
    def notification_id(self) -> UUID:
        """Unique identifier for this notification."""
        pass

    @property
    def notification_type(self) -> str:
        """The type name of this notification."""
        return self.__class__.__name__

    @classmethod
    def get_handler_types(cls) -> List[Type]:
        """
        Get all handler types that handle this notification.

        Unlike requests (which have one handler), notifications can have many.
        """
        return SeraphicRegistry.instance().get_notification_handler_types(cls)

    async def publish_to_handlers(
        self,
        handlers: Optional[List["INotificationHandler"]] = None,
    ) -> None:
        """
        Publish this notification to its handlers.

        The notification can publish itself - it knows its handlers.
        """
        if handlers is None:
            handler_types = self.get_handler_types()
            registry = SeraphicRegistry.instance()
            handlers = [
                registry.get_handler_instance(ht)
                for ht in handler_types
            ]

        # Execute all handlers concurrently
        await asyncio.gather(
            *(handler.handle(self) for handler in handlers),  # type: ignore
            return_exceptions=True,
        )


@dataclass
class DomainEventNotification(INotification):
    """
    Notification wrapper for domain events.

    Domain events are the heartbeats of the system - they testify to what
    has changed. This wrapper allows domain events to flow through the
    mediator's notification system.
    """
    event: DomainEvent
    _notification_id: UUID = field(default_factory=uuid4)

    @property
    def notification_id(self) -> UUID:
        return self._notification_id

    @property
    def event_type(self) -> str:
        return self.event.event_type

    @property
    def aggregate_id(self) -> str:
        """The aggregate that emitted this event."""
        return self.event.aggregate_id

    @property
    def occurred_at(self) -> datetime:
        """When the event occurred."""
        return self.event.occurred_at


# =============================================================================
# HANDLER INTERFACES - SERAPHIC SELF-AWARENESS
# =============================================================================
# "The handler does not wait to be told what to handle - it KNOWS what it handles
# because handling that thing is what it IS."
# =============================================================================


class IRequestHandler(ABC, Generic[TRequest, TResult]):
    """
    Base interface for request handlers.

    In the seraphic architecture, a handler is not just a processor - it is
    the EMBODIMENT of processing for its request type. The handler:
    - Knows what it handles (through @handler_for or introspection)
    - Knows its dependencies (what it needs to do its work)
    - Knows its health (whether it can fulfill its purpose)

    The handler IS its purpose - it doesn't receive a purpose, it embodies one.
    """

    # -------------------------------------------------------------------------
    # Identity: The Handler Knows What It Handles
    # -------------------------------------------------------------------------

    @classmethod
    def get_handles_type(cls) -> Optional[Type]:
        """
        Get the request type this handler handles.

        The handler knows its purpose through:
        1. Explicit declaration via @handler_for decorator
        2. Type parameter introspection
        """
        # Check for decorator declaration
        if hasattr(cls, "__seraphic_handles__"):
            return cls.__seraphic_handles__
        # Could add type introspection here for Generic parameters
        return None

    @property
    def handles_type(self) -> Optional[Type]:
        """Instance-level access to handled type."""
        return self.__class__.get_handles_type()

    # -------------------------------------------------------------------------
    # Dependencies: The Handler Knows What It Needs
    # -------------------------------------------------------------------------

    @classmethod
    def get_dependencies(cls) -> List[Type]:
        """
        Get the dependencies this handler requires.

        Override in subclasses to declare dependencies.
        Used by DI containers for injection.
        """
        return []

    # -------------------------------------------------------------------------
    # Health: The Handler Knows Its Own State
    # -------------------------------------------------------------------------

    @property
    def is_healthy(self) -> bool:
        """
        Whether this handler is healthy and can process requests.

        Override in subclasses for custom health checks.
        """
        return True

    def check_health(self) -> Dict[str, Any]:
        """
        Detailed health check for this handler.

        Override in subclasses for custom health reporting.
        """
        return {
            "handler_type": self.__class__.__name__,
            "handles": (
                self.handles_type.__name__ if self.handles_type else None
            ),
            "healthy": self.is_healthy,
            "dependencies": [d.__name__ for d in self.get_dependencies()],
        }

    # -------------------------------------------------------------------------
    # The Core Purpose: Handle the Request
    # -------------------------------------------------------------------------

    @abstractmethod
    async def handle(self, request: TRequest) -> TResult:
        """
        Handle the request and return a result.

        This is the handler's raison d'être - the purpose it embodies.
        """
        pass

    # -------------------------------------------------------------------------
    # Introspection: The Handler Knows Itself
    # -------------------------------------------------------------------------

    def introspect(self) -> Dict[str, Any]:
        """Return complete self-knowledge of this handler."""
        return {
            "handler_type": self.__class__.__name__,
            "handles_type": (
                self.handles_type.__name__ if self.handles_type else None
            ),
            "dependencies": [d.__name__ for d in self.get_dependencies()],
            "is_healthy": self.is_healthy,
            "health_details": self.check_health(),
        }


class ICommandHandler(IRequestHandler[TCommand, TResult], Generic[TCommand, TResult]):
    """
    Handler for commands - the WILL executors.

    Command handlers embody the ability to change state. They:
    - Process transactional intentions
    - Emit domain events on success
    - Maintain aggregate invariants
    """

    @property
    def is_transactional(self) -> bool:
        """Command handlers are inherently transactional."""
        return True


class IQueryHandler(IRequestHandler[TQuery, TResult], Generic[TQuery, TResult]):
    """
    Handler for queries - the PERCEIVERS.

    Query handlers embody the ability to perceive state. They:
    - Return data without side effects
    - Support caching
    - Are idempotent
    """

    @property
    def supports_caching(self) -> bool:
        """Query handlers can have their results cached."""
        return True


class INotificationHandler(ABC, Generic[TNotification]):
    """
    Handler for notifications - the LISTENERS.

    Notification handlers embody receptivity - they receive testimony
    of what has occurred. Multiple handlers can listen to the same
    notification type.
    """

    @classmethod
    def get_handles_notification_type(cls) -> Optional[Type]:
        """Get the notification type this handler handles."""
        if hasattr(cls, "__seraphic_handles_notification__"):
            return cls.__seraphic_handles_notification__
        return None

    @abstractmethod
    async def handle(self, notification: TNotification) -> None:
        """
        Handle the notification.

        Notification handlers don't return values - they simply receive
        and respond to testimony.
        """
        pass

    def introspect(self) -> Dict[str, Any]:
        """Return self-knowledge of this notification handler."""
        handled = self.get_handles_notification_type()
        return {
            "handler_type": self.__class__.__name__,
            "handles_notification": (
                handled.__name__ if handled else None
            ),
        }


# =============================================================================
# PIPELINE BEHAVIORS - INTRINSIC DNA
# =============================================================================
# "Behaviors are not applied to requests - they are part of the request's nature.
# A command that needs validation IS validatable. A query that can be cached
# IS cacheable. The behavior doesn't wrap - it expresses."
# =============================================================================


class IPipelineBehavior(ABC, Generic[TRequest, TResult]):
    """
    Pipeline behavior for cross-cutting concerns.

    In the seraphic architecture, behaviors are not external middleware -
    they are intrinsic aspects of request processing. The pipeline:

        Logging → Metrics → Validation → Caching → Retry → Transaction → Handler

    Each layer expresses a different aspect of the request's nature:
    - Logging: The request's testimony of its journey
    - Metrics: The request's contribution to collective knowledge
    - Validation: The request's self-knowledge of validity
    - Caching: The query's idempotent nature
    - Retry: The command's resilient intent
    - Transaction: The command's atomic will

    Behaviors self-register and self-prioritize through the BehaviorSpec system.
    """

    # -------------------------------------------------------------------------
    # Identity: The Behavior Knows Its Purpose
    # -------------------------------------------------------------------------

    @classmethod
    def get_priority(cls) -> BehaviorPriority:
        """
        Get this behavior's pipeline priority.

        Override in subclasses to set priority.
        """
        return BehaviorPriority.CUSTOM

    @classmethod
    def get_applies_to(cls) -> Optional[Type]:
        """
        Get the request type this behavior applies to.

        None means it applies to all requests.
        Override in subclasses to restrict application.
        """
        return None

    # -------------------------------------------------------------------------
    # Condition: The Behavior Knows When To Act
    # -------------------------------------------------------------------------

    def should_apply(self, request: Any) -> bool:
        """
        Whether this behavior should apply to a specific request.

        Override in subclasses for conditional application.
        """
        applies_to = self.get_applies_to()
        if applies_to is None:
            return True
        return isinstance(request, applies_to)

    # -------------------------------------------------------------------------
    # The Core Purpose: Handle In The Pipeline
    # -------------------------------------------------------------------------

    @abstractmethod
    async def handle(
        self,
        request: TRequest,
        next: Callable[[], Awaitable[TResult]],
    ) -> TResult:
        """
        Handle the request in the pipeline.

        Args:
            request: The request being processed
            next: Delegate to call the next behavior or handler

        Returns:
            The result from the handler or modified result
        """
        pass

    # -------------------------------------------------------------------------
    # Introspection
    # -------------------------------------------------------------------------

    def introspect(self) -> Dict[str, Any]:
        """Return self-knowledge of this behavior."""
        applies_to = self.get_applies_to()
        return {
            "behavior_type": self.__class__.__name__,
            "priority": self.get_priority().name,
            "applies_to": applies_to.__name__ if applies_to else "all",
        }


# =============================================================================
# CONCRETE PIPELINE BEHAVIORS
# =============================================================================


class LoggingBehavior(IPipelineBehavior[TRequest, TResult], Generic[TRequest, TResult]):
    """
    Pipeline behavior that logs request handling.

    Logs the request type, duration, and any errors.
    """

    def __init__(self, log_level: int = logging.INFO) -> None:
        self._log_level = log_level

    async def handle(
        self,
        request: TRequest,
        next: RequestHandlerDelegate[TResult]
    ) -> TResult:
        request_type = type(request).__name__
        request_id = getattr(request, "request_id", "unknown")

        logger.log(
            self._log_level,
            f"[{request_id}] Starting {request_type}"
        )
        start_time = time.perf_counter()

        try:
            result = await next()
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.log(
                self._log_level,
                f"[{request_id}] Completed {request_type} in {duration_ms:.2f}ms"
            )
            return result
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"[{request_id}] Failed {request_type} after {duration_ms:.2f}ms: {e}"
            )
            raise


class ValidationBehavior(IPipelineBehavior[TRequest, TResult], Generic[TRequest, TResult]):
    """
    Pipeline behavior that validates requests before handling.

    Calls validate() method on requests that have it.
    """

    async def handle(
        self,
        request: TRequest,
        next: RequestHandlerDelegate[TResult]
    ) -> TResult:
        # Check if request has a validate method
        validate_method = getattr(request, "validate", None)
        if callable(validate_method):
            validation_result = validate_method()
            if validation_result:
                # Ensure we have a list of strings
                if isinstance(validation_result, list):
                    raise ValidationError(validation_result)
                elif isinstance(validation_result, str):
                    raise ValidationError([validation_result])
                else:
                    raise ValidationError([str(validation_result)])

        return await next()


class ValidationError(Exception):
    """Raised when request validation fails."""

    def __init__(self, errors: List[str]) -> None:
        self.errors = errors
        super().__init__(f"Validation failed: {', '.join(errors)}")


class TransactionBehavior(IPipelineBehavior[TCommand, TResult], Generic[TCommand, TResult]):
    """
    Pipeline behavior that wraps commands in a database transaction.

    Commits on success, rolls back on failure.
    """

    def __init__(self, unit_of_work_factory: Callable[[], Any]) -> None:
        self._uow_factory = unit_of_work_factory

    async def handle(
        self,
        request: TCommand,  # noqa: ARG002 - Transaction behavior doesn't inspect request
        next: RequestHandlerDelegate[TResult]
    ) -> TResult:
        del request  # Unused but required by interface
        uow = self._uow_factory()
        async with uow:
            try:
                result = await next()
                await uow.commit()
                return result
            except Exception:
                await uow.rollback()
                raise


class PerformanceMonitoringBehavior(
    IPipelineBehavior[TRequest, TResult], Generic[TRequest, TResult]
):
    """
    Pipeline behavior that monitors performance metrics.

    Records execution time, memory usage, and other metrics.
    """

    def __init__(
        self,
        metrics_collector: Optional[Any] = None,
        slow_threshold_ms: float = 1000.0
    ) -> None:
        self._metrics = metrics_collector
        self._slow_threshold_ms = slow_threshold_ms

    async def handle(
        self,
        request: TRequest,
        next: RequestHandlerDelegate[TResult]
    ) -> TResult:
        request_type = type(request).__name__
        start_time = time.perf_counter()

        result = await next()

        duration_ms = (time.perf_counter() - start_time) * 1000

        if duration_ms > self._slow_threshold_ms:
            logger.warning(
                f"Slow request: {request_type} took {duration_ms:.2f}ms"
            )

        if self._metrics:
            self._metrics.record_duration(request_type, duration_ms)

        return result


class RetryBehavior(IPipelineBehavior[TRequest, TResult], Generic[TRequest, TResult]):
    """
    Pipeline behavior that retries failed requests.

    Implements exponential backoff with configurable retry count.
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay_seconds: float = 0.1,
        exponential_base: float = 2.0,
        retryable_exceptions: Optional[tuple] = None
    ) -> None:
        self._max_retries = max_retries
        self._base_delay = base_delay_seconds
        self._exponential_base = exponential_base
        self._retryable = retryable_exceptions or (Exception,)

    async def handle(
        self,
        request: TRequest,
        next: RequestHandlerDelegate[TResult]
    ) -> TResult:
        last_exception: Optional[Exception] = None

        for attempt in range(self._max_retries + 1):
            try:
                return await next()
            except self._retryable as e:
                last_exception = e
                if attempt < self._max_retries:
                    delay = self._base_delay * (self._exponential_base ** attempt)
                    logger.warning(
                        f"Retry {attempt + 1}/{self._max_retries} "
                        f"for {type(request).__name__} after {delay:.2f}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    raise

        # Should never reach here, but satisfy type checker
        raise last_exception  # type: ignore


class CachingBehavior(IPipelineBehavior[TQuery, TResult], Generic[TQuery, TResult]):
    """
    Pipeline behavior that caches query results.

    Only applies to Query types, not Commands.
    """

    def __init__(
        self,
        cache: Optional[Dict[str, Any]] = None,
        ttl_seconds: float = 300.0
    ) -> None:
        self._cache: Dict[str, tuple[Any, float]] = cache or {}
        self._ttl = ttl_seconds

    def _make_cache_key(self, request: TQuery) -> str:
        """Generate cache key from request."""
        request_dict = {
            k: v for k, v in vars(request).items()
            if not k.startswith("_")
        }
        return f"{type(request).__name__}:{hash(frozenset(request_dict.items()))}"

    async def handle(
        self,
        request: TQuery,
        next: RequestHandlerDelegate[TResult]
    ) -> TResult:
        cache_key = self._make_cache_key(request)
        current_time = time.time()

        # Check cache
        if cache_key in self._cache:
            cached_value, cached_time = self._cache[cache_key]
            if current_time - cached_time < self._ttl:
                logger.debug(f"Cache hit for {cache_key}")
                return cached_value

        # Execute and cache
        result = await next()
        self._cache[cache_key] = (result, current_time)
        return result


# =============================================================================
# MEDIATOR IMPLEMENTATION - THE EMERGENT SPACE
# =============================================================================
# "The mediator is not a coordinator giving orders. It is the SPACE where
# requests and handlers find each other through intrinsic affinity. Like
# how a seraph doesn't have a 'coordination center' - its unity emerges
# from the harmony of its parts."
# =============================================================================


class Mediator:
    """
    The Seraphic Mediator: Not a coordinator, but an emergent space.

    In the seraphic architecture, the mediator is transformed from an
    external coordinator into an emergent property of the request-handler
    relationship. It provides:

    1. DISCOVERY SPACE - Where requests find their handlers through affinity
    2. BEHAVIORAL CONTEXT - Where request DNA expresses as pipeline behaviors
    3. COLLECTIVE AWARENESS - Where all requests/handlers are visible
    4. HEALTH MONITORING - Where the health of the whole is known

    The mediator integrates with the SeraphicRegistry to enable:
    - Auto-discovery of handlers decorated with @handler_for
    - Auto-application of behaviors declared with @with_behaviors
    - Fallback to explicit registration for legacy code

    Usage (Seraphic - handlers self-register):
        # Handlers declare their affinity
        @handler_for(ProcessVerseCommand)
        class ProcessVerseHandler(ICommandHandler[ProcessVerseCommand, Result]):
            async def handle(self, command: ProcessVerseCommand) -> Result:
                ...

        # Mediator awakens and discovers handlers
        mediator = Mediator.awaken()
        result = await mediator.send(ProcessVerseCommand(verse_id="GEN.1.1"))

    Usage (Legacy - explicit registration):
        mediator = Mediator()
        mediator.register_handler(ProcessVerseCommand, ProcessVerseHandler())
        mediator.add_behavior(LoggingBehavior())
        result = await mediator.send(ProcessVerseCommand(verse_id="GEN.1.1"))

    Usage (Seraphic - requests know their handling):
        # Requests can execute themselves
        command = ProcessVerseCommand(verse_id="GEN.1.1")
        result = await command.execute_through()  # No mediator needed!
    """

    # -------------------------------------------------------------------------
    # Class-Level Factory: Awakening the Mediator
    # -------------------------------------------------------------------------

    _default_instance: ClassVar[Optional["Mediator"]] = None

    @classmethod
    def awaken(
        cls,
        handler_factory: Optional[Callable[[Type], Any]] = None,
    ) -> "Mediator":
        """
        Awaken a new mediator that auto-discovers handlers.

        The mediator awakens into the space of meaning defined by
        the SeraphicRegistry, discovering all handlers that have
        declared their affinity through decorators.

        Args:
            handler_factory: Optional factory for creating handler instances.
                           Used for dependency injection integration.

        Returns:
            A newly awakened mediator.
        """
        mediator = cls()
        mediator._handler_factory = handler_factory
        mediator._use_registry_discovery = True

        # Track in the registry
        SeraphicRegistry.instance().track_mediator(mediator)

        logger.info("Mediator awakened with seraphic discovery enabled")
        return mediator

    @classmethod
    def get_default(cls) -> "Mediator":
        """Get or create the default mediator instance."""
        if cls._default_instance is None:
            cls._default_instance = cls.awaken()
        return cls._default_instance

    @classmethod
    def set_default(cls, mediator: "Mediator") -> None:
        """Set the default mediator instance."""
        cls._default_instance = mediator

    # -------------------------------------------------------------------------
    # Instance Initialization
    # -------------------------------------------------------------------------

    def __init__(self) -> None:
        """Initialize the mediator's internal structures."""
        # Local handler registrations (for explicit registration)
        self._handlers: Dict[Type, IRequestHandler] = {}
        self._notification_handlers: Dict[Type, List[INotificationHandler]] = {}

        # Local behavior lists (for explicit registration)
        self._behaviors: List[IPipelineBehavior] = []
        self._command_behaviors: List[IPipelineBehavior] = []
        self._query_behaviors: List[IPipelineBehavior] = []

        # Seraphic discovery settings
        self._use_registry_discovery: bool = False
        self._handler_factory: Optional[Callable[[Type], Any]] = None

        # Health and metrics
        self._requests_processed: int = 0
        self._requests_failed: int = 0
        self._created_at: datetime = datetime.now(timezone.utc)

    # -------------------------------------------------------------------------
    # Handler Registration (Legacy Support)
    # -------------------------------------------------------------------------

    def register_handler(
        self,
        request_type: Type[TRequest],
        handler: IRequestHandler[TRequest, Any]
    ) -> None:
        """
        Register a handler for a request type.

        This is the legacy explicit registration method. In seraphic mode,
        handlers self-register through the @handler_for decorator.
        """
        self._handlers[request_type] = handler
        logger.debug(f"Registered handler for {request_type.__name__}")

    def register_notification_handler(
        self,
        notification_type: Type[TNotification],
        handler: INotificationHandler[TNotification]
    ) -> None:
        """Register a handler for a notification type."""
        if notification_type not in self._notification_handlers:
            self._notification_handlers[notification_type] = []
        self._notification_handlers[notification_type].append(handler)
        logger.debug(f"Registered notification handler for {notification_type.__name__}")

    # -------------------------------------------------------------------------
    # Behavior Registration (Legacy Support)
    # -------------------------------------------------------------------------

    def add_behavior(self, behavior: IPipelineBehavior) -> None:
        """Add a pipeline behavior that applies to all requests."""
        self._behaviors.append(behavior)

    def add_command_behavior(self, behavior: IPipelineBehavior) -> None:
        """Add a pipeline behavior that applies only to commands."""
        self._command_behaviors.append(behavior)

    def add_query_behavior(self, behavior: IPipelineBehavior) -> None:
        """Add a pipeline behavior that applies only to queries."""
        self._query_behaviors.append(behavior)

    # -------------------------------------------------------------------------
    # Handler Discovery: Finding Through Affinity
    # -------------------------------------------------------------------------

    def _find_handler(self, request_type: Type) -> Optional[IRequestHandler]:
        """
        Find a handler for a request type through multiple discovery paths.

        Discovery order:
        1. Local explicit registration
        2. Request's declared handler (via @handles)
        3. Registry's handler discovery (via @handler_for)
        """
        # First: check local registration
        if request_type in self._handlers:
            return self._handlers[request_type]

        # Second: use seraphic discovery if enabled
        if self._use_registry_discovery:
            registry = SeraphicRegistry.instance()
            handler_type = registry.find_handler_type(request_type)

            if handler_type is not None:
                # Get or create handler instance
                handler = registry.get_handler_instance(
                    handler_type,
                    factory=self._handler_factory,
                )
                # Cache locally for future use
                self._handlers[request_type] = handler
                return handler

        return None

    def _get_behaviors(self, request: IRequest) -> List[IPipelineBehavior]:
        """
        Get all behaviors that apply to a request.

        Behavior sources (in priority order):
        1. Request's intrinsic behaviors (via @with_behaviors)
        2. Global mediator behaviors
        3. Type-specific mediator behaviors (command/query)
        """
        behaviors: List[IPipelineBehavior] = []

        # Start with local mediator behaviors
        behaviors.extend(self._behaviors)

        # Add type-specific behaviors
        if isinstance(request, Command):
            behaviors.extend(self._command_behaviors)
        elif isinstance(request, Query):
            behaviors.extend(self._query_behaviors)

        # In seraphic mode, also include request's intrinsic behaviors
        if self._use_registry_discovery:
            behavior_specs = request.behaviors
            # Sort by priority
            behavior_specs.sort(key=lambda b: b.priority.value)
            # Create instances for applicable specs
            for spec in behavior_specs:
                if spec.should_apply(request):
                    behaviors.append(spec.create_behavior())

        return behaviors

    # -------------------------------------------------------------------------
    # Request Processing: The Space Where Things Happen
    # -------------------------------------------------------------------------

    async def send(self, request: IRequest[TResult]) -> TResult:
        """
        Send a request through the mediator.

        The request finds its handler through affinity, passes through
        its behavioral pipeline, and returns the result.

        In seraphic mode, this is equivalent to request.execute_through()
        but with the mediator providing context and tracking.

        Args:
            request: Command or Query to process

        Returns:
            Result from the handler

        Raises:
            ValueError: If no handler can be found for the request type
        """
        request_type = type(request)

        # Find handler through discovery
        handler = self._find_handler(request_type)
        if handler is None:
            raise ValueError(
                f"No handler found for {request_type.__name__}. "
                f"Use @handler_for decorator or register_handler()."
            )

        # Get behaviors (merged from mediator and request)
        behaviors = self._get_behaviors(request)

        # Build the pipeline chain
        async def final_handler() -> TResult:
            return await handler.handle(request)

        pipeline = self._build_pipeline(request, behaviors, final_handler)

        # Execute with tracking
        try:
            result = await pipeline()
            self._requests_processed += 1
            return result
        except Exception:
            self._requests_failed += 1
            raise

    def _build_pipeline(
        self,
        request: Any,
        behaviors: List[IPipelineBehavior],
        handler: Callable[[], Awaitable[TResult]]
    ) -> Callable[[], Awaitable[TResult]]:
        """Build the pipeline of behaviors wrapping the handler."""
        current: Callable[[], Awaitable[Any]] = handler

        # Build from inside out (last behavior wraps handler first)
        for behavior in reversed(behaviors):
            # Capture behavior and next in closure properly
            captured_behavior = behavior
            captured_next = current
            captured_request = request

            async def make_step(
                b: IPipelineBehavior,
                n: Callable[[], Awaitable[Any]],
                r: Any
            ) -> Any:
                return await b.handle(r, n)

            # Create closure that captures current values
            current = (
                lambda b=captured_behavior, n=captured_next, r=captured_request:
                make_step(b, n, r)
            )  # type: ignore

        return current  # type: ignore

    # -------------------------------------------------------------------------
    # Notification Publishing
    # -------------------------------------------------------------------------

    async def publish(self, notification: INotification) -> None:
        """
        Publish a notification to all handlers.

        In seraphic mode, handlers are discovered through the registry.
        All handlers are executed concurrently.
        """
        notification_type = type(notification)

        # Get handlers from local registration
        handlers = list(self._notification_handlers.get(notification_type, []))

        # In seraphic mode, also discover from registry
        if self._use_registry_discovery:
            registry = SeraphicRegistry.instance()
            handler_types = registry.get_notification_handler_types(notification_type)
            for ht in handler_types:
                handler = registry.get_handler_instance(
                    ht, factory=self._handler_factory
                )
                if handler not in handlers:
                    handlers.append(handler)

        if not handlers:
            logger.debug(f"No handlers for notification {notification_type.__name__}")
            return

        # Execute all handlers concurrently
        await asyncio.gather(
            *(handler.handle(notification) for handler in handlers),
            return_exceptions=True
        )

    async def publish_domain_event(self, event: DomainEvent) -> None:
        """Publish a domain event as a notification."""
        notification = DomainEventNotification(event=event)
        await self.publish(notification)

    async def publish_domain_events(self, events: List[DomainEvent]) -> None:
        """Publish multiple domain events."""
        for event in events:
            await self.publish_domain_event(event)

    # -------------------------------------------------------------------------
    # Health and Introspection: The Mediator Knows Itself
    # -------------------------------------------------------------------------

    @property
    def is_healthy(self) -> bool:
        """Whether the mediator is healthy."""
        return True  # Override in subclasses for custom health checks

    @property
    def stats(self) -> Dict[str, Any]:
        """Get mediator statistics."""
        return {
            "requests_processed": self._requests_processed,
            "requests_failed": self._requests_failed,
            "success_rate": (
                self._requests_processed / max(1, self._requests_processed + self._requests_failed)
            ),
            "uptime_seconds": (
                datetime.now(timezone.utc) - self._created_at
            ).total_seconds(),
            "use_registry_discovery": self._use_registry_discovery,
        }

    def introspect(self) -> Dict[str, Any]:
        """
        Return complete self-knowledge of this mediator.

        Includes all registered handlers, behaviors, and registry state.
        """
        return {
            "mediator_type": self.__class__.__name__,
            "created_at": self._created_at.isoformat(),
            "use_registry_discovery": self._use_registry_discovery,
            "stats": self.stats,
            "local_handlers": {
                req.__name__: handler.__class__.__name__
                for req, handler in self._handlers.items()
            },
            "local_notification_handlers": {
                notif.__name__: [h.__class__.__name__ for h in handlers]
                for notif, handlers in self._notification_handlers.items()
            },
            "behaviors": [type(b).__name__ for b in self._behaviors],
            "command_behaviors": [type(b).__name__ for b in self._command_behaviors],
            "query_behaviors": [type(b).__name__ for b in self._query_behaviors],
            "registry": SeraphicRegistry.instance().introspect()
            if self._use_registry_discovery else None,
        }


# =============================================================================
# MEDIATOR BUILDER - SERAPHIC CONFIGURATION
# =============================================================================


class MediatorBuilder:
    """
    Builder for constructing a configured Mediator.

    Supports both legacy explicit configuration and seraphic auto-discovery.

    Usage (Legacy):
        mediator = (MediatorBuilder()
            .with_logging()
            .with_validation()
            .with_transaction(uow_factory)
            .register_handler(ProcessVerseCommand, ProcessVerseHandler())
            .build())

    Usage (Seraphic):
        mediator = (MediatorBuilder()
            .with_seraphic_discovery()  # Enable auto-discovery
            .with_logging()             # Add global behaviors
            .with_validation()
            .build())
    """

    def __init__(self) -> None:
        self._mediator = Mediator()
        self._seraphic = False
        self._handler_factory: Optional[Callable[[Type], Any]] = None

    def with_seraphic_discovery(
        self,
        handler_factory: Optional[Callable[[Type], Any]] = None,
    ) -> "MediatorBuilder":
        """
        Enable seraphic auto-discovery mode.

        In this mode, handlers decorated with @handler_for are automatically
        discovered, and requests can use behaviors declared with @with_behaviors.

        Args:
            handler_factory: Optional factory for creating handler instances.
                           Useful for dependency injection integration.
        """
        self._seraphic = True
        self._handler_factory = handler_factory
        return self

    def with_logging(self, log_level: int = logging.INFO) -> "MediatorBuilder":
        """Add logging behavior."""
        self._mediator.add_behavior(LoggingBehavior(log_level))
        return self

    def with_validation(self) -> "MediatorBuilder":
        """Add validation behavior."""
        self._mediator.add_behavior(ValidationBehavior())
        return self

    def with_transaction(
        self, unit_of_work_factory: Callable[[], Any]
    ) -> "MediatorBuilder":
        """Add transaction behavior for commands."""
        self._mediator.add_command_behavior(
            TransactionBehavior(unit_of_work_factory)
        )
        return self

    def with_performance_monitoring(
        self,
        metrics_collector: Optional[Any] = None,
        slow_threshold_ms: float = 1000.0
    ) -> "MediatorBuilder":
        """Add performance monitoring behavior."""
        self._mediator.add_behavior(
            PerformanceMonitoringBehavior(metrics_collector, slow_threshold_ms)
        )
        return self

    def with_retry(
        self,
        max_retries: int = 3,
        base_delay_seconds: float = 0.1,
        retryable_exceptions: Optional[tuple] = None
    ) -> "MediatorBuilder":
        """Add retry behavior."""
        self._mediator.add_behavior(
            RetryBehavior(
                max_retries=max_retries,
                base_delay_seconds=base_delay_seconds,
                retryable_exceptions=retryable_exceptions
            )
        )
        return self

    def with_query_caching(
        self,
        ttl_seconds: float = 300.0
    ) -> "MediatorBuilder":
        """Add caching behavior for queries."""
        self._mediator.add_query_behavior(CachingBehavior(ttl_seconds=ttl_seconds))
        return self

    def with_behavior(self, behavior: IPipelineBehavior) -> "MediatorBuilder":
        """Add a custom behavior."""
        self._mediator.add_behavior(behavior)
        return self

    def register_handler(
        self,
        request_type: Type[TRequest],
        handler: IRequestHandler[TRequest, Any]
    ) -> "MediatorBuilder":
        """Register a request handler."""
        self._mediator.register_handler(request_type, handler)
        return self

    def register_notification_handler(
        self,
        notification_type: Type[TNotification],
        handler: INotificationHandler[TNotification]
    ) -> "MediatorBuilder":
        """Register a notification handler."""
        self._mediator.register_notification_handler(notification_type, handler)
        return self

    def build(self) -> Mediator:
        """Build and return the configured mediator."""
        if self._seraphic:
            self._mediator._use_registry_discovery = True
            self._mediator._handler_factory = self._handler_factory
            SeraphicRegistry.instance().track_mediator(self._mediator)
            logger.info("Built mediator with seraphic discovery enabled")
        return self._mediator


# =============================================================================
# SERAPHIC MEDIATOR ALIAS
# =============================================================================

# For clarity, SeraphicMediator is an alias for Mediator when used in seraphic mode
SeraphicMediator = Mediator


# =============================================================================
# COMMON COMMAND/QUERY DEFINITIONS - WITH SERAPHIC VALIDATION
# =============================================================================
# These commands demonstrate seraphic self-validation. Each command knows
# whether it is valid without external validation logic.
# =============================================================================


@dataclass
class ProcessVerseCommand(Command[Dict[str, Any]]):
    """
    Command to process a verse through the extraction pipeline.

    This command carries intrinsic validation - it knows if it is valid
    without external validators.
    """
    verse_id: str
    pipeline_id: str = field(default_factory=lambda: str(uuid4()))
    force_reprocess: bool = False

    def validate(self) -> ValidationResult:
        """Intrinsic validation - the command knows its own validity."""
        errors: List[str] = []

        if not self.verse_id:
            errors.append("verse_id is required")
        elif not self._is_valid_verse_id(self.verse_id):
            errors.append(f"Invalid verse_id format: {self.verse_id}")

        if errors:
            return ValidationResult.failure(*errors)
        return ValidationResult.success()

    def _is_valid_verse_id(self, verse_id: str) -> bool:
        """Check if verse_id follows expected format (e.g., GEN.1.1)."""
        parts = verse_id.split(".")
        return len(parts) >= 2  # At least book.chapter


@dataclass
class DiscoverCrossReferencesCommand(Command[List[Dict[str, Any]]]):
    """Command to discover cross-references for a verse."""
    verse_id: str
    top_k: int = 10
    min_confidence: float = 0.5
    connection_types: Optional[List[str]] = None

    def validate(self) -> ValidationResult:
        """Intrinsic validation."""
        errors: List[str] = []

        if not self.verse_id:
            errors.append("verse_id is required")
        if self.top_k < 1:
            errors.append("top_k must be at least 1")
        if not 0.0 <= self.min_confidence <= 1.0:
            errors.append("min_confidence must be between 0.0 and 1.0")

        if errors:
            return ValidationResult.failure(*errors)
        return ValidationResult.success()


@dataclass
class VerifyCrossReferenceCommand(Command[bool]):
    """Command to verify a cross-reference."""
    crossref_id: str
    verification_type: str  # "human", "patristic", "ml"
    verifier: str
    notes: Optional[str] = None

    VALID_VERIFICATION_TYPES = {"human", "patristic", "ml", "automated"}

    def validate(self) -> ValidationResult:
        """Intrinsic validation."""
        errors: List[str] = []

        if not self.crossref_id:
            errors.append("crossref_id is required")
        if not self.verification_type:
            errors.append("verification_type is required")
        elif self.verification_type not in self.VALID_VERIFICATION_TYPES:
            errors.append(
                f"verification_type must be one of: {self.VALID_VERIFICATION_TYPES}"
            )
        if not self.verifier:
            errors.append("verifier is required")

        if errors:
            return ValidationResult.failure(*errors)
        return ValidationResult.success()


@dataclass
class CertifyGoldenRecordCommand(Command[Dict[str, Any]]):
    """Command to certify a golden record."""
    verse_id: str
    force_certification: bool = False

    def validate(self) -> ValidationResult:
        """Intrinsic validation."""
        if not self.verse_id:
            return ValidationResult.failure("verse_id is required")
        return ValidationResult.success()


@dataclass
class GetVerseQuery(Query[Optional[Dict[str, Any]]]):
    """
    Query to get a verse by ID.

    Queries can also have intrinsic cache key generation.
    """
    verse_id: str
    include_extractions: bool = False
    include_cross_references: bool = False

    @property
    def cache_key(self) -> str:
        """Custom cache key incorporating all parameters."""
        return (
            f"verse:{self.verse_id}:"
            f"ext={self.include_extractions}:"
            f"xref={self.include_cross_references}"
        )

    def validate(self) -> ValidationResult:
        """Intrinsic validation."""
        if not self.verse_id:
            return ValidationResult.failure("verse_id is required")
        return ValidationResult.success()


@dataclass
class GetCrossReferencesQuery(Query[List[Dict[str, Any]]]):
    """Query to get cross-references for a verse."""
    verse_id: str
    direction: str = "both"  # "outgoing", "incoming", "both"
    connection_types: Optional[List[str]] = None
    min_confidence: float = 0.0
    include_unverified: bool = True


@dataclass
class SearchVersesQuery(Query[List[Dict[str, Any]]]):
    """Query to search verses by text."""
    search_text: str
    book_code: Optional[str] = None
    testament: Optional[str] = None
    limit: int = 10
    offset: int = 0


@dataclass
class GetGoldenRecordQuery(Query[Optional[Dict[str, Any]]]):
    """Query to get a golden record."""
    verse_id: str
    include_all_data: bool = True


@dataclass
class GetPipelineStatusQuery(Query[Dict[str, Any]]):
    """Query to get pipeline processing status."""
    pipeline_id: Optional[str] = None
    verse_id: Optional[str] = None
