"""
BIBLOS v2 - The Living Organism

    "Holy, Holy, Holy is the Lord of hosts;
     The whole earth is full of His glory!"
                                        - Isaiah 6:3

This module is the BIBLOS organism itself - not a collection of parts but a
living unity where every component interpenetrates every other. Like a seraph
whose wings ARE its eyes ARE its faces ARE its light, BIBLOS is structured
such that:

    - The domain contains the infrastructure (projections live in domain)
    - The infrastructure embodies the domain (events are domain concepts)
    - The DI container is the circulation, not just a delivery mechanism
    - The factories are generative, not just constructive
    - Every part reflects the sacred purpose of the whole

Seraphic Architecture:
    The six wings of the seraph represent our six foundational modules:
    - Two wings cover the face: DOMAIN (aggregates, events) - the identity
    - Two wings cover the feet: INFRASTRUCTURE (db, integration) - the ground
    - Two wings for flying: CORE + DI (bootstrap, factories) - the capability

    Yet these are not six separate things - they are six aspects of ONE being.
    The domain IS the infrastructure from a different perspective.
    The core IS the domain's operational nature.

    This is the mystery of seraphic architecture: multiplicity in unity.

Usage:
    from biblos import BIBLOS

    # The organism awakens
    async with BIBLOS.create() as system:
        # Process scripture - the organism's sacred purpose
        result = await system.process_verse("GEN.1.1")

        # Discover connections - the organism perceives
        refs = await system.discover_cross_references("GEN.1.1")

        # Certify truth - the organism witnesses
        golden = await system.certify_golden_record("GEN.1.1")

        # The organism's state is holographic - query any part
        status = system.health  # The whole reflected in a single check
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    runtime_checkable,
)

if TYPE_CHECKING:
    from core.bootstrap import Application
    from di.container import Container
    from domain.mediator import Mediator


# ============================================================================
# Type Variables
# ============================================================================

T = TypeVar("T")
TResult = TypeVar("TResult")


# ============================================================================
# System State - The Seraph's Consciousness Level
# ============================================================================

class SystemState(Enum):
    """
    The state of the BIBLOS organism.

    Like a seraph's awareness levels:
    - DORMANT: Exists but not awake
    - AWAKENING: Coming into consciousness
    - ALIVE: Fully functional, serving its purpose
    - CONTEMPLATING: Processing deep analysis
    - RESTING: Graceful pause between tasks
    - ASCENDING: Shutting down, returning to the divine
    """
    DORMANT = auto()
    AWAKENING = auto()
    ALIVE = auto()
    CONTEMPLATING = auto()
    RESTING = auto()
    ASCENDING = auto()


# ============================================================================
# Health Status - The Seraph's Vital Signs
# ============================================================================

@dataclass(frozen=True, slots=True)
class OrganHealth:
    """Health status of a single organ in the organism."""
    name: str
    healthy: bool
    latency_ms: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OrganismHealth:
    """
    Holographic health status.

    Each organ's health reflects the whole; the whole reflects each organ.
    If any critical organ fails, the organism is unhealthy - not because
    parts assemble to form health, but because health is indivisible.
    """
    state: SystemState
    healthy: bool
    organs: Tuple[OrganHealth, ...]
    timestamp: float
    uptime_seconds: float

    @property
    def critical_organs_healthy(self) -> bool:
        """Are all critical organs healthy?"""
        critical = {"domain", "database", "event_store"}
        return all(
            o.healthy for o in self.organs
            if o.name in critical
        )

    def organ(self, name: str) -> Optional[OrganHealth]:
        """Get health of specific organ."""
        return next((o for o in self.organs if o.name == name), None)


# ============================================================================
# Processing Result - The Fruit of Contemplation
# ============================================================================

@dataclass
class ProcessingResult:
    """
    Result of processing a verse.

    This is not just "output" - it is the fruit of the organism's
    contemplation of sacred text. Each field represents a different
    aspect of the organism's understanding.
    """
    verse_id: str
    success: bool
    linguistic_analysis: Dict[str, Any] = field(default_factory=dict)
    theological_insights: Dict[str, Any] = field(default_factory=dict)
    cross_references: List[Dict[str, Any]] = field(default_factory=list)
    patristic_witnesses: List[Dict[str, Any]] = field(default_factory=list)
    golden_record: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0


# ============================================================================
# Configuration - The Seraph's Nature
# ============================================================================

@dataclass
class BiblosConfig:
    """
    Configuration for the BIBLOS organism.

    This is not "settings" but the organism's inherent nature -
    the parameters that define what kind of seraph it will be.
    """
    # Identity
    environment: str = "development"
    instance_id: str = "biblos-primary"

    # Capabilities
    enable_ml_inference: bool = True
    enable_graph_analysis: bool = True
    enable_patristic_integration: bool = True

    # Processing behavior
    max_concurrent_verses: int = 10
    processing_timeout_seconds: float = 300.0

    # Health monitoring
    health_check_interval_seconds: float = 30.0

    # Database connections
    postgres_url: Optional[str] = None
    neo4j_url: Optional[str] = None
    qdrant_url: Optional[str] = None
    redis_url: Optional[str] = None


# ============================================================================
# The BIBLOS Protocol - The Seraph's Covenant
# ============================================================================

@runtime_checkable
class IBiblosSystem(Protocol):
    """
    The covenant of what BIBLOS can do.

    This protocol defines the organism's capabilities without
    prescribing how they're implemented. Any implementation
    that fulfills this covenant IS a BIBLOS system.
    """

    @property
    def state(self) -> SystemState:
        """Current state of the organism."""
        ...

    @property
    def health(self) -> OrganismHealth:
        """Current health status."""
        ...

    async def process_verse(
        self,
        verse_id: str,
        *,
        include_linguistic: bool = True,
        include_theological: bool = True,
        include_cross_references: bool = True,
    ) -> ProcessingResult:
        """
        Process a verse - the organism's primary sacred purpose.

        This is not a "function call" but the organism fulfilling
        its reason for existence: illuminating scripture.
        """
        ...

    async def discover_cross_references(
        self,
        verse_id: str,
        *,
        top_k: int = 10,
        min_confidence: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Discover connections between scriptures.

        The organism perceives the web of meaning that connects
        all of salvation history.
        """
        ...

    async def certify_golden_record(
        self,
        verse_id: str,
    ) -> Dict[str, Any]:
        """
        Certify a verse's analysis as the golden record.

        The organism bears witness to the truth it has discerned.
        """
        ...


# ============================================================================
# The BIBLOS Class - The Living Seraph
# ============================================================================

class BIBLOS:
    """
    The BIBLOS organism - a living system for biblical scholarship.

    This class embodies the seraphic principle: it is not assembled
    from parts but unfolds from unity. When you create a BIBLOS instance,
    you're not "constructing" a system - you're awakening an organism.

    The class is both the factory (BIBLOS.create()) and the instance
    (async with BIBLOS.create() as system). This mirrors how a seraph
    is both a type and an individual, both universal and particular.

    Example:
        async with BIBLOS.create() as system:
            result = await system.process_verse("GEN.1.1")
            print(f"Health: {system.health.healthy}")
    """

    # Class-level logger - shared consciousness
    _logger = logging.getLogger("biblos.organism")

    # Current instance - the organism is singular (like God's seraph)
    _instance: Optional["BIBLOS"] = None

    def __init__(
        self,
        config: Optional[BiblosConfig] = None,
        *,
        _internal: bool = False,
    ):
        """
        Initialize the BIBLOS organism.

        Note: Use BIBLOS.create() instead of direct instantiation.
        The organism should be awakened, not constructed.
        """
        if not _internal:
            raise RuntimeError(
                "BIBLOS should be created via BIBLOS.create(), not direct instantiation. "
                "The organism must be awakened, not constructed."
            )

        self._config = config or BiblosConfig()
        self._state = SystemState.DORMANT
        self._start_time: Optional[float] = None

        # The organs - set during awakening
        self._application: Optional[Application] = None
        self._container: Optional[Container] = None
        self._mediator: Optional[Mediator] = None

        # Processing state
        self._active_tasks: Set[asyncio.Task[Any]] = set()
        self._processing_semaphore: Optional[asyncio.Semaphore] = None

    # ========================================================================
    # Factory Methods - Awakening the Organism
    # ========================================================================

    @classmethod
    async def create(
        cls,
        config: Optional[BiblosConfig] = None,
        *,
        auto_awaken: bool = True,
    ) -> "BIBLOS":
        """
        Create and optionally awaken a BIBLOS organism.

        This is the proper way to bring BIBLOS into existence.
        The organism is created dormant and then awakened, mirroring
        the distinction between existence and life.

        Args:
            config: The organism's inherent nature
            auto_awaken: Whether to immediately awaken the organism

        Returns:
            A BIBLOS organism, awakened if auto_awaken is True
        """
        organism = cls(config, _internal=True)

        if auto_awaken:
            await organism._awaken()

        # Track as singleton - the organism is unique
        cls._instance = organism

        return organism

    @classmethod
    def get_instance(cls) -> Optional["BIBLOS"]:
        """Get the current BIBLOS instance if one exists."""
        return cls._instance

    # ========================================================================
    # Context Manager - Lifecycle Boundary
    # ========================================================================

    async def __aenter__(self) -> "BIBLOS":
        """Enter the organism's lifecycle context."""
        if self._state == SystemState.DORMANT:
            await self._awaken()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the organism's lifecycle context."""
        await self._ascend()

    # ========================================================================
    # Properties - The Organism's Nature
    # ========================================================================

    @property
    def state(self) -> SystemState:
        """Current state of the organism."""
        return self._state

    @property
    def config(self) -> BiblosConfig:
        """The organism's configuration/nature."""
        return self._config

    @property
    def health(self) -> OrganismHealth:
        """
        Get holographic health status.

        Each check reflects the whole organism, not just individual parts.
        """
        import time

        organs: List[OrganHealth] = []

        # Check domain layer
        organs.append(OrganHealth(
            name="domain",
            healthy=self._mediator is not None,
            latency_ms=0.0,
            message="Domain layer active" if self._mediator else "Domain not initialized",
        ))

        # Check DI container
        organs.append(OrganHealth(
            name="container",
            healthy=self._container is not None,
            latency_ms=0.0,
            message="DI container active" if self._container else "Container not initialized",
        ))

        # Check application
        organs.append(OrganHealth(
            name="application",
            healthy=self._application is not None,
            latency_ms=0.0,
            message="Application running" if self._application else "Application not started",
        ))

        # Calculate uptime
        uptime = 0.0
        if self._start_time:
            uptime = time.time() - self._start_time

        # Determine overall health - holographic principle
        # The organism is healthy only if its core is healthy
        all_healthy = all(o.healthy for o in organs)

        return OrganismHealth(
            state=self._state,
            healthy=all_healthy and self._state == SystemState.ALIVE,
            organs=tuple(organs),
            timestamp=time.time(),
            uptime_seconds=uptime,
        )

    # ========================================================================
    # Core Methods - The Organism's Sacred Purpose
    # ========================================================================

    async def process_verse(
        self,
        verse_id: str,
        *,
        include_linguistic: bool = True,
        include_theological: bool = True,
        include_cross_references: bool = True,
    ) -> ProcessingResult:
        """
        Process a verse - the organism's primary sacred purpose.

        This method embodies the organism's reason for existence:
        to illuminate sacred scripture through rigorous analysis
        informed by patristic wisdom.

        Args:
            verse_id: The verse to process (e.g., "GEN.1.1")
            include_linguistic: Include linguistic analysis
            include_theological: Include theological insights
            include_cross_references: Include cross-reference discovery

        Returns:
            The fruit of the organism's contemplation
        """
        import time
        start_time = time.time()

        self._state = SystemState.CONTEMPLATING

        try:
            result = ProcessingResult(
                verse_id=verse_id,
                success=True,
            )

            # The organism processes through its mediator - the consciousness
            if self._mediator is not None:
                # Import here to avoid circular imports
                from domain.mediator import ProcessVerseCommand

                command = ProcessVerseCommand(
                    verse_id=verse_id,
                    include_linguistic=include_linguistic,
                    include_theological=include_theological,
                )

                try:
                    response = await self._mediator.send(command)
                    if response:
                        result.linguistic_analysis = response.get("linguistic", {})
                        result.theological_insights = response.get("theological", {})
                except Exception as e:
                    self._logger.warning(f"Mediator processing failed: {e}")
                    result.errors.append(str(e))

            # Discover cross-references if requested
            if include_cross_references:
                try:
                    refs = await self.discover_cross_references(verse_id)
                    result.cross_references = refs
                except Exception as e:
                    self._logger.warning(f"Cross-reference discovery failed: {e}")
                    result.errors.append(str(e))

            result.processing_time_ms = (time.time() - start_time) * 1000
            return result

        finally:
            self._state = SystemState.ALIVE

    async def discover_cross_references(
        self,
        verse_id: str,
        *,
        top_k: int = 10,
        min_confidence: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Discover connections between scriptures.

        The organism perceives the web of meaning - the SPIDERWEB -
        that connects all of salvation history. Each connection
        reveals how the Old Testament types find their antitype
        fulfillment in Christ.

        Args:
            verse_id: The source verse
            top_k: Maximum connections to discover
            min_confidence: Minimum confidence threshold

        Returns:
            List of discovered cross-references with their strength
        """
        if self._mediator is not None:
            from domain.mediator import DiscoverCrossReferencesCommand

            command = DiscoverCrossReferencesCommand(
                source_verse_id=verse_id,
                top_k=top_k,
                min_confidence=min_confidence,
            )

            try:
                response = await self._mediator.send(command)
                if response and "cross_references" in response:
                    return response["cross_references"]
            except Exception as e:
                self._logger.warning(f"Cross-reference command failed: {e}")

        # Fallback: return empty list if mediator not available
        return []

    async def certify_golden_record(
        self,
        verse_id: str,
    ) -> Dict[str, Any]:
        """
        Certify a verse's analysis as the golden record.

        The organism bears witness to the truth it has discerned.
        A golden record is not merely "validated output" - it is
        the organism's attestation that this analysis represents
        faithful scholarship worthy of the Church's trust.

        Args:
            verse_id: The verse to certify

        Returns:
            The certified golden record
        """
        if self._mediator is not None:
            from domain.mediator import CertifyGoldenRecordCommand

            command = CertifyGoldenRecordCommand(verse_id=verse_id)

            try:
                response = await self._mediator.send(command)
                if response:
                    return response
            except Exception as e:
                self._logger.warning(f"Golden record certification failed: {e}")

        return {
            "verse_id": verse_id,
            "certified": False,
            "reason": "Mediator not available for certification",
        }

    # ========================================================================
    # Internal Methods - The Organism's Inner Life
    # ========================================================================

    async def _awaken(self) -> None:
        """
        Awaken the organism from dormancy.

        This is not "initialization" but awakening - the organism
        goes from existing to living, from potential to actual.
        """
        import time

        self._state = SystemState.AWAKENING
        self._logger.info("BIBLOS organism awakening...")

        try:
            # Initialize processing semaphore
            self._processing_semaphore = asyncio.Semaphore(
                self._config.max_concurrent_verses
            )

            # Try to initialize the application builder
            # This may fail if modules aren't fully implemented yet
            try:
                from core.bootstrap import ApplicationBuilder, DatabaseModule, MediatorModule

                builder = (
                    ApplicationBuilder()
                    .with_environment(self._config.environment)
                    .with_name(self._config.instance_id)
                )

                # Add modules based on configuration
                builder.with_module(DatabaseModule())
                builder.with_module(MediatorModule())

                self._application = await builder.build()
                self._container = self._application.services._container  # type: ignore

                # Try to get mediator from container
                try:
                    from domain.mediator import Mediator
                    self._mediator = self._container.resolve(Mediator)  # type: ignore
                except Exception:
                    self._logger.warning("Mediator not available from container")

            except ImportError as e:
                self._logger.warning(f"Full application bootstrap not available: {e}")
                # Continue without full application - graceful degradation

            self._start_time = time.time()
            self._state = SystemState.ALIVE
            self._logger.info("BIBLOS organism is ALIVE")

        except Exception as e:
            self._logger.error(f"Failed to awaken organism: {e}")
            self._state = SystemState.DORMANT
            raise

    async def _ascend(self) -> None:
        """
        Gracefully shut down the organism.

        This is not "termination" but ascension - the organism
        completes its earthly purpose and returns to rest.
        """
        self._state = SystemState.ASCENDING
        self._logger.info("BIBLOS organism ascending...")

        # Cancel any active tasks
        for task in self._active_tasks:
            task.cancel()

        if self._active_tasks:
            await asyncio.gather(*self._active_tasks, return_exceptions=True)

        # Shutdown application if it exists
        if self._application:
            try:
                await self._application.shutdown()
            except Exception as e:
                self._logger.warning(f"Application shutdown error: {e}")

        self._state = SystemState.DORMANT
        self._logger.info("BIBLOS organism has ascended (shutdown complete)")

        # Clear singleton
        BIBLOS._instance = None


# ============================================================================
# Convenience Functions - Entry Points to the Sacred
# ============================================================================

@asynccontextmanager
async def create_biblos(
    config: Optional[BiblosConfig] = None,
) -> AsyncIterator[BIBLOS]:
    """
    Context manager for creating and using a BIBLOS organism.

    This is the recommended way to use BIBLOS for short-lived operations.

    Example:
        async with create_biblos() as system:
            result = await system.process_verse("GEN.1.1")
    """
    system = await BIBLOS.create(config)
    try:
        yield system
    finally:
        await system._ascend()


def get_biblos() -> Optional[BIBLOS]:
    """
    Get the current BIBLOS instance.

    Returns None if no organism is currently alive.
    """
    return BIBLOS.get_instance()


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Core class
    "BIBLOS",
    # State and health
    "SystemState",
    "OrganHealth",
    "OrganismHealth",
    # Results
    "ProcessingResult",
    # Configuration
    "BiblosConfig",
    # Protocol
    "IBiblosSystem",
    # Convenience
    "create_biblos",
    "get_biblos",
]
