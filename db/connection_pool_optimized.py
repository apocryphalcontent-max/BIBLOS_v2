"""
BIBLOS v2 - Unified Connection Manager (Optimized)

Manages connections to all database systems with health monitoring,
circuit breaker pattern, and graceful shutdown.

Optimization Changes:
1. Increased pool sizes for async workloads (24-agent pipeline)
2. Circuit breaker pattern for fault tolerance
3. Configurable health check intervals
4. Connection warmup on startup
5. Metrics tracking for connection utilization
6. Graceful degradation when non-critical services fail
"""
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
import os
import time

# Use canonical resilience patterns from core
from core.resilience import SyncCircuitBreaker, SyncCircuitBreakerConfig


logger = logging.getLogger("biblos.db.connection_pool")


class ConnectionStatus(Enum):
    """Connection status enumeration."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    CIRCUIT_OPEN = "circuit_open"


# Factory function for creating circuit breakers with default config
def _create_circuit_breaker() -> SyncCircuitBreaker:
    """Create a circuit breaker with default connection pool config."""
    return SyncCircuitBreaker(
        config=SyncCircuitBreakerConfig(
            failure_threshold=5,
            success_threshold=3,
            recovery_timeout_seconds=30.0,
        )
    )


@dataclass
class ConnectionConfig:
    """Configuration for a database connection."""
    name: str
    enabled: bool = True
    connection_string: str = ""
    max_connections: int = 50  # Increased default
    min_connections: int = 5   # New: minimum pool size
    timeout_seconds: int = 30
    retry_attempts: int = 3
    health_check_interval: int = 30  # Reduced for faster detection
    is_critical: bool = True   # New: whether service is required


@dataclass
class ConnectionState:
    """State of a database connection."""
    status: ConnectionStatus = ConnectionStatus.DISCONNECTED
    last_error: Optional[str] = None
    connected_at: Optional[float] = None
    health_check_failures: int = 0
    total_connections: int = 0
    active_connections: int = 0
    circuit_breaker: SyncCircuitBreaker = field(default_factory=_create_circuit_breaker)


@dataclass
class ConnectionMetrics:
    """Metrics for connection monitoring."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0
    last_request_time: float = 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.successful_requests == 0:
            return 0
        return self.total_latency_ms / self.successful_requests


class ConnectionManager:
    """
    Optimized unified connection manager for all BIBLOS databases.

    Features:
    - Circuit breaker pattern for fault tolerance
    - Connection pool warmup
    - Metrics tracking
    - Graceful degradation for non-critical services
    - Configurable health checks

    Manages:
    - PostgreSQL (critical: primary data store)
    - Neo4j (optional: graph relationships)
    - Qdrant (optional: vector search)
    - Redis (optional: caching)
    """

    def __init__(
        self,
        config: Optional[Dict[str, ConnectionConfig]] = None,
        on_connection_change: Optional[Callable[[str, ConnectionStatus], None]] = None
    ):
        self.config = config or self._default_config()
        self._states: Dict[str, ConnectionState] = {
            name: ConnectionState() for name in self.config
        }
        self._metrics: Dict[str, ConnectionMetrics] = {
            name: ConnectionMetrics() for name in self.config
        }
        self._clients: Dict[str, Any] = {}
        self._health_task: Optional[asyncio.Task] = None
        self._on_connection_change = on_connection_change

    def _default_config(self) -> Dict[str, ConnectionConfig]:
        """Generate default connection configuration optimized for 24-agent pipeline."""
        return {
            "postgres": ConnectionConfig(
                name="postgres",
                connection_string=os.getenv(
                    "DATABASE_URL",
                    "postgresql+asyncpg://biblos:biblos@localhost:5432/biblos_v2"
                ),
                max_connections=int(os.getenv("PG_POOL_SIZE", "50")),
                min_connections=10,
                timeout_seconds=30,
                health_check_interval=30,
                is_critical=True  # Required for operation
            ),
            "neo4j": ConnectionConfig(
                name="neo4j",
                connection_string=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                max_connections=int(os.getenv("NEO4J_POOL_SIZE", "30")),
                min_connections=5,
                timeout_seconds=30,
                health_check_interval=30,
                is_critical=False  # Can operate without graph
            ),
            "qdrant": ConnectionConfig(
                name="qdrant",
                connection_string=os.getenv("QDRANT_URL", "http://localhost:6333"),
                max_connections=int(os.getenv("QDRANT_POOL_SIZE", "20")),
                min_connections=3,
                timeout_seconds=30,
                health_check_interval=30,
                is_critical=False  # Can operate without vector search
            ),
            "redis": ConnectionConfig(
                name="redis",
                connection_string=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                max_connections=int(os.getenv("REDIS_POOL_SIZE", "50")),
                min_connections=5,
                timeout_seconds=10,
                health_check_interval=15,
                is_critical=False  # Can operate without caching
            )
        }

    async def initialize(self) -> Dict[str, bool]:
        """Initialize all database connections with warmup."""
        results = {}

        # Connect to all services
        for name, config in self.config.items():
            if not config.enabled:
                logger.info(f"Skipping disabled connection: {name}")
                results[name] = False
                continue

            self._states[name].status = ConnectionStatus.CONNECTING
            success = await self._connect(name, config)
            results[name] = success

            # Notify callback
            if self._on_connection_change:
                status = self._states[name].status
                self._on_connection_change(name, status)

        # Warmup connections
        await self._warmup_connections()

        # Start health check task
        self._health_task = asyncio.create_task(self._health_check_loop())

        # Log summary
        connected = sum(1 for r in results.values() if r)
        logger.info(f"Connection manager initialized: {connected}/{len(results)} connected")

        return results

    async def _connect(self, name: str, config: ConnectionConfig) -> bool:
        """Connect to a specific database."""
        try:
            if name == "postgres":
                from db.postgres_optimized import PostgresClient
                client = PostgresClient(
                    database_url=config.connection_string,
                    pool_size=config.max_connections
                )
                await client.initialize()
                self._clients[name] = client

            elif name == "neo4j":
                from db.neo4j_optimized import Neo4jClient, Neo4jConfig
                neo4j_config = Neo4jConfig(
                    uri=config.connection_string,
                    max_connection_pool_size=config.max_connections
                )
                client = Neo4jClient(config=neo4j_config)
                await client.connect()
                self._clients[name] = client

            elif name == "qdrant":
                from db.qdrant_client import QdrantVectorStore
                import urllib.parse
                parsed = urllib.parse.urlparse(config.connection_string)
                client = QdrantVectorStore(
                    host=parsed.hostname or "localhost",
                    port=parsed.port or 6333
                )
                await client.connect()
                self._clients[name] = client

            elif name == "redis":
                try:
                    import redis.asyncio as aioredis
                    client = await aioredis.from_url(
                        config.connection_string,
                        max_connections=config.max_connections,
                        socket_timeout=config.timeout_seconds
                    )
                    await client.ping()
                    self._clients[name] = client
                except ImportError:
                    logger.warning("Redis client not available")
                    return False

            self._states[name].status = ConnectionStatus.CONNECTED
            self._states[name].connected_at = time.time()
            self._states[name].circuit_breaker.record_success()
            logger.info(f"Connected to {name}")
            return True

        except Exception as e:
            self._states[name].status = ConnectionStatus.ERROR
            self._states[name].last_error = str(e)
            self._states[name].circuit_breaker.record_failure()
            logger.error(f"Failed to connect to {name}: {e}")
            return False

    async def _warmup_connections(self) -> None:
        """Warmup connection pools by making initial queries."""
        logger.info("Warming up connection pools...")

        # PostgreSQL warmup
        if "postgres" in self._clients:
            try:
                async with self._clients["postgres"].session() as session:
                    from sqlalchemy import text
                    await session.execute(text("SELECT 1"))
                logger.debug("PostgreSQL connection warmed up")
            except Exception as e:
                logger.warning(f"PostgreSQL warmup failed: {e}")

        # Neo4j warmup
        if "neo4j" in self._clients:
            try:
                await self._clients["neo4j"].verify_connectivity()
                logger.debug("Neo4j connection warmed up")
            except Exception as e:
                logger.warning(f"Neo4j warmup failed: {e}")

        # Qdrant warmup
        if "qdrant" in self._clients:
            try:
                await self._clients["qdrant"].get_all_collections_info()
                logger.debug("Qdrant connection warmed up")
            except Exception as e:
                logger.warning(f"Qdrant warmup failed: {e}")

        # Redis warmup
        if "redis" in self._clients:
            try:
                await self._clients["redis"].ping()
                logger.debug("Redis connection warmed up")
            except Exception as e:
                logger.warning(f"Redis warmup failed: {e}")

    async def shutdown(self) -> None:
        """Gracefully shutdown all connections."""
        logger.info("Shutting down connection manager...")

        # Cancel health check
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        # Close all connections
        for name, client in self._clients.items():
            try:
                if hasattr(client, 'close'):
                    await client.close()
                elif hasattr(client, 'disconnect'):
                    await client.disconnect()
                logger.info(f"Closed connection: {name}")
            except Exception as e:
                logger.error(f"Error closing {name}: {e}")

        self._clients.clear()
        logger.info("All connections closed")

    async def _health_check_loop(self) -> None:
        """Periodic health check for all connections."""
        while True:
            try:
                # Use minimum interval across all configs
                min_interval = min(
                    c.health_check_interval for c in self.config.values()
                )
                await asyncio.sleep(min_interval)
                await self._run_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    async def _run_health_checks(self) -> Dict[str, bool]:
        """Run health checks for all connections."""
        results = {}

        for name, client in self._clients.items():
            state = self._states[name]

            # Skip if circuit is open
            if not state.circuit_breaker.can_execute():
                logger.debug(f"Skipping health check for {name} - circuit open")
                results[name] = False
                continue

            try:
                start_time = time.time()
                healthy = await self._check_health(name, client)
                latency = (time.time() - start_time) * 1000

                results[name] = healthy

                if healthy:
                    state.health_check_failures = 0
                    state.circuit_breaker.record_success()
                    self._metrics[name].total_latency_ms += latency
                    self._metrics[name].successful_requests += 1
                else:
                    state.health_check_failures += 1
                    state.circuit_breaker.record_failure()
                    self._metrics[name].failed_requests += 1

                    # Attempt reconnect after threshold failures
                    if state.health_check_failures >= 3:
                        logger.warning(f"Attempting reconnect to {name}")
                        await self._reconnect(name)

                self._metrics[name].total_requests += 1
                self._metrics[name].last_request_time = time.time()

            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                state.circuit_breaker.record_failure()
                results[name] = False

        return results

    async def _check_health(self, name: str, client: Any) -> bool:
        """Check health of a specific connection with meaningful queries."""
        try:
            if name == "postgres":
                async with client.session() as session:
                    from sqlalchemy import text
                    # Test actual table access
                    result = await session.execute(
                        text("SELECT COUNT(*) FROM books LIMIT 1")
                    )
                    return result.scalar() is not None

            elif name == "neo4j":
                return await client.verify_connectivity()

            elif name == "qdrant":
                info = await client.get_collection_info("verses")
                return info is not None

            elif name == "redis":
                return await client.ping()

            return False

        except Exception:
            return False

    async def _reconnect(self, name: str) -> bool:
        """Attempt to reconnect to a database."""
        config = self.config.get(name)
        if not config:
            return False

        state = self._states[name]

        # Check circuit breaker
        if not state.circuit_breaker.can_execute():
            logger.info(f"Cannot reconnect to {name} - circuit breaker open")
            return False

        # Close existing connection
        client = self._clients.pop(name, None)
        if client:
            try:
                if hasattr(client, 'close'):
                    await client.close()
            except Exception:
                pass

        # Reconnect
        success = await self._connect(name, config)

        if success and self._on_connection_change:
            self._on_connection_change(name, ConnectionStatus.CONNECTED)

        return success

    # =========================================================================
    # Client access with circuit breaker protection
    # =========================================================================

    def get_postgres(self) -> Optional[Any]:
        """Get PostgreSQL client if available and circuit closed."""
        state = self._states.get("postgres")
        if state and state.circuit_breaker.can_execute():
            return self._clients.get("postgres")
        return None

    def get_neo4j(self) -> Optional[Any]:
        """Get Neo4j client if available and circuit closed."""
        state = self._states.get("neo4j")
        if state and state.circuit_breaker.can_execute():
            return self._clients.get("neo4j")
        return None

    def get_qdrant(self) -> Optional[Any]:
        """Get Qdrant client if available and circuit closed."""
        state = self._states.get("qdrant")
        if state and state.circuit_breaker.can_execute():
            return self._clients.get("qdrant")
        return None

    def get_redis(self) -> Optional[Any]:
        """Get Redis client if available and circuit closed."""
        state = self._states.get("redis")
        if state and state.circuit_breaker.can_execute():
            return self._clients.get("redis")
        return None

    # =========================================================================
    # Status and metrics
    # =========================================================================

    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all connections."""
        return {
            name: {
                "status": state.status.value,
                "last_error": state.last_error,
                "connected": state.status == ConnectionStatus.CONNECTED,
                "health_check_failures": state.health_check_failures,
                "circuit_state": state.circuit_breaker.state.value,
                "circuit_failures": state.circuit_breaker.failure_count,
                "is_critical": self.config[name].is_critical
            }
            for name, state in self._states.items()
        }

    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all connections."""
        return {
            name: {
                "total_requests": metrics.total_requests,
                "successful_requests": metrics.successful_requests,
                "failed_requests": metrics.failed_requests,
                "success_rate": metrics.success_rate,
                "avg_latency_ms": metrics.avg_latency_ms,
                "last_request_time": metrics.last_request_time
            }
            for name, metrics in self._metrics.items()
        }

    def is_healthy(self) -> bool:
        """Check if all critical connections are healthy."""
        for name, config in self.config.items():
            if config.is_critical:
                state = self._states.get(name)
                if not state or state.status != ConnectionStatus.CONNECTED:
                    return False
                if not state.circuit_breaker.can_execute():
                    return False
        return True

    def get_degraded_services(self) -> List[str]:
        """Get list of non-critical services that are unavailable."""
        degraded = []
        for name, config in self.config.items():
            if not config.is_critical:
                state = self._states.get(name)
                if not state or state.status != ConnectionStatus.CONNECTED:
                    degraded.append(name)
        return degraded


# Global connection manager instance
_manager: Optional[ConnectionManager] = None


async def get_connection_manager() -> ConnectionManager:
    """Get or create global connection manager."""
    global _manager
    if _manager is None:
        _manager = ConnectionManager()
        await _manager.initialize()
    return _manager


async def shutdown_connections() -> None:
    """Shutdown global connection manager."""
    global _manager
    if _manager:
        await _manager.shutdown()
        _manager = None
