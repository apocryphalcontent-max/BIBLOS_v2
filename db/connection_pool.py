"""
BIBLOS v2 - Unified Connection Manager

Manages connections to all database systems with health monitoring
and graceful shutdown.
"""
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
import os


logger = logging.getLogger("biblos.db.connection_pool")


class ConnectionStatus(Enum):
    """Connection status enumeration."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class ConnectionConfig:
    """Configuration for a database connection."""
    name: str
    enabled: bool = True
    connection_string: str = ""
    max_connections: int = 10
    timeout_seconds: int = 30
    retry_attempts: int = 3
    health_check_interval: int = 60


@dataclass
class ConnectionState:
    """State of a database connection."""
    status: ConnectionStatus = ConnectionStatus.DISCONNECTED
    last_error: Optional[str] = None
    connected_at: Optional[float] = None
    health_check_failures: int = 0


class ConnectionManager:
    """
    Unified connection manager for all BIBLOS databases.

    Manages:
    - PostgreSQL (primary data store)
    - Neo4j (graph relationships)
    - Qdrant (vector search)
    - Redis (caching)
    """

    def __init__(self, config: Optional[Dict[str, ConnectionConfig]] = None):
        self.config = config or self._default_config()
        self._states: Dict[str, ConnectionState] = {
            name: ConnectionState() for name in self.config
        }
        self._clients: Dict[str, Any] = {}
        self._health_task: Optional[asyncio.Task] = None

    def _default_config(self) -> Dict[str, ConnectionConfig]:
        """Generate default connection configuration."""
        return {
            "postgres": ConnectionConfig(
                name="postgres",
                connection_string=os.getenv(
                    "DATABASE_URL",
                    "postgresql+asyncpg://biblos:biblos@localhost:5432/biblos_v2"
                ),
                max_connections=20
            ),
            "neo4j": ConnectionConfig(
                name="neo4j",
                connection_string=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                max_connections=10
            ),
            "qdrant": ConnectionConfig(
                name="qdrant",
                connection_string=os.getenv("QDRANT_URL", "http://localhost:6333"),
                max_connections=10
            ),
            "redis": ConnectionConfig(
                name="redis",
                connection_string=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                max_connections=20
            )
        }

    async def initialize(self) -> Dict[str, bool]:
        """Initialize all database connections."""
        results = {}

        for name, config in self.config.items():
            if not config.enabled:
                logger.info(f"Skipping disabled connection: {name}")
                results[name] = False
                continue

            self._states[name].status = ConnectionStatus.CONNECTING
            success = await self._connect(name, config)
            results[name] = success

        # Start health check task
        self._health_task = asyncio.create_task(self._health_check_loop())

        return results

    async def _connect(self, name: str, config: ConnectionConfig) -> bool:
        """Connect to a specific database."""
        try:
            if name == "postgres":
                from db.postgres import PostgresClient
                client = PostgresClient(database_url=config.connection_string)
                await client.initialize()
                self._clients[name] = client

            elif name == "neo4j":
                from db.neo4j_client import Neo4jClient
                # Parse connection string for Neo4j
                client = Neo4jClient(uri=config.connection_string)
                await client.connect()
                self._clients[name] = client

            elif name == "qdrant":
                from db.qdrant_client import QdrantVectorStore
                # Parse host/port from URL
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
                        max_connections=config.max_connections
                    )
                    await client.ping()
                    self._clients[name] = client
                except ImportError:
                    logger.warning("Redis client not available")
                    return False

            self._states[name].status = ConnectionStatus.CONNECTED
            self._states[name].connected_at = asyncio.get_event_loop().time()
            logger.info(f"Connected to {name}")
            return True

        except Exception as e:
            self._states[name].status = ConnectionStatus.ERROR
            self._states[name].last_error = str(e)
            logger.error(f"Failed to connect to {name}: {e}")
            return False

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
                await asyncio.sleep(60)  # Check every minute
                await self._run_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    async def _run_health_checks(self) -> Dict[str, bool]:
        """Run health checks for all connections."""
        results = {}

        for name, client in self._clients.items():
            try:
                healthy = await self._check_health(name, client)
                results[name] = healthy

                if healthy:
                    self._states[name].health_check_failures = 0
                else:
                    self._states[name].health_check_failures += 1

                    # Attempt reconnect after 3 failures
                    if self._states[name].health_check_failures >= 3:
                        logger.warning(f"Attempting reconnect to {name}")
                        await self._reconnect(name)

            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                results[name] = False

        return results

    async def _check_health(self, name: str, client: Any) -> bool:
        """Check health of a specific connection."""
        try:
            if name == "postgres":
                async with client.session() as session:
                    await session.execute("SELECT 1")
                return True

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

        # Close existing connection
        client = self._clients.pop(name, None)
        if client:
            try:
                if hasattr(client, 'close'):
                    await client.close()
            except Exception:
                pass

        # Reconnect
        return await self._connect(name, config)

    # Client access
    def get_postgres(self) -> Optional[Any]:
        """Get PostgreSQL client."""
        return self._clients.get("postgres")

    def get_neo4j(self) -> Optional[Any]:
        """Get Neo4j client."""
        return self._clients.get("neo4j")

    def get_qdrant(self) -> Optional[Any]:
        """Get Qdrant client."""
        return self._clients.get("qdrant")

    def get_redis(self) -> Optional[Any]:
        """Get Redis client."""
        return self._clients.get("redis")

    # Status
    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all connections."""
        return {
            name: {
                "status": state.status.value,
                "last_error": state.last_error,
                "connected": state.status == ConnectionStatus.CONNECTED,
                "health_check_failures": state.health_check_failures
            }
            for name, state in self._states.items()
        }

    def is_healthy(self) -> bool:
        """Check if all required connections are healthy."""
        required = ["postgres"]  # Only postgres is truly required
        return all(
            self._states[name].status == ConnectionStatus.CONNECTED
            for name in required
            if name in self._states
        )


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
