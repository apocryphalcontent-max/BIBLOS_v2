"""
BIBLOS v2 - Database Instrumentation

Auto-instrumentation wrappers for database clients with OpenTelemetry tracing.
Provides tracing for PostgreSQL (via SQLAlchemy), Neo4j, and Redis operations.
"""
from __future__ import annotations

import functools
import time
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, Optional, TypeVar

from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from observability import get_tracer, get_logger
from observability.metrics import record_db_query

T = TypeVar("T")
tracer = get_tracer("biblos.db")
logger = get_logger(__name__)


def trace_db_operation(
    operation: str,
    database: str = "unknown",
    include_result: bool = False,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for tracing database operations.

    Args:
        operation: Name of the operation (query, insert, update, etc.)
        database: Database type (postgres, neo4j, redis)
        include_result: Whether to record result count in span

    Returns:
        Decorated function with tracing
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            with tracer.start_as_current_span(
                f"db.{database}.{operation}",
                kind=SpanKind.CLIENT,
            ) as span:
                span.set_attribute("db.system", database)
                span.set_attribute("db.operation", operation)

                start_time = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)

                    duration = time.perf_counter() - start_time
                    span.set_attribute("db.duration_ms", duration * 1000)

                    if include_result and result is not None:
                        if isinstance(result, list):
                            span.set_attribute("db.result_count", len(result))
                        elif hasattr(result, "__len__"):
                            span.set_attribute("db.result_count", len(result))

                    span.set_status(Status(StatusCode.OK))
                    record_db_query(operation, duration, database)

                    return result

                except Exception as e:
                    duration = time.perf_counter() - start_time
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    span.set_attribute("db.duration_ms", duration * 1000)

                    record_db_query(operation, duration, database)
                    logger.error(
                        f"Database operation failed: {operation}",
                        database=database,
                        operation=operation,
                        error=str(e),
                    )
                    raise

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            with tracer.start_as_current_span(
                f"db.{database}.{operation}",
                kind=SpanKind.CLIENT,
            ) as span:
                span.set_attribute("db.system", database)
                span.set_attribute("db.operation", operation)

                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)

                    duration = time.perf_counter() - start_time
                    span.set_attribute("db.duration_ms", duration * 1000)

                    if include_result and result is not None:
                        if isinstance(result, list):
                            span.set_attribute("db.result_count", len(result))

                    span.set_status(Status(StatusCode.OK))
                    record_db_query(operation, duration, database)

                    return result

                except Exception as e:
                    duration = time.perf_counter() - start_time
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)

                    record_db_query(operation, duration, database)
                    raise

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


@asynccontextmanager
async def traced_db_session(database: str = "postgres"):
    """
    Async context manager for tracing database session lifecycle.

    Usage:
        async with traced_db_session("postgres") as span:
            # Database operations
            span.set_attribute("custom.attribute", "value")
    """
    with tracer.start_as_current_span(
        f"db.{database}.session",
        kind=SpanKind.CLIENT,
    ) as span:
        span.set_attribute("db.system", database)
        start_time = time.perf_counter()

        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
        finally:
            duration = time.perf_counter() - start_time
            span.set_attribute("db.session_duration_ms", duration * 1000)


class InstrumentedPostgresClient:
    """
    Wrapper for PostgreSQL client with automatic tracing.

    Wraps the PostgresClient to add OpenTelemetry spans to all operations.
    """

    def __init__(self, client):
        self._client = client
        self._tracer = get_tracer("biblos.db.postgres")

    async def initialize(self) -> None:
        """Initialize with tracing."""
        with self._tracer.start_as_current_span(
            "db.postgres.initialize",
            kind=SpanKind.CLIENT,
        ) as span:
            await self._client.initialize()
            span.set_status(Status(StatusCode.OK))

    async def close(self) -> None:
        """Close with tracing."""
        with self._tracer.start_as_current_span(
            "db.postgres.close",
            kind=SpanKind.CLIENT,
        ):
            await self._client.close()

    @asynccontextmanager
    async def session(self):
        """Get traced database session."""
        async with traced_db_session("postgres") as span:
            async with self._client.session() as session:
                yield session

    @trace_db_operation("get_verse", "postgres", include_result=False)
    async def get_verse(self, reference: str):
        """Get verse with tracing."""
        return await self._client.get_verse(reference)

    @trace_db_operation("get_verses_by_book", "postgres", include_result=True)
    async def get_verses_by_book(self, book_code: str):
        """Get verses by book with tracing."""
        return await self._client.get_verses_by_book(book_code)

    @trace_db_operation("upsert_verse", "postgres")
    async def upsert_verse(self, verse_data):
        """Upsert verse with tracing."""
        return await self._client.upsert_verse(verse_data)

    @trace_db_operation("batch_upsert_verses", "postgres")
    async def batch_upsert_verses(self, verses):
        """Batch upsert with tracing."""
        return await self._client.batch_upsert_verses(verses)

    @trace_db_operation("get_crossrefs", "postgres", include_result=True)
    async def get_crossrefs_for_verse(self, verse_ref: str):
        """Get cross-references with tracing."""
        return await self._client.get_crossrefs_for_verse(verse_ref)

    @trace_db_operation("add_crossref", "postgres")
    async def add_crossref(self, source_ref: str, target_ref: str, connection_type: str, **kwargs):
        """Add cross-reference with tracing."""
        return await self._client.add_crossref(source_ref, target_ref, connection_type, **kwargs)

    @trace_db_operation("save_extraction", "postgres")
    async def save_extraction_result(self, verse_ref: str, agent_name: str, extraction_type: str, data, confidence: float, **kwargs):
        """Save extraction result with tracing."""
        return await self._client.save_extraction_result(verse_ref, agent_name, extraction_type, data, confidence, **kwargs)

    @trace_db_operation("find_similar", "postgres", include_result=True)
    async def find_similar_verses(self, embedding, limit: int = 10, threshold: float = 0.7):
        """Find similar verses with tracing."""
        return await self._client.find_similar_verses(embedding, limit, threshold)

    @trace_db_operation("get_statistics", "postgres")
    async def get_statistics(self):
        """Get statistics with tracing."""
        return await self._client.get_statistics()


class InstrumentedNeo4jClient:
    """
    Wrapper for Neo4j client with automatic tracing.

    Wraps the Neo4jClient to add OpenTelemetry spans to all graph operations.
    """

    def __init__(self, client):
        self._client = client
        self._tracer = get_tracer("biblos.db.neo4j")

    async def connect(self) -> None:
        """Connect with tracing."""
        with self._tracer.start_as_current_span(
            "db.neo4j.connect",
            kind=SpanKind.CLIENT,
        ) as span:
            await self._client.connect()
            span.set_status(Status(StatusCode.OK))

    async def close(self) -> None:
        """Close with tracing."""
        with self._tracer.start_as_current_span(
            "db.neo4j.close",
            kind=SpanKind.CLIENT,
        ):
            await self._client.close()

    @trace_db_operation("verify_connectivity", "neo4j")
    async def verify_connectivity(self) -> bool:
        """Verify connectivity with tracing."""
        return await self._client.verify_connectivity()

    @trace_db_operation("create_indexes", "neo4j")
    async def create_indexes(self) -> None:
        """Create indexes with tracing."""
        return await self._client.create_indexes()

    @trace_db_operation("create_verse_node", "neo4j")
    async def create_verse_node(self, reference: str, properties: Dict[str, Any]):
        """Create verse node with tracing."""
        return await self._client.create_verse_node(reference, properties)

    @trace_db_operation("get_verse_node", "neo4j")
    async def get_verse_node(self, reference: str):
        """Get verse node with tracing."""
        return await self._client.get_verse_node(reference)

    @trace_db_operation("create_cross_reference", "neo4j")
    async def create_cross_reference(self, source_ref: str, target_ref: str, rel_type: str, properties: Optional[Dict[str, Any]] = None):
        """Create cross-reference relationship with tracing."""
        return await self._client.create_cross_reference(source_ref, target_ref, rel_type, properties)

    @trace_db_operation("get_cross_references", "neo4j", include_result=True)
    async def get_cross_references(self, verse_ref: str, direction: str = "both", rel_types=None):
        """Get cross-references with tracing."""
        return await self._client.get_cross_references(verse_ref, direction, rel_types)

    @trace_db_operation("create_church_father", "neo4j")
    async def create_church_father(self, name: str, properties: Dict[str, Any]):
        """Create church father node with tracing."""
        return await self._client.create_church_father(name, properties)

    @trace_db_operation("link_father_to_verse", "neo4j")
    async def link_father_to_verse(self, father_name: str, verse_ref: str, citation_type: str = "CITED_BY", properties=None):
        """Link father to verse with tracing."""
        return await self._client.link_father_to_verse(father_name, verse_ref, citation_type, properties)

    @trace_db_operation("find_shortest_path", "neo4j", include_result=True)
    async def find_shortest_path(self, source_ref: str, target_ref: str, max_depth: int = 5):
        """Find shortest path with tracing."""
        return await self._client.find_shortest_path(source_ref, target_ref, max_depth)

    @trace_db_operation("get_verse_neighborhood", "neo4j")
    async def get_verse_neighborhood(self, verse_ref: str, depth: int = 2):
        """Get verse neighborhood with tracing."""
        return await self._client.get_verse_neighborhood(verse_ref, depth)

    @trace_db_operation("get_graph_statistics", "neo4j")
    async def get_graph_statistics(self):
        """Get graph statistics with tracing."""
        return await self._client.get_graph_statistics()


class InstrumentedRedisClient:
    """
    Wrapper for Redis client with automatic tracing.

    Wraps Redis operations to add OpenTelemetry spans for cache monitoring.
    """

    def __init__(self, client):
        self._client = client
        self._tracer = get_tracer("biblos.db.redis")

    async def get(self, key: str) -> Optional[bytes]:
        """Get value with tracing."""
        with self._tracer.start_as_current_span(
            "db.redis.get",
            kind=SpanKind.CLIENT,
        ) as span:
            span.set_attribute("db.system", "redis")
            span.set_attribute("db.operation", "GET")
            span.set_attribute("db.redis.key", key[:50])  # Truncate key

            start_time = time.perf_counter()
            result = await self._client.get(key)
            duration = time.perf_counter() - start_time

            span.set_attribute("db.duration_ms", duration * 1000)
            span.set_attribute("cache.hit", result is not None)

            record_db_query("get", duration, "redis")

            return result

    async def set(
        self,
        key: str,
        value: bytes,
        ex: Optional[int] = None,
        px: Optional[int] = None,
    ) -> bool:
        """Set value with tracing."""
        with self._tracer.start_as_current_span(
            "db.redis.set",
            kind=SpanKind.CLIENT,
        ) as span:
            span.set_attribute("db.system", "redis")
            span.set_attribute("db.operation", "SET")
            span.set_attribute("db.redis.key", key[:50])
            if ex:
                span.set_attribute("db.redis.ttl_seconds", ex)

            start_time = time.perf_counter()
            result = await self._client.set(key, value, ex=ex, px=px)
            duration = time.perf_counter() - start_time

            span.set_attribute("db.duration_ms", duration * 1000)

            record_db_query("set", duration, "redis")

            return result

    async def delete(self, *keys: str) -> int:
        """Delete keys with tracing."""
        with self._tracer.start_as_current_span(
            "db.redis.delete",
            kind=SpanKind.CLIENT,
        ) as span:
            span.set_attribute("db.system", "redis")
            span.set_attribute("db.operation", "DEL")
            span.set_attribute("db.redis.key_count", len(keys))

            start_time = time.perf_counter()
            result = await self._client.delete(*keys)
            duration = time.perf_counter() - start_time

            span.set_attribute("db.duration_ms", duration * 1000)
            span.set_attribute("db.deleted_count", result)

            record_db_query("delete", duration, "redis")

            return result

    async def exists(self, *keys: str) -> int:
        """Check key existence with tracing."""
        with self._tracer.start_as_current_span(
            "db.redis.exists",
            kind=SpanKind.CLIENT,
        ) as span:
            span.set_attribute("db.system", "redis")
            span.set_attribute("db.operation", "EXISTS")

            start_time = time.perf_counter()
            result = await self._client.exists(*keys)
            duration = time.perf_counter() - start_time

            span.set_attribute("db.duration_ms", duration * 1000)

            record_db_query("exists", duration, "redis")

            return result

    async def mget(self, *keys: str):
        """Multi-get with tracing."""
        with self._tracer.start_as_current_span(
            "db.redis.mget",
            kind=SpanKind.CLIENT,
        ) as span:
            span.set_attribute("db.system", "redis")
            span.set_attribute("db.operation", "MGET")
            span.set_attribute("db.redis.key_count", len(keys))

            start_time = time.perf_counter()
            result = await self._client.mget(*keys)
            duration = time.perf_counter() - start_time

            span.set_attribute("db.duration_ms", duration * 1000)
            hits = sum(1 for r in result if r is not None)
            span.set_attribute("cache.hits", hits)
            span.set_attribute("cache.misses", len(keys) - hits)

            record_db_query("mget", duration, "redis")

            return result


def wrap_postgres_client(client) -> InstrumentedPostgresClient:
    """Wrap a PostgresClient with instrumentation."""
    return InstrumentedPostgresClient(client)


def wrap_neo4j_client(client) -> InstrumentedNeo4jClient:
    """Wrap a Neo4jClient with instrumentation."""
    return InstrumentedNeo4jClient(client)


def wrap_redis_client(client) -> InstrumentedRedisClient:
    """Wrap a Redis client with instrumentation."""
    return InstrumentedRedisClient(client)
