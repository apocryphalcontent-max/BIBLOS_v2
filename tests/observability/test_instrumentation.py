"""
Tests for observability/instrumentation.py - Database Instrumentation.

Covers:
- trace_db_operation decorator
- traced_db_session context manager
- InstrumentedPostgresClient wrapper
- InstrumentedNeo4jClient wrapper
- InstrumentedRedisClient wrapper
- Wrapper factory functions
"""
import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock


# =============================================================================
# trace_db_operation Decorator Tests
# =============================================================================

class TestTraceDbOperationDecorator:
    """Tests for trace_db_operation decorator."""

    @pytest.fixture
    def mock_tracer(self):
        """Create a mock tracer."""
        tracer = Mock()
        mock_span = Mock()
        mock_span.set_attribute = Mock()
        mock_span.set_status = Mock()
        mock_span.record_exception = Mock()
        mock_span.__enter__ = Mock(return_value=mock_span)
        mock_span.__exit__ = Mock(return_value=False)
        tracer.start_as_current_span = Mock(return_value=mock_span)
        return tracer, mock_span

    @pytest.mark.asyncio
    async def test_async_function_success(self, mock_tracer):
        """Test decorator on async function with successful execution."""
        tracer, mock_span = mock_tracer

        with patch("observability.instrumentation.tracer", tracer):
            with patch("observability.instrumentation.record_db_query"):
                from observability.instrumentation import trace_db_operation

                @trace_db_operation("query", "postgres")
                async def test_func():
                    return "result"

                result = await test_func()

                assert result == "result"
                tracer.start_as_current_span.assert_called_once()
                mock_span.set_attribute.assert_any_call("db.system", "postgres")
                mock_span.set_attribute.assert_any_call("db.operation", "query")

    @pytest.mark.asyncio
    async def test_async_function_error(self, mock_tracer):
        """Test decorator on async function with error."""
        tracer, mock_span = mock_tracer

        with patch("observability.instrumentation.tracer", tracer):
            with patch("observability.instrumentation.record_db_query"):
                from observability.instrumentation import trace_db_operation

                @trace_db_operation("query", "postgres")
                async def test_func():
                    raise ValueError("Test error")

                with pytest.raises(ValueError):
                    await test_func()

                mock_span.record_exception.assert_called_once()

    @pytest.mark.asyncio
    async def test_include_result_count_list(self, mock_tracer):
        """Test that result count is recorded for list results."""
        tracer, mock_span = mock_tracer

        with patch("observability.instrumentation.tracer", tracer):
            with patch("observability.instrumentation.record_db_query"):
                from observability.instrumentation import trace_db_operation

                @trace_db_operation("query", "postgres", include_result=True)
                async def test_func():
                    return [1, 2, 3, 4, 5]

                await test_func()

                mock_span.set_attribute.assert_any_call("db.result_count", 5)

    def test_sync_function_success(self, mock_tracer):
        """Test decorator on sync function."""
        tracer, mock_span = mock_tracer

        with patch("observability.instrumentation.tracer", tracer):
            with patch("observability.instrumentation.record_db_query"):
                from observability.instrumentation import trace_db_operation

                @trace_db_operation("query", "postgres")
                def test_func():
                    return "result"

                result = test_func()

                assert result == "result"
                tracer.start_as_current_span.assert_called_once()


# =============================================================================
# traced_db_session Context Manager Tests
# =============================================================================

class TestTracedDbSession:
    """Tests for traced_db_session context manager."""

    @pytest.fixture
    def mock_tracer(self):
        """Create a mock tracer."""
        tracer = Mock()
        mock_span = Mock()
        mock_span.set_attribute = Mock()
        mock_span.set_status = Mock()
        mock_span.record_exception = Mock()
        mock_span.__enter__ = Mock(return_value=mock_span)
        mock_span.__exit__ = Mock(return_value=False)
        tracer.start_as_current_span = Mock(return_value=mock_span)
        return tracer, mock_span

    @pytest.mark.asyncio
    async def test_session_success(self, mock_tracer):
        """Test successful session context."""
        tracer, mock_span = mock_tracer

        with patch("observability.instrumentation.tracer", tracer):
            from observability.instrumentation import traced_db_session

            async with traced_db_session("postgres") as span:
                assert span is mock_span

            mock_span.set_attribute.assert_any_call("db.system", "postgres")

    @pytest.mark.asyncio
    async def test_session_error(self, mock_tracer):
        """Test session context with error."""
        tracer, mock_span = mock_tracer

        with patch("observability.instrumentation.tracer", tracer):
            from observability.instrumentation import traced_db_session

            with pytest.raises(ValueError):
                async with traced_db_session("postgres"):
                    raise ValueError("Test error")

            mock_span.record_exception.assert_called_once()


# =============================================================================
# InstrumentedPostgresClient Tests
# =============================================================================

class TestInstrumentedPostgresClient:
    """Tests for InstrumentedPostgresClient wrapper."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock PostgresClient."""
        client = AsyncMock()
        client.initialize = AsyncMock()
        client.close = AsyncMock()
        client.get_verse = AsyncMock(return_value=Mock(reference="GEN.1.1"))
        client.get_verses_by_book = AsyncMock(return_value=[Mock(), Mock()])
        client.upsert_verse = AsyncMock()
        client.batch_upsert_verses = AsyncMock(return_value=10)
        client.get_crossrefs_for_verse = AsyncMock(return_value=[])
        client.add_crossref = AsyncMock()
        client.save_extraction_result = AsyncMock()
        client.find_similar_verses = AsyncMock(return_value=[])
        client.get_statistics = AsyncMock(return_value={})
        return client

    @pytest.fixture
    def mock_tracer(self):
        """Create a mock tracer."""
        tracer = Mock()
        mock_span = Mock()
        mock_span.set_attribute = Mock()
        mock_span.set_status = Mock()
        mock_span.__enter__ = Mock(return_value=mock_span)
        mock_span.__exit__ = Mock(return_value=False)
        tracer.start_as_current_span = Mock(return_value=mock_span)
        return tracer

    @pytest.mark.asyncio
    async def test_initialize(self, mock_client, mock_tracer):
        """Test initialize with tracing."""
        with patch("observability.instrumentation.get_tracer", return_value=mock_tracer):
            from observability.instrumentation import InstrumentedPostgresClient

            instrumented = InstrumentedPostgresClient(mock_client)
            await instrumented.initialize()

            mock_client.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_close(self, mock_client, mock_tracer):
        """Test close with tracing."""
        with patch("observability.instrumentation.get_tracer", return_value=mock_tracer):
            from observability.instrumentation import InstrumentedPostgresClient

            instrumented = InstrumentedPostgresClient(mock_client)
            await instrumented.close()

            mock_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_verse(self, mock_client, mock_tracer):
        """Test get_verse with tracing."""
        with patch("observability.instrumentation.get_tracer", return_value=mock_tracer):
            with patch("observability.instrumentation.tracer", mock_tracer):
                with patch("observability.instrumentation.record_db_query"):
                    from observability.instrumentation import InstrumentedPostgresClient

                    instrumented = InstrumentedPostgresClient(mock_client)
                    result = await instrumented.get_verse("GEN.1.1")

                    assert result.reference == "GEN.1.1"
                    mock_client.get_verse.assert_called_once_with("GEN.1.1")

    @pytest.mark.asyncio
    async def test_get_statistics(self, mock_client, mock_tracer):
        """Test get_statistics with tracing."""
        with patch("observability.instrumentation.get_tracer", return_value=mock_tracer):
            with patch("observability.instrumentation.tracer", mock_tracer):
                with patch("observability.instrumentation.record_db_query"):
                    from observability.instrumentation import InstrumentedPostgresClient

                    instrumented = InstrumentedPostgresClient(mock_client)
                    await instrumented.get_statistics()

                    mock_client.get_statistics.assert_called_once()


# =============================================================================
# InstrumentedNeo4jClient Tests
# =============================================================================

class TestInstrumentedNeo4jClient:
    """Tests for InstrumentedNeo4jClient wrapper."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Neo4jClient."""
        client = AsyncMock()
        client.connect = AsyncMock()
        client.close = AsyncMock()
        client.verify_connectivity = AsyncMock(return_value=True)
        client.create_indexes = AsyncMock()
        client.create_verse_node = AsyncMock(return_value="node-123")
        client.get_verse_node = AsyncMock()
        client.create_cross_reference = AsyncMock()
        client.get_cross_references = AsyncMock(return_value=[])
        client.create_church_father = AsyncMock()
        client.link_father_to_verse = AsyncMock()
        client.find_shortest_path = AsyncMock(return_value=[])
        client.get_verse_neighborhood = AsyncMock()
        client.get_graph_statistics = AsyncMock(return_value={})
        return client

    @pytest.fixture
    def mock_tracer(self):
        """Create a mock tracer."""
        tracer = Mock()
        mock_span = Mock()
        mock_span.set_attribute = Mock()
        mock_span.set_status = Mock()
        mock_span.__enter__ = Mock(return_value=mock_span)
        mock_span.__exit__ = Mock(return_value=False)
        tracer.start_as_current_span = Mock(return_value=mock_span)
        return tracer

    @pytest.mark.asyncio
    async def test_connect(self, mock_client, mock_tracer):
        """Test connect with tracing."""
        with patch("observability.instrumentation.get_tracer", return_value=mock_tracer):
            from observability.instrumentation import InstrumentedNeo4jClient

            instrumented = InstrumentedNeo4jClient(mock_client)
            await instrumented.connect()

            mock_client.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_verify_connectivity(self, mock_client, mock_tracer):
        """Test verify_connectivity with tracing."""
        with patch("observability.instrumentation.get_tracer", return_value=mock_tracer):
            with patch("observability.instrumentation.tracer", mock_tracer):
                with patch("observability.instrumentation.record_db_query"):
                    from observability.instrumentation import InstrumentedNeo4jClient

                    instrumented = InstrumentedNeo4jClient(mock_client)
                    result = await instrumented.verify_connectivity()

                    assert result is True

    @pytest.mark.asyncio
    async def test_create_verse_node(self, mock_client, mock_tracer):
        """Test create_verse_node with tracing."""
        with patch("observability.instrumentation.get_tracer", return_value=mock_tracer):
            with patch("observability.instrumentation.tracer", mock_tracer):
                with patch("observability.instrumentation.record_db_query"):
                    from observability.instrumentation import InstrumentedNeo4jClient

                    instrumented = InstrumentedNeo4jClient(mock_client)
                    result = await instrumented.create_verse_node(
                        "GEN.1.1",
                        {"text": "In the beginning..."}
                    )

                    assert result == "node-123"


# =============================================================================
# InstrumentedRedisClient Tests
# =============================================================================

class TestInstrumentedRedisClient:
    """Tests for InstrumentedRedisClient wrapper."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Redis client."""
        client = AsyncMock()
        client.get = AsyncMock(return_value=b"value")
        client.set = AsyncMock(return_value=True)
        client.delete = AsyncMock(return_value=1)
        client.exists = AsyncMock(return_value=1)
        client.mget = AsyncMock(return_value=[b"val1", None, b"val3"])
        return client

    @pytest.fixture
    def mock_tracer(self):
        """Create a mock tracer."""
        tracer = Mock()
        mock_span = Mock()
        mock_span.set_attribute = Mock()
        mock_span.set_status = Mock()
        mock_span.__enter__ = Mock(return_value=mock_span)
        mock_span.__exit__ = Mock(return_value=False)
        tracer.start_as_current_span = Mock(return_value=mock_span)
        return tracer

    @pytest.mark.asyncio
    async def test_get_cache_hit(self, mock_client, mock_tracer):
        """Test get with cache hit."""
        with patch("observability.instrumentation.get_tracer", return_value=mock_tracer):
            with patch("observability.instrumentation.record_db_query"):
                from observability.instrumentation import InstrumentedRedisClient

                instrumented = InstrumentedRedisClient(mock_client)
                result = await instrumented.get("test_key")

                assert result == b"value"
                mock_span = mock_tracer.start_as_current_span.return_value.__enter__.return_value
                mock_span.set_attribute.assert_any_call("cache.hit", True)

    @pytest.mark.asyncio
    async def test_get_cache_miss(self, mock_client, mock_tracer):
        """Test get with cache miss."""
        mock_client.get = AsyncMock(return_value=None)

        with patch("observability.instrumentation.get_tracer", return_value=mock_tracer):
            with patch("observability.instrumentation.record_db_query"):
                from observability.instrumentation import InstrumentedRedisClient

                instrumented = InstrumentedRedisClient(mock_client)
                result = await instrumented.get("missing_key")

                assert result is None
                mock_span = mock_tracer.start_as_current_span.return_value.__enter__.return_value
                mock_span.set_attribute.assert_any_call("cache.hit", False)

    @pytest.mark.asyncio
    async def test_set_with_ttl(self, mock_client, mock_tracer):
        """Test set with TTL."""
        with patch("observability.instrumentation.get_tracer", return_value=mock_tracer):
            with patch("observability.instrumentation.record_db_query"):
                from observability.instrumentation import InstrumentedRedisClient

                instrumented = InstrumentedRedisClient(mock_client)
                result = await instrumented.set("key", b"value", ex=3600)

                assert result is True
                mock_client.set.assert_called_once_with("key", b"value", ex=3600, px=None)

    @pytest.mark.asyncio
    async def test_delete(self, mock_client, mock_tracer):
        """Test delete with tracing."""
        with patch("observability.instrumentation.get_tracer", return_value=mock_tracer):
            with patch("observability.instrumentation.record_db_query"):
                from observability.instrumentation import InstrumentedRedisClient

                instrumented = InstrumentedRedisClient(mock_client)
                result = await instrumented.delete("key1", "key2")

                assert result == 1
                mock_span = mock_tracer.start_as_current_span.return_value.__enter__.return_value
                mock_span.set_attribute.assert_any_call("db.redis.key_count", 2)

    @pytest.mark.asyncio
    async def test_mget_cache_stats(self, mock_client, mock_tracer):
        """Test mget records cache hit/miss stats."""
        with patch("observability.instrumentation.get_tracer", return_value=mock_tracer):
            with patch("observability.instrumentation.record_db_query"):
                from observability.instrumentation import InstrumentedRedisClient

                instrumented = InstrumentedRedisClient(mock_client)
                result = await instrumented.mget("key1", "key2", "key3")

                assert len(result) == 3
                mock_span = mock_tracer.start_as_current_span.return_value.__enter__.return_value
                # 2 hits (val1, val3), 1 miss (None)
                mock_span.set_attribute.assert_any_call("cache.hits", 2)
                mock_span.set_attribute.assert_any_call("cache.misses", 1)


# =============================================================================
# Wrapper Factory Function Tests
# =============================================================================

class TestWrapperFactories:
    """Tests for wrapper factory functions."""

    def test_wrap_postgres_client(self):
        """Test wrap_postgres_client factory."""
        with patch("observability.instrumentation.get_tracer"):
            from observability.instrumentation import wrap_postgres_client, InstrumentedPostgresClient

            mock_client = Mock()
            wrapped = wrap_postgres_client(mock_client)

            assert isinstance(wrapped, InstrumentedPostgresClient)
            assert wrapped._client is mock_client

    def test_wrap_neo4j_client(self):
        """Test wrap_neo4j_client factory."""
        with patch("observability.instrumentation.get_tracer"):
            from observability.instrumentation import wrap_neo4j_client, InstrumentedNeo4jClient

            mock_client = Mock()
            wrapped = wrap_neo4j_client(mock_client)

            assert isinstance(wrapped, InstrumentedNeo4jClient)
            assert wrapped._client is mock_client

    def test_wrap_redis_client(self):
        """Test wrap_redis_client factory."""
        with patch("observability.instrumentation.get_tracer"):
            from observability.instrumentation import wrap_redis_client, InstrumentedRedisClient

            mock_client = Mock()
            wrapped = wrap_redis_client(mock_client)

            assert isinstance(wrapped, InstrumentedRedisClient)
            assert wrapped._client is mock_client
