"""
Tests for db/postgres.py - PostgreSQL Client.

Covers:
- PostgresClient initialization and configuration
- Connection pool management
- Session context manager
- CRUD operations for books, verses, cross-references
- Batch operations optimization
- Vector search with pgvector
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from contextlib import asynccontextmanager


# =============================================================================
# PostgresClient Initialization Tests
# =============================================================================

class TestPostgresClientInit:
    """Tests for PostgresClient initialization."""

    def test_default_configuration(self):
        """Test default configuration values."""
        from db.postgres import PostgresClient

        client = PostgresClient()

        assert client.pool_size == 10
        assert client.max_overflow == 20
        assert client.echo is False
        assert "postgresql+asyncpg" in client.database_url

    def test_custom_configuration(self):
        """Test custom configuration values."""
        from db.postgres import PostgresClient

        client = PostgresClient(
            database_url="postgresql+asyncpg://custom:custom@localhost/test",
            pool_size=5,
            max_overflow=10,
            echo=True
        )

        assert client.pool_size == 5
        assert client.max_overflow == 10
        assert client.echo is True
        assert "custom" in client.database_url

    def test_env_variable_fallback(self, monkeypatch):
        """Test DATABASE_URL environment variable fallback."""
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://env:env@envhost/envdb")

        from db.postgres import PostgresClient
        client = PostgresClient()

        assert "envhost" in client.database_url


# =============================================================================
# Connection Lifecycle Tests
# =============================================================================

class TestPostgresClientLifecycle:
    """Tests for PostgresClient connection lifecycle."""

    @pytest.mark.asyncio
    async def test_initialize_creates_engine(self):
        """Test that initialize creates the engine and session factory."""
        from db.postgres import PostgresClient

        with patch("db.postgres.create_async_engine") as mock_engine:
            with patch("db.postgres.async_sessionmaker") as mock_session:
                mock_engine.return_value = Mock()
                mock_session.return_value = Mock()

                client = PostgresClient()
                await client.initialize()

                mock_engine.assert_called_once()
                mock_session.assert_called_once()
                assert client._engine is not None
                assert client._session_factory is not None

    @pytest.mark.asyncio
    async def test_close_disposes_engine(self):
        """Test that close disposes the engine."""
        from db.postgres import PostgresClient

        mock_engine = Mock()
        mock_engine.dispose = AsyncMock()

        client = PostgresClient()
        client._engine = mock_engine

        await client.close()

        mock_engine.dispose.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_with_no_engine(self):
        """Test close when engine is not initialized."""
        from db.postgres import PostgresClient

        client = PostgresClient()
        # Should not raise
        await client.close()


# =============================================================================
# Session Context Manager Tests
# =============================================================================

class TestPostgresSession:
    """Tests for session context manager."""

    @pytest.mark.asyncio
    async def test_session_commits_on_success(self):
        """Test that session commits on successful completion."""
        from db.postgres import PostgresClient

        mock_session = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()

        mock_factory = Mock()

        @asynccontextmanager
        async def session_cm():
            yield mock_session

        mock_factory.return_value = session_cm()

        client = PostgresClient()
        client._session_factory = mock_factory

        async with client.session():
            pass

        mock_session.commit.assert_called_once()
        mock_session.rollback.assert_not_called()

    @pytest.mark.asyncio
    async def test_session_rollback_on_error(self):
        """Test that session rolls back on error."""
        from db.postgres import PostgresClient

        mock_session = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()

        mock_factory = Mock()

        @asynccontextmanager
        async def session_cm():
            yield mock_session

        mock_factory.return_value = session_cm()

        client = PostgresClient()
        client._session_factory = mock_factory

        with pytest.raises(ValueError):
            async with client.session():
                raise ValueError("Test error")

        mock_session.rollback.assert_called_once()


# =============================================================================
# Book Operations Tests
# =============================================================================

class TestBookOperations:
    """Tests for book operations."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock PostgresClient."""
        from db.postgres import PostgresClient
        client = PostgresClient()

        mock_session = AsyncMock()

        @asynccontextmanager
        async def session_cm():
            yield mock_session

        client._session_factory = Mock(return_value=session_cm())
        client._mock_session = mock_session
        return client

    @pytest.mark.asyncio
    async def test_get_book_found(self, mock_client):
        """Test getting an existing book."""
        from db.models import Book

        mock_book = Book(code="GEN", name_english="Genesis", order_num=1)
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_book
        mock_client._mock_session.execute = AsyncMock(return_value=mock_result)

        result = await mock_client.get_book("GEN")

        assert result is not None
        assert result.code == "GEN"

    @pytest.mark.asyncio
    async def test_get_book_not_found(self, mock_client):
        """Test getting a non-existent book."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_client._mock_session.execute = AsyncMock(return_value=mock_result)

        result = await mock_client.get_book("NONEXISTENT")

        assert result is None


# =============================================================================
# Verse Operations Tests
# =============================================================================

class TestVerseOperations:
    """Tests for verse operations."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock PostgresClient."""
        from db.postgres import PostgresClient
        client = PostgresClient()

        mock_session = AsyncMock()
        mock_session.add = Mock()
        mock_session.flush = AsyncMock()

        @asynccontextmanager
        async def session_cm():
            yield mock_session

        client._session_factory = Mock(return_value=session_cm())
        client._mock_session = mock_session
        return client

    @pytest.mark.asyncio
    async def test_get_verse_found(self, mock_client):
        """Test getting an existing verse."""
        from db.models import Verse

        mock_verse = Mock(spec=Verse)
        mock_verse.reference = "GEN.1.1"
        mock_verse.text_english = "In the beginning..."

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_verse
        mock_client._mock_session.execute = AsyncMock(return_value=mock_result)

        result = await mock_client.get_verse("GEN.1.1")

        assert result is not None
        assert result.reference == "GEN.1.1"

    @pytest.mark.asyncio
    async def test_upsert_verse_new(self, mock_client):
        """Test upserting a new verse."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_client._mock_session.execute = AsyncMock(return_value=mock_result)

        verse_data = {
            "reference": "GEN.1.1",
            "chapter": 1,
            "verse_num": 1,
            "text_english": "In the beginning..."
        }

        await mock_client.upsert_verse(verse_data)

        mock_client._mock_session.add.assert_called_once()
        mock_client._mock_session.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_upsert_verse_existing(self, mock_client):
        """Test upserting an existing verse."""
        from db.models import Verse

        existing_verse = Mock(spec=Verse)
        existing_verse.reference = "GEN.1.1"

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = existing_verse
        mock_client._mock_session.execute = AsyncMock(return_value=mock_result)

        verse_data = {
            "reference": "GEN.1.1",
            "text_english": "Updated text"
        }

        await mock_client.upsert_verse(verse_data)

        # Should not add new, should update existing
        mock_client._mock_session.add.assert_not_called()
        assert existing_verse.text_english == "Updated text"


# =============================================================================
# Batch Operations Tests
# =============================================================================

class TestBatchOperations:
    """Tests for batch operations."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock PostgresClient."""
        from db.postgres import PostgresClient
        client = PostgresClient()

        mock_session = AsyncMock()
        mock_session.add = Mock()
        mock_session.flush = AsyncMock()

        @asynccontextmanager
        async def session_cm():
            yield mock_session

        client._session_factory = Mock(return_value=session_cm())
        client._mock_session = mock_session
        return client

    @pytest.mark.asyncio
    async def test_batch_upsert_verses_empty(self, mock_client):
        """Test batch upsert with empty list."""
        count = await mock_client.batch_upsert_verses([])
        assert count == 0

    @pytest.mark.asyncio
    async def test_batch_upsert_verses(self, mock_client):
        """Test batch upsert with multiple verses."""
        # Mock empty existing result
        mock_scalars = Mock()
        mock_scalars.all.return_value = []

        mock_result = Mock()
        mock_result.scalars.return_value = mock_scalars

        mock_client._mock_session.execute = AsyncMock(return_value=mock_result)

        verses = [
            {"reference": f"GEN.1.{i}", "chapter": 1, "verse_num": i}
            for i in range(1, 6)
        ]

        count = await mock_client.batch_upsert_verses(verses)

        assert count == 5
        assert mock_client._mock_session.add.call_count == 5


# =============================================================================
# Cross-Reference Operations Tests
# =============================================================================

class TestCrossReferenceOperations:
    """Tests for cross-reference operations."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock PostgresClient."""
        from db.postgres import PostgresClient
        client = PostgresClient()

        mock_session = AsyncMock()
        mock_session.add = Mock()
        mock_session.flush = AsyncMock()

        @asynccontextmanager
        async def session_cm():
            yield mock_session

        client._session_factory = Mock(return_value=session_cm())
        client._mock_session = mock_session
        return client

    @pytest.mark.asyncio
    async def test_add_crossref_success(self, mock_client):
        """Test adding a cross-reference successfully."""
        from db.models import Verse

        source_verse = Mock(spec=Verse)
        source_verse.id = 1
        source_verse.reference = "GEN.1.1"

        target_verse = Mock(spec=Verse)
        target_verse.id = 2
        target_verse.reference = "JHN.1.1"

        mock_scalars = Mock()
        mock_scalars.all.return_value = [source_verse, target_verse]

        mock_result = Mock()
        mock_result.scalars.return_value = mock_scalars

        mock_client._mock_session.execute = AsyncMock(return_value=mock_result)

        result = await mock_client.add_crossref(
            "GEN.1.1",
            "JHN.1.1",
            "typological",
            confidence=0.9
        )

        mock_client._mock_session.add.assert_called_once()
        mock_client._mock_session.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_crossref_missing_verse(self, mock_client):
        """Test adding a cross-reference with missing verse."""
        from db.models import Verse

        source_verse = Mock(spec=Verse)
        source_verse.id = 1
        source_verse.reference = "GEN.1.1"

        # Only source exists, target missing
        mock_scalars = Mock()
        mock_scalars.all.return_value = [source_verse]

        mock_result = Mock()
        mock_result.scalars.return_value = mock_scalars

        mock_client._mock_session.execute = AsyncMock(return_value=mock_result)

        result = await mock_client.add_crossref(
            "GEN.1.1",
            "MISSING.1.1",
            "typological"
        )

        assert result is None
        mock_client._mock_session.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_batch_add_crossrefs_empty(self, mock_client):
        """Test batch add crossrefs with empty list."""
        count = await mock_client.batch_add_crossrefs([])
        assert count == 0


# =============================================================================
# Vector Search Tests
# =============================================================================

class TestVectorSearch:
    """Tests for vector search operations."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock PostgresClient."""
        from db.postgres import PostgresClient
        client = PostgresClient()

        mock_session = AsyncMock()

        @asynccontextmanager
        async def session_cm():
            yield mock_session

        client._session_factory = Mock(return_value=session_cm())
        client._mock_session = mock_session
        return client

    @pytest.mark.asyncio
    async def test_find_similar_verses(self, mock_client):
        """Test finding similar verses by embedding."""
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            ("GEN.1.1", "In the beginning...", 0.95),
            ("JHN.1.1", "In the beginning was the Word...", 0.88),
        ]
        mock_client._mock_session.execute = AsyncMock(return_value=mock_result)

        embedding = [0.1] * 768
        results = await mock_client.find_similar_verses(
            embedding,
            limit=10,
            threshold=0.7
        )

        assert len(results) == 2
        assert results[0]["reference"] == "GEN.1.1"
        assert results[0]["similarity"] == 0.95


# =============================================================================
# Statistics Tests
# =============================================================================

class TestStatistics:
    """Tests for statistics operations."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock PostgresClient."""
        from db.postgres import PostgresClient
        client = PostgresClient()

        mock_session = AsyncMock()

        @asynccontextmanager
        async def session_cm():
            yield mock_session

        client._session_factory = Mock(return_value=session_cm())
        client._mock_session = mock_session
        return client

    @pytest.mark.asyncio
    async def test_get_statistics(self, mock_client):
        """Test getting database statistics."""
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            ("books", 66),
            ("verses", 31102),
            ("cross_references", 5000),
            ("patristic_citations", 500),
            ("extraction_results", 10000),
        ]
        mock_client._mock_session.execute = AsyncMock(return_value=mock_result)

        stats = await mock_client.get_statistics()

        assert stats["books"] == 66
        assert stats["verses"] == 31102
        assert stats["cross_references"] == 5000


# =============================================================================
# Global Client Tests
# =============================================================================

class TestGlobalClient:
    """Tests for global client functions."""

    @pytest.mark.asyncio
    async def test_get_db_client_creates_singleton(self):
        """Test that get_db_client creates a singleton."""
        from db import postgres

        # Reset global
        postgres._client = None

        with patch.object(postgres.PostgresClient, "initialize", new_callable=AsyncMock):
            client1 = await postgres.get_db_client()
            client2 = await postgres.get_db_client()

            assert client1 is client2

        # Cleanup
        postgres._client = None
