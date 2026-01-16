"""
Tests for db/neo4j_client.py - Neo4j Graph Database Client.

Covers:
- Neo4jClient initialization
- Connection lifecycle
- Verse node operations
- Cross-reference operations
- Church Father connections
- Thematic category operations
- Graph analysis (paths, neighborhoods)
- Statistics
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import dataclass


# =============================================================================
# Initialization Tests
# =============================================================================

class TestNeo4jClientInit:
    """Tests for Neo4jClient initialization."""

    def test_default_configuration(self):
        """Test default configuration values."""
        from db.neo4j_client import Neo4jClient

        client = Neo4jClient()

        assert "bolt://localhost:7687" in client.uri
        assert client.user == "neo4j"
        assert client._driver is None

    def test_custom_configuration(self):
        """Test custom configuration values."""
        from db.neo4j_client import Neo4jClient

        client = Neo4jClient(
            uri="bolt://custom:7474",
            user="custom_user",
            password="custom_pass"
        )

        assert client.uri == "bolt://custom:7474"
        assert client.user == "custom_user"
        assert client.password == "custom_pass"

    def test_env_variable_fallback(self, monkeypatch):
        """Test environment variable fallback."""
        monkeypatch.setenv("NEO4J_URI", "bolt://envhost:7687")
        monkeypatch.setenv("NEO4J_USER", "envuser")
        monkeypatch.setenv("NEO4J_PASSWORD", "envpass")

        from db.neo4j_client import Neo4jClient
        client = Neo4jClient()

        assert "envhost" in client.uri
        assert client.user == "envuser"
        assert client.password == "envpass"


# =============================================================================
# Data Classes Tests
# =============================================================================

class TestDataClasses:
    """Tests for GraphNode and GraphRelationship dataclasses."""

    def test_graph_node_creation(self):
        """Test GraphNode creation."""
        from db.neo4j_client import GraphNode

        node = GraphNode(
            id="node-123",
            labels=["Verse", "OT"],
            properties={"reference": "GEN.1.1", "text": "In the beginning..."}
        )

        assert node.id == "node-123"
        assert "Verse" in node.labels
        assert node.properties["reference"] == "GEN.1.1"

    def test_graph_relationship_creation(self):
        """Test GraphRelationship creation."""
        from db.neo4j_client import GraphRelationship

        rel = GraphRelationship(
            id="rel-456",
            type="TYPOLOGICALLY_FULFILLS",
            start_node="node-123",
            end_node="node-789",
            properties={"confidence": 0.95}
        )

        assert rel.id == "rel-456"
        assert rel.type == "TYPOLOGICALLY_FULFILLS"
        assert rel.properties["confidence"] == 0.95


# =============================================================================
# Connection Lifecycle Tests
# =============================================================================

class TestNeo4jClientLifecycle:
    """Tests for connection lifecycle."""

    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test successful connection."""
        from db.neo4j_client import Neo4jClient

        with patch("db.neo4j_client.NEO4J_AVAILABLE", True):
            with patch("db.neo4j_client.AsyncGraphDatabase") as mock_db:
                mock_driver = Mock()
                mock_db.driver.return_value = mock_driver

                client = Neo4jClient()
                await client.connect()

                mock_db.driver.assert_called_once()
                assert client._driver is mock_driver

    @pytest.mark.asyncio
    async def test_connect_when_unavailable(self):
        """Test connect when Neo4j is not available."""
        from db.neo4j_client import Neo4jClient

        with patch("db.neo4j_client.NEO4J_AVAILABLE", False):
            client = Neo4jClient()
            await client.connect()

            # Should not raise, just log warning
            assert client._driver is None

    @pytest.mark.asyncio
    async def test_close_connection(self):
        """Test closing connection."""
        from db.neo4j_client import Neo4jClient

        mock_driver = AsyncMock()
        client = Neo4jClient()
        client._driver = mock_driver

        await client.close()

        mock_driver.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_no_driver(self):
        """Test close when no driver."""
        from db.neo4j_client import Neo4jClient

        client = Neo4jClient()
        # Should not raise
        await client.close()


# =============================================================================
# Connectivity Verification Tests
# =============================================================================

class TestConnectivityVerification:
    """Tests for connectivity verification."""

    @pytest.mark.asyncio
    async def test_verify_connectivity_success(self):
        """Test successful connectivity verification."""
        from db.neo4j_client import Neo4jClient

        mock_driver = AsyncMock()
        mock_driver.verify_connectivity = AsyncMock()

        client = Neo4jClient()
        client._driver = mock_driver

        result = await client.verify_connectivity()

        assert result is True
        mock_driver.verify_connectivity.assert_called_once()

    @pytest.mark.asyncio
    async def test_verify_connectivity_no_driver(self):
        """Test verification with no driver."""
        from db.neo4j_client import Neo4jClient

        client = Neo4jClient()
        result = await client.verify_connectivity()

        assert result is False

    @pytest.mark.asyncio
    async def test_verify_connectivity_service_unavailable(self):
        """Test verification when service unavailable."""
        from db.neo4j_client import Neo4jClient, ServiceUnavailable

        mock_driver = AsyncMock()
        mock_driver.verify_connectivity = AsyncMock(
            side_effect=ServiceUnavailable("Service down")
        )

        client = Neo4jClient()
        client._driver = mock_driver

        result = await client.verify_connectivity()

        assert result is False

    @pytest.mark.asyncio
    async def test_verify_connectivity_auth_error(self):
        """Test verification with authentication error."""
        from db.neo4j_client import Neo4jClient, AuthError

        mock_driver = AsyncMock()
        mock_driver.verify_connectivity = AsyncMock(
            side_effect=AuthError("Invalid credentials")
        )

        client = Neo4jClient()
        client._driver = mock_driver

        result = await client.verify_connectivity()

        assert result is False


# =============================================================================
# Verse Node Operations Tests
# =============================================================================

class TestVerseNodeOperations:
    """Tests for verse node operations."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Neo4jClient."""
        from db.neo4j_client import Neo4jClient

        client = Neo4jClient()
        mock_driver = AsyncMock()
        client._driver = mock_driver
        return client, mock_driver

    @pytest.mark.asyncio
    async def test_create_verse_node(self, mock_client):
        """Test creating a verse node."""
        client, mock_driver = mock_client

        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_record = {"id": "node-123"}
        mock_result.single = AsyncMock(return_value=mock_record)
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_driver.session.return_value.__aenter__.return_value = mock_session

        result = await client.create_verse_node(
            "GEN.1.1",
            {"text": "In the beginning..."}
        )

        assert result == "node-123"
        mock_session.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_verse_node_no_driver(self):
        """Test create verse node without driver."""
        from db.neo4j_client import Neo4jClient

        client = Neo4jClient()
        result = await client.create_verse_node("GEN.1.1", {})

        assert result is None

    @pytest.mark.asyncio
    async def test_get_verse_node_found(self, mock_client):
        """Test getting an existing verse node."""
        client, mock_driver = mock_client

        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_record = {
            "id": "node-123",
            "labels": ["Verse"],
            "v": {"reference": "GEN.1.1", "text": "In the beginning..."}
        }
        mock_result.single = AsyncMock(return_value=mock_record)
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_driver.session.return_value.__aenter__.return_value = mock_session

        result = await client.get_verse_node("GEN.1.1")

        assert result is not None
        assert result.id == "node-123"
        assert "Verse" in result.labels

    @pytest.mark.asyncio
    async def test_get_verse_node_not_found(self, mock_client):
        """Test getting a non-existent verse node."""
        client, mock_driver = mock_client

        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value=None)
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_driver.session.return_value.__aenter__.return_value = mock_session

        result = await client.get_verse_node("NONEXISTENT.1.1")

        assert result is None


# =============================================================================
# Cross-Reference Operations Tests
# =============================================================================

class TestCrossReferenceOperations:
    """Tests for cross-reference operations."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Neo4jClient."""
        from db.neo4j_client import Neo4jClient

        client = Neo4jClient()
        mock_driver = AsyncMock()
        client._driver = mock_driver
        return client, mock_driver

    @pytest.mark.asyncio
    async def test_create_cross_reference(self, mock_client):
        """Test creating a cross-reference relationship."""
        client, mock_driver = mock_client

        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value={"id": "rel-456"})
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_driver.session.return_value.__aenter__.return_value = mock_session

        result = await client.create_cross_reference(
            "GEN.1.1",
            "JHN.1.1",
            "TYPOLOGICALLY_FULFILLS",
            {"confidence": 0.95}
        )

        assert result == "rel-456"

    @pytest.mark.asyncio
    async def test_create_cross_reference_unknown_type(self, mock_client):
        """Test creating cross-reference with unknown type logs warning."""
        client, mock_driver = mock_client

        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value={"id": "rel-456"})
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_driver.session.return_value.__aenter__.return_value = mock_session

        # Should log warning but still proceed
        result = await client.create_cross_reference(
            "GEN.1.1",
            "JHN.1.1",
            "UNKNOWN_TYPE"
        )

        assert result == "rel-456"

    @pytest.mark.asyncio
    async def test_get_cross_references_outgoing(self, mock_client):
        """Test getting outgoing cross-references."""
        client, mock_driver = mock_client

        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.data = AsyncMock(return_value=[
            {"source": "GEN.1.1", "target": "JHN.1.1", "rel_type": "TYPOLOGICALLY_FULFILLS", "props": {}}
        ])
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_driver.session.return_value.__aenter__.return_value = mock_session

        results = await client.get_cross_references("GEN.1.1", direction="outgoing")

        assert len(results) == 1
        assert results[0]["target"] == "JHN.1.1"

    @pytest.mark.asyncio
    async def test_get_cross_references_with_type_filter(self, mock_client):
        """Test getting cross-references with type filter."""
        client, mock_driver = mock_client

        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.data = AsyncMock(return_value=[])
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_driver.session.return_value.__aenter__.return_value = mock_session

        await client.get_cross_references(
            "GEN.1.1",
            rel_types=["TYPOLOGICALLY_FULFILLS", "QUOTES"]
        )

        # Verify the query was constructed with type filter
        call_args = mock_session.run.call_args
        assert "TYPOLOGICALLY_FULFILLS|QUOTES" in str(call_args)


# =============================================================================
# Church Father Operations Tests
# =============================================================================

class TestChurchFatherOperations:
    """Tests for Church Father operations."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Neo4jClient."""
        from db.neo4j_client import Neo4jClient

        client = Neo4jClient()
        mock_driver = AsyncMock()
        client._driver = mock_driver
        return client, mock_driver

    @pytest.mark.asyncio
    async def test_create_church_father(self, mock_client):
        """Test creating a Church Father node."""
        client, mock_driver = mock_client

        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value={"id": "father-123"})
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_driver.session.return_value.__aenter__.return_value = mock_session

        result = await client.create_church_father(
            "John Chrysostom",
            {"century": 4, "location": "Constantinople"}
        )

        assert result == "father-123"

    @pytest.mark.asyncio
    async def test_link_father_to_verse(self, mock_client):
        """Test linking a Church Father to a verse."""
        client, mock_driver = mock_client

        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value={"id": "link-789"})
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_driver.session.return_value.__aenter__.return_value = mock_session

        result = await client.link_father_to_verse(
            "John Chrysostom",
            "GEN.1.1",
            citation_type="CITED_BY",
            properties={"work": "Homilies on Genesis"}
        )

        assert result == "link-789"


# =============================================================================
# Thematic Category Operations Tests
# =============================================================================

class TestThematicCategoryOperations:
    """Tests for thematic category operations."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Neo4jClient."""
        from db.neo4j_client import Neo4jClient

        client = Neo4jClient()
        mock_driver = AsyncMock()
        client._driver = mock_driver
        return client, mock_driver

    @pytest.mark.asyncio
    async def test_create_thematic_category(self, mock_client):
        """Test creating a thematic category."""
        client, mock_driver = mock_client

        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value={"id": "theme-123"})
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_driver.session.return_value.__aenter__.return_value = mock_session

        result = await client.create_thematic_category(
            "Creation",
            "Passages about God's creative work"
        )

        assert result == "theme-123"

    @pytest.mark.asyncio
    async def test_create_thematic_category_with_parent(self, mock_client):
        """Test creating a thematic category with parent."""
        client, mock_driver = mock_client

        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value={"id": "theme-456"})
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_driver.session.return_value.__aenter__.return_value = mock_session

        result = await client.create_thematic_category(
            "Genesis Creation",
            "Creation account in Genesis",
            parent="Creation"
        )

        assert result == "theme-456"
        # Should have two run calls - one for create, one for linking parent
        assert mock_session.run.call_count == 2

    @pytest.mark.asyncio
    async def test_tag_verse_with_theme(self, mock_client):
        """Test tagging a verse with a theme."""
        client, mock_driver = mock_client

        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value={"id": "tag-123"})
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_driver.session.return_value.__aenter__.return_value = mock_session

        result = await client.tag_verse_with_theme("GEN.1.1", "Creation")

        assert result == "tag-123"


# =============================================================================
# Graph Analysis Tests
# =============================================================================

class TestGraphAnalysis:
    """Tests for graph analysis operations."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Neo4jClient."""
        from db.neo4j_client import Neo4jClient

        client = Neo4jClient()
        mock_driver = AsyncMock()
        client._driver = mock_driver
        return client, mock_driver

    @pytest.mark.asyncio
    async def test_find_shortest_path(self, mock_client):
        """Test finding shortest path between verses."""
        client, mock_driver = mock_client

        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value={
            "nodes": ["GEN.1.1", "ISA.40.1", "JHN.1.1"],
            "relationships": ["REFERENCES", "TYPOLOGICALLY_FULFILLS"]
        })
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_driver.session.return_value.__aenter__.return_value = mock_session

        result = await client.find_shortest_path("GEN.1.1", "JHN.1.1")

        assert len(result) == 1
        assert len(result[0]["nodes"]) == 3

    @pytest.mark.asyncio
    async def test_find_shortest_path_no_path(self, mock_client):
        """Test finding shortest path when no path exists."""
        client, mock_driver = mock_client

        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value=None)
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_driver.session.return_value.__aenter__.return_value = mock_session

        result = await client.find_shortest_path("GEN.1.1", "REV.22.21")

        assert result == []

    @pytest.mark.asyncio
    async def test_get_verse_neighborhood(self, mock_client):
        """Test getting verse neighborhood graph."""
        client, mock_driver = mock_client

        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value={
            "nodes": [{"reference": "GEN.1.2"}, {"reference": "JHN.1.1"}],
            "rels": []
        })
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_driver.session.return_value.__aenter__.return_value = mock_session

        result = await client.get_verse_neighborhood("GEN.1.1", depth=2)

        assert "nodes" in result
        assert len(result["nodes"]) == 2

    @pytest.mark.asyncio
    async def test_get_verse_neighborhood_no_driver(self):
        """Test get neighborhood without driver."""
        from db.neo4j_client import Neo4jClient

        client = Neo4jClient()
        result = await client.get_verse_neighborhood("GEN.1.1")

        assert result == {"nodes": [], "relationships": []}


# =============================================================================
# Statistics Tests
# =============================================================================

class TestStatistics:
    """Tests for statistics operations."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Neo4jClient."""
        from db.neo4j_client import Neo4jClient

        client = Neo4jClient()
        mock_driver = AsyncMock()
        client._driver = mock_driver
        return client, mock_driver

    @pytest.mark.asyncio
    async def test_get_graph_statistics(self, mock_client):
        """Test getting graph statistics."""
        client, mock_driver = mock_client

        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value={
            "verses": 31102,
            "fathers": 50,
            "themes": 31,
            "relationships": 100000
        })
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_driver.session.return_value.__aenter__.return_value = mock_session

        stats = await client.get_graph_statistics()

        assert stats["verses"] == 31102
        assert stats["church_fathers"] == 50
        assert stats["thematic_categories"] == 31
        assert stats["relationships"] == 100000

    @pytest.mark.asyncio
    async def test_get_graph_statistics_no_driver(self):
        """Test get statistics without driver."""
        from db.neo4j_client import Neo4jClient

        client = Neo4jClient()
        stats = await client.get_graph_statistics()

        assert stats == {}


# =============================================================================
# Relationship Types Tests
# =============================================================================

class TestRelationshipTypes:
    """Tests for relationship type constants."""

    def test_relationship_types_defined(self):
        """Test that relationship types are defined."""
        from db.neo4j_client import Neo4jClient

        types = Neo4jClient.RELATIONSHIP_TYPES

        assert "REFERENCES" in types
        assert "QUOTES" in types
        assert "ALLUDES_TO" in types
        assert "TYPOLOGICALLY_FULFILLS" in types
        assert "PROPHETICALLY_FULFILLS" in types
        assert "THEMATICALLY_CONNECTED" in types
        assert "LITURGICALLY_USED" in types
        assert "VERBAL_PARALLEL" in types
        assert "NARRATIVE_PARALLEL" in types
        assert "CITED_BY" in types
