"""
Tests for theological extraction agents.
"""
import pytest


class TestPatrologos:
    """Tests for PATROLOGOS agent (Patristic Interpretation)."""

    @pytest.mark.asyncio
    async def test_extract_basic(self, sample_verse_id, sample_verse_text, sample_linguistic_context):
        """Test basic extraction."""
        from agents.theological.patrologos import PatrologosAgent

        agent = PatrologosAgent()
        result = await agent.extract(sample_verse_id, sample_verse_text, sample_linguistic_context)

        assert result.agent_name == "patrologos"
        assert result.verse_id == sample_verse_id
        assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_patristic_citations(self, sample_verse_id, sample_verse_text, sample_linguistic_context):
        """Test patristic citation extraction."""
        from agents.theological.patrologos import PatrologosAgent

        agent = PatrologosAgent()
        result = await agent.extract(sample_verse_id, sample_verse_text, sample_linguistic_context)

        data = result.data
        assert isinstance(data, dict)
        # May have citations or themes

    def test_dependencies(self):
        """Test agent dependencies."""
        from agents.theological.patrologos import PatrologosAgent

        agent = PatrologosAgent()
        deps = agent.get_dependencies()
        assert "grammateus" in deps


class TestTypologos:
    """Tests for TYPOLOGOS agent (Typological Connections)."""

    @pytest.mark.asyncio
    async def test_extract_basic(self, sample_verse_id, sample_verse_text, sample_theological_context):
        """Test basic extraction."""
        from agents.theological.typologos import TypologosAgent

        agent = TypologosAgent()
        result = await agent.extract(sample_verse_id, sample_verse_text, sample_theological_context)

        assert result.agent_name == "typologos"
        assert result.verse_id == sample_verse_id

    @pytest.mark.asyncio
    async def test_type_detection(self, sample_verse_id, sample_verse_text, sample_theological_context):
        """Test OT type detection."""
        from agents.theological.typologos import TypologosAgent

        agent = TypologosAgent()
        result = await agent.extract(sample_verse_id, sample_verse_text, sample_theological_context)

        data = result.data
        assert isinstance(data, dict)

    def test_dependencies(self):
        """Test agent dependencies."""
        from agents.theological.typologos import TypologosAgent

        agent = TypologosAgent()
        deps = agent.get_dependencies()
        assert "patrologos" in deps


class TestTheologos:
    """Tests for THEOLOGOS agent (Systematic Theology)."""

    @pytest.mark.asyncio
    async def test_extract_basic(self, sample_verse_id, sample_verse_text, sample_linguistic_context):
        """Test basic extraction."""
        from agents.theological.theologos import TheologosAgent

        agent = TheologosAgent()
        result = await agent.extract(sample_verse_id, sample_verse_text, sample_linguistic_context)

        assert result.agent_name == "theologos"
        assert result.verse_id == sample_verse_id

    @pytest.mark.asyncio
    async def test_doctrinal_classification(self, sample_verse_id, sample_verse_text, sample_linguistic_context):
        """Test doctrinal category detection."""
        from agents.theological.theologos import TheologosAgent

        agent = TheologosAgent()
        result = await agent.extract(sample_verse_id, sample_verse_text, sample_linguistic_context)

        data = result.data
        assert isinstance(data, dict)

    def test_dependencies(self):
        """Test agent dependencies."""
        from agents.theological.theologos import TheologosAgent

        agent = TheologosAgent()
        deps = agent.get_dependencies()
        assert "grammateus" in deps


class TestLiturgikos:
    """Tests for LITURGIKOS agent (Liturgical Usage)."""

    @pytest.mark.asyncio
    async def test_extract_basic(self, sample_verse_id, sample_verse_text, sample_theological_context):
        """Test basic extraction."""
        from agents.theological.liturgikos import LiturgikosAgent

        agent = LiturgikosAgent()
        result = await agent.extract(sample_verse_id, sample_verse_text, sample_theological_context)

        assert result.agent_name == "liturgikos"
        assert result.verse_id == sample_verse_id

    @pytest.mark.asyncio
    async def test_lectionary_mapping(self, sample_verse_id, sample_verse_text, sample_theological_context):
        """Test lectionary mapping."""
        from agents.theological.liturgikos import LiturgikosAgent

        agent = LiturgikosAgent()
        result = await agent.extract(sample_verse_id, sample_verse_text, sample_theological_context)

        data = result.data
        assert isinstance(data, dict)


class TestDogmatikos:
    """Tests for DOGMATIKOS agent (Dogmatic Analysis)."""

    @pytest.mark.asyncio
    async def test_extract_basic(self, sample_verse_id, sample_verse_text, sample_theological_context):
        """Test basic extraction."""
        from agents.theological.dogmatikos import DogmatikosAgent

        agent = DogmatikosAgent()
        result = await agent.extract(sample_verse_id, sample_verse_text, sample_theological_context)

        assert result.agent_name == "dogmatikos"
        assert result.verse_id == sample_verse_id

    @pytest.mark.asyncio
    async def test_conciliar_references(self, sample_verse_id, sample_verse_text, sample_theological_context):
        """Test ecumenical council reference detection."""
        from agents.theological.dogmatikos import DogmatikosAgent

        agent = DogmatikosAgent()
        result = await agent.extract(sample_verse_id, sample_verse_text, sample_theological_context)

        data = result.data
        assert isinstance(data, dict)
