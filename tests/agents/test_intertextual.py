"""
Tests for intertextual extraction agents.
"""
import pytest


class TestSyndesmos:
    """Tests for SYNDESMOS agent (Cross-Reference Connections)."""

    @pytest.mark.asyncio
    async def test_extract_basic(self, sample_verse_id, sample_verse_text, sample_theological_context):
        """Test basic extraction."""
        from agents.intertextual.syndesmos import SyndesmosAgent

        agent = SyndesmosAgent()
        result = await agent.extract(sample_verse_id, sample_verse_text, sample_theological_context)

        assert result.agent_name == "syndesmos"
        assert result.verse_id == sample_verse_id
        assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_cross_reference_detection(self, sample_verse_id, sample_verse_text, sample_theological_context):
        """Test cross-reference detection."""
        from agents.intertextual.syndesmos import SyndesmosAgent

        agent = SyndesmosAgent()
        result = await agent.extract(sample_verse_id, sample_verse_text, sample_theological_context)

        data = result.data
        assert isinstance(data, dict)

    def test_connection_types(self):
        """Test supported connection types."""
        from agents.intertextual.syndesmos import SyndesmosAgent, ConnectionType

        # Should have all required connection types
        types = [e.value for e in ConnectionType]
        assert "thematic" in types
        assert "typological" in types
        assert "verbal" in types


class TestHarmonikos:
    """Tests for HARMONIKOS agent (Parallel Passage Harmonization)."""

    @pytest.mark.asyncio
    async def test_extract_basic(self, sample_verse_id, sample_verse_text, sample_theological_context):
        """Test basic extraction."""
        from agents.intertextual.harmonikos import HarmonikosAgent

        agent = HarmonikosAgent()
        result = await agent.extract(sample_verse_id, sample_verse_text, sample_theological_context)

        assert result.agent_name == "harmonikos"
        assert result.verse_id == sample_verse_id

    @pytest.mark.asyncio
    async def test_synoptic_detection(self):
        """Test synoptic parallel detection."""
        from agents.intertextual.harmonikos import HarmonikosAgent

        agent = HarmonikosAgent()
        # Matthew verse should find Mark/Luke parallels
        result = await agent.extract("MAT.3.1", "In those days came John the Baptist", {
            "agent_results": {}
        })

        data = result.data
        assert isinstance(data, dict)


class TestAllographos:
    """Tests for ALLOGRAPHOS agent (Quotation/Allusion Detection)."""

    @pytest.mark.asyncio
    async def test_extract_basic(self, sample_verse_id, sample_verse_text, sample_theological_context):
        """Test basic extraction."""
        from agents.intertextual.allographos import AllographosAgent

        agent = AllographosAgent()
        result = await agent.extract(sample_verse_id, sample_verse_text, sample_theological_context)

        assert result.agent_name == "allographos"
        assert result.verse_id == sample_verse_id

    @pytest.mark.asyncio
    async def test_quotation_markers(self):
        """Test quotation marker detection."""
        from agents.intertextual.allographos import AllographosAgent

        agent = AllographosAgent()
        # Text with quotation marker
        text = "As it is written: 'The voice of one crying in the wilderness'"
        result = await agent.extract("MAT.3.3", text, {"agent_results": {}})

        data = result.data
        assert isinstance(data, dict)


class TestParadeigma:
    """Tests for PARADEIGMA agent (Example/Precedent Identification)."""

    @pytest.mark.asyncio
    async def test_extract_basic(self, sample_verse_id, sample_verse_text, sample_theological_context):
        """Test basic extraction."""
        from agents.intertextual.paradeigma import ParadeigmaAgent

        agent = ParadeigmaAgent()
        result = await agent.extract(sample_verse_id, sample_verse_text, sample_theological_context)

        assert result.agent_name == "paradeigma"
        assert result.verse_id == sample_verse_id

    @pytest.mark.asyncio
    async def test_figure_detection(self):
        """Test biblical figure detection."""
        from agents.intertextual.paradeigma import ParadeigmaAgent

        agent = ParadeigmaAgent()
        # Text mentioning Abraham
        text = "Abraham believed God and it was credited to him as righteousness"
        result = await agent.extract("ROM.4.3", text, {"agent_results": {}})

        data = result.data
        assert isinstance(data, dict)


class TestTopos:
    """Tests for TOPOS agent (Common Topic/Motif Analysis)."""

    @pytest.mark.asyncio
    async def test_extract_basic(self, sample_verse_id, sample_verse_text, sample_theological_context):
        """Test basic extraction."""
        from agents.intertextual.topos import ToposAgent

        agent = ToposAgent()
        result = await agent.extract(sample_verse_id, sample_verse_text, sample_theological_context)

        assert result.agent_name == "topos"
        assert result.verse_id == sample_verse_id

    @pytest.mark.asyncio
    async def test_motif_detection(self):
        """Test motif pattern detection."""
        from agents.intertextual.topos import ToposAgent

        agent = ToposAgent()
        # Text with covenant motif
        text = "I will establish my covenant between me and you"
        result = await agent.extract("GEN.17.7", text, {"agent_results": {}})

        data = result.data
        assert isinstance(data, dict)

    def test_motif_categories(self):
        """Test supported motif categories."""
        from agents.intertextual.topos import ToposAgent, MotifCategory

        # Test that MotifCategory enum has expected categories
        categories = [c.value for c in MotifCategory]
        assert "divine_action" in categories
        assert "cosmic" in categories
        assert "cultic" in categories

        # Test that ToposAgent has motif patterns for key biblical motifs
        agent = ToposAgent()
        patterns = list(agent.BIBLICAL_MOTIFS.keys())
        assert "exodus" in patterns
        assert "covenant" in patterns
