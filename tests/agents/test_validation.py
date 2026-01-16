"""
Tests for validation agents.
"""
import pytest


class TestElenktikos:
    """Tests for ELENKTIKOS agent (Cross-Agent Consistency)."""

    @pytest.mark.asyncio
    async def test_extract_basic(self, sample_verse_id, sample_verse_text, sample_theological_context):
        """Test basic extraction."""
        from agents.validation.elenktikos import ElenktikosAgent

        agent = ElenktikosAgent()
        result = await agent.extract(sample_verse_id, sample_verse_text, sample_theological_context)

        assert result.agent_name == "elenktikos"
        assert result.verse_id == sample_verse_id

    @pytest.mark.asyncio
    async def test_consistency_check(self, sample_theological_context):
        """Test cross-agent consistency checking."""
        from agents.validation.elenktikos import ElenktikosAgent

        agent = ElenktikosAgent()
        result = await agent.extract(
            "GEN.1.1",
            "In the beginning God created the heaven and the earth.",
            sample_theological_context
        )

        data = result.data
        assert isinstance(data, dict)
        # Should have validation info
        assert "validation_passed" in data or "consistency" in str(data)


class TestKritikos:
    """Tests for KRITIKOS agent (Quality Scoring)."""

    @pytest.mark.asyncio
    async def test_extract_basic(self, sample_verse_id, sample_verse_text, sample_theological_context):
        """Test basic extraction."""
        from agents.validation.kritikos import KritikosAgent

        agent = KritikosAgent()
        result = await agent.extract(sample_verse_id, sample_verse_text, sample_theological_context)

        assert result.agent_name == "kritikos"
        assert result.verse_id == sample_verse_id

    @pytest.mark.asyncio
    async def test_quality_dimensions(self, sample_theological_context):
        """Test quality dimension scoring."""
        from agents.validation.kritikos import KritikosAgent

        agent = KritikosAgent()
        result = await agent.extract(
            "GEN.1.1",
            "In the beginning God created the heaven and the earth.",
            sample_theological_context
        )

        data = result.data
        assert isinstance(data, dict)

    def test_quality_thresholds(self):
        """Test quality threshold definitions."""
        from agents.validation.kritikos import KritikosAgent, QualityDimension

        dims = [e.value for e in QualityDimension]
        assert "completeness" in dims
        assert "accuracy" in dims
        assert "consistency" in dims


class TestHarmonizer:
    """Tests for HARMONIZER agent (Result Harmonization)."""

    @pytest.mark.asyncio
    async def test_extract_basic(self, sample_verse_id, sample_verse_text, sample_theological_context):
        """Test basic extraction."""
        from agents.validation.harmonizer import HarmonizerAgent

        agent = HarmonizerAgent()
        result = await agent.extract(sample_verse_id, sample_verse_text, sample_theological_context)

        assert result.agent_name == "harmonizer"
        assert result.verse_id == sample_verse_id

    @pytest.mark.asyncio
    async def test_conflict_resolution(self, sample_theological_context):
        """Test conflict resolution between agents."""
        from agents.validation.harmonizer import HarmonizerAgent

        # Add conflicting results
        context = {
            **sample_theological_context,
            "agent_results": {
                "agent1": {"data": {"value": "A"}, "confidence": 0.8},
                "agent2": {"data": {"value": "B"}, "confidence": 0.7}
            }
        }

        agent = HarmonizerAgent()
        result = await agent.extract("GEN.1.1", "Test text", context)

        data = result.data
        assert isinstance(data, dict)


class TestProsecutor:
    """Tests for PROSECUTOR agent (Challenge Generation)."""

    @pytest.mark.asyncio
    async def test_extract_basic(self, sample_verse_id, sample_verse_text, sample_theological_context):
        """Test basic extraction."""
        from agents.validation.prosecutor import ProsecutorAgent

        agent = ProsecutorAgent()
        result = await agent.extract(sample_verse_id, sample_verse_text, sample_theological_context)

        assert result.agent_name == "prosecutor"
        assert result.verse_id == sample_verse_id

    @pytest.mark.asyncio
    async def test_challenge_generation(self, sample_theological_context):
        """Test challenge generation."""
        from agents.validation.prosecutor import ProsecutorAgent

        agent = ProsecutorAgent()
        result = await agent.extract(
            "GEN.1.1",
            "In the beginning God created the heaven and the earth.",
            sample_theological_context
        )

        data = result.data
        assert isinstance(data, dict)
        # May have challenges list


class TestWitness:
    """Tests for WITNESS agent (Defense Responses)."""

    @pytest.mark.asyncio
    async def test_extract_basic(self, sample_verse_id, sample_verse_text, sample_theological_context):
        """Test basic extraction."""
        from agents.validation.witness import WitnessAgent

        agent = WitnessAgent()
        result = await agent.extract(sample_verse_id, sample_verse_text, sample_theological_context)

        assert result.agent_name == "witness"
        assert result.verse_id == sample_verse_id

    @pytest.mark.asyncio
    async def test_defense_response(self, sample_theological_context):
        """Test defense response to challenges."""
        from agents.validation.witness import WitnessAgent

        # Add challenges to context
        context = {
            **sample_theological_context,
            "challenges": [
                {"type": "evidence", "claim": "Test claim", "severity": "medium"}
            ]
        }

        agent = WitnessAgent()
        result = await agent.extract("GEN.1.1", "Test text", context)

        data = result.data
        assert isinstance(data, dict)
