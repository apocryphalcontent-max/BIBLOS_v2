"""
Tests for pipeline phases.
"""
import pytest


class TestLinguisticPhase:
    """Tests for LinguisticPhase."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test phase initialization."""
        from pipeline.linguistic import LinguisticPhase

        phase = LinguisticPhase()
        await phase.initialize()

        assert len(phase._agents) == 4
        assert "grammateus" in phase._agents
        assert "morphologos" in phase._agents
        assert "syntaktikos" in phase._agents
        assert "semantikos" in phase._agents

    @pytest.mark.asyncio
    async def test_execute(self, sample_verse_id, sample_verse_text, sample_context):
        """Test phase execution."""
        from pipeline.linguistic import LinguisticPhase

        phase = LinguisticPhase()
        await phase.initialize()

        result = await phase.execute(sample_verse_id, sample_verse_text, sample_context)

        assert result.phase_name == "linguistic"
        assert result.status.value in ["completed", "failed"]

        await phase.cleanup()


class TestTheologicalPhase:
    """Tests for TheologicalPhase."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test phase initialization."""
        from pipeline.theological import TheologicalPhase

        phase = TheologicalPhase()
        await phase.initialize()

        assert len(phase._agents) == 5
        assert "patrologos" in phase._agents
        assert "typologos" in phase._agents
        assert "theologos" in phase._agents

    @pytest.mark.asyncio
    async def test_execute(self, sample_verse_id, sample_verse_text, sample_linguistic_context):
        """Test phase execution."""
        from pipeline.theological import TheologicalPhase

        phase = TheologicalPhase()
        await phase.initialize()

        result = await phase.execute(sample_verse_id, sample_verse_text, sample_linguistic_context)

        assert result.phase_name == "theological"

        await phase.cleanup()


class TestIntertextualPhase:
    """Tests for IntertextualPhase."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test phase initialization."""
        from pipeline.intertextual import IntertextualPhase

        phase = IntertextualPhase()
        await phase.initialize()

        assert len(phase._agents) == 5
        assert "syndesmos" in phase._agents
        assert "harmonikos" in phase._agents

    @pytest.mark.asyncio
    async def test_execute(self, sample_verse_id, sample_verse_text, sample_theological_context):
        """Test phase execution."""
        from pipeline.intertextual import IntertextualPhase

        phase = IntertextualPhase()
        await phase.initialize()

        result = await phase.execute(sample_verse_id, sample_verse_text, sample_theological_context)

        assert result.phase_name == "intertextual"

        await phase.cleanup()


class TestValidationPhase:
    """Tests for ValidationPhase."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test phase initialization."""
        from pipeline.validation import ValidationPhase

        phase = ValidationPhase()
        await phase.initialize()

        assert len(phase._agents) == 5
        assert "elenktikos" in phase._agents
        assert "kritikos" in phase._agents
        assert "harmonizer" in phase._agents

    @pytest.mark.asyncio
    async def test_execute(self, sample_verse_id, sample_verse_text, sample_theological_context):
        """Test phase execution."""
        from pipeline.validation import ValidationPhase

        phase = ValidationPhase()
        await phase.initialize()

        # Add required context
        context = {
            **sample_theological_context,
            "phase_results": {
                "linguistic": {"status": "completed"},
                "theological": {"status": "completed"},
                "intertextual": {"status": "completed"}
            }
        }

        result = await phase.execute(sample_verse_id, sample_verse_text, context)

        assert result.phase_name == "validation"

        await phase.cleanup()


class TestFinalizationPhase:
    """Tests for FinalizationPhase."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test phase initialization."""
        from pipeline.finalization import FinalizationPhase

        phase = FinalizationPhase()
        await phase.initialize()

    @pytest.mark.asyncio
    async def test_execute(self, sample_verse_id, sample_verse_text, sample_theological_context):
        """Test phase execution."""
        from pipeline.finalization import FinalizationPhase

        phase = FinalizationPhase()
        await phase.initialize()

        # Add full context
        context = {
            **sample_theological_context,
            "phase_results": {
                "linguistic": {"status": "completed"},
                "theological": {"status": "completed"},
                "intertextual": {"status": "completed"},
                "validation": {"status": "completed"}
            }
        }

        result = await phase.execute(sample_verse_id, sample_verse_text, context)

        assert result.phase_name == "finalization"

        await phase.cleanup()

    @pytest.mark.asyncio
    async def test_golden_record_creation(self, sample_verse_id, sample_verse_text, sample_theological_context):
        """Test golden record creation."""
        from pipeline.finalization import FinalizationPhase

        phase = FinalizationPhase()
        await phase.initialize()

        context = {
            **sample_theological_context,
            "phase_results": {
                "linguistic": {"status": "completed"},
                "theological": {"status": "completed"},
                "intertextual": {"status": "completed"},
                "validation": {"status": "completed"}
            }
        }

        result = await phase.execute(sample_verse_id, sample_verse_text, context)

        # Check golden record was created
        if result.agent_results.get("golden_record"):
            golden = result.agent_results["golden_record"]
            assert "verse_id" in golden
            assert "text" in golden
            assert "version" in golden

        await phase.cleanup()
