"""
Tests for pipeline orchestrator.
"""
import pytest


class TestPipelineConfig:
    """Tests for PipelineConfig."""

    def test_default_config(self):
        """Test default pipeline configuration."""
        from pipeline.orchestrator import PipelineConfig

        config = PipelineConfig()
        assert "linguistic" in config.phases
        assert "theological" in config.phases
        assert "intertextual" in config.phases
        assert "validation" in config.phases
        assert "finalization" in config.phases
        assert config.fail_fast is False
        assert config.timeout_seconds == 600

    def test_custom_config(self):
        """Test custom pipeline configuration."""
        from pipeline.orchestrator import PipelineConfig

        config = PipelineConfig(
            phases=["linguistic", "theological"],
            fail_fast=True,
            timeout_seconds=300
        )
        assert len(config.phases) == 2
        assert config.fail_fast is True
        assert config.timeout_seconds == 300


class TestPipelineResult:
    """Tests for PipelineResult."""

    def test_result_creation(self):
        """Test pipeline result creation."""
        from pipeline.orchestrator import PipelineResult

        result = PipelineResult(
            verse_id="GEN.1.1",
            status="completed",
            phase_results={},
            golden_record=None,
            start_time=0.0,
            end_time=1.0,
            errors=[]
        )

        assert result.verse_id == "GEN.1.1"
        assert result.status == "completed"
        assert result.duration == 1.0

    def test_result_to_dict(self):
        """Test result serialization."""
        from pipeline.orchestrator import PipelineResult

        result = PipelineResult(
            verse_id="GEN.1.1",
            status="completed",
            phase_results={},
            golden_record={"data": "test"},
            start_time=0.0,
            end_time=1.0,
            errors=[]
        )

        d = result.to_dict()
        assert d["verse_id"] == "GEN.1.1"
        assert d["status"] == "completed"
        assert d["duration"] == 1.0
        assert d["golden_record"] == {"data": "test"}


class TestPipelineOrchestrator:
    """Tests for PipelineOrchestrator."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test orchestrator initialization."""
        from pipeline.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()
        await orchestrator.initialize()

        assert orchestrator._initialized is True
        assert len(orchestrator._phases) == 5

        await orchestrator.cleanup()

    @pytest.mark.asyncio
    async def test_execute_single_verse(self, sample_verse_id, sample_verse_text):
        """Test single verse execution."""
        from pipeline.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()
        await orchestrator.initialize()

        result = await orchestrator.execute(sample_verse_id, sample_verse_text)

        assert result.verse_id == sample_verse_id
        assert result.status in ["completed", "partial", "error"]
        assert result.duration > 0

        await orchestrator.cleanup()

    @pytest.mark.asyncio
    async def test_execute_with_metadata(self, sample_verse_id, sample_verse_text):
        """Test execution with metadata."""
        from pipeline.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()
        await orchestrator.initialize()

        metadata = {"book": "GEN", "chapter": 1, "verse": 1}
        result = await orchestrator.execute(sample_verse_id, sample_verse_text, metadata)

        assert result.verse_id == sample_verse_id

        await orchestrator.cleanup()

    @pytest.mark.asyncio
    async def test_execute_batch(self, sample_verse_id, sample_verse_text):
        """Test batch execution."""
        from pipeline.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()
        await orchestrator.initialize()

        verses = [
            {"verse_id": "GEN.1.1", "text": "In the beginning"},
            {"verse_id": "GEN.1.2", "text": "And the earth was without form"}
        ]

        results = await orchestrator.execute_batch(verses, parallel=2)

        assert len(results) == 2
        assert results[0].verse_id == "GEN.1.1"
        assert results[1].verse_id == "GEN.1.2"

        await orchestrator.cleanup()

    @pytest.mark.asyncio
    async def test_get_phase_status(self):
        """Test phase status retrieval."""
        from pipeline.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()
        await orchestrator.initialize()

        status = orchestrator.get_phase_status()

        assert "linguistic" in status
        assert "theological" in status
        assert status["linguistic"] == "initialized"

        await orchestrator.cleanup()

    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test orchestrator cleanup."""
        from pipeline.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()
        await orchestrator.initialize()
        assert orchestrator._initialized is True

        await orchestrator.cleanup()
        assert orchestrator._initialized is False
        assert len(orchestrator._phases) == 0
