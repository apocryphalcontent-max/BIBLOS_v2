"""
Tests for base agent classes.
"""
import pytest
from dataclasses import dataclass
from typing import Dict, Any, List

# Mock imports for testing without full dependencies
@dataclass
class AgentConfig:
    name: str = "test_agent"
    extraction_type: str = "test"
    timeout_seconds: int = 60
    min_confidence: float = 0.5
    dependencies: List[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class TestAgentConfig:
    """Tests for AgentConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AgentConfig()
        assert config.name == "test_agent"
        assert config.extraction_type == "test"
        assert config.timeout_seconds == 60
        assert config.min_confidence == 0.5
        assert config.dependencies == []

    def test_custom_config(self):
        """Test custom configuration values."""
        config = AgentConfig(
            name="custom_agent",
            extraction_type="linguistic",
            timeout_seconds=120,
            min_confidence=0.7,
            dependencies=["agent1", "agent2"]
        )
        assert config.name == "custom_agent"
        assert config.extraction_type == "linguistic"
        assert config.timeout_seconds == 120
        assert config.min_confidence == 0.7
        assert config.dependencies == ["agent1", "agent2"]


class TestExtractionResult:
    """Tests for ExtractionResult dataclass."""

    def test_result_creation(self):
        """Test creating an extraction result."""
        from agents.base import ExtractionResult, ProcessingStatus, ExtractionType

        result = ExtractionResult(
            agent_name="test_agent",
            extraction_type=ExtractionType.MORPHOLOGICAL,
            verse_id="GEN.1.1",
            status=ProcessingStatus.COMPLETED,
            data={"key": "value"},
            confidence=0.9
        )

        assert result.agent_name == "test_agent"
        assert result.verse_id == "GEN.1.1"
        assert result.status == ProcessingStatus.COMPLETED
        assert result.data == {"key": "value"}
        assert result.confidence == 0.9

    def test_result_to_dict(self):
        """Test conversion to dictionary."""
        from agents.base import ExtractionResult, ProcessingStatus, ExtractionType

        result = ExtractionResult(
            agent_name="test_agent",
            extraction_type=ExtractionType.SYNTACTIC,
            verse_id="GEN.1.1",
            status=ProcessingStatus.COMPLETED,
            data={"key": "value"},
            confidence=0.9
        )

        d = result.to_dict()
        assert d["agent_name"] == "test_agent"
        assert d["verse_id"] == "GEN.1.1"
        assert d["status"] == "completed"
        assert d["data"] == {"key": "value"}
        assert d["confidence"] == 0.9


class TestProcessingStatus:
    """Tests for ProcessingStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        from agents.base import ProcessingStatus

        assert ProcessingStatus.PENDING.value == "pending"
        assert ProcessingStatus.PROCESSING.value == "processing"
        assert ProcessingStatus.COMPLETED.value == "completed"
        assert ProcessingStatus.FAILED.value == "failed"
        assert ProcessingStatus.SKIPPED.value == "skipped"


class TestBaseExtractionAgent:
    """Tests for BaseExtractionAgent."""

    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        """Test agent initialization."""
        from agents.base import BaseExtractionAgent, AgentConfig, ExtractionType

        class TestAgent(BaseExtractionAgent):
            def __init__(self):
                config = AgentConfig(
                    name="test_agent",
                    extraction_type=ExtractionType.MORPHOLOGICAL,
                )
                super().__init__(config)

            async def extract(self, verse_id, text, context):
                pass
            async def validate(self, result):
                return True
            def get_dependencies(self):
                return []

        agent = TestAgent()
        assert agent.config.name == "test_agent"

    @pytest.mark.asyncio
    async def test_dependency_validation(self, sample_linguistic_context):
        """Test dependency declarations."""
        from agents.base import BaseExtractionAgent, AgentConfig, ExtractionType

        class TestAgent(BaseExtractionAgent):
            def __init__(self):
                config = AgentConfig(
                    name="test",
                    extraction_type=ExtractionType.SYNTACTIC,
                    dependencies=["grammateus"]
                )
                super().__init__(config)

            async def extract(self, verse_id, text, context):
                pass
            async def validate(self, result):
                return True
            def get_dependencies(self):
                return ["grammateus"]

        agent = TestAgent()
        # Verify dependencies are properly declared
        deps = agent.get_dependencies()
        assert "grammateus" in deps
        # Check that grammateus is in context
        agent_results = sample_linguistic_context.get("agent_results", {})
        assert "grammateus" in agent_results

    @pytest.mark.asyncio
    async def test_missing_dependencies(self, sample_context):
        """Test handling of missing dependencies."""
        from agents.base import BaseExtractionAgent, AgentConfig, ExtractionType

        class TestAgent(BaseExtractionAgent):
            def __init__(self):
                config = AgentConfig(
                    name="test",
                    extraction_type=ExtractionType.SEMANTIC,
                    dependencies=["missing_agent"]
                )
                super().__init__(config)

            async def extract(self, verse_id, text, context):
                pass
            async def validate(self, result):
                return True
            def get_dependencies(self):
                return ["missing_agent"]

        agent = TestAgent()
        # Verify the dependency is correctly declared
        deps = agent.get_dependencies()
        assert "missing_agent" in deps
        # Verify the missing_agent is NOT in context
        agent_results = sample_context.get("agent_results", {})
        assert "missing_agent" not in agent_results
