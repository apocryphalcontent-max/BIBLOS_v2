"""
Tests for the Stream Orchestrator.

Run with: pytest tests/pipeline/test_stream_orchestrator.py -v
"""
import asyncio
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch

from pipeline.stream_orchestrator import (
    StreamOrchestrator,
    StreamOrchestratorConfig,
    VerseState,
    StreamPipelineResult,
)
from pipeline.event_bus import (
    EventBus,
    EventMessage,
    StreamTopic,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def orchestrator_config():
    """Create test orchestrator configuration."""
    return StreamOrchestratorConfig(
        phases=["linguistic", "theological"],
        phase_dependencies={
            "linguistic": [],
            "theological": ["linguistic"],
        },
        consumer_group="test-orchestrator",
        phase_timeout_seconds=30,
        verse_timeout_seconds=60,
    )


@pytest.fixture
def mock_event_bus():
    """Create mock event bus."""
    mock = AsyncMock(spec=EventBus)
    mock.publish = AsyncMock(return_value="msg-123")
    mock.subscribe = AsyncMock()
    mock.acknowledge = AsyncMock(return_value=1)
    mock.ensure_consumer_group = AsyncMock()
    mock.health_check = AsyncMock(return_value={"healthy": True})
    mock._redis = AsyncMock()
    mock._redis.set = AsyncMock()
    mock._redis.get = AsyncMock(return_value=None)
    mock._redis.delete = AsyncMock(return_value=1)
    mock._redis.scan = AsyncMock(return_value=(0, []))
    return mock


@pytest.fixture
async def orchestrator(orchestrator_config, mock_event_bus):
    """Create orchestrator with mock event bus."""
    orch = StreamOrchestrator(config=orchestrator_config, event_bus=mock_event_bus)
    orch._event_bus = mock_event_bus
    orch._initialized = True
    yield orch
    if orch._running:
        await orch.stop()


# =============================================================================
# VERSE STATE TESTS
# =============================================================================

class TestVerseState:
    """Tests for VerseState dataclass."""

    def test_create_verse_state(self):
        """Test creating a verse state."""
        state = VerseState(
            verse_id="GEN.1.1",
            text="In the beginning...",
            correlation_id="corr-123",
        )

        assert state.verse_id == "GEN.1.1"
        assert state.status == "pending"
        assert state.completed_phases == []
        assert state.current_phase is None

    def test_to_dict(self):
        """Test converting state to dictionary."""
        state = VerseState(
            verse_id="GEN.1.1",
            text="In the beginning...",
            correlation_id="corr-123",
            status="processing",
            completed_phases=["linguistic"],
        )

        d = state.to_dict()

        assert d["verse_id"] == "GEN.1.1"
        assert d["status"] == "processing"
        assert d["completed_phases"] == ["linguistic"]

    def test_from_dict(self):
        """Test creating state from dictionary."""
        data = {
            "verse_id": "GEN.1.1",
            "text": "In the beginning...",
            "correlation_id": "corr-123",
            "status": "completed",
            "completed_phases": ["linguistic", "theological"],
            "phase_results": {"linguistic": {"confidence": 0.9}},
            "started_at": 1000.0,
            "updated_at": 1001.0,
            "completed_at": 1002.0,
            "errors": [],
            "retry_count": 0,
            "metadata": {},
        }

        state = VerseState.from_dict(data)

        assert state.verse_id == "GEN.1.1"
        assert state.status == "completed"
        assert len(state.completed_phases) == 2

    def test_is_phase_complete(self):
        """Test checking phase completion."""
        state = VerseState(
            verse_id="GEN.1.1",
            text="...",
            correlation_id="123",
            completed_phases=["linguistic"],
        )

        assert state.is_phase_complete("linguistic") is True
        assert state.is_phase_complete("theological") is False

    def test_are_dependencies_met(self):
        """Test checking phase dependencies."""
        state = VerseState(
            verse_id="GEN.1.1",
            text="...",
            correlation_id="123",
            completed_phases=["linguistic"],
        )

        dependencies = {
            "linguistic": [],
            "theological": ["linguistic"],
            "intertextual": ["linguistic", "theological"],
        }

        assert state.are_dependencies_met("linguistic", dependencies) is True
        assert state.are_dependencies_met("theological", dependencies) is True
        assert state.are_dependencies_met("intertextual", dependencies) is False

    def test_get_next_phase(self):
        """Test getting next phase to execute."""
        state = VerseState(
            verse_id="GEN.1.1",
            text="...",
            correlation_id="123",
            completed_phases=["linguistic"],
        )

        phases = ["linguistic", "theological", "intertextual"]
        dependencies = {
            "linguistic": [],
            "theological": ["linguistic"],
            "intertextual": ["linguistic", "theological"],
        }

        next_phase = state.get_next_phase(phases, dependencies)

        assert next_phase == "theological"

    def test_get_next_phase_all_complete(self):
        """Test getting next phase when all complete."""
        state = VerseState(
            verse_id="GEN.1.1",
            text="...",
            correlation_id="123",
            completed_phases=["linguistic", "theological"],
        )

        phases = ["linguistic", "theological"]
        dependencies = {"linguistic": [], "theological": ["linguistic"]}

        next_phase = state.get_next_phase(phases, dependencies)

        assert next_phase is None

    def test_to_context(self):
        """Test converting state to context dictionary."""
        state = VerseState(
            verse_id="GEN.1.1",
            text="In the beginning...",
            correlation_id="123",
            completed_phases=["linguistic"],
            phase_results={
                "linguistic": {
                    "agent_results": {
                        "grammateus": {"data": "test", "confidence": 0.9}
                    }
                }
            },
        )

        context = state.to_context()

        assert context["verse_id"] == "GEN.1.1"
        assert context["text"] == "In the beginning..."
        assert context["completed_phases"] == ["linguistic"]
        assert "grammateus" in context["agent_results"]


# =============================================================================
# ORCHESTRATOR INITIALIZATION TESTS
# =============================================================================

class TestOrchestratorInit:
    """Tests for orchestrator initialization."""

    @pytest.mark.asyncio
    async def test_initialize(self, orchestrator_config, mock_event_bus):
        """Test orchestrator initialization."""
        orch = StreamOrchestrator(config=orchestrator_config, event_bus=mock_event_bus)

        await orch.initialize()

        assert orch._initialized is True
        mock_event_bus.ensure_consumer_group.assert_called()

    @pytest.mark.asyncio
    async def test_double_initialize(self, orchestrator):
        """Test initializing twice is safe."""
        await orchestrator.initialize()
        await orchestrator.initialize()  # Should be no-op

        assert orchestrator._initialized is True


# =============================================================================
# VERSE INGESTION TESTS
# =============================================================================

class TestVerseIngestion:
    """Tests for verse ingestion."""

    @pytest.mark.asyncio
    async def test_ingest_verse(self, orchestrator, mock_event_bus):
        """Test ingesting a verse."""
        correlation_id = await orchestrator.ingest_verse(
            verse_id="GEN.1.1",
            text="In the beginning God created...",
            metadata={"source": "test"},
        )

        assert correlation_id is not None
        mock_event_bus.publish.assert_called_once()

        # Check published message
        call_args = mock_event_bus.publish.call_args
        assert call_args[0][0] == StreamTopic.VERSE_INGESTED

    @pytest.mark.asyncio
    async def test_ingest_verse_normalizes_id(self, orchestrator, mock_event_bus):
        """Test verse ID normalization."""
        await orchestrator.ingest_verse(
            verse_id="gen 1:1",  # Mixed case with space and colon
            text="...",
        )

        call_args = mock_event_bus.publish.call_args
        event = call_args[0][1]
        assert event.payload["verse_id"] == "GEN.1.1"

    @pytest.mark.asyncio
    async def test_ingest_batch(self, orchestrator, mock_event_bus):
        """Test ingesting multiple verses."""
        verses = [
            {"verse_id": "GEN.1.1", "text": "In the beginning..."},
            {"verse_id": "GEN.1.2", "text": "And the earth was..."},
        ]

        correlation_ids = await orchestrator.ingest_batch(verses)

        assert len(correlation_ids) == 2
        assert mock_event_bus.publish.call_count == 2


# =============================================================================
# STATE MANAGEMENT TESTS
# =============================================================================

class TestStateManagement:
    """Tests for verse state management."""

    @pytest.mark.asyncio
    async def test_save_verse_state(self, orchestrator, mock_event_bus):
        """Test saving verse state to Redis."""
        state = VerseState(
            verse_id="GEN.1.1",
            text="...",
            correlation_id="123",
        )

        await orchestrator._save_verse_state(state)

        mock_event_bus._redis.set.assert_called_once()
        call_args = mock_event_bus._redis.set.call_args
        assert "biblos:verse_state:123" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_load_verse_state(self, orchestrator, mock_event_bus):
        """Test loading verse state from Redis."""
        state_data = {
            "verse_id": "GEN.1.1",
            "text": "...",
            "correlation_id": "123",
            "status": "processing",
            "completed_phases": ["linguistic"],
            "current_phase": "theological",
            "phase_results": {},
            "started_at": 1000.0,
            "updated_at": 1001.0,
            "completed_at": None,
            "errors": [],
            "retry_count": 0,
            "metadata": {},
        }
        mock_event_bus._redis.get.return_value = json.dumps(state_data)

        state = await orchestrator._load_verse_state("123")

        assert state is not None
        assert state.verse_id == "GEN.1.1"
        assert state.status == "processing"

    @pytest.mark.asyncio
    async def test_cleanup_verse_state(self, orchestrator, mock_event_bus):
        """Test cleaning up verse state."""
        orchestrator._verse_states["123"] = VerseState(
            verse_id="GEN.1.1",
            text="...",
            correlation_id="123",
        )

        await orchestrator._cleanup_verse_state("123")

        assert "123" not in orchestrator._verse_states
        mock_event_bus._redis.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_verse_status(self, orchestrator, mock_event_bus):
        """Test getting verse status."""
        orchestrator._verse_states["123"] = VerseState(
            verse_id="GEN.1.1",
            text="...",
            correlation_id="123",
            status="processing",
            completed_phases=["linguistic"],
        )

        status = await orchestrator.get_verse_status("123")

        assert status is not None
        assert status["verse_id"] == "GEN.1.1"
        assert status["status"] == "processing"


# =============================================================================
# GOLDEN RECORD TESTS
# =============================================================================

class TestGoldenRecord:
    """Tests for golden record building."""

    def test_build_golden_record(self, orchestrator):
        """Test building a golden record from state."""
        state = VerseState(
            verse_id="GEN.1.1",
            text="In the beginning...",
            correlation_id="123",
            status="completed",
            completed_phases=["linguistic", "theological"],
            phase_results={
                "linguistic": {
                    "agent_results": {
                        "grammateus": {"data": "test", "confidence": 0.95},
                    },
                    "metrics": {"phase_confidence": 0.95},
                },
                "theological": {
                    "agent_results": {
                        "patrologos": {"data": "test", "confidence": 0.9},
                    },
                    "metrics": {"phase_confidence": 0.9},
                },
            },
            started_at=1000.0,
            completed_at=1005.0,
        )

        golden = orchestrator._build_golden_record(state)

        assert golden["verse_id"] == "GEN.1.1"
        assert golden["text"] == "In the beginning..."
        assert golden["certification"]["level"] == "gold"  # avg 0.925 >= 0.9
        assert "grammateus" in golden["data"]["structural"]
        assert "patrologos" in golden["data"]["theological"]

    def test_build_golden_record_lower_confidence(self, orchestrator):
        """Test golden record with lower confidence."""
        state = VerseState(
            verse_id="GEN.1.1",
            text="...",
            correlation_id="123",
            completed_phases=["linguistic"],
            phase_results={
                "linguistic": {
                    "agent_results": {},
                    "metrics": {"phase_confidence": 0.6},
                },
            },
            started_at=1000.0,
            completed_at=1001.0,
        )

        golden = orchestrator._build_golden_record(state)

        assert golden["certification"]["level"] == "bronze"  # 0.6 < 0.75


# =============================================================================
# METRICS AND HEALTH TESTS
# =============================================================================

class TestMetricsAndHealth:
    """Tests for metrics and health checks."""

    def test_get_metrics(self, orchestrator):
        """Test getting orchestrator metrics."""
        orchestrator._metrics["verses_received"] = 10
        orchestrator._metrics["verses_completed"] = 8

        metrics = orchestrator.get_metrics()

        assert metrics["verses_received"] == 10
        assert metrics["verses_completed"] == 8
        assert "consumer_id" in metrics

    @pytest.mark.asyncio
    async def test_health_check(self, orchestrator, mock_event_bus):
        """Test health check."""
        orchestrator._running = True

        health = await orchestrator.health_check()

        assert health["healthy"] is True
        assert health["running"] is True
        assert "metrics" in health

    @pytest.mark.asyncio
    async def test_health_check_not_running(self, orchestrator, mock_event_bus):
        """Test health check when not running."""
        orchestrator._running = False

        health = await orchestrator.health_check()

        assert health["healthy"] is False

    @pytest.mark.asyncio
    async def test_list_processing_verses(self, orchestrator):
        """Test listing processing verses."""
        orchestrator._verse_states["123"] = VerseState(
            verse_id="GEN.1.1",
            text="...",
            correlation_id="123",
            status="processing",
        )
        orchestrator._verse_states["456"] = VerseState(
            verse_id="GEN.1.2",
            text="...",
            correlation_id="456",
            status="completed",
        )

        processing = await orchestrator.list_processing_verses()

        assert len(processing) == 1
        assert processing[0]["verse_id"] == "GEN.1.1"


# =============================================================================
# STREAM PIPELINE RESULT TESTS
# =============================================================================

class TestStreamPipelineResult:
    """Tests for StreamPipelineResult."""

    def test_from_verse_state(self):
        """Test creating result from verse state."""
        state = VerseState(
            verse_id="GEN.1.1",
            text="...",
            correlation_id="123",
            status="completed",
            completed_phases=["linguistic", "theological"],
            started_at=1000.0,
            completed_at=1005.0,
            errors=["test error"],
        )

        result = StreamPipelineResult.from_verse_state(state)

        assert result.verse_id == "GEN.1.1"
        assert result.status == "completed"
        assert result.processing_time == 5.0
        assert len(result.phases_executed) == 2
        assert len(result.errors) == 1
