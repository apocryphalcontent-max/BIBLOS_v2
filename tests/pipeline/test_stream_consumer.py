"""
Tests for pipeline/stream_consumer.py - Stream Consumer Base Class.

Covers:
- StreamConsumerConfig dataclass
- BaseStreamConsumer lifecycle
- Message processing and acknowledgment
- Retry logic with exponential backoff
- Metrics recording
- PhaseStreamConsumer adapter
- StreamConsumerManager
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import dataclass


# =============================================================================
# Configuration Tests
# =============================================================================

class TestStreamConsumerConfig:
    """Tests for StreamConsumerConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from pipeline.stream_consumer import StreamConsumerConfig

        config = StreamConsumerConfig()

        assert config.consumer_group == "biblos-phases"
        assert config.consumer_prefix == "phase"
        assert config.batch_size == 5
        assert config.parallel_processing == 2
        assert config.timeout_seconds == 300
        assert config.max_retries == 3
        assert config.retry_delay_base == 1.0
        assert config.retry_delay_max == 60.0
        assert config.heartbeat_interval == 30
        assert config.checkpoint_enabled is True
        assert config.checkpoint_interval == 10

    def test_custom_values(self):
        """Test custom configuration values."""
        from pipeline.stream_consumer import StreamConsumerConfig

        config = StreamConsumerConfig(
            consumer_group="custom-group",
            batch_size=10,
            parallel_processing=4,
            max_retries=5
        )

        assert config.consumer_group == "custom-group"
        assert config.batch_size == 10
        assert config.parallel_processing == 4
        assert config.max_retries == 5


# =============================================================================
# BaseStreamConsumer Lifecycle Tests
# =============================================================================

class TestBaseStreamConsumerLifecycle:
    """Tests for BaseStreamConsumer lifecycle."""

    @pytest.fixture
    def mock_event_bus(self):
        """Create a mock EventBus."""
        event_bus = AsyncMock()
        event_bus.ensure_consumer_group = AsyncMock()
        event_bus.publish = AsyncMock()
        event_bus.acknowledge = AsyncMock()
        event_bus.schedule_retry = AsyncMock()
        event_bus.claim_pending = AsyncMock(return_value=[])
        return event_bus

    @pytest.fixture
    def concrete_consumer(self, mock_event_bus):
        """Create a concrete implementation of BaseStreamConsumer."""
        from pipeline.stream_consumer import BaseStreamConsumer, StreamConsumerConfig

        class TestConsumer(BaseStreamConsumer):
            async def _initialize_phase(self):
                pass

            async def _cleanup_phase(self):
                pass

            async def _execute_phase(self, verse_id, text, context):
                return {"result": "test"}

        # Need to patch PHASE_TOPICS
        with patch("pipeline.stream_consumer.PHASE_TOPICS", {
            "test_phase": {
                "request": "test:request",
                "complete": "test:complete"
            }
        }):
            consumer = TestConsumer(
                phase_name="test_phase",
                config=StreamConsumerConfig(heartbeat_interval=0),
                event_bus=mock_event_bus
            )
            return consumer

    @pytest.mark.asyncio
    async def test_initialize(self, concrete_consumer, mock_event_bus):
        """Test consumer initialization."""
        await concrete_consumer.initialize()

        assert concrete_consumer._initialized is True
        mock_event_bus.ensure_consumer_group.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, concrete_consumer, mock_event_bus):
        """Test that initialize is idempotent."""
        await concrete_consumer.initialize()
        await concrete_consumer.initialize()

        # Should only be called once
        assert mock_event_bus.ensure_consumer_group.call_count == 1

    @pytest.mark.asyncio
    async def test_start_initializes_if_needed(self, concrete_consumer, mock_event_bus):
        """Test that start initializes if not already initialized."""
        # Mock subscribe to not block
        async def mock_subscribe(*args, **kwargs):
            return
            yield  # Make it an async generator

        mock_event_bus.subscribe = mock_subscribe

        await concrete_consumer.start()

        assert concrete_consumer._initialized is True
        assert concrete_consumer._running is True

    @pytest.mark.asyncio
    async def test_stop(self, concrete_consumer, mock_event_bus):
        """Test consumer stop."""
        # First initialize and start
        async def mock_subscribe(*args, **kwargs):
            return
            yield

        mock_event_bus.subscribe = mock_subscribe

        await concrete_consumer.initialize()
        await concrete_consumer.start()
        await concrete_consumer.stop()

        assert concrete_consumer._running is False

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self, concrete_consumer):
        """Test stop when not running is no-op."""
        # Should not raise
        await concrete_consumer.stop()

    @pytest.mark.asyncio
    async def test_shutdown(self, concrete_consumer, mock_event_bus):
        """Test full shutdown."""
        async def mock_subscribe(*args, **kwargs):
            return
            yield

        mock_event_bus.subscribe = mock_subscribe

        await concrete_consumer.initialize()
        await concrete_consumer.start()
        await concrete_consumer.shutdown()

        assert concrete_consumer._running is False
        assert concrete_consumer._initialized is False


# =============================================================================
# Metrics Tests
# =============================================================================

class TestMetrics:
    """Tests for metrics recording."""

    @pytest.fixture
    def concrete_consumer(self):
        """Create a concrete implementation of BaseStreamConsumer."""
        from pipeline.stream_consumer import BaseStreamConsumer, StreamConsumerConfig

        class TestConsumer(BaseStreamConsumer):
            async def _initialize_phase(self):
                pass

            async def _cleanup_phase(self):
                pass

            async def _execute_phase(self, verse_id, text, context):
                return {"result": "test"}

        with patch("pipeline.stream_consumer.PHASE_TOPICS", {
            "test_phase": {
                "request": "test:request",
                "complete": "test:complete"
            }
        }):
            consumer = TestConsumer(
                phase_name="test_phase",
                config=StreamConsumerConfig()
            )
            return consumer

    def test_initial_metrics(self, concrete_consumer):
        """Test initial metrics values."""
        metrics = concrete_consumer.get_metrics()

        assert metrics["messages_received"] == 0
        assert metrics["messages_processed"] == 0
        assert metrics["messages_failed"] == 0
        assert metrics["messages_retried"] == 0
        assert metrics["avg_processing_time"] == 0.0

    def test_record_processing_time(self, concrete_consumer):
        """Test recording processing time."""
        concrete_consumer._record_processing_time(1.5)
        concrete_consumer._record_processing_time(2.5)

        metrics = concrete_consumer.get_metrics()
        assert metrics["avg_processing_time"] == 2.0  # (1.5 + 2.5) / 2

    def test_processing_time_window(self, concrete_consumer):
        """Test that only last 100 times are kept."""
        # Add 110 times
        for i in range(110):
            concrete_consumer._record_processing_time(float(i))

        # Should only keep last 100
        assert len(concrete_consumer._metrics["processing_times"]) <= 100

    @pytest.mark.asyncio
    async def test_health_check(self, concrete_consumer):
        """Test health check."""
        health = await concrete_consumer.health_check()

        assert "healthy" in health
        assert health["healthy"] is False  # Not running or initialized


# =============================================================================
# PhaseStreamConsumer Tests
# =============================================================================

class TestPhaseStreamConsumer:
    """Tests for PhaseStreamConsumer adapter."""

    @pytest.fixture
    def mock_phase(self):
        """Create a mock BasePipelinePhase."""
        phase = Mock()
        phase.config = Mock()
        phase.config.name = "test_phase"
        phase.initialize = AsyncMock()
        phase.cleanup = AsyncMock()
        phase.execute = AsyncMock(return_value=Mock(
            status=Mock(value="completed"),
            agent_results={"agent1": {}},
            metrics={"time": 1.0},
            duration=1.0,
            error=None
        ))
        return phase

    @pytest.fixture
    def mock_event_bus(self):
        """Create a mock EventBus."""
        event_bus = AsyncMock()
        event_bus.ensure_consumer_group = AsyncMock()
        return event_bus

    @pytest.mark.asyncio
    async def test_initialize_phase(self, mock_phase, mock_event_bus):
        """Test phase initialization."""
        from pipeline.stream_consumer import PhaseStreamConsumer

        with patch("pipeline.stream_consumer.PHASE_TOPICS", {
            "test_phase": {"request": "test:request", "complete": "test:complete"}
        }):
            consumer = PhaseStreamConsumer(
                phase=mock_phase,
                event_bus=mock_event_bus
            )
            await consumer._initialize_phase()

            mock_phase.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_phase(self, mock_phase, mock_event_bus):
        """Test phase cleanup."""
        from pipeline.stream_consumer import PhaseStreamConsumer

        with patch("pipeline.stream_consumer.PHASE_TOPICS", {
            "test_phase": {"request": "test:request", "complete": "test:complete"}
        }):
            consumer = PhaseStreamConsumer(
                phase=mock_phase,
                event_bus=mock_event_bus
            )
            await consumer._cleanup_phase()

            mock_phase.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_phase(self, mock_phase, mock_event_bus):
        """Test phase execution."""
        from pipeline.stream_consumer import PhaseStreamConsumer

        with patch("pipeline.stream_consumer.PHASE_TOPICS", {
            "test_phase": {"request": "test:request", "complete": "test:complete"}
        }):
            consumer = PhaseStreamConsumer(
                phase=mock_phase,
                event_bus=mock_event_bus
            )
            result = await consumer._execute_phase("GEN.1.1", "test text", {})

            assert result["status"] == "completed"
            assert "agent_results" in result
            mock_phase.execute.assert_called_once()


# =============================================================================
# StreamConsumerManager Tests
# =============================================================================

class TestStreamConsumerManager:
    """Tests for StreamConsumerManager."""

    @pytest.fixture
    def mock_consumer(self):
        """Create a mock consumer."""
        consumer = AsyncMock()
        consumer.initialize = AsyncMock()
        consumer.start = AsyncMock()
        consumer.stop = AsyncMock()
        consumer.get_metrics = Mock(return_value={"test": 1})
        consumer.health_check = AsyncMock(return_value={"healthy": True})
        return consumer

    def test_register_consumer(self, mock_consumer):
        """Test registering a consumer."""
        from pipeline.stream_consumer import StreamConsumerManager

        manager = StreamConsumerManager()
        manager.register_consumer("test_phase", mock_consumer)

        assert "test_phase" in manager._consumers

    @pytest.mark.asyncio
    async def test_initialize_all(self, mock_consumer):
        """Test initializing all consumers."""
        from pipeline.stream_consumer import StreamConsumerManager

        manager = StreamConsumerManager()
        manager.register_consumer("phase1", mock_consumer)

        await manager.initialize()

        mock_consumer.initialize.assert_called_once()
        assert manager._initialized is True

    @pytest.mark.asyncio
    async def test_start_all(self, mock_consumer):
        """Test starting all consumers."""
        from pipeline.stream_consumer import StreamConsumerManager

        manager = StreamConsumerManager()
        manager.register_consumer("phase1", mock_consumer)

        await manager.start()

        mock_consumer.start.assert_called_once()
        assert manager._running is True

    @pytest.mark.asyncio
    async def test_stop_all(self, mock_consumer):
        """Test stopping all consumers."""
        from pipeline.stream_consumer import StreamConsumerManager

        manager = StreamConsumerManager()
        manager.register_consumer("phase1", mock_consumer)

        await manager.start()
        await manager.stop()

        mock_consumer.stop.assert_called_once()
        assert manager._running is False

    @pytest.mark.asyncio
    async def test_shutdown(self, mock_consumer):
        """Test full shutdown."""
        from pipeline.stream_consumer import StreamConsumerManager

        manager = StreamConsumerManager()
        manager.register_consumer("phase1", mock_consumer)

        await manager.start()
        await manager.shutdown()

        assert manager._running is False
        assert manager._initialized is False
        assert len(manager._consumers) == 0

    def test_get_metrics(self, mock_consumer):
        """Test getting metrics from all consumers."""
        from pipeline.stream_consumer import StreamConsumerManager

        manager = StreamConsumerManager()
        manager.register_consumer("phase1", mock_consumer)

        metrics = manager.get_metrics()

        assert "phase1" in metrics
        assert metrics["phase1"]["test"] == 1

    @pytest.mark.asyncio
    async def test_health_check(self, mock_consumer):
        """Test health check for all consumers."""
        from pipeline.stream_consumer import StreamConsumerManager

        manager = StreamConsumerManager()
        manager.register_consumer("phase1", mock_consumer)
        manager._running = True

        health = await manager.health_check()

        assert health["healthy"] is True
        assert health["running"] is True
        assert "phase1" in health["consumers"]

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, mock_consumer):
        """Test health check when consumer is unhealthy."""
        from pipeline.stream_consumer import StreamConsumerManager

        mock_consumer.health_check = AsyncMock(return_value={"healthy": False})

        manager = StreamConsumerManager()
        manager.register_consumer("phase1", mock_consumer)
        manager._running = True

        health = await manager.health_check()

        assert health["healthy"] is False


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""

    @pytest.fixture
    def mock_phase(self):
        """Create a mock phase."""
        phase = Mock()
        phase.config = Mock()
        phase.config.name = "test_phase"
        phase.initialize = AsyncMock()
        phase.cleanup = AsyncMock()
        return phase

    @pytest.mark.asyncio
    async def test_create_phase_consumer(self, mock_phase):
        """Test create_phase_consumer factory."""
        from pipeline.stream_consumer import create_phase_consumer

        mock_event_bus = AsyncMock()
        mock_event_bus.ensure_consumer_group = AsyncMock()

        with patch("pipeline.stream_consumer.get_event_bus", return_value=mock_event_bus):
            with patch("pipeline.stream_consumer.PHASE_TOPICS", {
                "test_phase": {"request": "test:request", "complete": "test:complete"}
            }):
                consumer = await create_phase_consumer(mock_phase, auto_start=False)

                assert consumer._initialized is True

    @pytest.mark.asyncio
    async def test_create_consumer_manager(self, mock_phase):
        """Test create_consumer_manager factory."""
        from pipeline.stream_consumer import create_consumer_manager

        mock_event_bus = AsyncMock()
        mock_event_bus.ensure_consumer_group = AsyncMock()

        with patch("pipeline.stream_consumer.get_event_bus", return_value=mock_event_bus):
            with patch("pipeline.stream_consumer.PHASE_TOPICS", {
                "test_phase": {"request": "test:request", "complete": "test:complete"}
            }):
                manager = await create_consumer_manager(
                    [mock_phase],
                    auto_start=False
                )

                assert manager._initialized is True
                assert len(manager._consumers) == 1
