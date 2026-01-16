"""
Tests for the Redis Streams Event Bus.

These tests require a running Redis instance.
Run with: pytest tests/pipeline/test_event_bus.py -v
"""
import asyncio
import pytest
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from pipeline.event_bus import (
    EventBus,
    EventBusConfig,
    EventMessage,
    StreamTopic,
    VerseEvent,
    PhaseRequestEvent,
    PhaseCompleteEvent,
    DeadLetterEvent,
    TOPICS,
    PHASE_TOPICS,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def event_config():
    """Create test event bus configuration."""
    return EventBusConfig(
        redis_url="redis://localhost:6379/1",  # Use DB 1 for tests
        max_connections=5,
        max_stream_length=1000,
        batch_size=5,
        max_retries=2,
    )


@pytest.fixture
def mock_redis():
    """Create mock Redis client."""
    mock = AsyncMock()
    mock.ping = AsyncMock(return_value=True)
    mock.xadd = AsyncMock(return_value="1234567890-0")
    mock.xreadgroup = AsyncMock(return_value=[])
    mock.xack = AsyncMock(return_value=1)
    mock.xrange = AsyncMock(return_value=[])
    mock.xgroup_create = AsyncMock()
    mock.xinfo_stream = AsyncMock(return_value={"length": 0})
    mock.xinfo_groups = AsyncMock(return_value=[])
    mock.xpending = AsyncMock(return_value={"pending": 0})
    mock.xpending_range = AsyncMock(return_value=[])
    mock.xclaim = AsyncMock(return_value=[])
    mock.xtrim = AsyncMock()
    mock.delete = AsyncMock(return_value=1)
    mock.set = AsyncMock()
    mock.setex = AsyncMock()
    mock.get = AsyncMock(return_value=None)
    mock.close = AsyncMock()
    mock.pipeline = MagicMock()
    return mock


@pytest.fixture
async def event_bus(mock_redis):
    """Create event bus with mocked Redis."""
    bus = EventBus(EventBusConfig())
    bus._redis = mock_redis
    bus._initialized = True
    yield bus
    await bus.shutdown()


# =============================================================================
# EVENT MESSAGE TESTS
# =============================================================================

class TestEventMessage:
    """Tests for EventMessage dataclass."""

    def test_create_message(self):
        """Test creating an event message."""
        msg = EventMessage(
            event_type="test_event",
            payload={"key": "value"},
        )

        assert msg.event_type == "test_event"
        assert msg.payload == {"key": "value"}
        assert msg.correlation_id is not None
        assert msg.timestamp is not None
        assert msg.retry_count == 0

    def test_to_dict(self):
        """Test converting message to dictionary."""
        msg = EventMessage(
            event_type="test",
            payload={"data": 123},
            correlation_id="test-id",
        )

        d = msg.to_dict()

        assert d["event_type"] == "test"
        assert json.loads(d["payload"]) == {"data": 123}
        assert d["correlation_id"] == "test-id"
        assert d["retry_count"] == "0"

    def test_from_dict(self):
        """Test creating message from dictionary."""
        data = {
            "event_type": "test",
            "payload": json.dumps({"data": 456}),
            "correlation_id": "abc123",
            "timestamp": "2024-01-01T00:00:00",
            "retry_count": "2",
            "source": "test",
            "version": "1.0.0",
        }

        msg = EventMessage.from_dict(data, message_id="msg-1")

        assert msg.event_type == "test"
        assert msg.payload == {"data": 456}
        assert msg.correlation_id == "abc123"
        assert msg.message_id == "msg-1"
        assert msg.retry_count == 2


class TestVerseEvent:
    """Tests for VerseEvent."""

    def test_create_verse_event(self):
        """Test creating a verse event."""
        event = VerseEvent.create(
            verse_id="GEN.1.1",
            text="In the beginning...",
            event_type="verse_ingested",
            metadata={"source": "test"},
        )

        assert event.event_type == "verse_ingested"
        assert event.payload["verse_id"] == "GEN.1.1"
        assert event.payload["text"] == "In the beginning..."
        assert event.payload["metadata"]["source"] == "test"


class TestPhaseEvents:
    """Tests for phase request and complete events."""

    def test_phase_request_event(self):
        """Test creating a phase request event."""
        event = PhaseRequestEvent.create(
            phase_name="linguistic",
            verse_id="GEN.1.1",
            text="In the beginning...",
            context={"completed_phases": []},
            correlation_id="corr-123",
        )

        assert event.event_type == "phase_linguistic_request"
        assert event.payload["phase_name"] == "linguistic"
        assert event.correlation_id == "corr-123"

    def test_phase_complete_event(self):
        """Test creating a phase complete event."""
        event = PhaseCompleteEvent.create(
            phase_name="linguistic",
            verse_id="GEN.1.1",
            status="completed",
            result={"confidence": 0.95},
            correlation_id="corr-123",
        )

        assert event.event_type == "phase_linguistic_complete"
        assert event.payload["status"] == "completed"
        assert event.payload["result"]["confidence"] == 0.95


# =============================================================================
# EVENT BUS TESTS
# =============================================================================

class TestEventBusPublish:
    """Tests for EventBus publishing."""

    @pytest.mark.asyncio
    async def test_publish_message(self, event_bus, mock_redis):
        """Test publishing a message."""
        msg = EventMessage(
            event_type="test",
            payload={"data": "test"},
        )

        result = await event_bus.publish(StreamTopic.VERSE_INGESTED, msg)

        assert result == "1234567890-0"
        mock_redis.xadd.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_with_max_len(self, event_bus, mock_redis):
        """Test publishing with custom max length."""
        msg = EventMessage(event_type="test", payload={})

        await event_bus.publish(StreamTopic.VERSE_INGESTED, msg, max_len=500)

        call_args = mock_redis.xadd.call_args
        assert call_args.kwargs["maxlen"] == 500

    @pytest.mark.asyncio
    async def test_publish_batch(self, event_bus, mock_redis):
        """Test publishing multiple messages."""
        messages = [
            EventMessage(event_type="test", payload={"i": i})
            for i in range(3)
        ]

        # Mock pipeline
        pipe_mock = AsyncMock()
        pipe_mock.execute = AsyncMock(return_value=["id1", "id2", "id3"])
        mock_redis.pipeline.return_value.__aenter__ = AsyncMock(return_value=pipe_mock)
        mock_redis.pipeline.return_value.__aexit__ = AsyncMock()

        results = await event_bus.publish_batch(StreamTopic.VERSE_INGESTED, messages)

        assert len(results) == 3


class TestEventBusConsumerGroups:
    """Tests for consumer group management."""

    @pytest.mark.asyncio
    async def test_create_consumer_group(self, event_bus, mock_redis):
        """Test creating a consumer group."""
        result = await event_bus.create_consumer_group(
            StreamTopic.VERSE_INGESTED,
            "test-group",
        )

        assert result is True
        mock_redis.xgroup_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_existing_group(self, event_bus, mock_redis):
        """Test creating an existing consumer group."""
        from redis.exceptions import ResponseError

        mock_redis.xgroup_create.side_effect = ResponseError("BUSYGROUP")

        result = await event_bus.create_consumer_group(
            StreamTopic.VERSE_INGESTED,
            "existing-group",
        )

        assert result is False


class TestEventBusAcknowledge:
    """Tests for message acknowledgment."""

    @pytest.mark.asyncio
    async def test_acknowledge_message(self, event_bus, mock_redis):
        """Test acknowledging a message."""
        result = await event_bus.acknowledge(
            StreamTopic.VERSE_INGESTED,
            "test-group",
            "1234567890-0",
        )

        assert result == 1
        mock_redis.xack.assert_called_once()

    @pytest.mark.asyncio
    async def test_acknowledge_batch(self, event_bus, mock_redis):
        """Test acknowledging multiple messages."""
        mock_redis.xack.return_value = 3

        result = await event_bus.acknowledge_batch(
            StreamTopic.VERSE_INGESTED,
            "test-group",
            ["id1", "id2", "id3"],
        )

        assert result == 3


class TestEventBusPending:
    """Tests for pending message handling."""

    @pytest.mark.asyncio
    async def test_get_pending_empty(self, event_bus, mock_redis):
        """Test getting pending messages when empty."""
        result = await event_bus.get_pending(
            StreamTopic.VERSE_INGESTED,
            "test-group",
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_get_pending_with_messages(self, event_bus, mock_redis):
        """Test getting pending messages."""
        mock_redis.xpending.return_value = {"pending": 2}
        mock_redis.xpending_range.return_value = [
            {
                "message_id": "id1",
                "consumer": "consumer1",
                "time_since_delivered": 5000,
                "times_delivered": 1,
            },
            {
                "message_id": "id2",
                "consumer": "consumer1",
                "time_since_delivered": 10000,
                "times_delivered": 2,
            },
        ]

        result = await event_bus.get_pending(
            StreamTopic.VERSE_INGESTED,
            "test-group",
        )

        assert len(result) == 2
        assert result[0]["message_id"] == "id1"


class TestEventBusReplay:
    """Tests for message replay."""

    @pytest.mark.asyncio
    async def test_read_stream(self, event_bus, mock_redis):
        """Test reading from a stream."""
        mock_redis.xrange.return_value = [
            ("id1", {"event_type": "test", "payload": "{}"}),
            ("id2", {"event_type": "test", "payload": "{}"}),
        ]

        result = await event_bus.read_stream(
            StreamTopic.VERSE_INGESTED,
            count=10,
        )

        assert len(result) == 2
        assert result[0][0] == "id1"


class TestEventBusDLQ:
    """Tests for dead letter queue."""

    @pytest.mark.asyncio
    async def test_send_to_dlq(self, event_bus, mock_redis):
        """Test sending to dead letter queue."""
        original = EventMessage(
            event_type="test",
            payload={"data": "test"},
            message_id="orig-123",
        )

        result = await event_bus.send_to_dlq(
            original,
            "Test error",
            StreamTopic.VERSE_INGESTED,
        )

        assert result == "1234567890-0"
        mock_redis.xadd.assert_called()

    @pytest.mark.asyncio
    async def test_send_to_dlq_disabled(self, event_bus, mock_redis):
        """Test DLQ disabled."""
        event_bus.config.enable_dlq = False

        result = await event_bus.send_to_dlq(
            EventMessage(event_type="test", payload={}),
            "Error",
            StreamTopic.VERSE_INGESTED,
        )

        assert result == ""
        mock_redis.xadd.assert_not_called()


class TestEventBusRetry:
    """Tests for retry logic."""

    def test_calculate_retry_delay(self, event_bus):
        """Test retry delay calculation."""
        # First retry
        delay0 = event_bus.calculate_retry_delay(0)
        assert 0.5 <= delay0 <= 1.5  # Base * jitter

        # Second retry (exponential)
        delay1 = event_bus.calculate_retry_delay(1)
        assert 1.0 <= delay1 <= 3.0

        # Check max cap
        delay_max = event_bus.calculate_retry_delay(10)
        assert delay_max <= event_bus.config.retry_delay_max


class TestEventBusStreamInfo:
    """Tests for stream information."""

    @pytest.mark.asyncio
    async def test_get_stream_info(self, event_bus, mock_redis):
        """Test getting stream info."""
        mock_redis.xinfo_stream.return_value = {
            "length": 100,
            "first-entry": ("id1", {}),
            "last-entry": ("id100", {}),
            "groups": 2,
        }

        result = await event_bus.get_stream_info(StreamTopic.VERSE_INGESTED)

        assert result["length"] == 100
        assert result["groups"] == 2

    @pytest.mark.asyncio
    async def test_get_stream_info_not_exists(self, event_bus, mock_redis):
        """Test getting info for non-existent stream."""
        from redis.exceptions import ResponseError

        mock_redis.xinfo_stream.side_effect = ResponseError("no such key")

        result = await event_bus.get_stream_info(StreamTopic.VERSE_INGESTED)

        assert result["length"] == 0
        assert result.get("exists") is False


class TestEventBusHealth:
    """Tests for health check."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, event_bus, mock_redis):
        """Test health check when healthy."""
        mock_redis.xinfo_stream.return_value = {"length": 10}

        result = await event_bus.health_check()

        assert result["healthy"] is True
        assert "latency_ms" in result

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self):
        """Test health check when not initialized."""
        bus = EventBus()

        result = await bus.health_check()

        assert result["healthy"] is False
        assert "error" in result


# =============================================================================
# TOPIC CONFIGURATION TESTS
# =============================================================================

class TestTopics:
    """Tests for topic configuration."""

    def test_topics_dict(self):
        """Test TOPICS dictionary."""
        assert "verse_ingested" in TOPICS
        assert TOPICS["verse_ingested"] == "stream:verse:ingested"

    def test_phase_topics(self):
        """Test PHASE_TOPICS configuration."""
        assert "linguistic" in PHASE_TOPICS
        assert "request" in PHASE_TOPICS["linguistic"]
        assert "complete" in PHASE_TOPICS["linguistic"]

    def test_all_phases_have_topics(self):
        """Test all phases have request and complete topics."""
        expected_phases = ["linguistic", "theological", "intertextual", "validation", "finalization"]

        for phase in expected_phases:
            assert phase in PHASE_TOPICS
            assert "request" in PHASE_TOPICS[phase]
            assert "complete" in PHASE_TOPICS[phase]
