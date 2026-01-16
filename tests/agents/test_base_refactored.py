"""
Tests for the refactored BaseExtractionAgent.

Verifies:
- Async context manager support
- LRU cache with TTL
- Factory methods for ExtractionResult
- Proper error handling hierarchy
- Type-safe extraction context
- Resource cleanup
"""
import asyncio
import pytest
import time
from datetime import datetime, timezone

from agents.base import (
    BaseExtractionAgent,
    AgentConfig,
    ExtractionType,
    ExtractionContext,
    ExtractionResult,
    AgentPhase,
    LRUCacheWithTTL,
)
from data.schemas import ProcessingStatus
from core.errors import BiblosAgentError, BiblosValidationError


# =============================================================================
# Test Agent Implementation
# =============================================================================


class TestAgent(BaseExtractionAgent):
    """Concrete implementation for testing."""

    def __init__(self, config: AgentConfig, fail_on: str = None):
        super().__init__(config)
        self.fail_on = fail_on
        self.setup_called = False
        self.cleanup_called = False

    async def _setup_resources(self):
        """Track resource setup."""
        self.setup_called = True
        await asyncio.sleep(0.01)  # Simulate setup work

    async def _cleanup_resources(self):
        """Track resource cleanup."""
        self.cleanup_called = True
        await asyncio.sleep(0.01)  # Simulate cleanup work

    async def extract(
        self,
        verse_id: str,
        text: str,
        context: ExtractionContext,
    ) -> ExtractionResult:
        """Test extraction logic."""
        # Simulate different failure modes
        if self.fail_on == "extract":
            raise BiblosAgentError("Extraction failed", "TEST_ERROR")
        if self.fail_on == "timeout":
            await asyncio.sleep(10)  # Will timeout

        # Successful extraction
        return ExtractionResult.success(
            agent_name=self.config.name,
            extraction_type=self.config.extraction_type,
            verse_id=verse_id,
            data={"text_length": len(text), "book": context.book},
            confidence=0.95,
        )

    async def validate(self, result: ExtractionResult) -> bool:
        """Test validation logic."""
        if self.fail_on == "validate":
            return False
        return result.confidence >= self.config.min_confidence

    def get_dependencies(self) -> list[str]:
        """Return test dependencies."""
        return []


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def agent_config():
    """Create test agent configuration."""
    return AgentConfig(
        name="test_agent",
        extraction_type=ExtractionType.SEMANTIC,
        phase=AgentPhase.LINGUISTIC,
        timeout_seconds=1,  # Short timeout for testing
        max_retries=2,
        cache_ttl_seconds=1,
        cache_max_size=10,
    )


@pytest.fixture
async def agent(agent_config):
    """Create and initialize test agent."""
    agent = TestAgent(agent_config)
    await agent.initialize()
    yield agent
    await agent.shutdown()


@pytest.fixture
def extraction_context():
    """Create test extraction context."""
    return ExtractionContext(
        book="GEN",
        book_name="Genesis",
        chapter=1,
        testament="OT",
        language="hebrew",
    )


# =============================================================================
# Test Async Context Manager
# =============================================================================


@pytest.mark.asyncio
async def test_async_context_manager(agent_config):
    """Test __aenter__ and __aexit__ methods."""
    agent = TestAgent(agent_config)

    # Before entering context
    assert not agent.state.is_initialized
    assert not agent.setup_called

    # Enter context
    async with agent as a:
        assert a is agent
        assert agent.state.is_initialized
        assert agent.setup_called
        assert not agent.cleanup_called

    # After exiting context
    assert not agent.state.is_initialized
    assert agent.cleanup_called


@pytest.mark.asyncio
async def test_async_context_manager_with_exception(agent_config):
    """Test context manager cleanup on exception."""
    agent = TestAgent(agent_config)

    with pytest.raises(RuntimeError):
        async with agent:
            assert agent.state.is_initialized
            raise RuntimeError("Test error")

    # Cleanup should still happen
    assert not agent.state.is_initialized
    assert agent.cleanup_called


# =============================================================================
# Test LRU Cache with TTL
# =============================================================================


def test_lru_cache_basic():
    """Test basic cache operations."""
    cache = LRUCacheWithTTL(max_size=3, ttl_seconds=10)

    # Create test results
    result1 = ExtractionResult.success(
        agent_name="test", extraction_type=ExtractionType.SEMANTIC,
        verse_id="GEN.1.1", data={"value": 1}, confidence=0.9
    )
    result2 = ExtractionResult.success(
        agent_name="test", extraction_type=ExtractionType.SEMANTIC,
        verse_id="GEN.1.2", data={"value": 2}, confidence=0.9
    )

    # Test set and get
    cache.set("key1", result1)
    cache.set("key2", result2)

    assert cache.get("key1") == result1
    assert cache.get("key2") == result2
    assert cache.get("key3") is None
    assert len(cache) == 2


def test_lru_cache_eviction():
    """Test LRU eviction when max size is reached."""
    cache = LRUCacheWithTTL(max_size=2, ttl_seconds=10)

    result1 = ExtractionResult.success(
        agent_name="test", extraction_type=ExtractionType.SEMANTIC,
        verse_id="GEN.1.1", data={"value": 1}, confidence=0.9
    )
    result2 = ExtractionResult.success(
        agent_name="test", extraction_type=ExtractionType.SEMANTIC,
        verse_id="GEN.1.2", data={"value": 2}, confidence=0.9
    )
    result3 = ExtractionResult.success(
        agent_name="test", extraction_type=ExtractionType.SEMANTIC,
        verse_id="GEN.1.3", data={"value": 3}, confidence=0.9
    )

    # Add 3 items to cache with max_size=2
    cache.set("key1", result1)
    cache.set("key2", result2)
    cache.set("key3", result3)  # Should evict key1

    assert cache.get("key1") is None  # Evicted
    assert cache.get("key2") == result2
    assert cache.get("key3") == result3
    assert len(cache) == 2


def test_lru_cache_ttl_expiration():
    """Test TTL expiration."""
    cache = LRUCacheWithTTL(max_size=10, ttl_seconds=0.1)  # 100ms TTL

    result = ExtractionResult.success(
        agent_name="test", extraction_type=ExtractionType.SEMANTIC,
        verse_id="GEN.1.1", data={"value": 1}, confidence=0.9
    )

    cache.set("key1", result)
    assert cache.get("key1") == result

    # Wait for expiration
    time.sleep(0.2)
    assert cache.get("key1") is None  # Expired


def test_lru_cache_clear():
    """Test cache clearing."""
    cache = LRUCacheWithTTL(max_size=10, ttl_seconds=10)

    for i in range(5):
        result = ExtractionResult.success(
            agent_name="test", extraction_type=ExtractionType.SEMANTIC,
            verse_id=f"GEN.1.{i}", data={"value": i}, confidence=0.9
        )
        cache.set(f"key{i}", result)

    assert len(cache) == 5
    count = cache.clear()
    assert count == 5
    assert len(cache) == 0


# =============================================================================
# Test Factory Methods
# =============================================================================


def test_extraction_result_success_factory():
    """Test ExtractionResult.success() factory method."""
    result = ExtractionResult.success(
        agent_name="test_agent",
        extraction_type=ExtractionType.SEMANTIC,
        verse_id="GEN.1.1",
        data={"key": "value"},
        confidence=0.95,
        processing_time_ms=100.5,
    )

    assert result.agent_name == "test_agent"
    assert result.extraction_type == ExtractionType.SEMANTIC
    assert result.verse_id == "GEN.1.1"
    assert result.status == ProcessingStatus.COMPLETED
    assert result.data == {"key": "value"}
    assert result.confidence == 0.95
    assert result.processing_time_ms == 100.5
    assert len(result.errors) == 0


def test_extraction_result_failure_factory():
    """Test ExtractionResult.failure() factory method."""
    result = ExtractionResult.failure(
        agent_name="test_agent",
        extraction_type=ExtractionType.SEMANTIC,
        verse_id="GEN.1.1",
        error="Something went wrong",
        processing_time_ms=50.0,
    )

    assert result.agent_name == "test_agent"
    assert result.status == ProcessingStatus.FAILED
    assert result.confidence == 0.0
    assert result.data == {}
    assert result.errors == ["Something went wrong"]


def test_extraction_result_needs_review_factory():
    """Test ExtractionResult.needs_review() factory method."""
    result = ExtractionResult.needs_review(
        agent_name="test_agent",
        extraction_type=ExtractionType.SEMANTIC,
        verse_id="GEN.1.1",
        data={"partial": "data"},
        confidence=0.6,
        warning="Validation failed",
        processing_time_ms=75.0,
    )

    # Uses FAILED status with warnings to indicate review needed
    assert result.status == ProcessingStatus.FAILED
    assert result.warnings == ["Validation failed"]
    assert result.data == {"partial": "data"}
    assert result.confidence == 0.6


# =============================================================================
# Test Error Handling
# =============================================================================


@pytest.mark.asyncio
async def test_error_handling_agent_error(agent_config, extraction_context):
    """Test BiblosAgentError handling."""
    agent = TestAgent(agent_config, fail_on="extract")
    await agent.initialize()

    result = await agent.process("GEN.1.1", "Test text", extraction_context)

    assert result.status == ProcessingStatus.FAILED
    assert "Extraction failed" in result.errors[0]
    assert result.confidence == 0.0
    assert agent.state.failed == 1

    await agent.shutdown()


@pytest.mark.asyncio
async def test_error_handling_validation_error(agent_config, extraction_context):
    """Test validation failure handling."""
    agent = TestAgent(agent_config, fail_on="validate")
    await agent.initialize()

    result = await agent.process("GEN.1.1", "Test text", extraction_context)

    # Validation failures result in COMPLETED status with warnings
    assert result.status == ProcessingStatus.COMPLETED
    assert "Validation failed" in result.warnings
    assert agent.state.successful == 1  # Still counted as successful

    await agent.shutdown()


@pytest.mark.asyncio
async def test_error_handling_timeout(agent_config, extraction_context):
    """Test timeout handling."""
    agent = TestAgent(agent_config, fail_on="timeout")
    await agent.initialize()

    result = await agent.process("GEN.1.1", "Test text", extraction_context)

    assert result.status == ProcessingStatus.FAILED
    assert any("timed out" in err.lower() for err in result.errors)

    await agent.shutdown()


# =============================================================================
# Test Type-Safe Context
# =============================================================================


@pytest.mark.asyncio
async def test_typed_context(agent, extraction_context):
    """Test ExtractionContext type safety."""
    # Test with typed context
    result = await agent.process("GEN.1.1", "Test text", extraction_context)
    assert result.status == ProcessingStatus.COMPLETED
    assert result.data["book"] == "GEN"

    # Test with dict context (should auto-convert)
    dict_context = {
        "book": "EXO",
        "book_name": "Exodus",
        "chapter": 1,
    }
    result = await agent.process("EXO.1.1", "Test text", dict_context)
    assert result.status == ProcessingStatus.COMPLETED
    assert result.data["book"] == "EXO"


# =============================================================================
# Test Resource Cleanup
# =============================================================================


@pytest.mark.asyncio
async def test_resource_cleanup_on_shutdown(agent_config):
    """Test proper resource cleanup."""
    agent = TestAgent(agent_config)
    await agent.initialize()

    assert agent.state.is_initialized
    assert agent.setup_called

    # Add some cache entries
    for i in range(5):
        context = ExtractionContext(book="GEN")
        await agent.process(f"GEN.1.{i}", f"Text {i}", context)

    assert len(agent._cache) > 0

    # Shutdown
    await agent.shutdown()

    assert not agent.state.is_initialized
    assert agent.cleanup_called
    assert len(agent._cache) == 0  # Cache cleared


@pytest.mark.asyncio
async def test_multiple_initialize_calls(agent_config):
    """Test that multiple initialize calls are safe."""
    agent = TestAgent(agent_config)

    await agent.initialize()
    assert agent.state.is_initialized

    # Second initialize should be a no-op
    await agent.initialize()
    assert agent.state.is_initialized

    await agent.shutdown()


# =============================================================================
# Test Caching Behavior
# =============================================================================


@pytest.mark.asyncio
async def test_cache_hit_miss(agent, extraction_context):
    """Test cache hit/miss tracking."""
    verse_id = "GEN.1.1"
    text = "Test text"

    # First call - cache miss
    result1 = await agent.process(verse_id, text, extraction_context)
    assert agent.state.cache_misses == 1
    assert agent.state.cache_hits == 0

    # Second call - cache hit
    result2 = await agent.process(verse_id, text, extraction_context)
    assert agent.state.cache_hits == 1
    assert result1.verse_id == result2.verse_id


@pytest.mark.asyncio
async def test_cache_disabled(agent_config, extraction_context):
    """Test processing with cache disabled."""
    agent_config.enable_caching = False
    agent = TestAgent(agent_config)
    await agent.initialize()

    verse_id = "GEN.1.1"
    text = "Test text"

    # Multiple calls should all execute
    result1 = await agent.process(verse_id, text, extraction_context)
    result2 = await agent.process(verse_id, text, extraction_context)

    assert agent.state.cache_hits == 0
    assert agent.state.cache_misses == 0  # Not tracked when disabled

    await agent.shutdown()


# =============================================================================
# Test Metrics
# =============================================================================


@pytest.mark.asyncio
async def test_metrics_tracking(agent, extraction_context):
    """Test metrics collection."""
    # Process multiple verses
    for i in range(5):
        await agent.process(f"GEN.1.{i}", f"Text {i}", extraction_context)

    metrics = agent.metrics.to_dict()

    assert metrics["total_processed"] == 5
    assert metrics["successful"] == 5
    assert metrics["failed"] == 0
    assert metrics["success_rate"] == 1.0
    assert metrics["avg_confidence"] > 0


@pytest.mark.asyncio
async def test_metrics_with_failures(agent_config, extraction_context):
    """Test metrics with mixed success/failure."""
    agent = TestAgent(agent_config, fail_on="extract")
    await agent.initialize()

    # Process some verses (will fail)
    for i in range(3):
        await agent.process(f"GEN.1.{i}", f"Text {i}", extraction_context)

    metrics = agent.metrics.to_dict()

    assert metrics["total_processed"] == 3
    assert metrics["successful"] == 0
    assert metrics["failed"] == 3
    assert metrics["success_rate"] == 0.0

    await agent.shutdown()


# =============================================================================
# Test Batch Processing
# =============================================================================


@pytest.mark.asyncio
async def test_batch_processing_sequential(agent):
    """Test sequential batch processing."""
    verses = [
        {"verse_id": f"GEN.1.{i}", "text": f"Text {i}", "context": {"book": "GEN"}}
        for i in range(5)
    ]

    results = await agent.process_batch(verses, parallel=False)

    assert len(results) == 5
    assert all(r.status == ProcessingStatus.COMPLETED for r in results)


@pytest.mark.asyncio
async def test_batch_processing_parallel(agent):
    """Test parallel batch processing."""
    verses = [
        {"verse_id": f"GEN.1.{i}", "text": f"Text {i}", "context": {"book": "GEN"}}
        for i in range(10)
    ]

    results = await agent.process_batch(verses, parallel=True)

    assert len(results) == 10
    assert all(r.status == ProcessingStatus.COMPLETED for r in results)


@pytest.mark.asyncio
async def test_batch_processing_with_progress(agent):
    """Test batch processing with progress callback."""
    verses = [
        {"verse_id": f"GEN.1.{i}", "text": f"Text {i}", "context": {"book": "GEN"}}
        for i in range(5)
    ]

    progress_calls = []

    def progress_callback(current, total):
        progress_calls.append((current, total))

    results = await agent.process_batch(verses, progress_callback=progress_callback)

    assert len(results) == 5
    assert len(progress_calls) == 5
    assert progress_calls[-1] == (5, 5)
