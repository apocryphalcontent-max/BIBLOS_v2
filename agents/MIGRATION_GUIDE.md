# Base Agent Refactoring - Migration Guide

## Overview

The `agents/base.py` and `agents/base_v2.py` modules have been consolidated into a single, improved `agents/base.py` implementation. This guide helps you migrate to the new refactored base class.

## What Changed

### Consolidated Features

The refactored `BaseExtractionAgent` combines the best patterns from both previous implementations:

**From base.py:**
- OpenTelemetry distributed tracing
- Comprehensive error handling hierarchy
- Integration with `core.errors` module
- Structured logging with `AgentLogger`
- Detailed span attributes for observability

**From base_v2.py:**
- Pydantic models for type safety
- LRU cache with TTL (time-to-live)
- Factory methods for `ExtractionResult`
- Typed `ExtractionContext`
- Better state management with `AgentState`

**New Features:**
- Async context manager support (`async with agent:`)
- Improved resource cleanup
- Better cache management with automatic eviction
- Enhanced metrics tracking

## Migration Steps

### 1. Update Imports

**Before:**
```python
from agents.base import BaseExtractionAgent, AgentConfig, ExtractionResult
# or
from agents.base_v2 import BaseExtractionAgent, AgentConfig, ExtractionResult
```

**After:**
```python
from agents.base import (
    BaseExtractionAgent,
    AgentConfig,
    ExtractionType,
    ExtractionContext,
    ExtractionResult,
    AgentPhase,
)
```

### 2. Update ExtractionType Usage

**Before (base.py):**
```python
from agents.base import ExtractionType

# ExtractionType was using auto()
ExtractionType.SEMANTIC  # Returns an enum with auto-generated values
```

**After:**
```python
from agents.base import ExtractionType

# ExtractionType now uses string values
ExtractionType.SEMANTIC  # Returns "semantic"
```

### 3. Use Factory Methods for ExtractionResult

**Before:**
```python
# Manual construction
result = ExtractionResult(
    agent_name=self.config.name,
    extraction_type=self.config.extraction_type,
    verse_id=verse_id,
    status=ProcessingStatus.COMPLETED,
    data=extracted_data,
    confidence=0.95,
)
```

**After:**
```python
# Use factory methods for clarity
result = ExtractionResult.success(
    agent_name=self.config.name,
    extraction_type=self.config.extraction_type,
    verse_id=verse_id,
    data=extracted_data,
    confidence=0.95,
)

# For failures
result = ExtractionResult.failure(
    agent_name=self.config.name,
    extraction_type=self.config.extraction_type,
    verse_id=verse_id,
    error="Extraction failed",
)

# For needs review
result = ExtractionResult.needs_review(
    agent_name=self.config.name,
    extraction_type=self.config.extraction_type,
    verse_id=verse_id,
    data=partial_data,
    confidence=0.6,
    warning="Validation warning",
)
```

### 4. Use Typed ExtractionContext

**Before:**
```python
# Context was a plain dict
async def extract(
    self,
    verse_id: str,
    text: str,
    context: Dict[str, Any],
) -> ExtractionResult:
    book = context.get("book", "")
    linguistic_results = context.get("linguistic_results", {})
```

**After:**
```python
# Context is now typed with Pydantic
from agents.base import ExtractionContext

async def extract(
    self,
    verse_id: str,
    text: str,
    context: ExtractionContext,
) -> ExtractionResult:
    book = context.book  # Type-safe access
    linguistic_results = context.linguistic_results
```

### 5. Use Async Context Manager

**Before:**
```python
agent = MyAgent(config)
await agent.initialize()
try:
    result = await agent.process(verse_id, text, context)
finally:
    await agent.shutdown()
```

**After:**
```python
# Cleaner resource management
async with MyAgent(config) as agent:
    result = await agent.process(verse_id, text, context)
# Automatic cleanup on exit
```

### 6. Update AgentConfig

**Before:**
```python
config = AgentConfig(
    name="my_agent",
    extraction_type=ExtractionType.SEMANTIC,
    # ... other settings
)
```

**After:**
```python
# New cache-related settings
config = AgentConfig(
    name="my_agent",
    extraction_type=ExtractionType.SEMANTIC,
    phase=AgentPhase.LINGUISTIC,  # New: pipeline phase
    cache_ttl_seconds=3600,  # New: cache TTL
    cache_max_size=1000,  # New: max cache size
    # ... other settings
)
```

### 7. Access Improved Metrics

**Before:**
```python
metrics = agent.metrics.to_dict()
success_rate = metrics["successful"] / metrics["total_processed"]
```

**After:**
```python
# Metrics now have computed properties
metrics = agent.metrics.to_dict()
success_rate = metrics["success_rate"]  # Computed property
avg_time = metrics["avg_time_ms"]  # Computed property
```

## Breaking Changes

### 1. ExtractionType Values

The `ExtractionType` enum now uses string values instead of auto-generated integers:

**Before:** `ExtractionType.SEMANTIC` → `<ExtractionType.SEMANTIC: 4>`
**After:** `ExtractionType.SEMANTIC` → `"semantic"`

**Impact:** If you're storing enum values in databases or files, you may need to migrate data.

**Migration:**
```python
# Old code that checked enum values
if extraction_type == ExtractionType.SEMANTIC:
    # This still works

# But if you serialized the value:
# Before: stored as integer (4)
# After: stored as string ("semantic")
```

### 2. Context Parameter Type

The `extract()` method now expects `ExtractionContext` instead of `Dict[str, Any]`.

**Impact:** You need to update your agent implementations to use the typed context.

**Migration:**
```python
# Update your agent's extract method signature
async def extract(
    self,
    verse_id: str,
    text: str,
    context: ExtractionContext,  # Changed from Dict[str, Any]
) -> ExtractionResult:
    # Access context fields directly
    book = context.book
    linguistic_results = context.linguistic_results
```

### 3. Cache Implementation

The cache now uses LRU with TTL instead of a simple dictionary.

**Impact:** Cache entries now expire after `cache_ttl_seconds` and oldest entries are evicted when cache is full.

**Migration:**
- If you were relying on indefinite caching, set `cache_ttl_seconds` to a large value
- If you need custom cache behavior, override `_get_cached()` and `_set_cached()`

## New Capabilities

### 1. Async Context Manager

```python
# Automatic resource management
async with MyAgent(config) as agent:
    result = await agent.process(verse_id, text, context)
# Resources automatically cleaned up
```

### 2. Factory Methods

```python
# Clear intent with factory methods
success_result = ExtractionResult.success(...)
failure_result = ExtractionResult.failure(...)
review_result = ExtractionResult.needs_review(...)
```

### 3. LRU Cache with TTL

```python
# Cache automatically manages size and expiration
config = AgentConfig(
    name="my_agent",
    extraction_type=ExtractionType.SEMANTIC,
    cache_max_size=1000,  # Max 1000 entries
    cache_ttl_seconds=3600,  # Entries expire after 1 hour
)
```

### 4. Enhanced State Tracking

```python
# Access agent state
print(f"Cache hits: {agent.state.cache_hits}")
print(f"Cache misses: {agent.state.cache_misses}")
print(f"Success rate: {agent.state.success_rate}")
print(f"Consecutive errors: {agent.state.consecutive_errors}")
```

## Example: Complete Migration

### Before (base.py)

```python
from agents.base import BaseExtractionAgent, AgentConfig, ExtractionType, ExtractionResult
from data.schemas import ProcessingStatus

class SemanticAgent(BaseExtractionAgent):
    async def extract(
        self,
        verse_id: str,
        text: str,
        context: Dict[str, Any],
    ) -> ExtractionResult:
        # Extract semantic data
        data = self._analyze_semantics(text)

        return ExtractionResult(
            agent_name=self.config.name,
            extraction_type=self.config.extraction_type,
            verse_id=verse_id,
            status=ProcessingStatus.COMPLETED,
            data=data,
            confidence=0.95,
        )

    async def validate(self, result: ExtractionResult) -> bool:
        return result.confidence >= self.config.min_confidence

    def get_dependencies(self) -> List[str]:
        return ["morphological_agent"]

# Usage
config = AgentConfig(
    name="semantic_agent",
    extraction_type=ExtractionType.SEMANTIC,
)
agent = SemanticAgent(config)
await agent.initialize()
result = await agent.process("GEN.1.1", "בְּרֵאשִׁית", {})
await agent.shutdown()
```

### After (refactored base.py)

```python
from agents.base import (
    BaseExtractionAgent,
    AgentConfig,
    ExtractionType,
    ExtractionContext,
    ExtractionResult,
    AgentPhase,
)

class SemanticAgent(BaseExtractionAgent):
    async def extract(
        self,
        verse_id: str,
        text: str,
        context: ExtractionContext,  # Typed context
    ) -> ExtractionResult:
        # Extract semantic data
        data = self._analyze_semantics(text, context.linguistic_results)

        # Use factory method
        return ExtractionResult.success(
            agent_name=self.config.name,
            extraction_type=self.config.extraction_type,
            verse_id=verse_id,
            data=data,
            confidence=0.95,
        )

    async def validate(self, result: ExtractionResult) -> bool:
        return result.confidence >= self.config.min_confidence

    def get_dependencies(self) -> List[str]:
        return ["morphological_agent"]

# Usage with async context manager
config = AgentConfig(
    name="semantic_agent",
    extraction_type=ExtractionType.SEMANTIC,
    phase=AgentPhase.LINGUISTIC,
    cache_ttl_seconds=3600,
    cache_max_size=1000,
)

async with SemanticAgent(config) as agent:
    context = ExtractionContext(
        book="GEN",
        chapter=1,
        testament="OT",
        language="hebrew",
    )
    result = await agent.process("GEN.1.1", "בְּרֵאשִׁית", context)
# Automatic cleanup
```

## Testing Your Migration

Run the test suite to verify your migration:

```bash
# Run refactored base tests
pytest tests/agents/test_base_refactored.py -v

# Run your agent-specific tests
pytest tests/agents/test_your_agent.py -v
```

## Deprecation Timeline

- **Current:** `agents/base_v2.py` is deprecated but still available
- **Next minor version:** Deprecation warnings when importing from `base_v2`
- **Next major version:** `base_v2.py` will be removed

## Support

If you encounter issues during migration:

1. Check the test file: `tests/agents/test_base_refactored.py`
2. Review the docstrings in `agents/base.py`
3. File an issue with details about your specific use case

## Benefits of Refactored Implementation

1. **Type Safety:** Pydantic models catch errors at runtime
2. **Better Caching:** LRU with TTL prevents unbounded growth
3. **Resource Management:** Async context managers ensure cleanup
4. **Observability:** Enhanced metrics and tracing
5. **Code Clarity:** Factory methods make intent explicit
6. **Maintainability:** Single source of truth for base agent logic

## Checklist

- [ ] Update all agent imports to use `agents.base`
- [ ] Replace `Dict[str, Any]` contexts with `ExtractionContext`
- [ ] Use factory methods for `ExtractionResult`
- [ ] Add async context manager usage where appropriate
- [ ] Update `AgentConfig` to include new cache settings
- [ ] Run tests to verify migration
- [ ] Update any serialized `ExtractionType` values in storage
- [ ] Review and update documentation
