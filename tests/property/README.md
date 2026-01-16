# Property-Based Testing for BIBLOS v2

This directory contains property-based tests using [Hypothesis](https://hypothesis.readthedocs.io/), a powerful framework for generating test cases based on specifications.

## Overview

Property-based testing complements traditional example-based unit tests by automatically generating hundreds or thousands of test cases to find edge cases that developers might not think of manually. With 31,000+ Bible verses, it's impossible to manually enumerate all edge cases—property-based testing solves this problem.

## Test Modules

### 1. `test_schema_integrity.py`
Tests schema validation, serialization, and Unicode handling:
- **Roundtrip serialization**: Schemas survive JSON serialization/deserialization
- **Unicode handling**: Greek (Ἐν ἀρχῇ), Hebrew (בְּרֵאשִׁית), Coptic, and emoji support
- **Validation logic**: Invalid schemas are correctly caught
- **Stateful testing**: Schema mutations preserve invariants

**Key Properties:**
- Valid schemas always pass validation
- Serialization is lossless
- Unicode text never crashes the system
- Schema fields maintain their types

### 2. `test_verse_parsing.py`
Tests verse ID parsing with malformed inputs:
- **Format validation**: `GEN.1.1` format checking
- **Malformed input handling**: Empty strings, wrong separators, Unicode in IDs
- **Normalization**: `GEN:1:1` → `GEN.1.1`, `gen.1.1` → `GEN.1.1`
- **Edge cases**: Very long strings, only whitespace, zero/negative numbers

**Key Properties:**
- Valid verse IDs match `BOOK.CHAPTER.VERSE` pattern
- Normalization is idempotent: `normalize(normalize(x)) = normalize(x)`
- Parser never crashes on invalid input
- Case-insensitive normalization works correctly

### 3. `test_cross_reference_invariants.py`
Tests cross-reference business rules and invariants:
- **Confidence bounds**: Always in [0, 1], no NaN or Infinity
- **Enum validation**: Connection types and strengths are valid
- **Bidirectional consistency**: Forward and reverse refs are symmetric
- **Self-reference detection**: Verses shouldn't reference themselves
- **Stateful collection**: Managing sets of cross-references

**Key Properties:**
- `0 ≤ confidence ≤ 1` for all cross-references
- Connection types are valid enum values
- Source ≠ target (verses don't self-reference)
- Bidirectional refs maintain symmetry

### 4. `test_pipeline_invariants.py`
Tests pipeline execution and golden record validation:
- **Time consistency**: `end_time ≥ start_time`, duration is non-negative
- **Status validity**: Pipeline status is always a valid enum
- **Phase aggregation**: Sum of phase times ≤ total pipeline time
- **Golden record certification**: Completed pipelines have valid certification
- **Stateful execution**: Pipeline progresses through phases correctly

**Key Properties:**
- Processing times are always non-negative
- Completed pipelines have all phases completed/skipped
- Agent counts are non-negative
- Golden records have valid certification structure

### 5. `test_ml_invariants.py`
Tests ML model outputs and numerical stability:
- **Embedding dimensions**: Consistent across all vectors
- **Similarity scores**: Cosine in [-1, 1], Euclidean ≥ 0
- **Batch consistency**: Batch processing = sequential processing
- **Determinism**: Same seed produces same output
- **Numerical stability**: No overflow, underflow, or NaN propagation

**Key Properties:**
- Embeddings have finite values (no NaN/Inf)
- `cosine_similarity(A, B) = cosine_similarity(B, A)` (symmetry)
- `cosine_similarity(A, A) = 1` (self-similarity)
- Top-k selection returns k highest confidence scores
- Softmax values sum to 1.0

## Custom Strategies

The `strategies.py` module provides domain-specific Hypothesis strategies:

### Biblical Data Strategies
- `verse_id_strategy()`: Generates verse IDs like `GEN.1.1`
- `verse_pair_strategy()`: Generates distinct verse pairs
- `biblical_text_strategy()`: Greek, Hebrew, Coptic text
- `unicode_text_strategy()`: Full Unicode character range

### Schema Strategies
- `verse_schema_strategy()`: Complete VerseSchema objects
- `cross_reference_schema_strategy()`: Cross-reference data
- `word_schema_strategy()`: Word-level analysis data
- `golden_record_schema_strategy()`: Pipeline output data

### ML Strategies
- `embedding_vector_strategy(dimension)`: Fixed-dimension embeddings
- `similarity_score_strategy(metric)`: Cosine, Euclidean, etc.
- `confidence_score_strategy()`: Scores in [0, 1]

## Running Property Tests

### Run all property tests
```bash
pytest tests/property/ -v
```

### Run with custom Hypothesis profile
```bash
# Quick dev profile (50 examples)
pytest tests/property/ --hypothesis-profile=dev

# Thorough profile (1000 examples)
pytest tests/property/ --hypothesis-profile=thorough

# CI profile (100 examples)
pytest tests/property/ --hypothesis-profile=ci
```

### Run only property tests (exclude unit tests)
```bash
pytest -m property
```

### Run stateful tests only
```bash
pytest -m stateful
```

### Reproduce a failing test
Hypothesis automatically saves failing examples to `.hypothesis/examples.db`. Re-running the test will reproduce the failure.

### Debug mode
```bash
pytest tests/property/ --hypothesis-profile=debug --hypothesis-verbosity=debug
```

## Hypothesis Configuration

Configuration profiles are in `.hypothesis/config.ini`:

- **default**: 200 examples, 800ms deadline (standard testing)
- **dev**: 50 examples, 300ms deadline (quick feedback)
- **ci**: 100 examples, 500ms deadline (CI pipelines)
- **thorough**: 1000 examples, 2000ms deadline (comprehensive testing)
- **debug**: 20 examples, verbose output (troubleshooting)

## Edge Cases Found

Property-based testing has discovered several edge cases:

### Schema Issues
- ✅ Unicode normalization in verse IDs (NFD vs NFC)
- ✅ Empty string handling in required fields
- ✅ Very long text (>10,000 characters) serialization

### Parsing Issues
- ✅ Malformed verse IDs with multiple separators: `GEN::1::1`
- ✅ Lowercase book codes before normalization: `gen.1.1`
- ✅ Trailing/leading whitespace in verse IDs

### Cross-Reference Issues
- ✅ Confidence scores exactly at boundary (0.0, 1.0)
- ✅ Self-referencing cross-references (unusual but not invalid)
- ✅ Empty notes/sources lists

### Pipeline Issues
- ✅ Zero-duration phases (very fast execution)
- ✅ Floating-point precision in time calculations
- ✅ Empty phase results in completed pipelines

### ML Issues
- ✅ Embedding vectors with all zeros (degenerate case)
- ✅ Numerical stability in softmax with large values
- ✅ Cosine similarity with zero-norm vectors

## Best Practices

### Writing Property Tests

1. **Focus on invariants**: Properties that should *always* be true
   ```python
   @given(confidence_score_strategy())
   def test_confidence_bounds(confidence):
       assert 0.0 <= confidence <= 1.0
   ```

2. **Test symmetry**: Operations that should be commutative
   ```python
   @given(emb1=embedding_vector(), emb2=embedding_vector())
   def test_similarity_symmetry(emb1, emb2):
       assert similarity(emb1, emb2) == similarity(emb2, emb1)
   ```

3. **Test idempotence**: Operations that stabilize
   ```python
   @given(verse_id=st.text())
   def test_normalize_idempotent(verse_id):
       assert normalize(normalize(verse_id)) == normalize(verse_id)
   ```

4. **Use examples for edge cases**
   ```python
   @given(text=st.text())
   @example("")  # Explicitly test empty string
   @example("Ἐν ἀρχῇ")  # Explicitly test Greek
   def test_unicode_handling(text):
       # ...
   ```

5. **Use assumptions to filter invalid inputs**
   ```python
   @given(value=st.floats())
   def test_division(value):
       assume(value != 0)  # Filter out zero
       result = 10 / value
       assert math.isfinite(result)
   ```

### Stateful Testing

Use state machines for testing sequences of operations:

```python
class PipelineStateMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.pipeline = create_pipeline()

    @rule()
    def execute_phase(self):
        self.pipeline.execute_next_phase()

    @invariant()
    def status_valid(self):
        assert self.pipeline.status in valid_statuses
```

## Integration with CI/CD

Property tests run in CI with the `ci` profile:

```yaml
# .github/workflows/test.yml
- name: Run property tests
  run: pytest tests/property/ --hypothesis-profile=ci
```

## Further Reading

- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [Property-Based Testing with Hypothesis](https://hypothesis.works/articles/what-is-property-based-testing/)
- [Choosing Properties for Property-Based Testing](https://fsharpforfunandprofit.com/posts/property-based-testing-2/)
- [Hypothesis Strategies](https://hypothesis.readthedocs.io/en/latest/data.html)

## Contributing

When adding new features to BIBLOS v2:

1. Add property tests for new schemas/validators
2. Update `strategies.py` with domain-specific generators
3. Test invariants and edge cases
4. Document any discovered edge cases in this README
